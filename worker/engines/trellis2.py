"""TRELLIS.2-4B adapter (blueprint P1-04). STATUS: NOT VERIFIED - no >= 24 GB NVIDIA GPU was available.

Written against microsoft/TRELLIS.2 at TRELLIS2_COMMIT (pipeline.run, preprocess_image,
pbr_attr_layout and o_voxel.postprocess.to_glb as used by that commit's app.py).
Run `python -m worker.main --smoke` inside worker/Dockerfile's image on a GPU host to verify.

Licence notes recorded in every manifest:
- TRELLIS.2 code and weights: MIT.
- The pipeline's own background remover is NOT used: our configurable segmenter runs instead
  (worker/segment.py; RMBG-2.0 only with SEGMENTER_ALLOW_NONCOMMERCIAL=1) and the pipeline's
  rembg model is never instantiated, so its weights are not even downloaded.
- Image conditioning uses DINOv3 (gated, custom licence, "Built with DINOv3" attribution).
- Texture baking uses nvdiffrast (NVIDIA Source Code License, non-commercial) - risk R1 / P5-06.
"""
from __future__ import annotations

import os
import time

from PIL import Image

from ..postprocess import cleanup
from ..segment import BiRefNetSegmenter, choose, segment
from .base import Engine, EngineError, EngineResult, Reporter

TRELLIS2_COMMIT = '75fbf0183001ed9876c8dbb35de6b68552ee08bd'  # github.com/microsoft/TRELLIS.2, 2026-06-05
WEIGHTS_REPO = 'microsoft/TRELLIS.2-4B'
PIPELINE_TYPES = {512: '512', 1024: '1024_cascade', 1536: '1536_cascade'}  # as in upstream app.py
# Sampler defaults copied from upstream app.py (the same values the public demo sends).
SAMPLERS = {
    'sparse_structure_sampler_params': {'steps': 12, 'guidance_strength': 7.5, 'guidance_rescale': 0.7, 'rescale_t': 5.0},
    'shape_slat_sampler_params': {'steps': 12, 'guidance_strength': 7.5, 'guidance_rescale': 0.5, 'rescale_t': 3.0},
    'tex_slat_sampler_params': {'steps': 12, 'guidance_strength': 1.0, 'guidance_rescale': 0.0, 'rescale_t': 3.0},
}
NVDIFFRAST_FACE_LIMIT = 16_777_216  # upstream: mesh.simplify(16777216) before baking
NOTICES = ['Built with DINOv3.', 'TRELLIS.2 (c) Microsoft Corporation, MIT License.',
           'Texture baking uses nvdiffrast (NVIDIA Source Code License, non-commercial).']


class _NoRembg:
    """Stands in for the pipeline's built-in background remover so its weights are never loaded."""

    def __init__(self, *args, **kwargs):
        self.requested = kwargs

    def to(self, device):
        pass

    cuda = cpu = lambda self: None

    def __call__(self, image):
        raise RuntimeError('built-in rembg disabled: worker/segment.py must provide an alpha mask')


class Trellis2Engine(Engine):
    id = 'trellis2-4b'
    textured = True

    def __init__(self):
        self.pipeline = None
        self.weights_revision = os.environ.get('TRELLIS2_WEIGHTS_REVISION')
        self.contrast_boost = os.environ.get('SEGMENT_CONTRAST_BOOST') == '1'

    def load(self) -> None:
        os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
        if not self.weights_revision:
            raise RuntimeError('TRELLIS2_WEIGHTS_REVISION must be set to a Hugging Face commit hash (reproducibility).')
        import torch  # noqa: PLC0415 - GPU image only
        from huggingface_hub import snapshot_download  # noqa: PLC0415
        from trellis2.pipelines import Trellis2ImageTo3DPipeline  # noqa: PLC0415
        from trellis2.pipelines import rembg as rembg_module  # noqa: PLC0415

        if not torch.cuda.is_available():
            raise RuntimeError('TRELLIS.2 needs an NVIDIA CUDA GPU.')
        local = snapshot_download(WEIGHTS_REPO, revision=self.weights_revision)
        original = rembg_module.BiRefNet
        rembg_module.BiRefNet = _NoRembg
        try:
            self.pipeline = Trellis2ImageTo3DPipeline.from_pretrained(local)
        finally:
            rembg_module.BiRefNet = original
        self.pipeline.cuda()
        args = self.pipeline._pretrained_args
        self.components = {'image_cond_model': args.get('image_cond_model'),
                           'pipeline_rembg_model_not_used': args.get('rembg_model')}
        self.segmenter = BiRefNetSegmenter(choose()).load()

    def run(self, image: Image.Image, params: dict, report: Reporter) -> EngineResult:
        import o_voxel  # noqa: PLC0415 - GPU image only
        import torch  # noqa: PLC0415

        report.stage('preparing', 'segment', 0.05)
        rgba, seg_info = segment(image, self.segmenter, self.contrast_boost)
        prepped = self.pipeline.preprocess_image(rgba)  # real alpha -> crop + premultiply only, no rembg
        report.stage('running', 'infer', 0.15)
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        pipeline_type = PIPELINE_TYPES[params['resolution']]
        try:
            meshes, (_, _, grid_res) = self.pipeline.run(prepped, seed=params['seed'], preprocess_image=False,
                                                         pipeline_type=pipeline_type, return_latent=True, **SAMPLERS)
            mesh = meshes[0]
            report.stage('postprocessing', 'extract', 0.6)
            if len(mesh.faces) == 0:
                raise EngineError('EMPTY_MESH', 'The model produced no surface for this image. Try another photo.', True)
            mesh.simplify(NVDIFFRAST_FACE_LIMIT)
            report.stage('postprocessing', 'bake', 0.7)
            textured = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs, coords=mesh.coords,
                attr_layout=self.pipeline.pbr_attr_layout, grid_size=grid_res, aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                # Upstream documents decimation_target as vertices (to_glb) and as faces (app.py). Half the face
                # budget stays within budget under either reading; gate V3 enforces the real face count.
                decimation_target=params['face_budget'] // 2, texture_size=params['texture_size'],
                remesh=True, remesh_band=1, remesh_project=0)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise EngineError('INFERENCE_OOM', 'The GPU ran out of memory. Try 512 resolution or External GPU.', True) from None
        gpu_seconds = time.monotonic() - started
        report.stage('postprocessing', 'cleanup', 0.8)
        textured, cleanup_report = cleanup(textured)
        glb = textured.export(file_type='glb', extension_webp=False)  # PNG textures: Blender/three.js safe
        torch.cuda.empty_cache()
        return EngineResult(glb=glb, gpu_seconds=round(gpu_seconds, 3), metadata={
            'engine': {'repo': 'github.com/microsoft/TRELLIS.2', 'commit': TRELLIS2_COMMIT, 'weights': WEIGHTS_REPO,
                       'weights_revision': self.weights_revision, 'pipeline_type': pipeline_type, 'samplers': SAMPLERS,
                       'components': self.components},
            'segmentation': seg_info,
            'cleanup': cleanup_report,
            'peak_vram_mb': int(torch.cuda.max_memory_allocated() // 2**20),
            'notices': NOTICES,
        })
