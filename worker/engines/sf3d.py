"""Stable Fast 3D adapter: the first local engine for small GPUs (e.g. 6 GB laptop GPUs) on Windows and Linux.

STATUS: NOT VERIFIED ON A GPU. Written against Stability-AI/stable-fast-3d at SF3D_COMMIT
(run.py and sf3d/system.py: SF3D.from_pretrained, model.run_image, sf3d.utils.resize_foreground).
Whether it fits a given GPU is never assumed: worker/selftest.py runs it on the real machine.

Memory-saving options (only what upstream supports, plus one exact refactor):
- texture (bake) resolution 512 / 1024 / 2048 (upstream --texture-resolution);
- bf16 autocast for the transformer on CUDA, as upstream run.py does;
- batch size 1, one job at a time (sequential);
- background removal on the CPU (rembg U^2-Net via onnxruntime), not on the GPU;
- low-VRAM mode: the marching-tetrahedra grid is queried in chunks instead of in one pass.
  The query and decoder are per-point, so the result is the same up to floating-point rounding;
  only peak memory changes;
- optional: move the model to system RAM while idle (SF3D_OFFLOAD_IDLE=1) to free VRAM for other apps.

Licence: Stability AI Community License (see docs/WINDOWS.md). Outputs belong to the user; commercial use
by organisations above US$1M annual revenue needs an enterprise licence; "Powered by Stability AI" attribution.
"""
from __future__ import annotations

import os
import time
from contextlib import nullcontext

from PIL import Image

from ..postprocess import cleanup
from ..segment import choose, make_segmenter, segment
from .base import Engine, EngineError, EngineResult, Reporter

SF3D_COMMIT = 'ff21fc491b4dc5314bf6734c7c0dabd86b5f5bb2'  # github.com/Stability-AI/stable-fast-3d, 2025-01-22
WEIGHTS_REPO = 'stabilityai/stable-fast-3d'  # gated: accept the licence on Hugging Face first
FOREGROUND_RATIO = 0.85  # upstream run.py default
DEFAULT_CHUNK = 262_144  # grid points per chunk in low-VRAM mode
NOTICES = ['Powered by Stability AI.',
           'This Stability AI Model is licensed under the Stability AI Community License, '
           'Copyright (c) Stability AI Ltd. All Rights Reserved.']


def chunked_triplane_to_meshes(model, triplanes, chunk: int, scale_tensor=None, cat=None):
    """sf3d.system.SF3D.triplane_to_meshes (at SF3D_COMMIT) with the grid query split into chunks.

    query_triplane and the decoder act on each grid point independently, so concatenating chunk
    results equals the single-pass result up to floating-point rounding (different batch sizes can
    take different kernel paths). scale_tensor/cat are injectable for CPU unit tests.
    """
    if scale_tensor is None:
        from sf3d.models.utils import scale_tensor  # noqa: PLC0415 - upstream helper
    if cat is None:
        import torch  # noqa: PLC0415

        def cat(parts):
            return torch.cat(parts, dim=-2)
    helper = model.isosurface_helper
    meshes = []
    for i in range(triplanes.shape[0]):
        triplane = triplanes[i]
        grid = scale_tensor(helper.grid_vertices.to(triplanes.device), helper.points_range, model.bbox)
        density, offset = [], []
        for start in range(0, grid.shape[0], chunk):
            values = model.query_triplane(grid[start:start + chunk], triplane)
            decoded = model.decoder(values, include=['vertex_offset', 'density'])
            density.append(decoded['density'])
            offset.append(decoded['vertex_offset'])
        sdf = cat(density) - model.cfg.isosurface_threshold
        deform = cat(offset).squeeze(0)
        mesh = helper(sdf.view(-1, 1), deform.view(-1, 3))
        mesh.v_pos = scale_tensor(mesh.v_pos, helper.points_range, model.bbox)
        meshes.append(mesh)
    return meshes


class Sf3dEngine(Engine):
    id = 'sf3d'
    textured = True
    selftest_ladder = ((2048, False), (2048, True), (1024, False), (1024, True), (512, True))

    def __init__(self):
        self.model = None
        self.weights_revision = os.environ.get('SF3D_WEIGHTS_REVISION')
        self.low_vram = False
        self.chunk = int(os.environ.get('SF3D_QUERY_CHUNK', DEFAULT_CHUNK))
        self.offload_idle = os.environ.get('SF3D_OFFLOAD_IDLE') == '1'
        self.device = 'cpu'

    @classmethod
    def install_problem(cls) -> str | None:
        """Checks code, compiled extensions (with their CUDA kernel) and cached weights without loading the model."""
        try:
            import torch  # noqa: PLC0415
        except ImportError:
            return 'PyTorch is not installed. Run scripts/windows/setup.ps1.'
        try:
            import sf3d.system  # noqa: F401, PLC0415
        except Exception as e:  # noqa: BLE001 - report any import failure verbatim
            return f'Stable Fast 3D code is not importable ({type(e).__name__}: {e}). Run scripts/windows/setup.ps1.'
        try:
            import texture_baker  # noqa: F401, PLC0415
            import uv_unwrapper  # noqa: F401, PLC0415
        except Exception as e:  # noqa: BLE001
            return f'SF3D extensions are not built ({e}). Install Visual Studio 2022 Build Tools and re-run setup.'
        if torch.cuda.is_available():
            try:
                tri = torch.tensor([[0, 1, 2]], dtype=torch.int32, device='cuda')
                uv = torch.tensor([[0.1, 0.1], [0.9, 0.1], [0.5, 0.9]], device='cuda')
                torch.ops.texture_baker_cpp.rasterize(uv, tri, 8)
            except (NotImplementedError, RuntimeError) as e:
                return ('texture_baker was built without its CUDA kernel, so baking cannot run on the GPU '
                        f'({str(e).splitlines()[0][:160]}). Install the CUDA Toolkit that matches PyTorch '
                        '(12.4) and re-run setup.')
        revision = os.environ.get('SF3D_WEIGHTS_REVISION')
        if not revision:
            return 'SF3D_WEIGHTS_REVISION is not set (setup records the downloaded revision).'
        from huggingface_hub import try_to_load_from_cache  # noqa: PLC0415

        for name in ('config.yaml', 'model.safetensors'):
            if not isinstance(try_to_load_from_cache(WEIGHTS_REPO, name, revision=revision), str):
                return (f'SF3D weights ({name}) are not downloaded. Accept the licence at '
                        'https://huggingface.co/stabilityai/stable-fast-3d and re-run setup.')
        return None

    def set_memory_mode(self, low_vram: bool) -> None:
        self.low_vram = low_vram

    def load(self) -> None:
        import torch  # noqa: PLC0415
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
        from sf3d.system import SF3D  # noqa: PLC0415

        if not self.weights_revision:
            raise RuntimeError('SF3D_WEIGHTS_REVISION must be set to the Hugging Face commit downloaded by setup.')
        config = hf_hub_download(WEIGHTS_REPO, 'config.yaml', revision=self.weights_revision)
        hf_hub_download(WEIGHTS_REPO, 'model.safetensors', revision=self.weights_revision)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = SF3D.from_pretrained(os.path.dirname(config), config_name='config.yaml',
                                          weight_name='model.safetensors')
        self.model.eval()
        self.model.to('cpu' if self.offload_idle else self.device)
        self.segmenter = make_segmenter(choose(os.environ.get('SEGMENTER', 'rembg-u2net'))).load()

    def _infer(self, image: Image.Image, texture_size: int):
        import torch  # noqa: PLC0415

        model = self.model
        if self.low_vram:  # per-instance override; the class method stays untouched
            model.triplane_to_meshes = lambda triplanes: chunked_triplane_to_meshes(model, triplanes, self.chunk)
        elif 'triplane_to_meshes' in vars(model):
            del model.triplane_to_meshes
        cuda = self.device == 'cuda'
        autocast = torch.autocast(device_type='cuda', dtype=torch.bfloat16) if cuda else nullcontext()
        with torch.no_grad(), autocast:  # upstream run.py: bf16 autocast on CUDA
            return model.run_image(image, bake_resolution=texture_size, remesh='none', vertex_count=-1)

    def run(self, image: Image.Image, params: dict, report: Reporter) -> EngineResult:
        import torch  # noqa: PLC0415
        from sf3d.utils import resize_foreground  # noqa: PLC0415

        texture = params['texture_size']
        report.stage('preparing', 'segment', 0.05)
        rgba, seg_info = segment(image, self.segmenter)
        prepped = resize_foreground(rgba, FOREGROUND_RATIO)
        cuda = self.device == 'cuda'
        if self.offload_idle:
            self.model.to(self.device)
        report.stage('running', 'infer', 0.2)
        if cuda:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(params['seed'])  # SF3D is feed-forward; recorded for completeness
        started = time.monotonic()
        try:
            mesh, _ = self._infer(prepped, texture)
        except torch.cuda.OutOfMemoryError:
            raise EngineError('INFERENCE_OOM', f'The GPU ran out of memory at {texture} px textures. Choose a smaller '
                              'texture size, or close other programs that use the GPU, and try again.', True) from None
        finally:
            if cuda:
                torch.cuda.empty_cache()
        seconds = time.monotonic() - started
        peak = int(torch.cuda.max_memory_allocated() // 2**20) if cuda else None
        if self.offload_idle:
            self.model.to('cpu')
            if cuda:
                torch.cuda.empty_cache()
        if len(mesh.faces) == 0:
            raise EngineError('EMPTY_MESH', 'The model produced no surface for this image. Try another photo.', True)
        report.stage('postprocessing', 'cleanup', 0.75)
        mesh, cleanup_report = cleanup(mesh)
        glb = mesh.export(file_type='glb', include_normals=True)  # as upstream run.py
        return EngineResult(glb=glb, gpu_seconds=round(seconds, 3) if cuda else None, metadata={
            'engine': {'repo': 'github.com/Stability-AI/stable-fast-3d', 'commit': SF3D_COMMIT, 'weights': WEIGHTS_REPO,
                       'weights_revision': self.weights_revision, 'device': self.device, 'texture_size': texture,
                       'low_vram': self.low_vram, 'query_chunk': self.chunk if self.low_vram else None,
                       'precision': 'bf16 autocast' if cuda else 'fp32', 'remesh': 'none',
                       'foreground_ratio': FOREGROUND_RATIO},
            'segmentation': seg_info,
            'cleanup': cleanup_report,
            'peak_vram_mb': peak,
            'notices': NOTICES,
        })
