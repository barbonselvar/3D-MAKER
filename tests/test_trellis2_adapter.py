"""Glue logic of worker/engines/trellis2.py against a MOCK of the upstream API surface.

These tests check that the adapter calls TRELLIS.2 the way commit TRELLIS2_COMMIT's
app.py does (seed, pipeline type, samplers, no built-in rembg, PNG export, OOM
mapping). They do NOT show that TRELLIS.2 runs: that needs a >= 24 GB NVIDIA GPU.
"""
import sys
import types

import numpy as np
import pytest
import trimesh
from PIL import Image

from worker.engines import trellis2
from worker.engines.base import EngineError


class OOM(Exception):
    pass


class Recorder:
    def __init__(self):
        self.stages = []

    def stage(self, status, stage, progress=None):
        self.stages.append((status, stage))

    def check_cancel(self):
        pass


@pytest.fixture()
def fake_stack(monkeypatch):
    calls = {}

    class FakeMesh:
        vertices = faces = np.zeros((10, 3))
        attrs = coords = None

        def simplify(self, n):
            calls['simplify'] = n

    class FakePipeline:
        pbr_attr_layout = {'base_color': slice(0, 3)}
        _pretrained_args = {'image_cond_model': {'name': 'DinoV3FeatureExtractor'},
                            'rembg_model': {'name': 'BiRefNet', 'args': {'model_name': 'briaai/RMBG-2.0'}}}

        @classmethod
        def from_pretrained(cls, path):
            calls['weights_path'] = path
            calls['rembg_cls_during_load'] = rembg_mod.BiRefNet
            return cls()

        def cuda(self):
            calls['cuda'] = True

        def preprocess_image(self, image):
            calls['preprocess_mode'] = image.mode
            return image.convert('RGB')

        def run(self, image, **kw):
            calls['run'] = kw
            if calls.get('oom'):
                raise OOM()
            return [FakeMesh()], (None, None, 1024)

    def to_glb(**kw):
        calls['to_glb'] = kw
        box = trimesh.creation.box()
        box.visual = trimesh.visual.TextureVisuals(uv=np.zeros((len(box.vertices), 2)),
                                                   material=trimesh.visual.material.PBRMaterial())
        return box

    class RealBiRefNet:
        pass

    rembg_mod = types.SimpleNamespace(BiRefNet=RealBiRefNet)
    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(is_available=lambda: True, reset_peak_memory_stats=lambda: None,
                                   max_memory_allocated=lambda: 30 * 2**30, empty_cache=lambda: None,
                                   OutOfMemoryError=OOM))
    pipelines = types.SimpleNamespace(Trellis2ImageTo3DPipeline=FakePipeline, rembg=rembg_mod)
    modules = {'torch': torch, 'trellis2': types.SimpleNamespace(pipelines=pipelines), 'trellis2.pipelines': pipelines,
               'trellis2.pipelines.rembg': rembg_mod,
               'huggingface_hub': types.SimpleNamespace(snapshot_download=lambda repo, revision: f'/models/{repo}@{revision}'),
               'o_voxel': types.SimpleNamespace(postprocess=types.SimpleNamespace(to_glb=to_glb))}
    for name, mod in modules.items():
        monkeypatch.setitem(sys.modules, name, mod)

    class FakeSegmenter:
        spec = trellis2.choose('birefnet')

        def __init__(self, spec):
            calls['segmenter'] = spec.model

        def load(self):
            return self

        def mask(self, rgb):
            m = np.zeros((rgb.size[1], rgb.size[0]), np.uint8)
            m[10:90, 10:90] = 255
            return Image.fromarray(m)

    monkeypatch.setattr(trellis2, 'BiRefNetSegmenter', FakeSegmenter)
    monkeypatch.setenv('TRELLIS2_WEIGHTS_REVISION', 'abc123')
    calls['rembg_mod'] = rembg_mod
    return calls


PARAMS = {'seed': 99, 'resolution': 1024, 'texture_size': 2048, 'face_budget': 200_000}


def run_engine(fake_stack, params=PARAMS):
    engine = trellis2.Trellis2Engine()
    engine.load()
    rec = Recorder()
    return engine, engine.run(Image.new('RGB', (100, 100), (90, 90, 90)), params, rec), rec


def test_requires_pinned_weights(monkeypatch, fake_stack):
    monkeypatch.delenv('TRELLIS2_WEIGHTS_REVISION')
    with pytest.raises(RuntimeError, match='TRELLIS2_WEIGHTS_REVISION'):
        trellis2.Trellis2Engine().load()


def test_load_never_instantiates_builtin_rembg(fake_stack):
    run_engine(fake_stack)
    assert fake_stack['rembg_cls_during_load'] is trellis2._NoRembg
    assert fake_stack['rembg_mod'].BiRefNet.__name__ == 'RealBiRefNet'  # restored afterwards
    assert fake_stack['weights_path'] == '/models/microsoft/TRELLIS.2-4B@abc123'
    assert fake_stack['segmenter'] == 'ZhengPeng7/BiRefNet'


def test_run_matches_upstream_app_usage(fake_stack):
    _, result, rec = run_engine(fake_stack)
    run = fake_stack['run']
    assert run['seed'] == 99 and run['pipeline_type'] == '1024_cascade' and run['preprocess_image'] is False
    assert run['return_latent'] is True and run['tex_slat_sampler_params']['guidance_strength'] == 1.0
    assert fake_stack['preprocess_mode'] == 'RGBA'  # our mask, so the pipeline only crops
    assert fake_stack['simplify'] == 16_777_216
    glb_args = fake_stack['to_glb']
    assert glb_args['decimation_target'] == 100_000 and glb_args['texture_size'] == 2048 and glb_args['grid_size'] == 1024
    assert result.glb[:4] == b'glTF' and b'EXT_texture_webp' not in result.glb
    meta = result.metadata
    assert meta['engine']['commit'] == trellis2.TRELLIS2_COMMIT and meta['engine']['weights_revision'] == 'abc123'
    assert meta['peak_vram_mb'] == 30 * 1024 and 'Built with DINOv3.' in meta['notices']
    assert meta['engine']['components']['pipeline_rembg_model_not_used']['args']['model_name'] == 'briaai/RMBG-2.0'
    assert [s for s, _ in rec.stages] == ['preparing', 'running', 'postprocessing', 'postprocessing', 'postprocessing']


@pytest.mark.parametrize('res,ptype', [(512, '512'), (1536, '1536_cascade')])
def test_resolution_mapping(fake_stack, res, ptype):
    run_engine(fake_stack, PARAMS | {'resolution': res})
    assert fake_stack['run']['pipeline_type'] == ptype


def test_oom_maps_to_inference_oom(fake_stack):
    fake_stack['oom'] = True
    with pytest.raises(EngineError) as e:
        run_engine(fake_stack)
    assert e.value.code == 'INFERENCE_OOM' and e.value.retryable
