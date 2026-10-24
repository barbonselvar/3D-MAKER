"""worker/engines/sf3d.py against a MOCK of Stable Fast 3D's API at SF3D_COMMIT.

Checks our wiring (weights pinning, bf16 autocast, bake resolution, low-VRAM chunking, CPU background
removal, OOM mapping, GLB export) and the install check. It does NOT show SF3D runs or fits in 6 GB:
that needs the real GPU (docs/WINDOWS.md, "Verify on your PC").
"""
import sys
import types
from contextlib import contextmanager

import numpy as np
import pytest
import trimesh
from PIL import Image

from worker.engines import sf3d
from worker.engines.base import EngineError

from .stub_engine import FIX


class OOM(Exception):
    pass


class Rec:
    def __init__(self):
        self.stages = []

    def stage(self, status, stage, progress=None):
        self.stages.append(stage)

    def check_cancel(self):
        pass


@pytest.fixture()
def fake(monkeypatch):
    calls = {'autocast': [], 'to': []}

    class FakeModel:
        def eval(self):
            return self

        def to(self, device):
            calls['to'].append(device)
            return self

        def run_image(self, image, bake_resolution, remesh, vertex_count):
            calls['run_image'] = {'mode': image.mode, 'size': image.size, 'bake': bake_resolution, 'remesh': remesh,
                                  'vertex_count': vertex_count, 'overridden': 'triplane_to_meshes' in vars(self)}
            if calls.get('oom'):
                raise OOM()
            if calls.get('empty'):
                return trimesh.Trimesh(), {}
            return trimesh.load(FIX / 'sf3d_style_512.glb', file_type='glb', force='mesh'), {}

    class SF3D:
        @classmethod
        def from_pretrained(cls, path, config_name, weight_name):
            calls['from_pretrained'] = (path, config_name, weight_name)
            return FakeModel()

    @contextmanager
    def autocast(device_type, dtype):
        calls['autocast'].append((device_type, dtype))
        yield

    @contextmanager
    def no_grad():
        yield

    torch = types.SimpleNamespace(
        bfloat16='bf16', autocast=autocast, no_grad=no_grad, manual_seed=lambda s: calls.setdefault('seed', s),
        cuda=types.SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None, reset_peak_memory_stats=lambda: None,
                                   max_memory_allocated=lambda: 4321 * 2**20, OutOfMemoryError=OOM))
    downloads = []

    def hf_hub_download(repo, name, revision):
        downloads.append((repo, name, revision))
        return f'/hf/{repo}/{revision}/{name}'

    modules = {
        'torch': torch,
        'huggingface_hub': types.SimpleNamespace(hf_hub_download=hf_hub_download),
        'sf3d': types.ModuleType('sf3d'),
        'sf3d.system': types.SimpleNamespace(SF3D=SF3D),
        'sf3d.utils': types.SimpleNamespace(resize_foreground=lambda img, ratio: calls.setdefault('ratio', ratio) and img),
    }
    for name, mod in modules.items():
        monkeypatch.setitem(sys.modules, name, mod)

    class FakeSegmenter:
        spec = sf3d.choose('rembg-u2net')

        def load(self):
            calls['segmenter'] = self.spec.name
            return self

        def mask(self, rgb):
            m = np.zeros((rgb.size[1], rgb.size[0]), np.uint8)
            m[20:80, 20:80] = 255
            return Image.fromarray(m)

    monkeypatch.setattr(sf3d, 'make_segmenter', lambda spec: FakeSegmenter())
    monkeypatch.setenv('SF3D_WEIGHTS_REVISION', 'abc123')
    monkeypatch.delenv('SEGMENTER', raising=False)
    calls['downloads'] = downloads
    return calls


PARAMS = {'seed': 5, 'texture_size': 1024, 'resolution': 1024, 'face_budget': 200_000}


def run(fake, params=PARAMS, low_vram=False, offload=False, monkeypatch=None):
    engine = sf3d.Sf3dEngine()
    engine.offload_idle = offload
    engine.load()
    engine.set_memory_mode(low_vram)
    rec = Rec()
    return engine, engine.run(Image.new('RGB', (100, 100), (200, 200, 200)), params, rec), rec


def test_load_pins_weights_and_uses_cpu_segmenter(fake):
    engine, result, rec = run(fake)
    assert fake['downloads'] == [('stabilityai/stable-fast-3d', 'config.yaml', 'abc123'),
                                 ('stabilityai/stable-fast-3d', 'model.safetensors', 'abc123')]
    assert fake['from_pretrained'] == ('/hf/stabilityai/stable-fast-3d/abc123', 'config.yaml', 'model.safetensors')
    assert fake['segmenter'] == 'rembg-u2net'
    assert result.metadata['segmentation']['method'] == 'rembg-u2net'


def test_run_matches_upstream_run_py(fake):
    _, result, rec = run(fake)
    call = fake['run_image']
    assert call['mode'] == 'RGBA' and call['bake'] == 1024 and call['remesh'] == 'none' and call['vertex_count'] == -1
    assert fake['autocast'] == [('cuda', 'bf16')] and fake['ratio'] == 0.85 and fake['seed'] == 5
    assert rec.stages == ['segment', 'infer', 'cleanup']
    assert result.glb[:4] == b'glTF' and result.metadata['peak_vram_mb'] == 4321
    meta = result.metadata['engine']
    assert meta['commit'] == sf3d.SF3D_COMMIT and meta['weights_revision'] == 'abc123' and meta['low_vram'] is False
    assert 'Powered by Stability AI.' in result.metadata['notices']


def test_low_vram_mode_installs_chunked_query_per_instance(fake):
    _, result, _ = run(fake, low_vram=True)
    assert fake['run_image']['overridden'] is True and result.metadata['engine']['query_chunk'] == sf3d.DEFAULT_CHUNK


def test_offload_idle_moves_model_between_devices(fake):
    run(fake, offload=True)
    assert fake['to'] == ['cpu', 'cuda', 'cpu']


def test_oom_maps_to_inference_oom(fake):
    fake['oom'] = True
    with pytest.raises(EngineError) as e:
        run(fake)
    assert e.value.code == 'INFERENCE_OOM' and '1024 px' in e.value.message and e.value.retryable


def test_empty_mesh_maps_to_empty_mesh(fake):
    fake['empty'] = True
    with pytest.raises(EngineError) as e:
        run(fake)
    assert e.value.code == 'EMPTY_MESH'


def test_requires_weights_revision(fake, monkeypatch):
    monkeypatch.delenv('SF3D_WEIGHTS_REVISION')
    with pytest.raises(RuntimeError, match='SF3D_WEIGHTS_REVISION'):
        sf3d.Sf3dEngine().load()


# --- chunked grid query equals the single-pass upstream computation -------------------------------

class GridModel:
    """NumPy stand-in: per-point query and decoder, like SF3D's grid_sample + MLP."""

    def __init__(self, n):
        rng = np.random.default_rng(0)
        self.points = rng.random((n, 3))
        self.cfg = types.SimpleNamespace(isosurface_threshold=10.0)
        self.bbox = None
        outer = self

        class Helper:
            points_range = (0, 1)

            class grid_vertices:  # noqa: N801 - mimics a tensor with .to()
                @staticmethod
                def to(device):
                    return outer.points

            def __call__(self, sdf, deform):
                return types.SimpleNamespace(sdf=sdf, deform=deform, v_pos=np.zeros((1, 3)))

        self.isosurface_helper = Helper()

    def query_triplane(self, positions, triplane):
        return (positions @ triplane)[None]  # [1, N, F]

    def decoder(self, values, include):
        return {'density': values[..., :1] * 3.0, 'vertex_offset': np.tanh(values)}


def single_pass(model, triplanes):
    out = []
    for t in triplanes:
        values = model.query_triplane(model.points, t)
        dec = model.decoder(values, include=None)
        out.append(((dec['density'] - 10.0).reshape(-1, 1), dec['vertex_offset'].squeeze(0).reshape(-1, 3)))
    return out


class TensorLike(np.ndarray):
    """ndarray with torch's .view(*shape) meaning (reshape)."""

    def view(self, *args):
        if args and all(isinstance(a, int) for a in args):
            return np.asarray(self).reshape(*args)
        return super().view(*args)


@pytest.mark.parametrize('chunk', [1, 7, 1000, 5000])
def test_chunked_query_equals_single_pass(chunk):
    model = GridModel(1000)
    planes = [np.eye(3), np.arange(9).reshape(3, 3) / 9]
    batch = type('Triplanes', (), {'shape': (2,), 'device': 'cpu', '__getitem__': lambda self, i: planes[i]})()
    meshes = sf3d.chunked_triplane_to_meshes(model, batch, chunk, scale_tensor=lambda x, a, b: x,
                                             cat=lambda parts: np.concatenate(parts, axis=-2).view(TensorLike))
    # Equal up to floating-point rounding: different batch sizes take different BLAS paths (seen in CI: 1 ulp).
    for mesh, (sdf, deform) in zip(meshes, single_pass(model, planes), strict=True):
        np.testing.assert_allclose(mesh.sdf, sdf, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(mesh.deform, deform, rtol=1e-12, atol=1e-12)


# --- install check --------------------------------------------------------------------------------

def test_install_problem_reports_missing_torch(monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', None)
    assert 'PyTorch is not installed' in sf3d.Sf3dEngine.install_problem()


def test_install_problem_reports_cpu_only_texture_baker(monkeypatch):
    def rasterize(*a):
        raise NotImplementedError("Could not run 'texture_baker_cpp::rasterize' with arguments from the 'CUDA' backend.")
    torch = types.SimpleNamespace(
        int32='int32', tensor=lambda *a, **k: None, cuda=types.SimpleNamespace(is_available=lambda: True),
        ops=types.SimpleNamespace(texture_baker_cpp=types.SimpleNamespace(rasterize=rasterize)))
    for name, mod in {'torch': torch, 'sf3d': types.ModuleType('sf3d'), 'sf3d.system': types.ModuleType('sf3d.system'),
                      'texture_baker': types.ModuleType('texture_baker'),
                      'uv_unwrapper': types.ModuleType('uv_unwrapper')}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    problem = sf3d.Sf3dEngine.install_problem()
    assert 'built without its CUDA kernel' in problem and 'CUDA Toolkit' in problem


def test_install_problem_reports_missing_weights(monkeypatch):
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    hub = types.SimpleNamespace(try_to_load_from_cache=lambda repo, name, revision: None)
    for name, mod in {'torch': torch, 'sf3d': types.ModuleType('sf3d'), 'sf3d.system': types.ModuleType('sf3d.system'),
                      'texture_baker': types.ModuleType('texture_baker'), 'uv_unwrapper': types.ModuleType('uv_unwrapper'),
                      'huggingface_hub': hub}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setenv('SF3D_WEIGHTS_REVISION', 'abc123')
    assert 'weights (config.yaml) are not downloaded' in sf3d.Sf3dEngine.install_problem()
