"""Self-test ladder, caching, out-of-memory step-down and report-driven worker behaviour.

The engine here is a MOCK (FakeGpuEngine): it raises INFERENCE_OOM above a chosen texture size and
otherwise returns an SF3D-style fixture GLB. It tests the decision logic, not real GPU memory.
"""
import pytest
from fastapi.testclient import TestClient

from app.internal import create_internal_app
from worker import selftest
from worker.engines.base import Engine, EngineError, EngineResult
from worker.main import ApiClient, Worker, prepare_engines

from .helpers import create_job, upload
from .stub_engine import FIX

RTX4050 = {'backend': 'cuda', 'devices': [{'name': 'NVIDIA GeForce RTX 4050 Laptop GPU', 'vram_total_mb': 6141,
                                           'vram_free_mb': 5329, 'driver': '566.36'}], 'torch': {'version': '2.6.0'}}
TOKEN = 'tok-worker-one-0123456789'


class FakeGpuEngine(Engine):
    """MOCK: 'fits' only up to max_ok texture size (and only in low-VRAM mode if low_needed)."""
    id = 'sf3d'
    textured = True
    selftest_ladder = ((2048, False), (2048, True), (1024, False), (1024, True), (512, True))
    weights_revision = 'rev1'

    def __init__(self, max_ok=1024, low_needed=False, crash=None, fixture='sf3d_style_{size}.glb'):
        self.max_ok, self.low_needed, self.crash, self.fixture = max_ok, low_needed, crash, fixture
        self.low_vram, self.runs = False, []

    def set_memory_mode(self, low_vram):
        self.low_vram = low_vram

    def load(self):
        pass

    def run(self, image, params, report):
        size = params['texture_size']
        self.runs.append((size, self.low_vram))
        if self.crash:
            raise self.crash
        if size > self.max_ok or (self.low_needed and not self.low_vram):
            raise EngineError('INFERENCE_OOM', f'out of memory at {size}', True)
        name = self.fixture.format(size=size if size in (512, 1024) else 1024)
        return EngineResult(glb=(FIX / name).read_bytes(), metadata={'peak_vram_mb': 4000 + size, 'mock': True})


def validator(glb, **kw):
    from worker.validate import validate_glb
    return validate_glb(glb, **kw)


def test_ladder_picks_best_setting_that_fits():
    engine = FakeGpuEngine(max_ok=1024)
    r = selftest.run_selftest(engine, validator, 'basecolor_normal', log=lambda m: None)
    assert (r['status'], r['max_texture_size'], r['low_vram'], r['peak_vram_mb']) == ('passed', 1024, False, 5024)
    assert engine.runs == [(2048, False), (2048, True), (1024, False)]
    assert [a['result'] for a in r['attempts']] == ['INFERENCE_OOM', 'INFERENCE_OOM', 'passed']


def test_ladder_uses_low_vram_mode_when_needed():
    r = selftest.run_selftest(FakeGpuEngine(max_ok=512, low_needed=True), validator, 'basecolor_normal', log=lambda m: None)
    assert (r['status'], r['max_texture_size'], r['low_vram']) == ('passed', 512, True)


def test_all_steps_out_of_memory():
    r = selftest.run_selftest(FakeGpuEngine(max_ok=256), validator, 'basecolor_normal', log=lambda m: None)
    assert r['status'] == 'out_of_memory' and 'down to 512 px' in r['message'] and r['max_texture_size'] is None


def test_other_errors_fail_without_trying_more():
    engine = FakeGpuEngine(crash=RuntimeError('texture_baker kernel missing'))
    r = selftest.run_selftest(engine, validator, 'basecolor_normal', log=lambda m: None)
    assert r['status'] == 'failed' and 'texture_baker kernel missing' in r['message'] and len(engine.runs) == 1


def test_invalid_output_fails_selftest():
    engine = FakeGpuEngine(fixture='bad_nan.glb')
    r = selftest.run_selftest(engine, validator, 'basecolor_normal', log=lambda m: None)
    assert r['status'] == 'failed' and 'failed validation' in r['message']


def test_cache_reused_until_fingerprint_changes(tmp_path):
    store = selftest.SelfTestStore(tmp_path / 'selftest.json')
    engine = FakeGpuEngine(max_ok=512, low_needed=True)
    first = selftest.ensure_selftest(engine, RTX4050, store, validator, 'basecolor_normal', log=lambda m: None)
    runs = len(engine.runs)
    again = selftest.ensure_selftest(engine, RTX4050, store, validator, 'basecolor_normal', log=lambda m: None)
    assert again == first and len(engine.runs) == runs and engine.low_vram is True  # mode restored from cache
    new_driver = RTX4050 | {'devices': [RTX4050['devices'][0] | {'driver': '572.16'}]}
    selftest.ensure_selftest(engine, new_driver, store, validator, 'basecolor_normal', log=lambda m: None)
    assert len(engine.runs) > runs


def test_step_down_after_real_oom():
    engine = FakeGpuEngine()
    passed = {'status': 'passed', 'max_texture_size': 1024, 'low_vram': False}
    r = selftest.step_down(engine, passed, 1024)
    assert (r['status'], r['max_texture_size'], r['low_vram']) == ('passed', 1024, True)
    r = selftest.step_down(engine, r, 1024)
    assert (r['max_texture_size'], r['low_vram']) == (512, True)
    r = selftest.step_down(engine, r, 512)
    assert r['status'] == 'out_of_memory' and r['max_texture_size'] is None


def test_state_dir_on_windows(monkeypatch):
    monkeypatch.delenv('MAKER_WORKER_STATE', raising=False)
    monkeypatch.setattr(selftest.platform, 'system', lambda: 'Windows')
    monkeypatch.setenv('LOCALAPPDATA', r'C:\Users\me\AppData\Local')
    assert str(selftest.state_dir()).endswith('3D-MAKER')


# --- engine preparation ---------------------------------------------------------------------------

class MissingEngine(FakeGpuEngine):
    @classmethod
    def install_problem(cls):
        return 'SF3D weights are not downloaded.'


class BrokenLoad(FakeGpuEngine):
    def load(self):
        raise OSError('CUDA error: no kernel image is available')


def test_prepare_engines_reports_not_installed_and_load_failures():
    loaded, reports = prepare_engines(['sf3d'], classes={'sf3d': MissingEngine})
    assert loaded == {} and reports['sf3d'] == {'installed': False, 'problem': 'SF3D weights are not downloaded.',
                                                'selftest': None}
    loaded, reports = prepare_engines(['sf3d'], classes={'sf3d': BrokenLoad})
    assert loaded == {} and 'no kernel image' in reports['sf3d']['problem']
    loaded, reports = prepare_engines(['sf3d'], classes={'sf3d': FakeGpuEngine})
    assert list(loaded) == ['sf3d'] and reports['sf3d']['installed'] is True


def test_prepare_engines_unknown_id():
    _, reports = prepare_engines(['magic'])
    assert reports['magic']['installed'] is False


# --- worker <-> API with reports ------------------------------------------------------------------

@pytest.fixture()
def internal(api):
    return TestClient(create_internal_app(context=api.app.state.ctx), headers={'Authorization': f'Bearer {TOKEN}'})


def make_worker(internal, tmp_path, engine):
    reports = {'sf3d': {'installed': True, 'problem': None, 'selftest': None}}
    return Worker(ApiClient(internal), {'sf3d': engine}, RTX4050, heartbeat_interval=3600, reports=reports,
                  store=selftest.SelfTestStore(tmp_path / 'st.json'))


def sf3d_status(api):
    targets = {t['id']: t for t in api.get('/api/v1/gpu-targets').json()['targets']}
    return {s['id']: s for s in targets['internal']['engine_status']}['sf3d'], targets['internal']


def test_untested_engine_is_not_offered_and_never_claims(api, internal, tmp_path):
    worker = make_worker(internal, tmp_path, FakeGpuEngine())
    worker.api.heartbeat(worker.probe, worker.engine_ids)
    status, target = sf3d_status(api)
    assert status['status'] == 'untested' and target['available'] is False
    assert worker.ready_engine_ids == []


def test_selftest_then_job_then_runtime_oom_steps_down(api, internal, tmp_path):
    engine = FakeGpuEngine(max_ok=1024)
    worker = make_worker(internal, tmp_path, engine)
    worker.run_selftests(log_fn=lambda m: None)
    worker.api.heartbeat(worker.probe, worker.engine_ids)
    status, target = sf3d_status(api)
    assert (status['status'], status['max_texture_size'], status['status_label']) == ('ready', 1024, 'Ready')
    assert target['engines'][0]['max_texture_size'] == 1024
    u = upload(api)
    assert create_job(api, u, engine='sf3d', params={'texture_size': 2048}).json()['error']['code'] == 'TEXTURE_TOO_LARGE'
    job_id = create_job(api, u, engine='sf3d', params={'texture_size': 1024}).json()['job_id']
    assert worker.run_once() and api.get(f'/api/v1/jobs/{job_id}').json()['status'] == 'completed'
    # The GPU gets busier (another program): the next 1024 px job runs out of memory.
    engine.max_ok = 512
    job_id = create_job(api, u, engine='sf3d', params={'texture_size': 1024}).json()['job_id']
    worker.run_once()
    assert api.get(f'/api/v1/jobs/{job_id}').json()['error']['code'] == 'INFERENCE_OOM'
    worker.api.heartbeat(worker.probe, worker.engine_ids)
    status, _ = sf3d_status(api)
    assert (status['status'], status['max_texture_size']) == ('ready', 1024)  # next step: 1024 + low-VRAM
    assert engine.low_vram is True
    assert selftest.SelfTestStore(tmp_path / 'st.json').get('sf3d', selftest.fingerprint(engine, RTX4050))['low_vram']


def test_out_of_memory_status_reaches_api(api, internal, tmp_path):
    worker = make_worker(internal, tmp_path, FakeGpuEngine(max_ok=128))
    worker.run_selftests(log_fn=lambda m: None)
    worker.api.heartbeat(worker.probe, worker.engine_ids)
    status, target = sf3d_status(api)
    assert status['status'] == 'out_of_memory' and status['status_label'] == 'Out of memory'
    assert target['available'] is False and 'Out of memory' in target['reason']
