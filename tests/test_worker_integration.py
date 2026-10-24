"""P1-07: worker loop <-> internal API <-> public API, using the MOCK StubEngine (tests/stub_engine.py)."""
import hashlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.internal import create_internal_app
from worker import engines as engine_registry
from worker.engines.base import EngineError
from worker.main import ApiClient, Worker

from .helpers import L40S_PROBE, create_job, upload
from .stub_engine import FIX, StubEngine

TOKEN = 'tok-worker-one-0123456789'  # matches the `settings` fixture


@pytest.fixture()
def internal(api):
    return TestClient(create_internal_app(context=api.app.state.ctx), headers={'Authorization': f'Bearer {TOKEN}'})


def make_worker(internal, engine=None):
    return Worker(ApiClient(internal), {'trellis2-4b': engine or StubEngine()}, L40S_PROBE, heartbeat_interval=3600)


def submit(api, worker):
    worker.api.heartbeat(worker.probe, worker.engine_ids)  # worker online -> internal target available
    r = create_job(api, upload(api), params={'seed': 42, 'texture_size': 1024})
    assert r.status_code == 202, r.text
    return r.json()['job_id']


def test_end_to_end_job_produces_validated_downloadable_glb(api, internal):
    engine = StubEngine()
    worker = make_worker(internal, engine)
    job_id = submit(api, worker)
    assert worker.run_once() is True
    job = api.get(f'/api/v1/jobs/{job_id}').json()
    assert job['status'] == 'completed', job
    assert job['gpu_seconds'] == 1.5 and engine.seen['seed'] == 42 and engine.seen['size'] == (64, 64)
    events = api.get(f'/api/v1/jobs/{job_id}/events').text
    for status in ('assigned', 'preparing', 'running', 'postprocessing', 'validating', 'exporting', 'completed'):
        assert f'"status":"{status}"' in events
    model = api.get(f"/api/v1/models/{job['model_id']}").json()
    files = {f['format']: f for f in model['files']}
    assert files['glb']['validated'] and files['glb']['validation']['passed']
    assert model['manifest']['params']['seed'] == 42
    assert model['manifest']['engine']['mock'] is True  # the manifest says this was the mock
    assert model['manifest']['input_sha256']
    assert 'MIT' in model['manifest']['licences'][0]
    r = api.get(files['glb']['url'], follow_redirects=False)
    assert r.status_code == 302 and '/api/v1/files/' in r.headers['location']
    glb = api.get(r.headers['location']).content
    assert hashlib.sha256(glb).hexdigest() == hashlib.sha256((FIX / 'pbr_cube.glb').read_bytes()).hexdigest()


def test_validation_failure_publishes_nothing(api, internal):
    worker = make_worker(internal, StubEngine(fixture='bad_nan.glb'))
    job_id = submit(api, worker)
    worker.run_once()
    job = api.get(f'/api/v1/jobs/{job_id}').json()
    assert job['status'] == 'failed' and job['error']['code'] == 'VALIDATION_FAILED'
    assert 'V3' in job['error']['message'] and job['model_id'] is None
    with api.app.state.ctx.db.connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM models').fetchone()[0] == 0


def test_engine_error_maps_to_job_error(api, internal):
    oom = EngineError('INFERENCE_OOM', 'The GPU ran out of memory. Try 512 resolution or External GPU.', retryable=True)
    worker = make_worker(internal, StubEngine(error=oom))
    job_id = submit(api, worker)
    worker.run_once()
    err = api.get(f'/api/v1/jobs/{job_id}').json()['error']
    assert err == {'code': 'INFERENCE_OOM', 'message': oom.message, 'retryable': True}


def test_unexpected_crash_is_reported_not_hung(api, internal):
    class Boom(StubEngine):
        def run(self, image, params, report):
            raise RuntimeError('kaboom')
    worker = make_worker(internal, Boom())
    job_id = submit(api, worker)
    worker.run_once()
    job = api.get(f'/api/v1/jobs/{job_id}').json()
    assert job['status'] == 'failed' and job['error']['code'] == 'ENGINE_ERROR' and 'kaboom' in job['error']['message']


def test_cancel_during_inference_stops_at_next_stage(api, internal):
    holder = {}

    def cancel_on_infer(stage):
        if stage == 'infer':
            api.post(f"/api/v1/jobs/{holder['job']}/cancel")
    engine = StubEngine(on_stage=cancel_on_infer)
    worker = make_worker(internal, engine)
    holder['job'] = submit(api, worker)
    worker.run_once()
    job = api.get(f"/api/v1/jobs/{holder['job']}").json()
    assert job['status'] == 'cancelled' and job['model_id'] is None
    assert '"status":"postprocessing"' not in api.get(f"/api/v1/jobs/{holder['job']}/events").text


def test_heartbeat_reports_cancel(api, internal):
    worker = make_worker(internal)
    job_id = submit(api, worker)
    job = worker.api.claim(worker.engine_ids)
    assert worker.api.heartbeat(worker.probe, worker.engine_ids, job['id']) is False
    api.post(f'/api/v1/jobs/{job_id}/cancel')
    assert worker.api.heartbeat(worker.probe, worker.engine_ids, job['id']) is True


def test_claim_returns_nothing_when_idle(internal):
    worker = make_worker(internal)
    assert worker.run_once() is False


def test_worker_auth_and_isolation(api, internal, settings):
    bad = TestClient(create_internal_app(context=api.app.state.ctx), headers={'Authorization': 'Bearer nope'})
    assert bad.post('/internal/v1/claim', json={'engines': ['trellis2-4b']}).status_code == 401
    assert TestClient(internal.app).post('/internal/v1/claim', json={'engines': []}).status_code == 401
    # Worker routes are not served by the public app at all.
    assert api.post('/internal/v1/claim', json={'engines': ['trellis2-4b']},
                    headers={'Authorization': f'Bearer {TOKEN}'}).status_code in (404, 405)
    # A second worker cannot touch the first worker's job.
    settings.worker_tokens['tok-worker-two-0123456789'] = 'w2'
    worker = make_worker(internal)
    job_id = submit(api, worker)
    worker.api.claim(worker.engine_ids)
    other = TestClient(internal.app, headers={'Authorization': 'Bearer tok-worker-two-0123456789'})
    assert other.post(f'/internal/v1/jobs/{job_id}/progress', json={'status': 'running'}).status_code == 404
    assert other.get(f'/internal/v1/jobs/{job_id}/input').status_code == 404


def test_complete_requires_exporting_glb_and_passed_validation(api, internal):
    worker = make_worker(internal)
    job_id = submit(api, worker)
    worker.api.claim(worker.engine_ids)
    good = {'manifest': {}, 'validation': {'glb': {'passed': True}}}
    assert internal.post(f'/internal/v1/jobs/{job_id}/complete', json=good).status_code == 409  # still assigned
    for status in ('preparing', 'validating', 'exporting'):
        assert internal.post(f'/internal/v1/jobs/{job_id}/progress', json={'status': status}).status_code == 200
    assert internal.post(f'/internal/v1/jobs/{job_id}/complete', json=good).json()['error']['code'] == 'ARTIFACT_MISSING'
    glb = (FIX / 'pbr_cube.glb').read_bytes()
    assert internal.put(f'/internal/v1/jobs/{job_id}/artifacts/glb', content=glb).status_code == 200
    bad = {'manifest': {}, 'validation': {'glb': {'passed': False}}}
    assert internal.post(f'/internal/v1/jobs/{job_id}/complete', json=bad).json()['error']['code'] == 'VALIDATION_FAILED'
    assert internal.post(f'/internal/v1/jobs/{job_id}/complete', json=good).status_code == 200


def test_worker_cannot_skip_validation_gate(api, internal):
    worker = make_worker(internal)
    job_id = submit(api, worker)
    worker.api.claim(worker.engine_ids)
    r = internal.post(f'/internal/v1/jobs/{job_id}/progress', json={'status': 'exporting'})
    assert r.status_code == 409 and r.json()['error']['code'] == 'TRANSITION_INVALID'
    r = internal.post(f'/internal/v1/jobs/{job_id}/progress', json={'status': 'completed'})
    assert r.status_code == 422


def test_unvalidated_format_not_downloadable(api, internal):
    worker = make_worker(internal)
    job_id = submit(api, worker)
    worker.run_once()
    model_id = api.get(f'/api/v1/jobs/{job_id}').json()['model_id']
    r = api.get(f'/api/v1/models/{model_id}/files/stl', follow_redirects=False)
    assert r.status_code == 404 and r.json()['error']['code'] == 'FORMAT_NOT_AVAILABLE'


def test_stub_engine_cannot_be_loaded_in_production():
    assert set(engine_registry.PRODUCTION_ENGINES) == {'trellis2-4b', 'sf3d'}
    assert all(not target.startswith('tests') for target in engine_registry.PRODUCTION_ENGINES.values())
    with pytest.raises(ValueError):
        engine_registry.load_engines(['stub'])
    production_sources = [p.read_text(encoding='utf-8') for p in Path('worker').rglob('*.py')] + [p.read_text(encoding='utf-8') for p in Path('app').rglob('*.py')]
    assert not any('stub_engine' in src or 'from tests' in src for src in production_sources)


def test_smoke_mode_writes_validated_glb_and_manifest(tmp_path):
    from worker.main import smoke

    engines = {'trellis2-4b': StubEngine()}
    assert not smoke(engines, L40S_PROBE, 'worker/smoke/sphere.png', str(tmp_path))  # 2048 px requested, fixture is 1024
    assert smoke(engines, L40S_PROBE, 'worker/smoke/sphere.png', str(tmp_path), {'texture_size': 1024})
    assert (tmp_path / 'trellis2-4b.glb').read_bytes()[:4] == b'glTF'
    assert '"passed": true' in (tmp_path / 'trellis2-4b.manifest.json').read_text(encoding='utf-8')
