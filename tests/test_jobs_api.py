"""P1-05: public job routes."""
import threading
import time

import pytest

from app.core import jobs

from .helpers import create_job, online_worker, upload


def err(r):
    return r.json()['error']['code']


def test_create_requires_existing_upload(api):
    online_worker(api)
    r = create_job(api, 'upl_missing')
    assert r.status_code == 404 and err(r) == 'UPLOAD_NOT_FOUND'


def test_internal_refused_without_worker_never_rerouted(api):
    r = create_job(api, upload(api))
    assert r.status_code == 409 and err(r) == 'TARGET_UNAVAILABLE'
    assert 'No GPU worker online' in r.json()['error']['message']
    with api.app.state.ctx.db.connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] == 0


def test_create_and_get_job(api):
    online_worker(api)
    r = create_job(api, upload(api), params={'resolution': 1024})
    assert r.status_code == 202
    body = r.json()
    assert body['status'] == 'queued' and body['job_id'].startswith('job_')
    job = api.get(f"/api/v1/jobs/{body['job_id']}").json()
    assert job['status'] == 'queued' and job['queue_position'] == 1
    assert job['params']['texture_size'] == 2048 and job['params']['face_budget'] == 200_000
    assert isinstance(job['params']['seed'], int)  # always recorded


def test_seed_is_kept_when_given(api):
    online_worker(api)
    job = create_job(api, upload(api), params={'seed': 123456}).json()['job']
    assert job['params']['seed'] == 123456


@pytest.mark.parametrize('params', [{'resolution': 777}, {'texture_size': 300}, {'face_budget': 10},
                                    {'seed': -1}, {'unknown': 1}])
def test_param_validation(api, params):
    online_worker(api)
    r = create_job(api, upload(api), params=params)
    assert r.status_code == 422 and err(r) == 'REQUEST_INVALID'


def test_unknown_engine_and_target(api):
    online_worker(api)
    u = upload(api)
    assert err(create_job(api, u, engine='magic')) == 'ENGINE_UNKNOWN'
    assert err(create_job(api, u, target='quantum')) == 'TARGET_UNKNOWN'


def test_engine_not_offered_by_worker(api):
    online_worker(api, engines=('sf3d',))
    r = create_job(api, upload(api))
    assert r.status_code == 409


def test_external_requires_consent_first(api):
    u = upload(api)
    r = create_job(api, u, target='external:modal')
    assert r.status_code == 422 and err(r) == 'CONSENT_REQUIRED'
    r = create_job(api, u, target='external:modal', consent={'external_processing': True})
    assert r.status_code == 409 and err(r) == 'TARGET_UNAVAILABLE'  # not configured: never silently routed
    r = create_job(api, u, target='external:hf-demo', consent={'external_processing': True})
    assert r.status_code == 409


def test_owner_scoping(settings):
    from fastapi.testclient import TestClient

    from app.main import create_app
    settings.api_token = 'owner-token-123'
    client = TestClient(create_app(settings), headers={'Authorization': 'Bearer owner-token-123'})
    online_worker(client)
    job_id = create_job(client, upload(client)).json()['job_id']
    with client.app.state.ctx.db.connect() as conn:
        conn.execute("UPDATE jobs SET owner_id='someone-else'")
    assert client.get(f'/api/v1/jobs/{job_id}').status_code == 404
    assert client.post(f'/api/v1/jobs/{job_id}/cancel').status_code == 404


def test_cancel_queued_job_and_terminal_conflict(api):
    online_worker(api)
    job_id = create_job(api, upload(api)).json()['job_id']
    r = api.post(f'/api/v1/jobs/{job_id}/cancel')
    assert r.status_code == 202 and r.json()['status'] == 'cancelled'
    r = api.post(f'/api/v1/jobs/{job_id}/cancel')
    assert r.status_code == 409 and err(r) == 'JOB_FINISHED'


def test_retry_new_seed_unless_pinned(api):
    online_worker(api)
    u = upload(api)
    a = create_job(api, u, params={'seed': 7}).json()['job_id']
    assert err(api.post(f'/api/v1/jobs/{a}/retry')) == 'JOB_ACTIVE'
    api.post(f'/api/v1/jobs/{a}/cancel')
    seeds = {api.post(f'/api/v1/jobs/{a}/retry').json()['job']['params']['seed'] for _ in range(3)}
    assert 7 not in seeds  # fresh random seed (2**31 space) on each unpinned retry
    b = create_job(api, u, params={'seed': 9, 'pin_seed': True}).json()['job_id']
    api.post(f'/api/v1/jobs/{b}/cancel')
    retried = api.post(f'/api/v1/jobs/{b}/retry').json()['job']
    assert retried['params']['seed'] == 9 and retried['retry_of'] == b and retried['upload_id'] == u


def parse_sse(text):
    events = []
    for block in text.strip().split('\n\n'):
        fields = dict(line.split(': ', 1) for line in block.splitlines() if ': ' in line and not line.startswith(':'))
        if 'event' in fields:
            events.append((fields['event'], fields['data']))
    return events


def test_sse_for_finished_job(api):
    online_worker(api)
    job_id = create_job(api, upload(api)).json()['job_id']
    api.post(f'/api/v1/jobs/{job_id}/cancel')
    events = parse_sse(api.get(f'/api/v1/jobs/{job_id}/events').text)
    assert [e for e, _ in events] == ['stage', 'stage', 'error']
    assert '"status":"cancelled"' in events[-1][1]


def test_sse_streams_live_progress(api):
    online_worker(api)
    job_id = create_job(api, upload(api)).json()['job_id']
    db = api.app.state.ctx.db

    def worker():
        time.sleep(0.3)
        with db.connect() as conn:
            jobs.claim(conn, 'w1', ['trellis2-4b'])
        for status in ['preparing', 'running', 'validating', 'exporting', 'completed']:
            time.sleep(0.1)
            with db.connect() as conn:
                jobs.transition(conn, job_id, status, worker_id='w1', stage=status)

    t = threading.Thread(target=worker)
    t.start()
    events = parse_sse(api.get(f'/api/v1/jobs/{job_id}/events').text)
    t.join()
    statuses = [__import__('json').loads(d)['status'] for e, d in events if e == 'stage']
    assert statuses == ['queued', 'assigned', 'preparing', 'running', 'validating', 'exporting', 'completed']
    assert events[-1][0] == 'done'


def test_sse_resumes_after_last_event_id(api):
    online_worker(api)
    job_id = create_job(api, upload(api)).json()['job_id']
    api.post(f'/api/v1/jobs/{job_id}/cancel')
    first = api.get(f'/api/v1/jobs/{job_id}/events').text
    first_id = [line for line in first.splitlines() if line.startswith('id: ')][0][4:]
    resumed = parse_sse(api.get(f'/api/v1/jobs/{job_id}/events', headers={'Last-Event-ID': first_id}).text)
    assert [e for e, _ in resumed] == ['stage', 'error']


def test_legacy_validation_errors_unchanged(api):
    r = api.post('/api/generate')
    assert r.status_code == 422 and 'detail' in r.json()
