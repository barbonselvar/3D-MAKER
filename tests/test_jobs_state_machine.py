"""P1-05: every arrow of the job state machine, including heartbeat loss, expiry and wall clock."""
import pytest

from app.core import jobs
from app.core.config import Settings
from app.core.db import Database

T0 = 1_800_000_000.0


@pytest.fixture()
def db(tmp_path):
    return Database(tmp_path / 'maker.db')


@pytest.fixture()
def settings(tmp_path):
    return Settings(data_dir=tmp_path, secret_key=b'k' * 32)


def new_job(conn, engine='trellis2-4b', target='internal', now=T0):
    conn.execute("INSERT OR IGNORE INTO uploads VALUES ('upl_1','local','key','sha',10,10,'image/png','image/png',0,?)", (T0,))
    return jobs.create(conn, owner_id='local', upload_id='upl_1', target=target, engine=engine,
                       params={'seed': 1}, now=now)


def test_transition_table_matches_spec():
    t = jobs.TRANSITIONS
    assert t['queued'] == {'assigned', 'cancelled', 'expired'}
    assert t['assigned'] >= {'preparing', 'cancelled', 'failed', 'queued'}
    assert 'completed' not in t['assigned'] and 'completed' in t['exporting']
    for s in ('completed', 'failed', 'cancelled', 'expired'):
        assert t[s] == frozenset()
    assert t['validating'] == {'exporting', 'cancelled', 'failed', 'queued'}
    for s in ('assigned', 'preparing', 'running', 'postprocessing'):
        assert not (t[s] & {'exporting', 'completed'}), f'{s} may not skip validation'
    for s in jobs.ACTIVE:  # never backwards except requeue
        idx = jobs.PIPELINE.index(s)
        assert not (t[s] & set(jobs.PIPELINE[1:idx]))


def test_happy_path_through_every_stage(db):
    with db.connect() as conn:
        job = new_job(conn)
        assert job['status'] == 'queued'
        claimed = jobs.claim(conn, 'w1', ['trellis2-4b'], now=T0 + 1)
        assert claimed['id'] == job['id'] and claimed['status'] == 'assigned' and claimed['worker_id'] == 'w1'
        for i, status in enumerate(['preparing', 'running', 'postprocessing', 'validating', 'exporting', 'completed']):
            row = jobs.transition(conn, job['id'], status, worker_id='w1', now=T0 + 2 + i)
            assert row['status'] == status
        assert row['progress'] == 1.0
        assert row['started_at'] == T0 + 2 and row['finished_at'] == T0 + 7
        events = [r['status'] for r in conn.execute('SELECT status FROM job_events WHERE job_id=? ORDER BY seq', (job['id'],))]
        assert events == ['queued', 'assigned', 'preparing', 'running', 'postprocessing', 'validating', 'exporting', 'completed']


@pytest.mark.parametrize('src', ['queued', 'assigned', 'preparing', 'running', 'postprocessing', 'validating', 'exporting'])
def test_cancel_from_every_non_terminal_state(db, src):
    with db.connect() as conn:
        job = new_job(conn)
        if src != 'queued':
            jobs.claim(conn, 'w1', ['trellis2-4b'], now=T0)
            path = jobs.PIPELINE[2:jobs.PIPELINE.index(src) + 1]
            for status in path:
                jobs.transition(conn, job['id'], status, worker_id='w1', now=T0)
        assert jobs.transition(conn, job['id'], 'cancelled', now=T0 + 1)['status'] == 'cancelled'


@pytest.mark.parametrize('src', ['assigned', 'preparing', 'running', 'postprocessing', 'validating', 'exporting'])
def test_fail_from_every_active_state(db, src):
    with db.connect() as conn:
        job = new_job(conn)
        jobs.claim(conn, 'w1', ['trellis2-4b'], now=T0)
        for status in jobs.PIPELINE[2:jobs.PIPELINE.index(src) + 1]:
            jobs.transition(conn, job['id'], status, worker_id='w1', now=T0)
        row = jobs.transition(conn, job['id'], 'failed', error=('INFERENCE_OOM', 'out of memory', True), worker_id='w1', now=T0)
        assert (row['status'], row['error_code'], row['error_retryable']) == ('failed', 'INFERENCE_OOM', 1)


def test_queued_cannot_fail_or_complete(db):
    with db.connect() as conn:
        job = new_job(conn)
        for bad in ('failed', 'completed', 'running'):
            with pytest.raises(jobs.TransitionError):
                jobs.transition(conn, job['id'], bad)


def test_terminal_states_are_final(db):
    with db.connect() as conn:
        job = new_job(conn)
        jobs.transition(conn, job['id'], 'cancelled')
        with pytest.raises(jobs.TransitionError):
            jobs.transition(conn, job['id'], 'assigned')


def test_no_backwards_moves(db):
    with db.connect() as conn:
        job = new_job(conn)
        jobs.claim(conn, 'w1', ['trellis2-4b'])
        jobs.transition(conn, job['id'], 'running', worker_id='w1')
        with pytest.raises(jobs.TransitionError):
            jobs.transition(conn, job['id'], 'preparing', worker_id='w1')


def test_other_worker_cannot_move_job(db):
    with db.connect() as conn:
        job = new_job(conn)
        jobs.claim(conn, 'w1', ['trellis2-4b'])
        with pytest.raises(jobs.TransitionError):
            jobs.transition(conn, job['id'], 'preparing', worker_id='w2')


def test_claim_respects_engine_target_and_order(db):
    with db.connect() as conn:
        a = new_job(conn, engine='sf3d', now=T0)
        b = new_job(conn, engine='trellis2-4b', now=T0 + 1)
        c = new_job(conn, engine='trellis2-4b', now=T0 + 2)
        new_job(conn, engine='trellis2-4b', target='external:modal', now=T0 - 5)
        assert jobs.claim(conn, 'w1', []) is None
        assert jobs.claim(conn, 'w1', ['trellis2-4b'])['id'] == b['id']
        assert jobs.claim(conn, 'w2', ['trellis2-4b'])['id'] == c['id']
        assert jobs.claim(conn, 'w3', ['trellis2-4b']) is None
        assert jobs.claim(conn, 'w3', ['sf3d'])['id'] == a['id']


def test_queue_expiry_after_30_minutes(db, settings):
    with db.connect() as conn:
        job = new_job(conn, now=T0)
        assert jobs.sweep(conn, settings, now=T0 + 29 * 60) == []
        assert jobs.sweep(conn, settings, now=T0 + 30 * 60 + 1) == [(job['id'], 'expired')]
        assert jobs.get(conn, job['id'])['error_code'] == 'QUEUE_EXPIRED'


def test_heartbeat_loss_requeues_once_then_fails(db, settings):
    with db.connect() as conn:
        job = new_job(conn, now=T0)
        jobs.claim(conn, 'w1', ['trellis2-4b'], now=T0)
        jobs.transition(conn, job['id'], 'running', worker_id='w1', now=T0)
        jobs.heartbeat(conn, job['id'], 'w1', now=T0 + 100)
        assert jobs.sweep(conn, settings, now=T0 + 200) == []  # 100 s since last heartbeat
        assert jobs.sweep(conn, settings, now=T0 + 221) == [(job['id'], 'queued')]
        row = jobs.get(conn, job['id'])
        assert (row['worker_id'], row['requeue_count'], row['started_at']) == (None, 1, None)
        jobs.claim(conn, 'w2', ['trellis2-4b'], now=T0 + 300)
        assert jobs.sweep(conn, settings, now=T0 + 421) == [(job['id'], 'failed')]
        assert jobs.get(conn, job['id'])['error_code'] == 'WORKER_LOST'


def test_wall_clock_limit(db, settings):
    with db.connect() as conn:
        job = new_job(conn, now=T0)
        jobs.claim(conn, 'w1', ['trellis2-4b'], now=T0)
        jobs.transition(conn, job['id'], 'preparing', worker_id='w1', now=T0)
        for t in range(60, 15 * 60, 60):
            jobs.heartbeat(conn, job['id'], 'w1', now=T0 + t)
            assert jobs.sweep(conn, settings, now=T0 + t) == []
        jobs.heartbeat(conn, job['id'], 'w1', now=T0 + 15 * 60 + 1)
        assert jobs.sweep(conn, settings, now=T0 + 15 * 60 + 1) == [(job['id'], 'failed')]
        assert jobs.get(conn, job['id'])['error_code'] == 'TIMEOUT'


def test_queue_position(db):
    with db.connect() as conn:
        a = new_job(conn, now=T0)
        b = new_job(conn, now=T0 + 1)
        assert jobs.queue_position(conn, jobs.get(conn, a['id'])) == 1
        assert jobs.queue_position(conn, jobs.get(conn, b['id'])) == 2
        jobs.claim(conn, 'w1', ['trellis2-4b'])
        assert jobs.queue_position(conn, jobs.get(conn, b['id'])) == 1
        assert jobs.queue_position(conn, jobs.get(conn, a['id'])) is None


def test_rollback_on_error(db):
    with pytest.raises(RuntimeError), db.connect() as conn:
        new_job(conn)
        raise RuntimeError
    with db.connect() as conn:
        assert conn.execute('SELECT COUNT(*) FROM jobs').fetchone()[0] == 0


def test_database_uses_wal(db):
    with db.connect() as conn:
        assert conn.execute('PRAGMA journal_mode').fetchone()[0] == 'wal'
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {'uploads', 'jobs', 'job_events', 'models', 'artifacts', 'workers', 'spend_ledger'} <= tables
