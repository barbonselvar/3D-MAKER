"""Job state machine and queue (docs/PLAN.md, contradiction C1).

    queued -> assigned -> preparing -> running -> postprocessing -> validating -> exporting -> completed
    any non-terminal state         -> cancelled  (user request)
    assigned .. exporting          -> failed     (engine/validation error, WORKER_LOST, TIMEOUT)
    queued                         -> expired    (no worker within 30 min)
    assigned .. exporting          -> queued     (worker silent for 120 s, once per job)

The worker may skip optional stages (e.g. preparing -> validating if an engine
has no separate postprocessing step) but can never skip the validation gate:
validating -> exporting -> completed is mandatory. No backwards moves except
the requeue above.
"""
from __future__ import annotations

import secrets
import sqlite3

from . import clock
from .config import Settings
from .db import dumps, loads

PIPELINE = ['queued', 'assigned', 'preparing', 'running', 'postprocessing', 'validating', 'exporting', 'completed']
ACTIVE = frozenset(PIPELINE[1:-1])
TERMINAL = frozenset({'completed', 'failed', 'cancelled', 'expired'})
STATUSES = frozenset(PIPELINE) | TERMINAL


def _allowed(src: str) -> frozenset[str]:
    if src in TERMINAL:
        return frozenset()
    if src == 'queued':
        return frozenset({'assigned', 'cancelled', 'expired'})
    later = PIPELINE[PIPELINE.index(src) + 1:]
    if 'validating' in later:  # optional stages may be skipped, the validation gate may not
        later = later[:later.index('validating') + 1]
    else:
        later = later[:1]
    return frozenset(later) | {'cancelled', 'failed', 'queued'}


TRANSITIONS: dict[str, frozenset[str]] = {s: _allowed(s) for s in STATUSES}


class TransitionError(Exception):
    """Raised for a transition the state machine does not allow (maps to HTTP 409)."""


def new_id(prefix: str) -> str:
    return f'{prefix}_{secrets.token_hex(12)}'


def record_event(conn: sqlite3.Connection, job_id: str, status: str, stage=None, progress=None, message=None, now=None):
    conn.execute(
        'INSERT INTO job_events (job_id, ts, status, stage, progress, message) VALUES (?,?,?,?,?,?)',
        (job_id, now if now is not None else clock.now(), status, stage, progress, message),
    )


def get(conn: sqlite3.Connection, job_id: str, owner_id: str | None = None) -> sqlite3.Row | None:
    if owner_id is None:
        return conn.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
    return conn.execute('SELECT * FROM jobs WHERE id=? AND owner_id=?', (job_id, owner_id)).fetchone()


def create(conn, *, owner_id: str, upload_id: str, target: str, engine: str, params: dict, retry_of=None, now=None):
    now = clock.now() if now is None else now
    job_id = new_id('job')
    conn.execute(
        'INSERT INTO jobs (id, owner_id, upload_id, target, engine, params, status, retry_of, created_at, queued_at)'
        " VALUES (?,?,?,?,?,?, 'queued', ?,?,?)",
        (job_id, owner_id, upload_id, target, engine, dumps(params), retry_of, now, now),
    )
    record_event(conn, job_id, 'queued', now=now)
    return get(conn, job_id)


def transition(conn, job_id: str, to: str, *, stage=None, progress=None, message=None,
               error: tuple[str, str, bool] | None = None, worker_id: str | None = None, now=None) -> sqlite3.Row:
    """Move job_id to `to`, enforcing TRANSITIONS. If worker_id is given the job must be assigned to it."""
    now = clock.now() if now is None else now
    job = get(conn, job_id)
    if job is None:
        raise KeyError(job_id)
    if worker_id is not None and job['worker_id'] != worker_id:
        raise TransitionError(f'job {job_id} is not assigned to worker {worker_id}')
    if to not in TRANSITIONS[job['status']]:
        raise TransitionError(f"cannot go from {job['status']} to {to}")
    sets: dict = {'status': to}
    if stage is not None:
        sets['stage'] = stage
    if progress is not None:
        sets['progress'] = max(0.0, min(1.0, float(progress)))
    if to == 'assigned':
        sets.update(assigned_at=now, heartbeat_at=now)
    if to in ACTIVE and to != 'assigned' and job['started_at'] is None:
        sets['started_at'] = now
    if to == 'queued':  # requeue after worker loss: fresh attempt
        sets.update(worker_id=None, stage=None, progress=0.0, assigned_at=None, started_at=None, heartbeat_at=None,
                    queued_at=now, requeue_count=job['requeue_count'] + 1)
    if to == 'completed':
        sets['progress'] = 1.0
    if to in TERMINAL:
        sets['finished_at'] = now
    if error is not None:
        sets.update(error_code=error[0], error_message=error[1], error_retryable=int(error[2]))
    cols = ', '.join(f'{k}=?' for k in sets)
    conn.execute(f'UPDATE jobs SET {cols} WHERE id=?', (*sets.values(), job_id))
    record_event(conn, job_id, to, sets.get('stage', job['stage']), sets.get('progress', job['progress']),
                 message or (error[1] if error else None), now=now)
    return get(conn, job_id)


def claim(conn, worker_id: str, engines: list[str], now=None) -> sqlite3.Row | None:
    """Atomically assign the oldest queued internal job this worker can run."""
    if not engines:
        return None
    now = clock.now() if now is None else now
    marks = ','.join('?' * len(engines))
    row = conn.execute(
        f"SELECT id FROM jobs WHERE status='queued' AND target='internal' AND engine IN ({marks})"
        ' ORDER BY queued_at, created_at LIMIT 1', engines).fetchone()
    if row is None:
        return None
    conn.execute('UPDATE jobs SET worker_id=? WHERE id=?', (worker_id, row['id']))
    return transition(conn, row['id'], 'assigned', worker_id=worker_id, now=now)


def heartbeat(conn, job_id: str, worker_id: str, now=None) -> sqlite3.Row:
    job = get(conn, job_id)
    if job is None or job['worker_id'] != worker_id:
        raise TransitionError('job not assigned to this worker')
    if job['status'] in ACTIVE:
        conn.execute('UPDATE jobs SET heartbeat_at=? WHERE id=?', (clock.now() if now is None else now, job_id))
    return get(conn, job_id)


def sweep(conn, settings: Settings, now=None) -> list[tuple[str, str]]:
    """Apply time-based rules. Returns [(job_id, new_status)] for every job it changed."""
    now = clock.now() if now is None else now
    changed = []
    for job in conn.execute("SELECT * FROM jobs WHERE status='queued' AND queued_at < ?",
                            (now - settings.queue_expiry,)).fetchall():
        transition(conn, job['id'], 'expired', error=('QUEUE_EXPIRED', 'No GPU worker picked up the job within 30 minutes.', True), now=now)
        changed.append((job['id'], 'expired'))
    active = ','.join(f"'{s}'" for s in ACTIVE)
    for job in conn.execute(f'SELECT * FROM jobs WHERE status IN ({active})').fetchall():
        started = job['started_at'] or job['assigned_at']
        if started is not None and now - started > settings.wall_clock_limit:
            transition(conn, job['id'], 'failed', error=('TIMEOUT', 'The job exceeded the 15-minute limit.', True), now=now)
            changed.append((job['id'], 'failed'))
        elif now - (job['heartbeat_at'] or job['assigned_at'] or now) > settings.heartbeat_lost:
            if job['requeue_count'] == 0:
                transition(conn, job['id'], 'queued', message='Worker stopped responding; job requeued once.', now=now)
                changed.append((job['id'], 'queued'))
            else:
                transition(conn, job['id'], 'failed', error=('WORKER_LOST', 'The GPU worker stopped responding twice.', True), now=now)
                changed.append((job['id'], 'failed'))
    return changed


def queue_position(conn, job: sqlite3.Row) -> int | None:
    if job['status'] != 'queued':
        return None
    ahead = conn.execute("SELECT COUNT(*) FROM jobs WHERE status='queued' AND target=? AND queued_at < ?",
                         (job['target'], job['queued_at'])).fetchone()[0]
    return ahead + 1


def to_dict(job: sqlite3.Row, position: int | None = None) -> dict:
    error = None
    if job['error_code']:
        error = {'code': job['error_code'], 'message': job['error_message'], 'retryable': bool(job['error_retryable'])}
    return {
        'id': job['id'], 'status': job['status'], 'stage': job['stage'], 'progress': job['progress'],
        'target': job['target'], 'engine': job['engine'], 'params': loads(job['params']),
        'upload_id': job['upload_id'], 'queue_position': position, 'retry_of': job['retry_of'],
        'created_at': clock.iso(job['created_at']), 'started_at': clock.iso(job['started_at']),
        'finished_at': clock.iso(job['finished_at']), 'error': error, 'model_id': job['model_id'],
        'gpu_seconds': job['gpu_seconds'], 'cost_usd': job['cost_usd'],
    }
