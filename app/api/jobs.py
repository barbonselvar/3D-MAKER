"""Public job routes under /api/v1 (blueprint section D)."""
from __future__ import annotations

import asyncio
import json
import secrets
from typing import Literal

from fastapi import APIRouter, Depends, Header, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from ..core import clock, jobs
from ..core.context import Context, ctx, current_owner
from ..core.db import loads
from ..core.engines import ENGINES
from ..core.errors import ApiError
from .gpu import internal_target

router = APIRouter(prefix='/api/v1')


class Params(BaseModel):
    model_config = ConfigDict(extra='forbid')
    resolution: Literal[512, 1024, 1536] = 1024
    seed: int | None = Field(None, ge=0, le=2**31 - 1)
    pin_seed: bool = False  # retries reuse the seed only when pinned
    texture_size: Literal[512, 1024, 2048, 4096] = 2048
    face_budget: int = Field(200_000, ge=1_000, le=2_000_000)
    emissive_eyes: bool = False  # optional heuristic (P5-03), off by default


class Consent(BaseModel):
    model_config = ConfigDict(extra='forbid')
    external_processing: bool = False
    max_cost_usd: float | None = Field(None, ge=0)


class JobRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    upload_id: str
    target: str
    engine: str = 'trellis2-4b'
    params: Params = Params()
    consent: Consent = Consent()


def check_target(c: Context, target: str, engine: str, consent: Consent, texture_size: int = 0) -> None:
    """Raise unless `target` can run `engine` now. Never reroutes: no silent failover between targets."""
    if engine not in ENGINES:
        raise ApiError(422, 'ENGINE_UNKNOWN', f'Unknown engine {engine!r}.')
    if target == 'internal':
        t = internal_target(c)
        if not t['available']:
            raise ApiError(409, 'TARGET_UNAVAILABLE', t['reason'], retryable=True)
        offered = {e['id']: e for e in t['engines']}
        if engine not in offered:
            status = next((s for s in t['engine_status'] if s['id'] == engine), None)
            detail = f" ({status['status_label']}: {status['reason']})" if status and status['reason'] else ''
            raise ApiError(409, 'TARGET_UNAVAILABLE', f'The local GPU cannot run {ENGINES[engine].label}{detail}. '
                           f"It offers: {', '.join(e['label'] for e in offered.values())}.")
        limit = offered[engine]['max_texture_size']
        if limit and texture_size > limit:
            raise ApiError(409, 'TEXTURE_TOO_LARGE', f'This GPU passed its {ENGINES[engine].label} self-test with textures up to '
                           f'{limit} px. Choose {limit} px or smaller.')
        return
    if target.startswith('external:'):
        if not consent.external_processing:
            raise ApiError(422, 'CONSENT_REQUIRED', 'External processing sends your image to a third-party GPU. Tick the consent box first.')
        if target == 'external:hf-demo':
            raise ApiError(409, 'TARGET_UNAVAILABLE', 'The Hugging Face public demo runs from your browser, not through this API.')
        if target == 'external:modal':
            raise ApiError(409, 'TARGET_UNAVAILABLE', 'Dedicated External GPU is not configured yet (Phase 2; needs owner approval).')
    raise ApiError(422, 'TARGET_UNKNOWN', f'Unknown target {target!r}.')


def owned_job(conn, job_id: str, owner: str):
    job = jobs.get(conn, job_id, owner)
    if job is None:
        raise ApiError(404, 'JOB_NOT_FOUND', 'Job not found.')
    return job


def job_body(conn, job) -> dict:
    return jobs.to_dict(job, jobs.queue_position(conn, job))


@router.post('/jobs', status_code=202)
def create_job(req: JobRequest, c: Context = Depends(ctx), owner: str = Depends(current_owner)):
    with c.db.connect() as conn:
        if conn.execute('SELECT 1 FROM uploads WHERE id=? AND owner_id=?', (req.upload_id, owner)).fetchone() is None:
            raise ApiError(404, 'UPLOAD_NOT_FOUND', 'Upload not found. Upload the image again.')
    check_target(c, req.target, req.engine, req.consent, req.params.texture_size)
    params = req.params.model_dump()
    if params['seed'] is None:
        params['seed'] = secrets.randbelow(2**31)  # recorded so the run is reproducible
    with c.db.connect() as conn:
        job = jobs.create(conn, owner_id=owner, upload_id=req.upload_id, target=req.target, engine=req.engine, params=params)
        return {'job_id': job['id'], 'status': job['status'], 'job': job_body(conn, job)}


@router.get('/jobs/{job_id}')
def get_job(job_id: str, c: Context = Depends(ctx), owner: str = Depends(current_owner)):
    with c.db.connect() as conn:
        jobs.sweep(conn, c.settings)
        return job_body(conn, owned_job(conn, job_id, owner))


@router.post('/jobs/{job_id}/cancel', status_code=202)
def cancel_job(job_id: str, c: Context = Depends(ctx), owner: str = Depends(current_owner)):
    with c.db.connect() as conn:
        job = owned_job(conn, job_id, owner)
        if job['status'] in jobs.TERMINAL:
            raise ApiError(409, 'JOB_FINISHED', f"The job already ended ({job['status']}).")
        # The worker learns about it on its next heartbeat/progress call and stops at the next stage boundary.
        job = jobs.transition(conn, job_id, 'cancelled', message='Cancelled by user.')
        return job_body(conn, job)


@router.post('/jobs/{job_id}/retry', status_code=202)
def retry_job(job_id: str, c: Context = Depends(ctx), owner: str = Depends(current_owner)):
    with c.db.connect() as conn:
        old = owned_job(conn, job_id, owner)
    if old['status'] not in jobs.TERMINAL:
        raise ApiError(409, 'JOB_ACTIVE', 'The job is still running; cancel it before retrying.')
    params = loads(old['params'])
    check_target(c, old['target'], old['engine'], Consent(external_processing=old['target'].startswith('external:')),
                 params.get('texture_size', 0))
    if not params.get('pin_seed'):
        params['seed'] = secrets.randbelow(2**31)
    with c.db.connect() as conn:
        job = jobs.create(conn, owner_id=owner, upload_id=old['upload_id'], target=old['target'], engine=old['engine'],
                          params=params, retry_of=old['id'])
        return {'job_id': job['id'], 'status': job['status'], 'job': job_body(conn, job)}


def _sse(event: str, data: dict, seq: int | None = None) -> str:
    head = f'id: {seq}\n' if seq is not None else ''
    return f'{head}event: {event}\ndata: {json.dumps(data, separators=(",", ":"))}\n\n'


@router.get('/jobs/{job_id}/events')
async def job_events(job_id: str, request: Request, c: Context = Depends(ctx), owner: str = Depends(current_owner),
                     last_event_id: str | None = Header(None)):
    """Server-Sent Events: `stage` per state change, then one `done` (completed) or `error` (failed/cancelled/expired)."""
    with c.db.connect() as conn:
        owned_job(conn, job_id, owner)
    poll = 0.25

    async def stream():
        seq = int(last_event_id) if (last_event_id or '').isdigit() else 0
        deadline = clock.now() + c.settings.queue_expiry + c.settings.wall_clock_limit * 2
        idle = 0.0
        while clock.now() < deadline:
            with c.db.connect() as conn:
                jobs.sweep(conn, c.settings)
                rows = conn.execute('SELECT * FROM job_events WHERE job_id=? AND seq>? ORDER BY seq', (job_id, seq)).fetchall()
                job = jobs.get(conn, job_id)
                body = job_body(conn, job)
            for r in rows:
                seq = r['seq']
                yield _sse('stage', {'status': r['status'], 'stage': r['stage'], 'progress': r['progress'],
                                     'message': r['message'], 'ts': clock.iso(r['ts'])}, seq)
            if job['status'] in jobs.TERMINAL:
                yield _sse('done' if job['status'] == 'completed' else 'error', body)
                return
            if await request.is_disconnected():
                return
            idle = 0.0 if rows else idle + poll
            if idle >= 15:
                idle = 0.0
                yield ': keep-alive\n\n'
            await asyncio.sleep(poll)

    return StreamingResponse(stream(), media_type='text/event-stream',
                             headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})
