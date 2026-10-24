"""Worker-only API under /internal/v1 (blueprint section D).

Served by a separate ASGI app on its own port, e.g.
    uvicorn app.internal:app --host 10.0.0.5 --port 8001
so it is never reachable through the public listener. Every call needs a
per-worker bearer token from MAKER_WORKER_TOKENS.
"""
from __future__ import annotations

import hashlib
import hmac
import re

from fastapi import APIRouter, Depends, FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, ConfigDict, Field

from .core import clock, jobs
from .core.config import Settings
from .core.context import Context, ctx
from .core.db import dumps
from .core.errors import ApiError, api_error_handler, validation_error_handler

router = APIRouter(prefix='/internal/v1')
MAX_ARTIFACT = 256 * 1024 * 1024
ARTIFACT_TYPES = {
    'glb': 'model/gltf-binary', 'glb_preview': 'model/gltf-binary', 'manifest': 'application/json',
    'thumbnail': 'image/png', 'obj_zip': 'application/zip', 'stl': 'model/stl', 'ply': 'application/octet-stream',
}


def worker_id(request: Request) -> str:
    supplied = request.headers.get('authorization', '').removeprefix('Bearer ').strip()
    for token, wid in request.app.state.ctx.settings.worker_tokens.items():
        if supplied and hmac.compare_digest(supplied.encode(), token.encode()):
            return wid
    raise ApiError(401, 'WORKER_UNAUTHORIZED', 'Unknown worker token.')


def assigned_job(conn, job_id: str, wid: str):
    job = jobs.get(conn, job_id)
    if job is None or job['worker_id'] != wid:
        raise ApiError(404, 'JOB_NOT_FOUND', 'Job not assigned to this worker.')
    if job['status'] == 'cancelled':
        raise ApiError(409, 'JOB_CANCELLED', 'The user cancelled this job.')
    if job['status'] in jobs.TERMINAL:
        raise ApiError(409, 'JOB_FINISHED', f"Job already {job['status']}.")
    return job


class HeartbeatBody(BaseModel):
    probe: dict
    engines: list[str]
    job_id: str | None = None


@router.post('/heartbeat')
def heartbeat(body: HeartbeatBody, c: Context = Depends(ctx), wid: str = Depends(worker_id)):
    now = clock.now()
    with c.db.connect() as conn:
        conn.execute(
            'INSERT INTO workers (id, kind, probe, engines, heartbeat_at, created_at) VALUES (?,?,?,?,?,?)'
            ' ON CONFLICT(id) DO UPDATE SET probe=excluded.probe, engines=excluded.engines, heartbeat_at=excluded.heartbeat_at',
            (wid, 'internal', dumps(body.probe), dumps(body.engines), now, now))
        cancel = False
        if body.job_id:
            job = jobs.get(conn, body.job_id)
            cancel = job is None or job['worker_id'] != wid or job['status'] not in jobs.ACTIVE
            if not cancel:
                jobs.heartbeat(conn, body.job_id, wid, now=now)
    return {'ok': True, 'cancel': cancel}


class ClaimBody(BaseModel):
    engines: list[str]


@router.post('/claim')
def claim(body: ClaimBody, c: Context = Depends(ctx), wid: str = Depends(worker_id)):
    with c.db.connect() as conn:
        jobs.sweep(conn, c.settings)
        job = jobs.claim(conn, wid, body.engines)
        if job is None:
            return Response(status_code=204)
        up = conn.execute('SELECT * FROM uploads WHERE id=?', (job['upload_id'],)).fetchone()
        return {'job': jobs.to_dict(job) | {
            'input': {'url': f"/internal/v1/jobs/{job['id']}/input", 'sha256': up['sha256'], 'width': up['width'],
                      'height': up['height'], 'has_alpha': bool(up['has_alpha'])}}}


@router.get('/jobs/{job_id}/input')
def job_input(job_id: str, c: Context = Depends(ctx), wid: str = Depends(worker_id)):
    with c.db.connect() as conn:
        job = assigned_job(conn, job_id, wid)
        key = conn.execute('SELECT storage_key FROM uploads WHERE id=?', (job['upload_id'],)).fetchone()[0]
    return Response(c.storage.get(key), media_type='image/png')


class ProgressBody(BaseModel):
    model_config = ConfigDict(extra='forbid')
    status: str | None = None
    stage: str | None = Field(None, max_length=40)
    progress: float | None = Field(None, ge=0, le=1)
    message: str | None = Field(None, max_length=500)


@router.post('/jobs/{job_id}/progress')
def progress(job_id: str, body: ProgressBody, c: Context = Depends(ctx), wid: str = Depends(worker_id)):
    if body.status in ('completed', 'failed', 'cancelled', 'expired', 'queued', 'assigned'):
        raise ApiError(422, 'STATUS_NOT_ALLOWED', f'Use the dedicated route for {body.status!r}.')
    with c.db.connect() as conn:
        job = assigned_job(conn, job_id, wid)
        if body.status and body.status != job['status']:
            try:
                job = jobs.transition(conn, job_id, body.status, stage=body.stage, progress=body.progress,
                                      message=body.message, worker_id=wid)
            except jobs.TransitionError as e:
                raise ApiError(409, 'TRANSITION_INVALID', str(e)) from None
        else:
            sets = {k: v for k, v in (('stage', body.stage), ('progress', body.progress)) if v is not None}
            if sets:
                conn.execute(f"UPDATE jobs SET {', '.join(f'{k}=?' for k in sets)} WHERE id=?", (*sets.values(), job_id))
            jobs.record_event(conn, job_id, job['status'], body.stage or job['stage'],
                              body.progress if body.progress is not None else job['progress'], body.message)
        jobs.heartbeat(conn, job_id, wid)
        return {'status': jobs.get(conn, job_id)['status']}


@router.put('/jobs/{job_id}/artifacts/{fmt}')
async def put_artifact(job_id: str, fmt: str, request: Request, c: Context = Depends(ctx), wid: str = Depends(worker_id)):
    if fmt not in ARTIFACT_TYPES:
        raise ApiError(422, 'FORMAT_UNKNOWN', f'Unknown artifact format {fmt!r}.')
    data = await request.body()
    if not data or len(data) > MAX_ARTIFACT:
        raise ApiError(413, 'ARTIFACT_SIZE', 'Artifact is empty or larger than 256 MB.')
    with c.db.connect() as conn:
        assigned_job(conn, job_id, wid)
        old = conn.execute('SELECT storage_key FROM artifacts WHERE job_id=? AND format=?', (job_id, fmt)).fetchone()
    key = c.storage.new_key()
    c.storage.put(key, data, ARTIFACT_TYPES[fmt])
    with c.db.connect() as conn:
        conn.execute('DELETE FROM artifacts WHERE job_id=? AND format=?', (job_id, fmt))
        conn.execute('INSERT INTO artifacts (id, job_id, format, storage_key, size, sha256, created_at) VALUES (?,?,?,?,?,?,?)',
                     (jobs.new_id('art'), job_id, fmt, key, len(data), hashlib.sha256(data).hexdigest(), clock.now()))
    if old:
        c.storage.delete(old['storage_key'])
    return {'format': fmt, 'size': len(data), 'sha256': hashlib.sha256(data).hexdigest()}


class CompleteBody(BaseModel):
    manifest: dict
    validation: dict[str, dict]  # artifact format -> validation report ({"passed": bool, ...})
    gpu_seconds: float | None = Field(None, ge=0)


@router.post('/jobs/{job_id}/complete')
def complete(job_id: str, body: CompleteBody, c: Context = Depends(ctx), wid: str = Depends(worker_id)):
    with c.db.connect() as conn:
        job = assigned_job(conn, job_id, wid)
        if job['status'] != 'exporting':
            raise ApiError(409, 'TRANSITION_INVALID', f"Complete is only allowed from exporting, not {job['status']}.")
        formats = {r['format'] for r in conn.execute('SELECT format FROM artifacts WHERE job_id=?', (job_id,))}
        if 'glb' not in formats:
            raise ApiError(422, 'ARTIFACT_MISSING', 'Upload the canonical GLB before completing.')
        if not body.validation.get('glb', {}).get('passed'):
            raise ApiError(422, 'VALIDATION_FAILED', 'The GLB did not pass validation; report a failure instead.')
        model_id = jobs.new_id('mdl')
        conn.execute('INSERT INTO models (id, owner_id, job_id, upload_id, engine, manifest, created_at) VALUES (?,?,?,?,?,?,?)',
                     (model_id, job['owner_id'], job_id, job['upload_id'], job['engine'], dumps(body.manifest), clock.now()))
        for fmt in formats:
            report = body.validation.get(fmt)
            ok = fmt == 'manifest' or bool(report and report.get('passed'))
            conn.execute('UPDATE artifacts SET model_id=?, validated=?, validation=? WHERE job_id=? AND format=?',
                         (model_id, int(ok), dumps(report) if report else None, job_id, fmt))
        conn.execute('UPDATE jobs SET model_id=?, gpu_seconds=? WHERE id=?', (model_id, body.gpu_seconds, job_id))
        jobs.transition(conn, job_id, 'completed', stage='done', worker_id=wid)
    return {'model_id': model_id}


class FailBody(BaseModel):
    code: str = Field(pattern=r'^[A-Z][A-Z0-9_]{2,40}$')
    message: str = Field(max_length=500)
    retryable: bool = False


@router.post('/jobs/{job_id}/fail')
def fail(job_id: str, body: FailBody, c: Context = Depends(ctx), wid: str = Depends(worker_id)):
    with c.db.connect() as conn:
        assigned_job(conn, job_id, wid)
        jobs.transition(conn, job_id, 'failed', error=(body.code, re.sub(r'\s+', ' ', body.message), body.retryable),
                        worker_id=wid)
    return {'status': 'failed'}


def create_internal_app(settings: Settings | None = None, context: Context | None = None) -> FastAPI:
    app = FastAPI(title='3D Maker worker API', version='0.2.0', docs_url=None, redoc_url=None, openapi_url=None)
    app.state.ctx = context or Context.create(settings or Settings.from_env())
    app.add_exception_handler(ApiError, api_error_handler)
    app.add_exception_handler(RequestValidationError, validation_error_handler)
    app.include_router(router)
    return app


def __getattr__(name):  # lazy `app` for `uvicorn app.internal:app`; importing the module has no side effects
    if name == 'app':
        return create_internal_app()
    raise AttributeError(name)
