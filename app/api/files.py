"""GET /api/v1/files/{key}: serves local-storage objects behind HMAC-signed, expiring URLs."""
from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse

from ..core.context import Context, ctx
from ..core.errors import ApiError

router = APIRouter(prefix='/api/v1')


@router.get('/files/{key}')
def signed_file(key: str, exp: int = Query(...), name: str = Query(...), sig: str = Query(...), c: Context = Depends(ctx)):
    problem = c.storage.verify(key, exp, name, sig)
    if problem:
        raise ApiError(403, problem, 'This download link is invalid or has expired. Request a new one.')
    if not c.storage.exists(key):
        raise ApiError(404, 'NOT_FOUND', 'File not found.')
    return FileResponse(c.storage.path(key), media_type=c.storage.content_type(key), filename=name)
