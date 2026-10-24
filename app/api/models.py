"""Model manifest and downloads (gallery listing/deletion arrive with P3-01)."""
from fastapi import APIRouter, Depends
from fastapi.responses import RedirectResponse

from ..core import clock
from ..core.context import Context, ctx, current_owner
from ..core.db import loads
from ..core.errors import ApiError

router = APIRouter(prefix='/api/v1')
EXTENSIONS = {'glb': 'glb', 'glb_preview': 'preview.glb', 'manifest': 'manifest.json', 'thumbnail': 'png',
              'obj_zip': 'obj.zip', 'stl': 'stl', 'ply': 'ply'}


def owned_model(conn, model_id: str, owner: str):
    row = conn.execute('SELECT * FROM models WHERE id=? AND owner_id=?', (model_id, owner)).fetchone()
    if row is None:
        raise ApiError(404, 'MODEL_NOT_FOUND', 'Model not found.')
    return row


@router.get('/models/{model_id}')
def get_model(model_id: str, c: Context = Depends(ctx), owner: str = Depends(current_owner)):
    with c.db.connect() as conn:
        m = owned_model(conn, model_id, owner)
        arts = conn.execute('SELECT * FROM artifacts WHERE model_id=? ORDER BY format', (model_id,)).fetchall()
    return {
        'id': m['id'], 'job_id': m['job_id'], 'upload_id': m['upload_id'], 'engine': m['engine'],
        'created_at': clock.iso(m['created_at']), 'manifest': loads(m['manifest']),
        # Only formats that passed validation are downloadable (gates V1-V4 / V7).
        'files': [{'format': a['format'], 'size': a['size'], 'sha256': a['sha256'], 'validated': bool(a['validated']),
                   'validation': loads(a['validation']), 'url': f"/api/v1/models/{model_id}/files/{a['format']}"
                   if a['validated'] else None} for a in arts],
    }


@router.get('/models/{model_id}/files/{fmt}')
def download(model_id: str, fmt: str, c: Context = Depends(ctx), owner: str = Depends(current_owner)):
    with c.db.connect() as conn:
        owned_model(conn, model_id, owner)
        art = conn.execute('SELECT * FROM artifacts WHERE model_id=? AND format=? AND validated=1', (model_id, fmt)).fetchone()
    if art is None:
        raise ApiError(404, 'FORMAT_NOT_AVAILABLE', f'{fmt} is not available for this model (missing or failed validation).')
    filename = f"3d-maker-{model_id.removeprefix('mdl_')[:10]}.{EXTENSIONS.get(fmt, fmt)}"
    return RedirectResponse(c.storage.presign(art['storage_key'], filename), status_code=302)
