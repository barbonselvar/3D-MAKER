"""POST /api/v1/uploads: validate, strip metadata, store."""
from fastapi import APIRouter, Depends, File, UploadFile

from ..core import clock
from ..core.context import Context, ctx, current_owner
from ..core.errors import ApiError
from ..core.images import MAX_BYTES, InvalidImage, sanitize
from ..core.jobs import new_id

router = APIRouter(prefix='/api/v1')


@router.post('/uploads', status_code=201)
async def create_upload(image: UploadFile = File(...), c: Context = Depends(ctx), owner: str = Depends(current_owner)):
    raw = await image.read(MAX_BYTES + 1)
    if len(raw) > MAX_BYTES:
        raise ApiError(413, 'INPUT_TOO_LARGE', 'Image is larger than 12 MB.')
    try:
        clean = sanitize(raw)
    except InvalidImage as e:
        raise ApiError(400, 'INPUT_INVALID', str(e)) from None
    key = c.storage.new_key()
    c.storage.put(key, clean.png, 'image/png')
    upload_id = new_id('upl')
    with c.db.connect() as conn:
        conn.execute(
            'INSERT INTO uploads (id, owner_id, storage_key, sha256, width, height, mime, original_mime, has_alpha, created_at)'
            ' VALUES (?,?,?,?,?,?,?,?,?,?)',
            (upload_id, owner, key, clean.sha256, clean.width, clean.height, 'image/png', clean.original_mime,
             int(clean.has_alpha), clock.now()),
        )
    return {'upload_id': upload_id, 'sha256': clean.sha256, 'width': clean.width, 'height': clean.height,
            'mime': 'image/png', 'has_alpha': clean.has_alpha}
