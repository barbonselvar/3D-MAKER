"""P1-06: POST /api/v1/uploads validation and sanitisation."""
import io

import pytest
from PIL import Image

from app.core import images


def encode(im: Image.Image, fmt: str, **kw) -> bytes:
    buf = io.BytesIO()
    im.save(buf, fmt, **kw)
    return buf.getvalue()


def post(api, data: bytes, name='x.png', mime='image/png'):
    return api.post('/api/v1/uploads', files={'image': (name, data, mime)})


def stored_bytes(api, upload_id) -> bytes:
    ctx = api.app.state.ctx
    with ctx.db.connect() as conn:
        key = conn.execute('SELECT storage_key FROM uploads WHERE id=?', (upload_id,)).fetchone()[0]
    return ctx.storage.get(key)


def stored_png(api, upload_id) -> Image.Image:
    return Image.open(io.BytesIO(stored_bytes(api, upload_id)))


@pytest.mark.parametrize('fmt,mime', [('PNG', 'image/png'), ('JPEG', 'image/jpeg'), ('WEBP', 'image/webp')])
def test_accepts_png_jpeg_webp_and_stores_png(api, fmt, mime):
    r = post(api, encode(Image.new('RGB', (64, 48), (200, 30, 30)), fmt), mime=mime)
    assert r.status_code == 201, r.text
    body = r.json()
    assert body['upload_id'].startswith('upl_')
    assert (body['width'], body['height'], body['mime']) == (64, 48, 'image/png')
    assert len(body['sha256']) == 64
    assert stored_png(api, body['upload_id']).format == 'PNG'


def test_type_comes_from_bytes_not_filename(api):
    jpeg = encode(Image.new('RGB', (8, 8)), 'JPEG')
    r = post(api, jpeg, name='lies.png', mime='image/png')
    assert r.status_code == 201
    with api.app.state.ctx.db.connect() as conn:
        assert conn.execute('SELECT original_mime FROM uploads').fetchone()[0] == 'image/jpeg'


@pytest.mark.parametrize('payload', [
    b'GIF89a' + b'\0' * 100,
    b'<svg xmlns="http://www.w3.org/2000/svg"/>',
    b'%PDF-1.7',
    b'\x89PNG\r\n\x1a\n' + b'garbage' * 10,  # PNG magic, corrupt body
    b'',
])
def test_rejects_non_images_and_corrupt_files(api, payload):
    r = post(api, payload)
    assert r.status_code == 400
    assert r.json()['error']['code'] == 'INPUT_INVALID'


def test_rejects_over_12_mb(api):
    r = post(api, b'\x89PNG\r\n\x1a\n' + b'0' * (12 * 1024 * 1024))
    assert r.status_code == 413
    assert r.json()['error']['code'] == 'INPUT_TOO_LARGE'


def test_rejects_over_40_megapixels(api):
    r = post(api, encode(Image.new('1', (7000, 6000)), 'PNG'))  # 42 MP, a few KB on disk
    assert r.status_code == 400
    assert 'megapixels' in r.json()['error']['message']


def test_decompression_bomb_rejected():
    bomb = encode(Image.new('1', (12000, 12000)), 'PNG')  # 144 MP in ~20 KB
    assert len(bomb) < 100_000
    with pytest.raises(images.InvalidImage):
        images.sanitize(bomb)


def test_exif_and_gps_stripped_and_orientation_applied(api):
    im = Image.new('RGB', (40, 20), (10, 200, 10))
    exif = Image.Exif()
    exif[0x0112] = 6  # orientation: rotate 90° CW
    exif[0x8825] = {2: (33.0, 35.0, 0.0), 1: 'N'}  # GPS IFD
    exif[0x010F] = 'CameraMaker'
    data = encode(im, 'JPEG', exif=exif.tobytes())
    assert b'CameraMaker' in data
    body = post(api, data, mime='image/jpeg').json()
    assert (body['width'], body['height']) == (20, 40)
    stored = stored_png(api, body['upload_id'])
    assert not stored.getexif()
    assert 'exif' not in stored.info and 'icc_profile' not in stored.info
    assert b'CameraMaker' not in stored_bytes(api, body['upload_id'])


def test_alpha_detection(api):
    rgba = Image.new('RGBA', (16, 16), (0, 0, 0, 0))
    rgba.putpixel((8, 8), (255, 0, 0, 255))
    assert post(api, encode(rgba, 'PNG')).json()['has_alpha'] is True
    opaque = Image.new('RGBA', (16, 16), (1, 2, 3, 255))
    assert post(api, encode(opaque, 'PNG')).json()['has_alpha'] is False


def test_icc_profile_converted_to_srgb_and_dropped(api):
    from PIL import ImageCms
    profile = ImageCms.ImageCmsProfile(ImageCms.createProfile('LAB')).tobytes()
    im = Image.new('RGB', (8, 8), (120, 60, 30))
    body = post(api, encode(im, 'PNG', icc_profile=profile)).json()
    assert 'icc_profile' not in stored_png(api, body['upload_id']).info
    assert b'iCCP' not in stored_bytes(api, body['upload_id'])


def test_upload_requires_token_when_configured(settings):
    from fastapi.testclient import TestClient

    from app.main import create_app
    settings.api_token = 'owner-secret-token'
    client = TestClient(create_app(settings))
    png = encode(Image.new('RGB', (8, 8)), 'PNG')
    assert post(client, png).status_code == 401
    r = client.post('/api/v1/uploads', files={'image': ('a.png', png, 'image/png')},
                    headers={'Authorization': 'Bearer owner-secret-token'})
    assert r.status_code == 201
