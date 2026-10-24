"""Shared helpers for API tests."""
import io

from PIL import Image

from app.core import clock
from app.core.db import dumps

L40S_PROBE = {'backend': 'cuda', 'devices': [{'name': 'NVIDIA L40S', 'vram_total_mb': 46068, 'vram_free_mb': 45512,
                                              'driver': '550.54.15', 'cuda': '12.4', 'compute_capability': '8.9'}],
              'torch': {'installed': True, 'cuda_available': True}, 'errors': []}


def png_bytes(size=(64, 64), color=(200, 40, 40)) -> bytes:
    buf = io.BytesIO()
    Image.new('RGB', size, color).save(buf, 'PNG')
    return buf.getvalue()


def upload(api) -> str:
    r = api.post('/api/v1/uploads', files={'image': ('a.png', png_bytes(), 'image/png')})
    assert r.status_code == 201, r.text
    return r.json()['upload_id']


def online_worker(api, worker_id='w1', probe=None, engines=('trellis2-4b',)):
    with api.app.state.ctx.db.connect() as conn:
        conn.execute('INSERT OR REPLACE INTO workers (id, kind, probe, engines, heartbeat_at, created_at) VALUES (?,?,?,?,?,?)',
                     (worker_id, 'internal', dumps(probe or L40S_PROBE), dumps(list(engines)), clock.now(), clock.now()))


def create_job(api, upload_id, **overrides):
    body = {'upload_id': upload_id, 'target': 'internal', 'engine': 'trellis2-4b'} | overrides
    return api.post('/api/v1/jobs', json=body)
