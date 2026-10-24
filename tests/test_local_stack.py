"""python -m app.local_stack: real servers on localhost ports with the MOCK FakeGpuEngine (no GPU)."""
import socket
import threading
import time

import httpx

from app import local_stack
from app.core.config import Settings

from .helpers import png_bytes
from .test_selftest import RTX4050, FakeGpuEngine


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


class FakeClass:
    """Class-like factory so prepare_engines can call install_problem() and instantiate."""

    @staticmethod
    def install_problem():
        return None

    def __new__(cls):
        return FakeGpuEngine(max_ok=512, low_needed=True)


def wait_for(fn, timeout=20):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = fn()
        if value:
            return value
        time.sleep(0.2)
    raise AssertionError('timed out')


def test_local_stack_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setenv('MAKER_WORKER_STATE', str(tmp_path / 'state'))
    port, worker_port, stop = free_port(), free_port(), threading.Event()
    settings = Settings(data_dir=tmp_path / 'data', secret_key=b'k' * 32)
    thread = threading.Thread(target=local_stack.run, daemon=True, kwargs=dict(
        port=port, worker_port=worker_port, engines=['sf3d'], settings=settings, stop=stop, caps=RTX4050,
        classes={'sf3d': FakeClass}))
    thread.start()
    base = f'http://127.0.0.1:{port}'
    try:
        def ready():
            try:
                t = httpx.get(base + '/api/v1/gpu-targets').json()['targets'][0]
            except httpx.HTTPError:
                return None
            return t if t['available'] else None
        internal = wait_for(ready)
        assert internal['engines'][0] == {'id': 'sf3d', 'label': 'Stable Fast 3D', 'quality': 'preview', 'textured': True,
                                          'max_texture_size': 512, 'est_seconds': internal['engines'][0]['est_seconds']}
        assert httpx.get(base + '/').status_code == 200
        up = httpx.post(base + '/api/v1/uploads', files={'image': ('a.png', png_bytes(), 'image/png')}).json()
        job = httpx.post(base + '/api/v1/jobs', json={'upload_id': up['upload_id'], 'target': 'internal', 'engine': 'sf3d',
                                                      'params': {'texture_size': 512}}).json()
        done = wait_for(lambda: (j := httpx.get(f"{base}/api/v1/jobs/{job['job_id']}").json())['status'] in
                        ('completed', 'failed') and j)
        assert done['status'] == 'completed', done
        model = httpx.get(f"{base}/api/v1/models/{done['model_id']}").json()
        glb = httpx.get(base + next(f['url'] for f in model['files'] if f['format'] == 'glb'), follow_redirects=True)
        assert glb.content[:4] == b'glTF'
        # The worker API listens on localhost only and rejects requests without this run's token.
        assert httpx.post(f'http://127.0.0.1:{worker_port}/internal/v1/claim', json={'engines': []}).status_code == 401
        assert (tmp_path / 'state' / 'selftest.json').exists()
    finally:
        stop.set()
        thread.join(10)
