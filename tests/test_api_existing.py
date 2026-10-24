"""Regression tests for the pre-existing API (ported from the blueprint audit smoke test)."""
import io

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from app.main import app


@pytest.fixture()
def client():
    return TestClient(app)


def png_circle(size=256) -> bytes:
    im = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(im).ellipse((40, 40, size - 40, size - 40), fill=(213, 250, 107, 255))
    buf = io.BytesIO()
    im.save(buf, 'PNG')
    return buf.getvalue()


def test_health_reports_no_ai(client):
    body = client.get('/api/health').json()
    assert body['status'] == 'ok'
    assert body['ai_enabled'] is False


def test_index_served(client):
    r = client.get('/')
    assert r.status_code == 200
    assert '3D Maker' in r.text


def test_circle_generates_watertight_glb(client):
    r = client.post('/api/generate', files={'image': ('c.png', png_circle(), 'image/png')}, data={'depth': '0.3'})
    assert r.status_code == 200
    body = r.json()
    assert body['method'].startswith('silhouette extrusion (not AI')
    glb = client.get(body['preview'])
    assert glb.status_code == 200
    mesh = trimesh.load(io.BytesIO(glb.content), file_type='glb', force='mesh')
    assert mesh.is_watertight
    assert len(mesh.faces) == body['faces']
    for fmt in ('obj', 'stl', 'ply'):
        assert client.get(f"/api/files/{body['id']}/model.{fmt}").status_code == 200


@pytest.mark.parametrize('path', [
    '/api/files/zz/model.glb',
    '/api/files/' + 'g' * 32 + '/model.glb',
    '/api/files/' + 'a' * 32 + '/passwd',
    '/api/files/../etc/passwd',
])
def test_bad_file_paths_404(client, path):
    assert client.get(path).status_code == 404


def test_noisy_photo_rejected(client):
    noise = (np.random.default_rng(0).random((256, 256, 3)) * 255).astype('uint8')
    buf = io.BytesIO()
    Image.fromarray(noise).save(buf, 'JPEG')
    r = client.post('/api/generate', files={'image': ('n.jpg', buf.getvalue(), 'image/jpeg')})
    assert r.status_code == 422


def test_non_image_rejected(client):
    r = client.post('/api/generate', files={'image': ('x.png', b'not an image', 'image/png')})
    assert r.status_code == 400


def test_depth_out_of_range(client):
    r = client.post('/api/generate', files={'image': ('c.png', png_circle(), 'image/png')}, data={'depth': '5'})
    assert r.status_code == 400
