import io
import os
import tempfile
from pathlib import Path
from uuid import uuid4

import numpy as np
import trimesh
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parent
OUTPUT = Path(os.environ.get('MAKER_OUTPUT', tempfile.gettempdir() + '/3d-maker-output'))
OUTPUT.mkdir(parents=True, exist_ok=True)
app = FastAPI(title='3D Maker', version='0.1.0')
app.mount('/static', StaticFiles(directory=ROOT / 'static'), name='static')

@app.get('/api/health')
def health():
    return {'status': 'ok', 'modes': ['silhouette'], 'ai_enabled': False, 'blender_available': bool(os.environ.get('BLENDER_BIN'))}

@app.post('/api/generate')
async def generate(image: UploadFile = File(...), depth: float = Form(0.25)):
    if not 0.02 <= depth <= 2:
        raise HTTPException(400, 'Depth must be between 0.02 and 2')
    raw = await image.read(12 * 1024 * 1024 + 1)
    if len(raw) > 12 * 1024 * 1024:
        raise HTTPException(413, 'Image too large (12 MB maximum)')
    try:
        im = Image.open(io.BytesIO(raw))
        if im.format not in ('PNG', 'JPEG', 'WEBP'):
            raise ValueError('Unsupported image format')
        im.thumbnail((512, 512))
        rgba = im.convert('RGBA')
    except Exception:
        raise HTTPException(400, 'Upload a valid PNG, JPG, or WEBP image') from None
    arr = np.asarray(rgba)
    alpha = arr[:, :, 3]
    # Prefer an alpha mask. Otherwise separate the object from the corner background color.
    if alpha.min() < 240:
        mask = alpha > 90
    else:
        rgb = arr[:, :, :3].astype(np.int16)
        corners = np.array([rgb[0,0], rgb[0,-1], rgb[-1,0], rgb[-1,-1]])
        bg = np.median(corners, axis=0)
        mask = np.linalg.norm(rgb - bg, axis=2) > 38
    from PIL import ImageFilter
    m = Image.fromarray((mask * 255).astype('uint8')).resize((128,128)).filter(ImageFilter.MedianFilter(3))
    mask = np.asarray(m) > 127
    if mask.sum() < 80 or mask.sum() > mask.size * 0.98:
        raise HTTPException(422, 'Could not isolate an object. Use a PNG with a transparent background.')
    # Contour extraction from binary pixel edges; largest polygon selected.
    import cv2
    contours, _ = cv2.findContours(mask.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    eps = 0.004 * cv2.arcLength(contour, True)
    pts = cv2.approxPolyDP(contour, eps, True)[:, 0, :].astype(float)
    if len(pts) < 3:
        raise HTTPException(422, 'Object outline is too small')
    pts[:, 1] = 127 - pts[:, 1]
    pts = (pts - 63.5) / 63.5
    poly = Polygon(pts).buffer(0)
    if poly.is_empty or poly.geom_type != 'Polygon':
        raise HTTPException(422, 'Cannot build a valid outline')
    mesh = trimesh.creation.extrude_polygon(poly, height=depth)
    mesh.apply_translation([0, 0, -depth/2])
    identifier = uuid4().hex
    folder = OUTPUT / identifier
    folder.mkdir()
    mesh.export(folder / 'model.glb')
    mesh.export(folder / 'model.obj')
    mesh.export(folder / 'model.stl')
    mesh.export(folder / 'model.ply')
    return {'id': identifier, 'preview': f'/api/files/{identifier}/model.glb', 'formats': ['glb','obj','stl','ply'], 'method': 'silhouette extrusion (not AI reconstruction)', 'faces': len(mesh.faces)}

@app.get('/api/files/{identifier}/{filename}')
def download(identifier: str, filename: str):
    if len(identifier) != 32 or not all(c in '0123456789abcdef' for c in identifier):
        raise HTTPException(404)
    if filename not in {'model.glb', 'model.obj', 'model.stl', 'model.ply'}:
        raise HTTPException(404)
    path = OUTPUT / identifier / filename
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, filename=filename if filename != 'model.glb' else '3d-maker.glb')


# Serve the studio (index.html plus ./vendor and ./js modules) at the site root.
# Mounted last so every /api route registered above takes precedence.
app.mount('/', StaticFiles(directory=ROOT / 'static', html=True), name='site')
