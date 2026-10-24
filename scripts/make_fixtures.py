"""Regenerate deterministic test fixtures under tests/fixtures/.

These are hand-built test assets, not AI output; they exist only to exercise
loaders, exporters and validators.
"""
import json
import struct
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

OUT = Path(__file__).resolve().parents[1] / 'tests' / 'fixtures'


def textured_cube() -> bytes:
    """1 m cube offset from the origin, with UVs and a checker baseColorTexture."""
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    box.unmerge_vertices()  # in place: per-face vertices so UVs are well defined
    uv = np.zeros((len(box.vertices), 2))
    for face in box.faces:
        uv[face] = [[0, 0], [1, 0], [1, 1]]
    checker = np.indices((64, 64)).sum(axis=0) // 8 % 2
    img = Image.fromarray(np.where(checker[..., None], [230, 120, 40], [30, 30, 30]).astype('uint8'))
    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=img, baseColorFactor=[255, 255, 255, 255], metallicFactor=0.0, roughnessFactor=0.8)
    box.visual = trimesh.visual.TextureVisuals(uv=uv, material=material)
    box.apply_translation([3.0, 0.5, -2.0])
    return trimesh.Scene(box).export(file_type='glb')


def glb_json(data: bytes) -> dict:
    length = struct.unpack_from('<I', data, 12)[0]
    return json.loads(data[20:20 + length])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cube = textured_cube()
    doc = glb_json(cube)
    assert doc['materials'][0]['pbrMetallicRoughness'].get('baseColorTexture'), 'texture missing'
    (OUT / 'textured_cube.glb').write_bytes(cube)
    print('wrote textured_cube.glb', len(cube), 'bytes')


if __name__ == '__main__':
    main()
