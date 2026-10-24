"""Regenerate deterministic test fixtures under tests/fixtures/.

These are hand-built test assets, not AI output; they exist only to exercise
loaders, exporters and validators.
"""
import io
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


def png(size=64, color=(200, 120, 40)) -> bytes:
    buf = io.BytesIO()
    Image.new('RGB', (size, size), color).save(buf, 'PNG')
    return buf.getvalue()


def cube_arrays(offset=(0.0, 0.0, 0.0)):
    box = trimesh.creation.box(extents=(1.0, 1.0, 1.0))
    box.unmerge_vertices()
    uv = np.zeros((len(box.vertices), 2))
    for face in box.faces:
        uv[face] = [[0, 0], [1, 0], [1, 1]]
    return box.vertices + np.asarray(offset), box.faces, uv


def build_glb(positions, faces, uvs=None, textures=None, mime='image/png', extensions_used=None) -> bytes:
    """Hand-rolled GLB writer so fixtures can contain defects real exporters refuse to write."""
    positions = np.asarray(positions, dtype=np.float32)
    indices = np.asarray(faces, dtype=np.uint32).reshape(-1)
    blobs, views, accessors = [], [], []

    def add(data: bytes, target=None):
        offset = sum(len(b) for b in blobs)
        blobs.append(data + b'\0' * (-len(data) % 4))
        view = {'buffer': 0, 'byteOffset': offset, 'byteLength': len(data)}
        if target:
            view['target'] = target
        views.append(view)
        return len(views) - 1

    finite = positions[np.isfinite(positions).all(axis=1)]
    accessors.append({'bufferView': add(positions.tobytes(), 34962), 'componentType': 5126, 'count': len(positions),
                      'type': 'VEC3', 'min': finite.min(axis=0).tolist(), 'max': finite.max(axis=0).tolist()})
    attributes = {'POSITION': 0}
    if uvs is not None:
        uvs = np.asarray(uvs, dtype=np.float32)
        accessors.append({'bufferView': add(uvs.tobytes(), 34962), 'componentType': 5126, 'count': len(uvs), 'type': 'VEC2'})
        attributes['TEXCOORD_0'] = 1
    accessors.append({'bufferView': add(indices.tobytes(), 34963), 'componentType': 5125, 'count': len(indices),
                      'type': 'SCALAR'})
    pbr, images, gl_textures = {'metallicFactor': 1.0, 'roughnessFactor': 1.0}, [], []
    for role, data in (textures or {}).items():
        images.append({'bufferView': add(data), 'mimeType': mime})
        gl_textures.append({'source': len(images) - 1})
        pbr[role] = {'index': len(gl_textures) - 1}
    doc = {'asset': {'version': '2.0', 'generator': '3d-maker test fixture'}, 'scene': 0,
           'scenes': [{'nodes': [0]}], 'nodes': [{'mesh': 0}],
           'meshes': [{'primitives': [{'attributes': attributes, 'indices': len(accessors) - 1, 'material': 0}]}],
           'materials': [{'pbrMetallicRoughness': pbr}], 'accessors': accessors, 'bufferViews': views,
           'buffers': [{'byteLength': sum(len(b) for b in blobs)}]}
    if images:
        doc.update(images=images, textures=gl_textures)
    if extensions_used:
        doc['extensionsUsed'] = extensions_used
    js = json.dumps(doc).encode()
    js += b' ' * (-len(js) % 4)
    binary = b''.join(blobs)
    total = 12 + 8 + len(js) + 8 + len(binary)
    return (b'glTF' + struct.pack('<II', 2, total) + struct.pack('<II', len(js), 0x4E4F534A) + js
            + struct.pack('<II', len(binary), 0x004E4942) + binary)


def crafted_fixtures() -> dict[str, bytes]:
    pbr = {'baseColorTexture': png(1024), 'metallicRoughnessTexture': png(1024, (0, 128, 255))}
    v, f, uv = cube_arrays()
    nan_v = v.copy()
    nan_v[3] = [np.nan, 0, 0]
    v2, f2, uv2 = cube_arrays((5, 0, 0))
    sliver = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=float)
    deg_v = np.vstack([v] + [sliver] * 20)
    deg_f = np.vstack([f] + [np.arange(3) + len(v) + 3 * i for i in range(20)])
    deg_uv = np.vstack([uv, np.zeros((60, 2))])
    good = build_glb(v, f, uv, pbr)
    return {
        'pbr_cube.glb': good,
        'sf3d_style_512.glb': sf3d_style(512),
        'sf3d_style_1024.glb': sf3d_style(1024),
        'pbr_cube_2048.glb': build_glb(v, f, uv, {'baseColorTexture': png(2048), 'metallicRoughnessTexture': png(2048, (0, 128, 255))}),
        'bad_nan.glb': build_glb(nan_v, f, uv, pbr),
        'bad_uv_range.glb': build_glb(v, f, uv * 1.5, pbr),
        'bad_webp.glb': build_glb(v, f, uv, pbr, mime='image/webp', extensions_used=['EXT_texture_webp']),
        'bad_truncated.glb': good[: len(good) // 2],
        'bad_fragmented.glb': build_glb(np.vstack([v, v2]), np.vstack([f, f2 + len(v)]), np.vstack([uv, uv2]), pbr),
        'bad_degenerate.glb': build_glb(deg_v, deg_f, deg_uv, pbr),
        'bad_untextured.glb': build_glb(v, f, uv, None),
    }


def sf3d_style(size=512) -> bytes:
    """Mimics how Stable Fast 3D (sf3d/system.py) assembles its output: trimesh PBRMaterial with a JPEG base
    colour texture, a JPEG normal map, scalar roughness/metallic, exported with include_normals=True.
    Geometry is a plain UV sphere: this is a format fixture, not model output."""
    sphere = trimesh.creation.uv_sphere(radius=0.5, count=[32, 32])
    v = sphere.vertices
    uv = np.stack([0.5 + np.arctan2(v[:, 1], v[:, 0]) / (2 * np.pi), 0.5 + np.arcsin(np.clip(v[:, 2] / 0.5, -1, 1)) / np.pi], 1)
    base = Image.fromarray(np.full((size, size, 3), (180, 60, 40), np.uint8))
    base.format = 'JPEG'
    normal = Image.fromarray(np.full((size, size, 3), (128, 128, 255), np.uint8))
    normal.format = 'JPEG'
    material = trimesh.visual.material.PBRMaterial(baseColorTexture=base, roughnessFactor=0.6, metallicFactor=0.1,
                                                   normalTexture=normal)
    mesh = trimesh.Trimesh(vertices=v, faces=sphere.faces,
                           visual=trimesh.visual.texture.TextureVisuals(uv=np.clip(uv, 0, 1), material=material))
    return mesh.export(file_type='glb', include_normals=True)


def smoke_subject() -> bytes:
    """Shaded sphere on transparency: a simple, unambiguous input for the GPU smoke run."""
    n = 512
    y, x = (np.mgrid[0:n, 0:n] - n / 2 + 0.5) / (n * 0.4)
    r2 = x**2 + y**2
    inside = r2 <= 1
    z = np.sqrt(np.clip(1 - r2, 0, 1))
    light = np.clip(0.25 + 0.75 * (z * 0.8 - x * 0.3 - y * 0.5), 0, 1)
    rgb = (np.stack([0.85 * light, 0.25 * light, 0.2 * light], -1) * 255).astype('uint8')
    alpha = (inside * 255).astype('uint8')
    buf = io.BytesIO()
    Image.fromarray(np.dstack([rgb, alpha])).save(buf, 'PNG')
    return buf.getvalue()


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
    smoke = OUT.parents[1] / 'worker' / 'smoke' / 'sphere.png'
    smoke.write_bytes(smoke_subject())
    print('wrote', smoke.relative_to(OUT.parents[1]))
    for name, data in crafted_fixtures().items():
        (OUT / name).write_bytes(data)
        print('wrote', name, len(data), 'bytes')


if __name__ == '__main__':
    main()
