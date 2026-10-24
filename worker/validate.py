"""Artifact validation gates V1-V4 (blueprint evaluation protocol).

V1 structural  Khronos glTF-Validator: 0 errors
V2 loadability Blender LTS imports it headless (three.js load is covered by the
               viewer/UI tests and the evaluation harness)
V3 mesh sanity no NaN/inf, degenerate faces < 0.1 %, faces <= budget, largest
               component >= 90 % of surface area unless multi-object
V4 materials   textures required by the engine's material profile present at the
               requested size, PNG/JPEG only (no WebP), UVs inside 0-1

The result decides which downloads are offered: a GLB is published only when
every required gate passed.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from .glb import GlbError, buffer_view_bytes, read_accessor, read_glb

HERE = Path(__file__).resolve().parent
UV_EPS = 1e-4

# Which textures each engine family must deliver (app/core/engines.py Engine.material).
MATERIAL_PROFILES = {
    # TRELLIS.2: base colour + packed metallic-roughness textures.
    'pbr_textures': {'baseColorTexture': True, 'metallicRoughnessTexture': True},
    # Stable Fast 3D: base colour + normal map textures; metallic/roughness are scalar factors upstream.
    'basecolor_normal': {'baseColorTexture': True, 'normalTexture': True},
    'basecolor': {'baseColorTexture': True},
}


@dataclass
class Gate:
    id: str
    name: str
    status: str  # passed | failed | skipped
    message: str = ''
    details: dict = field(default_factory=dict)


@dataclass
class Report:
    gates: list[Gate]
    required: list[str]

    @property
    def passed(self) -> bool:
        by_id = {g.id: g for g in self.gates}
        return all(by_id[r].status == 'passed' for r in self.required if r in by_id) and \
            not any(g.status == 'failed' for g in self.gates)

    def failures(self) -> list[Gate]:
        return [g for g in self.gates if g.status == 'failed' or (g.id in self.required and g.status != 'passed')]

    def to_dict(self) -> dict:
        return {'passed': self.passed, 'required': self.required, 'gates': [asdict(g) for g in self.gates]}


# --- V1 ------------------------------------------------------------------------------------------

def gate_v1(path: Path, node: str = 'node') -> Gate:
    script = HERE / 'validators' / 'gltf_validate.mjs'
    if shutil.which(node) is None or not (HERE / 'node_modules' / 'gltf-validator').is_dir():
        return Gate('V1', 'glTF-Validator', 'skipped', 'Node.js or gltf-validator not installed (run npm ci in worker/).')
    try:
        out = subprocess.run([node, str(script), str(path)], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=300)
    except subprocess.TimeoutExpired:
        return Gate('V1', 'glTF-Validator', 'failed', 'Validator timed out.')
    if out.returncode != 0:
        return Gate('V1', 'glTF-Validator', 'failed', (out.stderr.strip().splitlines() or ['validator crashed'])[-1])
    report = json.loads(out.stdout)
    issues = report['issues']
    errors = [m for m in issues['messages'] if m['severity'] == 0]
    details = {'validator': report.get('validatorVersion'), 'errors': issues['numErrors'],
               'warnings': issues['numWarnings'], 'first_errors': [f"{m['code']}: {m['message']}" for m in errors[:5]],
               'info': {k: report.get('info', {}).get(k) for k in ('totalTriangleCount', 'materialCount', 'hasTextures')}}
    ok = issues['numErrors'] == 0
    return Gate('V1', 'glTF-Validator', 'passed' if ok else 'failed',
                '' if ok else f"{issues['numErrors']} glTF error(s): {details['first_errors'][0] if errors else ''}", details)


# --- V2 ------------------------------------------------------------------------------------------

def gate_v2(path: Path, blender: str | None, expect_triangles: int | None, expect_materials: int | None) -> Gate:
    if not blender or shutil.which(blender) is None:
        return Gate('V2', 'Blender import', 'skipped', 'Blender not installed (set BLENDER_BIN).')
    cmd = [blender, '--background', '--factory-startup', '--python', str(HERE / 'blender' / 'import_check.py'), '--', str(path)]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=600)
    except subprocess.TimeoutExpired:
        return Gate('V2', 'Blender import', 'failed', 'Blender import timed out.')
    line = next((ln for ln in out.stdout.splitlines() if ln.startswith('IMPORT_CHECK_JSON:')), None)
    if line is None:
        return Gate('V2', 'Blender import', 'failed', f'Blender exited with {out.returncode} and no report.')
    details = json.loads(line.split(':', 1)[1])
    problems = []
    if not details.get('ok'):
        problems.append(details.get('error') or 'import produced no mesh or unresolved images')
    if expect_triangles is not None and details.get('triangles') not in (None, expect_triangles):
        problems.append(f"triangle count {details['triangles']} != {expect_triangles}")
    if expect_materials is not None and details.get('materials') not in (None, expect_materials):
        problems.append(f"material count {details['materials']} != {expect_materials}")
    return Gate('V2', 'Blender import', 'failed' if problems else 'passed', '; '.join(problems), details)


# --- V3 ------------------------------------------------------------------------------------------

def load_mesh(data: bytes) -> trimesh.Trimesh:
    scene = trimesh.load(io.BytesIO(data), file_type='glb', force='scene', process=False)
    meshes = [g for g in scene.dump() if isinstance(g, trimesh.Trimesh)] if scene.geometry else []
    if not meshes:
        return trimesh.Trimesh()
    return trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]


def gate_v3(data: bytes, face_budget: int, multi_object: bool) -> Gate:
    try:
        mesh = load_mesh(data)
    except Exception as e:  # noqa: BLE001 - any loader error is a failed gate
        return Gate('V3', 'Mesh sanity', 'failed', f'Mesh could not be loaded: {e}')
    faces = len(mesh.faces)
    if faces == 0:
        return Gate('V3', 'Mesh sanity', 'failed', 'The model contains no triangles.', {'faces': 0})
    finite = bool(np.isfinite(mesh.vertices).all())
    details = {'faces': faces, 'vertices': len(mesh.vertices), 'face_budget': face_budget, 'finite': finite}
    problems = []
    if not finite:
        problems.append('vertex positions contain NaN or infinity')
    else:
        extent = float(np.ptp(mesh.vertices, axis=0).max()) or 1.0
        area = mesh.area_faces
        degenerate = float((area <= (extent * 1e-7) ** 2).mean())
        # Engines duplicate vertices along UV seams; weld by position only so UV islands are not
        # mistaken for separate parts.
        welded = mesh.copy()
        welded.merge_vertices(merge_tex=True, merge_norm=True)
        labels = trimesh.graph.connected_component_labels(welded.face_adjacency, node_count=faces)
        per_component = np.bincount(labels, weights=area)
        largest = float(per_component.max() / per_component.sum()) if per_component.sum() > 0 else 0.0
        details.update(degenerate_fraction=round(degenerate, 6), components=int(per_component.size),
                       largest_component_area=round(largest, 4), bbox_extent=round(extent, 6))
        if degenerate >= 0.001:
            problems.append(f'{degenerate:.2%} degenerate faces (limit 0.1 %)')
        if not multi_object and largest < 0.9:
            problems.append(f'largest connected part covers {largest:.0%} of the surface (need 90 %)')
    if faces > face_budget:
        problems.append(f'{faces} faces exceeds the budget of {face_budget}')
    return Gate('V3', 'Mesh sanity', 'failed' if problems else 'passed', '; '.join(problems), details)


# --- V4 ------------------------------------------------------------------------------------------

def gate_v4(data: bytes, texture_size: int | None, profile: str = 'pbr_textures') -> Gate:
    try:
        doc, binary = read_glb(data)
    except (GlbError, ValueError) as e:
        return Gate('V4', 'Materials', 'failed', f'GLB could not be parsed: {e}')
    problems, textures = [], []
    if 'EXT_texture_webp' in doc.get('extensionsUsed', []):
        problems.append('WebP textures are not allowed (export with extension_webp=False)')

    def texture_info(tex_ref, role):
        tex = doc['textures'][tex_ref['index']]
        image = doc['images'][tex['source']] if 'source' in tex else None
        if image is None:
            problems.append(f'{role} texture has no image')
            return
        mime = image.get('mimeType')
        if mime not in ('image/png', 'image/jpeg'):
            problems.append(f'{role} texture is {mime or "external"}, expected PNG or JPEG')
            return
        size = Image.open(io.BytesIO(buffer_view_bytes(doc, binary, image['bufferView']))).size
        textures.append({'role': role, 'mime': mime, 'size': list(size)})
        if texture_size and size != (texture_size, texture_size):
            problems.append(f'{role} texture is {size[0]}x{size[1]}, expected {texture_size}x{texture_size}')

    required = MATERIAL_PROFILES[profile]
    names = {'baseColorTexture': 'base colour', 'metallicRoughnessTexture': 'metallic-roughness', 'normalTexture': 'normal'}
    materials = doc.get('materials', [])
    if not materials:
        problems.append('no materials')
    for i, mat in enumerate(materials):
        pbr = mat.get('pbrMetallicRoughness', {})
        slots = {'baseColorTexture': pbr.get('baseColorTexture'), 'metallicRoughnessTexture': pbr.get('metallicRoughnessTexture'),
                 'normalTexture': mat.get('normalTexture')}
        for slot, ref in slots.items():
            if ref is not None:
                texture_info(ref, f'material {i} {names[slot]}')
            elif required.get(slot):
                problems.append(f'material {i} has no {names[slot]} texture')

    uv_range = [np.inf, -np.inf]
    for mesh in doc.get('meshes', []):
        for prim in mesh['primitives']:
            if 'TEXCOORD_0' not in prim['attributes']:
                if 'material' in prim:
                    problems.append('a textured primitive has no UVs')
                continue
            uv = read_accessor(doc, binary, prim['attributes']['TEXCOORD_0']).astype(np.float64)
            if uv.size:
                uv_range = [min(uv_range[0], float(uv.min())), max(uv_range[1], float(uv.max()))]
    if np.isfinite(uv_range[0]) and (uv_range[0] < -UV_EPS or uv_range[1] > 1 + UV_EPS):
        problems.append(f'UVs span {uv_range[0]:.3f}..{uv_range[1]:.3f}, outside 0-1')
    details = {'profile': profile, 'materials': len(materials), 'textures': textures,
               'uv_range': uv_range if np.isfinite(uv_range[0]) else None}
    return Gate('V4', 'Materials', 'failed' if problems else 'passed', '; '.join(problems), details)


# --- entry point ---------------------------------------------------------------------------------

def validate_glb(data: bytes, *, face_budget: int, texture_size: int | None = None, require_pbr: bool = True,
                 multi_object: bool = False, blender: str | None = None, require_blender: bool | None = None,
                 profile: str | None = None) -> Report:
    """profile overrides require_pbr (True -> 'pbr_textures', False -> 'basecolor')."""
    profile = profile or ('pbr_textures' if require_pbr else 'basecolor')
    blender = blender if blender is not None else os.environ.get('BLENDER_BIN')
    if require_blender is None:
        require_blender = os.environ.get('VALIDATION_REQUIRE_BLENDER') == '1'
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'model.glb'
        path.write_bytes(data)
        v1 = gate_v1(path)
        v3 = gate_v3(data, face_budget, multi_object)
        v4 = gate_v4(data, texture_size, profile)
        v2 = gate_v2(path, blender, v3.details.get('faces'), v4.details.get('materials'))
    required = ['V1', 'V3', 'V4'] + (['V2'] if require_blender else [])
    return Report([v1, v2, v3, v4], required)


if __name__ == '__main__':
    import argparse

    ap = argparse.ArgumentParser(description='Run validation gates V1-V4 on a GLB.')
    ap.add_argument('glb')
    ap.add_argument('--face-budget', type=int, default=200_000)
    ap.add_argument('--texture-size', type=int)
    ap.add_argument('--profile', choices=sorted(MATERIAL_PROFILES), default='pbr_textures')
    a = ap.parse_args()
    r = validate_glb(Path(a.glb).read_bytes(), face_budget=a.face_budget, texture_size=a.texture_size, profile=a.profile)
    print(json.dumps(r.to_dict(), indent=2))
    raise SystemExit(0 if r.passed else 1)
