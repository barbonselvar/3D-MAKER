"""Blender headless import check (gate V2).

Run as: blender --background --factory-startup --python import_check.py -- model.glb
Prints one line "IMPORT_CHECK_JSON:{...}" and exits 0 when the import succeeded.
"""
import json
import sys

import bpy  # provided by Blender


def main() -> int:
    path = sys.argv[sys.argv.index('--') + 1]
    bpy.ops.wm.read_factory_settings(use_empty=True)
    try:
        bpy.ops.import_scene.gltf(filepath=path)
    except Exception as e:  # noqa: BLE001 - report any importer failure
        print('IMPORT_CHECK_JSON:' + json.dumps({'ok': False, 'error': str(e)}))
        return 1
    meshes = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    materials = {m.name for o in meshes for m in o.data.materials if m}
    missing = [i.name for i in bpy.data.images if i.source in ('FILE', 'PACKED') and not i.has_data and i.size[0] == 0]
    report = {
        'ok': bool(meshes) and not missing,
        'blender_version': bpy.app.version_string,
        'mesh_objects': len(meshes),
        'triangles': sum(sum(len(p.vertices) - 2 for p in o.data.polygons) for o in meshes),
        'materials': len(materials),
        'images': len(bpy.data.images),
        'missing_images': missing,
    }
    print('IMPORT_CHECK_JSON:' + json.dumps(report))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
