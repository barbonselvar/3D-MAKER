"""Deterministic mesh cleanup (blueprint pipeline stage 6). CPU-only, unit-tested.

Runs on the textured trimesh produced by o_voxel.postprocess.to_glb, keeping UVs
and materials intact (faces are removed, never re-triangulated).
"""
from __future__ import annotations

import numpy as np
import trimesh


def drop_small_components(mesh: trimesh.Trimesh, min_area_fraction: float = 0.005) -> tuple[trimesh.Trimesh, dict]:
    """Remove connected parts smaller than min_area_fraction of the surface (floating debris, rain specks)."""
    faces = len(mesh.faces)
    if faces == 0:
        return mesh, {'removed_components': 0, 'removed_faces': 0}
    welded = mesh.copy()
    welded.merge_vertices(merge_tex=True, merge_norm=True)  # UV seams must not split parts
    labels = trimesh.graph.connected_component_labels(welded.face_adjacency, node_count=faces)
    area = np.bincount(labels, weights=mesh.area_faces)
    keep_labels = area >= min_area_fraction * area.sum()
    keep = keep_labels[labels]
    removed = int((~keep_labels).sum())
    if removed:
        mesh = mesh.copy()
        mesh.update_faces(keep)
        mesh.remove_unreferenced_vertices()
    return mesh, {'removed_components': removed, 'removed_faces': int(faces - keep.sum()),
                  'components_kept': int(keep_labels.sum())}


def cleanup(mesh: trimesh.Trimesh, min_area_fraction: float = 0.005) -> tuple[trimesh.Trimesh, dict]:
    mesh, report = drop_small_components(mesh, min_area_fraction)
    return mesh, report
