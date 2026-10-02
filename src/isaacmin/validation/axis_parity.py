"""Bounded exact triangle-ray classification, independent of source occupancy.

Axis-aligned rays keep Rtree candidate boxes narrow on large terrain meshes.
Six opposing coordinate directions must agree; ambiguous boundary/tangency
queries use trimesh's original diagonal algorithm, never the expected label.
"""
from __future__ import annotations

import numpy as np
from trimesh.ray.ray_triangle import ray_triangle_id


def contains_axis_consensus(mesh, points, *, batch_size=128, coincidence_m=1e-7):
    """Return classifications and complete per-point independent ray evidence.

    This routine does not assert a mesh is closed or free of intersections;
    callers must retain their topology checks. Coincident triangle hits on a
    shared edge count once. Opposite orientations or surface-origin hits mark
    a query ambiguous and force the original algorithm rather than guessing.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Point queries must be finite XYZ triples")
    if not 1 <= batch_size <= 256 or not 0 < coincidence_m <= 1e-7:
        raise ValueError("Axis queries require bounded batches and fixed numerical precision")
    # Include all axes: a surface parallel to X/Z can otherwise give matching
    # parity in four directions while its query point lies on the boundary.
    directions = np.asarray([[1., 0, 0], [-1., 0, 0], [0, 0, 1.], [0, 0, -1.], [0, 1., 0], [0, -1., 0]])
    counts = np.zeros((len(points), len(directions)), dtype=np.int32)
    ambiguous = np.zeros_like(counts, dtype=bool)
    in_bounds = np.all(points >= mesh.bounds[0], axis=1) & np.all(points <= mesh.bounds[1], axis=1)
    selected = np.flatnonzero(in_bounds)
    # One shared BVH and immutable triangles/normals; only each batch's candidate
    # triangles and hits are materialized by the intersection implementation.
    triangles, normals, tree = mesh.triangles, mesh.face_normals, mesh.triangles_tree
    for direction_index, direction in enumerate(directions):
        for start in range(0, len(selected), batch_size):
            ids = selected[start:start+batch_size]
            face, ray, hit = ray_triangle_id(
                triangles, points[ids], np.tile(direction, (len(ids), 1)),
                triangles_normal=normals, tree=tree, multiple_hits=True)
            if not len(face):
                continue
            distance = (hit-points[ids[ray]]) @ direction
            order = np.lexsort((distance, ray))
            face, ray, distance = face[order], ray[order], distance[order]
            for local_ray in np.unique(ray):
                chosen = ray == local_ray
                distances, faces = distance[chosen], face[chosen]
                index = ids[local_ray]
                cuts = np.r_[0, np.flatnonzero(np.diff(distances) > coincidence_m)+1, len(distances)]
                for low, high in zip(cuts[:-1], cuts[1:]):
                    orientation = normals[faces[low:high]] @ direction
                    positive = np.any(orientation > 1e-12)
                    negative = np.any(orientation < -1e-12)
                    if abs(distances[low]) <= coincidence_m or positive == negative:
                        ambiguous[index, direction_index] = True
                    counts[index, direction_index] += 1
    parity = counts % 2 == 1
    disagreement = np.any(parity != parity[:, :1], axis=1)
    fallback = np.flatnonzero(in_bounds & (disagreement | ambiguous.any(axis=1)))
    inside = in_bounds & parity[:, 0]
    for start in range(0, len(fallback), batch_size):
        ids = fallback[start:start+batch_size]
        inside[ids] = mesh.contains(points[ids])
    evidence = {"axis_directions": directions, "axis_crossing_counts": counts,
                "axis_ambiguous": ambiguous, "axis_parity_disagreement": disagreement,
                "axis_fallback_indices": fallback, "axis_in_mesh_bounds": in_bounds}
    return inside, evidence
