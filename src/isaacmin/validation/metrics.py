"""Numerical acceptance checks with explicit units and independent inputs."""
from __future__ import annotations

import math
import numpy as np


def distances(a, b) -> np.ndarray:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3 or len(a) == 0:
        raise ValueError("Expected matching nonempty Nx3 point arrays")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Nonfinite sample")
    return np.linalg.norm(a - b, axis=1)


def compare_ground(render_points, collision_points, *, p999: float = .010, maximum: float = .020) -> dict:
    error = distances(render_points, collision_points)
    quantile = float(np.quantile(error, .999))
    largest = float(error.max())
    return {"samples": len(error), "p999_m": quantile, "max_m": largest,
            "pass": quantile <= p999 and largest <= maximum}


def compare_seam(a, b, maximum: float = .002) -> dict:
    error = distances(a, b)
    return {"samples": len(error), "max_gap_m": float(error.max()), "pass": bool(error.max() <= maximum)}


def compare_depth(measured, geometry, opaque_valid, *, semantics: str,
                  ray_cosines=None, tolerance_m: float = .02, min_samples: int = 10000) -> dict:
    measured, geometry = np.asarray(measured, float), np.asarray(geometry, float)
    valid = np.asarray(opaque_valid)
    if measured.shape != geometry.shape or valid.shape != measured.shape or valid.dtype != bool:
        raise ValueError("Depth, geometry and boolean mask must have matching shape")
    if semantics == "euclidean_range":
        cosines = np.asarray(ray_cosines, float)
        if cosines.shape != measured.shape or not np.isfinite(cosines).all() or (cosines <= 0).any() or (cosines > 1).any():
            raise ValueError("Range conversion requires positive per-pixel camera ray cosines")
        measured = measured * cosines
    elif semantics != "axial_z":
        raise ValueError("Depth semantics must be explicitly axial_z or euclidean_range")
    if valid.sum() < min_samples:
        raise ValueError("Insufficient opaque depth comparisons")
    if not np.isfinite(measured[valid]).all() or not np.isfinite(geometry[valid]).all():
        raise ValueError("Nonfinite depth in valid samples")
    if (measured[valid] <= 0).any() or (geometry[valid] <= 0).any():
        raise ValueError("Valid camera depth must be positive")
    error = np.abs(measured[valid] - geometry[valid])
    return {"samples": int(valid.sum()), "max_m": float(error.max()),
            "p999_m": float(np.quantile(error, .999)), "pass": bool((error <= tolerance_m).all()),
            "comparison_semantics": "axial_z"}


def unique_route_distance(points, *, grid_m: float = 0.25) -> dict:
    """Length of unique undirected quantized segments, without credit for laps.

    Subdivide longer moves before counting. Quantization introduces a measured
    discretization; raw trajectory length is retained separately. This is a
    coverage measure, not evidence that a route is traversable.
    """
    p = np.asarray(points, float)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 2 or not np.isfinite(p).all():
        raise ValueError("Expected a finite trajectory of at least two points")
    edges = {}
    raw = 0.0
    for a, b in zip(p[:-1], p[1:]):
        length = float(np.linalg.norm(b - a))
        raw += length
        segments = max(1, math.ceil(length / (grid_m * .5)))
        samples = np.linspace(a, b, segments + 1)
        cells = np.round(samples / grid_m).astype(np.int64)
        for ca, cb in zip(cells[:-1], cells[1:]):
            if np.array_equal(ca, cb):
                continue
            edge = tuple(sorted((tuple(ca), tuple(cb))))
            edges[edge] = float(np.linalg.norm(cb - ca)) * grid_m
    return {"raw_distance_m": raw, "unique_distance_m": min(raw, sum(edges.values())),
            "unique_segment_count": len(edges), "discretization_m": grid_m}


def critic_calibration(cases: list[dict]) -> dict:
    mandatory = {"seam", "floating_asset", "missing_material", "broken_opacity", "voxel_steps"}
    seen = {c["category"] for c in cases if c.get("severe") and c.get("held_out")}
    misses = [c["id"] for c in cases if c.get("severe") and not c.get("detected")]
    controls = [c for c in cases if not c.get("severe")]
    false_positives = [c["id"] for c in controls if c.get("detected")]
    return {"pass": mandatory <= seen and not misses and len(controls) > 0 and not false_positives,
            "missed_severe": misses, "missing_held_out_categories": sorted(mandatory - seen),
            "control_false_positives": false_positives, "control_count": len(controls)}


def mesh_integrity(vertices, triangles) -> dict:
    v, f = np.asarray(vertices, float), np.asarray(triangles, np.int64)
    if v.ndim != 2 or v.shape[1] != 3 or f.ndim != 2 or f.shape[1] != 3 or not len(f):
        raise ValueError("Expected nonempty triangle mesh")
    if not np.isfinite(v).all() or f.min() < 0 or f.max() >= len(v):
        raise ValueError("Invalid triangle positions or vertex indices")
    area2 = np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1)
    canonical = np.sort(f, axis=1)
    duplicates = len(canonical) - len(np.unique(canonical, axis=0))
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return {"vertices": len(v), "triangles": len(f), "degenerate_faces": int((area2 <= 1e-12).sum()),
            "duplicate_faces": duplicates, "boundary_edges": int((counts == 1).sum()),
            "nonmanifold_edges": int((counts > 2).sum())}
