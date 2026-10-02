"""Material-aware baseline and global drainage. Neither is the erosion backend.

Drainage follows a priority-flood outlet forest (Barnes et al., 2014).
Depression filling here is diagnostic: it never fills caves/water in geometry.
https://doi.org/10.1016/j.cageo.2013.04.024
"""
from __future__ import annotations

import heapq
from pathlib import Path

import numpy as np

from ..io import atomic_json, sha256_file

OFFSETS = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))


def priority_drainage(height, valid, *, spacing_m: float) -> dict:
    h, valid = np.asarray(height, float), np.asarray(valid, bool)
    if h.ndim != 2 or valid.shape != h.shape or not valid.any() or not np.isfinite(h[valid]).all():
        raise ValueError("Drainage requires finite valid terrain samples")
    if spacing_m <= 0:
        raise ValueError("Sample spacing must be positive")
    nz, nx = h.shape
    filled = np.full(h.shape, np.nan)
    receiver = np.full(h.shape, -2, np.int64)
    basin = np.full(h.shape, -1, np.int64)
    seen = np.zeros(h.shape, bool)
    heap = []
    outlet_id = 0
    for z, x in np.argwhere(valid):
        # A boundary against absent source is an open unknown outlet, not invented land.
        edge = any(not (0 <= z+dz < nz and 0 <= x+dx < nx) or not valid[z+dz, x+dx]
                   for dz, dx in OFFSETS[:4])
        if edge:
            filled[z, x], receiver[z, x], basin[z, x], seen[z, x] = h[z, x], -1, outlet_id, True
            heapq.heappush(heap, (h[z, x], int(z), int(x)))
            outlet_id += 1
    order = []
    while heap:
        elevation, z, x = heapq.heappop(heap)
        order.append(z * nx + x)
        for dz, dx in OFFSETS:
            qz, qx = z + dz, x + dx
            if not (0 <= qz < nz and 0 <= qx < nx) or not valid[qz, qx] or seen[qz, qx]:
                continue
            # Do not route diagonally through two absent cells.
            if dz and dx and not (valid[z, qx] and valid[qz, x]):
                continue
            seen[qz, qx] = True
            filled[qz, qx] = max(elevation, h[qz, qx])
            receiver[qz, qx] = z * nx + x
            basin[qz, qx] = basin[z, x]
            heapq.heappush(heap, (filled[qz, qx], qz, qx))
    if not np.all(seen[valid]):
        raise ValueError("Drainage left valid cells disconnected from an outlet")
    accumulation = valid.astype(float) * spacing_m**2
    flat_acc, flat_receiver = accumulation.ravel(), receiver.ravel()
    for index in reversed(order):
        parent = flat_receiver[index]
        if parent >= 0:
            flat_acc[parent] += flat_acc[index]
    return {"filled_height_m": filled, "receiver_flat_index": receiver, "basin_id": basin,
            "contributing_area_m2": accumulation, "depression_depth_m": filled-h,
            "valid": valid, "outlet_count": outlet_id, "traversal_order": np.asarray(order, np.int64)}


def reconstruct_baseline(height, valid, material, protection, *, iterations=32,
                         soil_limit_m=.85, rock_limit_m=.25) -> tuple[np.ndarray, dict]:
    """Constrained robust diffusion of quantized samples; preserves cliff breaks.

    Different substrate envelopes and original steep edges govern transport.
    This is a baseline only. Actual HighMap weathering must follow it and is
    independently evaluated. Protected features are enforced each iteration.
    """
    source = np.asarray(height, dtype=np.float64)
    valid = np.asarray(valid, bool)
    material = np.asarray(material)
    protection = np.asarray(protection, bool)
    if source.ndim != 2 or any(a.shape != source.shape for a in (valid, material, protection)):
        raise ValueError("Baseline masks must match height shape")
    if not valid.any() or not np.isfinite(source[valid]).all():
        raise ValueError("Missing or invalid source ground")
    if not 1 <= iterations <= 128:
        raise ValueError("Baseline iteration bound exceeded")
    working = np.where(valid, source, 0.)
    # caller uses 1=soil, 2=sediment, 3=rock; other/unknown substrates protected.
    soil = np.isin(material, [1, 2])
    rock = material == 3
    limit = np.where(soil, soil_limit_m, np.where(rock, rock_limit_m, 0.))
    limit[protection | ~valid] = 0
    weights = []
    for axis in (0, 1):
        a = [slice(None), slice(None)]
        b = [slice(None), slice(None)]
        a[axis], b[axis] = slice(None, -1), slice(1, None)
        a, b = tuple(a), tuple(b)
        difference = np.abs(working[a] - working[b])
        w = np.exp(-(difference / 1.5)**2) * (valid[a] & valid[b])
        # Abrupt rock/soil interfaces are real constraints, not a colour blend.
        w *= np.where(material[a] == material[b], 1., .15)
        w[difference > 3] = 0.
        weights.append((a, b, w))
    original = working.copy()
    for _ in range(iterations):
        flux = np.zeros_like(working)
        for a, b, w in weights:
            exchange = w * (working[b] - working[a])
            flux[a] += exchange
            flux[b] -= exchange
        working += .16 * flux + .04 * (original-working)
        working = np.clip(working, original-limit, original+limit)
        working[protection] = original[protection]
    working[~valid] = np.nan
    delta = working-source
    return working.astype(np.float32), {"method": "substrate_constrained_robust_diffusion",
            "role": "baseline_only_not_erosion", "iterations": iterations,
            "soil_envelope_m": soil_limit_m, "rock_envelope_m": rock_limit_m,
            "protected_samples": int((protection & valid).sum()),
            "protected_max_delta_m": float(np.max(np.abs(delta[protection & valid]), initial=0)),
            "max_delta_m": float(np.max(np.abs(delta[valid]))),
            "rms_delta_m": float(np.sqrt(np.mean(delta[valid]**2))),
            "remaining_visual_qualification": "not_run"}


def save_drainage(output: Path, height, valid, *, spacing_m: float, source_hash: str) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    result = priority_drainage(height, valid, spacing_m=spacing_m)
    arrays = {k: v for k, v in result.items() if isinstance(v, np.ndarray)}
    path = output / "global_drainage.npz"
    np.savez_compressed(path, **arrays)
    total = float(result["contributing_area_m2"][result["receiver_flat_index"] == -1].sum())
    expected = float(np.asarray(valid).sum() * spacing_m**2)
    if abs(total - expected) > 1e-5:
        raise ValueError("Drainage outlet area does not conserve contributing area")
    record = {"status": "numerically_verified_cpu", "source_hash": source_hash,
              "producer_sha256": sha256_file(Path(__file__)),
              "algorithm": "priority_flood_outlet_forest", "sample_spacing_m": spacing_m,
              "outlet_count": result["outlet_count"], "outlet_area_m2": total,
              "valid_source_area_m2": expected, "array_path": path.name, "sha256": sha256_file(path),
              "geometry_modified": False, "physical_hydrological_simulation": False,
              "boundary_policy": "known perimeter and missing-coverage boundaries are open outlets",
              "scope": "global declared source region; shared by child terrain tiles"}
    atomic_json(output / "drainage.json", record)
    return record
