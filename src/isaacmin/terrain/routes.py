"""Source-ground route candidates; no claim of contact or navigation success."""
from __future__ import annotations

import heapq
import math
import numpy as np


def route_candidate(height, valid, water, *, spacing_m: float, centre_index: tuple[int, int],
                    origin_xz: tuple[float, float], world_origin=(0., 0., 0.),
                    max_grade=.45) -> dict:
    h, valid, water = np.asarray(height, float), np.asarray(valid, bool), np.asarray(water, bool)
    if h.ndim != 2 or valid.shape != h.shape or water.shape != h.shape or spacing_m <= 0:
        raise ValueError("Invalid route surface contract")
    allowed = valid & ~water & np.isfinite(h)
    cells = np.argwhere(allowed)
    if not len(cells):
        return {"status": "blocked", "reason": "No dry supporting source ground"}
    start = tuple(cells[np.argmin(((cells-np.asarray(centre_index))**2).sum(axis=1))])
    nz, nx = h.shape

    def search(start):
        dist = {start: 0.}
        parent = {start: None}
        heap = [(0., start)]
        while heap:
            d, a = heapq.heappop(heap)
            if dist[a] != d:
                continue
            z, x = a
            for dz, dx in ((1, 0), (0, 1), (-1, 0), (0, -1), (1, 1), (-1, -1), (1, -1), (-1, 1)):
                qz, qx = z+dz, x+dx
                if not (0 <= qz < nz and 0 <= qx < nx) or not allowed[qz, qx]:
                    continue
                if dz and dx and not (allowed[z, qx] and allowed[qz, x]):
                    continue
                horizontal = spacing_m * math.hypot(dz, dx)
                rise = h[qz, qx] - h[z, x]
                grade = abs(rise) / horizontal
                if grade > max_grade:
                    continue
                cost = math.hypot(horizontal, rise) * (1 + grade**2 * 10)
                b, nd = (qz, qx), d+cost
                if nd < dist.get(b, float("inf")):
                    dist[b], parent[b] = nd, a
                    heapq.heappush(heap, (nd, b))
        return dist, parent

    distances, _ = search(start)
    endpoint = max(distances, key=distances.get)
    distances, parents = search(endpoint)
    end = max(distances, key=distances.get)
    path = []
    while end is not None:
        path.append(end)
        end = parents[end]
    path.reverse()
    ox, _, oz = world_origin
    points = [[origin_xz[0]+x*spacing_m-ox, -(origin_xz[1]+z*spacing_m-oz), float(h[z, x])]
              for z, x in path]
    p = np.asarray(points)
    length = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum()) if len(points) > 1 else 0
    return {"status": "candidate", "provenance": "authored_scenario_route_not_extracted_trail",
            "points_world_xyz": points, "surface_indices_zx": [list(map(int, a)) for a in path],
            "unique_polyline_length_m": length, "repeated_edges": 0, "corridor_width_m": 1.2,
            "max_source_grade": max_grade, "sample_spacing_m": spacing_m,
            "contact_validation": "not_run", "navigation_stack": "not_run_not_supplied",
            "limitations": ["Coarse source support does not establish final-surface clearance",
                            "No geometry changed and no trail wear authored by this planning operation"]}
