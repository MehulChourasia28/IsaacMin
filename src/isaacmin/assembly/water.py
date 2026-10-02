"""Source-derived exposed water surfaces, separate from supporting terrain."""
from __future__ import annotations

import numpy as np


def fluid_surface_mesh(kind, amount, minimum, surface_height, *, origin=(-1065.38, 0., 688.07), coverage_mask=None):
    """Every source water/known-air interface, with shared fluid-level corners.

    kind[y,z,x]:0 known air,1 solid,2 water,3 unknown/unsupported,4 other fluid,
    5 verified nonoccluding vegetation (distinct from source air).
    amount is the retained Java fluid amount1..8. Neighbor uncertainty is
    recorded explicitly; no unknown cell is promoted into air or source water.
    """
    kind, amount = np.asarray(kind), np.asarray(amount)
    minimum, origin = np.asarray(minimum, float), np.asarray(origin, float)
    if (kind.ndim != 3 or kind.shape != amount.shape or minimum.shape != (3,)
            or origin.shape != (3,) or np.asarray(surface_height).shape != kind.shape[1:]):
        raise ValueError("Fluid arrays or coordinate frame disagree")
    if not np.isin(kind, [0, 1, 2, 3, 4, 5]).all() or np.any((kind == 2) & ((amount < 1) | (amount > 8))):
        raise ValueError("Invalid retained source fluid states")
    coverage = np.ones(kind.shape, bool) if coverage_mask is None else np.asarray(coverage_mask, bool)
    if coverage.shape != kind.shape:
        raise ValueError("Fluid output coverage has a different shape")
    from scipy.ndimage import label
    bodies, body_count = label((kind == 2) & coverage)
    shape = kind.shape
    vertices, triangles, keys, face_cells, face_kinds, face_scopes, face_bodies = [], [], {}, [], [], [], []
    unknown_faces, unknown_corners = [], []
    source_cells = []

    def cell(y, z, x):
        return int(kind[y, z, x]) if 0 <= y < shape[0] and 0 <= z < shape[1] and 0 <= x < shape[2] else 3

    def height(y, z, x):
        category = cell(y, z, x)
        if category == 2:
            return 1. if cell(y+1, z, x) == 2 else float(amount[y, z, x])/9.
        return 0. if category in (0, 4, 5) else -1.

    def corner(y, z, x, dx, dz, centre):
        if centre >= 1:
            return 1.
        neighbors = [(y, z, x+dx), (y, z+dz, x)]
        a, b = (height(*p) for p in neighbors)
        if a >= 1. or b >= 1.:
            return 1.
        heights = [centre, a, b]
        if a > 0 or b > 0:
            neighbors.append((y, z+dz, x+dx))
            d = height(*neighbors[-1])
            if d >= 1.:
                return 1.
            heights.append(d)
        uncertain = [p for p in neighbors if cell(*p) == 3]
        if uncertain:
            unknown_corners.append({"cell_yzx": [y, z, x], "corner_dx_dz": [dx, dz],
                                    "unknown_neighbors_yzx": [list(p) for p in uncertain]})
        included = np.asarray([h for h in heights if h >= 0.])
        weights = np.where(included >= .8, 10., 1.)
        return float(np.sum(included*weights)/weights.sum())

    def face(points, y, z, x, face_kind):
        ids = []
        for px, py, pz in points:
            # Fluid corner rational arithmetic can differ by a final ULP when
            # approached from the adjacent source cell. Weld below1picometre.
            key = (round(px, 12), round(py, 12), round(pz, 12))
            if key not in keys:
                keys[key] = len(vertices)
                vertices.append([px+minimum[0]-origin[0], -pz-minimum[2]+origin[2], py+minimum[1]-origin[1]])
            ids.append(keys[key])
        triangles.extend(((ids[0], ids[1], ids[2]), (ids[0], ids[2], ids[3])))
        source = [int(x+minimum[0]), int(y+minimum[1]), int(z+minimum[2])]
        scope = "aquifer" if y+minimum[1]+1 < float(surface_height[z, x]) else "exterior"
        face_cells.extend([source, source]); face_kinds.extend([face_kind, face_kind])
        face_scopes.extend([scope, scope]); face_bodies.extend([int(bodies[y, z, x])]*2)

    for y, z, x in np.argwhere((kind == 2) & coverage):
        y, z, x = int(y), int(z), int(x)
        source_cells.append([x+int(minimum[0]), y+int(minimum[1]), z+int(minimum[2])])
        neighbours = {(0, 1, 0): cell(y+1, z, x), (0, -1, 0): cell(y-1, z, x),
                      (-1, 0, 0): cell(y, z, x-1), (1, 0, 0): cell(y, z, x+1),
                      (0, 0, -1): cell(y, z-1, x), (0, 0, 1): cell(y, z+1, x)}
        for direction, value in neighbours.items():
            if value == 3:
                unknown_faces.append({"cell_xyz": source_cells[-1], "direction_xyz": list(direction)})
        if not any(value in (0, 5) for value in neighbours.values()):
            continue
        centre = height(y, z, x)
        # Vertices in sourceXYZ, clockwise in sourceXZ -> upward in targetZ.
        h00, h01, h11, h10 = [corner(y, z, x, dx, dz, centre) for dx, dz in [(-1, -1), (-1, 1), (1, 1), (1, -1)]]
        top = [(x, y+h00, z), (x, y+h01, z+1), (x+1, y+h11, z+1), (x+1, y+h10, z)]
        if neighbours[(0, 1, 0)] in (0, 5):
            face(top, y, z, x, "top")
        if neighbours[(0, -1, 0)] in (0, 5):
            face([(x, y, z), (x+1, y, z), (x+1, y, z+1), (x, y, z+1)], y, z, x, "bottom")
        edges = [((-1, 0, 0), 1, 0), ((0, 0, 1), 2, 1), ((1, 0, 0), 3, 2), ((0, 0, -1), 0, 3)]
        for direction, a, b in edges:
            if neighbours[direction] in (0, 5):
                pa, pb = top[a], top[b]
                face([pa, pb, (pb[0], y, pb[2]), (pa[0], y, pa[2])], y, z, x, "side")
    from collections import Counter
    return {"vertices": vertices, "triangles": triangles, "triangle_source_cell_xyz": face_cells,
            "triangle_surface_kind": face_kinds, "triangle_scope": face_scopes, "triangle_body_id": face_bodies,
            "source_water_cells_xyz": source_cells, "source_body_count": int(body_count),
            "source_water_cells": len(source_cells), "surface_triangle_counts": dict(Counter(face_kinds)),
            "scope_triangle_counts": dict(Counter(face_scopes)), "unknown_adjacent_faces": unknown_faces,
            "uncertain_corner_neighbors": unknown_corners,
            "status": "incomplete" if unknown_faces or unknown_corners else "source_interfaces_constructed",
            "collision": "none; authoritative terrain supplies supporting ground",
            "surface_convention": "Java26.1 amount/9, samewaterabove=1, weighted fluid corner interpolation; no raster-only1mm z-fighting offset",
            "limits": ["Known air and verified nonoccluding glow_lichen interfaces; unsupported block-shape occlusion remains explicit",
                       "Static fluid surfaces preserve saved levels; no fluid dynamics or bubble simulation is claimed",
                       "Terrain-bank intersections and target transparency require exported-scene validation"]}


def source_water_from_ir(ir_directory, *, origin=(-1065.38, 0., 688.07), support_mask_path=None):
    """Decode retained exact fluid states without reducing stacked water to a heightmap."""
    import json
    from pathlib import Path
    from ..io import read_json, sha256_file
    from ..security import safe_path
    ir = Path(ir_directory)
    meta = read_json(ir / "world_ir.json")
    for entry in meta["files"]:
        if sha256_file(safe_path(ir, entry["path"], must_exist=True)) != entry["sha256"]:
            raise ValueError("Source water input differs from its immutable WorldIR")
    if set(c["data_version"] for c in meta["terrain_volume"]["chunk_records"]) != {4786}:
        raise ValueError("Fluid semantics require a separately verified runtime for this DataVersion")
    with np.load(ir / "natural_occupancy.npz", allow_pickle=False) as source:
        occupancy, validity, minimum = source["occupancy"], source["validity"], source["min_xyz"]
    kind = np.full(occupancy.shape, 3, np.uint8)
    kind[validity & (occupancy == 0)] = 0
    kind[validity & (occupancy == 1)] = 1
    kind[validity & (occupancy == 5)] = 4
    if support_mask_path is not None:
        with np.load(support_mask_path, allow_pickle=False) as support:
            if support["occupancy"].shape != kind.shape or not np.array_equal(support["min_xyz"], minimum):
                raise ValueError("Fluid bank support has a different coordinate frame")
            kind[support["structural_occupancy"] & validity] = 1
    amount = np.zeros(kind.shape, np.uint8)
    state_counts = {}

    def fluid_amount(state):
        name, properties = state["Name"], state.get("Properties", {})
        if name == "minecraft:water":
            level = int(properties["level"])
            if not 0 <= level <= 15:
                raise ValueError("Unexpected source water level")
            return 8 if level == 0 or level >= 8 else 8-level
        if name == "minecraft:bubble_column":
            return 8
        if properties.get("waterlogged") == "true":
            if name != "minecraft:glow_lichen":
                raise ValueError("Waterlogged block shape needs separately verified fluid-interface ownership: " + name)
            return 8
        return 0

    for item in meta["terrain_volume"]["chunk_records"]:
        with np.load(ir / item["volume_file"], allow_pickle=False) as chunk:
            palette = json.loads(str(chunk["palette_json"]))
            amounts = np.zeros(len(palette), np.uint8)
            for i, state in enumerate(palette):
                amounts[i] = fluid_amount(state)
                if amounts[i]:
                    key = json.dumps(state, sort_keys=True, separators=(",", ":"))
                    state_counts[key] = state_counts.get(key, 0)+int(np.count_nonzero(chunk["block_id"] == i))
            cx, _, cz = chunk["min_xyz"]
            for sy, block in zip(chunk["section_y"], chunk["block_id"]):
                selection = (slice(int(sy*16-minimum[1]), int(sy*16-minimum[1]+16)),
                             slice(int(cz-minimum[2]), int(cz-minimum[2]+16)),
                             slice(int(cx-minimum[0]), int(cx-minimum[0]+16)))
                liquid = amounts[block]
                amount[selection] = liquid
                kind[selection][liquid > 0] = 2
                nonoccluding = np.array([s["Name"] == "minecraft:glow_lichen" for s in palette])[block] & (liquid == 0)
                kind[selection][nonoccluding] = 5
    # One-cell source halo prevents artificial lowering at a cropped lake edge.
    # It is read from the same complete immutable save, never invented by padding.
    from ..source.anvil import Region, dimensions
    from ..source.provenance import begin_source_read, finish_source_read
    from ..source.semantics import classify, GROUND_CLASSES
    snapshot = Path(meta["source_path"])
    identity, before = begin_source_read(snapshot, "world_ir", meta["source_snapshot_sha256"])
    source_files = {entry["path"]: entry for entry in before}
    region_folder = dimensions(snapshot)[meta["dimension"]] / "region"
    h, nz, nx = kind.shape
    kind = np.pad(kind, ((0, 0), (1, 1), (1, 1)), constant_values=3)
    amount = np.pad(amount, ((0, 0), (1, 1), (1, 1)), constant_values=0)
    coverage = np.zeros(kind.shape, bool); coverage[:, 1:-1, 1:-1] = True
    minimum = minimum.copy(); minimum[[0, 2]] -= 1
    regions, chunks, used_files, missing_columns = {}, {}, set(), []
    boundary_columns = sorted({(z, x) for z in range(nz+2) for x in (0, nx+1)} |
                              {(z, x) for z in (0, nz+1) for x in range(nx+2)})
    for z, x in boundary_columns:
        sx, sz = int(x+minimum[0]), int(z+minimum[2]); cx, cz = sx//16, sz//16
        key = (cx//32, cz//32)
        if key not in regions:
            path = region_folder / f"r.{key[0]}.{key[1]}.mca"
            regions[key] = Region(path) if path.is_file() else None
            if path.is_file():
                used_files.add(path.relative_to(snapshot).as_posix())
        region = regions[key]
        if region is None or (cx, cz) not in region.locations:
            missing_columns.append([sx, sz]); continue
        if (cx, cz) not in chunks:
            chunks[cx, cz] = region.chunk(cx, cz)
        chunk = chunks[cx, cz]
        if not chunk.full:
            missing_columns.append([sx, sz]); continue
        if chunk.data_version != 4786:
            raise ValueError("Fluid halo DataVersion differs from verified source semantics")
        if chunk.metadata.get("external"):
            used_files.add((region.path.parent / f"c.{cx}.{cz}.mcc").relative_to(snapshot).as_posix())
        for y in range(h):
            sy = int(y+minimum[1])
            if sy//16 not in chunk.sections:
                continue
            state = chunk.block(sx, sy, sz)
            value = fluid_amount(state)
            if value:
                kind[y, z, x], amount[y, z, x] = 2, value
            elif state["Name"] == "minecraft:glow_lichen":
                kind[y, z, x] = 5
            elif state["Name"] in {"minecraft:air", "minecraft:cave_air", "minecraft:void_air"}:
                kind[y, z, x] = 0
            elif classify(state["Name"]) in GROUND_CLASSES or state["Name"] in {"minecraft:cobblestone", "minecraft:mossy_cobblestone"}:
                kind[y, z, x] = 1
            elif state["Name"] == "minecraft:lava":
                kind[y, z, x] = 4
    finish_source_read(snapshot, "world_ir", identity, before)
    with np.load(ir / "terrain_surface.npz", allow_pickle=False) as surface:
        heights = np.pad(surface["height"], ((1, 1), (1, 1)), constant_values=np.nan)
        result = fluid_surface_mesh(kind, amount, minimum, heights, origin=origin, coverage_mask=coverage)
    result.update(source_ir_sha256=sha256_file(ir / "world_ir.json"), exact_fluid_state_counts=state_counts,
                  source_data_version=4786, producer_sha256=sha256_file(Path(__file__)),
                  source_snapshot_sha256=meta["source_snapshot_sha256"],
                  halo={"width_cells": 1, "source_files": [source_files[name] for name in sorted(used_files)],
                        "missing_source_columns": missing_columns, "source_semantics_producer": identity["producer"],
                        "purpose": "Fluid face visibility and shared level corners at crop boundaries; halo is not additional output coverage"})
    if support_mask_path is not None:
        result["support_mask_sha256"] = sha256_file(Path(support_mask_path))
    return result


def source_water_mesh(surface, *, origin=(-1065.38, 0., 688.07)) -> dict:
    height = surface["height"]
    water = surface["water_height"]
    valid = surface["validity"] & surface["water_validity"] & (water > height)
    if not np.isfinite(water[valid]).all():
        raise ValueError("Nonfinite exposed water level")
    minimum = surface["min_xz"]
    ox, oy, oz = origin
    vertices, triangles, by_key = [], [], {}
    # A level is part of vertex identity so stacked/differing water bodies do not merge.
    for z, x in np.argwhere(valid):
        corners = ((x, z), (x+1, z), (x+1, z+1), (x, z+1))
        indices = []
        for cx, cz in corners:
            key = (int(cx), int(cz), float(water[z, x]))
            if key not in by_key:
                by_key[key] = len(vertices)
                vertices.append([float(minimum[0]+cx-ox), float(-minimum[1]-cz+oz), float(water[z, x]-oy)])
            indices.append(by_key[key])
        # Y=-sourceZ flips polygon orientation; upward normals require reversed winding.
        a, b, c, d = indices
        triangles.extend(((a, c, b), (a, d, c)))
    return {"vertices": vertices, "triangles": triangles,
            "source_columns": int(valid.sum()), "provenance": "source_exposed_water_top_faces",
            "collision": "none; underlying final terrain provides ground support",
            "sensor_exception": "transparent_water_depth_requires_target_renderer_semantics",
            "subsurface_aquifers": "retained_source_volume_separate_from_exposed_water",
            "qualification": "not_run"}
