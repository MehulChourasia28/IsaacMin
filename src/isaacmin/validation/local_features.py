"""Dense, source-defined thin-roof and portal diagnostics on actual triangles.

These measurements supplement the frozen centre/interface checks. They do not
introduce a new geometric tolerance or turn finite ray coverage into a safety
claim. Source feature endings, crop boundaries and unknowns remain explicit.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..io import atomic_json, sha256_file
from ..volumes.mesh_validation import _load_mesh
from ..volumes.support_validation import structural_support_for_validation
from ..volumes.source_evidence import verify_source_topology,recheck_source_topology


CONTEXT_BITS = {"source_feature_transition": 1, "crop_boundary": 2, "unknown_source": 4}
DIRECTIONS_XZ = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))


def _ray_hits(mesh, origins, directions):
    """Retain real triangle IDs/normals, with coincident hits deduplicated."""
    ray_records, distances, triangles, orientations = [], [], [], []
    duplicate_hits = 0
    for start in range(0, len(origins), 128):
        index_tri, index_ray, locations = mesh.ray.intersects_id(
            origins[start:start+128], directions[start:start+128],
            multiple_hits=True, return_locations=True)
        if not len(index_ray):
            continue
        distance = np.einsum("ij,ij->i", locations-origins[start+index_ray], directions[start+index_ray])
        order = np.lexsort((distance, index_ray))
        index_tri, index_ray, distance = index_tri[order], index_ray[order], distance[order]
        keep = np.r_[True, (np.diff(index_ray) != 0) | (np.diff(distance) > 1e-7)]
        duplicate_hits += int((~keep).sum())
        index_tri, index_ray, distance = index_tri[keep], index_ray[keep], distance[keep]
        ray_records.extend((start+index_ray).tolist())
        distances.extend(distance.tolist())
        triangles.extend(index_tri.tolist())
        orientations.extend(np.einsum("ij,ij->i", mesh.face_normals[index_tri], directions[start+index_ray]).tolist())
    rays = np.asarray(ray_records, dtype=np.int64)
    distances = np.asarray(distances, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)
    orientations = np.asarray(orientations, dtype=float)
    start = np.searchsorted(rays, np.arange(len(origins)), side="left")
    stop = np.searchsorted(rays, np.arange(len(origins)), side="right")
    return rays, distances, triangles, orientations, start, stop, duplicate_hits


def diagnose_local_features(ir_dir: Path, mesh_path: Path, output: Path, mesh_to_source, *,
                            support_manifest_path: Path | None = None,
                            max_source_roof_thickness_m: float = 2.,
                            stencil_offsets=(-.4, -.2, 0., .2, .4)) -> dict:
    """Measure local roofs/openings without changing frozen qualification limits.

    The default5x5 stencil lies inside each source unit face. Uniform interior
    roof patches and source transition edges are reported separately. Crossing
    order flags indicate possible overlapping/folded sheets or ray tangencies;
    they are not an exhaustive triangle/triangle self-intersection proof.
    """
    ir_dir, mesh_path, output = Path(ir_dir), Path(mesh_path), Path(output)
    graph,source_evidence=verify_source_topology(ir_dir)
    offsets = np.asarray(stencil_offsets, dtype=float)
    if (len(offsets) < 3 or not np.isfinite(offsets).all() or np.any(np.abs(offsets) >= .5)
            or len(set(offsets)) != len(offsets) or 0. not in offsets):
        raise ValueError("Diagnostic stencil requires unique offsets strictly inside a unit cell and includes its centre")
    if not 1 <= max_source_roof_thickness_m <= 6:
        raise ValueError("Thin roof diagnostic scope must lie between1m and protected6m source cover")
    ir_path = ir_dir / "world_ir.json"
    ir = json.loads(ir_path.read_text())
    for item in ir["files"]:
        from ..security import safe_path
        path = safe_path(ir_dir, item["path"], must_exist=True)
        if sha256_file(path) != item["sha256"]:
            raise ValueError("Source artifact changed before local-feature diagnostics")
    with np.load(ir_dir / "natural_occupancy.npz", allow_pickle=False) as data:
        occupancy, validity, minimum = data["occupancy"], data["validity"], data["min_xyz"].astype(int)
    with np.load(ir_dir / "terrain_surface.npz", allow_pickle=False) as data:
        protection = data["cave_column_protection"]
    with np.load(ir_dir / "topology/source_topology_labels.npz", allow_pickle=False) as data:
        covered = data["covered_air"]
    graph_path = ir_dir / "topology/source_topology_graph.json"
    structural, support_context = structural_support_for_validation(ir_dir, occupancy, support_manifest_path)
    solid = ((occupancy == 1) & validity) | structural
    air = (occupancy == 0) & validity
    mesh = _load_mesh(mesh_path, mesh_to_source)
    ny, nz, nx = solid.shape
    xmin, ymin, zmin = minimum

    def source_cell(xyz):
        x, y, z = np.floor(xyz).astype(int)-minimum
        if not (0 <= y < ny and 0 <= z < nz and 0 <= x < nx):
            return "crop"
        if not validity[y, z, x] or occupancy[y, z, x] == 3:
            return "unknown"
        return "air" if air[y, z, x] else "solid" if solid[y, z, x] else "excluded_non_ground"

    roofs, roof_context = [], []
    candidates = solid[1:] & covered[:-1]
    for z, x in np.argwhere(candidates.any(axis=0) & protection):
        non_solid = np.r_[np.flatnonzero(~solid[:, z, x]), ny]
        for bottom in np.flatnonzero(candidates[:, z, x])+1:
            top = int(non_solid[np.searchsorted(non_solid, bottom, side="right")])
            if top-bottom > max_source_roof_thickness_m:
                continue
            sides = []
            central_context = 0
            if bottom == 0 or top >= ny:
                central_context |= CONTEXT_BITS["crop_boundary"]
            elif not validity[bottom-1:top+1, z, x].all() or np.any(occupancy[bottom-1:top+1, z, x] == 3):
                central_context |= CONTEXT_BITS["unknown_source"]
            elif not (air[bottom-1, z, x] and air[top, z, x]):
                central_context |= CONTEXT_BITS["source_feature_transition"]
            for dx, dz in DIRECTIONS_XZ:
                xx, zz = x+dx, z+dz
                context = central_context
                if not (0 <= xx < nx and 0 <= zz < nz) or bottom == 0 or top >= ny:
                    context |= CONTEXT_BITS["crop_boundary"]
                elif not validity[bottom-1:top+1, zz, xx].all() or np.any(occupancy[bottom-1:top+1, zz, xx] == 3):
                    context |= CONTEXT_BITS["unknown_source"]
                elif not (solid[bottom:top, zz, xx].all() and air[bottom-1, zz, xx] and air[top, zz, xx]):
                    context |= CONTEXT_BITS["source_feature_transition"]
                sides.append(context)
            roofs.append([xmin+x+.5, zmin+z+.5, ymin+bottom, ymin+top])
            roof_context.append(sides)
    roofs = np.asarray(roofs, dtype=float).reshape((-1, 4))
    roof_context = np.asarray(roof_context, dtype=np.uint8).reshape((-1, 8))
    roof_points, roof_owner, roof_point_context = [], [], []
    for index, (x, z, bottom, top) in enumerate(roofs):
        for dz in offsets:
            for dx in offsets:
                sides = roof_context[index]
                context = 0
                for selected, side in ((dx < 0, 0), (dx > 0, 1), (dz < 0, 2), (dz > 0, 3)):
                    if selected:
                        context |= int(sides[side])
                for side, (sx, sz) in enumerate(DIRECTIONS_XZ[4:], start=4):
                    if dx*sx > 0 and dz*sz > 0:
                        context |= int(sides[side])
                # Centre measurements retain whole-cell context as well.
                if dx == 0 and dz == 0:
                    context = int(np.bitwise_or.reduce(sides))
                roof_points.append([x+dx, (bottom+top)/2, z+dz])
                roof_owner.append(index)
                roof_point_context.append(context)
    roof_points = np.asarray(roof_points, dtype=float).reshape((-1, 3))
    roof_owner = np.asarray(roof_owner, dtype=np.int64)
    roof_point_context = np.asarray(roof_point_context, dtype=np.uint8)
    xz, inverse = np.unique(roof_points[:, [0, 2]], axis=0, return_inverse=True)
    origin_y = float(mesh.bounds[0, 1])-1.
    roof_origins = np.column_stack((xz[:, 0], np.full(len(xz), origin_y), xz[:, 1]))
    roof_directions = np.tile([0., 1., 0.], (len(xz), 1))
    rr, rd, rt, rn, rs, re, roof_duplicates = _ray_hits(mesh, roof_origins, roof_directions)
    target_intervals = np.full((len(roof_points), 2), np.nan)
    potential_intersections = []
    for ray in range(len(xz)):
        values, normals = rd[rs[ray]:re[ray]], rn[rs[ray]:re[ray]]
        expected_sign = np.where(np.arange(len(normals)) % 2, 1., -1.)
        anomalous = np.flatnonzero(normals*expected_sign <= 1e-8)
        if len(values) % 2 or len(anomalous):
            potential_intersections.append({"kind": "roof_column", "source_xz": xz[ray].tolist(),
                "odd_crossing_count": bool(len(values) % 2), "nonalternating_or_tangent_crossing_indices": anomalous.tolist(),
                "crossing_source_y": (values+origin_y).tolist()})
    for index, xyz in enumerate(roof_points):
        ray = inverse[index]
        values = rd[rs[ray]:re[ray]]+origin_y
        interval = int(np.searchsorted(values, xyz[1], side="right"))
        if interval % 2 and 0 < interval < len(values):
            target_intervals[index] = values[interval-1:interval+1]
    thickness = target_intervals[:, 1]-target_intervals[:, 0]
    missing_roof = ~np.isfinite(thickness)

    portal_points, portal_origins, portal_directions, portal_owner, portal_context, portal_ids = [], [], [], [], [], []
    for portal in graph["portals"]:
        for face in portal["faces"]:
            centre = np.asarray(face["center_xyz"], dtype=float)
            axis = {"x": 0, "y": 1, "z": 2}[face["normal_axis"]]
            normal = np.zeros(3)
            normal[axis] = face["normal_sign"]
            tangents = [a for a in range(3) if a != axis]
            face_index = len(portal_ids)
            neighbor_context = []
            for da, db in DIRECTIONS_XZ:
                adjacent = centre.copy()
                adjacent[tangents] += [da, db]
                classes = {source_cell(adjacent+side*.35*normal) for side in (-1, 1)}
                context = 0
                if "crop" in classes:
                    context |= CONTEXT_BITS["crop_boundary"]
                if "unknown" in classes:
                    context |= CONTEXT_BITS["unknown_source"]
                if classes-{"air", "crop", "unknown"}:
                    context |= CONTEXT_BITS["source_feature_transition"]
                neighbor_context.append(context)
            face_context = int(np.bitwise_or.reduce(neighbor_context))
            portal_ids.append({"portal_id": portal["id"], "source_face": face,
                               "source_face_context": face_context, "source_neighbor_context": neighbor_context})
            for a in offsets:
                for b in offsets:
                    point = centre.copy()
                    point[tangents] += [a, b]
                    context = 0
                    # Test neighboring source aperture faces, retaining real
                    # wall/end/crop masks independently of target geometry.
                    for index, (da, db) in enumerate(DIRECTIONS_XZ):
                        if ((da == 0 or a == 0 or da*a > 0) and (db == 0 or b == 0 or db*b > 0)):
                            context |= neighbor_context[index]
                    portal_points.append(point)
                    portal_origins.append(point-.35*normal)
                    portal_directions.append(normal)
                    portal_owner.append(face_index)
                    portal_context.append(context)
    portal_points = np.asarray(portal_points).reshape((-1, 3))
    portal_origins = np.asarray(portal_origins).reshape((-1, 3))
    portal_directions = np.asarray(portal_directions).reshape((-1, 3))
    portal_context = np.asarray(portal_context, dtype=np.uint8)
    portal_face_context = np.asarray([face["source_face_context"] for face in portal_ids], dtype=np.uint8)
    portal_uniform_interior = portal_face_context[np.asarray(portal_owner, dtype=int)] == 0
    roof_uniform_interior = np.bitwise_or.reduce(roof_context, axis=1)[roof_owner] == 0
    pr, pd, pt, pn, ps, pe, portal_duplicates = _ray_hits(mesh, portal_origins, portal_directions)
    segment_crossings = np.zeros(len(portal_points), dtype=np.int32)
    portal_inside = np.zeros((len(portal_points), 3), dtype=bool)
    for ray in range(len(portal_points)):
        values, normals = pd[ps[ray]:pe[ray]], pn[ps[ray]:pe[ray]]
        segment_crossings[ray] = np.count_nonzero((values > 1e-7) & (values < .7-1e-7))
        portal_inside[ray] = [np.count_nonzero(values > d+1e-7) % 2 for d in (0., .35, .7)]
        if len(normals):
            expected_sign = np.where((np.arange(len(normals))+int(portal_inside[ray, 0])) % 2, 1., -1.)
            bad = np.flatnonzero(normals*expected_sign <= 1e-8)
            if len(bad):
                potential_intersections.append({"kind": "portal_segment_ray", "source_xyz": portal_points[ray].tolist(),
                    "nonalternating_or_tangent_crossing_indices": bad.tolist(), "distances_from_source_segment_origin": values.tolist()})
    portal_clearance = np.empty(len(portal_points), dtype=float)
    nearest_triangle = np.empty(len(portal_points), dtype=np.int64)
    for start in range(0, len(portal_points), 128):
        _, distance, triangle = mesh.nearest.on_surface(portal_points[start:start+128])
        portal_clearance[start:start+128], nearest_triangle[start:start+128] = distance, triangle

    output.mkdir(parents=True, exist_ok=True)
    raw = output / "local_feature_samples.npz"
    np.savez_compressed(raw, roof_source_intervals=roofs, roof_source_side_context=roof_context,
        roof_sample_source_xyz=roof_points, roof_sample_owner=roof_owner, roof_sample_context=roof_point_context,
        roof_target_intervals=target_intervals, roof_target_thickness=thickness,
        roof_ray_origins=roof_origins, roof_hit_ray=rr, roof_hit_distance=rd, roof_hit_triangle=rt, roof_hit_orientation=rn,
        portal_sample_source_xyz=portal_points, portal_sample_owner=np.asarray(portal_owner, dtype=np.int64),
        portal_sample_context=portal_context, portal_source_face_context=portal_face_context, portal_sample_inside=portal_inside,
        portal_segment_crossings=segment_crossings, portal_clearance=portal_clearance, portal_nearest_triangle=nearest_triangle,
        portal_ray_origins=portal_origins, portal_ray_directions=portal_directions,
        portal_hit_ray=pr, portal_hit_distance=pd, portal_hit_triangle=pt, portal_hit_orientation=pn)

    def summary(mask):
        measured = mask & ~missing_roof
        return {"samples": int(mask.sum()), "missing_support_at_source_midheight": int((mask & missing_roof).sum()),
                "minimum_measured_thickness_m": float(thickness[measured].min()) if measured.any() else None,
                "maximum_measured_thickness_m": float(thickness[measured].max()) if measured.any() else None}

    result = {"schema_version": 1, "kind": "IndependentLocalFeatureDiagnostic", "status": "measurements_complete",
        "qualification": "diagnostic_only; frozen centre/interface budgets are unchanged",
        "mesh_sha256": sha256_file(mesh_path), "source_ir_sha256": sha256_file(ir_path),
        "source_topology_sha256": sha256_file(graph_path), "validator_sha256": sha256_file(Path(__file__)),
        "source_payload_evidence":source_evidence,
        "mesh_to_source": np.asarray(mesh_to_source).tolist(), "structural_support": support_context,
        "sampling": {"stencil_offsets_m": offsets.tolist(), "source_roof_max_thickness_m": max_source_roof_thickness_m,
            "portal_segment_length_m": .7, "source_context_bits": CONTEXT_BITS,
            "roof_side_order": ["negative_x", "positive_x", "negative_z", "positive_z", "negative_x_negative_z", "negative_x_positive_z", "positive_x_negative_z", "positive_x_positive_z"],
            "source_neighbor_offsets": [list(offset) for offset in DIRECTIONS_XZ],
            "uniform_interior_definition": "Full3x3source neighborhood: all8neighbors retain identical solid roof span with known air immediately below and above; portal neighbors have known air on both sides of each aperture face",
            "edge_policy": "Neighboring source faces define transitions/endings before target tests; report interior and edge samples separately",
            "ray_coincidence_numerical_m": 1e-7},
        "roofs": {"source_roof_intervals": len(roofs), "unique_triangle_rays": len(xz),
            "all": summary(np.ones(len(thickness), dtype=bool)), "source_interior": summary(roof_uniform_interior),
            "source_transition_or_boundary": summary(~roof_uniform_interior),
            "missing_sample_indices": np.flatnonzero(missing_roof).tolist()},
        "portals": {"source_faces": len(portal_ids), "samples": len(portal_points),
            "samples_with_target_segment_crossings": int((segment_crossings > 0).sum()),
            "interior_samples_with_target_segment_crossings": int(((segment_crossings > 0) & portal_uniform_interior).sum()),
            "samples_with_inside_endpoint_or_centre": int(portal_inside.any(axis=1).sum()),
            "minimum_nearest_surface_clearance_m": float(portal_clearance.min()) if len(portal_clearance) else None,
            "crossing_sample_indices": np.flatnonzero(segment_crossings).tolist(), "faces": portal_ids},
        "potential_self_intersection_or_tangency_flags": potential_intersections,
        "self_intersection_scope": "Ray crossing parity/orientation diagnostics only; complete triangle-triangle intersection test not_run",
        "deduplicated_coincident_hits": {"roof": roof_duplicates, "portal": portal_duplicates},
        "mesh_positive_closed_volume": bool(mesh.is_volume),
        "files": [{"path": raw.name, "bytes": raw.stat().st_size, "sha256": sha256_file(raw)}],
        "limitations": ["Finite interior-face stencils do not prove arbitrary lateral clearance or navigation safety",
            "Source-defined open feature edges may taper naturally; their measurements are not labelled interior-roof failures",
            "Nearest-surface clearance is unsigned; inside flags and segment intersections must be interpreted with it",
            "No tolerance has been added or relaxed and no Isaac rendering/contact qualification follows from this diagnostic"]}
    recheck_source_topology(ir_dir,source_evidence)
    atomic_json(output / "local_feature_diagnostics.json", result)
    return result
