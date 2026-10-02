"""Compare native PhysX hit positions with independent final-triangle rays."""
from pathlib import Path

import numpy as np

from ..io import atomic_json, read_json, sha256_file
from .capture_plan import load_ground
from .metrics import compare_ground


def validate_dropped_probes(mesh_path: Path, capture_dir: Path, output: Path) -> dict:
    """Validate settled native body poses against complete oriented cube surfaces.

    A tilted cube does not rest at ground+half-edge. Dense samples cover all six
    faces; their signed nearest-triangle distances detect penetration and actual
    upward support. This does not replace dynamic traversal or the ray test.
    """
    import itertools
    import trimesh
    from scipy.spatial.transform import Rotation
    native_path = capture_dir / "capture_result.json"
    native = read_json(native_path)
    request = read_json(capture_dir / "capture_request.json")
    if native.get("physx_contact_rays", {}).get("final_ground_sha256") != sha256_file(mesh_path):
        raise ValueError("Dropped body poses refer to different final supporting geometry")
    contacts = native["contacts"]
    if len(contacts) != len(request["contact_probes"]) or not contacts:
        raise ValueError("Missing dropped-body pose measurements")
    mesh = load_ground(mesh_path, output/"native_ground_queries")
    records = []
    probe_jobs = []
    all_points, all_nearest, all_signed, all_probe_ids = [], [], [], []
    for index, (contact, specification) in enumerate(zip(contacts, request["contact_probes"])):
        if contact.get("id") != index or contact.get("probe_shape") != "cube":
            raise ValueError("Unsupported or mismatched measured probe shape")
        edge = float(contact["edge_length_m"])
        if not 0 < edge <= .5:
            raise ValueError("Invalid measured probe dimensions")
        q = np.asarray(contact["settled_orientation_wxyz"], float)
        position = np.asarray(contact["settled_position"], float)
        if q.shape != (4,) or position.shape != (3,) or not np.isfinite(q).all() or not np.isfinite(position).all() or abs(np.linalg.norm(q)-1) > 1e-5:
            raise ValueError("Invalid native dropped-body transform")
        # <=12.5mm grid gives <=8.9mm nearest sample on a face, below the
        # existing20mm support tolerance. This sample spacing is not a new gate.
        n = max(3, int(np.ceil(edge/.0125))+1)
        points = []
        for axis in range(3):
            other = [i for i in range(3) if i != axis]
            for side in (-edge/2, edge/2):
                for u, v in itertools.product(np.linspace(-edge/2, edge/2, n), repeat=2):
                    p = np.zeros(3); p[axis] = side; p[other] = [u, v]; points.append(p)
        points = np.unique(points, axis=0)
        points = Rotation.from_quat(np.r_[q[1:], q[0]]).apply(points)+position
        probe_jobs.append((index,contact,specification,edge,position,points))
    native_queries=getattr(mesh,'queries',None);offsets=np.r_[0,np.cumsum([len(job[-1]) for job in probe_jobs])]
    if native_queries is not None:
        points=np.concatenate([job[-1] for job in probe_jobs]);all_closest,all_distance,all_triangles=native_queries.closest(points)
        far=all_distance>.02;all_inside=np.zeros(len(points),bool)
        if far.any():all_inside[far]=native_queries.contains(points[far])
    for job_index,(index,contact,specification,edge,position,points) in enumerate(probe_jobs):
        if native_queries is None:closest,distance,triangles=trimesh.proximity.closest_point(mesh,points)
        else:
            selected=slice(offsets[job_index],offsets[job_index+1]);closest,distance,triangles=all_closest[selected],all_distance[selected],all_triangles[selected]
        normals = mesh.face_normals[triangles]
        signed = distance*np.sign(np.sum((points-closest)*normals, axis=1))
        support = (normals[:, 2] > .5) & (closest[:, 2] <= position[2]+.002)
        support_gap = float(np.min(distance[support])) if support.any() else None
        # Nearest-face normals can be ambiguous at edges. Resolve every sample
        # beyond the tolerance using full triangle inside/outside parity.
        far = distance > .02
        if far.any():
            inside = mesh.contains(points[far]) if native_queries is None else all_inside[selected][far]
            signed[far] = np.where(inside, -distance[far], distance[far])
        penetration = float(np.max(np.maximum(-signed, 0)))
        linear = float(np.linalg.norm(contact["linear_velocity_mps"]))
        angular = float(np.linalg.norm(contact["angular_velocity_radps"]))
        displacement = float(np.linalg.norm(position[:2]-np.asarray(specification["position"])[:2]))
        floor_delta = abs(float(position[2])-float(specification["ground_z"]))
        passed = (support_gap is not None and support_gap <= .02 and penetration <= .02
                  and linear <= .03 and angular <= .05 and displacement <= .5
                  and floor_delta <= edge*2)
        records.append({"id": index, "status": "pass" if passed else "fail", "upward_support_gap_m": support_gap,
                        "maximum_sampled_penetration_m": penetration, "surface_samples": len(points),
                        "linear_speed_m_s": linear, "angular_speed_rad_s": angular,
                        "horizontal_displacement_m": displacement, "source_floor_vertical_distance_m": floor_delta,
                        "native_upright_estimate_status": contact.get("status"),
                        "support_scope": specification.get("surface_scope", specification.get("support_source_class", "unspecified"))})
        all_points.extend(points); all_nearest.extend(closest); all_signed.extend(signed); all_probe_ids.extend([index]*len(points))
    output.mkdir(parents=True, exist_ok=True)
    samples = output / "dropped_probe_surface_samples.npz"
    np.savez_compressed(samples, cube_surface_points=np.asarray(all_points), closest_terrain_points=np.asarray(all_nearest),
                        signed_distance_m=np.asarray(all_signed), probe_id=np.asarray(all_probe_ids))
    result = {"status": "pass" if all(r["status"] == "pass" for r in records) else "fail", "probes": records,
              "validator_sha256": sha256_file(Path(__file__)),
              "capture_sha256": sha256_file(native_path), "request_sha256": sha256_file(capture_dir / "capture_request.json"),
              "final_ground_sha256": sha256_file(mesh_path), "samples": samples.name, "sample_sha256": sha256_file(samples),
              "method": "Measured native full body pose and velocities versus independent oriented-face/terrain closest points",
              "limits": ["Surface samples do not prove continuous shape separation between samples",
                         "A supported dropped cube does not establish route traversal or user navigation"]}
    if native_queries is not None:result["native_query_evidence"]=mesh.receipt()
    atomic_json(output / "dropped_probe_validation.json", result)
    return result


def validate_contact_rays(mesh_path: Path, capture_dir: Path, source_surface: Path, output: Path,
                          *, origin=(-1065.38, 0., 688.07)) -> dict:
    record_path = capture_dir / "physx_contact_rays.json"
    record = read_json(record_path)
    sample_path = capture_dir / "physx_contact_rays.npz"
    expected_hash = record.get("sha256", record.get("npz_sha256"))
    if expected_hash != sha256_file(sample_path):
        raise ValueError("Native contact ray bytes do not match their producer manifest")
    if record.get("final_ground_sha256") != sha256_file(mesh_path):
        raise ValueError("Native collision queries refer to different final ground")
    data = np.load(sample_path, allow_pickle=False)
    starts, directions = data["origins"], data["directions"]
    native_points, native_hits = data["position"], data["hit"]
    if starts.shape != directions.shape or starts.ndim != 2 or starts.shape[1] != 3 or not len(starts):
        raise ValueError("Invalid native collision query arrays")
    if (native_points.shape != starts.shape or native_hits.shape != (len(starts),)
            or native_hits.dtype != bool or data["collision_index"].shape != native_hits.shape
            or data["collision_index"].dtype.kind not in "iu"):
        raise ValueError("Invalid native collision hit arrays")
    maximum_distance = float(data["max_distance_m"])
    if not np.isfinite(maximum_distance) or maximum_distance <= 0:
        raise ValueError("Invalid native collision query distance")
    collider_paths, terrain_paths = record["collider_paths"], set(record["terrain_paths"])
    indices = data["collision_index"]
    if not terrain_paths or (indices < 0).any() or (indices >= len(collider_paths)).any():
        raise ValueError("Native collision identifiers are missing or out of bounds")
    terrain_hit = np.array([collider_paths[int(i)] in terrain_paths for i in indices])
    if (native_hits & ~terrain_hit).any():
        raise ValueError("Native collision rays hit an object other than the final supporting terrain")
    if not np.isfinite(starts).all() or not np.isfinite(directions).all():
        raise ValueError("Nonfinite native collision rays")
    if not np.allclose(np.linalg.norm(directions, axis=1), 1., atol=1e-6):
        raise ValueError("Nonunit native collision ray directions")
    mesh = load_ground(mesh_path, output/"native_ground_queries")
    independent = np.full(starts.shape, np.nan)
    normals = np.full(starts.shape, np.nan)
    batch = len(starts) if hasattr(mesh,"queries") else 256
    for start in range(0, len(starts), batch):
        points, rays, triangles = mesh.ray.intersects_location(starts[start:start+batch], directions[start:start+batch], multiple_hits=False)
        independent[start+rays] = points
        normals[start+rays] = mesh.face_normals[triangles]
    valid = native_hits & np.isfinite(independent).all(axis=1) & np.isfinite(native_points).all(axis=1)
    valid &= np.linalg.norm(independent-starts, axis=1) <= maximum_distance+1e-5
    native_range = np.linalg.norm(native_points-starts, axis=1)
    valid &= (native_range <= maximum_distance+1e-5) & (np.sum((native_points-starts)*directions, axis=1) >= 0)
    source = np.load(source_surface, allow_pickle=False)
    source_xyz = np.column_stack((independent[:, 0]+origin[0], independent[:, 2]+origin[1], -independent[:, 1]+origin[2]))
    groups = np.full(len(starts), "side_or_underside", dtype="U24")
    finite = np.flatnonzero(np.isfinite(source_xyz).all(axis=1))
    iz = np.floor((source_xyz[finite, 2]-source["min_xz"][1])/float(source["sample_spacing_m"])).astype(int)
    ix = np.floor((source_xyz[finite, 0]-source["min_xz"][0])/float(source["sample_spacing_m"])).astype(int)
    within = (ix >= 0) & (iz >= 0) & (ix < source["height"].shape[1]) & (iz < source["height"].shape[0])
    ids, iz, ix = finite[within], iz[within], ix[within]
    upward = normals[ids, 2] > .7
    upward &= source["validity"][iz, ix]
    covered = source_xyz[ids, 1] < source["height"][iz, ix]-2.
    groups[ids[upward & ~covered]] = "exterior"
    groups[ids[upward & covered]] = "cave_floor"
    failures = np.flatnonzero(~valid)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "ground_samples.npz"
    # Missing rays remain NaN in this raw array. Never drop misses to improve a quantile.
    np.savez_compressed(path, render_points=independent, collision_points=native_points,
                        sample_group=groups, ray_valid=valid, origins=starts, directions=directions)
    try:
        metrics = compare_ground(independent, native_points)
    except ValueError as exc:
        metrics = {"pass": False, "samples": len(starts), "reason": str(exc)}
    counts = {str(name): int(count) for name, count in zip(*np.unique(groups, return_counts=True))}
    result = {"status": "pass" if metrics["pass"] and not len(failures) else "fail",
              "validator_sha256": sha256_file(Path(__file__)),
              "metrics": metrics, "samples": path.name, "sample_sha256": sha256_file(path),
              "source_strata": counts, "failed_ray_indices": failures.tolist(),
              "native_record_sha256": sha256_file(record_path), "native_samples_sha256": sha256_file(sample_path),
              "final_ground_sha256": sha256_file(mesh_path), "source_surface_sha256": sha256_file(source_surface),
              "method": "Independent trimesh triangle intersections versus native PhysX scene queries",
              "limitations": ["Gravity/contact probes and supported-footprint traversal remain separate requirements",
                             "Surface strata use source top height; semantic cave identity is checked independently"]}
    if hasattr(mesh,"receipt"):
        result["native_query_evidence"]=mesh.receipt();result["method"]="Independent bounded CGAL original-triangle segments versus native PhysX scene queries"
    atomic_json(output / "contact_validation.json", result)
    return result
