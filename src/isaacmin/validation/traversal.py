"""Independently measure support along a native dynamic footprint trajectory."""
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..io import atomic_json, hash_object, read_json, sha256_file
from ..security import safe_path
from .capture_plan import load_ground
from .metrics import compare_ground


def validate_traversal(mesh_path: Path, traversal_dir: Path, output: Path) -> dict:
    record_path = traversal_dir / "traversal_result.json"
    request_path = traversal_dir / "traversal_request.json"
    record, request = read_json(record_path), read_json(request_path)
    samples_path = traversal_dir / "traversal_samples.jsonl"
    if (record.get("request_sha256") != sha256_file(request_path)
            or record.get("samples_sha256") != sha256_file(samples_path)
            or record.get("final_ground_sha256") != sha256_file(mesh_path)
            or request.get("final_ground_sha256") != sha256_file(mesh_path)):
        raise ValueError("Traversal input or raw measurement bytes changed")
    if request.get("scene_content_sha256") != hash_object(request.get("scene_dependency_files", [])):
        raise ValueError("Traversal scene dependency identity is missing or invalid")
    scene_root = Path(request["scene"]).parent
    for entry in request["scene_dependency_files"]:
        if sha256_file(safe_path(scene_root, entry["path"], must_exist=True)) != entry["sha256"]:
            raise ValueError("Traversal scene dependency bytes changed")
    samples = [json.loads(line) for line in samples_path.read_text().splitlines() if line.strip()]
    if len(samples) != record.get("sample_count") or len(samples) < 2:
        raise ValueError("Missing native trajectory samples")
    dimensions = np.asarray(record["footprint_xyz_m"], float)
    if dimensions.shape != (3,) or not np.isfinite(dimensions).all() or np.any(dimensions <= 0) or np.any(dimensions > 1):
        raise ValueError("Invalid native footprint dimensions")
    if not np.array_equal(dimensions, request["footprint_xyz_m"]):
        raise ValueError("Native probe dimensions differ from the requested footprint")
    local = np.array([[x, y, -dimensions[2]/2] for x in [-dimensions[0]/2, 0, dimensions[0]/2]
                      for y in [-dimensions[1]/2, 0, dimensions[1]/2]])
    mesh = load_ground(mesh_path, output/"native_ground_queries")
    positions, times, support_points, independent_points, native_points = [], [], [], [], []
    frame_results = []
    terrain_paths = set(record.get("terrain_paths", []))
    if not terrain_paths:
        raise ValueError("Native traversal did not identify the authoritative terrain colliders")
    jobs=[]
    for index, sample in enumerate(samples):
        position, q = np.asarray(sample["position"], float), np.asarray(sample["orientation_wxyz"], float)
        if position.shape != (3,) or q.shape != (4,) or not np.isfinite(position).all() or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1) > 1e-5:
            raise ValueError("Invalid measured body transform")
        corners = Rotation.from_quat(np.r_[q[1:], q[0]]).apply(local)+position
        native_corners = np.asarray(sample["footprint_bottom_samples_world"], float)
        if native_corners.shape != (9, 3) or not np.allclose(native_corners, corners, atol=1e-6, rtol=0):
            raise ValueError("Recorded footprint points disagree with the measured rigid body pose")
        origins = corners+[0., 0., .25]
        jobs.append((index,sample,position,corners,origins))
    if hasattr(mesh,'queries'):
        locations,normal_values,triangle_values=mesh.queries.first_hits(np.concatenate([job[4] for job in jobs]),np.tile([0.,0.,-1.],(len(jobs)*9,1)))
    for index,sample,position,corners,origins in jobs:
        if hasattr(mesh,'queries'):
            selected=slice(index*9,(index+1)*9);independent=locations[selected];normals=normal_values[selected]
        else:
            points,ray_ids,triangles=mesh.ray.intersects_location(origins,np.tile([0.,0.,-1.],(9,1)),multiple_hits=False)
            independent=np.full((9,3),np.nan);normals=np.full((9,3),np.nan);independent[ray_ids]=points;normals[ray_ids]=mesh.face_normals[triangles]
        hits = sample["terrain_support_rays"]
        if len(hits) != 9:
            raise ValueError("Missing footprint support rays")
        native = np.full((9, 3), np.nan)
        for i, hit in enumerate(hits):
            if hit["hit"]:
                if hit.get("collision") not in terrain_paths:
                    raise ValueError("Traversal support was provided by an unidentified collider")
                native[i] = hit["position"]
        valid = np.isfinite(independent).all(axis=1) & np.isfinite(native).all(axis=1)
        valid &= (np.linalg.norm(independent-origins, axis=1) <= 2.+1e-6) & (normals[:, 2] > 0)
        clearance = corners[:, 2]-independent[:, 2]
        gap = float(np.min(clearance)) if valid.all() else None
        penetration = float(np.max(np.maximum(-clearance, 0))) if valid.all() else None
        disagreement = float(np.max(np.linalg.norm(independent-native, axis=1))) if valid.all() else None
        # The same strict20mm bound is used for contact separation and penetration.
        # Every recorded footprint ray remains in the result, including misses.
        supported = (valid.all() and abs(gap) <= .02 and penetration <= .02 and disagreement <= .02)
        force = np.asarray(sample["measured_net_contact_force_n"], float)
        velocity = np.asarray(sample["linear_velocity_mps"], float)
        applied = np.asarray(sample["applied_force_n"], float)
        if any(a.shape != (3,) or not np.isfinite(a).all() for a in (force, velocity, applied)):
            raise ValueError("Nonfinite native force or velocity")
        if abs(applied[2]) > 1e-9 or np.any(abs(applied[:2]) > 12.+1e-6):
            raise ValueError("Probe drive exceeded its bounded horizontal-force contract")
        frame_results.append({"index": index, "status": "pass" if supported else "fail",
                              "support_gap_m": gap, "maximum_sampled_penetration_m": penetration,
                              "maximum_ray_disagreement_m": disagreement, "contact_force_z_n": float(force[2])})
        positions.append(position); times.append(float(sample["simulation_time_s"]))
        support_points.extend(corners); independent_points.extend(independent); native_points.extend(native)
    times, positions = np.asarray(times), np.asarray(positions)
    dt = np.diff(times)
    if not np.isfinite(times).all() or not np.allclose(dt, 1/30, atol=1e-5, rtol=0):
        raise ValueError("Traversal samples do not preserve the declared30Hz simulation cadence")
    try:
        metrics = compare_ground(np.asarray(independent_points), np.asarray(native_points))
    except ValueError as exc:
        metrics = {"pass": False, "samples": len(independent_points), "reason": str(exc)}
    requested = len(request["route_ground_xyz"])
    completed = record.get("completed_waypoint_indices") == list(range(requested))
    passed = (record.get("status") == "pass" and completed and metrics["pass"]
              and all(f["status"] == "pass" for f in frame_results)
              and record.get("height_teleports_after_initialization") == 0
              and any(f["contact_force_z_n"] > .01 for f in frame_results))
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "traversal_comparison.npz"
    np.savez_compressed(raw, simulation_time_s=times, positions=positions,
                        footprint_bottom_points=np.asarray(support_points),
                        render_points=np.asarray(independent_points), collision_points=np.asarray(native_points))
    result = {"status": "pass" if passed else "fail", "surface_scope": request["surface_scope"],
              "frame_count": len(samples), "frames": frame_results, "ray_metrics": metrics,
              "completed_requested_route": completed, "requested_route_points": requested,
              "travelled_distance_m": float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()),
              "maximum_observed_speed_m_s": float(np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1)/dt)),
              "native_record_sha256": sha256_file(record_path), "native_samples_sha256": sha256_file(samples_path),
              "request_sha256": sha256_file(request_path), "final_ground_sha256": sha256_file(mesh_path),
              "scene_content_sha256": request["scene_content_sha256"],
              "samples": raw.name, "sample_sha256": sha256_file(raw),
              "validator_sha256": sha256_file(Path(__file__)),
              "method": "Measured rigid-body footprint and native terrain rays versus independent final triangle queries at every30Hz sample",
              "navigation_stack": "not_run_not_supplied",
              "limitations": ["Nine footprint rays do not establish continuous shape clearance between sample positions",
                              "Only the recorded bounded support route was physically traversed"]}
    if hasattr(mesh,"receipt"):result["native_query_evidence"]=mesh.receipt()
    atomic_json(output / "traversal_validation.json", result)
    return result
