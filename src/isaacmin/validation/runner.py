"""Release authority, separate from construction and runtime model tools.

Only this module writes qualification records. Lack of evidence is never a pass.
Numerical validators consume raw measurements; producer success flags alone do
not establish material, temporal, visual or portability qualification.
"""
from __future__ import annotations

from pathlib import Path
import math
import numpy as np

from ..io import atomic_json, hash_object, read_json, sha256_file, utc_now
from ..security import safe_path
from .metrics import compare_ground, compare_depth, compare_seam, critic_calibration, unique_route_distance
from .profile import load_profile, validator_hash


GATE_NAMES = ["provenance", "source_fidelity", "exterior_refinement", "caves_and_overhangs",
              "continuous_surface", "ground_contact", "materials_and_assets", "ecology",
              "sensor_geometry", "visual_appearance", "temporal_quality", "long_range",
              "portable_delivery", "rebuild_repair"]


def gate(number: int, status: str, reason: str, *, metrics=None, evidence=None) -> dict:
    return {"id": f"Q{number:02d}", "name": GATE_NAMES[number], "status": status,
            "reason": reason, "metrics": metrics or {}, "evidence": evidence or [], "exceptions": []}


def validate_build(workspace: Path, build_id: str) -> dict:
    directory = safe_path(workspace / "worlds", build_id)
    profile_record = load_profile(directory / "quality_profile.json")
    manifest = read_json(directory / "build.json")
    report = evaluate(workspace, directory, manifest, profile_record)
    atomic_json(directory / "qualification.json", report)
    atomic_json(workspace / "reports" / (build_id + ".json"), report)
    return report


def evaluate(workspace: Path, directory: Path, manifest: dict, profile_record: dict) -> dict:
    thresholds = profile_record["profile"]["thresholds"]
    gates = [gate(i, "not_run", "Required independent evidence has not been produced") for i in range(14)]
    raw = directory / "evidence"
    scene = directory / "scene/world_physics.usda"
    if not scene.exists():
        scene = directory / "scene/world.usda"
    from ..packaging import ascii_dependency_closure
    closure = ascii_dependency_closure(scene) if scene.exists() else {"status": "fail", "files": []}
    scene_content_hash = hash_object(closure["files"]) if closure["status"] == "pass" else None
    export_record = manifest.get("stages", {}).get("scene", {})
    source_usd = directory / "scene/world.usda"
    mesh_path = directory / "scene/final_ground.obj"
    mesh_hash = sha256_file(mesh_path) if mesh_path.exists() else None
    actual_geometry = (scene_content_hash is not None and export_record.get("status") == "success"
                       and export_record.get("terrain_vertices", 0) > 0
                       and export_record.get("terrain_polygons", 0) > 0
                       and source_usd.exists() and export_record.get("usd_sha256") == sha256_file(source_usd)
                       and mesh_hash is not None and export_record.get("final_ground_sha256") == mesh_hash
                       and export_record.get("native_usd_geometry_verified") is True)
    blockers = list(manifest.get("blockers", []))
    profile_hash = profile_record["sha256"]

    source_path = safe_path(workspace, manifest.get("source_ir", "artifacts/source/region_corrected")) / "independent_decode_comparison.json"
    if source_path.exists():
        source = read_json(source_path)
        # This is source-stage evidence, not post-refinement source-fidelity qualification.
        gates[1] = gate(1, "not_run", "Independent source decode verified; final exported-surface comparison is pending",
                        metrics={"source_decode": source}, evidence=[str(source_path.relative_to(workspace))])

    topology_samples = raw / "topology_samples/source_mesh_validation.json"
    topology_graph = raw / "topology_graph/mesh_connectivity_comparison.json"
    if topology_samples.exists() and topology_graph.exists() and mesh_hash:
        sample_result, graph_result = read_json(topology_samples), read_json(topology_graph)
        from ..volumes import mesh_validation, source_evidence
        from . import axis_parity
        backend_valid = True
        try:
            native_samples = mesh_validation.verify_mesh_backend_evidence(sample_result)
            mesh_validation.verify_mesh_backend_evidence(graph_result)
            if not native_samples:
                backend_valid = sample_result.get("classifier_sha256") == sha256_file(Path(axis_parity.__file__))
        except (ValueError, OSError, KeyError):
            backend_valid = False
        if (sample_result.get("mesh_sha256") != mesh_hash or graph_result.get("mesh_sha256") != mesh_hash
                or sample_result.get("validator_sha256") != sha256_file(Path(mesh_validation.__file__))
                or graph_result.get("validator_sha256") != sha256_file(Path(mesh_validation.__file__))
                or not backend_valid
                or any(result.get("source_payload_evidence", {}).get("verifier_sha256")
                       != sha256_file(Path(source_evidence.__file__)) for result in (sample_result, graph_result))):
            gates[3] = gate(3, "fail", "Cave comparison belongs to different ground geometry or validation code")
        elif sample_result["status"] != "pass" or graph_result["status"] != "pass":
            gates[3] = gate(3, "fail", "Independent source occupancy, openings or connectivity comparison failed",
                            metrics={"samples": sample_result, "connectivity": graph_result})
        else:
            gates[3] = gate(3, "not_run", "Occupancy and connectivity checks pass; feature-specific roof deformation and target views remain required",
                            metrics={"samples": sample_result, "connectivity": graph_result})

    fidelity_path = raw / "source_fidelity/source_fidelity.json"
    if fidelity_path.exists() and mesh_hash:
        fidelity = read_json(fidelity_path)
        from . import source_fidelity
        valid = (fidelity.get("mesh_sha256") == mesh_hash
                 and fidelity.get("quality_profile_sha256") == profile_hash
                 and fidelity.get("validator_sha256") == sha256_file(Path(source_fidelity.__file__)))
        try:
            ir_directory = safe_path(workspace, manifest["source_ir"], must_exist=True)
            valid &= fidelity.get("source_ir_sha256") == sha256_file(ir_directory / "world_ir.json")
            decode_path = ir_directory / "independent_decode_comparison.json"
            decode = read_json(decode_path)
            consumed = {
                "topology_samples_report": topology_samples,
                "topology_graph_report": topology_graph,
                "column_ray_reconstruction": raw / "topology_graph/column_ray_reconstruction.npz",
                "point_classifications": raw / "topology_samples/source_mesh_samples.npz",
                "source_decode_report": decode_path,
                "source_decode_samples": safe_path(ir_directory, decode["samples_file"], must_exist=True),
                "exterior_delta": directory / "data/refinement/exterior_delta.npz",
                "frozen_source_budget": directory / "source_fidelity_profile.json"}
            valid &= set(fidelity.get("evidence_sha256", {})) == set(consumed)
            for name, path in consumed.items():
                valid &= path.is_file() and fidelity.get("evidence_sha256", {}).get(name) == sha256_file(path)
        except (OSError, ValueError, KeyError):
            valid = False
        for entry in fidelity.get("files", []):
            file = safe_path(fidelity_path.parent, entry["path"])
            valid &= file.is_file() and sha256_file(file) == entry["sha256"]
        if not valid:
            gates[1] = gate(1, "fail", "Source-fidelity measurements are stale or changed")
            gates[3] = gate(3, "fail", "Protected cave measurements are stale or changed")
        else:
            geometry_pass = fidelity["source_geometry_status"] == "pass"
            gates[1] = gate(1, "not_run" if geometry_pass else "fail",
                "Source geometry fits frozen budgets; target material and water correspondence remain required" if geometry_pass
                else "Final source geometry exceeds frozen refinement/protection budgets", metrics=fidelity)
            if fidelity["protected_cave_geometry_status"] == "fail":
                gates[3] = gate(3, "fail", "Protected cave interfaces, roof thickness or connectivity failed", metrics=fidelity)
            elif gates[3]["status"] != "fail":
                gates[3] = gate(3, "not_run", "Protected geometry measured; actual portal/interior/overhang view qualification remains required", metrics=fidelity)

    refinement_path = raw / "refinement/refinement_validation.json"
    if refinement_path.exists():
        refinement = read_json(refinement_path)
        verified = True
        for entry in refinement.get("inputs", []):
            path = Path(entry["path"]).resolve()
            verified &= path.is_relative_to(workspace.resolve()) and path.is_file() and sha256_file(path) == entry["sha256"]
        failed = not verified or refinement["status"] != "pass"
        gates[2] = gate(2, "fail" if failed else "not_run",
            "Native refinement or regional drainage consistency failed" if failed
            else "Native erosion, protection and regional runoff verified; exported biome forms and transitions need target inspection",
            metrics=refinement)

    water_path = raw / "water_interfaces/water_interface_comparison.json"
    if water_path.exists():
        from . import water_interfaces
        water = read_json(water_path)
        source_water = directory / "scene/source_water_interfaces.json"
        native_water = export_record.get("source_water_interfaces", {})
        water_valid = (water.get("status") == "pass" and source_water.is_file()
                       and water.get("water_mesh_sha256") == sha256_file(source_water)
                       and water.get("source_snapshot_sha256") == manifest["dependencies"]["source"]
                       and water.get("validator_sha256") == sha256_file(Path(water_interfaces.__file__))
                       and native_water.get("native_usd_geometry_verified") is True
                       and native_water.get("source_sha256") == sha256_file(source_water))
        for entry in water.get("files", []):
            path = safe_path(water_path.parent, entry["path"])
            water_valid &= path.is_file() and sha256_file(path) == entry["sha256"]
        if not water_valid:
            gates[1] = gate(1, "fail", "Independent saved-fluid interfaces or exact native USD water correspondence failed", metrics=water)
        else:
            gates[1]["metrics"]["water_interfaces"] = water

    def bound(name: str, *, required_paths=()) -> dict:
        data = read_json(raw / (name + ".json"))
        if (not actual_geometry or data.get("scene_content_sha256") != scene_content_hash
                or data.get("profile_sha256") != profile_hash):
            raise ValueError("Evidence is stale or refers to another scene/profile")
        paths = {file["path"] for file in data.get("files", [])}
        if not paths or any(data.get(field) not in paths for field in required_paths):
            raise ValueError("Every consumed measurement must be listed and hashed in the evidence manifest")
        for file in data.get("files", []):
            path = safe_path(raw, file["path"], must_exist=True)
            if sha256_file(path) != file["sha256"]:
                raise ValueError("Evidence file hash mismatch")
        return data

    # Only numerical components can be completed without calibrated visual analysis.
    if (raw / "ground_contact.json").exists() and scene.exists():
        try:
            from . import contact as contact_validator
            data = bound("ground_contact", required_paths=("samples", "native_capture", "gravity_measurement", "gravity_samples"))
            ray_comparison = read_json(raw / "contact/contact_validation.json")
            current_contact_validator = sha256_file(Path(contact_validator.__file__))
            ir_surface = safe_path(workspace, manifest["source_ir"]) / "terrain_surface.npz"
            if (ray_comparison.get("validator_sha256") != current_contact_validator
                    or ray_comparison.get("native_record_sha256") != sha256_file(raw / "isaac_static/physx_contact_rays.json")
                    or ray_comparison.get("native_samples_sha256") != sha256_file(raw / "isaac_static/physx_contact_rays.npz")
                    or ray_comparison.get("sample_sha256") != sha256_file(safe_path(raw, data["samples"]))
                    or ray_comparison.get("source_surface_sha256") != sha256_file(ir_surface)):
                raise ValueError("Independent contact rays or their source inputs are stale")
            with np.load(safe_path(raw, data["samples"]), allow_pickle=False) as arrays:
                result = compare_ground(arrays["render_points"], arrays["collision_points"],
                                        p999=thresholds["collision_p999_m"], maximum=thresholds["collision_max_m"])
                if len(np.unique(np.round(arrays["render_points"], 5), axis=0)) < 10000:
                    raise ValueError("Ground comparison requires at least10000 distinct stratified points")
                groups = set(arrays["sample_group"].tolist())
            contact_record = read_json(safe_path(raw, data["native_capture"]))
            contact = contact_record["contacts"]
            gravity = read_json(safe_path(raw, data["gravity_measurement"]))
            if (gravity.get("capture_sha256") != sha256_file(safe_path(raw, data["native_capture"]))
                    or gravity.get("validator_sha256") != current_contact_validator
                    or gravity.get("request_sha256") != sha256_file(raw / "isaac_static/capture_request.json")
                    or gravity.get("final_ground_sha256") != mesh_hash
                    or gravity.get("sample_sha256") != sha256_file(safe_path(raw, data["gravity_samples"]))):
                raise ValueError("Independent gravity/body-pose evidence is stale")
            if data.get("final_ground_sha256") != mesh_hash or data.get("independent_verification_status") != "pass":
                raise ValueError("Ground measurements are stale or independent ray verification failed")
            if "exterior" not in groups:
                raise ValueError("Ground samples exclude exterior support")
            source_topology = safe_path(workspace, manifest["source_ir"]) / "topology/source_topology_graph.json"
            if read_json(source_topology).get("component_count", 0) > 0 and "cave_floor" not in groups:
                raise ValueError("Ground samples exclude present caves and stacked levels")
            if (not contact or gravity.get("status") != "pass" or len(gravity["probes"]) != len(contact)
                    or any(p["status"] != "pass" for p in gravity["probes"])):
                result["pass"] = False
            from . import traversal as traversal_validator
            routes = data.get("supported_footprint_traversals", [])
            measured_scopes, measured_routes = set(), []
            bound_files = {entry["path"] for entry in data["files"]}
            for route in routes:
                for key in ("measurement", "samples", "native_record", "native_samples", "request"):
                    if route[key] not in bound_files:
                        raise ValueError("Every consumed traversal record and sample must be hash-bound")
                measured = read_json(safe_path(raw, route["measurement"]))
                if (measured.get("final_ground_sha256") != mesh_hash
                        or measured.get("sample_sha256") != sha256_file(safe_path(raw, route["samples"]))
                        or measured.get("native_record_sha256") != sha256_file(safe_path(raw, route["native_record"]))
                        or measured.get("native_samples_sha256") != sha256_file(safe_path(raw, route["native_samples"]))
                        or measured.get("request_sha256") != sha256_file(safe_path(raw, route["request"]))
                        or measured.get("validator_sha256") != sha256_file(Path(traversal_validator.__file__))):
                    raise ValueError("Independent traversal evidence is stale or changed")
                with np.load(safe_path(raw, route["samples"]), allow_pickle=False) as arrays:
                    measured_rays = compare_ground(arrays["render_points"], arrays["collision_points"],
                        p999=thresholds["collision_p999_m"], maximum=thresholds["collision_max_m"])
                    frames = len(arrays["simulation_time_s"])
                okay = (measured["status"] == "pass" and measured_rays["pass"]
                        and measured["completed_requested_route"] and measured["frame_count"] == frames
                        and len(measured["frames"]) == frames and frames > 1
                        and all(f["status"] == "pass" for f in measured["frames"]))
                if not okay:
                    result["pass"] = False
                else:
                    measured_scopes.add(measured["surface_scope"])
                measured_routes.append({"id": route["id"], "status": "pass" if okay else "fail",
                                        "surface_scope": measured["surface_scope"], "frames": frames,
                                        "travelled_distance_m": measured["travelled_distance_m"]})
            required_scopes = {"exterior"}
            if read_json(source_topology).get("component_count", 0) > 0:
                required_scopes.add("cave_floor")
            missing_scopes = sorted(required_scopes-measured_scopes)
            missing_contacts=data.get('missing_cave_contact_probes',[])
            missing_exterior_contacts=data.get('missing_exterior_contact_probes',[])
            result.update(supported_footprint_routes=measured_routes, missing_traversal_scopes=missing_scopes,
                          missing_cave_contact_probes=missing_contacts,
                          missing_exterior_contact_probes=missing_exterior_contacts,
                          navigation_stack="not_run_not_supplied")
            status = "fail" if not result["pass"] else "not_run" if missing_scopes or missing_contacts or missing_exterior_contacts else "pass"
            reason = ("Independent final-ground distances, settled body poses and recorded force-driven footprint routes pass"
                      if status == "pass" else "Independent ground/contact or physical traversal failed" if status == "fail"
                      else "Ground and gravity measurements pass; required contact locations or footprint traversal scopes are missing")
            gates[5] = gate(5, status, reason,
                            metrics=result, evidence=["evidence/ground_contact.json"])
        except (ValueError, OSError, KeyError) as exc:
            gates[5] = gate(5, "fail", str(exc))
    if (raw / "depth_comparison.json").exists() and scene.exists():
        try:
            from . import sensors
            data = bound("depth_comparison", required_paths=("samples",))
            measurement = read_json(raw / "depth/depth_validation.json")
            capture = raw / "isaac_static/capture_result.json"
            if (measurement.get("validator_sha256") != sha256_file(Path(sensors.__file__))
                    or measurement.get("final_ground_sha256") != mesh_hash
                    or measurement.get("capture_sha256") != sha256_file(capture)
                    or measurement.get("sample_sha256") != sha256_file(safe_path(raw, data["samples"]))):
                raise ValueError("Independent depth validation or its native source is stale")
            bound_paths = {entry["path"] for entry in data["files"]}
            for frame in read_json(capture)["frames"]:
                for role in ("rgb", "depth", "instance_segmentation"):
                    file = safe_path(capture.parent, frame[role], must_exist=True)
                    if (file.relative_to(raw).as_posix() not in bound_paths
                            or sha256_file(file) != frame[role+"_sha256"]):
                        raise ValueError("Depth gate excludes or changed a consumed native frame")
            with np.load(safe_path(raw, data["samples"]), allow_pickle=False) as arrays:
                result = compare_depth(arrays["render_depth"], arrays["geometry_axial_z"], arrays["opaque_valid"],
                                       semantics=data["depth_semantics"], ray_cosines=arrays.get("ray_cosines"),
                                       min_samples=thresholds["depth_samples_min"])
            if not data["same_camera_pose_and_timestamp"]:
                result["pass"] = False
            if data.get("independent_verification_status") != "pass":
                result["pass"] = False
            gates[8] = gate(8, "pass" if result["pass"] else "fail", "Metric opaque-surface depth versus independent geometry",
                            metrics=result, evidence=["evidence/depth_comparison.json"])
        except (ValueError, OSError, KeyError) as exc:
            gates[8] = gate(8, "fail", str(exc))
    if (raw / "long_range.json").exists() and scene.exists():
        try:
            data = bound("long_range", required_paths=("trajectory",))
            points = np.load(safe_path(raw, data["trajectory"]), allow_pickle=False)
            result = unique_route_distance(points)
            target = thresholds["unique_route_target_m"]
            if result["unique_distance_m"] < target:
                gates[11] = gate(11, "fail", "Captured unique route is shorter than the supported source target", metrics=result)
            else:
                gates[11] = gate(11, "not_run", "Distance verified; full cadence/return/revisit/continuous-recording checks pending", metrics=result)
        except (ValueError, OSError, KeyError) as exc:
            gates[11] = gate(11, "fail", str(exc))

    surface_path = raw / "continuous_surface/surface/continuous_surface.json"
    if surface_path.is_file() and mesh_hash:
        from . import continuous_surface, surface_overlap
        surface = read_json(surface_path)
        valid = (surface.get("validator_sha256") == sha256_file(Path(continuous_surface.__file__))
                 and surface.get("overlap_classifier_sha256") == sha256_file(Path(surface_overlap.__file__))
                 and surface.get("quality_profile_sha256") == sha256_file(directory / "quality_profile.json")
                 and any(entry.get("sha256") == mesh_hash for entry in surface.get("input_files", [])))
        for entry in surface.get("files", []):
            payload = safe_path(surface_path.parent, entry["path"])
            valid &= payload.is_file() and sha256_file(payload) == entry["sha256"]
        if not valid or surface.get("status") == "fail":
            gates[4] = gate(4, "fail", "Exact final triangle integrity/ownership failed or evidence changed", metrics=surface)
        else:
            gates[4] = gate(4, "not_run", "Triangle integrity measured; actual join crossings and normal/material continuity remain required", metrics=surface)

    ecology_path = raw / "ecology/ecology_report.json"
    if ecology_path.is_file():
        ecology = read_json(ecology_path)
        try:
            from ..assets.usd_provenance import verify_scene_closure
            from ..assets.network import ServiceError
            verify_scene_closure(source_usd, ecology["native_scene_dependencies"]["manifest"]["sha256"])
            valid = (ecology["scene_sha256"] == sha256_file(source_usd)
                     and ecology["ecology_manifest_sha256"] == sha256_file(directory / "data/ecology/ecology_manifest.json")
                     and ecology["source_ir_manifest_sha256"] == manifest.get("dependencies", {}).get("world_ir"))
            if not valid or ecology.get("hard_rule_violations") or ecology.get("status") == "fail":
                gates[7] = gate(7, "fail", "Exported-instance ecology violated source rules or its input identity changed", metrics=ecology)
            else:
                gates[7] = gate(7, "not_run", "Native instance rules and distributions measured; asset variety and target composition still need qualification", metrics=ecology)
        except (ValueError, OSError, KeyError, ServiceError) as exc:
            gates[7] = gate(7, "fail", str(exc))

    structure_path=raw/'usd_structure/usd_inspection.json'
    if structure_path.is_file():
        try:
            from ..assets import usd_inspection,collision_scope
            structure=read_json(structure_path)
            native_closure=directory/'scene/native_physics_dependency_closure.json'
            runtime=directory/'scene/runtime_dependencies.json'
            valid=(structure.get('scene_sha256')==sha256_file(scene)
                and structure.get('validator_sha256')==sha256_file(Path(usd_inspection.__file__))
                and structure.get('collision_scope_validator_sha256')==sha256_file(Path(collision_scope.__file__))
                and native_closure.is_file() and structure.get('dependency_closure_sha256')==sha256_file(native_closure)
                and runtime.is_file() and structure.get('runtime_dependency_sha256')==sha256_file(runtime))
            failed=not valid or structure.get('status')!='pass' or bool(structure.get('errors'))
            gates[6]=gate(6,'fail' if failed else 'not_run',
                'Native USD material/collider structure failed or its inputs changed' if failed else
                'Native USD bindings, texture files and explicit collider ownership verified; target material appearance remains unqualified',
                metrics=structure,evidence=[str(structure_path.relative_to(workspace))])
        except (ValueError,OSError,KeyError) as exc:
            gates[6]=gate(6,'fail',str(exc))

    review = manifest.get("stages", {}).get("visual_review", {})
    if review.get("status") == "blocked_calibration":
        gates[9] = gate(9, "blocked", review["reason"], evidence=[review["frozen_manifest"]])
    temporal_path = raw / "temporal/temporal_measurements.json"
    if temporal_path.is_file():
        try:
            from . import temporal as temporal_validator
            temporal = read_json(temporal_path)
            capture_path = raw / "isaac_motion/capture_result.json"
            trajectory_path = safe_path(temporal_path.parent, temporal["trajectory"])
            valid = (capture_path.is_file() and sha256_file(capture_path) == temporal.get("capture_sha256")
                     and trajectory_path.is_file() and sha256_file(trajectory_path) == temporal.get("trajectory_sha256")
                     and temporal.get("validator_sha256") == sha256_file(Path(temporal_validator.__file__)))
            if not temporal.get("inputs"):
                valid = False
            for entry in temporal.get("inputs", []):
                path = safe_path(capture_path.parent, entry["path"], must_exist=True)
                valid &= sha256_file(path) == entry["sha256"]
            if not valid or not temporal.get("continuous_frame_numbering") or temporal.get("missing_ground_frames"):
                gates[10] = gate(10, "fail", "Continuous native motion has missing ground, missing frames or changed input bytes", metrics=temporal)
            else:
                gates[10] = gate(10, "not_run", "Actual motion and ground optical-flow residuals measured; flagged regions, foliage, shadow and residency checks remain", metrics=temporal)
        except (ValueError, OSError, KeyError) as exc:
            gates[10] = gate(10, "fail", "Temporal measurement cannot be verified: " + str(exc))

    portable_path = raw / "portable_delivery/reproduction_comparison.json"
    portable_pointer = raw / "portable_delivery_pointer.json"
    if portable_pointer.is_file():
        try:
            pointer = read_json(portable_pointer)
            portable_path = safe_path(raw, pointer["report"], must_exist=True)
            if sha256_file(portable_path) != pointer["sha256"]:
                raise ValueError("Portable report differs from its current measured-attempt pointer")
        except (ValueError, OSError, KeyError) as exc:
            gates[12] = gate(12, "fail", str(exc))
            portable_path = raw / "__invalid_portable_pointer__"
    if portable_path.is_file() and scene.exists():
        try:
            from ..packaging import verify_package_files
            portable = read_json(portable_path)
            package = Path(portable["package"]).resolve()
            if not package.is_relative_to(workspace.resolve()):
                raise ValueError("Portable evidence names a package outside this workspace")
            package_manifest = read_json(package / "package.json")
            valid = (portable.get("validator_sha256") == sha256_file(workspace / "isaac_scripts/compare_reproduction.py")
                     and portable.get("scene_sha256") == sha256_file(scene)
                     and portable.get("package_manifest_sha256") == sha256_file(package / "package.json")
                     and portable.get("reference_sha256") == sha256_file(package / "reproduction/reference.json")
                     and verify_package_files(package, package_manifest["files"]))
            for entry in portable.get("files", []):
                path = safe_path(portable_path.parent, entry["path"], must_exist=True)
                valid &= path.stat().st_size == entry["bytes"] and sha256_file(path) == entry["sha256"]
            if not valid or portable.get("errors") or portable.get("status") == "fail":
                gates[12] = gate(12, "fail", "Fresh package reproduction failed or its exact evidence changed", metrics=portable)
            elif portable.get("status") == "pass" and not portable.get("gaps"):
                gates[12] = gate(12, "pass", "Bundled saved scene reproduces representative/free-view RGB, depth and native physical contacts in fresh pinned Isaac processes", metrics=portable)
            else:
                gates[12] = gate(12, "not_run", "Portable checks are incomplete", metrics=portable)
        except (ValueError, OSError, KeyError) as exc:
            gates[12] = gate(12, "fail", "Portable evidence cannot be verified: " + str(exc))

    if not actual_geometry:
        gates[0] = gate(0, "blocked", "No dependency-closed scene exists yet")
        for n in range(2, 14):
            gates[n] = gate(n, "blocked", "Requires the real exported and captured Isaac world")
    else:
        gates[0] = gate(0, "not_run", "Scene exists; complete licensing/dependency/provenance closure has not passed")
        provenance_path = raw / "provenance/provenance.json"
        if provenance_path.is_file():
            from . import provenance
            audit = read_json(provenance_path)
            valid = (audit.get("validator_sha256") == sha256_file(Path(provenance.__file__))
                     and audit.get("profile_sha256") == profile_hash and audit.get("build_id") == manifest["build_id"]
                     and hash_object(audit.get("native_closure", {}).get("files", [])) == scene_content_hash)
            for entry in audit.get("files", []):
                path = Path(entry["path"])
                valid &= path.is_file() and path.stat().st_size == entry["bytes"] and sha256_file(path) == entry["sha256"]
            if not valid or audit.get("errors") or audit.get("status") == "fail":
                gates[0] = gate(0, "fail", "Native provenance audit failed or its exact input bytes changed", metrics=audit)
            elif audit.get("gaps") or audit.get("status") != "pass":
                gates[0] = gate(0, "not_run", "File/licence checks completed but original provenance gaps remain", metrics=audit)
            else:
                gates[0] = gate(0, "pass", "Source, acquired licences, construction inputs and closed native payload verified", metrics=audit)
    technical = (0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13)
    technical_pass = all(gates[i]["status"] == "pass" for i in technical)
    all_pass = all(g["status"] == "pass" for g in gates)
    stage = "autonomously_qualified" if all_pass else "quality_candidate" if technical_pass else "geometry_world" if actual_geometry else None
    return {"schema_version": "1.0", "build_id": manifest["build_id"], "run_id": manifest["run_id"],
            "created_at_utc": utc_now(), "status": "pass" if all_pass else "incomplete",
            "stage": stage, "profile_sha256": profile_hash, "validator_sha256": validator_hash(),
            "scene_sha256": sha256_file(scene) if scene.exists() else None,
            "scene_content_sha256": scene_content_hash,
            "gates": gates, "blockers": blockers, "source_scope": manifest.get("scope"),
            "human_review": "not_requested", "navigation_stack": "not_run_not_supplied",
            "report": str(directory / "qualification.json"),
            "limitations": ["No visual quality or runtime performance claim follows from completed code or fixture tests",
                            "A geometry_world label does not mean an Isaac-qualified or realistic world"]}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--build", required=True)
    args = parser.parse_args()
    result = validate_build(args.workspace, args.build)
    print(result["status"])
    return 0 if result["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
