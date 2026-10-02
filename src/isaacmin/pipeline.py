"""Deterministic production jobs, with durable checkpoints and fail-closed gates."""
from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np

from .contracts import CoordinateFrame, seal_directory, verify_artifact
from .contracts.artifacts import promote
from .io import atomic_json, code_hash, hash_object, read_json, sha256_file, utc_now
from .jobs import Journal
from .project import resolve_project
from .security import safe_path
from .validation.profile import freeze_profile, validator_hash


def bootstrap(workspace: Path, world: Path | None) -> dict:
    from .bootstrap import inspect_host
    from .source import inspect_source, snapshot_world
    from .adapters.nim import probe_nim
    from .adapters.workers import probe_workers
    from .assets.providers import bootstrap_assets
    project = resolve_project(workspace, world)
    host = inspect_host(workspace, persist=True)
    snapshot = snapshot_world(Path(project["source_world"]), workspace / "work/snapshots")
    atomic_json(workspace / "artifacts/source/source_snapshot.json", snapshot)
    inventory = inspect_source(Path(snapshot["snapshot_path"]), workspace / "artifacts/source")
    capabilities = workspace / "state/nim_capabilities.json"
    nim = read_json(capabilities) if capabilities.exists() else probe_nim(workspace)
    assets = bootstrap_assets(workspace)
    workers = probe_workers(workspace)
    report = {"status": "pass" if workers["cross_runtime_qualified"] else "waiting_dependency",
              "host": host, "snapshot_sha256": snapshot["save_sha256"],
              "source_data_version": inventory["data_version"], "nim": nim,
              "assets": assets, "workers": workers,
              "reason": "Native compatibility must pass before a production world build"}
    atomic_json(workspace / "artifacts/bootstrap/bootstrap.json", report)
    return report


def _source(workspace: Path, project: dict) -> tuple[dict, dict]:
    from .source import snapshot_world, inspect_source
    from .source.provenance import reader_producer_identity
    snapshot = snapshot_world(Path(project["source_world"]), workspace / "work/snapshots")
    previous = workspace / "artifacts/source/source_snapshot.json"
    atomic_json(previous, snapshot)
    inventory_directory = workspace / "artifacts/source" / snapshot["save_sha256"] / "inventory"
    inventory_path = inventory_directory / "source_inventory.json"
    producer = reader_producer_identity("inventory")
    if inventory_path.exists() and read_json(inventory_path).get("producer") != producer:
        inventory_directory = inventory_directory.with_name("inventory_"+hash_object(producer)[:16])
        inventory_path = inventory_directory / "source_inventory.json"
    inventory = read_json(inventory_path) if inventory_path.exists() else inspect_source(
        Path(snapshot["snapshot_path"]), inventory_directory, snapshot_sha256=snapshot["save_sha256"])
    if inventory.get("source_snapshot_sha256") != snapshot["save_sha256"] or inventory.get("producer") != producer:
        raise ValueError("Inventory source or semantic interpretation is stale")
    return snapshot, inventory


def _job(workspace: Path, run_id: str, name: str, dependencies: dict, parameters: dict,
         output: Path, profile: dict, execute) -> dict:
    journal_path = workspace / "state/journal.sqlite"
    journal = Journal(journal_path)
    fingerprint = hash_object({"name": name, "dependencies": dependencies, "parameters": parameters})
    operation_id = f"{run_id}:{name}:{fingerprint}"
    record = journal.submit(run_id, operation_id, name, parameters, dependencies)
    def verify_expected():
        artifact = verify_artifact(output)
        if (artifact["input_hashes"] != dependencies or artifact["parameter_hash"] != hash_object(parameters)
                or artifact["quality_profile_hash"] != profile["sha256"]
                or artifact["toolchain_hash"] != dependencies["tools"]):
            raise ValueError("Artifact provenance differs from the requested job")
        return artifact
    if record["state"] == "succeeded":
        try:
            verify_expected()
            return record["result"]
        finally:
            journal.close()
    owner = journal.claim(record["id"])
    if owner is None:
        journal.close()
        raise RuntimeError("Stage is already running with an active lease")
    # A recovered worker never writes another owner's staging directory.
    staging = output.with_name("." + output.name + ".staging." + owner)
    staging.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    heartbeat_errors = []

    def heartbeat():
        j = Journal(journal_path)
        try:
            while not stop.wait(20):
                try:
                    j.heartbeat(record["id"], owner, {"staging": str(staging)})
                except Exception as exc:
                    heartbeat_errors.append(exc)
                    stop.set()
                    break
        finally:
            j.close()

    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        if output.exists():
            verify_expected()
            result = read_json(output / "stage_result.json")
            result["artifact"] = str(output / "artifact.json")
            journal.publish(record["id"], owner, result, lambda: None)
            return result
        result = execute(staging)
        if heartbeat_errors:
            raise RuntimeError("Lease heartbeat failed; this candidate cannot publish")
        if result.get("status") not in {"success", "pass", "validated", "source_graph_computed"}:
            raise RuntimeError("Stage did not produce a validated result: " + str(result.get("reason", result.get("status"))))
        # Store future artifact locations before sealing; raw process logs retain the
        # historical staging cwd. A crash-recovered adoption then has usable paths.
        result = json.loads(json.dumps(result).replace(str(staging), str(output)))
        atomic_json(staging / "stage_result.json", result)
        seal_directory(staging, kind=name, build_id=run_id, scope=parameters, inputs=dependencies,
                       parameters=parameters, toolchain_hash=dependencies["tools"],
                       profile_hash=profile["sha256"], coordinate_frame=profile["profile"]["context"]["coordinates"])
        result["artifact"] = str(output / "artifact.json")
        journal.publish(record["id"], owner, result, lambda: promote(staging, output))
        return result
    except BaseException as exc:
        from .jobs import LeaseLost
        try:
            journal.finish(record["id"], owner, state="failed", error={"type": type(exc).__name__, "reason": str(exc)})
        except LeaseLost:
            pass  # Stale worker has no authority to change the new owner's status.
        raise
    finally:
        stop.set()
        thread.join(timeout=5)
        journal.close()


def build(workspace: Path, world: Path | None, *, integration_only: bool = False,
          resume_run_id: str | None = None) -> dict:
    import fcntl
    workspace = workspace.resolve()
    (workspace / "state").mkdir(parents=True, exist_ok=True)
    # The entire candidate has one publisher, including native tools that author
    # several files. OS ownership is released on crashes; individual durable
    # refinement/volume jobs retain their own fenced SQLite leases.
    with safe_path(workspace, "state/build.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"status": "waiting_resource", "reason": "Another build or resume owns this workspace's heavy-worker lock"}
        try:
            return _build(workspace, world, integration_only=integration_only, resume_run_id=resume_run_id)
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _build(workspace: Path, world: Path | None, *, integration_only: bool = False,
           resume_run_id: str | None = None) -> dict:
    from .adapters.workers import probe_workers, refine_heightfield, reconstruct_occupancy
    from .source import extract_world_ir
    from .terrain.baseline import reconstruct_baseline
    from .validation.runner import validate_build

    project = resolve_project(workspace, world)
    snapshot, inventory = _source(workspace, project)
    from .assets.candidates import candidate_lighting
    lighting_recipe = candidate_lighting(workspace)
    from .assembly.material_recipe import selected_material_recipe, validate_material_recipe
    material_recipe = selected_material_recipe(workspace)
    from .assembly.foliage_recipe import selected_foliage_recipe
    foliage_recipe = selected_foliage_recipe(workspace)
    from .adapters.render_recipe import selected_capture_recipe
    capture_recipe = selected_capture_recipe(workspace)
    workers = probe_workers(workspace)
    lock_path = workspace / "artifacts/bootstrap/dependency-lock.json"
    tools_hash = sha256_file(lock_path) if lock_path.exists() else hash_object(workers)
    # Original provider shader groups can otherwise export without any surface
    # shader. Prepare and verify their native derivatives before freezing this
    # build's asset identity, without altering source masters or geometry.
    from .assets.target_materials import prepare_target_assets
    prepare_target_assets(workspace)
    assets_path = workspace / "state/asset_catalogue.json"
    from .source.provenance import reader_producer_identity
    source_producer = reader_producer_identity("world_ir")
    initial = workspace / "artifacts/source" / snapshot["save_sha256"] / "region"
    if (initial / "world_ir.json").exists() and read_json(initial / "world_ir.json").get("producer") != source_producer:
        initial = initial.with_name("region_"+hash_object(source_producer)[:16])
    if not (initial / "world_ir.json").exists():
        extract_world_ir(Path(snapshot["snapshot_path"]), initial,
                         center=tuple(project["centre_minecraft_xz"]), extent=128, snapshot_sha256=snapshot["save_sha256"])
    ir = read_json(initial / "world_ir.json")
    if ir.get("source_snapshot_sha256") != snapshot["save_sha256"] or ir.get("producer") != source_producer:
        raise ValueError("WorldIR source or semantic interpretation is stale")
    if ir["scope"]["requested_center_xz"] != project["centre_minecraft_xz"]:
        raise ValueError("WorldIR center differs from the requested project center")
    if ir["source_level_dat_sha256"] != sha256_file(Path(snapshot["snapshot_path"]) / "level.dat"):
        raise ValueError("WorldIR source metadata hash does not match snapshot")
    for entry in ir["files"]:
        if sha256_file(safe_path(initial, entry["path"], must_exist=True)) != entry["sha256"]:
            raise ValueError("WorldIR content was changed: " + entry["path"])
    from .volumes.source_support import derive_source_support
    support_directory = workspace / "artifacts/source" / snapshot["save_sha256"] / "support"
    support_identity = {"source_ir": sha256_file(initial / "world_ir.json"),
                        "producer": sha256_file(workspace / "src/isaacmin/volumes/source_support.py"),
                        "semantics": sha256_file(workspace / "src/isaacmin/source/semantics.py")}
    if (support_directory / "support_manifest.json").exists():
        previous_support = read_json(support_directory / "support_manifest.json")
        if (previous_support["source_ir_sha256"] != support_identity["source_ir"]
                or previous_support["producer_sha256"] != support_identity["producer"]
                or previous_support["semantic_policy_sha256"] != support_identity["semantics"]):
            support_directory = support_directory.with_name("support_"+hash_object(support_identity)[:16])
    if not (support_directory / "support_manifest.json").exists():
        derive_source_support(initial, support_directory)
    support_record = read_json(support_directory / "support_manifest.json")
    if (support_record["source_ir_sha256"] != support_identity["source_ir"]
            or support_record["producer_sha256"] != support_identity["producer"]
            or support_record["semantic_policy_sha256"] != support_identity["semantics"]):
        raise ValueError("Derived support mask provenance is stale")
    for entry in support_record["files"]:
        if sha256_file(safe_path(support_directory, entry["path"], must_exist=True)) != entry["sha256"]:
            raise ValueError("Derived source support changed")
    support_mask_path = support_directory / "support_mask.npz"
    from .volumes.topology import source_topology
    topology_record = initial / "topology/source_topology_graph.json"
    if not topology_record.exists():
        source_topology(initial)
    graph = read_json(topology_record)
    if graph["source_ir_sha256"] != sha256_file(initial / "world_ir.json"):
        raise ValueError("Source topology belongs to another WorldIR")
    for entry in graph["files"]:
        if sha256_file(safe_path(topology_record.parent, entry["path"], must_exist=True)) != entry["sha256"]:
            raise ValueError("Source topology content changed: " + entry["path"])
    macro_dir = workspace / "artifacts/source/macro_region"
    macro_record = macro_dir / "macro_surface.json"
    def macro_matches(path):
        if not path.exists():
            return False
        record = read_json(path)
        from .source.anvil import dimensions
        from .source.macro import macro_producer_identity
        import math
        source_root = Path(snapshot["snapshot_path"])
        if (record.get("source_snapshot_sha256") != snapshot["save_sha256"]
                or record.get("producer") != macro_producer_identity()):
            return False
        dimension = dimensions(source_root).get("minecraft:overworld")
        if dimension is None:
            return False
        cx, cz = project["centre_minecraft_xz"]
        xmin, zmin = (math.floor(cx/16)-64)*16, (math.floor(cz/16)-64)*16
        expected_regions = {
            (dimension / "region" / f"r.{rx}.{rz}.mca").relative_to(source_root).as_posix()
            for rx in range(xmin//512, (xmin+2047)//512+1)
            for rz in range(zmin//512, (zmin+2047)//512+1)
            if (dimension / "region" / f"r.{rx}.{rz}.mca").exists()}
        if (record["scope"]["bounds_blocks_xz"] != [xmin, zmin, xmin+2048, zmin+2048]
                or record["surface"]["sample_spacing_m"] != 4
                or record["surface"]["shape"] != [512, 512]
                or not record["source_regions"] or set(record["source_regions"]) != expected_regions):
            return False
        if (record["scope"]["requested_center_xz"] != project["centre_minecraft_xz"]
                or record["source_level_dat_sha256"] != sha256_file(Path(snapshot["snapshot_path"]) / "level.dat")):
            return False
        return all(safe_path(source_root, name).is_file()
                   and sha256_file(safe_path(source_root, name)) == digest
                   for name, digest in record["source_regions"].items())
    if not macro_matches(macro_record):
        macro_dir = workspace / "artifacts/source" / snapshot["save_sha256"] / "macro_region"
        macro_record = macro_dir / "macro_surface.json"
    if macro_record.exists() and not macro_matches(macro_record):
        from .source.macro import macro_producer_identity
        identity = {"producer": macro_producer_identity(), "snapshot": snapshot["save_sha256"],
                    "center": project["centre_minecraft_xz"], "extent": 2048, "spacing": 4}
        macro_dir = macro_dir.with_name("macro_region_"+hash_object(identity)[:16])
        macro_record = macro_dir / "macro_surface.json"
    if not macro_record.exists():
        from .source.macro import extract_macro_surface
        extract_macro_surface(Path(snapshot["snapshot_path"]), macro_dir,
                              center=tuple(project["centre_minecraft_xz"]))
    if not macro_matches(macro_record):
        raise ValueError("Macro context is stale for the immutable source snapshot")
    for entry in read_json(macro_record)["files"]:
        if sha256_file(safe_path(macro_dir, entry["path"], must_exist=True)) != entry["sha256"]:
            raise ValueError("Macro surface artifact was changed")
    from .source.independent import ensure_independent_comparisons
    source_comparisons = ensure_independent_comparisons(Path(snapshot["snapshot_path"]), initial, macro_dir)
    macro_path = macro_dir / "macro_surface.npz"
    drainage_identity = hash_object({"source": sha256_file(macro_path),
                                    "producer": sha256_file(workspace / "src/isaacmin/terrain/baseline.py"),
                                    "scope": "regional_priority_flood"})
    drainage_path = workspace / "artifacts/terrain/catchments" / drainage_identity / "global_drainage.npz"
    drainage_record = drainage_path.parent / "drainage.json"
    if (not drainage_path.exists() or not drainage_record.exists()
            or read_json(drainage_record)["source_hash"] != sha256_file(macro_path)
            or read_json(drainage_record).get("sha256") != sha256_file(drainage_path)
            or read_json(drainage_record).get("producer_sha256") != sha256_file(workspace / "src/isaacmin/terrain/baseline.py")):
        from .terrain.baseline import save_drainage
        with np.load(macro_path, allow_pickle=False) as macro:
            save_drainage(drainage_path.parent, macro["height"], macro["validity"],
                          spacing_m=float(macro["sample_spacing_m"]), source_hash=sha256_file(macro_path))
    dependencies = {"source": snapshot["save_sha256"], "tools": tools_hash,
                    "assets": sha256_file(assets_path) if assets_path.exists() else "unavailable",
                    "code": code_hash(workspace), "validator": validator_hash(),
                    "recipe": hash_object({"name": "temperate_meadow_constrained_highmap_v1", "soil_limit_m": .85, "rock_limit_m": .25}),
                    "world_ir": sha256_file(initial / "world_ir.json"),
                    "source_surface": sha256_file(initial / "terrain_surface.npz"),
                    "supporting_solids": sha256_file(support_directory / "support_manifest.json"),
                    "catchment": sha256_file(drainage_path),
                    "macro_source": sha256_file(macro_path),
                    "cave_components": sha256_file(initial / "topology/source_topology_graph.json"),
                    "materials": sha256_file(workspace / "state/terrain_materials.json") if (workspace / "state/terrain_materials.json").exists() else "unavailable",
                    "normalized_assets": sha256_file(workspace / "state/normalized_assets.json") if (workspace / "state/normalized_assets.json").exists() else "unavailable",
                    "procedural_canopy": sha256_file(workspace / "state/procedural_assets.json") if (workspace / "state/procedural_assets.json").exists() else "unavailable"}
    dependencies["lighting"] = sha256_file(workspace / "state/lighting_candidates.json")
    dependencies["references"] = sha256_file(workspace / "state/reference_catalogue.json")
    dependencies["host"] = sha256_file(workspace / "artifacts/bootstrap/host.json")
    if material_recipe:
        dependencies["terrain_material_recipe"] = hash_object(material_recipe)
        dependencies["shared_original_materials"] = material_recipe["candidate_manifest_sha256"]
    dependencies["capture_recipe"] = hash_object(capture_recipe)
    dependencies["foliage_material_recipe"] = hash_object(foliage_recipe)
    request_hash = hash_object({"dependencies": dependencies, "project": project, "integration_only": integration_only})
    build_id = "build_" + request_hash[:16]
    run_id = resume_run_id or "run_" + request_hash[:16]
    directory = workspace / "worlds" / build_id
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": "1.0", "build_id": build_id, "run_id": run_id,
                "created_at_utc": utc_now(), "dependencies": dependencies, "project": project,
                "integration_only": integration_only, "source_ir": str(initial.relative_to(workspace)),
                "independent_source": source_comparisons,
                "scope": {"initial_region_m": 128,
                    "target_extent_m": [2048, 2048], "centre_minecraft_xz": project["centre_minecraft_xz"]},
                "status": "running", "blockers": []}
    manifest["input_files"] = {name: {"path": str(path.relative_to(workspace)), "sha256": sha256_file(path)}
        for name, path in {"world_ir": initial / "world_ir.json", "source_surface": initial / "terrain_surface.npz",
            "supporting_solids": support_directory / "support_manifest.json", "catchment": drainage_path,
            "macro_source": macro_path, "cave_components": topology_record,
            "tools": lock_path, "assets": assets_path,
            "materials": workspace / "state/terrain_materials.json", "normalized_assets": workspace / "state/normalized_assets.json",
            "procedural_canopy": workspace / "state/procedural_assets.json", "lighting": workspace / "state/lighting_candidates.json",
            "references": workspace / "state/reference_catalogue.json", "host": workspace / "artifacts/bootstrap/host.json"}.items()
        if path.is_file()}
    if material_recipe:
        for name, path in {
            "terrain_material_recipe": workspace / "state/terrain_material_recipe.json",
            "shared_original_materials": workspace / material_recipe["candidate_manifest"],
        }.items():
            manifest["input_files"][name] = {"path": str(path.relative_to(workspace)), "sha256": sha256_file(path)}
    if (workspace / "state/render_recipe.json").is_file():
        path = workspace / "state/render_recipe.json"
        manifest["input_files"]["capture_recipe"] = {"path": str(path.relative_to(workspace)), "sha256": sha256_file(path)}
    if foliage_recipe:
        path = workspace / "state/foliage_material_recipe.json"
        manifest["input_files"]["foliage_material_recipe"] = {"path": str(path.relative_to(workspace)), "sha256": sha256_file(path)}
    profile = freeze_profile(directory / "quality_profile.json", {
        "dependencies": dependencies,
        "coordinates": project["coordinate_frame"], "scope": manifest["scope"],
        "capture_recipe": capture_recipe, "terrain_material_recipe": material_recipe,
        "foliage_material_recipe": foliage_recipe})
    from .validation.source_fidelity import freeze_source_fidelity_profile
    freeze_source_fidelity_profile(profile, directory / "source_fidelity_profile.json")
    atomic_json(workspace / "state/runs" / (run_id + ".json"), {
        "run_id": run_id, "build_id": build_id, "world": project["source_world"],
        "integration_only": integration_only, "request_hash": request_hash})
    if (directory / "qualification.json").exists() and read_json(directory / "qualification.json").get("stage") == "autonomously_qualified":
        # Revalidate exact bytes before reuse, preserving the last qualified world.
        previous_report = validate_build(workspace, build_id)
        if previous_report["stage"] == "autonomously_qualified":
            return previous_report
    from .jobs.repair_budget import exhausted_repairs
    exhausted = exhausted_repairs(workspace, dependencies["source"], dependencies["world_ir"])
    if exhausted:
        manifest["status"] = "blocked_repair_budget"
        manifest["exhausted_repairs"] = exhausted
        manifest["blockers"] = [entry["defect_class"]+": "+entry["reason"] for entry in exhausted]
        atomic_json(directory / "build.json", manifest)
        report = validate_build(workspace, build_id)
        report["status"] = "blocked_repair_budget"
        report["exhausted_repairs"] = exhausted
        return report
    if not workers["cross_runtime_qualified"]:
        manifest["status"] = "waiting_dependency"
        manifest["blockers"] = ["Native compatibility pending for " + name for name in ("highmap", "openvdb", "blender", "isaac", "openvdb_independent", "native_exchange")
                                if workers[name].get("status") not in {"pass", "success"}]
        atomic_json(directory / "build.json", manifest)
        report = validate_build(workspace, build_id)
        report["status"] = "waiting_dependency"
        return report
    if not (directory / "stage_checkpoints").exists() and any((directory / name).exists() for name in ("scene", "evidence", "data/bare_scene")):
        import shutil
        from datetime import datetime, timezone
        attempt = directory / "attempts" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        attempt.mkdir(parents=True)
        for relative in ("scene", "evidence", "data/bare_scene", "data/ecology", "data/scenario_route.json", "build.json", "qualification.json"):
            previous = directory / relative
            if previous.exists():
                target = attempt / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(previous, target)
        shutil.copyfile(directory / "quality_profile.json", attempt / "quality_profile.json")
    surface = np.load(initial / "terrain_surface.npz", allow_pickle=False)
    from .source.semantics import classify
    names = surface["block_names"]
    substrate = np.array([{"soil": 1, "sediment": 2, "rock": 3}.get(classify(str(n)), 0) for n in names])[surface["substrate_id"]]
    occupancy = np.load(initial / "natural_occupancy.npz", allow_pickle=False)
    topology = np.load(initial / "topology/source_topology_labels.npz", allow_pickle=False)
    source_y = np.arange(occupancy["occupancy"].shape[0]) + int(occupancy["min_xyz"][1])
    shallow_void = topology["covered_air"] & (surface["height"][None, :, :] - source_y[:, None, None] <= 6)
    from scipy.ndimage import binary_dilation
    # Six metres of source cover exceeds the maximum permitted top-surface change.
    # Every portal also receives a four-metre protection collar, enforced per iteration.
    exposed_water = surface["water_validity"] & (surface["water_height"] >= surface["height"])
    protect = shallow_void.any(axis=0) | exposed_water
    protect |= binary_dilation((topology["portal_labels"] > 0).any(axis=0), iterations=4)
    baseline, baseline_record = reconstruct_baseline(surface["height"], surface["validity"], substrate, protect)
    global_surface = np.load(macro_path, allow_pickle=False)
    global_flow = np.load(drainage_path, allow_pickle=False)["contributing_area_m2"]
    regional_flow_clip = float(10 * np.sqrt(np.mean(global_flow[global_surface["validity"]])))
    normalized_flow = np.clip(global_flow / max(regional_flow_clip, 1e-12), 0., 1.)
    # Global field samples are fixed world-coordinate samples, never reseeded per tile.
    from scipy.ndimage import map_coordinates
    minxz, globalmin = surface["min_xz"], global_surface["min_xz"] + global_surface["sample_offset_m"]
    z, x = np.indices(baseline.shape)
    gz = (minxz[1]+z+.5-globalmin[1])/float(global_surface["sample_spacing_m"])
    gx = (minxz[0]+x+.5-globalmin[0])/float(global_surface["sample_spacing_m"])
    shared_flow = map_coordinates(normalized_flow, [gz, gx], order=1, mode="nearest")

    def refine(output):
        result = refine_heightfield(baseline, output, protection=protect, global_runoff=shared_flow,
                                    global_runoff_normalized=True)
        if result.get("status") == "success":
            refined = np.load(output / "refinement.npz")["height"]
            np.savez_compressed(output / "exterior_delta.npz", source_height=surface["height"],
                                delta=refined-surface["height"], protection=protect)
            atomic_json(output / "baseline.json", baseline_record)
            atomic_json(output / "regional_flow_normalization.json", {
                "global_drainage_sha256": dependencies["catchment"],
                "regional_clip_area_m2": regional_flow_clip,
                "method": "HighMap stream-power weight clipped and normalized once over full source macroregion",
                "local_tile_renormalization": False})
        return result

    try:
        refinement = _job(workspace, run_id, "refinement", dependencies, {"extent_m": 128},
                          directory / "data/refinement", profile, refine)
        from .validation.refinement import validate_refinement
        refinement_validation = validate_refinement(initial / "terrain_surface.npz", directory / "data/refinement",
            macro_path, drainage_path, directory / "evidence/refinement")
        if refinement_validation["status"] != "pass":
            raise ValueError("Independent native refinement validation failed: " + str(refinement_validation["errors"]))
        volume = _job(workspace, run_id, "volume", dependencies, {"extent_m": 128, "voxel_size_m": 1},
                      directory / "data/volume", profile,
                      lambda out: reconstruct_occupancy(np.load(support_mask_path, allow_pickle=False)["occupancy"], out))
        from .assembly.pipeline import assemble_region
        from .assets.candidates import candidate_materials, candidate_placements
        materials = (validate_material_recipe(workspace, material_recipe) if material_recipe
                     else candidate_materials(workspace))
        exterior = {"path": str(directory / "data/refinement/exterior_delta.npz"),
                    "source_height_key": "source_height", "delta_key": "delta", "protection_key": "protection",
                    "source_min_xz": surface["min_xz"].tolist()}
        detail = {"subdivision_levels": 3, "soil_displacement_peak_to_peak_m": .02}
        # Placement planning needs the exact ground before instances exist. Keep
        # this first export and its immutable material bakes separate from the
        # populated scene, including when the second stage is interrupted.
        bare_scene = directory / "data/bare_scene"
        from .jobs.stage_checkpoint import run_stage
        from .assembly.reuse import assemble_bare
        from .assembly.capture_reuse import reuse_capture_plan, source_inventory
        inventory = source_inventory(workspace)
        if hash_object(inventory) != dependencies['code']:
            raise ValueError('Code changed after the build dependency snapshot')
        inventory_path = directory / 'execution_source_files.json'
        if inventory_path.exists() and read_json(inventory_path) != inventory:
            raise ValueError('Existing build execution inventory differs')
        atomic_json(inventory_path, inventory)
        stage_identity = {"dependencies": dependencies, "project": project, "detail": detail}
        scene_result = run_stage(directory, "bare_terrain", stage_identity,
            ["data/bare_scene", "evidence/bare_cache_selection.json"],
            lambda: assemble_bare(workspace, initial, bare_scene, materials=materials,
                                       origin=tuple(project["coordinate_frame"]["origin_blocks"]),
                                       support_mask_path=support_mask_path,
                                       volume_directory=directory / "data/volume", exterior_delta=exterior,
                                       geometry_detail=detail, terrain_material_recipe=material_recipe,
                                       source_fidelity_budget_path=directory / "source_fidelity_profile.json",
                                       evidence=directory / "evidence/bare_cache_selection.json"),
            accepted=lambda r: r.get("status") == "success")
        if scene_result.get("status") != "success":
            raise RuntimeError("Native terrain assembly failed: " + str(scene_result.get("reason", "inspect worker log")))
        manifest["stages"] = {"refinement": refinement, "independent_refinement": refinement_validation,
                              "volume": volume, "scene": scene_result}
        manifest["status"] = "geometry_world"
        atomic_json(directory / "build.json", manifest)
        ground_path = bare_scene / "final_ground.obj"
        ground_hash = sha256_file(ground_path)
        from .validation.worker import run_validation,measurement_completed
        plan_identity = dict(stage_identity, final_ground_sha256=ground_hash)
        def plan_captures():
            parameters = {
                "mesh_path": ground_path, "ir_dir": initial, "origin": project["coordinate_frame"]["origin_blocks"],
                "support_manifest_path": support_directory / "support_manifest.json"}
            value = reuse_capture_plan(workspace, directory, plan_identity, parameters)
            if value is None:
                value = run_validation(workspace, "capture_plan", directory / "evidence/planning", parameters)
            atomic_json(directory / "evidence/capture_plan.json", value)
            return value
        plan = run_stage(directory, "capture_plan", plan_identity,
            ["evidence/planning", "evidence/capture_plan.json"], plan_captures,
            accepted=lambda r: r.get("status") == "planned")
        route_path = directory / "data/scenario_route.json"
        atomic_json(route_path, {"points_world_xyz": plan["route_points_world_xyz"], "corridor_width_m": 1.2,
            "provenance": "authored_scenario_route_not_extracted_trail", "ground_sha256": ground_hash,
            "navigation_stack": "not_run_not_supplied"})
        ecology = run_stage(directory, "ecology", dict(stage_identity, final_ground_sha256=ground_hash,
            route_sha256=sha256_file(route_path)), ["data/ecology"],
            lambda: candidate_placements(workspace, initial, bare_scene / "scene.blend", directory / "data/ecology",
                runoff_path=directory / "data/refinement/refinement.npz", route_path=route_path),
            accepted=lambda r: bool(r.get("exporter_assets")))
        population = directory / "data/populated_export"
        population_identity = dict(stage_identity, ecology_sha256=sha256_file(directory / "data/ecology/ecology_manifest.json"),
                                   bare_cache_sha256=sha256_file(bare_scene / "bare_scene_cache.json"))
        scene_result = run_stage(directory, "native_population", population_identity, ["data/populated_export"],
            lambda: assemble_region(initial, population, materials=materials,
                                       origin=tuple(project["coordinate_frame"]["origin_blocks"]),
                                       support_mask_path=support_mask_path,
                                       volume_directory=directory / "data/volume", exterior_delta=exterior,
                                       geometry_detail=detail, assets=ecology["exporter_assets"],
                                       bare_scene_directory=bare_scene, terrain_material_recipe=material_recipe,
                                       foliage_material_recipe=foliage_recipe,
                                       source_fidelity_budget_path=directory / "source_fidelity_profile.json"),
            accepted=lambda r: r.get("status") == "success")
        from .assembly.finalize import instance_scene, collision_scene
        instance_result = run_stage(directory, "native_instances", dict(population_identity,
            export_closure_sha256=sha256_file(population / "native_dependency_closure.json")),
            ["data/instanced_export", "evidence/native_instancing"],
            lambda: instance_scene(workspace, population, directory / "data/instanced_export",
                                   directory / "evidence/native_instancing"),
            accepted=lambda r: r.get("status") == "lossless_native_instancing_verified",
            partial_outputs=['data/instanced_export.staging'])
        scene_result = run_stage(directory, "exact_collision", dict(population_identity,
            instance_closure_sha256=instance_result["closure_sha256"]),
            ["scene", "evidence/exact_collision"],
            lambda: collision_scene(workspace, directory / "data/instanced_export", population,
                directory / "scene", directory / "evidence/exact_collision", instance_result),
            accepted=lambda r: r.get("status") == "success", partial_outputs=['scene.staging'])
        ground_path = directory / "scene/final_ground.obj"
        if scene_result.get("status") != "success" or sha256_file(ground_path) != ground_hash:
            raise RuntimeError("Populated scene changed the final supporting terrain or failed export; placements require rebuilding")
        manifest["stages"] = {"refinement": refinement, "independent_refinement": refinement_validation,
                              "volume": volume, "scene": scene_result}
        manifest["status"] = "geometry_world" if scene_result.get("status") == "success" else "failed"
        # Save before expensive independent checks so interruption can resume from real artifacts.
        atomic_json(directory / "build.json", manifest)
        def measure(kind, output, parameters):
            try:
                relative = str(output.relative_to(directory))
                inputs = {}
                for key, value in parameters.items():
                    if isinstance(value, (str, Path)) and Path(value).is_file():
                        inputs[key] = sha256_file(Path(value))
                    elif isinstance(value, (str, Path)) and Path(value).is_dir():
                        # Native captures and geometry are already byte-bound by
                        # their complete parent checkpoints. Include their record
                        # identities so changed measurements cannot be reused.
                        for name in ('capture_result.json','traversal_result.json','world_ir.json',
                                     'native_dependency_closure.json','physx_contact_rays.json'):
                            file = Path(value) / name
                            if file.is_file():inputs[key+'/'+name] = sha256_file(file)
                return run_stage(directory, 'measure_'+kind+'_'+hash_object(relative)[:8],
                    dict(stage_identity, parameters={k:str(v) if isinstance(v,Path) else v for k,v in parameters.items()},
                         input_files=inputs, final_ground_sha256=ground_hash), [relative],
                    lambda: run_validation(workspace, kind, output, parameters),
                    accepted=measurement_completed)
            except Exception as exc:
                failure = {"status": "failed_to_measure", "kind": kind,
                           "type": type(exc).__name__, "reason": str(exc)}
                atomic_json(output / "failure.json", failure)
                manifest["blockers"].append(f"Independent {kind}: {type(exc).__name__}: {exc}")
                return failure
        # A missing native material is already a hard failure. Detect it before
        # long source queries and full-quality recordings, retaining the complete
        # prescribed capture plan for the repaired candidate.
        preflight = measure('usd_structure', directory / 'evidence/usd_structure_preflight',
                            {'scene': directory / 'scene/world.usda'})
        manifest['stages']['usd_structure_preflight'] = preflight
        if preflight.get('status') != 'pass' or preflight.get('errors'):
            raise RuntimeError('Native material/collider preflight failed; repair exported assets before capture')
        manifest["stages"]["independent_water"] = measure("water_interfaces", directory / "evidence/water_interfaces", {
            "snapshot": snapshot["snapshot_path"], "ir_dir": initial,
            "water_mesh_path": directory / "scene/source_water_interfaces.json",
            "origin": project["coordinate_frame"]["origin_blocks"]})
        manifest["stages"]["independent_ecology"] = measure("ecology", directory / "evidence/ecology", {
            "workspace": workspace, "scene": directory / "scene/world.usda",
            "ecology_manifest": directory / "data/ecology/ecology_manifest.json", "ir_directory": initial,
            "route_path": route_path, "runoff_path": directory / "data/refinement/refinement.npz"})
        transform = CoordinateFrame(tuple(project["coordinate_frame"]["origin_blocks"])).record()["world_to_source"]
        manifest["stages"]["continuous_surface"] = measure("continuous_surface", directory / "evidence/continuous_surface", {
            "mesh_path": ground_path, "source_ir": str(initial), "mesh_to_source": transform,
            "quality_profile_path": directory / "quality_profile.json", "topology_scope": "closed_initial_crop"})
        source_parameters = {"ir_dir": initial, "mesh_path": ground_path, "mesh_to_source": transform,
                             "support_manifest_path": support_directory / "support_manifest.json"}
        source_mesh = measure("source_mesh", directory / "evidence/topology_samples", source_parameters)
        connectivity = measure("source_connectivity", directory / "evidence/topology_graph", source_parameters)
        manifest["stages"]["topology"] = {"samples": source_mesh, "connectivity": connectivity}
        manifest["stages"]["source_fidelity"] = measure("source_fidelity", directory / "evidence/source_fidelity", {
            "ir_dir": initial, "mesh_path": ground_path, "frozen_budget_path": directory / "source_fidelity_profile.json",
            "exterior_delta_path": directory / "data/refinement/exterior_delta.npz",
            "topology_samples_path": directory / "evidence/topology_samples/source_mesh_validation.json",
            "topology_graph_path": directory / "evidence/topology_graph/mesh_connectivity_comparison.json",
            "support_manifest_path": support_directory / "support_manifest.json"})
        from .adapters.workers import capture_scene
        def capture_checkpoint(name, *, poses, illumination, lighting, physics_scene_name, scope):
            relative = 'evidence/' + name
            suffix = '' if physics_scene_name == 'world_physics.usda' else '_'+Path(physics_scene_name).stem
            closure_name = ('native_physics_dependency_closure.json' if not suffix else
                            'native_'+Path(physics_scene_name).stem+'_dependency_closure.json')
            return run_stage(directory, name, dict(stage_identity,
                scene_closure_sha256=sha256_file(directory / 'scene/native_dependency_closure.json'),
                poses=poses, contact_probes=plan['contact_probes'], illumination=illumination,
                renderer=capture_recipe, lighting=lighting, scope=scope),
                [relative, 'scene/' + physics_scene_name, 'scene/'+closure_name,
                 'scene/runtime_dependencies'+suffix+'.json'],
                lambda: capture_scene(directory / 'scene/world.usda', directory / relative,
                    poses=poses, contact_probes=plan['contact_probes'],
                    hdri=illumination['hdri'], lighting=lighting, lighting_parameters=illumination,
                    renderer_recipe=capture_recipe['renderer_recipe'],
                    physics_scene_name=physics_scene_name, scope=scope),
                # Reuse a completed failing measurement honestly; never infer a
                # quality pass or erase it by blindly repeating the same inputs.
                accepted=lambda r: r.get('status') in {'pass','fail'}
                    and len(r.get('frames',[])) == len(poses)
                    and r.get('native_reopen',{}).get('status') in {'pass','fail'})
        illumination = lighting_recipe["conditions"]["directional"]
        captures = capture_checkpoint('isaac_static', poses=plan['poses_static'],
            illumination=illumination, lighting='directional', physics_scene_name='world_physics.usda',
            scope='real_source_initial_region_static_candidate')
        manifest["stages"]["isaac_static"] = captures
        if (directory / 'scene/world_physics.usda').is_file():
            manifest['stages']['usd_structure'] = measure('usd_structure',directory/'evidence/usd_structure',
                {'scene':directory/'scene/world_physics.usda'})
        if captures.get("status") in {"pass", "fail"} and captures.get("frames"):
            manifest["stages"]["independent_depth"] = measure("depth", directory / "evidence/depth", {
                "mesh_path": ground_path, "capture_dir": directory / "evidence/isaac_static"})
        if (directory / "evidence/isaac_static/physx_contact_rays.json").exists():
            manifest["stages"]["independent_contact"] = measure("contact", directory / "evidence/contact", {
                "mesh_path": ground_path, "capture_dir": directory / "evidence/isaac_static",
                "source_surface": initial / "terrain_surface.npz", "origin": project["coordinate_frame"]["origin_blocks"]})
            manifest["stages"]["independent_gravity"] = measure("gravity", directory / "evidence/gravity", {
                "mesh_path": ground_path, "capture_dir": directory / "evidence/isaac_static"})
        # Failures remain visible; successful captures never imply visual qualification.
        capture_ready = (captures.get("native_reopen", {}).get("status") == "pass"
                         and len(captures.get("frames", [])) == len(plan["poses_static"]))
        # A contact failure blocks release but does not prevent independent
        # camera diagnostics or a recorded, bounded support traversal attempt.
        if capture_ready:
            diffuse = lighting_recipe["conditions"]["diffuse"]
            manifest["stages"]["isaac_static_diffuse"] = capture_checkpoint('isaac_static_diffuse',
                poses=plan['poses_static'], illumination=diffuse, lighting='diffuse',
                physics_scene_name='world_physics_diffuse.usda',
                scope='real_source_initial_region_diffuse_static_candidate')
            manifest["stages"]["isaac_motion"] = capture_checkpoint('isaac_motion',
                poses=plan['poses_motion'], illumination=illumination, lighting='directional',
                physics_scene_name='world_physics_motion.usda',
                scope='real_source_initial_region_continuous_motion_candidate')
            if manifest["stages"]["isaac_motion"].get("status") in {"pass", "fail"}:
                manifest["stages"]["temporal"] = measure("temporal", directory / "evidence/temporal", {
                    "capture_dir": directory / "evidence/isaac_motion"})
            from .adapters.traversal import traverse_scene
            support_traversals = []
            for index, route in enumerate(plan.get("support_routes", [])):
                native_directory = directory / "evidence/traversal" / f"route_{index:03d}" / "native"
                measured_directory = native_directory.parent / "independent"
                native = traverse_scene(workspace, directory / "scene/world_physics.usda", native_directory,
                    route_ground_xyz=route["points_world_xyz"], surface_scope=route["surface_scope"])
                entry = {"id": route["id"], "surface_scope": route["surface_scope"], "native": native}
                if (native_directory / "traversal_result.json").is_file():
                    entry["independent"] = measure("traversal", measured_directory, {
                        "mesh_path": ground_path, "traversal_dir": native_directory})
                    if entry["independent"].get("status") in {"pass", "fail"}:
                        entry["measurement"] = str((measured_directory / "traversal_validation.json").relative_to(directory / "evidence"))
                        entry["samples"] = str((measured_directory / "traversal_comparison.npz").relative_to(directory / "evidence"))
                        entry["native_record"] = str((native_directory / "traversal_result.json").relative_to(directory / "evidence"))
                        entry["native_samples"] = str((native_directory / "traversal_samples.jsonl").relative_to(directory / "evidence"))
                        entry["request"] = str((native_directory / "traversal_request.json").relative_to(directory / "evidence"))
                support_traversals.append(entry)
                manifest["stages"]["support_traversals"] = support_traversals
                atomic_json(directory / "build.json", manifest)
        from .validation.evidence import bind_measurement
        contact_record = manifest["stages"].get("independent_contact")
        if (contact_record and contact_record.get("status") in {"pass", "fail"}
                and manifest["stages"].get("independent_gravity", {}).get("status") in {"pass", "fail"}):
            traversals = []
            traversal_files = []
            for entry in manifest["stages"].get("support_traversals", []):
                if "measurement" not in entry:
                    continue
                traversals.append({key: entry[key] for key in ("id", "surface_scope", "measurement", "samples", "native_record", "native_samples", "request")})
                traversal_files.extend(directory / "evidence" / entry[key]
                                       for key in ("measurement", "samples", "native_record", "native_samples", "request"))
            bind_measurement(directory, "ground_contact", {
                "samples": "contact/ground_samples.npz", "native_capture": "isaac_static/capture_result.json",
                "gravity_measurement": "gravity/dropped_probe_validation.json",
                "gravity_samples": "gravity/dropped_probe_surface_samples.npz",
                "final_ground_sha256": ground_hash,
                "independent_verification_status": contact_record["status"],
                "supported_footprint_traversals": traversals,
                "missing_cave_contact_probes":plan.get('missing_cave_contact_probes',[]),
                "missing_exterior_contact_probes":plan.get('missing_exterior_contact_probes',[]),
                "missing_cave_support_routes": plan.get("missing_cave_support_routes", [])},
                [directory / "evidence/contact/ground_samples.npz", directory / "evidence/contact/contact_validation.json",
                 directory / "evidence/isaac_static/physx_contact_rays.json", directory / "evidence/isaac_static/physx_contact_rays.npz",
                 directory / "evidence/gravity/dropped_probe_validation.json", directory / "evidence/gravity/dropped_probe_surface_samples.npz",
                 directory / "evidence/isaac_static/capture_result.json", directory / "evidence/isaac_static/capture_request.json",
                 *traversal_files])
        depth_record = manifest["stages"].get("independent_depth")
        if depth_record and depth_record.get("status") in {"pass", "fail"}:
            capture_record = directory / "evidence/isaac_static/capture_result.json"
            native_frame_files = [capture_record.parent / frame[role]
                                  for frame in read_json(capture_record)["frames"]
                                  for role in ("rgb", "depth", "instance_segmentation")]
            bind_measurement(directory, "depth_comparison", {
                "samples": "depth/depth_samples.npz", "depth_semantics": "axial_z",
                "same_camera_pose_and_timestamp": True,
                "synchronization": "RGB and distance_to_image_plane read after same synchronous Replicator step",
                "independent_verification_status": depth_record["status"],
                "failed_frames": depth_record["failed_frames"],
                "excluded_surfaces": depth_record["exclusions"]},
                [directory / "evidence/depth/depth_samples.npz", directory / "evidence/depth/depth_validation.json",
                 capture_record, capture_record.parent / "capture_request.json", *native_frame_files])
        # Actual visibility is measured from saved native segmentation and depth,
        # independently of pose tags. Run each large mesh query in its own process.
        static_results = [directory / "evidence" / name / "capture_result.json"
                          for name in ("isaac_static", "isaac_static_diffuse")
                          if (directory / "evidence" / name / "capture_result.json").is_file()]
        if static_results:
            manifest["stages"]["feature_visibility"] = measure("feature_visibility",
                directory / "evidence/feature_visibility", {"mesh_path": ground_path, "ir_dir": initial,
                "capture_results": [str(p) for p in static_results],
                "origin": project["coordinate_frame"]["origin_blocks"]})
            feature_report = directory / "evidence/feature_visibility/observed/feature_visibility.json"
            visibility = measure("view_coverage", directory / "evidence/view_coverage", {
                "workspace": workspace, "scene": directory / "scene/world.usda",
                "capture_results": [str(p) for p in static_results],
                "ecology_manifest": directory / "data/ecology/ecology_manifest.json",
                "feature_evidence": feature_report if feature_report.is_file() else None})
            manifest["stages"]["view_coverage"] = visibility
            required_features = []
            if graph.get("features"):
                required_features.extend(("cave_interior", "cave_underside"))
            if graph.get("portals"):
                required_features.append("cave_portal")
            water_record = read_json(directory / "scene/source_water_interfaces.json")
            if water_record.get("triangles"):
                required_features.append("surface_water")
            from .adapters.visual_workflow import freeze_review, inspect_frozen
            review = freeze_review(workspace, static_results, directory / "quality_profile.json",
                directory / "evidence/visual_review", required_features=required_features,
                required_material_families=visibility.get("used_material_families", []),
                required_asset_families=visibility.get("used_asset_families", []),
                visibility_report=directory / "evidence/view_coverage/view_coverage.json" if visibility.get("status") == "measured" else None)
            manifest["stages"]["frozen_visual_review"] = review
            qualified_critic = workspace / "state/qualified_visual_critic.json"
            if qualified_critic.is_file():
                pointer = read_json(qualified_critic)
                calibration = safe_path(workspace, pointer["calibration_report"], must_exist=True)
                if sha256_file(calibration) != pointer["calibration_sha256"]:
                    raise ValueError("Qualified visual critic calibration bytes changed")
                manifest["stages"]["visual_review"] = inspect_frozen(workspace, review["manifest"],
                    calibration, directory / "evidence/visual_review")
            else:
                manifest["stages"]["visual_review"] = {
                    "status": "blocked_calibration", "frozen_manifest": review["manifest"],
                    "reason": "The required hosted visual critic has not passed independent full-category fault/clean-control calibration",
                    "qualification": "not_run; saved actual images and numerical checks remain available"}
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["blockers"].append(f"{type(exc).__name__}: {exc}")
    atomic_json(directory / "build.json", manifest)
    # Provenance is useful even when visual/contact qualification failed. Keep
    # the audited build manifest stable after this point, and store the audit
    # separately rather than creating a circular self-hash in build.json.
    if (directory / "scene/world.usda").is_file():
        from .validation.worker import run_validation
        try:
            from .jobs.stage_checkpoint import run_stage
            run_stage(directory,'provenance',{'manifest_sha256':sha256_file(directory/'build.json'),
                'dependencies':dependencies},['evidence/provenance'],
                lambda:run_validation(workspace,"provenance",directory/'evidence/provenance',{
                    'workspace':workspace,'build_dir':directory}),
                accepted=lambda r:r.get('status') in {'pass','fail','incomplete'})
        except Exception as exc:
            atomic_json(directory / "evidence/provenance/audit_failure.json", {
                "status": "fail", "type": type(exc).__name__, "reason": str(exc)})
    report = validate_build(workspace, build_id)
    # A copied package must be exercised independently. This also produces the
    # best truthful development deliverable when unrelated visual gates fail.
    if (directory / "scene/world.usda").is_file():
        from .packaging import package_build
        from .portable import verify_portable
        try:
            packaged = package_build(workspace, build_id, require_qualified=False)
            atomic_json(directory / "evidence/package_result.json", packaged)
            if packaged.get("status") == "pass":
                report["package"] = packaged["package"]
                package_manifest = read_json(Path(packaged["package"]) / "package.json")
                if package_manifest.get("reproduction", {}).get("reference"):
                    replay = verify_portable(workspace, Path(packaged["package"]), directory / "evidence/portable_delivery")
                    atomic_json(directory / "evidence/portable_replay_result.json", replay)
                    report = validate_build(workspace, build_id)
                    report["package"] = packaged["package"]
        except Exception as exc:
            atomic_json(directory / "evidence/portable_delivery_failure.json", {
                "status": "fail", "type": type(exc).__name__, "reason": str(exc)})
    if not integration_only:
        # Q11 measures the expanded world and cannot be a prerequisite for
        # beginning expansion. Every other gate remains required at local scope.
        missing_local = [g["id"] for g in report["gates"] if g["id"] != "Q11" and g["status"] != "pass"]
        report["scale_status"] = "blocked_by_local_qualification" if missing_local else "local_checks_pass_expansion_not_run"
        report["scale_prerequisites_remaining"] = missing_local
        report["expanded_world_built"] = False
    return report


def resume(workspace: Path, run_id: str) -> dict:
    record = read_json(safe_path(workspace / "state/runs", run_id + ".json", must_exist=True))
    journal = Journal(workspace / "state/journal.sqlite")
    journal.recover()
    journal.close()
    return build(workspace, Path(record["world"]), integration_only=record["integration_only"], resume_run_id=run_id)
