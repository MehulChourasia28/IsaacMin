"""Replay an unchanged population request without reconstructing cached terrain.

This is a native construction continuation, not a validation/qualification pass.
The original failed build and admission evidence remain unchanged.
"""
from pathlib import Path

from ..io import atomic_json, read_json, sha256_file, utc_now
from ..process import run_worker


def reuse_memory_estimate(bare_scene_bytes: int, unique_asset_bytes: int) -> dict:
    if type(bare_scene_bytes) is not int or type(unique_asset_bytes) is not int:
        raise ValueError("Native serialized sizes must be integers")
    if bare_scene_bytes <= 0 or unique_asset_bytes < 0:
        raise ValueError("A nonempty verified native scene is required")
    return {
        "estimated_memory_bytes": 2 * 2**30 + 7 * bare_scene_bytes + 13 * unique_asset_bytes,
        "estimated_disk_bytes": 4 * (bare_scene_bytes + unique_asset_bytes) + 2**30,
        "shared_memory_reserve_bytes": 24 * 2**30,
        "method": "2GiB allowance +7*verified cached blend bytes +13*unique source asset blend bytes",
        "basis": "Cached population loads the applied mesh and does not run OpenSubdiv or allocate construction intermediates; serialized mesh, USD exchange and asset buffers still need headroom",
        "estimate_is_measured_peak": False,
        "calibration": "Actual 2026-10-01 cached full-detail region export peaked at73.3621GiB; this model includes headroom over that observation, not a universal upper bound",
        "actual_shared_memory_guard": "unchanged; worker stops if available unified memory falls below24GiB",
    }


def populate_cached_request(workspace: Path, request_path: Path, evidence: Path) -> dict:
    from .native_cache import verify_bare, deduplicate_generated_textures
    from ..adapters.workers import native_environment

    root, request_path, evidence = map(lambda p: Path(p).resolve(), (workspace, request_path, evidence))
    if not request_path.is_relative_to(root) or not evidence.is_relative_to(root):
        raise ValueError("Population request and evidence must remain in the workspace")
    request = read_json(request_path)
    output = Path(request["output"]).resolve()
    bare = Path(request["bare_scene_directory"]).resolve()
    if not output.is_relative_to(root) or output == bare or output.is_relative_to(bare):
        raise ValueError("Population must not overwrite the immutable bare scene")
    if not request.get("assets"):
        raise ValueError("An explicit nonempty population is required")
    if any((output / n).exists() for n in ("world.usda", "content.usdc", "scene.blend", "export_result.json")):
        raise ValueError("Preserve existing population output; use a fresh destination")
    if evidence.exists() and any(evidence.iterdir()):
        raise ValueError("Population continuation evidence must use a fresh directory")
    evidence.mkdir(parents=True, exist_ok=True)
    cache = verify_bare(request, root, bare)
    assets = sorted({Path(a["blend"]).resolve() for a in request["assets"]})
    if any(not p.is_relative_to(root) or p.suffix != ".blend" or not p.is_file() for p in assets):
        raise ValueError("Only existing workspace native blend assets are accepted")
    inputs = [{"path": str(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}
              for p in [request_path, *assets]]
    resources = reuse_memory_estimate((bare / "scene.blend").stat().st_size,
                                     sum(p.stat().st_size for p in assets))
    record = {"schema_version": 1, "status": "running", "started_at_utc": utc_now(),
              "request_sha256": sha256_file(request_path), "inputs": inputs,
              "bare_cache_sha256": cache["manifest_sha256"], "resource_plan": resources,
              "exporter_sha256": sha256_file(root / "blender_scripts/export_scene.py"),
              "continuation_producer_sha256": sha256_file(Path(__file__)),
              "original_failed_build_modified": False, "quality_profile_modified": False,
              "geometry_reduction": False, "appearance_qualification": "not_run"}
    atomic_json(evidence / "continuation.json", record)
    process = run_worker([
        str(root / ".tools/blender/blender"), "--background", "--factory-startup",
        "--disable-autoexec", "--python-use-system-env", "--python-exit-code", "1",
        "--python", str(root / "blender_scripts/export_scene.py"), "--", "--request", str(request_path)],
        cwd=root, log_path=evidence / "worker.log", timeout=86400,
        environment=native_environment(root),
        estimated_memory_bytes=resources["estimated_memory_bytes"],
        estimated_disk_bytes=resources["estimated_disk_bytes"],
        shared_memory_reserve_bytes=resources["shared_memory_reserve_bytes"])
    record["actual_process"] = process
    record["completed_at_utc"] = utc_now()
    if process["exit_code"] or not (output / "export_result.json").is_file():
        record["status"] = "failed_native_export"
        atomic_json(evidence / "continuation.json", record)
        raise RuntimeError("Cached population failed; actual native evidence retained")
    exported = read_json(output / "export_result.json")
    if (exported.get("status") != "success" or not exported.get("native_usd_geometry_verified")
            or exported.get("final_ground_sha256") != cache["manifest"]["ground_sha256"]
            or not exported.get("bare_scene_reuse", {}).get("actual_payload_equality")):
        raise ValueError("Populated native export failed actual ground/cache equality")
    for item in inputs:
        if sha256_file(Path(item["path"])) != item["sha256"]:
            raise ValueError("Frozen population input changed during execution")
    if verify_bare(request, root, bare)["manifest_sha256"] != cache["manifest_sha256"]:
        raise ValueError("Bare cache changed during population")
    storage = deduplicate_generated_textures(output, root / ".cache/native_textures")
    atomic_json(output / "texture_storage.json", storage)
    record.update(status="native_population_exported", export_result={
        "path": str(output / "export_result.json"), "sha256": sha256_file(output / "export_result.json")},
        ground_sha256=exported["final_ground_sha256"],
        native_usd_geometry_verified=exported["native_usd_geometry_verified"],
        placed_native_objects=len(exported.get("placed_object_names", [])))
    atomic_json(evidence / "continuation.json", record)
    return record
