"""Freeze and execute portable probes using only bundled scripts and scene data."""
from pathlib import Path
import shutil

from .io import atomic_json, read_json, sha256_file
from .security import safe_path


def freeze_probe(workspace: Path, scene: Path, capture_dir: Path, destination: Path) -> dict:
    """Package actual native measurements; never invent a reference result."""
    record = read_json(capture_dir / "capture_result.json")
    reopened = read_json(capture_dir / "reopen_result.json")
    if (record.get("physics_scene_sha256") != sha256_file(scene)
            or reopened.get("scene_sha256") != sha256_file(scene)
            or reopened.get("status") != "pass"):
        raise ValueError("Portable reference must belong to this exact, freshly reopened physics scene")
    frames = record["frames"]
    if not frames:
        raise ValueError("Portable reference needs actual captured frames")
    # A bounded representative set, including a held-out view when available.
    selected = {0, len(frames)//2, len(frames)-1}
    selected.add(next((i for i, f in enumerate(frames) if f["pose"].get("held_out")), 0))
    target = destination / "reproduction"
    target.mkdir()
    reference = target / "reference"
    reference.mkdir()
    copied = []

    def copy(source, name, expected):
        if sha256_file(source) != expected:
            raise ValueError("Portable reference bytes changed: " + name)
        dest = safe_path(reference, name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        copied.append({"path": name, "sha256": expected, "bytes": dest.stat().st_size})

    chosen = []
    for index in sorted(selected):
        frame = frames[index]
        for role in ("rgb", "depth", "instance_segmentation"):
            copy(safe_path(capture_dir, frame[role], must_exist=True), frame[role], frame[role+"_sha256"])
        chosen.append(frame)
    rays = read_json(capture_dir / "physx_contact_rays.json")
    ray_file = Path(rays["samples"] if "samples" in rays else rays["path"]).name
    ray_path = capture_dir / ray_file
    copy(ray_path, ray_file, rays.get("samples_sha256", rays.get("sha256")))
    for name in ("capture_scene.py", "material_log.py", "ground_collision.py", "contact_rays.py", "reopen_scene.py", "reproduce_package.py", "compare_reproduction.py"):
        shutil.copyfile(workspace / "isaac_scripts" / name, target / name)
    settings = record.get("lighting_parameters", {})
    # Replay the full original sequence: skipped views alter renderer history.
    # Only the representative references need to occupy the portable package.
    request = {"scene": scene.name, "poses": [frame["pose"] for frame in frames],
               "contact_probes": read_json(capture_dir / "capture_request.json")["contact_probes"],
               "resolution": record["resolution"], "lighting": record["lighting"],
               "lighting_parameters": {"camera_default": settings.get("camera_default", {})},
               "hdri": None, "preserve_authored_scene": True,
               "renderer_recipe": record.get("renderer_recipe", "legacy_rtx_8"),
               "scope": "portable_saved_scene_reproduction_not_navigation"}
    atomic_json(target / "request.json", request)
    contract = {"schema_version": 1, "source_capture_sha256": sha256_file(capture_dir / "capture_result.json"),
                "scene_sha256": sha256_file(scene), "runtime_version": reopened["native_isaac_version"],
                "renderer_recipe": record.get("renderer_recipe", "legacy_rtx_8"),
                "native_renderer_settings": record.get("native_renderer_settings", {}),
                "frames": chosen, "replay_frame_indices": sorted(selected), "expected_frame_count": len(frames),
                "contacts": record["contacts"], "ray_samples": ray_file,
                "files": copied, "required_contact_ray_count": 10000,
                "thresholds": {"rgb_mean_codes_max": .5, "rgb_p999_codes_max": 2,
                               "depth_p999_m_max": .001, "depth_max_m": .01,
                               "contact_position_m_max": .001, "contact_orientation_degrees_max": .1,
                               "contact_ray_m_max": .001, "camera_matrix_abs_max": 1e-6},
                "limits": ["Reproduction checks preservation of the frozen scene, not its realism or source fidelity",
                           "No supplied navigation controller is run"]}
    atomic_json(target / "reference.json", contract)
    return {"status": "frozen", "request": "reproduction/request.json",
            "reference": "reproduction/reference.json", "frames": len(chosen),
            "reference_sha256": sha256_file(target / "reference.json"),
            "execution": "not_run", "navigation_stack": "not_run_not_supplied"}


def verify_portable(workspace: Path, package: Path, output: Path) -> dict:
    from .process import run_worker
    from .packaging import verify_package_files
    package = package.resolve()
    if not package.is_relative_to(workspace.resolve()):
        raise ValueError("Package must be inside the workspace")
    manifest = read_json(package / "package.json")
    if not verify_package_files(package, manifest["files"]):
        raise ValueError("Package bytes differ from their immutable inventory")
    if not manifest.get("reproduction", {}).get("reference"):
        return {"status": "blocked", "reason": "Package has no captured reproduction reference"}
    if output.exists():
        raise ValueError("Choose a new output directory; earlier reproduction attempts are immutable")
    output.mkdir(parents=True)
    runtime = Path(read_json(workspace / "state/native_tools.json")["isaac_python"])
    process = run_worker(["/usr/bin/python3", str(package / "reproduction/reproduce_package.py"),
                          "--package", str(package), "--isaac-python", str(runtime), "--output", str(output)],
                         cwd=output, log_path=output / "worker.log", timeout=7200,
                         estimated_memory_bytes=16*2**30,
                         estimated_disk_bytes=sum(f["bytes"] for f in manifest["files"])*2,
                         environment={"LD_LIBRARY_PATH": ""})
    path = output / "reproduction_comparison.json"
    if not path.is_file():
        return {"status": "failed", "reason": "Portable native execution failed", "process": process,
                "comparison": None}
    result = read_json(path)
    result["evidence"] = str(path)
    result["process_exit_code"] = process["exit_code"]
    build_evidence = workspace / "worlds" / manifest["build_id"] / "evidence"
    if output.resolve().is_relative_to(build_evidence.resolve()):
        atomic_json(build_evidence / "portable_delivery_pointer.json", {
            "report": path.resolve().relative_to(build_evidence.resolve()).as_posix(), "sha256": sha256_file(path)})
    return result
