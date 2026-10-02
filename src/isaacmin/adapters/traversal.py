"""Run the actual Isaac gravity-driven support probe in an immutable attempt."""
from pathlib import Path

import numpy as np

from ..io import atomic_json, hash_object, read_json, sha256_file
from ..packaging import ascii_dependency_closure
from ..process import run_worker


def traverse_scene(workspace: Path, scene: Path, output: Path, *, route_ground_xyz,
                   surface_scope: str, speed_mps: float = .35, maximum_simulation_seconds: float | None = None) -> dict:
    route = np.asarray(route_ground_xyz, float)
    if route.ndim != 2 or route.shape[1] != 3 or len(route) < 2 or not np.isfinite(route).all():
        raise ValueError("Traversal requires a finite source-supported route")
    if surface_scope not in {"exterior", "cave_floor", "portal"}:
        raise ValueError("Traversal requires an explicit support surface scope")
    if not .05 <= speed_mps <= .75:
        raise ValueError("Traversal speed is outside the bounded worker profile")
    if output.exists() and any(output.iterdir()):
        raise ValueError("Preserve the previous traversal and choose a new output directory")
    output.mkdir(parents=True, exist_ok=True)
    scene = scene.resolve()
    closure = ascii_dependency_closure(scene)
    if closure["status"] != "pass":
        raise ValueError("Support traversal requires a dependency-closed scene")
    distance = float(np.linalg.norm(np.diff(route, axis=0), axis=1).sum())
    duration = maximum_simulation_seconds or max(30., 30.+distance/speed_mps*2)
    if not 2 <= duration <= 1800:
        raise ValueError("Split support routes longer than the bounded thirty-minute diagnostic")
    script = workspace / "isaac_scripts/traverse_support.py"
    request = {"scene": str(scene), "output": str(output.resolve()), "route_ground_xyz": route.tolist(),
               "surface_scope": surface_scope, "scope": "actual_final_ground_support_diagnostic",
               "speed_mps": speed_mps, "maximum_simulation_seconds": duration,
               "footprint_xyz_m": [.35, .25, .2], "scene_dependency_files": closure["files"],
               "scene_content_sha256": hash_object(closure["files"]),
               "final_ground_sha256": sha256_file(scene.parent / "final_ground.obj"),
               "native_script_sha256": sha256_file(script), "navigation_stack": "not_run_not_supplied"}
    path = output / "traversal_request.json"
    atomic_json(path, request)
    isaac = Path(read_json(workspace / "state/native_tools.json")["isaac_python"])
    process = run_worker([str(isaac), "--no-ros-env", str(script), "--request", str(path)],
        cwd=workspace, log_path=output / "worker.log", timeout=min(86400., max(1800., duration*30)),
        environment={"LD_LIBRARY_PATH": ""})
    result_path = output / "traversal_result.json"
    if process["exit_code"] != 0 or not result_path.is_file():
        return {"status": "fail", "reason": "Native support probe did not complete", "process": process}
    result = read_json(result_path)
    if result.get("request_sha256") != sha256_file(path) or result.get("final_ground_sha256") != request["final_ground_sha256"]:
        raise ValueError("Native support probe returned measurements for different inputs")
    return {**result, "process": process}
