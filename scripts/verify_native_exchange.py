#!/usr/bin/env python3
"""Independent measurements of the current technical fixture, never map quality."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json

from isaacmin.adapters.workers import probe_workers, project_root
from isaacmin.io import atomic_json, sha256_file
from isaacmin.packaging import ascii_dependency_closure
from isaacmin.validation.worker import run_validation


def main():
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or root / "artifacts/bootstrap" / (
        "independent_exchange_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    base = root / "artifacts/bootstrap"
    mesh = base / "blender/final_ground.obj"
    capture = base / "cross_isaac"
    measurements = {}
    for kind in ("depth", "contact", "gravity"):
        parameters = {"mesh_path": mesh, "capture_dir": capture}
        if kind == "contact":
            parameters.update(source_surface=base / "cross_ir/terrain_surface.npz", origin=[0, 0, 0])
        measurements[kind] = run_validation(root, kind, output / kind, parameters)
    measurements["portable_render_closure"] = ascii_dependency_closure(base / "blender/world.usda")
    measurements["portable_physics_closure"] = ascii_dependency_closure(base / "blender/world_physics.usda")
    current = probe_workers(root)
    measurements["current_native_chain"] = {"status": "pass" if current["cross_runtime_qualified"] else "fail",
        "native_exchange_status": current["native_exchange"]["status"]}
    result = {"status": "pass" if all(x["status"] == "pass" for x in measurements.values()) else "fail",
        "scope": "explicit_synthetic_native_exchange_with_real_acquired_fern_and_source_style_water",
        "measurements": measurements, "native_chain_sha256": sha256_file(base / "cross_runtime.json"),
        "validator_script_sha256": sha256_file(Path(__file__)),
        "actual_map_quality": "not_qualified; known exhausted native geometry failure remains separate",
        "navigation_stack": "not_supplied_not_tested"}
    atomic_json(output / "independent_exchange.json", result)
    print(json.dumps({"status": result["status"], "evidence": str(output / "independent_exchange.json")}))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
