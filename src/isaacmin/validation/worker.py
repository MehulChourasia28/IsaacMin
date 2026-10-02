"""Isolated independent geometry checks with bounded shared-memory execution."""
from pathlib import Path

from ..io import atomic_json, read_json


KINDS = {"source_mesh", "source_connectivity", "source_fidelity", "capture_plan",
         "water_interfaces", "depth", "contact", "gravity", "traversal", "temporal", "ecology",
         "feature_visibility", "view_coverage", "provenance", "continuous_surface", "usd_structure"}


def measurement_completed(result):
    """Execution completeness only; preserve the measurement's exact verdict."""
    return result.get('status') in {'pass','fail','measured','measurements_pass',
        'measurements_complete','measurements_incomplete','incomplete'}


def execute(kind: str, parameters: dict, output: Path):
    if kind not in KINDS:
        raise ValueError("Unknown independent validation stage")
    parameters = {key: Path(value) if key.endswith("_path") or key.endswith("_dir") or key == "source_surface" else value
                  for key, value in parameters.items()}
    if kind == "source_mesh":
        from ..volumes.mesh_validation import validate_source_mesh
        return validate_source_mesh(output=output, **parameters)
    if kind == "source_connectivity":
        from ..volumes.mesh_validation import validate_source_connectivity
        return validate_source_connectivity(output=output, **parameters)
    if kind == "source_fidelity":
        from .source_fidelity import evaluate_source_fidelity
        return evaluate_source_fidelity(output=output, **parameters)
    if kind == "water_interfaces":
        from .water_interfaces import validate_water_interfaces
        return validate_water_interfaces(output=output, **parameters)
    if kind == "capture_plan":
        from .capture_plan import create_capture_plan
        from .cave_views import create_cave_view_plan
        from ..contracts import CoordinateFrame
        origin = tuple(parameters["origin"])
        plan = create_capture_plan(parameters["mesh_path"], parameters["ir_dir"] / "terrain_surface.npz",
                                   output / "capture_plan.json", origin=origin)
        cave = create_cave_view_plan(parameters["mesh_path"], parameters["ir_dir"], CoordinateFrame(origin),
            output / "cave_view_plan.json", support_manifest_path=parameters["support_manifest_path"])
        plan["poses_static"].extend(cave["poses_static"])
        plan["contact_probes"].extend(cave["contact_probes"])
        plan["missing_cave_contact_probes"] = cave["missing_contact_probes"]
        plan["support_routes"].extend(cave.get("support_routes", []))
        plan["missing_cave_support_routes"] = cave.get("missing_support_routes", ["Cave footprint route planning has not run"])
        plan["cave_features"] = cave["available_features"]
        plan["missing_cave_views"] = cave["missing_representative_features"]
        plan["limitations"] = ["A pose plan does not establish actual RGB, depth, contact or temporal quality"]
        atomic_json(output / "capture_plan.json", plan)
        return plan
    if kind == "depth":
        from .sensors import validate_capture_depth
        return validate_capture_depth(output_dir=output, **parameters)
    if kind == "contact":
        from .contact import validate_contact_rays
        return validate_contact_rays(output=output, **parameters)
    if kind == "gravity":
        from .contact import validate_dropped_probes
        return validate_dropped_probes(output=output, **parameters)
    if kind == "traversal":
        from .traversal import validate_traversal
        return validate_traversal(output=output, **parameters)
    if kind == "temporal":
        from .temporal import measure_motion
        return measure_motion(output=output, **parameters)
    if kind == "ecology":
        from ..assets.ecology_validation import collect_ecology
        return collect_ecology(output_directory=output, **parameters)
    if kind == 'usd_structure':
        from ..assets.usd_inspection import inspect_usd
        from ..io import sha256_file
        scene=Path(parameters['scene']);runtime=scene.parent/'runtime_dependencies.json'
        closure=scene.parent/'native_physics_dependency_closure.json'
        modules=[]
        if runtime.is_file():
            modules=[{'module':m['asset'],'path':m['resolved_path'],'sha256':m['sha256']}
                     for m in read_json(runtime)['modules']]
        result=inspect_usd(scene,output/'usd_inspection.json',runtime_modules=modules)
        result['dependency_closure_sha256']=sha256_file(closure) if closure.is_file() else None
        result['runtime_dependency_sha256']=sha256_file(runtime) if runtime.is_file() else None
        atomic_json(output/'usd_inspection.json',result)
        return result
    if kind == "feature_visibility":
        from .feature_visibility import collect_feature_visibility
        from ..contracts import CoordinateFrame
        parameters["coordinate_frame"] = CoordinateFrame(tuple(parameters.pop("origin")))
        # The subprocess request/log live in output; the collector owns a new,
        # immutable child directory containing its raw witnessed source labels.
        return collect_feature_visibility(output_dir=output / "observed", **parameters)
    if kind == "view_coverage":
        from ..assets.view_coverage import collect_view_coverage
        return collect_view_coverage(output_path=output / "view_coverage.json", **parameters)
    if kind == "provenance":
        from .provenance import audit_provenance
        return audit_provenance(output=output, **parameters)
    if kind == "continuous_surface":
        from .continuous_surface import validate_continuous_surface
        return validate_continuous_surface(output=output / "surface", **parameters)


def run_validation(workspace: Path, kind: str, output: Path, parameters: dict) -> dict:
    """Native scene arrays die with the worker instead of accumulating in the agent."""
    import sys
    from ..process import run_worker
    if kind not in KINDS:
        raise ValueError("Unknown independent validation stage")
    output.mkdir(parents=True, exist_ok=True)
    request = output / (kind+"_request.json")
    encoded = {k: str(v) if isinstance(v, Path) else v for k, v in parameters.items()}
    atomic_json(request, {"kind": kind, "parameters": encoded, "output": str(output.resolve())})
    result_path = output / (kind+"_result.json")
    if result_path.exists():
        raise ValueError("Independent validation output already exists; preserve the old attempt and choose a new directory")
    mesh = Path(parameters["mesh_path"]) if "mesh_path" in parameters else None
    if kind in {"ecology", "view_coverage", "usd_structure"}:
        mesh = Path(parameters["scene"]).parent / "final_ground.obj"
    estimate = max(4*2**30, mesh.stat().st_size*16 if mesh else 0)
    disk_estimate = 2*2**30
    if kind == 'source_fidelity':
        estimate=4*2**30
    elif mesh is not None and kind in {'source_mesh','source_connectivity','continuous_surface'}:
        from .native_ground import has_native_ground
        if has_native_ground(mesh):
            # Presence alone cannot select a cheaper resource declaration.
            # Verify the actual finalization-bound native array bytes first.
            native=mesh.parent/'native_precision'
            finalization=read_json(native/'finalization.json')
            from ..io import sha256_file
            if (sha256_file(native/'final_vertices.npy')!=finalization['final_vertices_sha256']
                    or sha256_file(native/'final_triangles.npy')!=finalization['final_triangles_sha256']):
                raise ValueError('Native finalization arrays changed before bounded validation')
            estimate=(32 if kind=='continuous_surface' else 12)*2**30
            if kind == 'continuous_surface':
                # Full original-face incidence uses temporary on-disk edge
                # records as well as the retained numerical measurements.
                import numpy as np
                triangles = np.load(native/'final_triangles.npy', mmap_mode='r', allow_pickle=False)
                disk_estimate = max(8*2**30, len(triangles)*128)
    environment = None
    if kind in {"ecology", "view_coverage", "usd_structure"}:
        from ..adapters.workers import native_environment
        environment = native_environment(workspace)
    process = run_worker([sys.executable, "-m", "isaacmin.validation.worker", "--request", str(request)],
        cwd=workspace, log_path=output / (kind+"_worker.log"), timeout=86400,
        estimated_memory_bytes=estimate, estimated_disk_bytes=disk_estimate, environment=environment)
    if process["exit_code"] != 0 or not result_path.is_file():
        raise RuntimeError("Independent validation failed to complete; inspect " + process["log"])
    return read_json(result_path)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = read_json(args.request)
    result = execute(request["kind"], request["parameters"], Path(request["output"]))
    atomic_json(Path(request["output"]) / (request["kind"]+"_result.json"), result)


if __name__ == "__main__":
    main()
