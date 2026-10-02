"""Open a self-contained IsaacMin package using the pinned Isaac Python runtime.

Run with that installation's python.sh, not the generator virtual environment.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            value.update(block)
    return value.hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--package", type=Path, default=Path(__file__).resolve().parent)
parser.add_argument("--headless", action="store_true")
parser.add_argument("--frames", type=int, default=0, help="Zero keeps an interactive viewer running")
parser.add_argument("--report", type=Path)
args, _ = parser.parse_known_args()
root = args.package.resolve()
manifest = json.loads((root / "package.json").read_text())
names = [record["path"] for record in manifest["files"]]
actual_names = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
if not names or len(set(names)) != len(names) or actual_names != set(names) | {"package.json"}:
    raise RuntimeError("Package file inventory differs")
for record in manifest["files"]:
    unresolved = root / record["path"]
    if any(path.is_symlink() for path in [unresolved, *unresolved.parents] if path.is_relative_to(root)):
        raise RuntimeError("Package contains a symbolic link")
    path = unresolved.resolve()
    if (not path.is_relative_to(root) or not path.is_file() or path.stat().st_size != record["bytes"]
            or digest(path) != record["sha256"]):
        raise RuntimeError("Package verification failed: " + record["path"])
scene = (root / manifest["entrypoint"]).resolve()
if not scene.is_relative_to(root):
    raise RuntimeError("Invalid package entrypoint")
if args.headless and args.frames <= 0:
    args.frames = 120
print("IsaacMin qualification:", manifest["status"])
reference_path = root / "reproduction/reference.json"
reference = json.loads(reference_path.read_text()) if reference_path.is_file() else {}
config_path = root / "render_configuration.json"
render_config = json.loads(config_path.read_text()) if config_path.is_file() else {}
viewer_renderer = "PathTracing" if reference.get("renderer_recipe",render_config.get("renderer_recipe")) == "pathtracing_1024" else "RayTracedLighting"
from isaacsim import SimulationApp
app = SimulationApp({"headless": args.headless, "renderer": viewer_renderer})
report = {"status": "failed", "entrypoint": scene.name, "package_sha256": manifest["package_sha256"],
          "package_byte_verification": "pass", "qualification": manifest["status"],
          "navigation_stack": "not_run_not_supplied", "contact_and_capture_reproduction": "not_run"}
try:
    import omni.usd
    import carb.tokens
    import carb.settings
    from pxr import Usd, UsdGeom, UsdUtils, UsdLux
    if not omni.usd.get_context().open_stage(str(scene)):
        raise RuntimeError("Isaac could not open the packaged USD")
    for _ in range(30):
        app.update()
    stage = omni.usd.get_context().get_stage()
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(str(scene))
    runtime_assets = set()
    runtime_modules = manifest.get("runtime_modules", [])
    if runtime_modules:
        kit = Path(carb.tokens.get_tokens_interface().resolve('${kit}'))
        version_file = kit.parent / "VERSION"
        actual_version = version_file.read_text().strip() if version_file.is_file() else None
        runtime_name = ("runtime_dependencies_" + scene.stem + ".json"
                        if scene.stem.startswith("world_physics_") else "runtime_dependencies.json")
        runtime = json.loads((root / runtime_name).read_text())
        if not actual_version or runtime["runtime_version"] != actual_version:
            raise RuntimeError("Package requires a different pinned Isaac runtime")
        for module in runtime_modules:
            if module["asset"] != "OmniSurface.mdl" or module["runtime_version"] != actual_version:
                raise RuntimeError("Unrecognized or mismatched runtime material module")
            candidates = [kit / "mdl/core/Base" / module["asset"], kit / "mdl/core" / module["asset"],
                          kit / "mdl" / module["asset"]]
            local_module = next((p for p in candidates if p.is_file()), None)
            if local_module is None or digest(local_module) != module["sha256"]:
                raise RuntimeError("Pinned runtime material differs: " + module["asset"])
            runtime_assets.add(module["asset"])
        transitive = runtime.get("transitive_module_files", [])
        if not transitive:
            raise RuntimeError("Missing transitive runtime material provenance")
        for module in transitive:
            original = module["path"]
            if "/mdl/" not in original:
                raise RuntimeError("Unrecognized material module search scope")
            local_module = (kit / "mdl" / original.split("/mdl/", 1)[1]).resolve()
            if not local_module.is_relative_to(kit.resolve()) or not local_module.is_file() or digest(local_module) != module["sha256"]:
                raise RuntimeError("Transitive runtime material differs")
        report.update(pinned_runtime_version=actual_version, runtime_material_byte_verification="pass",
                      transitive_material_modules=len(transitive))
    missing = [str(path) for path in unresolved if str(path) not in runtime_assets]
    if missing:
        raise RuntimeError("Unresolved package dependencies: " + repr(missing))
    meshes = [str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh)]
    if not meshes:
        raise RuntimeError("No native mesh geometry in package")
    # Native exposure is a runtime setting rather than a USD camera attribute.
    # Restore the recorded controls so the interactive viewer uses the actual
    # capture recipe. A viewer is still not a repetition of the probe sequence.
    request_path = root / "reproduction/request.json"
    reference_path = root / "reproduction/reference.json"
    if request_path.is_file() and reference_path.is_file():
        request = json.loads(request_path.read_text())
        reference = json.loads(reference_path.read_text())
        recipe = request.get("lighting_parameters", {}).get("camera_default", {})
        controls = {key: value for key, value in reference.get("native_renderer_settings", {}).items()
                    if value is not None}
        if recipe:
            controls.update({"/rtx/post/histogram/enabled": bool(recipe.get("auto_exposure", False)),
                "/rtx/post/motionblur/enabled": bool(recipe.get("motion_blur", False)),
                "/rtx/post/tonemap/filmIso": float(recipe["iso"]),
                "/rtx/post/tonemap/fNumber": float(recipe["f_number"]),
                "/rtx/post/tonemap/exposureTime": float(recipe["shutter_seconds"]),
                "/rtx/post/tonemap/whitepoint": list(UsdLux.BlackbodyTemperatureAsRgb(
                    float(recipe.get("white_balance_kelvin", 5500))))})
        settings = carb.settings.get_settings()
        for key, value in controls.items():
            if settings.get(key) is None:
                raise RuntimeError("Pinned runtime lacks the recorded viewer control: " + key)
            settings.set(key, value)
        report["viewer_controls_readback"] = {key: settings.get(key) for key in controls}
    if render_config.get('initial_camera_response'):
        response = render_config['initial_camera_response']
        settings = carb.settings.get_settings()
        controls = {'/rtx/post/histogram/enabled': False,
            '/rtx/post/tonemap/filmIso': float(response.get('iso',100)),
            '/rtx/post/tonemap/fNumber': float(response['f_number']),
            '/rtx/post/tonemap/exposureTime': float(response['f_number'])**2*100/float(response.get('iso',100))/2**float(response['ev100'])}
        for key,value in controls.items():
            if settings.get(key) is None:raise RuntimeError('Pinned viewer lacks requested camera response')
            settings.set(key,value)
        report.setdefault('viewer_controls_readback',{}).update({key:settings.get(key) for key in controls})
    cameras = [str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdGeom.Camera)
               and str(prim.GetPath()).startswith("/Replicator/")]
    if render_config.get('initial_camera'):
        initial = stage.GetPrimAtPath(render_config['initial_camera'])
        if not initial or not initial.IsA(UsdGeom.Camera):raise RuntimeError('Saved outdoor camera is missing')
        cameras = [render_config['initial_camera']]
    if cameras:
        import omni.kit.viewport.utility
        viewport = omni.kit.viewport.utility.get_active_viewport()
        report["saved_capture_camera"] = cameras[0]
        if viewport is not None:
            viewport.set_active_camera(cameras[0])
            report["active_viewport_camera"] = str(viewport.camera_path)
        else:
            report["active_viewport_camera"] = "not_available_in_headless_mode"
    count = 0
    while app.is_running() and (not args.frames or count < args.frames):
        app.update()
        count += 1
    report.update(status="opened", native_mesh_prims=len(meshes), updated_frames=count,
                  source_save_access_required=False, unresolved_dependencies=missing,
                  pinned_runtime_modules=manifest.get("runtime_modules", []),
                  limitation="Opening alone does not reproduce required contact and sensor acceptance probes")
except BaseException as exc:
    report["error"] = type(exc).__name__ + ": " + str(exc)
    raise
finally:
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2)+"\n")
    app.close()
