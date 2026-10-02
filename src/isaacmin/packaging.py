"""Portable scene packaging, with explicit unqualified labels and secret exclusion."""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path

from .io import atomic_json, hash_object, read_json, sha256_file, utc_now
from .security import safe_path


def ascii_dependency_closure(scene: Path) -> dict:
    root = scene.parent.resolve()
    pending, visited, errors = [scene.resolve()], set(), []
    files = []
    variant = scene.stem.startswith("world_physics_")
    runtime_manifest = root / ("runtime_dependencies_"+scene.stem+".json" if variant else "runtime_dependencies.json")
    runtime_modules = {}
    if runtime_manifest.exists():
        runtime = read_json(runtime_manifest)
        if runtime.get("status") == "pass":
            for module in runtime.get("modules", []):
                # These are pinned Isaac standard-library modules, not arbitrary
                # missing assets permitted by a user-controlled filename list.
                if (module.get("asset") == "OmniSurface.mdl" and module.get("sha256")
                        and module.get("runtime_version") not in {None, "", "not_available"}):
                    resolved = Path(module.get("resolved_path", ""))
                    transitive = runtime.get("transitive_module_files", [])
                    transitive_valid = bool(transitive) and all(Path(item["path"]).is_file()
                        and sha256_file(Path(item["path"])) == item["sha256"] for item in transitive)
                    if (transitive_valid and resolved.is_file() and resolved.name == module["asset"]
                            and sha256_file(resolved) == module["sha256"]):
                        runtime_modules[module["asset"]] = module
    used_runtime_modules = {}
    native_record_path = root / ("native_"+scene.stem+"_dependency_closure.json" if variant else
                                "native_physics_dependency_closure.json" if scene.name == "world_physics.usda"
                                else "native_dependency_closure.json")
    native_files = {}
    if native_record_path.exists():
        native = read_json(native_record_path)
        try:
            native_root = safe_path(root, native["root_asset"], must_exist=True)
            valid_native = (native_root == scene.resolve()
                            and native.get("status") == "pass" and native.get("method") == "UsdUtils.ComputeAllDependencies"
                            and native.get("unresolved_paths") == [] and native.get("absolute_asset_paths") == []
                            and sha256_file(native_root) == native["root_sha256"])
            if native.get("runtime_dependencies_manifest"):
                declared_runtime = safe_path(root, native["runtime_dependencies_manifest"], must_exist=True)
                valid_native &= (declared_runtime == runtime_manifest
                                 and sha256_file(declared_runtime) == native.get("runtime_dependencies_sha256"))
            for entry in native.get("files", []):
                item = safe_path(root, entry["path"], must_exist=True)
                valid_native &= item.stat().st_size == entry["bytes"] and sha256_file(item) == entry["sha256"]
            if valid_native:
                native_files = {entry["path"]: entry for entry in native["files"]}
        except (KeyError, OSError, ValueError):
            native_files = {}
    used_native_closure = False
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        visited.add(path)
        if not path.is_relative_to(root) or path.is_symlink():
            errors.append({"path": str(path), "reason": "dependency_outside_scene"})
            continue
        if not path.is_file():
            errors.append({"path": str(path), "reason": "missing_dependency"})
            continue
        files.append({"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path), "bytes": path.stat().st_size})
        if path.suffix.lower() in {".usd", ".usda", ".usdc", ".usdz"}:
            with path.open("rb") as handle:
                prefix = handle.read(16)
            if not prefix.startswith(b"#usda"):
                if path.relative_to(root).as_posix() in native_files:
                    used_native_closure = True
                    for name, entry in native_files.items():
                        dependency = safe_path(root, name, must_exist=True)
                        if dependency not in visited:
                            visited.add(dependency)
                            files.append(entry)
                else:
                    errors.append({"path": str(path), "reason": "binary_usd_requires_verified_native_usdutils_closure"})
                continue
            content = path.read_text()
            # Asset paths in exported ASCII USD; native clean-session reopen is still required.
            for ref in re.findall(r"@([^@]+)@", content):
                if ref in runtime_modules:
                    used_runtime_modules[ref] = runtime_modules[ref]
                    continue
                if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", ref) or Path(ref).is_absolute():
                    errors.append({"path": ref, "reason": "absolute_or_remote_asset_path"})
                    continue
                resolved = (path.parent / ref).resolve()
                pending.append(resolved)
    if used_runtime_modules:
        files.append({"path": runtime_manifest.name, "sha256": sha256_file(runtime_manifest),
                      "bytes": runtime_manifest.stat().st_size})
    if used_native_closure:
        files.append({"path": native_record_path.name, "sha256": sha256_file(native_record_path),
                      "bytes": native_record_path.stat().st_size})
    return {"status": "pass" if not errors else "fail", "files": sorted(files, key=lambda f: f["path"]),
            "runtime_modules": list(used_runtime_modules.values()),
            "errors": errors, "method": "ASCII asset dependency traversal plus hashed pinned runtime modules; native reopen remains Q12"}


def package_build(workspace: Path, build_id: str, *, require_qualified: bool,
                  reference_capture: Path | None = None) -> dict:
    from .validation.runner import validate_build
    directory = safe_path(workspace / "worlds", build_id)
    qualification = validate_build(workspace, build_id)
    if require_qualified and qualification["stage"] != "autonomously_qualified":
        return {"status": "blocked", "reason": "All applicable frozen gates must pass before qualified packaging",
                "build_id": build_id, "stage": qualification["stage"]}
    scene = directory / "scene/world_physics.usda"
    if not scene.exists():
        scene = directory / "scene/world.usda"
    if not scene.exists():
        return {"status": "blocked", "reason": "No actual USD world to package", "build_id": build_id}
    closure = ascii_dependency_closure(scene)
    if closure["status"] != "pass":
        return {"status": "blocked", "reason": "Scene dependency closure failed", "closure": closure}
    destination = workspace / "packages" / (build_id + ("_qualified" if require_qualified else "_unqualified"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".package-", dir=destination.parent))
    try:
        for record in closure["files"]:
            source = safe_path(scene.parent, record["path"], must_exist=True)
            target = safe_path(staging, record["path"])
            if source.name.startswith(".env"):
                raise ValueError("Secrets cannot be packaged")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        stable_qualification = {k: v for k, v in qualification.items() if k not in {"created_at_utc", "report"}}
        atomic_json(staging / "qualification.json", stable_qualification)
        atomic_json(staging / "dependency_closure.json", closure)
        shutil.copyfile(directory / "quality_profile.json", staging / "quality_profile.json")
        credits = workspace / "ASSET_CREDITS.md"
        if credits.exists():
            shutil.copyfile(credits, staging / "ASSET_CREDITS.md")
        loader = workspace / "isaac_scripts/load_world.py"
        if loader.exists():
            shutil.copyfile(loader, staging / "load_world.py")
        lock = workspace / "artifacts/bootstrap/dependency-lock.json"
        if lock.exists():
            shutil.copyfile(lock, staging / "dependency-lock.json")
        reference_capture = reference_capture or directory / "evidence/isaac_static"
        reproduction = {"status": "not_run", "reason": "No actual reference capture has been recorded"}
        if (reference_capture / "capture_result.json").is_file():
            from .portable import freeze_probe
            reproduction = freeze_probe(workspace, scene, reference_capture, staging)
        # Scan copied binary and text bytes, not just filenames. A package can
        # never claim credential exclusion merely because .env was omitted.
        from .validation.provenance import scan_delivery_file
        scanned_files = []
        for path in sorted(staging.rglob("*")):
            if not path.is_file():
                continue
            scan = scan_delivery_file(path)
            if scan["credential_pattern_found"] or scan["forbidden_filename"]:
                raise ValueError("Delivery contains a forbidden credential or secret filename")
            scanned_files.append({"path": path.relative_to(staging).as_posix(), "sha256": scan["sha256"],
                                  "bytes": path.stat().st_size})
        manifest = {"build_id": build_id, "stage": qualification["stage"],
                    "entrypoint": scene.name,
                    "runtime_modules": closure.get("runtime_modules", []),
                    "status": "qualified" if require_qualified else "unqualified_development_artifact",
                    "files": scanned_files, "reproduction": reproduction,
                    "source_save_included": False, "credentials_included": False,
                    "credential_scan_scope": "NVIDIA token syntax in all packaged bytes and forbidden .env filenames",
                    "native_clean_reopen": qualification["gates"][12]["status"]}
        # Identity uses stable scene bytes/profile/gate results, not report timestamps.
        manifest["package_sha256"] = hash_object({"files": manifest["files"], "entrypoint": scene.name,
            "runtime_modules": manifest["runtime_modules"], "status": manifest["status"]})
        atomic_json(staging / "package.json", manifest)
        if destination.exists():
            old = read_json(destination / "package.json")
            if old["package_sha256"] != manifest["package_sha256"]:
                destination = destination.with_name(destination.name + "_" + manifest["package_sha256"][:10])
            else:
                if verify_package_files(destination, manifest["files"]):
                    return {"status": "pass", "package": str(destination), "stage": qualification["stage"]}
                # Preserve the damaged package for inspection and publish a new
                # verified copy; matching metadata alone never establishes reuse.
                destination = destination.with_name(destination.name + "_repaired_" + utc_now().replace(":", "").replace(".", ""))
        if destination.exists():
            if verify_package_files(destination, manifest["files"]):
                return {"status": "pass", "package": str(destination), "stage": qualification["stage"]}
            destination = destination.with_name(destination.name + "_repaired_" + utc_now().replace(":", "").replace(".", ""))
        os.rename(staging, destination)
        return {"status": "pass", "package": str(destination), "stage": qualification["stage"],
                "package_sha256": manifest["package_sha256"]}
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def verify_package_files(directory: Path, expected_files: list[dict]) -> bool:
    try:
        existing = read_json(directory / "package.json")
        names = [entry["path"] for entry in expected_files]
        if not names or len(names) != len(set(names)) or existing["files"] != expected_files:
            return False
        actual_names = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
        if actual_names != set(names) | {"package.json"}:
            return False
        for entry in expected_files:
            if any(p.is_symlink() for p in [directory / entry["path"], *(directory / entry["path"]).parents]
                   if p.is_relative_to(directory)):
                return False
            path = safe_path(directory, entry["path"], must_exist=True)
            if path.is_symlink() or path.stat().st_size != entry["bytes"] or sha256_file(path) != entry["sha256"]:
                return False
        return True
    except (KeyError, ValueError, OSError):
        return False
