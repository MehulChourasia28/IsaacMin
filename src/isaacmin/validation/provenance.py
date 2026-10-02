"""Read-only provenance audit of a candidate's actual source and export inputs.

This is separate from construction success. Missing original execution records
remain gaps even when their present-day reproduction scripts can be hashed.
"""
from pathlib import Path
import hashlib
import re

from ..io import atomic_json, code_hash, hash_object, read_json, sha256_file
from ..packaging import ascii_dependency_closure
from ..security import safe_path
from .profile import load_profile


def scan_delivery_file(path: Path, *, chunk_bytes=1024*1024) -> dict:
    """Hash and search bytes without emitting any matching credential value."""
    digest = hashlib.sha256()
    found = False
    tail = b""
    pattern = re.compile(rb"nvapi-[A-Za-z0-9_-]{20,}")
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
            found |= pattern.search(tail+chunk) is not None
            tail = (tail+chunk)[-128:]
    return {"sha256": digest.hexdigest(), "credential_pattern_found": found,
            "forbidden_filename": path.name.startswith(".env")}


def audit_provenance(workspace: Path, build_dir: Path, output: Path) -> dict:
    workspace, build_dir, output = map(lambda p: Path(p).resolve(), (workspace, build_dir, output))
    errors, gaps, files, licences = [], [], {}, []

    def bind(path, expected=None, *, role):
        path = Path(path)
        if not path.is_absolute():
            path = safe_path(workspace, str(path))
        path = path.resolve()
        if not path.is_relative_to(workspace) or not path.is_file():
            errors.append({"role": role, "reason": "missing_or_outside_workspace_file", "path": str(path)})
            return None
        digest = sha256_file(path)
        if expected is not None and digest != expected:
            errors.append({"role": role, "reason": "content_changed", "path": str(path)})
        files[str(path)] = {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}
        return path

    def document(path, expected=None, *, role):
        actual = bind(path, expected, role=role)
        return read_json(actual) if actual else {}

    manifest = document(build_dir / "build.json", role="build")
    profile_path = build_dir / "quality_profile.json"
    bind(profile_path, role="profile")
    profile = load_profile(profile_path)
    dependencies = manifest["dependencies"]
    for name, entry in manifest.get("input_files", {}).items():
        bind(entry["path"], entry["sha256"], role="construction_input:"+name)
    if not manifest.get("input_files"):
        gaps.append({"role": "build", "reason": "complete_named_input_file_inventory_not_recorded"})
    if profile["profile"]["context"]["dependencies"] != dependencies:
        errors.append({"role": "profile", "reason": "frozen_dependency_set_differs"})
    if code_hash(workspace) != dependencies["code"]:
        errors.append({"role": "code", "reason": "generation_code_changed_since_build"})

    snapshot_path = workspace / "work/snapshots" / dependencies["source"] / "snapshot.json"
    snapshot = document(snapshot_path, role="source_snapshot")
    if snapshot:
        if hash_object(snapshot["files"]) != dependencies["source"]:
            errors.append({"role": "source_snapshot", "reason": "inventory_hash_differs"})
        source_root = Path(snapshot["snapshot_path"])
        actual_names = {p.relative_to(source_root).as_posix() for p in source_root.rglob("*")
                        if p.is_file() and p.name != "session.lock"}
        if actual_names != {entry["path"] for entry in snapshot["files"]}:
            errors.append({"role": "source_snapshot", "reason": "snapshot_inventory_has_extra_or_missing_files"})
        for entry in snapshot["files"]:
            bind(safe_path(source_root, entry["path"]), entry["sha256"], role="source_payload")
    ir_dir = safe_path(workspace, manifest["source_ir"])
    ir = document(ir_dir / "world_ir.json", dependencies["world_ir"], role="world_ir")
    for entry in ir.get("files", []):
        bind(safe_path(ir_dir, entry["path"]), entry["sha256"], role="world_ir_payload")

    declared = {
        "tools": "artifacts/bootstrap/dependency-lock.json", "assets": "state/asset_catalogue.json",
        "normalized_assets": "state/normalized_assets.json", "procedural_canopy": "state/procedural_assets.json",
        "materials": "state/terrain_materials.json", "lighting": "state/lighting_candidates.json",
        "references": "state/reference_catalogue.json", "host": "artifacts/bootstrap/host.json"}
    documents = {key: document(workspace / path, dependencies.get(key), role=key)
                 for key, path in declared.items()}
    toolchain = documents["tools"]
    for entry in [*toolchain.get("native_binaries", []), *toolchain.get("runtime_material_modules", []),
                  *[toolchain[k] for k in ("native_build_lock", "orchestrator_requirements") if k in toolchain]]:
        bind(entry["path"], entry["sha256"], role="pinned_native_dependency")

    known_asset_ids = set()
    for asset in documents["assets"].get("assets", []):
        identity = asset["asset_id"]
        known_asset_ids.add(identity)
        if asset.get("licence") != "CC0-1.0" or not asset.get("source_page"):
            errors.append({"role": "asset", "asset_id": identity, "reason": "missing_eligible_licence_or_source"})
        licence_path = asset.get("licence_snapshot_path")
        if not licence_path or not asset.get("licence_text_sha256"):
            gaps.append({"role": "asset", "asset_id": identity, "reason": "missing_licence_snapshot"})
        else:
            bind(licence_path, asset["licence_text_sha256"], role="asset_licence")
        licences.append({"asset_id": identity, "licence": asset.get("licence"), "url": asset.get("licence_url")})
        for entry in asset.get("files", []):
            bind(entry["path"], entry["sha256"], role="acquired_asset")
    for asset in documents["normalized_assets"].get("assets", []):
        if asset["asset_id"] not in known_asset_ids:
            errors.append({"role": "normalized_asset", "asset_id": asset["asset_id"], "reason": "unbound_master_asset"})
        bind(asset["output_blend"], asset["output_sha256"], role="normalized_asset")
        preparation=asset.get('target_material_preparation')
        if preparation:
            from ..assets.target_materials import verify_prepared_asset
            try:
                if not verify_prepared_asset(workspace,asset):
                    raise ValueError('Prepared material producer changed')
                original=preparation['input_asset']
                bind(original['output_blend'],original['output_sha256'],role='original_normalized_asset')
                for role in ('report','process'):
                    entry=preparation[role]
                    document(entry['path'],entry['sha256'],role='material_preparation_'+role)
            except (ValueError,KeyError,OSError) as exc:
                errors.append({'role':'material_preparation','asset_id':asset['asset_id'],'reason':str(exc)})
    for asset in documents["procedural_canopy"].get("assets", []):
        known_asset_ids.add(asset["asset_id"])
        for entry in asset.get("files", []):
            bind(entry["path"], entry["sha256"], role="procedural_asset")
        if not asset.get("generator_execution_script_sha256"):
            gaps.append({"role": "procedural_asset", "asset_id": asset["asset_id"],
                         "reason": "original_generation_execution_script_hash_not_recorded"})
        generation_path = Path(asset["blend"]).parent / "generation.json"
        generation = document(generation_path, role="procedural_generator")
        if not generation.get("generator_licence") or not generation.get("generator_source") or not generation.get("source_hashes"):
            errors.append({"role": "procedural_generator", "reason": "missing_generator_licence_or_sources", "asset_id": asset["asset_id"]})
        licences.append({"asset_id": asset["asset_id"], "generator_licence": generation.get("generator_licence"),
                         "generated_geometry_is_scan": False})
    for reference in documents["references"].get("references", []):
        if reference.get("licence") != "CC0-1.0" or not reference.get("not_scene_evidence"):
            errors.append({"role": "reference", "reason": "unlicensed_or_misclassified_reference", "asset_id": reference.get("asset_id")})
        bind(reference["original_path"], reference["original_sha256"], role="original_reference")
        bind(reference["board"]["path"], reference["board"]["sha256"], role="derived_reference_board")

    # Bind the parameters actually consumed by native construction, including
    # the final instance catalogue. A recipe name alone cannot close provenance.
    native_export=manifest.get('stages',{}).get('scene',{})
    request_relative='data/populated_export/assembly_request.json' if native_export.get('post_export') else 'scene/assembly_request.json'
    for relative in (request_relative, "scene/export_result.json", "data/ecology/ecology_manifest.json",
                     "data/refinement/worker_result.json", "data/volume/worker_result.json"):
        document(build_dir / relative, role="executed_native_inputs")
    request_path=build_dir/request_relative
    request = read_json(request_path) if request_path.is_file() else {}
    if native_export.get('post_export'):
        post=native_export['post_export']
        assembly=document(build_dir/'data/populated_export/assembly_result.json',
            native_export.get('source_assembly_result_sha256'),role='original_native_assembly')
        bind(build_dir/'scene/blender_export_result.json',post['source_blender_export_sha256'],role='original_blender_export')
        sharing=document(post['instancing_evidence']['path'],post['instancing_evidence']['sha256'],role='lossless_native_instancing')
        if sharing:
            bind(sharing['source_scene'],sharing['source_scene_sha256'],role='uninstanced_source_stage')
            bind(Path(sharing['source_scene']).parent/'native_dependency_closure.json',sharing['source_closure_sha256'],role='uninstanced_source_closure')
            bind(Path(sharing['scene']).parent/'native_dependency_closure.json',sharing['closure_sha256'],role='instanced_source_closure')
        collision=document(build_dir/'scene/collision_preparation.json',role='exact_collision_preparation')
        if (post.get('exact_collision')!=collision or post.get('geometry_material_transform_reduction') is not False
                or sharing.get('status')!='lossless_native_instancing_verified'
                or collision.get('status')!='all_saved_collision_triangles_match_rendered_source'):
            errors.append({'role':'post_export','reason':'missing_or_changed_lossless_geometry_proof'})
        if str(request_path) not in assembly.get('argv',[]):
            errors.append({'role':'native_assembly','reason':'request_not_bound_to_executed_argument_array'})
    for asset in request.get("assets", []):
        if asset.get("asset_id") not in known_asset_ids:
            errors.append({"role": "exported_asset", "asset_id": asset.get("asset_id"), "reason": "unbound_exported_asset"})
    scene = build_dir / "scene/world_physics.usda"
    if not scene.exists():
        scene = build_dir / "scene/world.usda"
    closure = ascii_dependency_closure(scene) if scene.is_file() else {"status": "fail", "files": [], "errors": ["missing_scene"]}
    if closure["status"] != "pass":
        errors.append({"role": "native_scene", "reason": "unclosed_dependencies", "details": closure["errors"]})
    delivery_scans = []
    for entry in closure["files"]:
        path = safe_path(scene.parent, entry["path"], must_exist=True)
        scanned = scan_delivery_file(path)
        delivery_scans.append({"path": entry["path"], **scanned})
        if scanned["sha256"] != entry["sha256"] or scanned["credential_pattern_found"] or scanned["forbidden_filename"]:
            errors.append({"role": "delivery_payload", "reason": "changed_bytes_or_secret", "path": entry["path"]})
    result = {"status": "fail" if errors else "incomplete" if gaps else "pass",
              "method": "independent_declared_input_and_native_payload_audit_v1", "validator_sha256": sha256_file(Path(__file__)),
              "profile_sha256": profile["sha256"], "build_id": manifest["build_id"], "files": list(files.values()),
              "source_snapshot_sha256": dependencies["source"], "native_closure": closure,
              "delivery_byte_scan": delivery_scans, "licences": licences, "errors": errors, "gaps": gaps,
              "limits": ["Credential scan recognizes NVIDIA API token syntax and forbidden local-secret filenames; no claim to detect every possible secret format",
                         "Licence declarations and acquired licence snapshots are checked; no source-save redistribution rights are inferred",
                         "This audit does not qualify material appearance, ecology or runtime performance"]}
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "provenance.json", result)
    return result
