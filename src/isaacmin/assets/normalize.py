"""Validated native asset normalization behind a secret-free Blender process."""
from pathlib import Path
import json
import os
import shutil
import uuid

from isaacmin.process import run_worker
from .network import ServiceError, atomic_json, digest


def normalize_asset(workspace: Path, record: dict, blender: Path, environment: dict | None = None):
    workspace = Path(workspace).resolve()
    files = [f for f in record["files"] if f.get("role") == "blend"]
    if len(files) != 1:
        raise ServiceError("unavailable_variant", "Normalization requires exactly one acquired Blender master")
    source = Path(files[0]["path"])
    for master in record['files']:
        if digest(Path(master['path'])) != master['sha256']:
            raise ServiceError("corrupt_master", "Master Blender or declared dependency checksum changed")
    folder = workspace / "assets/normalized" / record["asset_id"]
    staging = workspace / "assets/normalized/.staging" / str(uuid.uuid4())
    staging.mkdir(parents=True)
    request = {"asset_id": record["asset_id"], "source_blend": str(source), "source_sha256": files[0]["sha256"],
               "allowed_images": {str(Path(f['path']).resolve()):f['sha256'] for f in record['files'] if f.get('role') == 'model_dependency'},
               "output_blend": str(staging / "normalized.blend"), "output_report": str(staging / "normalization.json")}
    atomic_json(staging / "request.json", request)
    result = run_worker([str(Path(blender).resolve()), "-b", "--factory-startup", "--disable-autoexec", "--python-exit-code", "1", "--python",
                         str(workspace / "blender_scripts/normalize_assets.py"), "--", str(staging / "request.json")],
                        cwd=workspace, log_path=staging / "normalization.log", timeout=600, environment=environment)
    if result["exit_code"] != 0 or not (staging / "normalization.json").is_file() or not (staging / "normalized.blend").is_file():
        raise ServiceError("normalization_failed", "Blender normalization failed; staging and process log retained")
    report = json.loads((staging / "normalization.json").read_text())
    if not report.get("objects") or report.get("source_sha256") != files[0]["sha256"]:
        raise ServiceError("invalid_worker_output", "Native asset artifact has invalid provenance or empty geometry")
    for obj in report["objects"]:
        if obj["triangles"] <= 0 or abs(obj["normalized_contact_z"]) > 1e-5:
            raise ServiceError("invalid_contact_anchor", "Native prototype lacks a valid local base anchor")
    report["output_sha256"] = digest(staging / "normalized.blend")
    report["output_blend"] = str(folder / "normalized.blend")
    report["normalizer_sha256"] = digest(workspace / "blender_scripts/normalize_assets.py")
    report["process"] = result
    atomic_json(staging / "normalization.json", report)
    # Preserve existing candidate until the new artifact has validated.
    if folder.exists():
        archived = workspace / "assets/normalized/.history" / (record["asset_id"] + "-" + uuid.uuid4().hex)
        archived.parent.mkdir(parents=True, exist_ok=True)
        os.replace(folder, archived)
    folder.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, folder)
    return report
