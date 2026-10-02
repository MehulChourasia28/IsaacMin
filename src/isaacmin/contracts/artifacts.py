from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

from ..io import atomic_json, hash_object, read_json, sha256_file, utc_now
from ..security import safe_path


class ArtifactError(ValueError):
    pass


@dataclass(frozen=True)
class FileRecord:
    path: str
    byte_size: int
    sha256: str
    role: str

    @classmethod
    def capture(cls, root: Path, path: Path, role: str = "data") -> "FileRecord":
        relative = path.relative_to(root).as_posix()
        safe_path(root, relative, must_exist=True)
        if path.is_symlink():
            raise ArtifactError("Artifact files cannot be symlinks")
        if path.name.startswith(".env"):
            raise ArtifactError("Secrets cannot be packaged")
        return cls(relative, path.stat().st_size, sha256_file(path), role)


@dataclass(frozen=True)
class Artifact:
    schema_version: str
    artifact_id: str
    kind: str
    build_id: str
    scope: dict
    created_at_utc: str
    producer: dict
    input_hashes: dict
    parameter_hash: str
    toolchain_hash: str
    quality_profile_hash: str
    content_sha256: str
    files: list[dict]
    coordinate_frame: dict
    status: str
    diagnostics: list


def seal_directory(staging: Path, *, kind: str, build_id: str, scope: dict,
                   inputs: dict, parameters: dict, toolchain_hash: str,
                   profile_hash: str, coordinate_frame: dict) -> Artifact:
    files = [asdict(FileRecord.capture(staging, p)) for p in sorted(staging.rglob("*"))
             if p.is_file() and p.name != "artifact.json"]
    if not files:
        raise ArtifactError("Cannot seal an empty artifact")
    content_hash = hash_object(files)
    artifact = Artifact("1.0", content_hash, kind, build_id, scope, utc_now(),
                        {"name": "isaacmin", "version": "0.1.0"}, inputs,
                        hash_object(parameters), toolchain_hash, profile_hash,
                        content_hash, files, coordinate_frame, "validated", [])
    atomic_json(staging / "artifact.json", asdict(artifact))
    verify_artifact(staging)
    return artifact


def verify_artifact(root: Path) -> dict:
    data = read_json(root / "artifact.json")
    required = set(Artifact.__dataclass_fields__)
    if required != set(data) or data["schema_version"] != "1.0":
        raise ArtifactError("Unsupported or incomplete artifact envelope")
    if data["status"] not in {"validated", "promoted"}:
        raise ArtifactError("Artifact is not validated")
    files = data["files"]
    if not files or len({r["path"] for r in files}) != len(files):
        raise ArtifactError("Empty or duplicate artifact files")
    for record in files:
        path = safe_path(root, record["path"], must_exist=True)
        if path.stat().st_size != record["byte_size"] or sha256_file(path) != record["sha256"]:
            raise ArtifactError("Artifact content hash mismatch: " + record["path"])
    if hash_object(files) != data["content_sha256"]:
        raise ArtifactError("Artifact index hash mismatch")
    return data


def promote(staging: Path, destination: Path) -> dict:
    data = verify_artifact(staging)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = verify_artifact(destination)
        if existing["content_sha256"] != data["content_sha256"]:
            raise ArtifactError("Refusing to overwrite a different immutable artifact")
        return existing
    data["status"] = "promoted"
    atomic_json(staging / "artifact.json", data)
    os.rename(staging, destination)
    fd = os.open(destination.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return data


def validate_array(array, *, shape: tuple[int, ...], validity=None, minimum=None, maximum=None) -> dict:
    import numpy as np
    a = np.asarray(array)
    if a.shape != shape or a.dtype.kind not in "fiu":
        raise ArtifactError("Array shape or dtype does not match contract")
    mask = np.ones(shape, bool) if validity is None else np.asarray(validity)
    if mask.shape != shape or mask.dtype != bool:
        raise ArtifactError("Validity mask must be boolean with the same shape")
    values = a[mask]
    if not len(values) or not np.isfinite(values).all():
        raise ArtifactError("No valid finite samples")
    if minimum is not None and values.min() < minimum:
        raise ArtifactError("Array below physical range")
    if maximum is not None and values.max() > maximum:
        raise ArtifactError("Array above physical range")
    return {"shape": list(shape), "dtype": str(a.dtype), "valid_count": int(mask.sum()),
            "min": float(values.min()), "max": float(values.max())}
