"""Stable content-hashed save snapshots, with no writes to the user's save."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from .nbt import SourceError


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def file_inventory(world: Path) -> list[dict]:
    entries = []
    for path in sorted(Path(world).rglob("*")):
        if path.is_symlink():
            raise SourceError(f"Symlink in source save is unsupported: {path.relative_to(world)}")
        if not path.is_file() or path.name == "session.lock":
            continue
        before = path.stat()
        digest = sha256(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
            raise SourceError("Source changed during hashing; retry after save becomes stable")
        entries.append({"path": path.relative_to(world).as_posix(), "bytes": after.st_size, "sha256": digest})
    return entries


def snapshot_world(world: Path, output: Path) -> dict:
    world,output=Path(world).resolve(),Path(output).resolve()
    if world.is_file():
        from .archive import snapshot_archive
        return snapshot_archive(world,output)
    return _snapshot_directory(world,output)


def _snapshot_directory(world: Path, output: Path, source_reference: dict | None = None) -> dict:
    world, output = Path(world).resolve(), Path(output).resolve()
    if not (world / "level.dat").is_file():
        raise SourceError("Expected a Java save directory containing level.dat")
    if output == world or world in output.parents:
        raise SourceError("Snapshot destination must be outside the input save")
    initial = file_inventory(world)
    save_hash = canonical_hash(initial)
    destination = output / save_hash
    manifest_path = destination / "snapshot.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("save_sha256") != save_hash or file_inventory(destination / "save") != initial:
            raise SourceError("Existing snapshot failed content verification")
        return {**manifest, **(source_reference or {})}
    output.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=output))
    try:
        for entry in initial:
            source = world / entry["path"]
            target = staging / "save" / entry["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            if sha256(target) != entry["sha256"]:
                raise SourceError("Snapshot copy differs from stable source hash")
            target.chmod(0o444)
        final = file_inventory(world)
        if initial != final:
            raise SourceError("Source save changed during snapshot; candidate rejected")
        manifest = {
            "schema_version": 1, "kind": "SourceSnapshot", "status": "validated",
            "save_sha256": save_hash, "original_path": str(world),
            "snapshot_path": str(destination / "save"),
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "files": initial,
            "stability": {"before_after_source_hashes_equal": True, "copied_bytes_verified": True,
                          "source_write_operations": 0, "excluded_files": ["session.lock"],
                          "scope": "all regular save files except the volatile Minecraft session lock"},
        }
        manifest.update(source_reference or {})
        write_json(staging / "snapshot.json", manifest)
        os.replace(staging, destination)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_source_unchanged(snapshot_manifest: dict) -> dict:
    if snapshot_manifest.get("original_kind")=="archive":
        digest=sha256(Path(snapshot_manifest["original_path"]))
        expected=snapshot_manifest["archive_sha256"]
        return {"status":"pass" if digest==expected else "fail","original_archive_sha256":expected,"current_archive_sha256":digest,"scope":"complete original archive bytes"}
    current = file_inventory(Path(snapshot_manifest["original_path"]))
    digest = canonical_hash(current)
    return {"status": "pass" if digest == snapshot_manifest["save_sha256"] else "fail",
            "original_save_sha256": snapshot_manifest["save_sha256"], "current_save_sha256": digest,
            "scope": "all regular input files except session.lock"}
