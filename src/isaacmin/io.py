from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .security import redact


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def hash_object(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    """Durable replace; staging and destination are on the same filesystem."""
    path.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(redact(value), sort_keys=True, indent=2, allow_nan=False) + "\n"
    fd, name = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        dirfd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(dirfd)
        finally:
            os.close(dirfd)
    finally:
        Path(name).unlink(missing_ok=True)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def code_hash(workspace: Path) -> str:
    roots = [workspace / p for p in ("src", "scripts", "blender_scripts", "isaac_scripts", "recipes")]
    files = sorted(p for root in roots if root.exists() for p in root.rglob("*")
                   if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc")
    return hash_object({str(p.relative_to(workspace)): sha256_file(p) for p in files})

