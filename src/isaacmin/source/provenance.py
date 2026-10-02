"""Bind decoded interpretations to exact save bytes and reader semantics."""
from __future__ import annotations

import json
from pathlib import Path

from .nbt import SourceError
from .snapshot import canonical_hash, file_inventory, sha256


def reader_producer_identity(stage: str) -> dict:
    if stage not in {"inventory", "world_ir", "surface_objects"}:
        raise ValueError("Unknown source reader stage")
    names = (stage + ".py", "provenance.py", "anvil.py", "nbt.py", "semantics.py", "snapshot.py")
    root = Path(__file__).parent
    return {"name": "isaacmin.source." + stage, "provenance_version": 2,
            "code_sha256": {name: sha256(root / name) for name in names}}


def begin_source_read(snapshot: Path, stage: str, expected_hash: str | None = None) -> tuple[dict, list[dict]]:
    files = file_inventory(snapshot)
    digest = canonical_hash(files)
    manifest_path = snapshot.parent / "snapshot.json"
    if manifest_path.is_file():
        record = json.loads(manifest_path.read_text())
        if record.get("save_sha256") != digest or record.get("files") != files:
            raise SourceError("Source snapshot manifest does not match actual source bytes")
    if expected_hash is not None and expected_hash != digest:
        raise SourceError("Supplied source snapshot identity differs from actual source bytes")
    return {"source_snapshot_sha256": digest, "producer": reader_producer_identity(stage)}, files


def finish_source_read(snapshot: Path, stage: str, identity: dict, initial_files: list[dict]) -> None:
    if file_inventory(snapshot) != initial_files:
        raise SourceError("Source changed while decoding; no coherent candidate can be published")
    if reader_producer_identity(stage) != identity["producer"]:
        raise SourceError("Source parser or semantics changed while decoding")
