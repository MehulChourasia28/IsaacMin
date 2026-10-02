import gzip
import json
from pathlib import Path

import pytest
from nbtlib import Compound, Int

from isaacmin.source.inventory import inspect_source
from isaacmin.source.nbt import SourceError
from isaacmin.source.provenance import begin_source_read, finish_source_read, reader_producer_identity
from isaacmin.source.snapshot import snapshot_world
from isaacmin.source.world_ir import extract_world_ir
from test_source import chunk_nbt, make_region, nbt_bytes


def make_source(path):
    path.mkdir()
    (path / "level.dat").write_bytes(gzip.compress(nbt_bytes({"Data": Compound({"DataVersion": Int(4786)})})))
    make_region(path / "region/r.0.0.mca", chunk_nbt(0, 0), 0, 0, external=True)


def test_inventory_and_ir_bind_external_payload_and_reader(tmp_path):
    world = tmp_path / "world"
    make_source(world)
    snapshot = snapshot_world(world, tmp_path / "snapshots")
    save = Path(snapshot["snapshot_path"])
    inventory = inspect_source(save, tmp_path / "inventory")
    ir = extract_world_ir(save, tmp_path / "ir", center=(16, 16), extent=32)
    for stage, report in (("inventory", inventory), ("world_ir", ir)):
        assert report["source_snapshot_sha256"] == snapshot["save_sha256"]
        assert report["producer"] == reader_producer_identity(stage)
        assert {entry["path"] for entry in report["source_dependency_closure"]} == {
            "level.dat", "region/r.0.0.mca", "region/c.0.0.mcc"}
        assert report["files"]
    assert inventory["dimensions"][0]["external_chunk_count"] == 1
    # Identical arrays cannot make a stale semantic producer reusable.
    ir["producer"]["code_sha256"]["semantics.py"] = "old-reader-revision"
    path = tmp_path / "ir/world_ir.json"
    path.write_text(json.dumps(ir))
    with pytest.raises(SourceError, match="another candidate"):
        extract_world_ir(save, tmp_path / "ir", center=(16, 16), extent=32)
    assert json.loads(path.read_text())["producer"] == ir["producer"]


def test_source_change_during_decoding_cannot_publish_coherent_identity(tmp_path):
    world = tmp_path / "world"
    make_source(world)
    identity, files = begin_source_read(world, "inventory")
    external = world / "region/c.0.0.mcc"
    external.write_bytes(external.read_bytes() + b"changed")
    with pytest.raises(SourceError, match="Source changed"):
        finish_source_read(world, "inventory", identity, files)
