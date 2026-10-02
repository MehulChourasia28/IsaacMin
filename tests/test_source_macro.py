import zlib

import numpy as np
import pytest

from isaacmin.source.macro import extract_macro_surface, macro_producer_identity
from isaacmin.source.nbt import SourceError
from isaacmin.source.snapshot import sha256, snapshot_world
from test_source import chunk_nbt, make_region


def test_macro_binds_external_payload_snapshot_and_producer(tmp_path):
    world = tmp_path / "world"
    world.mkdir()
    (world / "level.dat").write_bytes(b"fixture metadata hashed but not interpreted by macro")
    region = world / "region/r.0.0.mca"
    make_region(region, chunk_nbt(0, 0), 0, 0, external=True)
    before_region_hash = sha256(region)
    snapshot = snapshot_world(world, tmp_path / "snapshots")
    first = extract_macro_surface(snapshot["snapshot_path"], tmp_path / "first", (16, 16), extent=32)
    assert first["source_snapshot_sha256"] == snapshot["save_sha256"]
    assert first["producer"] == macro_producer_identity()
    assert {entry["path"] for entry in first["source_dependency_closure"]} == {
        "level.dat", "region/r.0.0.mca", "region/c.0.0.mcc"}
    assert first["scope"]["full_chunks"] == 1
    assert len(first["scope"]["exclusions"]) == 3
    # The external source changes without touching its Anvil stub or level.dat.
    # Use the independent NBT writer to preserve actual length fields.
    from test_source import nbt_bytes
    from nbtlib import Int, String, List, Compound, Byte
    raw = nbt_bytes({"DataVersion": Int(4786), "xPos": Int(0), "zPos": Int(0), "Status": String("minecraft:full"),
                     "sections": List[Compound]([Compound({"Y": Byte(-1), "block_states": Compound({"palette": List[Compound]([Compound({"Name": String("minecraft:dirt")})])})})])})
    (region.parent / "c.0.0.mcc").write_bytes(zlib.compress(raw))
    assert sha256(region) == before_region_hash
    changed = snapshot_world(world, tmp_path / "snapshots")
    second = extract_macro_surface(changed["snapshot_path"], tmp_path / "second", (16, 16), extent=32)
    assert second["source_snapshot_sha256"] != first["source_snapshot_sha256"]
    assert second["source_regions"] == first["source_regions"]
    assert second["source_dependency_closure"] != first["source_dependency_closure"]
    with np.load(tmp_path / "second/macro_surface.npz", allow_pickle=False) as data:
        assert np.all(data["substrate"][data["validity"]] == "minecraft:dirt")
    with pytest.raises(SourceError, match="Supplied macro snapshot hash"):
        extract_macro_surface(changed["snapshot_path"], tmp_path / "wrong", (16, 16), extent=32, snapshot_sha256="0"*64)
