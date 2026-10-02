import gzip
import json

import nbtlib
import numpy as np
import pytest
from nbtlib import Byte, Compound, Int, List, LongArray, String

from isaacmin.source.world_ir import extract_world_ir
from isaacmin.validation.source_water import audit_source_water
from test_source import make_region, nbt_bytes


def make_water_fixture(tmp_path):
    world = tmp_path / "save"
    world.mkdir()
    (world / "level.dat").write_bytes(gzip.compress(nbt_bytes({"Data": Compound({"DataVersion": Int(4786), "spawn": Compound({"pos": nbtlib.IntArray([0, 16, 0])})})})))
    states = [{"Name": "minecraft:air"}, {"Name": "minecraft:stone"},
        {"Name": "minecraft:water", "Properties": {"level": "0"}},
        {"Name": "minecraft:water", "Properties": {"level": "3"}},
        {"Name": "minecraft:water", "Properties": {"level": "8"}},
        {"Name": "minecraft:bubble_column", "Properties": {"drag": "true"}},
        {"Name": "minecraft:glow_lichen", "Properties": {"waterlogged": "true", "north": "true"}}]
    values = np.zeros((32, 32, 32), dtype=np.uint8)
    values[:16] = 1
    values[3, 8, 8:11] = [2, 5, 6]
    values[4, 8, 8:11] = 0
    values[17, 8, 8] = 3
    values[17, 8, 11] = 4
    palette = List[Compound]([Compound({key: Compound({k: String(v) for k, v in val.items()}) if isinstance(val, dict) else String(val) for key, val in state.items()}) for state in states])
    for cx in (-1, 0):
        for cz in (-1, 0):
            sections = []
            for sy in (0, 1):
                data = values[sy*16:sy*16+16, (cz+1)*16:(cz+2)*16, (cx+1)*16:(cx+2)*16]
                packed = [sum(int(v) << (i*4) for i, v in enumerate(group)) for group in data.ravel().reshape(-1, 16)]
                sections.append(Compound({"Y": Byte(sy), "block_states": Compound({"palette": palette, "data": LongArray(packed)}),
                    "biomes": Compound({"palette": List[String]([String("minecraft:plains")])})}))
            make_region(world / f"region/r.{cx//32}.{cz//32}.mca", nbt_bytes({"DataVersion": Int(4786), "xPos": Int(cx), "zPos": Int(cz),
                        "Status": String("minecraft:full"), "sections": List[Compound](sections)}), cx, cz)
    ir = tmp_path / "ir"
    extract_world_ir(world, ir, center=(0, 0), extent=32)
    return world, ir


def test_independent_water_audit_keeps_stacked_fluids_levels_and_lichen(tmp_path):
    world, ir = make_water_fixture(tmp_path)
    report = audit_source_water(world, ir, tmp_path / "water")
    assert report["status"] == "pass"
    assert report["source_state_comparisons"] == 5
    assert report["water_block_level_counts"] == {"0": 1, "3": 1, "8": 1}
    assert report["bubble_column_cells"] == report["waterlogged_cells"] == 1
    assert report["free_upper_air_interfaces"] == 5
    assert report["free_upper_air_interfaces_under_cover"] == 3
    assert report["minimum_free_upper_boundary_y"] == 4
    assert report["maximum_free_interfaces_in_one_xz_column"] == 2
    assert report["water_air_interfaces_lost_by_max_column_only"] == 1
    with np.load(tmp_path / "water/source_water_samples.npz") as data:
        level_three = data["block_level"] == 3
        assert np.allclose(data["fluid_own_or_full_height"][level_three], 5/9)
        assert len(data["free_upper_source_xyz"]) == 5
    target = ir / "natural_occupancy.npz"
    target.write_bytes(target.read_bytes()+b"altered")
    with pytest.raises(ValueError, match="Retained source artifact changed"):
        audit_source_water(world, ir, tmp_path / "stale")
