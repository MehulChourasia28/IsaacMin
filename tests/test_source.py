from __future__ import annotations

import gzip
import io
import json
import struct
import zlib
from pathlib import Path

import nbtlib
import numpy as np
import pytest
from nbtlib import Byte, Compound, Int, List, LongArray, String

from isaacmin.source.anvil import Chunk, Region, dimensions, unpack_indices
from isaacmin.source.nbt import SourceError, decompress, loads
from isaacmin.source.semantics import classify
from isaacmin.source.snapshot import file_inventory, snapshot_world, verify_source_unchanged


def nbt_bytes(value):
    stream=io.BytesIO()
    nbtlib.File(value).write(stream)
    return stream.getvalue()


def make_region(path: Path, raw: bytes, cx=0, cz=0, codec=2, external=False):
    payload=zlib.compress(raw) if codec==2 else raw
    header=bytearray(8192)
    count=1 if external else (len(payload)+5+4095)//4096
    struct.pack_into(">I",header,4*((cz%32)*32+cx%32),(2<<8)|count)
    body=struct.pack(">I",1 if external else len(payload)+1)+bytes([codec|(128 if external else 0)])+(b"" if external else payload)
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(header+body+bytes(count*4096-len(body)))
    if external:
        (path.parent/f"c.{cx}.{cz}.mcc").write_bytes(payload)


def chunk_nbt(cx=-1,cz=-1):
    return nbt_bytes({"DataVersion":Int(4786),"xPos":Int(cx),"zPos":Int(cz),"Status":String("minecraft:full"),
                      "sections":List[Compound]([Compound({"Y":Byte(-1),"block_states":Compound({"palette":List[Compound]([Compound({"Name":String("minecraft:stone")})])})})])})


def test_nbt_roundtrip_independent_writer_and_unicode():
    value={"a":Int(-100),"b":String("negative x, 🪨 and\x00"),"longs":LongArray([-1,2**63-1]),"nested":Compound({"value":Byte(-7)})}
    assert loads(nbt_bytes(value))=={"a":-100,"b":"negative x, 🪨 and\x00","longs":[-1,2**63-1],"nested":{"value":-7}}


def test_nbt_rejects_truncation_depth_and_extra_data():
    raw=nbt_bytes({"x":Int(1)})
    with pytest.raises(SourceError): loads(raw[:-1])
    with pytest.raises(SourceError): loads(raw+b"bad")
    with pytest.raises(SourceError): loads(b"\x03\x00\x00\x00\x00\x00\x00")


def test_decompression_limits_and_codecs():
    raw=nbt_bytes({"x":Int(4786)})
    assert decompress(gzip.compress(raw),1)==raw
    assert decompress(zlib.compress(raw),2)==raw
    assert decompress(raw,3)==raw
    with pytest.raises(SourceError): decompress(zlib.compress(b"x"*10000),2,128)
    with pytest.raises(SourceError): decompress(raw,4)
    with pytest.raises(SourceError): decompress(zlib.compress(raw)[:-2],2)


def test_padded_palettes_and_legacy_cross_word():
    # Construct bit strings independently of the production shifts/masks.
    expected=[(i*7)%17 for i in range(4096)]
    width=5
    bits="".join(format(v,f"0{width}b")[::-1] for v in expected)
    continuous=[int(bits[i:i+64][::-1],2) for i in range(0,len(bits),64)]
    padded=[]
    for offset in range(0,len(expected),12):
        part="".join(format(v,"05b")[::-1] for v in expected[offset:offset+12])
        padded.append(int(part[::-1],2))
    assert unpack_indices(continuous,4096,17,4,False)==expected
    assert unpack_indices(padded,4096,17,4,True)==expected
    with pytest.raises(SourceError): unpack_indices(padded[:-1],4096,17,4,True)
    with pytest.raises(SourceError): unpack_indices(None,4096,2,4,True)
    with pytest.raises(SourceError): unpack_indices([15]*256,4096,2,4,True)


def test_negative_region_coordinates_external_chunk_and_missing(tmp_path):
    path=tmp_path/"r.-1.-1.mca"
    make_region(path,chunk_nbt(),-1,-1,external=True)
    region=Region(path)
    assert list(region.locations)==[(-1,-1)]
    assert region.chunk(-1,-1).block(-1,-1,-1)["Name"]=="minecraft:stone"
    with pytest.raises(SourceError,match="Missing chunk"): region.chunk(-2,-1)
    with pytest.raises(SourceError,match="Unstored section"): region.chunk(-1,-1).block(-1,0,-1)
    (tmp_path/"c.-1.-1.mcc").unlink()
    with pytest.raises(SourceError,match="external chunk"): region.chunk(-1,-1)


def test_malformed_region_sector_does_not_become_air(tmp_path):
    path=tmp_path/"r.0.0.mca"
    header=bytearray(8192)
    struct.pack_into(">I",header,0,(2<<8)|1)
    path.write_bytes(header)
    with pytest.raises(SourceError,match="sector"): Region(path)


def test_modern_dimensions_ambiguous_layout_and_unknown_semantics(tmp_path):
    folder=tmp_path/"dimensions/minecraft/overworld/region"
    folder.mkdir(parents=True)
    assert dimensions(tmp_path)["minecraft:overworld"]==folder.parent
    (tmp_path/"region").mkdir()
    with pytest.raises(SourceError,match="Ambiguous"): dimensions(tmp_path)
    assert classify("mod:stone")=="unknown"
    assert classify("minecraft:new_unknown_block")=="unknown"
    assert classify("minecraft:cave_air")=="air"
    assert classify("minecraft:water")=="water"
    assert classify("minecraft:oak_leaves")=="vegetation"
    assert classify("minecraft:stone_brick_stairs")=="structure"


def test_snapshot_is_stable_independent_and_detects_mutation(tmp_path):
    world=tmp_path/"world"
    world.mkdir()
    (world/"level.dat").write_bytes(gzip.compress(nbt_bytes({"Data":Compound({"DataVersion":Int(4786)})})))
    (world/"session.lock").write_bytes(b"volatile")
    (world/"region").mkdir()
    (world/"region/test").write_bytes(b"source")
    before=file_inventory(world)
    snapshot=snapshot_world(world,tmp_path/"snapshots")
    assert file_inventory(world)==before
    assert Path(snapshot["snapshot_path"]).joinpath("region/test").read_bytes()==b"source"
    assert snapshot_world(world,tmp_path/"snapshots")["save_sha256"]==snapshot["save_sha256"]
    assert verify_source_unchanged(snapshot)["status"]=="pass"
    (world/"region/test").write_bytes(b"changed")
    assert verify_source_unchanged(snapshot)["status"]=="fail"
    assert Path(snapshot["snapshot_path"]).joinpath("region/test").read_bytes()==b"source"


def test_snapshot_rejects_symlink_and_nested_destination(tmp_path):
    world=tmp_path/"world"
    world.mkdir()
    (world/"level.dat").write_bytes(b"data")
    with pytest.raises(SourceError,match="outside"): snapshot_world(world,world/"out")
    (world/"unsafe").symlink_to(tmp_path/"unrelated")
    with pytest.raises(SourceError,match="Symlink"): snapshot_world(world,tmp_path/"snapshots")


def test_stacked_source_floors_retained_and_missing_chunk_explicit(tmp_path):
    from isaacmin.source.world_ir import extract_world_ir
    world=tmp_path/"world"
    world.mkdir()
    level={"Data":Compound({"DataVersion":Int(4786),"spawn":Compound({"pos":nbtlib.IntArray([0,4,0])})})}
    (world/"level.dat").write_bytes(gzip.compress(nbt_bytes(level)))
    # A two-floor source volume: full stone floors at y=0 and y=8 with air
    # between/above. A heightfield alone cannot preserve the lower floor.
    palette=List[Compound]([Compound({"Name":String("minecraft:air")}),Compound({"Name":String("minecraft:stone")})])
    values=np.zeros((16,16,16),dtype=np.int64)
    values[0]=1
    values[8]=1
    encoded=[]
    for group in values.ravel().reshape(-1,16):
        encoded.append(sum(int(value)<<(i*4) for i,value in enumerate(group)))
    root={"DataVersion":Int(4786),"xPos":Int(0),"zPos":Int(0),"Status":String("minecraft:full"),"sections":List[Compound]([
        Compound({"Y":Byte(0),"block_states":Compound({"palette":palette,"data":LongArray(encoded)}),"biomes":Compound({"palette":List[String]([String("minecraft:plains")])})})])}
    make_region(world/"region/r.0.0.mca",nbt_bytes(root))
    report=extract_world_ir(world,tmp_path/"ir",bounds=(0,0,32,32),center=(8,8),extent=32)
    assert report["scope"]["full_chunks"]==1
    assert len(report["scope"]["exclusions"])==3
    with np.load(tmp_path/"ir/terrain_surface.npz") as surface:
        assert np.all(surface["height"][:16,:16]==9)
        assert np.all(surface["validity"][:16,:16])
        assert not surface["validity"][16:,16:].any()
    with np.load(tmp_path/"ir/supporting_surfaces.npz") as floors:
        assert set(floors["xyz"][:,1])=={1,9}
    with np.load(tmp_path/"ir/natural_occupancy.npz") as volume:
        assert np.all(volume["occupancy"][:,16:,16:]==3)
        assert np.all(volume["occupancy"][1:8,:16,:16]==0)
