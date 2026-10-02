import json

import numpy as np
import pytest

from isaacmin.io import atomic_json, sha256_file
from isaacmin.source.nbt import SourceError
from isaacmin.volumes.source_support import derive_source_support


def support_fixture(path):
    path.mkdir()
    (path / "terrain_volume").mkdir()
    names = ["minecraft:air", "minecraft:stone", "minecraft:cobblestone", "minecraft:mossy_cobblestone",
             "minecraft:chest", "minecraft:spawner", "minecraft:oak_leaves"]
    placements = {(-31, -30, 50): 1, (-32, -31, 48): 2, (-16, 1, 63): 3,
                  (-17, -1, 52): 4, (-28, -16, 51): 5, (-20, 3, 54): 6}
    minimum = np.asarray([-32, -32, 48])
    occupancy = np.zeros((48, 16, 32), dtype=np.uint8)
    for xyz, block in placements.items():
        y, z, x = (np.asarray(xyz)-minimum)[[1, 2, 0]]
        occupancy[y, z, x] = 1 if block == 1 else 2
    np.savez_compressed(path / "natural_occupancy.npz", occupancy=occupancy, validity=np.ones_like(occupancy, dtype=bool),
                        min_xyz=minimum, voxel_size_m=np.asarray(1.))
    chunks = []
    for cx in (-2, -1):
        # Deliberately unordered sections demonstrate coordinate mapping uses
        # the actual section Y value, never its position in the stored array.
        sections = [0, -2, -1]
        ids = np.zeros((3, 16, 16, 16), dtype=np.uint16)
        for (x, y, z), block in placements.items():
            if x//16 == cx:
                ids[sections.index(y//16), y%16, z%16, x%16] = block
        name = f"terrain_volume/c.{cx}.3.npz"
        np.savez_compressed(path / name, block_id=ids, section_y=np.asarray(sections),
                            palette_json=np.asarray(json.dumps([{"Name": n} for n in names])), min_xyz=np.asarray([cx*16, -32, 48]))
        chunks.append({"chunk_xz": [cx, 3], "section_y": sections, "volume_file": name})
    manifest = {"source_snapshot_sha256": "synthetic-support-coordinate-fixture", "terrain_volume": {"chunk_records": chunks},
                "files": [{"path": p.relative_to(path).as_posix(), "sha256": sha256_file(p)} for p in sorted(path.rglob("*.npz"))]}
    atomic_json(path / "world_ir.json", manifest)
    return occupancy, placements


def test_structural_floor_positions_are_exact_and_source_classification_unchanged(tmp_path):
    ir = tmp_path / "ir"
    original, placements = support_fixture(ir)
    hashes = {str(p): sha256_file(p) for p in ir.rglob("*") if p.is_file()}
    support, manifest = derive_source_support(ir, tmp_path / "support")
    assert support.dtype == bool
    assert support.shape == original.shape
    assert manifest["natural_solid_cells"] == 1
    assert manifest["structural_stone_cells"] == 2
    assert support.sum() == 3
    assert {tuple(record["source_xyz_min_corner"]) for record in manifest["structural_blocks"]} == {(-32, -31, 48), (-16, 1, 63)}
    assert {record["source_block_state"]["Name"] for record in manifest["excluded_structure_objects"]} == {"minecraft:chest", "minecraft:spawner"}
    assert manifest["excluded_non_ground_counts"]["minecraft:oak_leaves"] == 1
    for xyz, block in placements.items():
        y, z, x = (np.asarray(xyz)-[-32, -32, 48])[[1, 2, 0]]
        assert bool(support[y, z, x]) == (block in (1, 2, 3))
    assert hashes == {str(p): sha256_file(p) for p in ir.rglob("*") if p.is_file()}
    with pytest.raises(SourceError, match="Preserve prior"):
        derive_source_support(ir, tmp_path / "support")


def test_unknown_support_cells_and_changed_retained_palettes_fail(tmp_path):
    ir = tmp_path / "ir"
    support_fixture(ir)
    path = ir / "natural_occupancy.npz"
    data = dict(np.load(path, allow_pickle=False))
    data["occupancy"][0, 0, 0] = 3
    np.savez_compressed(path, **data)
    manifest = json.loads((ir / "world_ir.json").read_text())
    for entry in manifest["files"]:
        if entry["path"] == path.name:
            entry["sha256"] = sha256_file(path)
    atomic_json(ir / "world_ir.json", manifest)
    with pytest.raises(SourceError, match="complete known"):
        derive_source_support(ir)
    data["occupancy"][0, 0, 0] = 0
    np.savez_compressed(path, **data)
    with pytest.raises(SourceError, match="differs from retained"):
        derive_source_support(ir)


def test_added_masonry_is_independently_required_as_floor_and_roof(tmp_path):
    from isaacmin.contracts.coordinates import CoordinateFrame
    from isaacmin.validation.cave_views import create_cave_view_plan
    from isaacmin.volumes.mesh_validation import validate_source_mesh, validate_source_connectivity
    from isaacmin.volumes.topology import source_topology
    from test_source_connectivity import voxel_boundary_mesh
    ir = tmp_path / "chamber"
    (ir / "terrain_volume").mkdir(parents=True)
    occupancy = np.zeros((32, 32, 32), dtype=np.uint8)
    occupancy[:16] = 1
    occupancy[4:9, 12:20, 8:24] = 0
    occupancy[3, 12:20, 8:24] = 2
    occupancy[9, 14:18, 14:18] = 2
    np.savez_compressed(ir / "natural_occupancy.npz", occupancy=occupancy, validity=np.ones_like(occupancy, bool), min_xyz=np.asarray([0, 0, 0]), voxel_size_m=np.asarray(1.))
    np.savez_compressed(ir / "terrain_surface.npz", height=np.full((32, 32), 16), validity=np.ones((32, 32), bool))
    chunks = []
    for cx in range(2):
        for cz in range(2):
            name = f"terrain_volume/c.{cx}.{cz}.npz"
            ids = np.stack([occupancy[y:y+16, cz*16:cz*16+16, cx*16:cx*16+16] for y in (0, 16)])
            np.savez_compressed(ir / name, block_id=ids, section_y=np.asarray([0, 1]), min_xyz=np.asarray([cx*16, 0, cz*16]),
                                palette_json=np.asarray(json.dumps([{"Name": n} for n in ("minecraft:air", "minecraft:stone", "minecraft:cobblestone")])))
            chunks.append({"chunk_xz": [cx, cz], "section_y": [0, 1], "volume_file": name})
    atomic_json(ir / "world_ir.json", {"content_sha256": "synthetic-chamber-structural-floor-and-roof", "source_snapshot_sha256": "fixture",
        "terrain_volume": {"chunk_records": chunks}, "files": [{"path": p.relative_to(ir).as_posix(), "sha256": sha256_file(p)} for p in sorted(ir.rglob("*.npz"))]})
    source_topology(ir)
    support, record = derive_source_support(ir, tmp_path / "support")
    support_manifest = tmp_path / "support/support_manifest.json"
    full_mesh = voxel_boundary_mesh(support)
    full_mesh.export(tmp_path / "full.obj")
    natural_mesh = voxel_boundary_mesh(occupancy == 1)
    natural_mesh.export(tmp_path / "missing_masonry.obj")
    for filename, expected in (("full.obj", "pass"), ("missing_masonry.obj", "fail")):
        output = tmp_path / (filename + "_checks")
        sampled = validate_source_mesh(ir, tmp_path / filename, output / "samples", np.eye(4), support_manifest_path=support_manifest)
        graph = validate_source_connectivity(ir, tmp_path / filename, output / "graph", np.eye(4), support_manifest_path=support_manifest)
        assert sampled["status"] == graph["status"] == expected
        assert sampled["structural_support"]["structural_cells"] == record["structural_stone_cells"]
        assert graph["structural_support"]["structural_roof_intervals"] > 0
        assert bool(sampled["structural_support"]["inside_failures"]) == (expected == "fail")
        assert bool(graph["structural_support"]["inside_failures"]) == (expected == "fail")
    frame = CoordinateFrame()
    full_mesh.apply_transform(frame.record()["source_to_world"])
    full_mesh.export(tmp_path / "world.obj")
    no_masonry = create_cave_view_plan(tmp_path / "world.obj", ir, frame)
    assert not no_masonry["poses_static"]
    planned = create_cave_view_plan(tmp_path / "world.obj", ir, frame, support_manifest_path=support_manifest)
    assert planned["poses_static"] and planned["contact_probes"]
    assert all(p["support_source_class"] == "structural_masonry" for p in planned["poses_static"])
