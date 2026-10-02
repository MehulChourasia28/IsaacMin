import json
from pathlib import Path

import numpy as np

from isaacmin.volumes.topology import source_topology


def test_cross_chunk_cave_portal_thin_roof_and_unknown(tmp_path):
    # Explicit source-resolution adversarial fixture: a passage crosses x=16,
    # reaches open sky through one mouth, and has a one-voxel natural roof.
    shape=(8,8,32)
    occupancy=np.ones(shape,dtype=np.uint8)
    occupancy[5:]=0
    occupancy[2:4,3:5,2:30]=0
    occupancy[2:6,3:5,2:4]=0
    # Unknown hole must not manufacture a second portal or a connected room.
    occupancy[2:4,3:5,30]=3
    valid=occupancy!=3
    np.savez_compressed(tmp_path/"natural_occupancy.npz",occupancy=occupancy,validity=valid,min_xyz=np.asarray([-32,-4,10]))
    height=np.full((8,32),1,dtype=np.float32)
    height[3:5,2:4]=-2
    np.savez_compressed(tmp_path/"terrain_surface.npz",height=height,validity=np.ones((8,32),dtype=bool))
    (tmp_path/"world_ir.json").write_text(json.dumps({"content_sha256":"fixture-cross-chunk-thin-roof"}))
    report=source_topology(tmp_path)
    assert report["component_count"]==1
    assert report["portal_count"]==1
    cave=report["features"][0]
    assert cave["bounds_xyz_blocks"][0] < -16 < cave["bounds_xyz_blocks"][3]
    assert cave["connected_to_observed_sky"]
    assert cave["touches_unknown_source"]
    assert not cave["touches_crop_boundary"]
    assert cave["minimum_vertical_roof_thickness_m"]==1
    assert report["target_comparison"]["status"]=="not_run"
    assert report["portals"][0]["robot_passable"]=="not_inferred"
    first_id=cave["id"]
    assert source_topology(tmp_path)["features"][0]["id"]==first_id


def test_diagonal_cells_not_falsely_connected(tmp_path):
    occupancy=np.ones((5,5,5),dtype=np.uint8)
    occupancy[4]=0
    occupancy[1,1,1]=0
    occupancy[2,2,2]=0
    np.savez_compressed(tmp_path/"natural_occupancy.npz",occupancy=occupancy,validity=np.ones_like(occupancy,dtype=bool),min_xyz=np.asarray([0,0,0]))
    np.savez_compressed(tmp_path/"terrain_surface.npz",height=np.full((5,5),4),validity=np.ones((5,5),dtype=bool))
    (tmp_path/"world_ir.json").write_text(json.dumps({"content_sha256":"fixture-diagonal"}))
    report=source_topology(tmp_path)
    assert report["component_count"]==2
    assert report["portal_count"]==0
    assert not any(c["connected_to_observed_sky"] for c in report["features"])
