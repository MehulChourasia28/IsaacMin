import json

import numpy as np
import pytest
import trimesh

from isaacmin.source.nbt import SourceError
from isaacmin.volumes.mesh_validation import _load_mesh,validate_source_mesh
from isaacmin.volumes.topology import source_topology


def fixture_source(folder):
    occupancy=np.zeros((32,32,32),dtype=np.uint8)
    occupancy[:16]=1
    np.savez_compressed(folder/"natural_occupancy.npz",occupancy=occupancy,validity=np.ones_like(occupancy,dtype=bool),min_xyz=np.asarray([0,0,0]))
    np.savez_compressed(folder/"terrain_surface.npz",height=np.full((32,32),16),validity=np.ones((32,32),dtype=bool))
    (folder/"world_ir.json").write_text(json.dumps({"content_sha256":"independent-mesh-fixture"}))
    source_topology(folder)


def test_independent_solver_rejects_injected_missing_ground(tmp_path):
    fixture_source(tmp_path)
    exact=trimesh.creation.box(extents=(32,16,32))
    exact.apply_translation((16,8,16))
    exact.export(tmp_path/"exact.obj")
    result=validate_source_mesh(tmp_path,tmp_path/"exact.obj",tmp_path/"exact_report",np.eye(4))
    assert result["status"]=="pass"
    assert result["sample_count"]>=10000
    assert result["mismatch_count"]==0
    inverted=exact.copy()
    inverted.invert()
    inverted.export(tmp_path/"inverted.obj")
    result=validate_source_mesh(tmp_path,tmp_path/"inverted.obj",tmp_path/"inverted_report",np.eye(4))
    assert result["status"]=="fail"
    assert result["mismatch_count"]==0
    assert not result["mesh"]["positive_closed_volume"]
    # A new technically valid box that omits the entire source top half must
    # fail source occupancy even though it remains closed and watertight.
    damaged=trimesh.creation.box(extents=(32,8,32))
    damaged.apply_translation((16,4,16))
    damaged.export(tmp_path/"missing_ground.obj")
    result=validate_source_mesh(tmp_path,tmp_path/"missing_ground.obj",tmp_path/"damaged_report",np.eye(4))
    assert result["status"]=="fail"
    assert result["mesh"]["watertight"]
    assert result["mismatch_count"]>1000
    assert result["isaac_contact"]["status"]=="not_run"


def test_mesh_coordinate_transform_is_required(tmp_path):
    path=tmp_path/"cube.obj"
    trimesh.creation.box().export(path)
    with pytest.raises(SourceError,match="explicit"):
        _load_mesh(path,None)


def test_attribute_seams_close_but_nonzero_geometric_cracks_do_not(tmp_path):
    cube=trimesh.creation.box()
    vertices=cube.vertices[cube.faces].reshape(-1,3)
    faces=np.arange(len(vertices)).reshape(-1,3)
    path=tmp_path/"attribute_seams.npz"
    np.savez(path,vertices=vertices,faces=faces,mesh_to_source=np.eye(4))
    assert _load_mesh(path,None).is_volume
    # A small real separation must not disappear in decimal-coordinate welding.
    vertices[0,0]+=1e-9
    np.savez(path,vertices=vertices,faces=faces,mesh_to_source=np.eye(4))
    assert not _load_mesh(path,None).is_watertight
