"""Adversarial synthetic source fixtures for the independent target-ray graph."""
import json

import numpy as np
import trimesh

from isaacmin.volumes.mesh_validation import validate_source_connectivity
from isaacmin.volumes.topology import source_topology


def voxel_boundary_mesh(occupied):
    vertices=[]
    faces=[]
    index={}
    def vertex(v):
        if v not in index:
            index[v]=len(vertices)
            vertices.append(v)
        return index[v]
    for y,z,x in np.argwhere(occupied):
        panels=[((-1,0,0),[(x,y,z),(x,y,z+1),(x,y+1,z+1),(x,y+1,z)]),
                ((1,0,0),[(x+1,y,z),(x+1,y+1,z),(x+1,y+1,z+1),(x+1,y,z+1)]),
                ((0,-1,0),[(x,y,z),(x+1,y,z),(x+1,y,z+1),(x,y,z+1)]),
                ((0,1,0),[(x,y+1,z),(x,y+1,z+1),(x+1,y+1,z+1),(x+1,y+1,z)]),
                ((0,0,-1),[(x,y,z),(x,y+1,z),(x+1,y+1,z),(x+1,y,z)]),
                ((0,0,1),[(x,y,z+1),(x+1,y,z+1),(x+1,y+1,z+1),(x,y+1,z+1)])]
        for (dx,dy,dz),corners in panels:
            yy,zz,xx=y+dy,z+dz,x+dx
            if 0<=yy<occupied.shape[0] and 0<=zz<occupied.shape[1] and 0<=xx<occupied.shape[2] and occupied[yy,zz,xx]:
                continue
            ids=[vertex(tuple(map(float,p))) for p in corners]
            faces.extend([[ids[0],ids[1],ids[2]],[ids[0],ids[2],ids[3]]])
    return trimesh.Trimesh(vertices=np.asarray(vertices),faces=np.asarray(faces),process=False)


def cave_fixture(path):
    occupancy=np.zeros((24,24,24),dtype=np.uint8)
    occupancy[:10]=1
    occupancy[6:9,9:15,4:20]=0
    occupancy[6:10,10:14,4:8]=0
    ground=occupancy==1
    height=24-np.argmax(ground[::-1],axis=0)
    np.savez_compressed(path/"natural_occupancy.npz",occupancy=occupancy,validity=np.ones_like(ground),min_xyz=np.asarray([0,0,0]))
    np.savez_compressed(path/"terrain_surface.npz",height=height,validity=np.ones(height.shape,dtype=bool))
    (path/"world_ir.json").write_text(json.dumps({"content_sha256":"synthetic-cross-column-sky-cave"}))
    graph=source_topology(path)
    assert any(f["connected_to_observed_sky"] for f in graph["features"])
    return ground


def test_column_intervals_preserve_and_detect_capped_sky_connection(tmp_path):
    original=cave_fixture(tmp_path)
    mesh=voxel_boundary_mesh(original)
    assert mesh.is_volume
    mesh.export(tmp_path/"original.obj")
    result=validate_source_connectivity(tmp_path,tmp_path/"original.obj",tmp_path/"original_graph",np.eye(4))
    assert result["status"]=="pass"
    assert result["classified_source_centres"]==24**3
    assert result["protected_occupancy_mismatches"]==0
    assert not result["source_component_splits"]
    assert not result["target_component_merges"]
    assert result["roof_thickness"]["positive_thickness_and_inside_sign_preserved"]
    assert result["roof_thickness"]["one_metre_source_roof_samples"]>0
    assert result["roof_thickness"]["one_metre_roof_target_min_m"]==1
    # Cap the skylight above the source's covered-air/exterior transition.
    # Checking only the mouth positions would not prove this path to sky.
    capped=original.copy()
    capped[9,10:14,4:8]=True
    voxel_boundary_mesh(capped).export(tmp_path/"capped.obj")
    result=validate_source_connectivity(tmp_path,tmp_path/"capped.obj",tmp_path/"capped_graph",np.eye(4))
    assert result["status"]=="fail"
    assert result["source_component_splits"]
    assert result["changed_sky_connections"]
    displaced=mesh.copy()
    displaced.apply_translation((100,0,0))
    displaced.export(tmp_path/"missing_all_columns.obj")
    result=validate_source_connectivity(tmp_path,tmp_path/"missing_all_columns.obj",tmp_path/"missing_graph",np.eye(4))
    assert result["status"]=="fail"
    assert not result["roof_thickness"]["positive_thickness_and_inside_sign_preserved"]


def test_unintended_connection_between_closed_caves_is_rejected(tmp_path):
    occupancy=np.zeros((16,16,16),dtype=np.uint8)
    occupancy[:12]=1
    occupancy[3:6,4:7,3:6]=0
    occupancy[3:6,4:7,9:12]=0
    ground=occupancy==1
    np.savez_compressed(tmp_path/"natural_occupancy.npz",occupancy=occupancy,validity=np.ones_like(ground),min_xyz=np.asarray([0,0,0]))
    np.savez_compressed(tmp_path/"terrain_surface.npz",height=np.full((16,16),12),validity=np.ones((16,16),dtype=bool))
    (tmp_path/"world_ir.json").write_text(json.dumps({"content_sha256":"synthetic-two-caves"}))
    source_topology(tmp_path)
    breached=ground.copy()
    breached[4,5,6:9]=False
    voxel_boundary_mesh(breached).export(tmp_path/"breached.obj")
    result=validate_source_connectivity(tmp_path,tmp_path/"breached.obj",tmp_path/"breached_graph",np.eye(4))
    assert result["status"]=="fail"
    assert result["target_component_merges"]
    assert result["protected_occupancy_mismatches"]==3
