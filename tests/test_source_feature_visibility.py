import numpy as np
import pytest
import trimesh
import json
from pathlib import Path

from isaacmin.validation.feature_visibility import bounded_first_hits, _source_labels
from isaacmin.validation.feature_visibility import collect_feature_visibility
from isaacmin.contracts.coordinates import CoordinateFrame
from test_source_connectivity import cave_fixture


def test_bounded_independent_triangle_segments_match_original_and_keep_occluders():
    outer=trimesh.creation.box(extents=[8,8,8])
    inner=trimesh.creation.box(extents=[2,2,2]);inner.invert()
    mesh=trimesh.util.concatenate([outer,inner])
    rng=np.random.default_rng(632)
    origins=rng.uniform(-7,7,(128,3));directions=rng.normal(size=origins.shape)
    directions/=np.linalg.norm(directions,axis=1)[:,None]
    lengths=rng.uniform(.2,15,len(origins))
    distance,faces=bounded_first_hits(mesh,origins,directions,lengths)
    hits,rays,_=mesh.ray.intersects_location(origins,directions,multiple_hits=False)
    expected=np.full(len(origins),np.nan)
    values=np.linalg.norm(hits-origins[rays],axis=1)
    valid=values<=lengths[rays]
    expected[rays[valid]]=values[valid]
    assert np.allclose(distance,expected,atol=1e-9,equal_nan=True)
    # The source/native depth might name the far wall. The complete prefix is
    # queried, so an earlier independent blocker is retained as a discrepancy.
    distances,_=bounded_first_hits(mesh,[[-6,0,0]],[[1,0,0]],[11])
    assert distances[0] == 2
    assert not np.isclose(distances[0],10)


def test_visibility_source_association_never_promotes_unknown_or_nonair():
    labels=np.full((4,4,4),7,dtype=np.int32);valid=np.ones_like(labels,bool);air=np.ones_like(labels,bool)
    valid[1,2,3]=False;air[2,1,1]=False
    points=np.asarray([[3.5,1.5,2.5],[1.5,2.5,1.5],[1.5,1.5,1.5],[-.5,1,1]])
    assert _source_labels(points,np.zeros(3),labels,valid,air).tolist()==[0,0,7,0]


def test_visibility_segment_bad_inputs_fail_without_dropping_rows():
    with pytest.raises(ValueError,match="Nonfinite"):
        bounded_first_hits(trimesh.creation.box(),[[np.nan,0,0]],[[1,0,0]],[1])


def test_standalone_feature_collector_rejects_stale_topology_before_mesh_or_capture(tmp_path):
    cave_fixture(tmp_path)
    # A minimal constructed IR can bind the fixture source while its topology
    # graph still carries the correct original source identity.
    ir=tmp_path/"world_ir.json";meta=json.loads(ir.read_text());meta.setdefault("files",[])
    ir.write_text(json.dumps(meta))
    from isaacmin.source.snapshot import sha256
    graph_path=tmp_path/"topology/source_topology_graph.json";graph=json.loads(graph_path.read_text())
    graph["source_ir_sha256"]=sha256(ir);graph_path.write_text(json.dumps(graph))
    labels=tmp_path/"topology/source_topology_labels.npz";labels.write_bytes(labels.read_bytes()+b"changed")
    with pytest.raises(ValueError,match="topology payload changed"):
        collect_feature_visibility(tmp_path/"missing.obj",tmp_path,[],CoordinateFrame(),tmp_path/"rejected")
