from pathlib import Path

import numpy as np
import pytest
import trimesh

from isaacmin.validation.global_intersections import validate_global_intersections

ROOT=Path(__file__).resolve().parents[1]
pytestmark=pytest.mark.skipif(not (ROOT/'.tools/geometry_validation/build_manifest.json').exists(),reason='Pinned isolated native CGAL validator not installed')


def run(tmp_path,vertices,faces,name='check',limit=2_000_000):
    path=tmp_path/name;path.mkdir()
    np.save(path/'vertices.npy',np.asarray(vertices,np.float32));np.save(path/'triangles.npy',np.asarray(faces,np.int32))
    return validate_global_intersections(ROOT,path/'vertices.npy',path/'triangles.npy',path/'evidence',maximum_leaf_faces=limit)


def test_actual_native_predicate_excludes_shared_boundary_but_finds_coplanar_fold(tmp_path):
    v=[[0,0,0],[1,0,0],[0,1,0],[1,1,0]]
    assert run(tmp_path,v,[[0,1,2],[1,3,2]],'valid')['status']=='pass'
    v[3]=[.25,.25,0]
    result=run(tmp_path,v,[[0,1,2],[1,3,2]],'fold')
    assert result['status']=='fail' and result['complete']
    assert result['intersection_pair_count']==1


def test_noncoplanar_intersection_and_degenerate_original_face_are_detected(tmp_path):
    v=[[0,0,0],[2,0,0],[0,2,0],[.5,.5,-1],[.5,.5,1],[1.5,.5,0]]
    result=run(tmp_path,v,[[0,1,2],[3,4,5]],'cross')
    assert result['intersection_pair_count']==1
    result=run(tmp_path,[[0,0,0],[1,0,0],[2,0,0]],[[0,1,2]],'degenerate')
    assert result['degenerate_triangles']==1


def test_adaptive_partition_matches_monolithic_pairs_without_duplicates(tmp_path):
    boxes=[]
    for x in (-10,0,10):
        box=trimesh.creation.box();box.apply_translation([x,0,0]);boxes.append(box)
    mesh=trimesh.util.concatenate(boxes)
    extra=trimesh.Trimesh([[0,0,-1],[0,0,1],[1,0,0]],[[0,1,2]],process=False)
    mesh=trimesh.util.concatenate((mesh,extra))
    a=run(tmp_path,mesh.vertices,mesh.faces,'monolithic')
    b=run(tmp_path,mesh.vertices,mesh.faces,'partitioned',limit=16)
    assert a['complete'] and b['complete']
    assert b['native_measurements']['completed_leaves']>1
    def pairs(name):return set(map(tuple,np.fromfile(tmp_path/name/'evidence/intersection_pairs.u64',dtype='<u8').reshape(-1,2)))
    assert pairs('monolithic')==pairs('partitioned')
    assert a['intersection_pair_count']==b['intersection_pair_count']>0


def test_unsplittable_cell_is_incomplete_never_a_pass(tmp_path):
    result=run(tmp_path,[[0,0,0],[1,0,0],[0,1,0]],[[0,1,2],[0,1,2]],limit=1)
    assert result['status']=='incomplete' and not result['complete']


def test_recorded_actual_float32_zero_normal_star_is_a_global_overlap_regression(tmp_path):
    # Exact coordinates/identity from the retained real-map attempt-3 witness
    # actual_incident_triangles.npz; this reduced test is algorithmic evidence.
    # All five faces have positive area, but four face pairs overlap in area.
    vertices=np.array([
        [33.56636047363281,-32.9330940246582,72.],
        [33.65182876586914,-32.93450927734375,72.],
        [33.581993103027344,-32.93335723876953,72.],
        [33.72053909301758,-32.93566131591797,72.],
        [33.736968994140625,-32.93593215942383,72.],
        [33.682987213134766,-32.935035705566406,72.],
    ],np.float32)
    faces=np.array([[0,1,2],[0,3,1],[1,4,5],[1,3,4],[2,1,5]],np.int32)
    triangle=vertices[faces].astype(float)
    assert np.all(np.linalg.norm(np.cross(triangle[:,1]-triangle[:,0],triangle[:,2]-triangle[:,0]),axis=1)>0)
    result=run(tmp_path,vertices,faces,'actual_star')
    assert result['status']=='fail' and result['complete'] and result['intersection_pair_count']==4
    from itertools import combinations
    from isaacmin.validation.surface_overlap import exact_coplanar_intersection
    expected={pair for pair in combinations(range(len(faces)),2) if exact_coplanar_intersection(triangle[pair[0]],triangle[pair[1]])['positive_area_overlap']}
    actual=set(map(tuple,np.fromfile(tmp_path/'actual_star/evidence/intersection_pairs.u64',dtype='<u8').reshape(-1,2)))
    assert actual==expected
