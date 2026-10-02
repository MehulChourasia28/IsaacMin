from pathlib import Path
import numpy as np
import pytest
import trimesh
from isaacmin.validation.native_segments import query_native_segments

ROOT=Path(__file__).resolve().parents[1]
pytestmark=pytest.mark.skipif(not (ROOT/'.tools/geometry_validation/segment_build_manifest.json').exists(),reason='Pinned isolated CGAL segment validator unavailable')


def run(tmp_path,vertices,faces,segments,name='query',limit=2_000_000):
    path=tmp_path/name;path.mkdir()
    np.save(path/'vertices.npy',np.asarray(vertices,np.float32));np.save(path/'triangles.npy',np.asarray(faces,np.int32))
    return query_native_segments(ROOT,path/'vertices.npy',path/'triangles.npy',segments,path/'evidence',maximum_leaf_faces=limit)


def test_actual_native_stacked_roof_queries_match_triangle_solver(tmp_path):
    a=trimesh.creation.box(extents=[4,4,1]);a.apply_translation([0,0,.5])
    b=trimesh.creation.box(extents=[4,4,1]);b.apply_translation([0,0,3.5])
    mesh=trimesh.util.concatenate((a,b))
    segments=np.array([[[.3,.2,-1],[.3,.2,5]],[[.3,.2,1.1],[.3,.2,2.9]],[[1.2,-.7,-1],[1.2,-.7,5]]])
    hits,report=run(tmp_path,mesh.vertices,mesh.faces,segments)
    assert report['complete'] and not report['coplanar_segment_intersections']
    for i,segment in enumerate(segments):
        wanted=np.array([0,1,3,4]) if i!=1 else np.empty(0)
        actual=np.unique(hits['first'][hits['query']==i,2])
        np.testing.assert_allclose(actual,wanted)
        direction=segment[1]-segment[0];length=np.linalg.norm(direction);direction/=length
        points,ray,face=mesh.ray.intersects_location([segment[0]],[direction],multiple_hits=True)
        points=points[np.linalg.norm(points-segment[0],axis=1)<=length]
        np.testing.assert_allclose(actual,np.unique(points[:,2]))


def test_partitioned_segment_evidence_matches_monolithic_original_ids(tmp_path):
    boxes=[]
    for x in (-10,0,10):
        box=trimesh.creation.box();box.apply_translation([x,0,0]);boxes.append(box)
    mesh=trimesh.util.concatenate(boxes)
    query=np.array([[[-12,.15,.2],[12,.15,.2]],[[0,-2,.1],[0,2,.1]],[[0,0,-2],[0,0,2]]])
    a,ar=run(tmp_path,mesh.vertices,mesh.faces,query,'one')
    b,br=run(tmp_path,mesh.vertices,mesh.faces,query,'partition',16)
    assert ar['complete'] and br['complete'] and br['native_measurements']['completed_leaves']>1
    np.testing.assert_array_equal(a,b)


def test_coplanar_segments_endpoints_and_near_parallel_hits_not_discarded(tmp_path):
    v=np.array([[0,0,0],[1,0,0],[0,1,0]],np.float32)
    q=np.array([[[-.1,.2,0],[.8,.2,0]],[[.2,.2,-1e-9],[.4,.2,1e-9]],[[.2,.2,0],[.2,.2,1]]])
    hits,report=run(tmp_path,v,[[0,1,2]],q)
    assert report['complete'] and report['coplanar_segment_intersections']==1
    assert list(hits['kind'])==[1,0,0]
    np.testing.assert_allclose(hits['first'][1],[.3,.2,0],atol=1e-15)
    np.testing.assert_allclose(hits['first'][2],[.2,.2,0],atol=1e-15)


def test_unsplittable_segment_query_is_incomplete_never_silently_empty(tmp_path):
    _,report=run(tmp_path,[[0,0,0],[1,0,0],[0,1,0]],[[0,1,2],[0,1,2]],[[[.2,.2,-1],[.2,.2,1]]],limit=1)
    assert report['status']=='incomplete' and not report['complete']


def test_six_axis_parity_and_opposite_diagonal_boundary_resolution(tmp_path):
    from isaacmin.validation.native_source import DIRECTIONS,classify_native_parity
    mesh=trimesh.creation.box(extents=[2,2,2])
    points=np.array([[0,0,0],[.73,.42,.17],[2,.13,.23],[1,0,0]])
    origins=np.repeat(points,len(DIRECTIONS),axis=0);directions=np.tile(DIRECTIONS,(len(points),1))
    segments=np.stack([origins,origins+directions*10],axis=1)
    hits,report=run(tmp_path,mesh.vertices,mesh.faces,segments)
    inside,evidence=classify_native_parity(hits,points)
    assert report['complete']
    np.testing.assert_array_equal(inside[:3],[True,True,False])
    assert not evidence['unresolved'][:3].any()
    assert evidence['unresolved'][3] and evidence['diagonal_fallback'][3]


def test_opposite_facing_coincident_faces_remain_parity_ambiguous(tmp_path):
    from isaacmin.validation.native_source import DIRECTIONS,classify_native_parity
    # A sheet and its duplicate reverse cannot become a closed solid by parity.
    v=np.array([[-1,-1,0],[1,-1,0],[0,1,0]],np.float32)
    point=np.array([[0,0,-.03]])
    directions=DIRECTIONS;origins=np.repeat(point,len(directions),axis=0)
    hits,_=run(tmp_path,v,[[0,1,2],[0,2,1]],np.stack([origins,origins+directions*10],axis=1))
    inside,evidence=classify_native_parity(hits,point)
    assert evidence['axis_disagreement'][0] or evidence['ambiguous_rays'][0].any()
    assert evidence['unresolved'][0]
