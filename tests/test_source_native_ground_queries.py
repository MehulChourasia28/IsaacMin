"""Actual CPU native geometry queries on explicitly synthetic fixtures."""
from pathlib import Path
import numpy as np
import pytest
import trimesh
from isaacmin.validation.native_proximity import query_native_proximity
from isaacmin.validation.ground_queries import GroundQueries
from isaacmin.validation.cave_views import create_cave_view_plan
from test_source_native_integration import write_native_fixture
from test_source_fidelity import real_reader_fixture

ROOT=Path(__file__).resolve().parents[1]
pytestmark=pytest.mark.skipif(not (ROOT/'.tools/geometry_validation/proximity_build_manifest.json').exists(),reason='Pinned isolated CGAL proximity worker unavailable')


def test_global_proximity_keeps_every_leaf_and_nearest_original_face(tmp_path):
    first=trimesh.creation.box();second=trimesh.creation.box();second.apply_translation((20,0,3));mesh=trimesh.util.concatenate([first,second])
    v=tmp_path/'v.npy';f=tmp_path/'f.npy';np.save(v,np.asarray(mesh.vertices,np.float32));np.save(f,np.asarray(mesh.faces,np.int32))
    points=np.array([[0,0,2],[20,.2,4],[10,0,0],[-.5,0,0],[20,0,3]])
    measured,report=query_native_proximity(ROOT,v,f,points,tmp_path/'query',maximum_leaf_faces=12)
    closest,distance,_=trimesh.proximity.closest_point(mesh,points)
    assert report['complete'] and report['native_measurements']['completed_leaves']>1
    np.testing.assert_allclose(np.sqrt(measured['distance2']),distance,rtol=0,atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(points-measured['point'],axis=1),distance,rtol=0,atol=1e-12)
    assert np.array_equal(measured['triangle'][:2]//12,[0,1])


def test_ground_batches_preserve_occlusion_boundary_and_clearance(tmp_path):
    mesh=trimesh.creation.box(extents=(4,4,1));mesh.apply_translation((0,0,-.5));obj=write_native_fixture(tmp_path/'scene',mesh.vertices,mesh.faces);queries=GroundQueries(obj,tmp_path/'evidence/queries')
    points=np.array([[0,0,.5],[0,0,-.5],[0,0,0.]])
    assert queries.contains(points,reject_ambiguous=True).tolist()==[False,True,True]
    with pytest.raises(ValueError,match='ambiguous'):queries.contains(points)
    hit,normal,ids=queries.first_hits(np.array([[0,0,.5],[3,3,.5]]),np.tile([0,0,-1.],(2,1)))
    assert ids[0]>=0 and ids[1]==-1 and hit[0,2]==0 and normal[0,2]==1
    assert queries.unobstructed([[0,0,.2],[0,0,.2]],[[0,0,.4],[0,0,-.1]]).tolist()==[True,False]
    assert queries.receipt()['sha256']


def test_batched_native_cave_plan_matches_legacy_source_scopes(tmp_path):
    ir,mesh,frame,_=real_reader_fixture(tmp_path);mesh.apply_transform(frame.record()['source_to_world']);legacy=tmp_path/'legacy.obj';mesh.export(legacy)
    before=create_cave_view_plan(legacy,ir,frame,candidate_budget=32,max_interior_views=3,max_portal_views=2)
    native=write_native_fixture(tmp_path/'scene',mesh.vertices,mesh.faces)
    after=create_cave_view_plan(native,ir,frame,tmp_path/'evidence/cave_plan.json',candidate_budget=32,max_interior_views=3,max_portal_views=2)
    assert after['available_features']==before['available_features']
    assert len(after['poses_static'])==len(before['poses_static'])
    assert after['support_routes'] and after['native_query_evidence']
    for pose in after['poses_static']:
        assert pose['camera_nearest_triangle_m']>=.12 and pose['support_normal'][2]>=.9
    for route in after['support_routes']:
        assert len(route['footprint_ground_world_xyz'])==7 and route['body_minimum_sampled_mesh_clearance_m']>=.01
