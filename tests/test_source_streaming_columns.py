import numpy as np
import pytest
import trimesh

from isaacmin.validation.streaming_columns import streamed_column_intersections


def deduplicate(rays,heights):
    order=np.lexsort((heights,rays));rays=np.asarray(rays)[order];heights=np.asarray(heights)[order]
    keep=np.r_[True,(rays[1:]!=rays[:-1])|(np.diff(heights)>1e-7)] if len(rays) else np.zeros(0,bool)
    return rays[keep],heights[keep]


def reference(mesh,minimum,shape):
    z,x=np.indices(shape);origins=np.column_stack((x.ravel()+minimum[0]+.5,np.full(x.size,-50.),z.ravel()+minimum[1]+.5))
    locations,rays,_=mesh.ray.intersects_location(origins,np.tile([0.,1.,0.],(len(origins),1)),multiple_hits=True)
    return deduplicate(rays,np.asarray(locations).reshape(-1,3)[:,1])


@pytest.mark.parametrize('translated',[False,True])
def test_streaming_matches_original_solver_on_stacked_roofs_and_edge_rays(translated):
    floor=trimesh.creation.box(extents=(6,2,6));floor.apply_translation((3,0,3))
    roof=trimesh.creation.box(extents=(6,1,6));roof.apply_translation((3,5.5,3))
    raised=trimesh.creation.box(extents=(2,1,2));raised.apply_translation((3,10.5,3))
    mesh=trimesh.util.concatenate((floor,roof,raised))
    # Quarter-turn + translation exercise the exact source/world axis map.
    frame=np.asarray([[1,0,0,-100.5],[0,0,1,4.],[0,-1,0,20.5],[0,0,0,1.]]) if translated else np.eye(4)
    native=mesh.copy();native.apply_transform(np.linalg.inv(frame))
    result=streamed_column_intersections(native.vertices,native.faces,frame,[0,0],(6,6),triangle_batch=5,candidate_batch=7)
    actual=deduplicate(result['ray_index'],result['source_y']);expected=reference(mesh,[0,0],(6,6))
    assert np.array_equal(actual[0],expected[0]);assert np.allclose(actual[1],expected[1],atol=1e-12,rtol=0)
    assert result['metrics']['original_triangles_visited']==len(mesh.faces)
    assert result['metrics']['requested_columns']==36
    assert result['metrics']['maximum_candidate_batch']<=7
    assert result['exact_edge_hit'].any()


def test_nonparallel_near_vertical_intersection_is_retained_and_difference_reported():
    vertices=np.asarray([[.5-2e-9,-10,0],[.5+6e-9,10,0],[.5-2e-9,10,1.]])
    faces=np.asarray([[0,1,2]])
    result=streamed_column_intersections(vertices,faces,np.eye(4),[0,0],(1,1))
    assert len(result['source_y'])==1
    assert abs(result['source_y'][0]-5)<1e-5
    assert result['metrics']['hits_below_trimesh_plane_cosine_cutoff']==1
    mesh=trimesh.Trimesh(vertices,faces,process=False)
    assert not len(reference(mesh,[0,0],(1,1))[0])


def test_vertical_tangent_and_invalid_original_faces_are_not_silently_deleted():
    vertices=np.asarray([[.5,-1,0],[.5,1,0],[.5,1,1.]])
    result=streamed_column_intersections(vertices,np.asarray([[0,1,2]]),np.eye(4),[0,0],(1,1))
    assert result['vertical_face_tangent_ray_triangle'].tolist()==[[0,0]]
    with pytest.raises(ValueError,match='Degenerate'):
        streamed_column_intersections(vertices,np.asarray([[0,0,2]]),np.eye(4),[0,0],(1,1))
    vertices[0,0]=np.nan
    with pytest.raises(ValueError,match='Nonfinite'):
        streamed_column_intersections(vertices,np.asarray([[0,1,2]]),np.eye(4),[0,0],(1,1))
