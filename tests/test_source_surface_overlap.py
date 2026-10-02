import numpy as np
import trimesh

from isaacmin.validation.surface_overlap import exact_coplanar_intersection,shared_edge_overlap_candidates


def test_exact_predicate_distinguishes_shared_edge_overlap_and_parallel_separation():
    a=np.asarray([[0,0,0],[1,0,0],[.3,.2,0]],float)
    b=np.asarray([[1,0,0],[0,0,0],[.6,.1,0]],float)
    proof=exact_coplanar_intersection(a,b)
    assert proof['positive_area_overlap']
    assert 0<proof['projected_overlap_area_m2']<.1
    b[2,1]=-.1
    assert not exact_coplanar_intersection(a,b)['positive_area_overlap']
    # No epsilon converts a separated parallel surface into a true overlap.
    b=a.copy();b[:,2]+=np.nextafter(0.,1.)
    assert exact_coplanar_intersection(a,b)['status']=='not_exactly_coplanar'


def test_shared_edge_fold_detection_is_independent_of_winding_and_translation():
    vertices=np.asarray([[0,0,0],[1,0,0],[.3,.2,0],[.6,.1,0]],float)+[33.,-32.,72.]
    faces=np.asarray([[0,1,2],[1,0,3]])
    result=shared_edge_overlap_candidates(vertices,faces,[0],[1],[[0,1]])
    assert len(result)==1
    assert result[0]['face_ids']==[0,1]
    # Reversing the second face does not remove the physical overlap.
    faces[1]=faces[1,::-1]
    assert len(shared_edge_overlap_candidates(vertices,faces,[0],[1],[[0,1]]))==1
    vertices[3,1]=-32.1
    assert not shared_edge_overlap_candidates(vertices,faces,[0],[1],[[0,1]])


def test_tiny_binary_triangle_overlap_is_not_rounded_to_zero():
    a=np.asarray([[0,0,0],[2**-20,0,0],[0,2**-20,0]],float)
    proof=exact_coplanar_intersection(a,a[[1,0,2]])
    assert proof['positive_area_overlap']
    assert proof['projected_overlap_area_exact']==f'1/{2**41}'


def test_closed_positive_consistently_wound_surface_still_fails_for_folded_fan(tmp_path):
    from isaacmin.validation.continuous_surface import validate_continuous_surface
    vertices=np.asarray([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                         [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1],[1.1,0,1]],float)
    faces=[[0,2,1],[0,3,2]]
    for i in range(4):
        j=(i+1)%4;faces.extend([[i,j,j+4],[i,j+4,i+4]])
    faces.extend([[4,5,8],[5,6,8],[6,7,8],[7,4,8]])
    mesh=trimesh.Trimesh(vertices,faces,process=False)
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.volume>0
    path=tmp_path/'folded.npz';np.savez(path,vertices=vertices,faces=np.asarray(faces))
    result=validate_continuous_surface(path,tmp_path/'report')
    assert result['status']=='fail'
    assert result['failures']['coplanar_shared_edge_overlaps']==2
    assert result['failures']['inconsistent_winding_edges']==0
    assert result['failures']['shell_orientation_failures']==0
    assert result['global_self_intersection_test']=='not_run'
