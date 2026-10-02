"""Independent exact geometry properties of the measured native fold repair."""
from collections import Counter
from pathlib import Path
import itertools
import numpy as np
import pytest
from isaacmin.volumes.planar_patch import triangulate_patch
from isaacmin.validation.surface_overlap import exact_coplanar_intersection


def boundary(faces):
    directed=Counter((int(a),int(b)) for face in faces for a,b in zip(face,np.roll(face,-1)))
    return {edge for edge in directed if (edge[1],edge[0]) not in directed}


def overlaps(vertices,faces):
    return sum(exact_coplanar_intersection(vertices[a],vertices[b])['positive_area_overlap']
               for a,b in itertools.combinations(faces,2))


@pytest.mark.parametrize('axes',[(0,1,2),(2,0,1),(1,0,2)])
def test_measured_fold_repair_preserves_sampling_boundary_and_all_points(axes):
    data=np.load(Path(__file__).parent/'fixtures/native_planar_fold.npz')
    vertices=data['vertices'][:,axes].copy();original=vertices.copy();faces=data['triangles']
    assert overlaps(vertices,faces)>0
    result,report=triangulate_patch(vertices,faces)
    assert overlaps(vertices,result)==0
    np.testing.assert_array_equal(vertices,original)
    np.testing.assert_array_equal(np.unique(result),np.unique(faces))
    assert boundary(result)==boundary(faces)
    assert len(result)==len(faces)
    original_max=np.linalg.norm(vertices[faces].astype(float)-vertices[faces[:,[1,2,0]]].astype(float),axis=2).max()
    assert report['maximum_edge_m']<=original_max
    again,_=triangulate_patch(vertices,faces[::-1])
    np.testing.assert_array_equal(result,again)


def test_repair_rejects_nonplanar_surface_and_self_crossing_fixed_boundary():
    data=np.load(Path(__file__).parent/'fixtures/native_planar_fold.npz')
    vertices=data['vertices'].copy();faces=data['triangles']
    moved=vertices.copy();moved[0,2]+=.001
    with pytest.raises(ValueError,match='not exactly planar'):
        triangulate_patch(moved,faces)
    # The original five-face fan alone has a self-crossing fixed boundary;
    # blindly retriangulating it would change the surface ownership contract.
    centre=np.flatnonzero(np.all(vertices==[np.float32(33.65182876586914),np.float32(-32.93450927734375),np.float32(72)],axis=1))[0]
    star=faces[np.any(faces==centre,axis=1)]
    with pytest.raises(ValueError,match='self-intersects'):
        triangulate_patch(vertices,star)


def test_actual_nonplanar_native_crossing_retriangulation():
    from isaacmin.volumes.projected_patch import triangulate_projected_patch
    data=np.load(Path(__file__).parent/'fixtures/native_nonplanar_crossing.npz')
    points,faces=data['vertices'],data['triangles']
    expected={tuple(sorted(t)) for t in ([0,1,2],[0,2,3],[0,3,5],[3,4,5])}
    for axis in (1,2):
        repaired,report=triangulate_projected_patch(points,faces,axis)
        assert {tuple(sorted(t)) for t in repaired}==expected
        assert not report['exactly_planar_original_patch']
        assert report['vertices_preserved']==6 and report['boundary_edges_preserved']==6
        assert report['replacement_faces']==report['original_faces']==4
        assert report['maximum_edge_m']<=np.linalg.norm(points[faces].astype(float)-points[faces[:,[1,2,0]]].astype(float),axis=2).max()
    with pytest.raises(ValueError,match='boundary self-intersects'):
        triangulate_projected_patch(points,faces,0)
