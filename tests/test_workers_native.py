"""Meaningful external CPU tests; never claim Isaac on a mocked subprocess."""
from pathlib import Path
import numpy as np
import pytest
from isaacmin.adapters.workers import project_root,refine_heightfield,reconstruct_occupancy


def test_invalid_native_arrays_fail_before_process(tmp_path):
    with pytest.raises(ValueError,match='finite'):
        refine_heightfield(np.full((8,8),np.nan),tmp_path)
    with pytest.raises(ValueError,match='shape'):
        refine_heightfield(np.zeros((8,8)),tmp_path,protection=np.zeros((2,2)))
    with pytest.raises(ValueError,match='Boolean'):
        reconstruct_occupancy(np.ones((4,4,4),dtype='u1'),tmp_path)


@pytest.mark.skipif(not (project_root()/'.tools/build/native-gcc13/isaacmin_highmap').is_file(),reason='Native HighMap not installed')
def test_actual_cpu_erosion_constraints_and_shared_runoff(tmp_path):
    x,y=np.meshgrid(np.arange(33),np.arange(33))
    height=(40+0.1*x+0.07*y+0.2*np.sin(x/4)).astype('f4')
    protect=(x==16)&(y<20)
    runoff=((x+1)*(y+1)/1089).astype('f4')
    result=refine_heightfield(height,tmp_path,protection=protect,global_runoff=runoff,global_runoff_normalized=True)
    assert result['status']=='success'
    arrays=np.load(result['arrays'])
    np.testing.assert_array_equal(arrays['height'][protect],height[protect])
    assert np.count_nonzero(arrays['delta'])>500
    assert np.all(arrays['delta']>=-0.20-1e-5)
    np.testing.assert_allclose(arrays['runoff'],runoff,rtol=0,atol=0)
    assert result['global_drainage_sha256']
    # Different tile extents and execution order must preserve identical shared points.
    a=refine_heightfield(height[:,:24],tmp_path/'a',protection=protect[:,:24],
                         global_runoff=runoff[:,:24],global_runoff_normalized=True)
    b=refine_heightfield(height[:,9:],tmp_path/'b',protection=protect[:,9:],
                         global_runoff=runoff[:,9:],global_runoff_normalized=True)
    np.testing.assert_array_equal(np.load(a['arrays'])['height'][:,9:24],np.load(b['arrays'])['height'][:,:15])


@pytest.mark.skipif(not (project_root()/'.tools/openvdb_worker').is_file(),reason='Native OpenVDB not installed')
def test_actual_volume_mesh_is_closed_and_retains_tunnel(tmp_path):
    occupied=np.zeros((12,12,12),bool);occupied[:9,:,:]=True
    occupied[2:6,4:8,:]=False
    result=reconstruct_occupancy(occupied,tmp_path)
    assert result['status']=='success'
    vertices=[];edges={};triangles=[]
    for line in Path(result['mesh']).read_text().splitlines():
        if line.startswith('v '):vertices.append([float(v) for v in line.split()[1:]])
        elif line.startswith('f '):
            face=[int(v)-1 for v in line.split()[1:]]
            triangles.extend((face[0],face[i],face[i+1]) for i in range(1,len(face)-1))
            for a,b in zip(face,face[1:]+face[:1]):
                key=tuple(sorted((a,b)));edges[key]=edges.get(key,0)+1
    assert all(count==2 for count in edges.values())
    points=np.asarray(vertices)
    tri=points[np.asarray(triangles)]
    signed_volume=np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()/6
    assert signed_volume>0, 'OBJ winding must orient closed terrain solids outward'
    # Geometry includes an internal floor and ceiling, not only exterior top.
    internal=(points[:,1]>-7)&(points[:,1]<-4)&(points[:,0]>1)&(points[:,0]<10)
    assert np.any(internal&(np.abs(points[:,2]-1.5)<0.1))
    assert np.any(internal&(np.abs(points[:,2]-5.5)<0.1))


@pytest.mark.skipif(not (project_root()/'.tools/openvdb_worker').is_file(),reason='Native OpenVDB not installed')
def test_source_projection_preserves_thin_roof_between_constraint_samples(tmp_path):
    import trimesh
    from isaacmin.volumes.interface_projection import constrain_interfaces
    occupied=np.zeros((12,12,12),bool);occupied[:9]=True
    occupied[2:8,3:9,:]=False
    covered=np.zeros_like(occupied);covered[2:8,3:9,:]=True
    native=reconstruct_occupancy(occupied,tmp_path/'native')
    candidate=constrain_interfaces(Path(native['mesh']),tmp_path/'projected',supporting_solid=occupied,
        covered_air=covered,min_xyz=np.array([0,0,0]),exterior_height=np.full((12,12),9.),
        exterior_validity=np.ones((12,12),bool),known_source_air=~occupied)
    target=trimesh.load(candidate['mesh'],force='mesh',process=False)
    assert target.is_volume
    # Held-out locations are deliberately different from the construction grid.
    x,z=np.meshgrid(5+np.array([-.33,.11,.37]),5+np.array([-.31,.07,.36]))
    origins=np.column_stack((x.ravel(),-z.ravel(),np.full(x.size,12.)))
    points,rays,_=target.ray.intersects_location(origins,np.tile([0,0,-1.],(x.size,1)),multiple_hits=True)
    for index in range(x.size):
        levels=np.sort(points[rays==index,2])
        np.testing.assert_allclose(levels[-3:],[1.5,7.5,8.5],rtol=0,atol=.002)
        assert abs(levels[-1]-levels[-2]-1)<=.004
