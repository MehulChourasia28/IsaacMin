import numpy as np
import trimesh
import json
from isaacmin.io import sha256_file

from isaacmin.validation.continuous_surface import validate_continuous_surface


def save(path,mesh):
    np.savez(path,vertices=mesh.vertices,faces=mesh.faces)
    return path


def test_closed_surface_and_inward_void_are_oriented_by_solid_parity(tmp_path):
    outer=trimesh.creation.box(extents=[4,4,4]);inner=trimesh.creation.box();inner.invert()
    mesh=trimesh.util.concatenate([outer,inner]);path=save(tmp_path/'void.npz',mesh)
    report=validate_continuous_surface(path,tmp_path/'pass')
    assert report['status']=='pass'
    assert report['edge_connected_shells']==2
    assert report['shell_orientation']['status']=='pass'
    assert report['full_Q04_status']=='not_run'
    # Flipping the void shell alone keeps total signed volume positive and all
    # edge incidences valid. Independent shell air/solid orientation must fail.
    inner.invert();path=save(tmp_path/'wrong.npz',trimesh.util.concatenate([outer,inner]))
    report=validate_continuous_surface(path,tmp_path/'wrong')
    assert report['status']=='fail'
    assert report['failures']['shell_orientation_failures']==1


def test_exact_attribute_seams_are_not_physical_cracks_but_nonzero_gaps_are(tmp_path):
    cube=trimesh.creation.box();v=cube.vertices[cube.faces].reshape(-1,3);f=np.arange(len(v)).reshape(-1,3)
    path=tmp_path/'seams.npz';np.savez(path,vertices=v,faces=f)
    report=validate_continuous_surface(path,tmp_path/'seams')
    assert report['status']=='pass' and report['attribute_seam_duplicate_positions']>0
    v[0,0]+=1e-9;np.savez(tmp_path/'crack.npz',vertices=v,faces=f)
    report=validate_continuous_surface(tmp_path/'crack.npz',tmp_path/'crack')
    assert report['status']=='fail' and report['failures']['unexplained_boundary_edges']>0


def test_duplicate_opposite_face_and_float32_collapse_are_explicit(tmp_path):
    cube=trimesh.creation.box();faces=np.vstack([cube.faces,cube.faces[0,::-1]])
    path=tmp_path/'duplicate.npz';np.savez(path,vertices=cube.vertices,faces=faces)
    report=validate_continuous_surface(path,tmp_path/'duplicate')
    assert report['status']=='fail' and report['failures']['duplicate_triangles']==1
    # A valid double triangle whose third vertex is lost in the actual target
    # position format must fail before native subdivision/normals are trusted.
    vertices=np.asarray([[1000.,0,0],[1001.,0,0],[1000.,1e-7,0]])
    vertices[:,1]+=1000.
    np.savez(tmp_path/'precision.npz',vertices=vertices,faces=np.asarray([[0,1,2]]))
    report=validate_continuous_surface(tmp_path/'precision.npz',tmp_path/'precision',numerics_only=True)
    assert report['status']=='fail'
    assert report['failures']['float64_degenerate_triangles']==0
    assert report['failures']['native_float32_degenerate_triangles']==1


def test_open_payload_requires_named_seam_and_does_not_qualify_world(tmp_path):
    cube=trimesh.creation.box();keep=~np.all(cube.vertices[cube.faces][:,:,0]==.5,axis=1)
    cube.update_faces(keep);path=save(tmp_path/'payload.npz',cube)
    failed=validate_continuous_surface(path,tmp_path/'missing',topology_scope='runtime_payload')
    assert failed['status']=='fail'
    report=validate_continuous_surface(path,tmp_path/'declared',topology_scope='runtime_payload',
        declared_seam_planes=[{'axis':'x','coordinate_m':.5,'join_id':'east','neighbor_owner':'next_payload'}])
    assert report['status']=='pass'
    assert report['declared_payload_seam_edge_counts']=={'east':4}
    assert report['full_Q04_status']=='not_run'


def test_nonfinite_native_array_rows_are_reported_without_deletion(tmp_path):
    path=tmp_path/'native';path.mkdir();np.save(path/'vertices.npy',np.asarray([[np.nan,0,0],[1,0,0],[0,1,0]]));np.save(path/'triangles.npy',np.asarray([[0,1,2]],np.int32))
    report=validate_continuous_surface(path,tmp_path/'nonfinite')
    assert report['status']=='fail' and report['nonfinite_vertex_count']==1


def test_duplicated_face_ownership_is_not_hidden_by_opposite_winding(tmp_path):
    cube=trimesh.creation.box();faces=np.vstack([cube.faces,cube.faces[0,::-1]])
    path=tmp_path/'mesh.npz';np.savez(path,vertices=cube.vertices,faces=faces)
    owner=tmp_path/'owners.npz';np.savez(owner,owner_id=np.r_[np.zeros(12,np.int32),1],
        owner_names=np.asarray(['left_payload','right_payload']),input_sha256=np.asarray(sha256_file(path)))
    report=validate_continuous_surface(path,tmp_path/'result',face_ownership_path=owner)
    assert report['status']=='fail'
    assert report['face_ownership']['duplicate_faces_with_different_owners']==1


def test_initial_crop_caps_are_labelled_but_internal_capped_cubes_fail(tmp_path):
    source=tmp_path/'source';source.mkdir()
    occupancy=np.ones((1,1,2),np.uint8)
    np.savez(source/'natural_occupancy.npz',occupancy=occupancy,min_xyz=np.zeros(3,dtype=int))
    artifact=source/'natural_occupancy.npz'
    (source/'world_ir.json').write_text(json.dumps({'files':[{'path':artifact.name,'sha256':sha256_file(artifact)}]}))
    closed=trimesh.creation.box(extents=[2,1,1]);closed.apply_translation([1,.5,.5]);path=save(tmp_path/'closed.npz',closed)
    report=validate_continuous_surface(path,tmp_path/'closed',source_ir=source,mesh_to_source=np.eye(4))
    assert report['status']=='pass'
    assert set(report['source_crop_boundary']['plane_face_counts'].values())=={2}
    assert 'not an internal tile join' in report['source_crop_boundary']['interpretation']
    left=trimesh.creation.box();left.apply_translation([.5,.5,.5]);right=left.copy();right.apply_translation([1,0,0])
    joined=save(tmp_path/'bad_join.npz',trimesh.util.concatenate([left,right]))
    result=validate_continuous_surface(joined,tmp_path/'bad_join',source_ir=source,mesh_to_source=np.eye(4),topology_scope='composed_world')
    assert result['status']=='fail'
    assert result['failures']['nonmanifold_edges']>0 or result['failures']['duplicate_triangles']>0
