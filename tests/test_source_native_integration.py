"""Actual CPU native predicates on declared synthetic integration fixtures."""
import json
import copy
from pathlib import Path
import numpy as np
import pytest
from isaacmin.io import sha256_file
from isaacmin.validation.native_ground import bind_native_ground
from isaacmin.validation.source_fidelity import evaluate_source_fidelity
from isaacmin.volumes.mesh_validation import validate_source_mesh,validate_source_connectivity,verify_mesh_backend_evidence
from test_source_fidelity import real_reader_fixture

ROOT=Path(__file__).resolve().parents[1]
pytestmark=pytest.mark.skipif(not (ROOT/'.tools/geometry_validation/obj_compare_build_manifest.json').exists(),reason='Pinned isolated native OBJ comparator unavailable')


def write_native_fixture(scene,vertices,faces,*,decimals=None):
    scene.mkdir();native=scene/'native_precision';native.mkdir()
    vertices=np.asarray(vertices,np.float32);faces=np.asarray(faces,np.int32)
    np.save(native/'final_vertices.npy',vertices);np.save(native/'final_triangles.npy',faces)
    (native/'finalization.json').write_text(json.dumps({'fixture_only':True,'final_vertices_sha256':sha256_file(native/'final_vertices.npy'),'final_triangles_sha256':sha256_file(native/'final_triangles.npy')}))
    obj=scene/'final_ground.obj'
    with obj.open('w') as stream:
        for p in vertices:stream.write('v '+' '.join(format(float(v),'.17g' if decimals is None else '.6f') for v in p)+'\n')
        for p in faces:stream.write('f '+' '.join(str(int(v)+1) for v in p)+'\n')
    return obj


def test_obj_array_identity_rejects_decimal_rounding_and_changed_indices(tmp_path):
    v=np.array([[.123456789,0,0],[1,0,0],[0,1,0]],np.float32);f=np.array([[0,1,2]],np.int32)
    exact=write_native_fixture(tmp_path/'exact',v,f)
    assert bind_native_ground(exact,tmp_path/'proof')['report']['status']=='pass'
    approximate=write_native_fixture(tmp_path/'rounded',v,f,decimals=6)
    with pytest.raises(ValueError,match='exactly preserve'):bind_native_ground(approximate,tmp_path/'rejected')
    changed=write_native_fixture(tmp_path/'index',v,f)
    changed.write_text(changed.read_text().replace('f 1 2 3','f 1 3 2'))
    with pytest.raises(ValueError,match='exactly preserve'):bind_native_ground(changed,tmp_path/'index_rejected')


def test_production_native_source_graph_and_fidelity_keep_frozen_contract(tmp_path):
    ir,mesh,frame,_=real_reader_fixture(tmp_path)
    mesh.apply_transform(frame.record()['source_to_world']);path=write_native_fixture(tmp_path/'scene',mesh.vertices,mesh.faces)
    output=tmp_path/'evidence'
    samples=validate_source_mesh(ir,path,output/'topology_samples',frame.record()['world_to_source'])
    graph=validate_source_connectivity(ir,path,output/'topology_graph',frame.record()['world_to_source'])
    assert samples['status']==graph['status']=='pass'
    assert samples['classifier_kind']=='bounded_native_segments'
    assert samples['sample_count']>=10000 and samples['mismatch_count']==0
    assert graph['protected_occupancy_mismatches']==graph['vertical_tangent_ray_triangle_pairs']==0
    assert verify_mesh_backend_evidence(samples) and verify_mesh_backend_evidence(graph)
    incomplete=copy.deepcopy(samples)
    incomplete['backend_evidence']['evidence_files']=[entry for entry in incomplete['backend_evidence']['evidence_files'] if entry['role']!='source_point_and_portal_queries']
    with pytest.raises(ValueError,match='measurement closure'):verify_mesh_backend_evidence(incomplete)
    measured=evaluate_source_fidelity(ir,path,output/'source_fidelity',frozen_budget_path=tmp_path/'source_fidelity_profile.json',exterior_delta_path=tmp_path/'exterior_delta.npz',topology_samples_path=output/'topology_samples/source_mesh_validation.json',topology_graph_path=output/'topology_graph/mesh_connectivity_comparison.json')
    assert measured['status']=='measurements_pass'
    # A modified raw native query cannot hide behind unchanged top-level rows.
    query=output/'topology_samples/native_backend/native_queries/unique_hits.npy'
    with query.open('ab') as stream:stream.write(b'changed')
    with pytest.raises(ValueError,match='changed'):verify_mesh_backend_evidence(samples)
