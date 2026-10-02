import json
import numpy as np
import pytest
from isaacmin.validation.disk_integrity import validate_disk_retriangulation


def run(tmp_path,mutation=None):
    vertices=np.array([[0,0,0],[1,0,.1],[1,1,0],[0,1,.1],[3,0,0],[4,0,0],[3,1,0]],np.float32)
    before=np.array([[0,1,2],[0,2,3],[4,5,6]],np.int32);after=np.array([[0,1,3],[1,2,3],[4,5,6]],np.int32)
    moved=vertices.copy()
    if mutation=='move':moved[5,0]+=.1
    if mutation=='outside':after[2]=[4,6,5]
    if mutation=='boundary':after[1]=after[1,::-1]
    for name,value in [('ov',vertices),('of',before),('nv',moved),('nf',after)]:np.save(tmp_path/(name+'.npy'),value)
    np.savez(tmp_path/'patch.npz',face_ids=np.array([0,1]));(tmp_path/'context.json').write_text(json.dumps({'fixture_only':True}))
    return validate_disk_retriangulation(tmp_path/'ov.npy',tmp_path/'of.npy',tmp_path/'nv.npy',tmp_path/'nf.npy',tmp_path/'patch.npz',tmp_path/'evidence',context_path=tmp_path/'context.json',cumulative_attempt=1)


def test_nonplanar_disk_ownership_does_not_claim_surface_equality(tmp_path):
    report=run(tmp_path)
    assert report['status']=='pass' and not report['surface_equality_claim']
    assert report['original_disk']['euler_characteristic']==report['replacement_disk']['euler_characteristic']==1


@pytest.mark.parametrize('mutation,reason',[('move','changed_vertex_coordinates'),('outside','unannounced_changed_faces'),('boundary','oriented_boundary_changed')])
def test_unowned_change_or_boundary_reversal_cannot_pass(tmp_path,mutation,reason):
    report=run(tmp_path,mutation)
    assert report['status']=='fail' and report['failures'][reason]
