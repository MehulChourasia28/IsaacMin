import json
from pathlib import Path

import numpy as np
import pytest

from isaacmin.validation.patch_integrity import validate_planar_patch


def check(tmp_path, *, move=False, outside_change=False, folded=False):
    # A planar square changed to its other diagonal, plus a separate triangle.
    vertices=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0],[3,0,0],[4,0,0],[3,1,0]],np.float32)
    before=np.array([[0,1,2],[0,2,3],[4,5,6]],np.int32)
    after=np.array([[0,1,3],[1,2,3],[4,5,6]],np.int32)
    if folded:after[:2]=[[0,1,2],[1,3,2]]
    changed=vertices.copy()
    if move:changed[6,2]=np.float32(.0001)
    if outside_change:after[2]=[4,6,5]
    for name,array in [('old_vertices',vertices),('old_triangles',before),('vertices',changed),('triangles',after)]:np.save(tmp_path/(name+'.npy'),array)
    np.savez(tmp_path/'patch.npz',face_ids=np.array([0,1]),old_triangles=before[:2],new_triangles=after[:2])
    (tmp_path/'context.json').write_text(json.dumps({'kind':'fixture_only'}))
    return validate_planar_patch(tmp_path/'old_vertices.npy',tmp_path/'old_triangles.npy',tmp_path/'vertices.npy',tmp_path/'triangles.npy',tmp_path/'patch.npz',tmp_path/'evidence',context_path=tmp_path/'context.json',cumulative_attempt=1)


def test_exact_planar_diagonal_change_preserves_owned_disk(tmp_path):
    report=check(tmp_path)
    assert report['status']=='pass'
    assert report['whole_array_faces_compared']==3 and report['actually_changed_faces']==2
    assert report['preserved_oriented_boundary_edges']==4


@pytest.mark.parametrize('argument,failure',[('move','changed_vertex_positions'),('outside_change','changed_faces_outside_declared_patch'),('folded','positive_area_overlap_pairs')])
def test_ownership_changes_and_fold_cannot_pass(tmp_path,argument,failure):
    report=check(tmp_path,**{argument:True})
    assert report['status']=='fail' and report['failures'][failure]>0
