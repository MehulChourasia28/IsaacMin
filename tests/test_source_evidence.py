import json

import numpy as np
import pytest

from isaacmin.volumes.source_evidence import verify_source_topology,recheck_source_topology
from isaacmin.volumes.mesh_validation import validate_source_mesh,validate_source_connectivity
from isaacmin.validation.local_features import diagnose_local_features
from test_source_mesh_validation import fixture_source


@pytest.mark.parametrize("validator",[validate_source_mesh,validate_source_connectivity,diagnose_local_features])
def test_public_geometry_checks_reject_valid_but_altered_topology_before_mesh_loading(tmp_path,validator):
    fixture_source(tmp_path)
    labels=tmp_path/"topology/source_topology_labels.npz"
    with np.load(labels,allow_pickle=False) as data:arrays={key:data[key] for key in data.files}
    # A valid readable archive can still lie about protected cave air. The
    # manifest must reject it before opening even the missing target mesh.
    arrays["covered_air"][0,0,0]=~arrays["covered_air"][0,0,0]
    np.savez_compressed(labels,**arrays)
    with pytest.raises(ValueError,match="topology payload changed"):
        validator(tmp_path,tmp_path/"missing.obj",tmp_path/"evidence",np.eye(4))


@pytest.mark.parametrize("mutation",["missing_labels","duplicate_labels","different_ir"])
def test_source_graph_requires_unambiguous_labels_bound_to_current_ir(tmp_path,mutation):
    fixture_source(tmp_path)
    path=tmp_path/"topology/source_topology_graph.json"
    graph=json.loads(path.read_text())
    if mutation=="missing_labels":graph["files"]=[]
    elif mutation=="duplicate_labels":graph["files"].append(dict(graph["files"][0]))
    else:graph["source_ir_sha256"]="0"*64
    path.write_text(json.dumps(graph))
    with pytest.raises(ValueError):verify_source_topology(tmp_path)


def test_source_payload_rechecked_before_evidence_publication(tmp_path):
    fixture_source(tmp_path)
    _,context=verify_source_topology(tmp_path)
    path=tmp_path/"topology/source_topology_graph.json"
    graph=json.loads(path.read_text());graph["features"]=[{"id":"changed-during-validation"}]
    path.write_text(json.dumps(graph))
    with pytest.raises(ValueError,match="changed during geometry validation"):
        recheck_source_topology(tmp_path,context)
