"""Unit cache contracts only: fixture receipts do not establish native success."""
from copy import deepcopy
from pathlib import Path
import pytest

from isaacmin.assembly.capture_reuse import reuse_capture_plan, source_inventory, OWNED
from isaacmin.io import atomic_json, hash_object, read_json, sha256_file


def fixture(tmp_path):
    root = tmp_path
    for name in ('src/isaacmin/pipeline.py', 'src/isaacmin/validation/capture_plan.py'):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('# unit producer fixture')
    previous, current = (root / 'worlds' / name for name in ('previous', 'current'))
    for build in (previous, current):
        build.mkdir(parents=True)
        (build / 'ground.obj').write_text('unit mesh identity only')
    inventory = source_inventory(root)
    atomic_json(previous / 'execution_source_files.json', inventory)
    identity = {'dependencies': {'code': hash_object(inventory), 'source_surface': 'fixture-source'},
                'final_ground_sha256': sha256_file(previous / 'ground.obj'), 'detail': {'levels': 3}}
    plan = {'status': 'planned', 'final_ground_sha256': identity['final_ground_sha256'],
            'source_surface_sha256': 'fixture-source', 'limitations': ['unit fixture; no native execution'],
            'missing_cave_contact_probes': ['must remain missing']}
    output = previous / OWNED[0]
    parameters = {'mesh_path': previous / 'ground.obj', 'ir_dir': str(root / 'ir'), 'origin': [1, 0, 2]}
    atomic_json(output / 'capture_plan_request.json', {'kind': 'capture_plan', 'output': str(output),
        'parameters': {k: str(v) if isinstance(v, Path) else v for k, v in parameters.items()}})
    atomic_json(output / 'capture_plan_worker.process.json', {'exit_code': 0, 'status': 'process_exited',
        'child_reaped': True, 'interrupted': False, 'timed_out': False, 'resource_limited': False,
        'scope': 'unit fixture only', 'argv': ['worker', '--request', str(output / 'capture_plan_request.json')]})
    for name in (OWNED[1], OWNED[0] + '/capture_plan.json', OWNED[0] + '/capture_plan_result.json'):
        atomic_json(previous / name, plan)
    files = [previous / OWNED[1], *output.iterdir()]
    checkpoint = {'name': 'capture_plan', 'status': 'completed', 'identity': identity,
        'identity_sha256': hash_object(identity), 'owned_outputs': OWNED, 'result': plan,
        'files': [{'path': str(p.relative_to(previous)), 'bytes': p.stat().st_size,
                   'sha256': sha256_file(p)} for p in files]}
    atomic_json(previous / 'stage_checkpoints/capture_plan.json', checkpoint)
    parameters['mesh_path'] = current / 'ground.obj'
    return root, previous, current, deepcopy(identity), parameters, plan


def test_orchestration_change_reuses_exact_plan_and_original_receipts(tmp_path):
    root, previous, current, identity, parameters, plan = fixture(tmp_path)
    (root / 'src/isaacmin/pipeline.py').write_text('# changed orchestrator fixture')
    identity['dependencies']['code'] = hash_object(source_inventory(root))
    assert reuse_capture_plan(root, current, identity, parameters) == plan
    receipt = OWNED[0] + '/capture_plan_worker.process.json'
    assert (current / receipt).read_bytes() == (previous / receipt).read_bytes()
    proof = read_json(current / OWNED[0] / 'reuse.json')
    assert proof['status'] == 'reused_completed_plan'
    assert proof['qualification'] == 'not_inferred'
    assert read_json(current / OWNED[1])['missing_cave_contact_probes'] == ['must remain missing']


@pytest.mark.parametrize('change', ['producer', 'mesh', 'dependency', 'output', 'untracked',
                                   'inventory', 'partial', 'parameters', 'receipt'])
def test_changes_or_partial_execution_prevent_plan_reuse(tmp_path, change):
    root, previous, current, identity, parameters, plan = fixture(tmp_path)
    if change == 'producer':
        (root / 'src/isaacmin/validation/capture_plan.py').write_text('# modified planner')
    elif change == 'mesh':
        (current / 'ground.obj').write_text('changed geometry')
    elif change == 'dependency':
        identity['detail']['levels'] = 2
    elif change == 'output':
        (previous / OWNED[1]).write_text('{}')
    elif change == 'untracked':
        (previous / OWNED[0] / 'unexpected.json').write_text('{}')
    elif change == 'inventory':
        atomic_json(previous / 'execution_source_files.json', {})
    elif change == 'parameters':
        parameters['origin'] = [0, 0, 0]
    else:
        path = previous / 'stage_checkpoints/capture_plan.json'
        record = read_json(path)
        if change == 'partial':
            record['status'] = 'interrupted_or_failed'
        else:
            receipt = previous / OWNED[0] / 'capture_plan_worker.process.json'
            value = read_json(receipt); value['timed_out'] = True; atomic_json(receipt, value)
            for item in record['files']:
                if item['path'] == str(receipt.relative_to(previous)):
                    item.update(bytes=receipt.stat().st_size, sha256=sha256_file(receipt))
        atomic_json(path, record)
    identity['dependencies']['code'] = hash_object(source_inventory(root))
    assert reuse_capture_plan(root, current, identity, parameters) is None
    assert not (current / OWNED[1]).exists()
    assert read_json(current / OWNED[0] / 'reuse.json')['candidates'][0]['status'] == 'rejected'


def test_external_query_proof_must_still_match_its_original_bytes(tmp_path):
    root, previous, current, identity, parameters, plan = fixture(tmp_path)
    proof = root / 'external_query_proof.json'
    atomic_json(proof, {'scope': 'unit fixture; not a native query'})
    plan['native_query_evidence'] = {'path': str(proof), 'sha256': sha256_file(proof),
                                     'bytes': proof.stat().st_size}
    for name in (OWNED[1], OWNED[0] + '/capture_plan.json', OWNED[0] + '/capture_plan_result.json'):
        atomic_json(previous / name, plan)
    checkpoint = previous / 'stage_checkpoints/capture_plan.json'
    record = read_json(checkpoint)
    record['result'] = plan
    for item in record['files']:
        path = previous / item['path']
        item.update(bytes=path.stat().st_size, sha256=sha256_file(path))
    atomic_json(checkpoint, record)
    atomic_json(proof, {'scope': 'changed external measurement'})
    assert reuse_capture_plan(root, current, identity, parameters) is None
    assert not (current / OWNED[1]).exists()
