"""Reuse completed planning, with original execution receipts left unchanged.

Only orchestration changes may differ. Every planning/geometry/validation
producer, dependency, output and request parameter remains bound to bytes.
This is a camera plan cache, never a renderer or qualification cache.
"""
from copy import deepcopy
from pathlib import Path
import shutil

from isaacmin.io import atomic_json, hash_object, read_json, sha256_file, utc_now

CODE_ROOTS = ('src', 'scripts', 'blender_scripts', 'isaac_scripts', 'recipes')
ORCHESTRATION = frozenset((
    'src/isaacmin/pipeline.py', 'src/isaacmin/assembly/reuse.py',
    'src/isaacmin/assembly/capture_reuse.py',
))
OWNED = ['evidence/planning', 'evidence/capture_plan.json']


def source_inventory(workspace):
    root = Path(workspace).resolve()
    return {str(p.relative_to(root)): sha256_file(p)
            for folder in CODE_ROOTS for p in sorted((root / folder).rglob('*'))
            if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc'}


def _identity_without_code(identity):
    result = deepcopy(identity)
    result['dependencies'].pop('code')
    return result


def _owned(build, relative):
    relative = Path(relative)
    path = build / relative
    if (relative.is_absolute() or '..' in relative.parts or path.is_symlink()
            or not path.resolve().is_relative_to(build) or not path.is_file()):
        raise ValueError('Planning cache entry is not an owned file')
    return path


def _verify_candidate(previous, checkpoint, identity, inventory, parameters):
    record = read_json(checkpoint)
    if (record.get('status') != 'completed' or record.get('name') != 'capture_plan'
            or record.get('owned_outputs') != OWNED
            or record.get('identity_sha256') != hash_object(record['identity'])
            or _identity_without_code(record['identity']) != _identity_without_code(identity)):
        raise ValueError('Planning dependency identity differs or stage is incomplete')
    old_inventory = read_json(_owned(previous, 'execution_source_files.json'))
    if hash_object(old_inventory) != record['identity']['dependencies']['code']:
        raise ValueError('Original execution source inventory is not bound to this stage')
    actual_changes = sorted(k for k in old_inventory.keys() | inventory.keys()
                            if old_inventory.get(k) != inventory.get(k))
    if set(actual_changes) - ORCHESTRATION:
        raise ValueError('A planning or non-orchestration producer changed')
    files = record.get('files', [])
    if not files:
        raise ValueError('Planning output inventory is empty')
    expected = set()
    for item in files:
        name = item['path']
        if name != OWNED[1] and not name.startswith(OWNED[0] + '/'):
            raise ValueError('Planning output exceeds its owned paths')
        if name in expected:
            raise ValueError('Duplicate planning output')
        expected.add(name)
        path = _owned(previous, name)
        if path.stat().st_size != item['bytes'] or sha256_file(path) != item['sha256']:
            raise ValueError('Planning output bytes changed: ' + name)
    actual = {str(p.relative_to(previous)) for p in (previous / OWNED[0]).rglob('*') if p.is_file()}
    actual.add(OWNED[1])
    reuse_record = previous / OWNED[0] / 'reuse.json'
    if actual != expected or (reuse_record.exists()
            and read_json(reuse_record).get('status') == 'reused_completed_plan'):
        raise ValueError('Planning inventory differs or is a chained cache')
    request = read_json(previous / OWNED[0] / 'capture_plan_request.json')
    old_parameters = dict(request['parameters'])
    old_mesh = Path(old_parameters.pop('mesh_path'))
    current_parameters = {k: str(v) if isinstance(v, Path) else v for k, v in parameters.items()}
    current_mesh = Path(current_parameters.pop('mesh_path'))
    if (request.get('kind') != 'capture_plan'
            or request.get('output') != str(previous / OWNED[0])
            or old_parameters != current_parameters
            or not old_mesh.resolve().is_relative_to(previous)
            or any(sha256_file(p) != identity['final_ground_sha256'] for p in (old_mesh, current_mesh))):
        raise ValueError('Actual planning request or supporting mesh changed')
    process = read_json(previous / OWNED[0] / 'capture_plan_worker.process.json')
    if (process.get('exit_code') != 0 or process.get('status') != 'process_exited'
            or any(process.get(k) is not False for k in ('interrupted', 'timed_out', 'resource_limited'))
            or process.get('child_reaped') is not True
            or process.get('argv', [])[-2:] != ['--request', str(previous / OWNED[0] / 'capture_plan_request.json')]):
        raise ValueError('Original native planning worker did not complete normally')
    result = record['result']
    if (result.get('status') != 'planned'
            or result.get('final_ground_sha256') != identity['final_ground_sha256']
            or result.get('source_surface_sha256') != identity['dependencies']['source_surface']
            or any(read_json(previous / p) != result for p in (
                OWNED[1], OWNED[0] + '/capture_plan.json', OWNED[0] + '/capture_plan_result.json'))):
        raise ValueError('Saved plan differs from the completed worker result')
    # Query receipts can refer to immutable native proofs outside the stage's
    # owned output directory. Keep those original paths and verify the complete
    # recorded closure instead of trusting just the copied plan's bytes.
    from isaacmin.validation.evidence_closure import verify_json_evidence
    verify_json_evidence(previous / OWNED[1])
    cave_plan = previous / OWNED[0] / 'cave_view_plan.json'
    if cave_plan.is_file():
        verify_json_evidence(cave_plan)
    return record, actual_changes


def reuse_capture_plan(workspace, directory, identity, parameters):
    root, directory = Path(workspace).resolve(), Path(directory).resolve()
    inventory = source_inventory(root)
    if hash_object(inventory) != identity['dependencies']['code']:
        raise ValueError('Current code changed after build dependencies were frozen')
    report = {'schema_version': 1, 'created_at_utc': utc_now(), 'qualification': 'not_inferred',
              'status': 'no_matching_completed_plan', 'candidates': []}
    evidence = directory / 'evidence/planning/reuse.json'
    candidates = sorted((root / 'worlds').glob('*/stage_checkpoints/capture_plan.json'),
                        key=lambda p: p.stat().st_mtime_ns, reverse=True)
    for checkpoint in candidates:
        previous = checkpoint.parent.parent.resolve()
        if previous == directory:
            continue
        entry = {'build': str(previous.relative_to(root))}
        report['candidates'].append(entry)
        try:
            _owned(root, str(checkpoint.relative_to(root)))
            record, changes = _verify_candidate(previous, checkpoint, identity, inventory, parameters)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            entry.update(status='rejected', reason=str(exc))
            continue
        for item in record['files']:
            target = directory / item['path']
            if target.exists():
                raise ValueError('Planning reuse must not overwrite an existing output')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(previous / item['path'], target)
            if sha256_file(target) != item['sha256']:
                raise ValueError('Planning bytes changed during reuse')
        entry.update(status='verified_original_execution', checkpoint_sha256=sha256_file(checkpoint),
                     execution_inventory_sha256=sha256_file(previous / 'execution_source_files.json'),
                     changed_orchestration_files=changes)
        report.update(status='reused_completed_plan', selected=entry,
                      original_receipts='Copied unchanged; their paths and timestamps refer to the original execution')
        atomic_json(evidence, report)
        return record['result']
    atomic_json(evidence, report)
    return None
