"""An explicit, content-bound capture choice; appearance remains separately tested."""
from pathlib import Path
from isaacmin.io import read_json, sha256_file
from isaacmin.security import safe_path

PRODUCERS = ('isaac_scripts/capture_scene.py', 'isaac_scripts/material_log.py',
             'isaac_scripts/ground_collision.py', 'isaac_scripts/contact_rays.py',
             'src/isaacmin/adapters/workers.py')


def selected_capture_recipe(workspace):
    root = Path(workspace).resolve()
    path = root / 'state/render_recipe.json'
    if not path.is_file():
        return {'renderer_recipe': 'legacy_rtx_8'}
    record = read_json(path)
    if record.get('renderer_recipe') != 'pathtracing_1024':
        raise ValueError('Unsupported explicit native capture recipe')
    if record.get('producer_files') != {name: sha256_file(root / name) for name in PRODUCERS}:
        raise ValueError('Selected capture producer changed; retain and requalify its evidence')
    evidence = record['component_evidence']
    file = safe_path(root, evidence['path'], must_exist=True)
    if sha256_file(file) != evidence['sha256']:
        raise ValueError('Selected renderer component evidence changed')
    proof = read_json(file)
    if proof.get('status') != 'component_pass' or proof.get('renderer_recipe') != record['renderer_recipe']:
        raise ValueError('Selected native renderer lacks actual component evidence')
    for entry in proof.get('files', []):
        path = safe_path(root, entry['path'], must_exist=True)
        if sha256_file(path) != entry['sha256']:
            raise ValueError('Selected renderer component input changed')
    return record
