"""Explicit material candidates; selecting a recipe never qualifies appearance."""
from pathlib import Path

from isaacmin.io import read_json, sha256_file
from isaacmin.security import safe_path
from .material_assignment import REQUIRED

SHARED_ORIGINAL = 'source_shared_original_pbr_v1'
PRODUCERS = ('src/isaacmin/assembly/material_mdl.py',
             'src/isaacmin/assembly/material_assignment.py',
             'recipes/materials/shared_original.mdl.in')


def validate_material_recipe(workspace, recipe, materials=None):
    """Check original channel bytes and producer identity before native execution."""
    root = Path(workspace).resolve()
    if recipe.get('mode') != SHARED_ORIGINAL:
        raise ValueError('Unsupported explicit terrain material recipe')
    candidate = safe_path(root, recipe['candidate_manifest'], must_exist=True)
    if sha256_file(candidate) != recipe['candidate_manifest_sha256']:
        raise ValueError('Selected original material manifest changed')
    expected = {name: sha256_file(root / name) for name in PRODUCERS}
    if recipe.get('producer_files') != expected:
        raise ValueError('Selected shared material producer changed; qualify the new candidate explicitly')
    data = read_json(candidate)
    if data.get('status') != 'original_channels_acquired_and_decoded':
        raise ValueError('Shared materials need acquired and decoded original scans')
    specs = data['materials']
    if len(specs) != len(REQUIRED) or {s['asset_id'] for s in specs} != set(REQUIRED):
        raise ValueError('Shared materials must retain all source material families')
    for spec in specs:
        repeat = float(spec['repeat_m'])
        if not .1 <= repeat <= 20 or spec.get('licence') != 'CC0-1.0':
            raise ValueError('Original scans need verified licence and physical repeat')
        for role in ('base_color', 'roughness', 'normal', 'height'):
            supplied = Path(spec[role])
            relative = supplied.resolve().relative_to(root).as_posix() if supplied.is_absolute() else spec[role]
            file = safe_path(root, relative, must_exist=True)
            proof = spec['original_channel_proof'][role]
            if (sha256_file(file) != proof['sha256']
                    or min(proof['width'], proof['height']) < 4096
                    or min(proof['width'], proof['height']) / repeat < 1024):
                raise ValueError('Original material bytes or physical sampling differ from the selected candidate')
    if materials is not None and materials != specs:
        raise ValueError('Assembly materials differ from the selected original scan manifest')
    return specs


def selected_material_recipe(workspace):
    root = Path(workspace).resolve()
    path = root / 'state/terrain_material_recipe.json'
    if not path.is_file():
        return None
    recipe = read_json(path)
    validate_material_recipe(root, recipe)
    return recipe
