"""Explicit tree-leaf transport selection after native component measurements."""
from pathlib import Path

from isaacmin.io import read_json, sha256_file
from isaacmin.security import safe_path

PRODUCERS = ('src/isaacmin/assembly/foliage_mdl.py', 'recipes/materials/thin_leaf.mdl.in')
RECIPE = 'original_textures_thin_leaf_v1'


def validate_foliage_recipe(workspace, record):
    root = Path(workspace).resolve()
    if record.get('recipe') != RECIPE or record.get('tissue_transmission_fraction') != .35:
        raise ValueError('Unsupported explicit tree-leaf transport recipe')
    if record.get('producer_files') != {name: sha256_file(root / name) for name in PRODUCERS}:
        raise ValueError('Tree-leaf transport producer changed; native measurements required')
    entry = record['component_evidence']
    proof_path = safe_path(root, entry['path'], must_exist=True)
    if sha256_file(proof_path) != entry['sha256']:
        raise ValueError('Tree-leaf optical evidence changed')
    proof = read_json(proof_path)
    if proof.get('status') != 'technical_component_pass' or proof.get('recipe') != RECIPE:
        raise ValueError('Tree-leaf transport lacks actual native component measurements')
    for file in proof['files']:
        path = safe_path(root, file['path'], must_exist=True)
        if sha256_file(path) != file['sha256']:
            raise ValueError('Tree-leaf native component dependency changed')
    return record


def selected_foliage_recipe(workspace):
    path = Path(workspace) / 'state/foliage_material_recipe.json'
    return validate_foliage_recipe(workspace, read_json(path)) if path.is_file() else None
