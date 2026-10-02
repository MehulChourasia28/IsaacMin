"""Selection/provenance fixtures; these do not establish native asset success."""
from pathlib import Path
import pytest
from isaacmin.io import atomic_json, sha256_file, read_json
from isaacmin.assets.canopy import select_canopy_candidates, load_canopy_candidates


def generation_fixture(root, seed):
    folder = root / 'fixtures' / str(seed)
    folder.mkdir(parents=True)
    script = folder / 'frozen.py'
    script.write_text('# selection test fixture, never executed by Blender\n')
    blend = folder / 'fixture.blend'
    blend.write_bytes(b'Not a native Blender artifact; selection fixture only')
    request = folder / 'request.json'
    atomic_json(request, {'seed': seed})
    maps = folder / 'maps'
    maps.mkdir()
    atomic_json(maps / 'maps_manifest.json', {'scope': 'selection_fixture'})
    manifest = folder / 'inputs.json'
    atomic_json(manifest, {'executed_script': str(script), 'execution_script_sha256': sha256_file(script),
                          'files': [{'path': str(maps / 'maps_manifest.json'), 'sha256': sha256_file(maps / 'maps_manifest.json')}]})
    source = folder / 'generation.json'
    atomic_json(source, {'status': 'external_tool_verified', 'recipe_species': 'oak', 'asset_id': 'procedural_oak',
        'generator_script_sha256': sha256_file(script), 'input_manifest': str(manifest),
        'input_manifest_sha256': sha256_file(manifest), 'request_sha256': sha256_file(request),
        'output_blend': str(blend), 'output_sha256': sha256_file(blend), 'maps_manifest': str(maps / 'maps_manifest.json'),
        'variants': [{'seed': seed, 'objects': ['fixture_trunk', 'fixture_leaves']}]})
    return source, script


def test_separate_variant_blends_retain_both_seeds_and_old_state(tmp_path):
    first, _ = generation_fixture(tmp_path, 1)
    second, _ = generation_fixture(tmp_path, 2)
    select_canopy_candidates(tmp_path, [first])
    previous = read_json(tmp_path / 'state/procedural_assets.json')
    selected = select_canopy_candidates(tmp_path, [first, second])
    assert selected['qualification'] == 'not_run'
    assert [v['seed'] for _, v in load_canopy_candidates(tmp_path)['oak']] == [1, 2]
    history = list((tmp_path / 'state/procedural_asset_history').glob('*.json'))
    assert len(history) == 1 and read_json(history[0]) == previous


def test_changed_original_execution_cannot_be_adopted_or_used(tmp_path):
    source, script = generation_fixture(tmp_path, 1)
    select_canopy_candidates(tmp_path, [source])
    script.write_text('# changed original execution fixture\n')
    with pytest.raises(ValueError, match='dependency changed'):
        load_canopy_candidates(tmp_path)
    with pytest.raises(ValueError, match='input changed'):
        select_canopy_candidates(tmp_path, [source])
