"""Cache identity and byte-storage guards; no external tool result is simulated."""
import json
from pathlib import Path

import pytest

from isaacmin.assembly.native_cache import (
    PRODUCERS, PRECISION_FILES, copy_native_precision_evidence,
    deduplicate_generated_textures, input_identity, record_bare, sha, verify_bare,
)


def input_files(tmp_path):
    root = tmp_path / "workspace"
    for name in (*PRODUCERS, ".tools/blender/blender"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(("identity fixture: " + name).encode())
    terrain = root / "terrain.obj"
    terrain.write_bytes(b"identity-only mesh bytes")
    texture = root / "scan.png"
    texture.write_bytes(b"identity-only image bytes")
    request = {"terrain_mesh": str(terrain), "terrain_translation": [0, 0, 0],
               "minecraft_origin": [1, 2, 3], "geometry_detail": {"subdivision_levels": 3},
               "materials": [{"name": "soil", "repeat_m": 1.3, "base_color": str(texture)}]}
    return root, request


def test_cache_identity_excludes_population_but_binds_source_scale_and_producer(tmp_path):
    root, request = input_files(tmp_path)
    baseline = input_identity(request, root)
    populated = dict(request, output="another/output", assets=[{"id": "fern"}],
                     bare_scene_directory="bare")
    assert input_identity(populated, root) == baseline
    for key, replacement in (("minecraft_origin", [1, 2, 3.001]),
                             ("geometry_detail", {"subdivision_levels": 2}),
                             ("terrain_material_recipe", {"mode": "source_shared_original_pbr_v1"})):
        assert input_identity(dict(request, **{key: replacement}), root) != baseline
    texture = Path(request["materials"][0]["base_color"])
    original = texture.read_bytes()
    texture.write_bytes(original + b" changed")
    assert input_identity(request, root) != baseline
    texture.write_bytes(original)
    producer = root / PRODUCERS[0]
    producer.write_bytes(producer.read_bytes() + b" changed")
    assert input_identity(request, root) != baseline


def test_cache_rejects_changed_bytes_and_escaping_paths_before_open(tmp_path):
    root, request = input_files(tmp_path)
    cache = root / "bare"
    cache.mkdir()
    payload = cache / "scene.blend"
    payload.write_bytes(b"guard-test payload; never opened as Blender")
    record = {"input_identity": input_identity(request, root), "files": [
        {"path": payload.name, "sha256": sha(payload), "bytes": payload.stat().st_size}]}
    manifest = cache / "bare_scene_cache.json"
    manifest.write_text(json.dumps(record))
    payload.write_bytes(b"altered")
    with pytest.raises(ValueError, match="bytes changed"):
        verify_bare(request, root, cache)
    record["files"][0]["path"] = "../terrain.obj"
    manifest.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="escapes"):
        verify_bare(request, root, cache)


def test_texture_dedup_retains_portable_bytes_and_excludes_provider_masters(tmp_path):
    output = tmp_path / "scene"
    (output / "textures").mkdir(parents=True)
    (output / "transition_bakes").mkdir()
    content = b"byte-storage fixture; no decoder claim"
    a = output / "textures/a.png"
    b = output / "transition_bakes/b.png"
    provider = tmp_path / "provider.png"
    for path in (a, b, provider):
        path.write_bytes(content)
    provider_inode = provider.stat().st_ino
    store = tmp_path / "store"
    result = deduplicate_generated_textures(output, store)
    assert len(result["files"]) == 2
    assert result["unique_encoded_bytes"] == len(content)
    assert a.stat().st_ino == b.stat().st_ino
    assert a.stat().st_mode & 0o222 == 0
    assert provider.stat().st_ino == provider_inode
    assert provider.stat().st_mode & 0o200
    next(store.rglob("*.png")).unlink()
    assert a.read_bytes() == b.read_bytes() == content
    with pytest.raises(ValueError, match="outside portable"):
        deduplicate_generated_textures(output, output / "store")


def test_texture_dedup_rejects_corrupt_content_address(tmp_path):
    output = tmp_path / "scene"
    (output / "textures").mkdir(parents=True)
    texture = output / "textures/a.png"
    texture.write_bytes(b"original encoded bytes")
    digest = sha(texture)
    store = tmp_path / "store"
    collision = store / digest[:2] / (digest + ".png")
    collision.parent.mkdir(parents=True)
    collision.write_bytes(b"corrupted store bytes")
    with pytest.raises(ValueError, match="collision"):
        deduplicate_generated_textures(output, store)
    assert texture.read_bytes() == b"original encoded bytes"


def bare_contract_fixture(tmp_path, *, precision=True):
    """Protocol-only inputs; no Blender, renderer, or geometry qualification."""
    import numpy as np
    root, request = input_files(tmp_path)
    bare = root / 'bare'
    bare.mkdir()
    for name in ('scene.blend', 'final_ground.obj', 'world.usda'):
        (bare / name).write_bytes(b'unit contract fixture; not a native tool result')
    finalization = None
    if precision:
        folder = bare / 'native_precision'
        folder.mkdir()
        np.save(folder / 'final_vertices.npy', np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], np.float32))
        np.save(folder / 'final_triangles.npy', np.array([[0, 1, 2]], np.int32))
        finalization = {'status': 'unit_fixture_not_geometry_evidence',
                        'final_vertices_sha256': sha(folder / 'final_vertices.npy'),
                        'final_triangles_sha256': sha(folder / 'final_triangles.npy')}
        (folder / 'finalization.json').write_text(json.dumps(finalization))
        (folder / 'input_vertices.npy').write_bytes(b'not part of minimal precision evidence')
    export = {'status': 'success', 'native_usd_geometry_verified': True,
              'texture_files': [], 'native_precision_finalization': finalization,
              'final_ground_sha256': sha(bare / 'final_ground.obj'),
              'native_usd_terrain_vertices': 3, 'native_usd_terrain_triangles': 1,
              'native_terrain_payload': {'scope': 'unit contract only'}}
    (bare / 'export_result.json').write_text(json.dumps(export))
    (bare / 'native_dependency_closure.json').write_text(json.dumps({'status': 'pass', 'files': [{'path': 'world.usda'}]}))
    return root, request, bare, export


def test_bare_precision_arrays_are_bound_and_copied_exactly_once(tmp_path):
    root, request, bare, _ = bare_contract_fixture(tmp_path)
    record = record_bare(request, root, bare)
    assert record['schema_version'] == 2
    assert set(PRECISION_FILES) <= {item['path'] for item in record['files']}
    verified = verify_bare(request, root, bare)
    output = root / 'populated'
    receipt = copy_native_precision_evidence(bare, output, verified)
    assert receipt['status'] == 'copied_verified_bytes' and receipt['qualification'] == 'not_inferred'
    assert {p.name for p in (output / 'native_precision').iterdir()} == {Path(p).name for p in PRECISION_FILES}
    for name in PRECISION_FILES:
        assert (bare / name).read_bytes() == (output / name).read_bytes()
    with pytest.raises(ValueError, match='immutable'):
        copy_native_precision_evidence(bare, output, verified)


@pytest.mark.parametrize('name', PRECISION_FILES)
def test_precision_change_is_detected_before_cache_open_and_after_verification(tmp_path, name):
    root, request, bare, _ = bare_contract_fixture(tmp_path)
    record_bare(request, root, bare)
    verified = verify_bare(request, root, bare)
    path = bare / name
    path.write_bytes(path.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='bytes changed'):
        verify_bare(request, root, bare)
    with pytest.raises(ValueError, match='bytes changed'):
        copy_native_precision_evidence(bare, root / 'populated', verified)
    assert not (root / 'populated/native_precision').exists()


def test_record_rejects_declared_precision_array_hash_mismatch(tmp_path):
    root, request, bare, export = bare_contract_fixture(tmp_path)
    export['native_precision_finalization']['final_vertices_sha256'] = '0' * 64
    (bare / 'native_precision/finalization.json').write_text(json.dumps(export['native_precision_finalization']))
    (bare / 'export_result.json').write_text(json.dumps(export))
    with pytest.raises(ValueError, match='declared array hash mismatch'):
        record_bare(request, root, bare)
    assert not (bare / 'bare_scene_cache.json').exists()


def test_record_rejects_finalization_different_from_export(tmp_path):
    root, request, bare, export = bare_contract_fixture(tmp_path)
    export['native_precision_finalization']['status'] = 'different declaration'
    (bare / 'export_result.json').write_text(json.dumps(export))
    with pytest.raises(ValueError, match='differs from export declaration'):
        record_bare(request, root, bare)


def test_older_precision_cache_is_not_retroactively_completed(tmp_path):
    root, request, bare, _ = bare_contract_fixture(tmp_path)
    record = record_bare(request, root, bare)
    record['schema_version'] = 1
    record.pop('native_precision_evidence')
    record['files'] = [item for item in record['files'] if item['path'] not in PRECISION_FILES]
    manifest = bare / 'bare_scene_cache.json'
    manifest.write_text(json.dumps(record))
    original = manifest.read_bytes()
    with pytest.raises(ValueError, match='predates recorded native precision evidence'):
        verify_bare(request, root, bare)
    assert manifest.read_bytes() == original


def test_older_cache_without_precision_declaration_remains_explicitly_without_it(tmp_path):
    root, request, bare, _ = bare_contract_fixture(tmp_path, precision=False)
    record = record_bare(request, root, bare)
    record['schema_version'] = 1
    record.pop('native_precision_evidence')
    (bare / 'bare_scene_cache.json').write_text(json.dumps(record))
    verified = verify_bare(request, root, bare)
    copied = copy_native_precision_evidence(bare, root / 'populated', verified)
    assert copied == {'status': 'not_declared', 'files': [], 'qualification': 'not_inferred'}
    assert not (root / 'populated').exists()
