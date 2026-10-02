"""Construction edge cases; these fixtures cannot establish Isaac realism."""
from pathlib import Path

import numpy as np
import pytest
import trimesh

from isaacmin.contracts.coordinates import CoordinateFrame
from isaacmin.io import atomic_json, hash_object, read_json

from isaacmin.terrain import naturalize_exterior as candidate

@pytest.fixture
def source_case(tmp_path):
    n = 20
    x, z = np.meshgrid(np.arange(n+1), np.arange(n+1))
    vertices = np.column_stack((x.ravel(), -z.ravel(), np.zeros(x.size)))
    ids = np.arange(x.size).reshape(x.shape)
    a, b, c, d = (q.ravel() for q in (ids[:-1, :-1], ids[1:, :-1], ids[1:, 1:], ids[:-1, 1:]))
    faces = np.concatenate((np.column_stack((a, b, c)), np.column_stack((a, c, d))))
    mesh = tmp_path / 'input.obj'
    with mesh.open('w') as stream:
        np.savetxt(stream, vertices, fmt='v %.17g %.17g %.17g')
        np.savetxt(stream, faces+1, fmt='f %d %d %d')
    height = np.zeros((n, n), np.float32)
    source = tmp_path / 'surface.npz'
    np.savez(source, height=height, validity=np.ones_like(height, bool),
        block_names=np.array(['minecraft:grass_block']), substrate_id=np.zeros_like(height, np.int32),
        water_validity=np.zeros_like(height, bool), water_height=height, min_xz=np.array([0, 0]))
    protection = np.zeros_like(height, bool)
    protection[9:11, 9:11] = True
    delta = np.full_like(height, -.1)
    delta[protection] = 0
    refinement = tmp_path / 'refinement.npz'
    np.savez(refinement, source_height=height, delta=delta, protection=protection)
    frame = CoordinateFrame((0., 0., 0.)).record()
    budget = {'coordinates': frame, 'protected_interface_numerical_m': .002,
              'recipe': {'soil_and_sediment_baseline_m': .75, 'rock_baseline_m': .25,
                         'highmap_max_lowering_m': .2}}
    frozen = tmp_path / 'frozen.json'
    atomic_json(frozen, {'budget': budget, 'sha256': hash_object(budget)})
    return mesh, source, refinement, frozen


def test_real_refinement_and_exact_protected_boundary_coordinates(source_case, tmp_path):
    result = candidate.naturalize(*source_case, (0., 0., 0.), tmp_path / 'candidate')
    before = trimesh.load(source_case[0], process=False)
    after = np.load(tmp_path / 'candidate/vertices.npy')
    movement = np.load(tmp_path / 'candidate/movement.npz')
    fixed = movement['fixed_vertex_mask']
    assert result['status'] == 'candidate_requires_independent_validation'
    assert result['eligible_vertices'] > 0
    assert np.array_equal(after[fixed], np.asarray(before.vertices, np.float32)[fixed])
    assert np.min(after[:, 2]) < -.09
    assert np.all(after[:, 2] >= -.10001)
    assert np.array_equal(np.load(tmp_path / 'candidate/triangles.npy'), before.faces)
    reloaded = trimesh.load(tmp_path / 'candidate/terrain.obj', process=False)
    assert np.array_equal(np.asarray(reloaded.vertices, np.float32), after)
    assert result['native_export_and_renderer'] == 'not_run'


def test_frozen_budget_and_coordinate_frame_cannot_be_overridden(source_case, tmp_path):
    with pytest.raises(ValueError, match='origin differs'):
        candidate.naturalize(*source_case, (100., 0., 0.), tmp_path / 'wrong_frame')
    record = read_json(source_case[3])
    record['budget']['recipe']['soil_and_sediment_baseline_m'] = 100.
    atomic_json(source_case[3], record)
    with pytest.raises(ValueError, match='budget changed'):
        candidate.naturalize(*source_case, (0., 0., 0.), tmp_path / 'altered_budget')


def test_unknown_source_cannot_be_filled(source_case, tmp_path):
    with np.load(source_case[1]) as data:
        fields = {k: data[k] for k in data.files}
    fields['validity'][10, 10] = False
    np.savez(source_case[1], **fields)
    with pytest.raises(ValueError, match='complete finite source'):
        candidate.naturalize(*source_case, (0., 0., 0.), tmp_path / 'unknown_source')


def test_wrong_refinement_height_field_is_rejected(source_case, tmp_path):
    with np.load(source_case[2]) as data:
        fields = {k: data[k] for k in data.files}
    fields['source_height'][8, 8] = 1.
    np.savez(source_case[2], **fields)
    with pytest.raises(ValueError, match='same complete finite source'):
        candidate.naturalize(*source_case, (0., 0., 0.), tmp_path / 'wrong_height')


def test_cache_reuse_checks_real_output_bytes(source_case, tmp_path):
    path, receipt = candidate.construct_cached(*source_case, (0., 0., 0.), tmp_path / 'cache')
    modified = path.stat().st_mtime_ns
    assert candidate.construct_cached(*source_case, (0., 0., 0.), tmp_path / 'cache') == (path, receipt)
    assert path.stat().st_mtime_ns == modified
    with path.open('a') as stream:
        stream.write('# modified bytes\n')
    with pytest.raises(ValueError, match='cache bytes changed'):
        candidate.construct_cached(*source_case, (0., 0., 0.), tmp_path / 'cache')


def test_interrupted_construction_is_preserved(source_case, tmp_path):
    output = tmp_path / 'interrupted'
    output.mkdir()
    witness = output / 'partial.json'
    witness.write_text('{"status":"interrupted"}\n')
    with pytest.raises(FileExistsError):
        candidate.naturalize(*source_case, (0., 0., 0.), output)
    assert witness.read_text() == '{"status":"interrupted"}\n'
