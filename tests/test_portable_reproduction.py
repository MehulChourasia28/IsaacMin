import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from isaacmin.io import atomic_json, sha256_file
from isaacmin.packaging import verify_package_files

_spec = importlib.util.spec_from_file_location('portable_compare', Path(__file__).parents[1] / 'isaac_scripts/compare_reproduction.py')
compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compare)


def fixture(tmp_path):
    package, output = tmp_path / 'package', tmp_path / 'replayed'
    reference = package / 'reproduction/reference'
    captured = output / 'capture'
    reference.mkdir(parents=True)
    captured.mkdir(parents=True)
    pose = {'position': [0, 0, .6], 'look_at': [0, 1, .6], 'kind': 'static', 'held_out': True}
    frame = {'frame': 0, 'pose': pose, 'rgb': 'rgb.png', 'depth': 'depth.npy', 'instance_segmentation': 'labels.npy',
             'camera_world_matrix_row_vectors': np.eye(4).tolist(), 'fx_pixels': 640, 'fy_pixels': 640,
             'cx_pixels': 640, 'cy_pixels': 360, 'instance_id_to_prim_path': {'1': '/World/Terrain_FinalGround'}}
    for root in (reference, captured):
        Image.fromarray(np.full((16, 16, 3), 120, np.uint8)).save(root / 'rgb.png')
        np.save(root / 'depth.npy', np.full((16, 16), 2, np.float32))
        np.save(root / 'labels.npy', np.ones((16, 16), np.int32))
        np.savez(root / 'physx_contact_rays.npz', hit=np.ones(12000, bool),
                 origins=np.zeros((12000, 3)), directions=np.tile([0, 0, -1], (12000, 1)),
                 source_face_index=np.arange(12000), position=np.zeros((12000, 3)))
    for role in ('rgb', 'depth', 'instance_segmentation'):
        frame[role+'_sha256'] = sha256_file(reference / frame[role])
    contact = {'id': 0, 'settled_position': [0, 0, .1], 'settled_orientation_wxyz': [1, 0, 0, 0]}
    atomic_json(package / 'package.json', {'test_fixture_only': True})
    atomic_json(package / 'reproduction/reference.json', {
        'files': [{'path': p.name, 'sha256': sha256_file(p)} for p in reference.iterdir()],
        'scene_sha256': 'fixture-scene', 'runtime_version': 'fixture-runtime',
        'frames': [frame]*3, 'replay_frame_indices': [0, 1, 2], 'expected_frame_count': 3,
        'contacts': [contact], 'ray_samples': 'physx_contact_rays.npz',
        'thresholds': compare.THRESHOLDS, 'required_contact_ray_count': 10000})
    atomic_json(package / 'reproduction/request.json', {'poses': [pose]*3})
    actual = {'frames': [frame]*3, 'contacts': [contact], 'scene_sha256': 'fixture-scene', 'authored_scene_preserved': True}
    atomic_json(captured / 'capture_result.json', actual)
    atomic_json(output / 'capture_request.json', {})
    atomic_json(output / 'reopen/reopen_result.json', {'status': 'pass', 'scene_sha256': 'fixture-scene',
                                                    'native_isaac_version': 'fixture-runtime'})
    atomic_json(captured / 'physx_contact_rays.json', {'sha256': sha256_file(captured / 'physx_contact_rays.npz')})
    return package, output, actual


def test_portable_comparison_rejects_changed_contact_without_hiding_equal_images(tmp_path):
    package, output, actual = fixture(tmp_path)
    assert compare.compare(package, output)['status'] == 'pass'
    actual = copy.deepcopy(actual)
    actual['contacts'][0]['settled_position'][2] += .002
    atomic_json(output / 'capture/capture_result.json', actual)
    result = compare.compare(package, output)
    assert result['status'] == 'fail'
    assert all(frame['status'] == 'pass' for frame in result['frame_results'])
    assert result['contact_results'][0]['status'] == 'fail'


def test_portable_comparison_rejects_changed_render_and_weakened_threshold(tmp_path):
    package, output, actual = fixture(tmp_path)
    rgb = output / 'capture/rgb.png'
    Image.fromarray(np.full((16, 16, 3), 123, np.uint8)).save(rgb)
    for frame in actual['frames']:
        frame['rgb_sha256'] = sha256_file(rgb)
    atomic_json(output / 'capture/capture_result.json', actual)
    assert compare.compare(package, output)['status'] == 'fail'
    reference_path = package / 'reproduction/reference.json'
    reference = json.loads(reference_path.read_text())
    reference['thresholds']['rgb_mean_codes_max'] = 10
    atomic_json(reference_path, reference)
    with pytest.raises(ValueError, match='thresholds'):
        compare.compare(package, output)


def test_portable_comparison_rejects_stale_raw_bytes(tmp_path):
    package, output, _ = fixture(tmp_path)
    np.save(output / 'capture/depth.npy', np.full((16, 16), 3, np.float32))
    with pytest.raises(ValueError, match='payload'):
        compare.compare(package, output)


def test_package_inventory_rejects_unrecorded_secret_file(tmp_path):
    scene = tmp_path / 'world.usda'
    scene.write_text('#usda 1.0\n')
    files = [{'path': scene.name, 'sha256': sha256_file(scene), 'bytes': scene.stat().st_size}]
    atomic_json(tmp_path / 'package.json', {'files': files})
    assert verify_package_files(tmp_path, files)
    (tmp_path / '.env').write_bytes(b'')
    assert not verify_package_files(tmp_path, files)
