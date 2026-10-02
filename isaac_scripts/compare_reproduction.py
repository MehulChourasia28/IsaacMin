"""Independent raw-array comparison, runnable in pinned Isaac Python without Kit.

Thresholds are frozen in the package before the first replay. A failure is
retained; this module has no authority to alter thresholds or scene bytes.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

THRESHOLDS = {'rgb_mean_codes_max': .5, 'rgb_p999_codes_max': 2,
              'depth_p999_m_max': .001, 'depth_max_m': .01,
              'contact_position_m_max': .001, 'contact_orientation_degrees_max': .1,
              'contact_ray_m_max': .001, 'camera_matrix_abs_max': 1e-6}


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def payload(root, name, expected):
    path = (root / name).resolve()
    if (Path(name).is_absolute() or not path.is_relative_to(root.resolve())
            or not path.is_file() or digest(path) != expected):
        raise ValueError('Reproduction raw payload missing, escaped or changed')
    return path


def label_codes(array, frame, mapping):
    result = np.zeros(array.shape, dtype=np.int32)
    labels = frame['instance_id_to_prim_path']
    for label in np.unique(array):
        description = json.dumps(labels.get(str(int(label)), {'unmapped_id': int(label)}), sort_keys=True)
        code = mapping.setdefault(description, len(mapping)+1)
        result[array == label] = code
    return result


def compare(package, output):
    package, output = Path(package).resolve(), Path(output).resolve()
    reference_path = package / 'reproduction/reference.json'
    reference = json.loads(reference_path.read_text())
    root = package / 'reproduction/reference'
    for entry in reference['files']:
        payload(root, entry['path'], entry['sha256'])
    capture_dir = output / 'capture'
    actual_path = capture_dir / 'capture_result.json'
    actual = json.loads(actual_path.read_text())
    reopened_path = output / 'reopen/reopen_result.json'
    reopened = json.loads(reopened_path.read_text())
    request = json.loads((package / 'reproduction/request.json').read_text())
    thresholds = reference['thresholds']
    if thresholds != THRESHOLDS or reference['required_contact_ray_count'] != 10000:
        raise ValueError('Reproduction thresholds differ from the fixed supported protocol')
    errors, gaps, frame_results, measured_files = [], [], [], []

    def bind(path):
        measured_files.append({'path': str(path.relative_to(output)), 'sha256': digest(path),
                               'bytes': path.stat().st_size})

    for path in (actual_path, reopened_path, output / 'capture_request.json'):
        bind(path)
    if (actual.get('scene_sha256') != reference['scene_sha256']
            or reopened.get('scene_sha256') != reference['scene_sha256']
            or reopened.get('native_isaac_version') != reference['runtime_version']
            or reopened.get('status') != 'pass' or not actual.get('authored_scene_preserved')):
        errors.append('Scene, collision, runtime or clean-reopen identity differs')
    if (actual.get('renderer_recipe', 'legacy_rtx_8') != reference.get('renderer_recipe', 'legacy_rtx_8')
            or actual.get('native_renderer_settings', {}) != reference.get('native_renderer_settings', {})):
        errors.append('Native renderer settings differ from the frozen reference')
    if len(actual['frames']) != reference['expected_frame_count']:
        errors.append('Representative frame count differs')
    selected = [actual['frames'][i] for i in reference['replay_frame_indices'] if i < len(actual['frames'])]
    for before, after in zip(reference['frames'], selected):
        paths = {}
        for side, directory, frame in (('before', root, before), ('after', capture_dir, after)):
            for role in ('rgb', 'depth', 'instance_segmentation'):
                path = payload(directory, frame[role], frame[role+'_sha256'])
                paths[(side, role)] = path
                if side == 'after':
                    bind(path)
        rgb_a, rgb_b = [np.asarray(Image.open(paths[(side, 'rgb')]).convert('RGB'), dtype=np.int16)
                        for side in ('before', 'after')]
        depth_a, depth_b = [np.load(paths[(side, 'depth')], allow_pickle=False) for side in ('before', 'after')]
        if rgb_a.shape != rgb_b.shape or depth_a.shape != depth_b.shape:
            raise ValueError('Native capture shape changed')
        rgb_delta = np.abs(rgb_a-rgb_b)
        valid_a, valid_b = [np.isfinite(d) & (d > 0) for d in (depth_a, depth_b)]
        valid = valid_a & valid_b
        depth_delta = np.abs(depth_a[valid].astype(float)-depth_b[valid].astype(float))
        if not len(depth_delta):
            raise ValueError('No valid depth to reproduce')
        mapping = {}
        labels = [label_codes(np.load(paths[(side, 'instance_segmentation')], allow_pickle=False).squeeze(), frame, mapping)
                  for side, frame in (('before', before), ('after', after))]
        pose_error = float(np.max(np.abs(np.asarray(before['camera_world_matrix_row_vectors'])
                                        - np.asarray(after['camera_world_matrix_row_vectors']))))
        intrinsics_equal = all(abs(before[key]-after[key]) < 1e-7 for key in
                               ('fx_pixels', 'fy_pixels', 'cx_pixels', 'cy_pixels'))
        metrics = {'reference_frame': before['frame'], 'replayed_frame': after['frame'],
                   'rgb_mean_codes': float(rgb_delta.mean()), 'rgb_p999_codes': float(np.quantile(rgb_delta, .999)),
                   'rgb_max_codes': int(rgb_delta.max()), 'depth_p999_m': float(np.quantile(depth_delta, .999)),
                   'depth_max_m': float(depth_delta.max()), 'valid_depth_pixels': int(valid.sum()),
                   'depth_valid_masks_equal': bool(np.array_equal(valid_a, valid_b)),
                   'semantic_labels_equal': bool(np.array_equal(*labels)),
                   'camera_matrix_max_difference': pose_error, 'intrinsics_equal': intrinsics_equal,
                   'pose_request_equal': before['pose'] == after['pose']}
        okay = (metrics['rgb_mean_codes'] <= thresholds['rgb_mean_codes_max']
                and metrics['rgb_p999_codes'] <= thresholds['rgb_p999_codes_max']
                and metrics['depth_p999_m'] <= thresholds['depth_p999_m_max']
                and metrics['depth_max_m'] <= thresholds['depth_max_m']
                and metrics['depth_valid_masks_equal'] and metrics['semantic_labels_equal']
                and pose_error <= thresholds['camera_matrix_abs_max']
                and intrinsics_equal and metrics['pose_request_equal'])
        metrics['status'] = 'pass' if okay else 'fail'
        frame_results.append(metrics)
        if not okay:
            errors.append('Frame reproduction differs: '+str(before['frame']))
    ray_path = capture_dir / 'physx_contact_rays.npz'
    ray_manifest_path = capture_dir / 'physx_contact_rays.json'
    rays = json.loads(ray_manifest_path.read_text())
    payload(capture_dir, ray_path.name, rays['sha256'])
    bind(ray_path)
    bind(ray_manifest_path)
    with np.load(root / reference['ray_samples'], allow_pickle=False) as a, np.load(ray_path, allow_pickle=False) as b:
        matched = (np.array_equal(a['hit'], b['hit']) and np.array_equal(a['origins'], b['origins'])
                   and np.array_equal(a['directions'], b['directions'])
                   and np.array_equal(a['source_face_index'], b['source_face_index']))
        hit = a['hit'] & b['hit'] if a['hit'].shape == b['hit'].shape else np.zeros(0, bool)
        ray_error = float(np.linalg.norm(a['position'][hit]-b['position'][hit], axis=1).max()) if hit.any() else None
        ray_count = len(a['hit'])
        if (not matched or ray_error is None or ray_error > thresholds['contact_ray_m_max']
                or ray_count < reference['required_contact_ray_count']):
            errors.append('Native contact rays differ or coverage is insufficient')
    contact_results = []
    if len(actual['contacts']) != len(reference['contacts']):
        errors.append('Dropped probe count differs')
    if not reference['contacts']:
        gaps.append('Reference contains no dropped physical probes')
    for a, b in zip(reference['contacts'], actual['contacts']):
        error = float(np.linalg.norm(np.asarray(a['settled_position'])-b['settled_position']))
        qa, qb = (np.asarray(c['settled_orientation_wxyz'], float) for c in (a, b))
        angle = float(np.degrees(2*np.arccos(np.clip(abs(np.dot(qa, qb)/(np.linalg.norm(qa)*np.linalg.norm(qb))), 0, 1))))
        okay = np.isfinite(error+angle) and error <= thresholds['contact_position_m_max'] and angle <= thresholds['contact_orientation_degrees_max']
        contact_results.append({'id': a['id'], 'position_difference_m': error,
                                'orientation_difference_degrees': angle, 'status': 'pass' if okay else 'fail'})
        if not okay:
            errors.append('Dropped physical probe changed: '+str(a['id']))
    if not any(p.get('held_out') for p in request['poses']):
        gaps.append('No independently selected held-out/free viewpoint in this reference')
    if len(frame_results) < 3:
        gaps.append('Fewer than three representative native viewpoints')
    result = {'status': 'fail' if errors else 'incomplete' if gaps else 'pass',
              'method': 'fresh_pinned_Isaac_saved_scene_RGB_depth_contact_reproduction',
              'package_manifest_sha256': digest(package / 'package.json'),
              'package': str(package),
              'reference_sha256': digest(reference_path), 'scene_sha256': reference['scene_sha256'],
              'runtime_version': reference['runtime_version'], 'thresholds': thresholds,
              'frame_results': frame_results, 'contact_results': contact_results,
              'ray_count': ray_count, 'ray_samples_equal': matched, 'maximum_contact_ray_difference_m': ray_error,
              'errors': errors, 'gaps': gaps, 'files': measured_files,
              'validator_sha256': digest(Path(__file__)),
              'package_unchanged': True, 'source_save_referenced': False, 'generation_modules_imported': False,
              'appearance_qualification': 'not_run', 'navigation_stack': 'not_run_not_supplied'}
    (output / 'reproduction_comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.package, args.output)
    raise SystemExit(0 if result['status'] == 'pass' else 2)
