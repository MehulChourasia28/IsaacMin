#!/usr/bin/env python3
"""Measure native cutout/depth and optical response; never issue a forest visual pass."""
from pathlib import Path
import argparse
import numpy as np
from PIL import Image

from isaacmin.io import read_json, atomic_json, sha256_file, utc_now
from isaacmin.assembly.foliage_recipe import PRODUCERS, RECIPE, validate_foliage_recipe
from isaacmin.packaging import ascii_dependency_closure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--select', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.exists():
        raise ValueError('Use a new immutable native optical measurement record')
    files = set()

    def bind(path):
        path = Path(path).resolve()
        path.relative_to(root)
        files.add(path)
        return path

    def load(path):
        return read_json(bind(path))

    captures = {}
    failures = []
    for name, directory in [('preview', args.preview / 'preview'),
                            ('opaque_control', args.candidate / 'opaque_control'),
                            ('thin_candidate', args.candidate / 'thin_candidate')]:
        directory = directory.resolve()
        capture = directory / 'capture'
        report = load(capture / 'capture_result.json')
        adapter = load(capture / 'adapter_result.json')
        reopen = load(capture / 'reopen_result.json')
        load(capture / 'capture_request.json')
        bind(capture / 'kit.log')
        if any(r.get('status') != 'pass' for r in (report, adapter, reopen)):
            failures.append(name + ': native capture/contact/reopen failed')
        if report.get('renderer_recipe') != 'pathtracing_1024' or report.get('path_tracing_sample_budget') != 1024:
            failures.append(name + ': actual native sampling differs')
        if name != 'preview':
            diagnostic = load(capture / 'native_material_diagnostics.json')
            if diagnostic['status'] != 'no_observed_native_material_errors' or diagnostic['errors']:
                failures.append(name + ': native material compilation/import error')
            material = load(directory / 'foliage_material/material_manifest.json')
            if material['producer_sha256'] != sha256_file(root / PRODUCERS[0]) or material['template_sha256'] != sha256_file(root / PRODUCERS[1]):
                failures.append(name + ': material constructor changed')
            for item in material['materials']:
                bind(directory / item['module'])
        closure = ascii_dependency_closure(directory / 'world.usda')
        if closure['status'] != 'pass':
            failures.append(name + ': native scene closure failed')
        for entry in closure['files']:
            path = bind(directory / entry['path'])
            if sha256_file(path) != entry['sha256']:
                failures.append(name + ': changed scene dependency')
        for frame in report['frames']:
            for role in ('rgb', 'depth', 'instance_segmentation'):
                path = bind(capture / frame[role])
                if sha256_file(path) != frame[role + '_sha256']:
                    failures.append(name + ': native frame bytes changed')
            if frame['path_tracing_sample_budget'] != 1024 or frame['path_tracing_subframes_per_capture'] != 16:
                failures.append(name + ': frame sampling differs')
        captures[name] = (capture, report)
    expected_poses = [f['pose'] for f in captures['preview'][1]['frames']]
    if len(expected_poses) != 6 or any([f['pose'] for f in r['frames']] != expected_poses for _, r in captures.values()):
        raise ValueError('Optical comparisons require the same complete six native camera poses')

    measured = []
    for index, pose in enumerate(expected_poses):
        samples = {}
        for name, (directory, capture) in captures.items():
            frame = capture['frames'][index]
            depth = np.load(directory / frame['depth'], allow_pickle=False)
            segmentation = np.load(directory / frame['instance_segmentation'], allow_pickle=False)
            labels = frame['instance_id_to_prim_path']
            ids = [int(key) for key, value in labels.items() if 'MeasurementCard' in str(value)]
            mask = np.isin(segmentation, ids)
            rgb = np.asarray(Image.open(directory / frame['rgb']))[..., :3]
            if mask.shape != depth.shape or rgb.shape[:2] != depth.shape or mask.sum() < 1000:
                raise ValueError('Optical card has insufficient actual segmented coverage')
            samples[name] = {'depth': depth, 'mask': mask, 'rgb': rgb}
        reference = samples['opaque_control']
        thin = samples['thin_candidate']
        union = reference['mask'] | thin['mask']
        common = reference['mask'] & thin['mask']
        silhouette = int(np.count_nonzero(reference['mask'] != thin['mask']))
        preview_silhouette = int(np.count_nonzero(samples['preview']['mask'] != thin['mask']))
        finite = np.isfinite(reference['depth'][union]) & np.isfinite(thin['depth'][union])
        difference = np.abs(reference['depth'][union] - thin['depth'][union])
        maximum_depth = float(difference.max()) if len(difference) and finite.all() else None
        if not common.any():
            raise ValueError('Candidate and control have no common actual leaf pixels')
        rgb_difference = thin['rgb'][common].astype(float) - reference['rgb'][common].astype(float)
        item = {'pose': pose['kind'], 'card_pixels': int(common.sum()),
            'candidate_control_silhouette_mismatch_pixels': silhouette,
            'candidate_preview_silhouette_mismatch_pixels': preview_silhouette,
            'candidate_control_maximum_depth_error_m': maximum_depth,
            'candidate_minus_control_mean_rgb_codes': rgb_difference.mean(axis=0).tolist(),
            'preview_mean_rgb_codes': samples['preview']['rgb'][samples['preview']['mask']].mean(axis=0).tolist(),
            'opaque_mean_rgb_codes': reference['rgb'][common].mean(axis=0).tolist(),
            'thin_mean_rgb_codes': thin['rgb'][common].mean(axis=0).tolist(),
            'candidate_saturated_pixel_fraction': float((thin['rgb'][common] == 255).any(axis=1).mean())}
        if silhouette or preview_silhouette or maximum_depth is None or maximum_depth > 1e-5:
            failures.append(pose['kind'] + ': thin transport changed native cutout/depth')
        if item['candidate_saturated_pixel_fraction'] > .01:
            failures.append(pose['kind'] + ': diagnostic loses optical response to clipping')
        measured.append(item)
    for species in ('oak', 'birch'):
        views = [m for m in measured if m['pose'] in (species + '_front', species + '_back')]
        if max(float(np.mean(m['candidate_minus_control_mean_rgb_codes'])) for m in views) <= 1:
            failures.append(species + ': no measured backlight transmission response')
    result = {'status': 'fail' if failures else 'technical_component_pass', 'recipe': RECIPE,
        'at_utc': utc_now(), 'renderer_recipe': 'pathtracing_1024', 'measurements': measured,
        'failures': failures, 'scope': 'Actual original leaf photograph test cards; no forest or Minecraft world visual qualification',
        'optical_parameters': 'Inferred 65/35 reflect/transmit split and native exported IOR; not measured spectra',
        'temporal_appearance': 'not_run', 'held_out_canopy_appearance': 'not_run', 'world_qualification': 'not_run',
        'files': [{'path': str(p.relative_to(root)), 'sha256': sha256_file(p)} for p in sorted(files)],
        'producer_files': {name: sha256_file(root / name) for name in PRODUCERS},
        'measurement_script_sha256': sha256_file(Path(__file__))}
    atomic_json(output, result)
    if args.select and not failures:
        selection = {'recipe': RECIPE, 'tissue_transmission_fraction': .35,
            'component_evidence': {'path': str(output.relative_to(root)), 'sha256': sha256_file(output)},
            'producer_files': result['producer_files'], 'world_appearance': 'not_run'}
        validate_foliage_recipe(root, selection)
        path = root / 'state/foliage_material_recipe.json'
        if path.exists():
            previous = root / 'state/foliage_recipe_history' / (sha256_file(path) + '.json')
            if not previous.exists():
                atomic_json(previous, read_json(path))
        atomic_json(path, selection)
    print({'status': result['status'], 'failures': failures, 'evidence': str(output),
           'selected': args.select and not failures, 'world_appearance': 'not_run'})
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
