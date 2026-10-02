#!/usr/bin/env python3
"""Select the actual measured Spark capture recipe, without qualifying a world."""
from pathlib import Path
import argparse

from isaacmin.io import atomic_json, read_json, sha256_file, utc_now
from isaacmin.adapters.workers import probe_workers
from isaacmin.adapters.render_recipe import PRODUCERS, selected_capture_recipe
from isaacmin.assets.usd_provenance import validated_native_renderer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--independent-exchange', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    chain_path = root / 'artifacts/bootstrap/cross_runtime.json'
    independent_path = args.independent_exchange.resolve()
    output = args.output.resolve()
    if output.exists():
        raise ValueError('Use a new immutable renderer component record')
    chain = read_json(chain_path)
    independent = read_json(independent_path)
    if not probe_workers(root)['cross_runtime_qualified']:
        raise ValueError('Current native compatibility chain is missing, changed or failed')
    if (chain.get('status') != 'pass' or validated_native_renderer(chain) != 'PathTracing'
            or independent.get('status') != 'pass'
            or independent['native_chain_sha256'] != sha256_file(chain_path)):
        raise ValueError('Independent current native capture, depth and contact measurements are required')
    if chain.get('native_material_import', {}).get('status') != 'no_observed_native_material_errors':
        raise ValueError('Current native material import diagnostics are required')
    producers = {name: sha256_file(root / name) for name in PRODUCERS}
    result = {'status': 'component_pass', 'created_at_utc': utc_now(),
        'renderer_recipe': 'pathtracing_1024', 'producer_files': producers,
        'scope': 'Actual complete ARM64 worker, native USD, original materials, render, depth and ground-contact compatibility fixture',
        'files': [{'path': str(p.relative_to(root)), 'sha256': sha256_file(p)}
                  for p in (chain_path, independent_path)],
        'actual_frames': len(chain['frames']), 'generated_display_frames_used': False,
        'world_appearance': 'not_run', 'renderer_rng_reproducibility': 'not_qualified'}
    atomic_json(output, result)
    selected = {'renderer_recipe': 'pathtracing_1024', 'producer_files': producers,
        'component_evidence': {'path': str(output.relative_to(root)), 'sha256': sha256_file(output)},
        'world_appearance': 'not_run'}
    path = root / 'state/render_recipe.json'
    if path.exists():
        previous = root / 'state/render_recipe_history' / (sha256_file(path) + '.json')
        if not previous.exists():
            atomic_json(previous, read_json(path))
    atomic_json(path, selected)
    selected_capture_recipe(root)
    print({'selected': 'pathtracing_1024', 'actual_frames': len(chain['frames']),
           'component_evidence': str(output), 'world_appearance': 'not_run'})


if __name__ == '__main__':
    main()
