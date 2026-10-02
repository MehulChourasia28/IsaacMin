#!/usr/bin/env python3
"""Bind measured actual-source precision correction to its retained failed history.

This only allows a fresh production construction. It neither modifies a repair
ledger nor promotes a world, waives source checks, or qualifies appearance.
"""
from pathlib import Path
import argparse

from isaacmin.io import atomic_json, read_json, sha256_file, utc_now
from isaacmin.jobs.geometry_resolution import geometry_producers, verify_geometry_resolution
from isaacmin.jobs.repair_budget import exhausted_repairs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--attempt-directory', type=Path)
    source.add_argument('--evidence-manifest', type=Path,
                        help='Explicit actual evidence roles and native execution provenance for a new layout')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--activate', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    attempt = args.attempt_directory.resolve() if args.attempt_directory else None
    output = args.output.resolve()
    if output.exists():
        raise ValueError('Use a new immutable precision-resolution record')
    pointer_path = root / 'state/known_failures.json'
    state = read_json(pointer_path)
    matching = [entry for entry in state['failures']
                if entry['defect_class'] == 'native_float32_precision_tessellation']
    if len(matching) != 1:
        raise ValueError('Expected one retained source-specific precision defect')
    entry = matching[0]
    scene = attempt / 'production_replay' if attempt else None
    evidence = attempt / 'independent_final' if attempt else None
    explicit = read_json(args.evidence_manifest) if args.evidence_manifest else None
    roles = {role: (root / path).resolve() for role, path in explicit['files'].items()} if explicit else {
        'finalization': scene / 'native_precision/finalization.json',
        'global_intersections': scene / 'native_precision/global_intersections/global_intersections.json',
        'continuous_surface': evidence / 'evidence/continuous_surface/surface/continuous_surface.json',
        'topology_samples': evidence / 'evidence/topology_samples/source_mesh_validation.json',
        'topology_graph': evidence / 'evidence/topology_graph/mesh_connectivity_comparison.json',
        'source_fidelity': evidence / 'evidence/source_fidelity/source_fidelity.json',
        'edge_witnesses': evidence / 'edge_witness_regression/result.json',
        'edge_queries': evidence / 'edge_witness_regression/native_queries/segment_queries.json',
        'quality_profile': evidence / 'quality_profile.json',
        'source_fidelity_profile': evidence / 'source_fidelity_profile.json',
        'authoritative_obj': scene / 'final_ground.obj',
        'native_vertices': scene / 'native_precision/final_vertices.npy',
        'native_triangles': scene / 'native_precision/final_triangles.npy',
        'construction_request': scene / 'request.json',
    }
    if any(not path.is_relative_to(root) or path.is_symlink() for path in roles.values()):
        raise ValueError('Actual precision evidence must remain within the project')
    result = {key: entry[key] for key in ('defect_class', 'source_snapshot_sha256',
        'source_ir_sha256', 'ledger_sha256')}
    result.update(kind='MeasuredNativePrecisionResolution', status='geometry_component_resolved',
        created_at_utc=utc_now(), scope='actual retained native region geometry and protected source only',
        original_ledger=entry['ledger'], original_history_preserved=True,
        files={role: {'path': str(path.relative_to(root)), 'sha256': sha256_file(path)}
               for role, path in roles.items()},
        producer_files={name: sha256_file(root / name) for name in geometry_producers()},
        recorder_sha256=sha256_file(Path(__file__)),
        production_policy='All independent native, source, material, ecology, render and world gates still run',
        appearance_qualification='not_run', world_qualification='not_run',
        original_native_process=(explicit['native_execution'] if explicit else
                                 'interrupted by host reboot; no exit status invented'),
        recovered_geometry=(explicit['geometry_provenance'] if explicit else
                            'Original final native arrays retained; exact OBJ regenerated and independently compared'))
    if explicit:
        result['evidence_manifest'] = {'path': str(args.evidence_manifest.resolve().relative_to(root)),
                                       'sha256': sha256_file(args.evidence_manifest)}
    atomic_json(output, result)
    candidate = dict(entry, resolution={'path': str(output.relative_to(root)), 'sha256': sha256_file(output)})
    # A failed verification retains the candidate as failed evidence but never
    # publishes it to the state consumed by production.
    try:
        if verify_geometry_resolution(root, candidate) is not True:
            raise ValueError('Actual-source precision evidence did not verify')
    except Exception as exc:
        atomic_json(output.with_suffix('.verification.json'), {'status': 'fail',
            'reason': str(exc), 'record_sha256': sha256_file(output), 'activated': False})
        raise
    atomic_json(output.with_suffix('.verification.json'), {'status': 'component_pass',
        'record_sha256': sha256_file(output), 'verified_at_utc': utc_now(),
        'world_qualification': 'not_run'})
    if args.activate:
        previous = root / 'state/known_failure_history' / (sha256_file(pointer_path) + '.json')
        if not previous.exists():
            atomic_json(previous, state)
        entry['resolution'] = candidate['resolution']
        state['updated_at_utc'] = utc_now()
        atomic_json(pointer_path, state)
        remaining = exhausted_repairs(root, entry['source_snapshot_sha256'], entry['source_ir_sha256'])
        if any(e['defect_class'] == entry['defect_class'] for e in remaining):
            raise RuntimeError('Verified correction was not consumed by normal production admission')
    print({'status': 'geometry_component_resolved', 'record': str(output), 'activated': args.activate,
           'world_appearance': 'not_run'})


if __name__ == '__main__':
    main()
