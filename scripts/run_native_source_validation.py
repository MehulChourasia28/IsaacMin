#!/usr/bin/env python3
"""Run the normal independent source gates on a retained actual native region."""
from pathlib import Path
import argparse
from isaacmin.io import atomic_json, read_json, sha256_file, utc_now
from isaacmin.validation.profile import freeze_profile
from isaacmin.validation.source_fidelity import freeze_source_fidelity_profile, evaluate_source_fidelity
from isaacmin.validation.continuous_surface import validate_continuous_surface
from isaacmin.volumes.mesh_validation import validate_source_mesh, validate_source_connectivity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-directory', type=Path, required=True)
    parser.add_argument('--source-ir', type=Path, required=True)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.exists():
        raise ValueError('Use a fresh immutable validation directory')
    output.mkdir(parents=True)
    mesh = args.scene_directory.resolve() / 'final_ground.obj'
    ir = args.source_ir.resolve()
    request = read_json(args.request.resolve())
    project = read_json(root / 'state/resolved_project.json')
    frame = project['coordinate_frame']
    support = Path(request['support_manifest'])
    exterior = Path(request['exterior_delta']['path'])
    profile = freeze_profile(output / 'quality_profile.json', {
        'coordinates': frame, 'scope': 'actual retained native region; source and geometry gates only',
        'mesh_sha256': sha256_file(mesh), 'source_ir_sha256': sha256_file(ir / 'world_ir.json')})
    freeze_source_fidelity_profile(profile, output / 'source_fidelity_profile.json')
    evidence = output / 'evidence'
    results = {}
    def retain(stage, report):
        results[stage] = report['status']
        atomic_json(output / 'checkpoint.json', {'at_utc': utc_now(), 'status': 'running',
            'completed_stages': results, 'appearance_qualification': 'not_run'})
        print({'stage': stage, 'status': report['status']}, flush=True)
    try:
        retain('continuous_surface', validate_continuous_surface(mesh, evidence / 'continuous_surface/surface',
            source_ir=ir, mesh_to_source=frame['world_to_source']))
        retain('source_samples', validate_source_mesh(ir, mesh, evidence / 'topology_samples',
            frame['world_to_source'], sample_count=10000, seed=1729, support_manifest_path=support))
        retain('connectivity', validate_source_connectivity(ir, mesh, evidence / 'topology_graph',
            frame['world_to_source'], support_manifest_path=support))
        retain('source_fidelity', evaluate_source_fidelity(ir, mesh, evidence / 'source_fidelity',
            frozen_budget_path=output / 'source_fidelity_profile.json', exterior_delta_path=exterior,
            topology_samples_path=evidence / 'topology_samples/source_mesh_validation.json',
            topology_graph_path=evidence / 'topology_graph/mesh_connectivity_comparison.json',
            support_manifest_path=support))
    except BaseException as exc:
        atomic_json(output / 'checkpoint.json', {'at_utc': utc_now(), 'status': 'incomplete',
            'completed_stages': results, 'exception_type': type(exc).__name__, 'reason': str(exc),
            'appearance_qualification': 'not_run'})
        raise
    atomic_json(output / 'result.json', {'at_utc': utc_now(),
        'status': 'component_measurements_pass' if all(
            status == ('measurements_pass' if stage == 'source_fidelity' else 'pass')
            for stage, status in results.items()) else 'fail',
        'stages': results, 'scope': 'actual native geometry and independent source preservation only',
        'script_sha256': sha256_file(Path(__file__)), 'world_appearance': 'not_run'})


if __name__ == '__main__':
    main()
