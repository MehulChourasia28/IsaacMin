"""Resolve one measured precision defect using independent actual-map evidence."""
from pathlib import Path
from isaacmin.io import read_json, sha256_file
from isaacmin.security import safe_path
from isaacmin.validation.evidence_closure import verify_json_evidence
from isaacmin.volumes.mesh_validation import verify_mesh_backend_evidence


def geometry_producers():
    from isaacmin.volumes.native_precision import PRODUCERS
    return tuple(sorted(set(PRODUCERS) | {
        'blender_scripts/export_scene.py', 'blender_scripts/precision_mesh.py',
        'blender_scripts/precise_subdivision.py', 'blender_scripts/mesh_arrays.py',
        'src/isaacmin/terrain/naturalize_exterior.py',
        'src/isaacmin/volumes/construction.py', 'src/isaacmin/assembly/material_assignment.py',
        '.tools/native_precision/build_manifest.json'}))


def verify_geometry_resolution(workspace, failure):
    """A repaired component permits construction; all world gates still run."""
    root = Path(workspace).resolve()
    pointer = failure.get('resolution')
    if pointer is None:
        return False
    path = safe_path(root, pointer['path'], must_exist=True)
    if sha256_file(path) != pointer['sha256']:
        raise ValueError('Native precision resolution record changed')
    record = read_json(path)
    if (failure['defect_class'] != 'native_float32_precision_tessellation'
            or record.get('kind') != 'MeasuredNativePrecisionResolution'
            or record.get('status') != 'geometry_component_resolved'):
        raise ValueError('Unsupported source-specific repair resolution')
    for key in ('defect_class', 'source_snapshot_sha256', 'source_ir_sha256', 'ledger_sha256'):
        if record.get(key) != failure[key]:
            raise ValueError('Precision resolution belongs to different source or failed history')
    expected = {name: sha256_file(root / name) for name in geometry_producers()}
    if record.get('producer_files') != expected:
        raise ValueError('Measured precision constructor changed; new actual-source evidence is required')
    required = {'finalization', 'global_intersections', 'continuous_surface', 'topology_samples',
                'topology_graph', 'source_fidelity', 'edge_witnesses', 'quality_profile',
                'source_fidelity_profile', 'authoritative_obj', 'native_vertices', 'native_triangles',
                'construction_request', 'edge_queries'}
    if set(record['files']) != required:
        raise ValueError('Incomplete actual-source precision resolution proof')
    documents = {}
    files = {}
    for role, entry in record['files'].items():
        actual = safe_path(root, entry['path'], must_exist=True)
        if sha256_file(actual) != entry['sha256']:
            raise ValueError('Precision correction evidence changed: ' + role)
        files[role] = actual
        if actual.suffix == '.json':
            documents[role] = read_json(actual)
    final = documents['finalization']
    if (final['final_vertices_sha256'] != record['files']['native_vertices']['sha256']
            or final['final_triangles_sha256'] != record['files']['native_triangles']['sha256']
            or final['zero_native_normals'] or final['nonfinite_native_normals']
            or final['faces_below_existing_area_threshold'] or final['bad_normal_vertex_ids']
            or final['actual_normal_count'] != final['vertices']
            or final['producer_sha256'] != expected['blender_scripts/precision_mesh.py']):
        raise ValueError('Actual native geometry measurements did not resolve the precision defect')
    glob = documents['global_intersections']
    if (glob['status'] != 'pass' or not glob['complete'] or glob['intersection_pair_count']
            or glob['candidate_triangles'] != final['triangles'] or glob['candidate_vertices'] != final['vertices']):
        raise ValueError('Complete native triangle intersection freedom is required')
    inputs = {e['role']: e['sha256'] for e in glob['input_files']}
    if (inputs['candidate_vertices'] != final['final_vertices_sha256']
            or inputs['candidate_triangles'] != final['final_triangles_sha256']):
        raise ValueError('Intersection proof belongs to another actual mesh')
    mesh_hash = record['files']['authoritative_obj']['sha256']
    for role in ('topology_samples', 'topology_graph', 'source_fidelity'):
        report = documents[role]
        status = 'measurements_pass' if role == 'source_fidelity' else 'pass'
        if (report['status'] != status or report['mesh_sha256'] != mesh_hash
                or report['source_ir_sha256'] != failure['source_ir_sha256']):
            raise ValueError('Independent protected-source geometry has not passed: ' + role)
        if role != 'source_fidelity' and verify_mesh_backend_evidence(report) is not True:
            raise ValueError('Original native-array source proof is required')
    from isaacmin.volumes import mesh_validation
    from isaacmin.validation import source_fidelity, continuous_surface
    for role, module in [('topology_samples', mesh_validation), ('topology_graph', mesh_validation),
                         ('source_fidelity', source_fidelity), ('continuous_surface', continuous_surface)]:
        if documents[role]['validator_sha256'] != sha256_file(Path(module.__file__)):
            raise ValueError('Resolution evidence predates the current validator: ' + role)
    surface = documents['continuous_surface']
    witness = documents['edge_witnesses']
    if (surface['status'] != 'pass' or any(surface['failures'].values())
            or not any(e['sha256'] == mesh_hash for e in surface['input_files'])
            or witness['status'] != 'pass' or any(witness['inside']) or any(witness['unresolved'])
            or witness['query_sha256'] != record['files']['edge_queries']['sha256']
            or witness['mesh_vertices_sha256'] != final['final_vertices_sha256']
            or witness['mesh_triangles_sha256'] != final['final_triangles_sha256']):
        raise ValueError('Continuous surface or held-out intrusion regression failed')
    from isaacmin.validation.profile import load_profile
    profile = load_profile(files['quality_profile'])
    budget = documents['source_fidelity_profile']['budget']
    if (budget['quality_profile_sha256'] != profile['sha256']
            or budget['protected_interface_numerical_m'] != .002
            or budget['protected_roof_max_thickness_loss_m'] != .004):
        raise ValueError('Resolution must retain the original strict protected-source limits')
    detail = documents['construction_request']['geometry_detail']
    if detail != {'subdivision_levels': 3, 'soil_displacement_peak_to_peak_m': .02,
                  'subdivision_method': 'bilinear_double_no_limit_v1'}:
        raise ValueError('Precision resolution changed the required geometry detail')
    for role in ('global_intersections', 'continuous_surface', 'edge_queries'):
        verify_json_evidence(files[role])
    return True
