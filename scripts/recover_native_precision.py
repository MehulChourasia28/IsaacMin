#!/usr/bin/env python3
"""Recover an exact OBJ from completed, verified native geometry after a reboot."""
import argparse
from pathlib import Path
import numpy as np

from isaacmin.io import atomic_json, read_json, sha256_file, utc_now
from isaacmin.process import run_worker
from isaacmin.validation.global_intersections import _verify_record, file_record
from isaacmin.validation.native_ground import bind_native_ground


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-directory', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    scene = args.scene_directory.resolve()
    evidence = args.evidence.resolve()
    if evidence.exists():
        raise ValueError('Use a new immutable recovery evidence directory')
    evidence.mkdir(parents=True)
    native = scene / 'native_precision'
    finalization = read_json(native / 'finalization.json')
    if (finalization['zero_native_normals'] or finalization['nonfinite_native_normals']
            or finalization['faces_below_existing_area_threshold']
            or finalization['actual_normal_count'] != finalization['vertices']
            or finalization['producer_sha256'] != sha256_file(root / 'blender_scripts/precision_mesh.py')):
        raise ValueError('Completed native normal/nondegeneracy evidence is missing or changed')
    global_path = native / 'global_intersections/global_intersections.json'
    global_report = read_json(global_path)
    if (sha256_file(global_path) != finalization['global_intersections_sha256']
            or global_report['status'] != 'pass' or not global_report['complete']
            or global_report['intersection_pair_count'] != 0):
        raise ValueError('Completed entire-mesh intersection proof is required')
    for entry in [*global_report['input_files'], *global_report['producer_files'], *global_report['files']]:
        _verify_record(entry)
    vp, fp = native / 'final_vertices.npy', native / 'final_triangles.npy'
    for path, key in [(vp, 'final_vertices_sha256'), (fp, 'final_triangles_sha256')]:
        if sha256_file(path) != finalization[key]:
            raise ValueError('Retained native geometry bytes changed')
    v, f = np.load(vp, mmap_mode='r'), np.load(fp, mmap_mode='r')
    if (v.dtype != np.float32 or f.dtype != np.int32 or not v.flags.c_contiguous
            or not f.flags.c_contiguous or v.shape != (finalization['vertices'], 3)
            or f.shape != (finalization['triangles'], 3)):
        raise ValueError('Original native array shape/layout mismatch')
    build_path = root / '.tools/native_precision/build_manifest.json'
    build = read_json(build_path)
    for entry in build['workers']['write_native_obj']['files']:
        _verify_record(entry)
    obj = scene / 'final_ground.obj'
    if obj.exists():
        raise ValueError('Existing OBJ retained; this recovery only writes a missing output')
    process = run_worker([str(root / '.tools/native_precision/write_native_obj'),
        str(vp), str(v.offset), str(len(v)), str(fp), str(f.offset), str(len(f)), str(obj)],
        cwd=root, log_path=evidence / 'obj_writer.log', timeout=3600,
        estimated_memory_bytes=2 * 2**30, estimated_disk_bytes=len(v) * 80 + len(f) * 36)
    if process['exit_code'] != 0:
        raise RuntimeError('Recovery writer did not complete; partial output is not a valid OBJ')
    identity = bind_native_ground(obj, evidence)
    report = {'schema_version': 1, 'status': 'exact_obj_recovered', 'at_utc': utc_now(),
        'scene_directory': str(scene), 'scope': 'completed native arrays exported; original interrupted process remains incomplete',
        'geometry_regenerated': False, 'original_process_exit_code': None,
        'original_process_status': 'interrupted_by_host_reboot_before_final_process_receipt',
        'recovery_boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'files': [file_record(p, role) for p, role in [(vp, 'native_vertices'), (fp, 'native_triangles'),
            (native / 'finalization.json', 'actual_native_finalization'), (global_path, 'complete_global_intersections'),
            (build_path, 'exact_obj_writer_build'), (obj, 'authoritative_obj'),
            (identity['identity_report'], 'independent_all_coordinate_and_index_comparison')]],
        'producer_sha256': sha256_file(Path(__file__)), 'source_and_world_qualification': 'not_run'}
    atomic_json(evidence / 'recovery.json', report)
    print({'status': report['status'], 'vertices': len(v), 'triangles': len(f), 'evidence': str(evidence)})


if __name__ == '__main__':
    main()
