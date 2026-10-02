"""Native CPU USD sharing and exact collision as ordinary production stages."""
from pathlib import Path
import shutil

from isaacmin.io import atomic_json, read_json, sha256_file
from isaacmin.process import run_worker


def _execute(root, script, arguments, evidence, memory_gib):
    root, evidence = Path(root).resolve(), Path(evidence).resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    result = run_worker([str(root / '.venv/bin/python'), str(root / 'scripts' / script), *arguments],
        cwd=root, log_path=evidence / 'worker.log', timeout=3600,
        environment={'PYTHONPATH': str(root / '.tools/usd/lib/python') + ':' + str(root / 'src'),
                     'LD_LIBRARY_PATH': str(root / '.tools/native/usr/lib/aarch64-linux-gnu') + ':' + str(root / '.tools/native/usr/lib'),
                     'PXR_WORK_THREAD_LIMIT': '8'},
        estimated_memory_bytes=memory_gib * 2**30, estimated_disk_bytes=28 * 2**30)
    atomic_json(evidence / 'process_result.json', result)
    if result['exit_code'] != 0:
        raise RuntimeError('Native finalization failed: ' + script + '; see ' + str(evidence))
    return result


def instance_scene(root, source, output, evidence):
    source, output, evidence = map(lambda p: Path(p).resolve(), (source, output, evidence))
    process = _execute(root, 'share_native_scene.py', [
        '--source', str(source / 'world.usda'), '--destination', str(output),
        '--evidence', str(evidence / 'sharing.json')], evidence, 64)
    result = read_json(evidence / 'sharing.json')
    if result['status'] != 'lossless_native_instancing_verified':
        raise ValueError('No complete lossless native instance proof')
    return dict(result, process_receipt=str(evidence / 'process_result.json'),
                sharing_evidence={'path':str(evidence / 'sharing.json'),
                                  'sha256':sha256_file(evidence / 'sharing.json')})


def collision_scene(root, source, ground_source, output, evidence, instance_result):
    source, ground_source, output, evidence = map(lambda p: Path(p).resolve(),
                                                 (source, ground_source, output, evidence))
    _execute(root, 'prepare_exact_collision_scene.py', [
        '--source', str(source / 'world.usda'), '--destination', str(output),
        '--ground-source', str(ground_source)], evidence, 40)
    collision = read_json(output / 'collision_preparation.json')
    if collision['status'] != 'all_saved_collision_triangles_match_rendered_source':
        raise ValueError('Exact collision input geometry comparison is incomplete')
    original_export = read_json(ground_source / 'export_result.json')
    original_result = read_json(ground_source / 'assembly_result.json')
    shutil.copyfile(ground_source / 'export_result.json', output / 'blender_export_result.json')
    export = dict(original_export, usd_sha256=sha256_file(output / 'world.usda'),
        native_instance_count=instance_result['native_instances'],
        post_export={'source_blender_export_sha256': sha256_file(output / 'blender_export_result.json'),
                     'instancing_evidence': instance_result['sharing_evidence'],
                     'native_prototypes': instance_result['native_prototypes'],
                     'exact_collision': collision,
                     'geometry_material_transform_reduction': False,
                     'target_renderer_qualification': 'not_run'})
    atomic_json(output / 'export_result.json', export)
    result = dict(original_result, **{k: v for k, v in export.items() if k != 'status'})
    result.update(status='success', root_usd=str(output / 'world.usda'),
                  source_assembly_result_sha256=sha256_file(ground_source / 'assembly_result.json'),
                  qualification='not_qualified')
    atomic_json(output / 'assembly_result.json', result)
    return result
