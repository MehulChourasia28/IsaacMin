"""Prepare a separate portable scene with exact collision partitions on CPU."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root / 'src'), str(root / 'isaac_scripts')]
from isaacmin.io import atomic_json, sha256_file
from isaacmin.assets.usd_provenance import verify_scene_closure
from isaacmin.assembly.usd_instances import _closure
from ground_collision import configure_ground_collision, verify_exact_parts
from pxr import Usd, Sdf


def prepare(source, destination, ground_source):
    source, destination, ground_source = map(lambda p: Path(p).resolve(), (source, destination, ground_source))
    if destination.exists() or destination.is_relative_to(source.parent):
        raise ValueError('Use a fresh candidate outside the original scene')
    started = time.monotonic()
    closure_sha = sha256_file(source.parent / 'native_dependency_closure.json')
    closure = verify_scene_closure(source, closure_sha)
    staging = destination.with_name(destination.name + '.staging')
    if staging.exists():
        raise ValueError('Preserve earlier collision staging')
    staging.mkdir(parents=True)
    for item in closure['files']:
        original = Path(item['path']); relative = original.relative_to(source.parent)
        target = staging / relative; target.parent.mkdir(parents=True, exist_ok=True)
        if relative.as_posix() == 'content.usdc':
            # A fresh crate removes unreachable old array blocks left by Sdf.Save.
            if not Sdf.Layer.FindOrOpen(str(original)).Export(str(target)):
                raise ValueError('Could not compact the unchanged native content layer')
        else:
            shutil.copyfile(original, target)
    original_export = json.loads((ground_source / 'export_result.json').read_text())
    if sha256_file(ground_source / 'final_ground.obj') != original_export['final_ground_sha256']:
        raise ValueError('Original exact ground changed')
    extra = ['final_ground.obj', 'source_water_interfaces.json',
             'native_precision/final_vertices.npy', 'native_precision/final_triangles.npy',
             'native_precision/finalization.json']
    for name in extra:
        original = ground_source / name
        if original.is_file():
            target = staging / name; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
            if sha256_file(target) != sha256_file(original):
                raise ValueError('Ground copy differed')
    scene = staging / source.name
    stage = Usd.Stage.Open(str(scene))
    collision = configure_ground_collision(stage)
    stage.GetRootLayer().Save()
    stage = Usd.Stage.Open(str(scene))
    collision['saved_parts_rechecked'] = verify_exact_parts(stage)
    output_closure = _closure(scene)
    atomic_json(staging / 'native_dependency_closure.json', output_closure)
    verify_scene_closure(source, closure_sha)
    atomic_json(staging / 'collision_preparation.json', {
        'status': 'all_saved_collision_triangles_match_rendered_source',
        'source_scene': str(source), 'source_closure_sha256': closure_sha,
        'source_ground_sha256': original_export['final_ground_sha256'],
        'collision': collision, 'producer_sha256': sha256_file(root / 'isaac_scripts/ground_collision.py'),
        'native_PhysX_and_render_qualification': 'not_run',
        'elapsed_seconds': time.monotonic() - started})
    staging.rename(destination)
    print(json.dumps({'status': 'prepared', 'scene': str(destination / source.name),
                      'parts': len(collision['collision_meshes']),
                      'source_triangles': sum(c['triangles'] for c in collision['partition_comparisons']),
                      'elapsed_seconds': time.monotonic() - started}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--source', required=True)
    p.add_argument('--destination', required=True); p.add_argument('--ground-source', required=True)
    a = p.parse_args(); prepare(a.source, a.destination, a.ground_source)
