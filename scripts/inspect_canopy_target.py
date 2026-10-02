#!/usr/bin/env python3
"""Actual canopy on an explicit technical terrain fixture, never a saved-world view."""
from pathlib import Path
import argparse
import json
import shutil
import numpy as np
import trimesh

from isaacmin.io import atomic_json, read_json, sha256_file
from isaacmin.assembly.pipeline import assemble_region
from isaacmin.assembly.material_recipe import selected_material_recipe, validate_material_recipe
from isaacmin.assets.candidates import candidate_lighting
from isaacmin.adapters.workers import capture_scene


def support(mesh, x, y):
    points, _, _ = mesh.ray.intersects_location(np.asarray([[x, y, 50.]]),
        np.asarray([[0., 0., -1.]]), multiple_hits=True)
    if not len(points):
        raise RuntimeError('Technical camera or tree has no actual ground support')
    return float(points[:, 2].max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixture', type=Path,
        default=Path('artifacts/development/shared_original_4k_assembly_repair01_20261001'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    if output.exists():
        raise ValueError('Use a fresh immutable inspection directory')
    output.mkdir(parents=True)
    generation = args.generation.resolve()
    canopy = read_json(generation)
    blend = Path(canopy['output_blend'])
    if sha256_file(blend) != canopy['output_sha256']:
        raise ValueError('Actual generated canopy bytes changed')
    variant = canopy['variants'][0]
    fixture = args.fixture.resolve()
    for name in ('technical_ir', 'technical_volume'):
        shutil.copytree(fixture / name, output / name)
    ir = output / 'technical_ir'
    # Remove derived input-conditioning output from the copied fixture. It is
    # never source data; each inspection executes the current constructor.
    cached_input = output / 'technical_volume/native_precision_input'
    if cached_input.exists():
        shutil.rmtree(cached_input)
    recipe = selected_material_recipe(root)
    if recipe is None:
        raise ValueError('Canopy inspection requires the explicit original4K ground candidate')
    common = dict(materials=validate_material_recipe(root, recipe), origin=(0, 0, 0),
        volume_directory=output / 'technical_volume', source_water_mesh_path=ir / 'water_interfaces.json',
        geometry_detail={'subdivision_levels': 3, 'soil_displacement_peak_to_peak_m': .02},
        exterior_delta={'path': str(ir / 'delta.npz'), 'source_height_key': 'height',
            'delta_key': 'delta', 'protection_key': 'protection', 'source_min_xz': [0, 0]},
        terrain_material_recipe=recipe)
    bare = assemble_region(ir, output / 'bare', **common)
    if bare['status'] != 'success':
        raise RuntimeError('Technical supporting terrain export failed')
    mesh = trimesh.load(output / 'bare/final_ground.obj', force='mesh', process=False)
    x, y = 8.5, -8.5
    z = support(mesh, x, y)
    anchors = np.asarray(variant['contact_anchors_local_m']) + [x, y, z]
    offsets = [float(a[2] - support(mesh, a[0], a[1])) for a in anchors]
    asset = {'id': 'canopy_target_candidate', 'asset_id': canopy['asset_id'], 'blend': str(blend),
        'objects': variant['objects'], 'alpha_mode': 'cutout', 'position': [x, y, z], 'scale': 1, 'yaw': 0}
    protocol = {'scope': 'actual_native_canopy_on_synthetic_technical_terrain',
        'minecraft_world_evidence': False, 'generation': str(generation),
        'generation_sha256': sha256_file(generation), 'blend_sha256': canopy['output_sha256'],
        'tree_root_xyz': [x, y, z], 'root_anchor_ground_offsets_m': offsets,
        'root_contact_status': 'pass' if max(map(abs, offsets)) <= .02 else 'fail',
        'root_contact_scope': 'diagnostic placement only; world grounding must be tested separately',
        'appearance_qualification': 'not_run', 'renderer_recipe': 'pathtracing_1024',
        'script_sha256': sha256_file(Path(__file__))}
    atomic_json(output / 'protocol.json', protocol)
    populated = assemble_region(ir, output / 'scene', assets=[asset],
        bare_scene_directory=output / 'bare', **common)
    if populated['status'] != 'success':
        raise RuntimeError('Actual canopy native USD export failed')
    poses = []
    for kind, px, py, clearance, target in [
        ('canopy_whole', 1., -1., 1.5, [x, y, z + 5.]),
        ('canopy_trunk_close', 6.8, -8.5, .6, [x, y, z + 1.1]),
        ('canopy_understorey', 10.5, -6.5, .6, [x, y, z + 4.5])]:
        poses.append({'kind': kind, 'position': [px, py, support(mesh, px, py) + clearance],
                      'look_at': target, 'ground_clearance_m': clearance,
                      'support_reference': 'independent_final_OBJ_ray'})
    px, py = 2.5, -2.5
    probe_ground = support(mesh, px, py)
    probes = [{'position': [px, py, probe_ground + 1.], 'ground_z': probe_ground}]
    lighting = candidate_lighting(root)['conditions']['directional']
    capture = capture_scene(output / 'scene/world.usda', output / 'capture', poses=poses,
        contact_probes=probes, hdri=lighting['hdri'], lighting_parameters=lighting,
        renderer_recipe='pathtracing_1024', timeout=7200,
        scope='actual_canopy_technical_fixture_not_minecraft_world')
    atomic_json(output / 'inspection.json', {'protocol': protocol, 'export_status': populated['status'],
        'capture_status': capture['status'], 'appearance_qualification': 'not_run',
        'frames': len(capture.get('frames', []))})
    print(json.dumps({'output': str(output), 'capture_status': capture['status'],
                      'appearance_qualification': 'not_run'}))


if __name__ == '__main__':
    main()
