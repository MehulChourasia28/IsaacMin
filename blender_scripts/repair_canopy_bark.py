"""Measure bark UVs in a native canopy without changing its geometry or masters."""
import bpy
import hashlib
import json
import sys
from pathlib import Path
import numpy as np

from isaacmin.assets.bark_uv import metric_bark_uv
import isaacmin.assets.bark_uv as bark_module


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def mesh_identity(mesh):
    """Native arrays, not bounds or counts alone, identify retained geometry."""
    proof = {}
    for name, collection, property_name, width, dtype in [
        ('positions', mesh.vertices, 'co', 3, np.float32),
        ('loop_vertices', mesh.loops, 'vertex_index', 1, np.int32),
        ('face_loop_starts', mesh.polygons, 'loop_start', 1, np.int32),
        ('face_loop_counts', mesh.polygons, 'loop_total', 1, np.int32),
        ('face_material_indices', mesh.polygons, 'material_index', 1, np.int32),
    ]:
        array = np.empty(len(collection) * width, dtype=dtype)
        collection.foreach_get(property_name, array)
        proof[name] = hashlib.sha256(memoryview(array).cast('B')).hexdigest()
        del array
    return proof


request_path = Path(sys.argv[sys.argv.index('--') + 1]).resolve()
request = json.loads(request_path.read_text())
parent_path = Path(request['parent_generation']).resolve()
parent = json.loads(parent_path.read_text())
parent_blend = Path(parent['output_blend']).resolve()
output = Path(request['output']).resolve()
if output.exists():
    raise RuntimeError('Use a new immutable bark correction output directory')
inputs = json.loads(Path(request['input_manifest']).read_text())


def verify_inputs():
    if inputs['execution_script_sha256'] != sha(__file__):
        raise RuntimeError('Frozen bark correction script changed')
    for item in inputs['files']:
        if sha(item['path']) != item['sha256']:
            raise RuntimeError('Frozen bark correction input changed')
    if sha(parent_blend) != parent['output_sha256']:
        raise RuntimeError('Original generated canopy changed')


verify_inputs()
output.mkdir(parents=True)
bpy.ops.wm.open_mainfile(filepath=str(parent_blend), load_ui=False, use_scripts=False)
objects = [bpy.data.objects[name] for v in parent['variants'] for name in v['objects']]
before = {obj.name: mesh_identity(obj.data) for obj in objects}
corrections = []
for variant in parent['variants']:
    trunks = [bpy.data.objects[name] for name in variant['objects'] if name.endswith('_trunk')]
    if len(trunks) != 1:
        raise RuntimeError('Exactly one original trunk per canopy variant is required')
    obj = trunks[0]
    mesh = obj.data
    layer = mesh.uv_layers.active
    if not layer:
        raise RuntimeError('Native bevel UVs are missing')
    points = np.empty((len(mesh.vertices), 3), np.float32)
    edges = np.empty((len(mesh.edges), 2), np.int32)
    loops = np.empty(len(mesh.loops), np.int32)
    native_uv = np.empty((len(mesh.loops), 2), np.float32)
    mesh.vertices.foreach_get('co', points.ravel())
    mesh.edges.foreach_get('vertices', edges.ravel())
    mesh.loops.foreach_get('vertex_index', loops)
    layer.data.foreach_get('uv', native_uv.ravel())
    result, evidence = metric_bark_uv(points, edges, loops, native_uv,
                                     request.get('repeat_m', [.6, .6]))
    evidence['before_uv_sha256'] = hashlib.sha256(native_uv.tobytes()).hexdigest()
    evidence['after_uv_sha256'] = hashlib.sha256(result.tobytes()).hexdigest()
    layer.data.foreach_set('uv', result.ravel())
    layer.name = 'MetricBarkUV'
    mesh.update()
    verified = np.empty_like(result)
    layer.data.foreach_get('uv', verified.ravel())
    if not np.array_equal(result, verified):
        raise RuntimeError('Native Blender did not retain the exact metric UVs')
    evidence['native_blender_uv_roundtrip'] = 'exact'
    evidence['producer_sha256'] = sha(bark_module.__file__)
    evidence['object'] = obj.name
    variant['bark_uv'] = evidence
    corrections.append(evidence)
    del points, edges, loops, native_uv, result, verified
after = {obj.name: mesh_identity(obj.data) for obj in objects}
if before != after:
    raise RuntimeError('Bark UV correction changed native canopy geometry or material assignments')
for text in list(bpy.data.texts):
    bpy.data.texts.remove(text)
blend = output / (parent['recipe_species'] + '_variants.blend')
bpy.ops.wm.save_as_mainfile(filepath=str(blend), compress=True)
verify_inputs()
parent.update(output_blend=str(blend), output_sha256=sha(blend), isaac_qualification='not_run')
parent.setdefault('postprocesses', []).append({
    'operation': 'metric_bark_uv', 'parent_generation': str(parent_path),
    'parent_generation_sha256': sha(parent_path), 'parent_blend': str(parent_blend),
    'parent_blend_sha256': sha(parent_blend), 'execution_script_sha256': sha(__file__),
    'request_sha256': sha(request_path), 'input_manifest': str(Path(request['input_manifest']).resolve()),
    'input_manifest_sha256': sha(request['input_manifest']),
    'geometry_before': before, 'geometry_after': after, 'geometry_bytes_unchanged': True,
    'corrections': corrections, 'blender_version': bpy.app.version_string,
    'target_renderer_qualification': 'not_run',
})
(output / 'generation.json').write_text(json.dumps(parent, indent=2) + '\n')
print(json.dumps({'status': 'external_tool_verified', 'output': str(output),
                  'geometry_bytes_unchanged': True, 'target_renderer_qualification': 'not_run'}))
