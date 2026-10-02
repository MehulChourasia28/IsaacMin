"""Original licensed asset in CPU Cycles: diagnostic, never Isaac world evidence."""
import json
import sys
from pathlib import Path
import bpy
from mathutils import Vector

request = json.loads(Path(sys.argv[sys.argv.index('--') + 1]).read_text())
out = Path(request['output']); out.mkdir(parents=True, exist_ok=False)
card = json.loads(Path(request['card']).read_text())
master = next(Path(f['path']) for f in card['files'] if f['path'].endswith('.blend'))
bpy.ops.wm.open_mainfile(filepath=str(master), use_scripts=False, load_ui=False)
target = bpy.data.objects[request['object_name']]
for obj in bpy.context.scene.objects:
    obj.hide_render = obj != target
target.hide_render = False
scene = bpy.context.scene
scene.render.engine = 'CYCLES'; scene.cycles.device = 'CPU'
scene.cycles.samples = 64; scene.cycles.use_denoising = False
scene.render.threads_mode = 'FIXED'; scene.render.threads = 8
scene.render.resolution_x = 720; scene.render.resolution_y = 900
scene.render.resolution_percentage = 100
scene.world = bpy.data.worlds.new('OriginalAssetDiagnosticSky'); scene.world.use_nodes = True
nodes = scene.world.node_tree.nodes
sky = nodes.new('ShaderNodeTexSky'); sky.sky_type = 'NISHITA'; sky.sun_elevation = .75
scene.world.node_tree.links.new(sky.outputs['Color'], nodes.get('Background').inputs['Color'])
nodes.get('Background').inputs['Strength'].default_value = .4
corners = [target.matrix_world @ Vector(p) for p in target.bound_box]
lo = Vector(tuple(min(p[i] for p in corners) for i in range(3)))
hi = Vector(tuple(max(p[i] for p in corners) for i in range(3)))
centre = (lo + hi) * .5; height = hi.z - lo.z
bpy.ops.object.camera_add(location=centre + Vector((height*.9, -height*1.3, -height*.4)))
camera = bpy.context.object
camera.rotation_euler = (centre-camera.location).to_track_quat('-Z', 'Y').to_euler()
camera.data.lens = 42; scene.camera = camera
scene.view_settings.view_transform = 'AgX'; scene.view_settings.exposure = 0
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = str(out/'original_blender_diagnostic.png')
(out/'diagnostic.json').write_text(json.dumps(dict(scope='original_asset_CPU_Blender_only_not_Isaac_world',
    original_master=str(master), original_object=target.name, original_geometry_materials_unchanged=True,
    vertex_count=len(target.data.vertices), bounds_world=[list(lo),list(hi)],
    native_engine='Cycles_CPU', blender_version=bpy.app.version_string,
    samples=64, denoising=False, scripts_enabled=False,
    compatibility='Diagnostic only; review worker log for newer-master conversion warnings'), indent=2))
bpy.ops.render.render(write_still=True)
