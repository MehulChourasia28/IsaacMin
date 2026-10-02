"""CPU Blender diagnostic only: never Isaac qualification evidence."""
import bpy
import math
import json
import sys
from pathlib import Path
from mathutils import Vector

folder=Path(sys.argv[sys.argv.index('--')+1]).resolve()
record=json.loads((folder/'generation.json').read_text())
bpy.ops.wm.open_mainfile(filepath=record['output_blend'],use_scripts=False,load_ui=False)
names=record['variants'][0]['objects']
for obj in bpy.context.scene.objects:
    obj.hide_render=obj.name not in names
scene=bpy.context.scene
scene.render.engine='CYCLES';scene.cycles.device='CPU';scene.cycles.samples=64;scene.cycles.use_denoising=False
scene.render.threads_mode='FIXED';scene.render.threads=8
scene.render.resolution_x=640;scene.render.resolution_y=800;scene.render.resolution_percentage=100
scene.world=bpy.data.worlds.new('DiagnosticSky');scene.world.use_nodes=True
nodes=scene.world.node_tree.nodes
sky=nodes.new('ShaderNodeTexSky');sky.sky_type='NISHITA';sky.sun_elevation=.75
scene.world.node_tree.links.new(sky.outputs['Color'],nodes.get('Background').inputs['Color'])
nodes.get('Background').inputs['Strength'].default_value=.4
bpy.ops.object.camera_add(location=(17,-20,2))
camera=bpy.context.object
camera.rotation_euler=(Vector((0,0,6.5))-camera.location).to_track_quat('-Z','Y').to_euler()
camera.data.lens=45;scene.camera=camera
scene.render.image_settings.file_format='PNG';scene.render.filepath=str(folder/'blender_diagnostic_only.png')
bpy.ops.render.render(write_still=True)
