"""Native Blender export of the authoritative outdoor surface arrays."""
from pathlib import Path
import sys
import json
import numpy as np
import bpy

request=json.loads(Path(sys.argv[sys.argv.index('--')+1]).read_text())
root=Path(request['workspace']);sys.path.insert(0,str(root/'src'))
from isaacmin.io import read_json,atomic_json,sha256_file,utc_now

terrain=Path(request['terrain']);out=Path(request['output']);out.mkdir(parents=True,exist_ok=True)
if not bpy.app.build_options.usd:raise RuntimeError('Native Blender USD exporter unavailable')
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.preferences.filepaths.use_scripts_auto_execute=False
bpy.context.scene.unit_settings.system='METRIC'
bpy.context.scene.unit_settings.scale_length=1.
partition=request.get('partition')
if partition:
    from isaacmin.terrain.surface_partitions import window_array,grid_triangles
    points=window_array(terrain,'vertices.npy',partition['halo'])
    z0,z1,x0,x1=partition['halo'];faces=grid_triangles(z1-z0+1,x1-x0+1)
else:
    points=np.load(terrain/'vertices.npy',mmap_mode='r');faces=np.load(terrain/'triangles.npy',mmap_mode='r')
mesh=bpy.data.meshes.new('Terrain_FinalGround_Surface')
mesh.vertices.add(len(points));mesh.vertices.foreach_set('co',points.ravel())
mesh.loops.add(faces.size);mesh.loops.foreach_set('vertex_index',faces.ravel())
mesh.polygons.add(len(faces));mesh.polygons.foreach_set('loop_start',np.arange(len(faces),dtype=np.int32)*3)
mesh.polygons.foreach_set('loop_total',np.full(len(faces),3,np.int32))
mesh.polygons.foreach_set('use_smooth',np.ones(len(faces),bool))
mesh.update()
obj=bpy.data.objects.new('Terrain_FinalGround_Surface',mesh);bpy.context.collection.objects.link(obj)
obj['surface_scope']='above_ground_navigation'
obj['terrain_manifest_sha256']=sha256_file(terrain/'terrain.json')
# Write native vertex attributes; USD authoring also verifies these on reopen.
attributes=[] if partition else [('IsaacMinMaterialWeights012','FLOAT_VECTOR'),('IsaacMinMaterialWeight3','FLOAT'),
            ('IsaacMinSurfaceRestPosition','FLOAT_VECTOR'),('IsaacMinSurfaceRestNormal','FLOAT_VECTOR')]
for name,kind in attributes:mesh.attributes.new(name=name,type=kind,domain='POINT')
if not partition:
    weights=np.load(terrain/'weights.npy',mmap_mode='r')
    mesh.attributes[attributes[0][0]].data.foreach_set('vector',np.ascontiguousarray(weights[:,:3]).ravel())
    mesh.attributes[attributes[1][0]].data.foreach_set('value',np.ascontiguousarray(weights[:,3]))
    for name,file in [('IsaacMinSurfaceRestPosition','rest_positions.npy'),('IsaacMinSurfaceRestNormal','rest_normals.npy')]:
        mesh.attributes[name].data.foreach_set('vector',np.load(terrain/file,mmap_mode='r').ravel())
if not partition and weights.shape[1] != 4:
    for index in range(weights.shape[1]):
        mesh.attributes.new(name='IsaacMinSurfaceWeight'+str(index),type='FLOAT',domain='POINT')
    for index in range(weights.shape[1]):
        mesh.attributes['IsaacMinSurfaceWeight'+str(index)].data.foreach_set('value',np.ascontiguousarray(weights[:,index]))
bpy.context.view_layer.objects.active=obj;obj.select_set(True)
if not partition:bpy.ops.wm.save_as_mainfile(filepath=str(out/'surface.blend'),check_existing=False)
settings=dict(filepath=str(out/'outdoor_terrain.usdc'),export_materials=False,export_normals=True,
    export_uvmaps=True,root_prim_path='/World/Outdoor',export_custom_properties=True,
    custom_properties_namespace='isaacmin',selected_objects_only=True)
available=bpy.ops.wm.usd_export.get_rna_type().properties.keys()
if partition:settings['root_prim_path']+='/'+partition['name']
settings={k:v for k,v in settings.items() if k in available}
result=bpy.ops.wm.usd_export(**settings)
if result!={'FINISHED'}:raise RuntimeError('Native Blender USD export failed')
atomic_json(out/'export_result.json',dict(status='native_Blender_USD_export_complete',at_utc=utc_now(),
    blender_version=bpy.app.version_string,vertices=len(points),triangles=len(faces),
    usd_sha256=sha256_file(out/'outdoor_terrain.usdc'),terrain_manifest_sha256=sha256_file(terrain/'terrain.json'),
    qualification='not_run',source_save_modified=False,producer_sha256=sha256_file(Path(__file__))))
print({'native_Blender_export':True,'vertices':len(points),'triangles':len(faces)})
