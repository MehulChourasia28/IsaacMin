"""Construct native Blender geometry from actual double-precision bilinear output."""
from pathlib import Path
import json
import bpy,numpy as np
from isaacmin.volumes.native_precision import subdivide_precise


def construct(request,output,root):
    output=Path(output);result=subdivide_precise(root,Path(request['terrain_mesh']),request['geometry_detail']['subdivision_levels'],output)
    report=json.loads((output/'subdivision.json').read_text());v=np.load(output/'vertices.npy');q=np.load(output/'quads.npy')
    mesh=bpy.data.meshes.new('PreciseBilinearGround');mesh.vertices.add(len(v));mesh.vertices.foreach_set('co',v.reshape(-1))
    mesh.loops.add(q.size);mesh.loops.foreach_set('vertex_index',q.reshape(-1));mesh.polygons.add(len(q))
    mesh.polygons.foreach_set('loop_start',np.arange(len(q),dtype=np.int32)*4);mesh.polygons.foreach_set('loop_total',np.full(len(q),4,np.int32));mesh.update(calc_edges=True)
    terrain=bpy.data.objects.new('Terrain_FinalGround',mesh);bpy.context.scene.collection.objects.link(terrain)
    terrain.select_set(True);bpy.context.view_layer.objects.active=terrain
    return terrain,report['input'],{'manifest':result,'method':'bilinear_double_no_limit_v1','counts':report['output']}
