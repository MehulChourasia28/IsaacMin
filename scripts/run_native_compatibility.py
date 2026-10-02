#!/usr/bin/env python3
"""Real native chain on an explicitly synthetic fixture, never map/realism evidence."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import json
import numpy as np
from isaacmin.adapters.workers import refine_heightfield,reconstruct_occupancy,capture_scene,project_root,_sha
from isaacmin.assembly.pipeline import assemble_region

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--renderer-recipe',default='legacy_rtx_8',
                    choices=['legacy_rtx_8','native_resolution_rtx_128','pathtracing_1024'])
parser.add_argument('--shared-original-materials',action='store_true')
args=parser.parse_args()
root=project_root();base=root/'artifacts/bootstrap'
# Retain complete previous attempts before any worker writes a same-named output.
# This is a technical fixture command; it never repairs actual-map geometry.
previous=[base/name for name in ('cross_ir','cross_highmap','cross_volume',
    'cross_bare_blender','cross_populated_export','cross_instanced_export',
    'cross_native_instancing','cross_exact_collision',
    'blender','cross_isaac','cross_runtime.json') if (base/name).exists()]
if previous:
    archived=base/'attempts'/('native_chain_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    archived.mkdir(parents=True,exist_ok=False)
    records=[]
    for path in previous:
        records.append({'original_path':str(path.relative_to(root)),'archived_name':path.name})
        path.rename(archived/path.name)
    (archived/'archive_manifest.json').write_text(json.dumps({
        'reason':'preserve native compatibility before fresh execution','entries':records},indent=2)+'\n')
ir=base/'cross_ir';ir.mkdir(parents=True,exist_ok=True)
x,z=np.meshgrid(np.arange(17),np.arange(17))
height=np.floor(9+0.1*x+0.08*z+0.3*np.sin(x/3)).astype('f4')
ys=np.arange(20)[:,None,None]
occ=ys<height[None,:,:]
occ[2:6,6:10,:]=False
protection=np.zeros(height.shape,bool);protection[5:11,:]=True
np.savez_compressed(ir/'natural_occupancy.npz',occupancy=occ.astype('u1'),
                    validity=np.ones(occ.shape,bool),min_xyz=np.array([0,0,0]),voxel_size_m=1)
water_validity=np.zeros(height.shape,bool);water_validity[13:16,13:16]=True
np.savez_compressed(ir/'terrain_surface.npz',height=height,validity=np.ones(height.shape,bool),
                    water_validity=water_validity,water_height=height+1,min_xz=np.array([0,0]),sample_spacing_m=1.0,
                    biome=np.where(np.indices(height.shape)[0]<8,'minecraft:forest','minecraft:plains'),
                    substrate_id=np.where(np.indices(height.shape)[1]<4,0,1).astype('i2'),
                    block_names=np.asarray(['minecraft:dirt','minecraft:grass_block']))
from isaacmin.assembly.water import fluid_surface_mesh
fluid_kind=occ.astype('u1');fluid_amount=np.zeros_like(fluid_kind)
for iz,ix in np.argwhere(water_validity):
    fluid_kind[int(height[iz,ix]),iz,ix]=2;fluid_amount[int(height[iz,ix]),iz,ix]=8
fluid_kind[2,6:10,2:5]=2;fluid_amount[2,6:10,2:5]=8
water_interfaces=fluid_surface_mesh(fluid_kind,fluid_amount,np.array([0,0,0]),height,origin=(0,0,0))
water_interfaces['source_data_kind']='explicit_synthetic_native_integration_fixture'
water_path=ir/'water_interfaces.json';water_path.write_text(json.dumps(water_interfaces)+'\n')
refine=refine_heightfield(height,base/'cross_highmap',protection=protection)
if refine['status']!='success':raise RuntimeError(refine)
delta=np.load(refine['arrays'])['delta']
np.savez_compressed(ir/'delta.npz',height=height,source_height=height,delta=delta,protection=protection)
from isaacmin.contracts.coordinates import CoordinateFrame
from isaacmin.validation.profile import freeze_profile
from isaacmin.validation.source_fidelity import freeze_source_fidelity_profile
profile=freeze_profile(ir/'quality_profile.json',{'coordinates':CoordinateFrame((0,0,0)).record(),
    'scope':'explicit_synthetic_native_integration_fixture'})
freeze_source_fidelity_profile(profile,ir/'source_fidelity_profile.json')
volume=reconstruct_occupancy(occ,base/'cross_volume')
materials=json.loads((root/'state/terrain_materials.json').read_text())['materials'][:2]
material_recipe=None
if args.shared_original_materials:
    from isaacmin.assembly.material_recipe import PRODUCERS,SHARED_ORIGINAL,validate_material_recipe
    candidate='state/material_candidates/shared_mdl_4k_20261001.json'
    material_recipe={'mode':SHARED_ORIGINAL,'candidate_manifest':candidate,
        'candidate_manifest_sha256':_sha(root/candidate),
        'producer_files':{name:_sha(root/name) for name in PRODUCERS}}
    materials=validate_material_recipe(root,material_recipe)
normalized=json.loads((root/'state/normalized_assets.json').read_text())['assets']
assets=[]
for record in normalized:
    if record['asset_id']=='fern_02':
        # A real downloaded leaf cutout is included to expose target import problems.
        path=record.get('normalized_blend') or record.get('output_blend')
        if path:
            obj=record['objects'][0] if record.get('objects') else None
            name=obj['object_name'] if isinstance(obj,dict) else obj
            assets.append({'id':'fern_02','alpha_mode':'cutout','blend':path,'objects':[name] if name else [],
                           'position':[8.5,-4.5,float(height[4,8])],'scale':1,'yaw':0})
assembly_arguments=dict(materials=materials,origin=(0,0,0),
                      terrain_material_recipe=material_recipe,
                      source_fidelity_budget_path=ir/'source_fidelity_profile.json',
                      source_water_mesh_path=water_path,
                      volume_directory=base/'cross_volume',geometry_detail={'subdivision_levels':3,'soil_displacement_peak_to_peak_m':.02},exterior_delta={
                          'path':str(ir/'delta.npz'),'source_height_key':'height','delta_key':'delta',
                          'protection_key':'protection','source_min_xz':[0,0]})
bare=assemble_region(ir,base/'cross_bare_blender',**assembly_arguments)
if bare['status']!='success':raise RuntimeError(bare)
result=assemble_region(ir,base/'cross_populated_export',assets=assets,
                      bare_scene_directory=base/'cross_bare_blender',**assembly_arguments)
if result['status']!='success':raise RuntimeError(result)
from isaacmin.assembly.finalize import instance_scene,collision_scene
instances=instance_scene(root,base/'cross_populated_export',base/'cross_instanced_export',
                         base/'cross_native_instancing')
result=collision_scene(root,base/'cross_instanced_export',base/'cross_populated_export',
                       base/'blender',base/'cross_exact_collision',instances)
(base/'blender/result.json').write_text(json.dumps(result,indent=2)+'\n')
poses=[{'position':[3.5+i*0.025,-3.5,float(height[3,3])+0.6],
        'look_at':[12.5,-5.5,float(height[3,3])+0.3],'kind':'motion'} for i in range(12)]
poses.extend([{'position':[7.8,-4.5,10.28],'look_at':[8.5,-4.5,10.1],'kind':'material_alpha'},
              {'position':[8.5,-5.2,10.28],'look_at':[8.5,-4.5,10.1],'kind':'material_alpha'}])
poses.append({'position':[1.5,-8.,3.6],'look_at':[5.,-8.,2.7],'kind':'aquifer_water'})
scene=base/'blender/world.usda'
import trimesh
mesh=trimesh.load(base/'blender/final_ground.obj',force='mesh',process=False)
for pose in poses:
    if pose.get('kind')!='motion':continue
    x,y,_=pose['position']
    hits,_,_=mesh.ray.intersects_location(np.asarray([[x,y,30.0]]),np.asarray([[0,0,-1]]),multiple_hits=False)
    if len(hits)!=1:raise RuntimeError('Native compatibility camera has no measured exterior support')
    pose['position'][2]=float(hits[0,2])+.6
    pose['look_at']=[x+6,y-2,pose['position'][2]-.3]
    pose['ground_follow_reference']='independent_final_OBJ_ray'
    pose['ground_clearance_m']=.6

probes=[{'position':[2.5,-2.5,float(height[2,2])+1]},
        {'position':[4,-8,3.5]}]
for probe in probes:
    hits,_,_=mesh.ray.intersects_location(np.asarray([probe['position']]),
                                         np.asarray([[0,0,-1]]),multiple_hits=True)
    if not len(hits):raise RuntimeError('Independent final OBJ support ray missed fixture')
    probe['ground_z']=float(hits[:,2].max())
    probe['reference']='independent_trimesh_ray_on_final_blender_obj'
    p=probe['position']
    footprint=np.asarray([[p[0]+dx,p[1]+dy,p[2]] for dx in [-.15,0,.15] for dy in [-.15,0,.15]])
    floor,_,_=mesh.ray.intersects_location(footprint,np.tile([0,0,-1],(len(footprint),1)),multiple_hits=False)
    if len(floor)!=len(footprint) or np.ptp(floor[:,2])>.012:
        raise RuntimeError('Technical contact probe requires an independently measured near-flat support footprint')
    probe['footprint_support_spread_m']=float(np.ptp(floor[:,2]))
(base/'blender/contact_reference.json').write_text(json.dumps(probes,indent=2)+'\n')
captures=capture_scene(scene,base/'cross_isaac',poses=poses,
                       contact_probes=probes,
                       renderer_recipe=args.renderer_recipe,
                       scope='synthetic_native_cross_runtime_only')
dependencies=[root/'.tools/build/native-gcc13/isaacmin_highmap',root/'.tools/openvdb_worker',
 root/'.tools/blender/blender',root/'blender_scripts/export_scene.py',root/'isaac_scripts/capture_scene.py',
 root/'isaac_scripts/reopen_scene.py',root/'isaac_scripts/contact_rays.py',root/'isaac_scripts/material_log.py',root/'scripts/run_native_compatibility.py',
 root/'isaac_scripts/ground_collision.py',root/'src/isaacmin/assembly/finalize.py',
 root/'src/isaacmin/assembly/usd_instances.py',root/'scripts/share_native_scene.py',
 root/'scripts/prepare_exact_collision_scene.py',
 base/'cross_native_instancing/sharing.json',base/'blender/collision_preparation.json',
 root/'src/isaacmin/adapters/workers.py',root/'src/isaacmin/assembly/pipeline.py',
 root/'src/isaacmin/assembly/native_cache.py',
 root/'blender_scripts/mesh_arrays.py',root/'src/isaacmin/assembly/water.py',
 root/'src/isaacmin/assembly/material_assignment.py',root/'src/isaacmin/assembly/material_bake.py',
 base/'highmap/worker_result.json',base/'openvdb/worker_result.json',
 base/'openvdb_independent/comparison.json',base/'blender/export_result.json',
 base/'blender/final_ground.obj',base/'blender/native_dependency_closure.json',
 base/'cross_isaac/capture_request.json',base/'cross_isaac/capture_result.json',
 *sorted(ir.glob('*.npz')),ir/'quality_profile.json',ir/'source_fidelity_profile.json',
 water_path,base/'blender/source_water_interfaces.json']
from isaacmin.assembly.native_cache import PRODUCERS as ASSEMBLY_PRODUCERS
dependencies.extend(root/name for name in ASSEMBLY_PRODUCERS)
if material_recipe:dependencies.append(root/material_recipe['candidate_manifest'])
dependencies.extend(base/'cross_bare_blender'/name for name in (
    'bare_scene_cache.json','export_result.json','native_dependency_closure.json','scene.blend','final_ground.obj'))
for item in json.loads((base/'cross_bare_blender/bare_scene_cache.json').read_text())['files']:
 dependencies.append(base/'cross_bare_blender'/item['path'])
for item in json.loads((base/'blender/native_dependency_closure.json').read_text())['files']:
 dependencies.append(base/'blender'/item['path'])
if captures.get('physics_scene'):
 dependencies.extend([Path(captures['physics_scene']),base/'blender/runtime_dependencies.json',
                      base/'blender/native_physics_dependency_closure.json',base/'cross_isaac/reopen_result.json',
                      base/'cross_isaac/physx_contact_rays.json',base/'cross_isaac/physx_contact_rays.npz'])
if captures.get('motion_video',{}).get('status')=='pass':dependencies.append(Path(captures['motion_video']['path']))
for frame in captures.get('frames',[]):
 dependencies.extend(base/'cross_isaac'/frame[key] for key in ('rgb','depth','instance_segmentation'))
tool_config=root/'state/native_tools.json'
runtime=Path(json.loads(tool_config.read_text())['isaac_python']) if tool_config.is_file() else Path('/home/mehulchourasia/IsaacSim/_build/linux-aarch64/release/python.sh')
dependencies.extend([runtime,runtime.parent/'VERSION'])
captures['chain_provenance']=[{'path':str(path.resolve().relative_to(root)) if path.resolve().is_relative_to(root) else str(path.resolve()),
                              'sha256':_sha(path)} for path in sorted(set(dependencies)) if path.is_file()]
missing=[str(path) for path in dependencies if not path.is_file()]
if missing:captures.update(status='failed',missing_chain_artifacts=missing)
(base/'cross_runtime.json').write_text(json.dumps(captures,indent=2)+'\n')
print(json.dumps({'status':captures['status'],'evidence':'artifacts/bootstrap/cross_runtime.json'}))
