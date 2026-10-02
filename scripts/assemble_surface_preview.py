"""Compose new outdoor geometry and source-driven instances in native USD.

The supplied library scene is a development asset/material donor only. Every
old ground, collider, water surface and placement is deactivated before the new
surface is composed. No historical qualification is inherited.
"""
from pathlib import Path
import argparse,os,shutil
import numpy as np
from pxr import Usd,UsdGeom,UsdShade,Sdf,Vt,Gf
from isaacmin.io import read_json,atomic_json,sha256_file,utc_now
from isaacmin.assembly.usd_instances import _closure
from isaacmin.terrain.surface_navigation import SurfaceQuery,source_fields
from isaacmin.assembly.surface_population import random_cells
from isaacmin.assembly.surface_canopy import copy_canopy_libraries,embedded_woody_root,select_canopy_prototype
from isaacmin.assembly.outdoor_lighting import capture_configuration,author_outdoor_lighting

parser=argparse.ArgumentParser();parser.add_argument('--request',required=True)
args=parser.parse_args();request=read_json(Path(args.request));root=Path(request['workspace'])
source=Path(request['library_scene']);terrain=Path(request['terrain']);out=Path(request['output'])
scene=out/'scene';scene.mkdir(parents=True,exist_ok=False)
for entry in read_json(source/'native_dependency_closure.json')['files']:
    p=source/entry['path'];q=scene/entry['path']
    if p.is_symlink() or not p.resolve().is_relative_to(source) or sha256_file(p)!=entry['sha256']:
        raise RuntimeError('Donor dependency changed')
    q.parent.mkdir(parents=True,exist_ok=True)
    if p.suffix in ('.mdl','.usda','.json'):shutil.copyfile(p,q)
    else:os.link(p,q)
(scene/'world.usda').rename(scene/'donor_root.usda')
stage=Usd.Stage.CreateNew(str(scene/'world.usda'))
overrides=Sdf.Layer.CreateNew(str(scene/'outdoor_overrides.usdc'))
stage.GetRootLayer().subLayerPaths=[overrides.identifier.rsplit('/',1)[-1],'donor_root.usda']
stage.SetDefaultPrim(stage.GetPrimAtPath('/World'))
UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(stage,1.)
stage.SetEditTarget(overrides)
prototypes={};by_object={}
index_path=source/'asset_index.json'
if index_path.is_file():
    for entry in read_json(index_path)['prototypes']:
        key=(entry['blend_sha256'],entry['object_name'])
        prototypes[key]=entry['prototype'];by_object[entry['object_name']]=entry['prototype']
        if not stage.GetPrimAtPath(entry['prototype']):
            raise ValueError('Standalone asset index names a missing prototype')
instances=[];old_ground=[];old_water=[]
for prim in stage.Traverse():
    obj=prim.GetAttribute('isaacmin:isaacmin_source_object').Get()
    if prim.IsInstance() and obj:
        refs=prim.GetMetadata('references').GetAppliedItems()
        key=(str(prim.GetAttribute('isaacmin:isaacmin_source_blend_sha256').Get()),str(obj))
        if len(refs)!=1 or refs[0].assetPath:raise ValueError('Expected local native prototype')
        prototypes[key]=str(refs[0].primPath);by_object[str(obj)]=str(refs[0].primPath);instances.append(prim)
    if prim.IsA(UsdGeom.Mesh) and 'Terrain_FinalGround' in str(prim.GetPath()):old_ground.append(prim)
    if prim.IsA(UsdGeom.Mesh) and 'SourceSurfaceWater' in str(prim.GetPath()):old_water.append(prim)
for prim in instances+old_ground+old_water:prim.SetActive(False)
for path in ['/World/MeadowGroundCover','/IsaacMinCollision']:
    prim=stage.GetPrimAtPath(path)
    if prim:prim.SetActive(False)
from isaacmin.assembly.surface_ground import attach_surface_ground
material_count,active_material_indices=attach_surface_ground(stage,terrain,scene,request['blender_export'])
# CPU displacement and shader share frozen metric chart coordinates.
mdl=scene/'shared_material/IsaacMinScans.mdl';text=mdl.read_text()
old='float3 p = state::transform_point(state::coordinate_internal,state::coordinate_world,state::position());'
new='float3 p = scene::data_lookup_float3("IsaacMinSurfaceRestPosition", state::transform_point(state::coordinate_internal,state::coordinate_world,state::position()));'
if old not in text:raise RuntimeError('Expected known native scan projection')
text=text.replace(old,new)
needle='    float3 w012 = scene::data_lookup_float3'
text=text.replace(needle,'    float3 chart_normal = math::normalize(scene::data_lookup_float3("IsaacMinSurfaceRestNormal",gn));\n'+needle)
text=text.replace(', p, gn);',', p, chart_normal);');mdl.write_text(text)
material=read_json(scene/'shared_material/material_manifest.json');material.update(
    module_sha256=sha256_file(mdl),physical_displacement_mapping='same frozen rest primvars and three-patch original scan coordinates',
    qualification='not_run_for_new_surface',recipe='outdoor_stochastic_scan_v1')
atomic_json(scene/'shared_material/material_manifest.json',material)
if material_count!=4:
    from isaacmin.assembly.surface_materials import prepare_surface_library
    terrain_record=read_json(terrain/'terrain.json');masters=read_json(Path(terrain_record['material_manifest']['path']))
    if sha256_file(Path(terrain_record['material_manifest']['path']))!=terrain_record['material_manifest']['sha256']:
        raise RuntimeError('Surface material manifest changed')
    surface_material=prepare_surface_library(root,masters['materials'],scene/'outdoor_material',active_indices=active_material_indices)
    atomic_json(out/'active_surface_materials.json',dict(
        active_indices=active_material_indices,material_order=terrain_record['material_order'],
        weight_sha256=sha256_file(terrain/'weights.npy'),
        proof='All omitted channels equal exactly zero at every vertex; same source maps and sampling on every contributing channel',
        appearance_regression='not_run'))
    shader=UsdShade.Shader(stage.GetPrimAtPath('/IsaacMinMaterials/SharedOriginalPBR/Shader'))
    shader.GetPrim().GetAttribute('info:mdl:sourceAsset').Set(Sdf.AssetPath('./outdoor_material/IsaacMinScans.mdl'))
    mat=stage.GetPrimAtPath('/IsaacMinMaterials/SharedOriginalPBR')
    mat.GetAttribute('isaacmin:manifest').Set(Sdf.AssetPath('./outdoor_material/material_manifest.json'))
    mat.GetAttribute('isaacmin:materialFamilies').Set(surface_material['material_order'])
    mat.GetAttribute('isaacmin:physicalRepeatMetres').Set([float(m['repeat_m']) for m in masters['materials']])
    mat.GetAttribute('isaacmin:scanDependencies').Set([Sdf.AssetPath('./outdoor_material/'+p['path']) for p in surface_material['textures']])
    mat.GetAttribute('isaacmin:recipe').Set('outdoor_biome_materials_v1')

population=read_json(Path(request['population'])/'population.json')
from isaacmin.assembly.living_grass import author_living_grass
author_living_grass(stage,scene)
from isaacmin.assembly.surface_instances import author_surface_instances
ground_cover_count=author_surface_instances(stage,'/World/OutdoorGroundCover',request['population'],population,prototypes)
query=SurfaceQuery(terrain);objects=read_json(Path(request['objects'])/'objects.json');trees=[];rejected={}
conifers=copy_canopy_libraries(root,scene,request.get('canopy_libraries',[]))
canopy_rules=read_json(root/request['canopy_rules'])['rules'] if request.get('canopy_rules') else []
tree_fields=source_fields(terrain/'material_fields.npz')
for index,tree in enumerate(objects['trees']):
    x,y=tree['source_x']-query.origin[0],-tree['source_z']+query.origin[2]
    z=float(query.heights(x,y))
    if not np.isfinite(z):continue
    yaw=float(random_cells(int(tree['source_x']),int(tree['source_z']),82)*360)
    ix=int(np.floor(tree['source_x']-tree_fields['min_xz'][0]));iz=int(np.floor(tree['source_z']-tree_fields['min_xz'][1]))
    biome=str(tree_fields['biome'][iz,ix])
    selected=select_canopy_prototype(tree,biome,conifers,canopy_rules)
    if selected:
        prototype,canopy_rule=selected
        support=embedded_woody_root(query,x,y,yaw,prototype['contact_anchors_local_m'],prototype['dimensions_m'][2],
            maximum_burial_m=prototype.get('maximum_burial_m',.6),
            maximum_burial_fraction=prototype.get('maximum_burial_fraction_of_height',.03))
        if support is None:
            rejected['canopy_root_footprint']=rejected.get('canopy_root_footprint',0)+1;continue
        supported,root_placement=support
        top=UsdGeom.Xform.Define(stage,f'/World/OutdoorTrees/Tree{index:05d}')
        top.AddTranslateOp().Set(Gf.Vec3d(x,y,supported));top.AddRotateZOp().Set(yaw)
        child=UsdGeom.Xform.Define(stage,top.GetPath().AppendChild('whole_tree')).GetPrim()
        child.GetReferences().AddReference(prototype['scene_asset'],Sdf.Path(prototype['usd_prim']))
        child.SetInstanceable(True)
        trees.append(dict(tree,position_world_xyz=[x,y,supported],variant=prototype['object_name'],
            dimensions_m=prototype['dimensions_m'],yaw_degrees=yaw,
            usd_path=str(top.GetPath()),
            canopy_recipe=canopy_rule,asset_id=prototype['asset_id'],source_biome=biome,
            source_species_interpretation='Recipe-selected original whole tree at source tree site; functional group, not exact botanical taxonomy',
            root_placement=root_placement))
        continue
    names={'oak':('oak_3729','oak_3730'),'birch':('white_birch_4729','white_birch_4730')}.get(tree['species'])
    if names is None:rejected[tree['species']]=rejected.get(tree['species'],0)+1;continue
    name=names[int(random_cells(int(tree['source_x']),int(tree['source_z']),81)>.5)]
    top=UsdGeom.Xform.Define(stage,f'/World/OutdoorTrees/Tree{index:05d}')
    top.AddTranslateOp().Set(Gf.Vec3d(x,y,z))
    top.AddRotateZOp().Set(yaw)
    for part in ('trunk','leaves'):
        child=UsdGeom.Xform.Define(stage,top.GetPath().AppendChild(part)).GetPrim()
        child.GetReferences().AddInternalReference(by_object[name+'_'+part]);child.SetInstanceable(True)
    trees.append(dict(tree,position_world_xyz=[x,y,z],variant=name,usd_path=str(top.GetPath())))
# Measure the prepared crowns, including larger replacements of voxel trees.
# Bounding extents drive only a recorded exposure estimate, not appearance gates.
crown_bounds=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render'],useExtentsHint=True)
for tree in trees:
    bound=crown_bounds.ComputeWorldBound(stage.GetPrimAtPath(tree['usd_path'])).ComputeAlignedRange()
    if bound.IsEmpty():raise ValueError('Prepared tree has no measurable native extent')
    tree.update(bounds_world_min=list(bound.GetMin()),bounds_world_max=list(bound.GetMax()))
if request.get('scenery_libraries'):
    from scipy.spatial.transform import Rotation
    from isaacmin.assembly.surface_scenery import scatter_rocks
    scenery=copy_canopy_libraries(root,scene,request['scenery_libraries'],namespace='scenery_library')
    rock_record=scatter_rocks(terrain,request['objects'],scenery,route=request.get('route'))
    for i,placement in enumerate(rock_record['placements']):
        top=UsdGeom.Xform.Define(stage,f'/World/OutdoorScenery/Rock{i:06d}')
        top.AddTranslateOp().Set(Gf.Vec3d(*placement['position_world_xyz']))
        quat=Rotation.from_matrix(placement['rotation_matrix']).as_quat()
        top.AddOrientOp().Set(Gf.Quatf(float(quat[3]),Gf.Vec3f(*quat[:3])))
        child=UsdGeom.Xform.Define(stage,top.GetPath().AppendChild('original')).GetPrim()
        child.GetReferences().AddReference(placement['scene_asset'],Sdf.Path(placement['usd_prim']))
        child.SetInstanceable(True)
    atomic_json(out/'scenery.json',rock_record)
shrub_count=0
if request.get('shrub_libraries'):
    from isaacmin.assembly.surface_shrubs import scatter_shrubs
    shrubs=copy_canopy_libraries(root,scene,request['shrub_libraries'],namespace='shrub_library')
    shrub_record=scatter_shrubs(terrain,shrubs,objects=request['objects'],route=request.get('route'))
    for i,placement in enumerate(shrub_record['placements']):
        top=UsdGeom.Xform.Define(stage,f'/World/OutdoorShrubs/Shrub{i:06d}')
        top.AddTranslateOp().Set(Gf.Vec3d(*placement['position_world_xyz']))
        top.AddRotateZOp().Set(placement['yaw_degrees'])
        child=UsdGeom.Xform.Define(stage,top.GetPath().AppendChild('original')).GetPrim()
        child.GetReferences().AddReference(placement['scene_asset'],Sdf.Path(placement['usd_prim']))
        child.SetInstanceable(True)
    shrub_count=len(shrub_record['placements'])
    atomic_json(out/'shrubs.json',shrub_record)
litter_count=0
if request.get('litter_libraries'):
    from isaacmin.assembly.surface_litter import scatter_litter
    litter=copy_canopy_libraries(root,scene,request['litter_libraries'],namespace='litter_library')
    litter_record=scatter_litter(terrain,request['objects'],litter,out/'litter',route=request.get('route'))
    litter_count=author_surface_instances(stage,'/World/OutdoorLitter',out/'litter',litter_record,{},litter=True)
# Preserve source water levels and fit the shoreline to the final actual ground.
from isaacmin.assembly.surface_water import build_surface_water
water_material=UsdShade.Material.Define(stage,'/IsaacMinMaterials/OutdoorWater')
shader=UsdShade.Shader.Define(stage,'/IsaacMinMaterials/OutdoorWater/Shader')
shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
water_module=scene/'water_material/IsaacMinClearWater.mdl';water_module.parent.mkdir(exist_ok=True)
shutil.copyfile(root/'recipes/materials/clear_water.mdl',water_module)
shader.SetSourceAsset(Sdf.AssetPath('./water_material/IsaacMinClearWater.mdl'),'mdl');shader.SetSourceAssetSubIdentifier('IsaacMinClearWater','mdl')
shader.CreateOutput('out',Sdf.ValueTypeNames.Token)
water_material.CreateSurfaceOutput('mdl').ConnectToSource(shader.ConnectableAPI(),'out')
def author_water(wp,wf,index):
    water=UsdGeom.Mesh.Define(stage,'/World/Outdoor/SurfaceWater'+('_'+str(index) if index else ''))
    water.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(wp))
    water.CreateFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(wf),3,np.int32)))
    water.CreateFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(wf.ravel()))
    water.CreateSubdivisionSchemeAttr().Set('none')
    UsdShade.MaterialBindingAPI.Apply(water.GetPrim()).Bind(water_material)
_,_,water_record=build_surface_water(terrain,sink=author_water)
atomic_json(out/'water.json',water_record)
atomic_json(out/'trees.json',dict(status='source_positioned_tree_candidates',trees=trees,rejected_species=rejected,
    root_footprint_qualification='not_run; final ground centre support used',qualification='not_run'))
request_capture=capture_configuration(root,request['capture_template']);poses=[]
if 'poses' not in request:
    from isaacmin.outdoor_pipeline import outdoor_camera_poses
    request['poses']=outdoor_camera_poses(terrain,request['objects'])
for original in request['poses']:
    pose=dict(original);p=np.asarray(pose['position'],float);look=np.asarray(pose['look_at'],float)
    dz=float(query.heights(p[0],p[1]))+.6-p[2];p[2]+=dz;look[2]+=dz
    pose.update(position=p.tolist(),look_at=look.tolist(),surface_scope='outdoor',scout_height_m=.6)
    from isaacmin.assembly.outdoor_lighting import canopy_camera_response,daylight_camera_response
    pose=canopy_camera_response(pose,trees)
    pose=daylight_camera_response(pose,request_capture)
    pose.pop('visual_ground_support',None);poses.append(pose)
request_capture.update(scene=str(scene/'world.usda'),output=str(out/'preview'),poses=poses,
    contact_probes=[],scope='actual_outdoor_visual_iteration_not_qualification',
    native_denoising=request.get('native_denoising',False),preserve_authored_scene=True)
lighting=author_outdoor_lighting(stage,scene,request_capture,poses)
overrides.Save();stage.GetRootLayer().Save()
request_capture['hdri']=str(scene/lighting['hdri'])
atomic_json(out/'preview_request.json',request_capture)
atomic_json(scene/'native_dependency_closure.json',_closure(scene/'world.usda'))
atomic_json(out/'assembly.json',dict(status='native_outdoor_scene_ready',at_utc=utc_now(),
    surface=read_json(terrain/'terrain.json'),ground_cover_instances=ground_cover_count,trees=len(trees),
    original_shrub_instances=shrub_count,
    original_litter_instances=litter_count,
    source_water_columns=water_record['source_water_columns'],structures='excluded_by_user',underground='excluded_by_user',
    native_Blender_geometry_bytes_verified=True,qualification='not_run',collision='next_stage_exact_final_triangles',
    asset_library=str(source),authored_lighting=lighting,producer_sha256=sha256_file(Path(__file__))))
print({'status':'native_outdoor_scene_ready','ground_cover':ground_cover_count,'trees':len(trees)})
