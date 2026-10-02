"""Create an immutable derived world with exactly planar water retriangulation."""
from pathlib import Path
import argparse,os,shutil
import numpy as np
from pxr import Usd,UsdGeom,UsdShade,Sdf,Vt
from isaacmin.io import read_json,atomic_json,sha256_file,utc_now
from isaacmin.assembly.usd_instances import _closure,PropertyFingerprint
from isaacmin.assembly.surface_water import build_surface_water

p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
source=a.build.resolve();out=a.output.resolve();base=read_json(source/'outdoor_build.json')
if base.get('status')!='geometry_world' or out.exists():raise ValueError('Use completed source and fresh output')
old=Path(base['scene']).parent;scene=out/'assembled/scene';scene.mkdir(parents=True)
closure=read_json(old/'native_dependency_closure.json')
for row in closure['files']:
    src=old/row['path'];dst=scene/row['path']
    if src.is_symlink() or sha256_file(src)!=row['sha256']:raise ValueError('Source scene dependency changed')
    if row['path']=='outdoor_overrides.usdc':continue
    dst.parent.mkdir(parents=True,exist_ok=True)
    if src.suffix in ('.usda','.json'):shutil.copyfile(src,dst)
    else:os.link(src,dst)
original=Sdf.Layer.FindOrOpen(str(old/'outdoor_overrides.usdc'))
replacement=Sdf.Layer.CreateNew(str(scene/'outdoor_overrides.usdc'))
for prim in original.rootPrims:
    if not Sdf.CopySpec(original,prim.path,replacement,prim.path):raise ValueError('Native layer copy failed')
stage=Usd.Stage.Open(str(scene/'world.usda'));stage.SetEditTarget(replacement)
water_paths=[prim.GetPath() for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh) and str(prim.GetPath()).startswith('/World/Outdoor/SurfaceWater')]
for path in water_paths:
    if not stage.RemovePrim(path):raise ValueError('Could not replace source water ownership')
material=UsdShade.Material(stage.GetPrimAtPath('/IsaacMinMaterials/OutdoorWater'))
if not material:raise ValueError('Original water material missing')
progress=out/'compaction_progress.json'
def sink(points,faces,index):
    water=UsdGeom.Mesh.Define(stage,'/World/Outdoor/SurfaceWater'+('_'+str(index) if index else ''))
    water.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
    water.CreateFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(faces),3,np.int32)))
    water.CreateFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(faces.ravel()))
    water.CreateSubdivisionSchemeAttr().Set('none')
    UsdShade.MaterialBindingAPI.Apply(water.GetPrim()).Bind(material)
    atomic_json(progress,dict(at_utc=utc_now(),completed_water_parts=index+1))
_,_,water=build_surface_water(source/'terrain',sink=sink)
before=read_json(source/'assembled/water.json')
old_levels={r['source_level_m']:r for r in before['levels']};new_levels={r['source_level_m']:r for r in water['levels']}
if old_levels.keys()!=new_levels.keys() or before['source_water_columns']!=water['source_water_columns']:
    raise ValueError('Source water levels/columns changed')
areas=[]
for level,row in new_levels.items():
    delta=row['surface_area_m2']-old_levels[level]['surface_area_m2']
    if abs(delta)>1e-6:raise ValueError('Water area differs from original fine triangulation')
    areas.append(dict(level_m=level,area_difference_m2=delta,source_columns=row['source_columns']))
# Export fresh bytes, avoiding unreachable old water arrays in append-only crate storage.
temporary=scene/'outdoor_overrides_compacted.usdc';replacement.Export(str(temporary))
del stage,replacement
os.replace(temporary,scene/'outdoor_overrides.usdc')
replacement=Sdf.Layer.FindOrOpen(str(scene/'outdoor_overrides.usdc'));replacement.Reload()
def prims(layer):
    todo=list(layer.rootPrims)
    while todo:
        prim=todo.pop()
        if str(prim.path).startswith('/World/Outdoor/SurfaceWater'):continue
        yield prim;todo.extend(prim.nameChildren.values())
# Compare every authored non-water property, including full instance arrays and
# metadata, after native reopen. Inherited terrain/collision bytes were hardlinked
# and hash-verified independently above; their storage and shape do not change.
fingerprints=PropertyFingerprint();checked=0
for prim in prims(original):
    other=replacement.GetPrimAtPath(prim.path)
    if other is None or set(prim.ListInfoKeys())!=set(other.ListInfoKeys()):raise ValueError('Non-water prim changed')
    for key in prim.ListInfoKeys():
        if fingerprints.value(prim.GetInfo(key),prim.path)!=fingerprints.value(other.GetInfo(key),prim.path):
            raise ValueError('Non-water metadata changed: '+str(prim.path)+':'+key)
    if set(prim.properties.keys())!=set(other.properties.keys()):raise ValueError('Non-water property set changed')
    for name,prop in prim.properties.items():
        target=other.properties[name]
        if set(prop.ListInfoKeys())!=set(target.ListInfoKeys()):raise ValueError('Property metadata fields changed')
        for key in prop.ListInfoKeys():
            if fingerprints.value(prop.GetInfo(key),prim.path)!=fingerprints.value(target.GetInfo(key),prim.path):
                raise ValueError('Non-water property changed: '+str(prop.path)+':'+key)
        checked+=1
if {str(p.path) for p in prims(original)}!={str(p.path) for p in prims(replacement)}:raise ValueError('Non-water prim set changed')
del fingerprints
atomic_json(out/'assembled/water.json',water)
atomic_json(scene/'native_dependency_closure.json',_closure(scene/'world.usda'))
# Reuse source/terrain metadata explicitly, without pretending it was rebuilt.
for folder in ('terrain','producer_sources'):
    for src in (source/folder).rglob('*'):
        if src.is_file():
            dst=out/folder/src.relative_to(source/folder);dst.parent.mkdir(parents=True,exist_ok=True);os.link(src,dst)
for name in ('route_candidate.json','biome_coverage.json'):
    if (source/name).is_file():shutil.copyfile(source/name,out/name)
for name in ('assembly.json','trees.json','scenery.json','shrubs.json','exact_collision.json'):
    shutil.copyfile(source/'assembled'/name,out/'assembled'/name)
request=read_json(source/'assembled/preview_request.json')
request.update(scene=str(scene/'world.usda'),hdri=str(scene/Path(request['hdri']).relative_to(old)),output=str(out/'assembled/preview'))
atomic_json(out/'assembled/preview_request.json',request)
proof=dict(at_utc=utc_now(),status='exact_planar_water_surface_retriangulated',source_build=str(source),
    source_closure_sha256=sha256_file(old/'native_dependency_closure.json'),
    water_triangles_before=before['triangles'],water_triangles_after=water['triangles'],
    unchanged_non_water_properties=checked,area_comparisons=areas,
    terrain_materials_vegetation_collision_changed=False,shoreline_clipping_changed=False,
    producer_sha256={str(p):sha256_file(p) for p in [Path(__file__),Path('src/isaacmin/assembly/planar_water.py'),Path('src/isaacmin/assembly/surface_water.py')]},
    native_appearance_comparison='not_run',qualification='not_qualified')
atomic_json(out/'planar_water_derivation.json',proof)
base.update(scene=str(scene/'world.usda'),derived_from=str(source),derivation=proof,qualification='not_run')
atomic_json(out/'outdoor_build.json',base)
print({k:proof[k] for k in ('status','water_triangles_before','water_triangles_after','unchanged_non_water_properties')})
