"""Freeze reusable prototypes/materials without importing a previous map's world."""
from pathlib import Path
import argparse
import hashlib
import os
import shutil
import numpy as np
from pxr import Usd, UsdGeom, UsdUtils, Sdf
from isaacmin.io import read_json, atomic_json, sha256_file, utc_now
from isaacmin.assembly.usd_instances import _closure

parser=argparse.ArgumentParser(); parser.add_argument('--request', required=True)
r=read_json(Path(parser.parse_args().request))
source=Path(r['source']).resolve(); out=Path(r['output']).resolve()
out.mkdir(parents=True, exist_ok=False)
index=read_json(Path(r['prototype_index']))
mask=Usd.StagePopulationMask()
for path in ('/IsaacMinAssetPrototypes','/IsaacMinMaterials','/World/_materials'):
    mask.Add(Sdf.Path(path))
stage=Usd.Stage.OpenMasked(str(source/'world.usda'),mask)
stage.ExpandPopulationMask()
allowed=('/IsaacMinAssetPrototypes','/IsaacMinMaterials','/World/_materials')
for prim in stage.TraverseAll():
    path=str(prim.GetPath())
    if prim.IsA(UsdGeom.Mesh) and not path.startswith('/IsaacMinAssetPrototypes/'):
        raise ValueError('A source-world mesh entered the reusable asset mask: '+path)
    if prim.IsA(UsdGeom.Camera) or prim.IsA(UsdGeom.PointInstancer):
        raise ValueError('Map camera or population entered the asset library')
closed=read_json(source/'native_dependency_closure.json')
copied={}
for item in closed['files']:
    p=source/item['path']
    if p.suffix.lower() in ('.usd','.usda','.usdc'):continue
    if p.is_symlink() or not p.resolve().is_relative_to(source) or sha256_file(p)!=item['sha256']:
        raise ValueError('Original native dependency changed')
    q=out/item['path']; q.parent.mkdir(parents=True,exist_ok=True)
    if p.suffix in ('.mdl','.json'):shutil.copyfile(p,q)
    else:os.link(p,q)
    copied[str(p.resolve())]=q.relative_to(out).as_posix()
flat=stage.Flatten()
def relative(path):
    if not path:return path
    p=Path(path)
    resolved=str(p.resolve()) if p.is_absolute() else str((source/p).resolve())
    if resolved in copied:return './'+copied[resolved]
    if p.suffix in ('.usd','.usda','.usdc'):
        raise ValueError('Flattened library still references a map layer: '+path)
    if not p.is_absolute() and len(p.parts)==1:
        return path  # Runtime built-in MDL modules are checked by the closure resolver.
    raise ValueError('Asset dependency escaped the frozen library: '+path)
UsdUtils.ModifyAssetPaths(flat,relative)
flat.Export(str(out/'library.usdc'))
result=Usd.Stage.CreateNew(str(out/'world.usda'))
result.GetRootLayer().subLayerPaths=['library.usdc']
world=UsdGeom.Xform.Define(result,'/World').GetPrim()
result.SetDefaultPrim(world); UsdGeom.SetStageMetersPerUnit(result,1.)
UsdGeom.SetStageUpAxis(result,UsdGeom.Tokens.z)
world.CreateAttribute('isaacmin:assetIndex',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('./asset_index.json'))
proof=[]
for item in index:
    a=stage.GetPrimAtPath(item['prototype']); b=result.GetPrimAtPath(item['prototype'])
    if not a or not b:raise ValueError('Required reusable prototype absent')
    checks=[]
    for prim in Usd.PrimRange(a,Usd.PrimAllPrimsPredicate):
        if not prim.IsA(UsdGeom.Mesh):continue
        target=result.GetPrimAtPath(prim.GetPath())
        for name in ('points','faceVertexCounts','faceVertexIndices','normals'):
            left=prim.GetAttribute(name).Get();right=target.GetAttribute(name).Get()
            if left is None and right is None:continue
            left=np.asarray(left);right=np.asarray(right)
            if not np.array_equal(left,right):raise ValueError('Asset extraction changed native geometry')
            checks.append(dict(mesh=str(prim.GetPath()),attribute=name,sha256=hashlib.sha256(left.tobytes()).hexdigest()))
    if not checks:raise ValueError('Reusable prototype contains no verified geometry')
    proof.append(dict(prototype=item['prototype'],checks=checks))
atomic_json(out/'asset_index.json',dict(schema_version=1,prototypes=index,
    contents='source-independent native prototypes and material candidates; no terrain, cameras or population',
    qualification='not_inferred'))
result.GetRootLayer().Save()
atomic_json(out/'native_dependency_closure.json',_closure(out/'world.usda'))
atomic_json(out/'extraction.json',dict(at_utc=utc_now(),status='native_asset_library_extracted',
    original_library_scene=str(source/'world.usda'),prototype_index_sha256=sha256_file(Path(r['prototype_index'])),
    prototypes=len(index),original_geometry_equal=True,geometry_proof=proof,
    source_world_geometry_included=False,source_world_cameras_included=False,
    qualification='structural verification only; native scene replay next',producer_sha256=sha256_file(Path(__file__))))
print({'standalone_prototypes':len(index),'source_world_geometry_included':False},flush=True)
