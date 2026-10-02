"""Read-only native USD sampling for the browser's labelled construction diagram.

Points come from actual composed asset meshes and instance placements. This never
exports substitute assets into the Isaac world or evaluates its visual quality.
"""
import argparse
from pathlib import Path
import numpy as np
from pxr import Usd, UsdGeom
from isaacmin.io import read_json, atomic_json, sha256_file, utc_now

parser=argparse.ArgumentParser();parser.add_argument('--request',required=True)
request=read_json(Path(parser.parse_args().request));scene=Path(request['scene']);out=Path(request['output'])
out.mkdir(parents=True,exist_ok=True);source_hash=sha256_file(scene)
stage=Usd.Stage.Open(str(scene));cache=UsdGeom.XformCache();samples={};parts=[];groups=[]

def source_points(mesh,limit):
    prim=mesh.GetPrim();original=prim.GetPrimInPrototype() if prim.IsInstanceProxy() else prim
    key=(str(original.GetPath()),limit)
    if key not in samples:
        value=np.asarray(mesh.GetPointsAttr().Get(),dtype=np.float32)
        indices=np.linspace(0,len(value)-1,min(limit,len(value)),dtype=int) if len(value) else []
        samples[key]=value[indices].copy()
    return samples[key]

def append(points,base,rank,color):
    if not len(points):return
    count=len(points);parts.append(np.column_stack((points,np.full(count,base),np.full(count,rank),
        np.broadcast_to(color,(count,3)))).astype('<f4'))

for label,path,budget,color in [
    ('Trees','/World/OutdoorTrees',150000,[.51,.77,.43]),
    ('Rocks','/World/OutdoorScenery',18000,[.73,.67,.55]),
    ('Understory','/World/OutdoorShrubs',26000,[.68,.85,.46])]:
    root=stage.GetPrimAtPath(path)
    children=list(root.GetChildren()) if root else []
    selected=children[::max(1,(len(children)+799)//800)]
    per=max(24,min(1800,budget//max(1,len(selected))))
    start=sum(len(p) for p in parts)
    for index,top in enumerate(selected):
        meshes=[UsdGeom.Mesh(p) for p in Usd.PrimRange(top,Usd.TraverseInstanceProxies()) if p.IsA(UsdGeom.Mesh)]
        base=float(cache.GetLocalToWorldTransform(top).ExtractTranslation()[2])
        for mesh in meshes:
            points=source_points(mesh,max(6,per//max(1,len(meshes))))
            matrix=np.asarray(cache.GetLocalToWorldTransform(mesh.GetPrim()))
            world=points@matrix[:3,:3]+matrix[3,:3]
            name=str(mesh.GetPath()).lower()
            tint=[.70,.53,.32] if label=='Trees' and any(s in name for s in ('trunk','bark','stem')) else color
            append(world,base,index/max(1,len(selected)),tint)
    groups.append(dict(name=label,start=start,count=sum(len(p) for p in parts)-start,
        actual_instances=len(children),represented_instances=len(selected),geometry='sampled actual composed USD mesh vertices'))

for label,path,limit,color in [('Ground cover','/World/OutdoorGroundCover',18000,[.68,.76,.33]),
                              ('Forest litter','/World/OutdoorLitter',8000,[.73,.55,.31])]:
    root=stage.GetPrimAtPath(path);start=sum(len(p) for p in parts);total=0;represented=0
    instancers=[UsdGeom.PointInstancer(p) for p in Usd.PrimRange(root) if p.IsA(UsdGeom.PointInstancer)] if root else []
    for instancer in instancers:
        positions=np.asarray(instancer.GetPositionsAttr().Get(),dtype=np.float32);total+=len(positions)
        select=np.linspace(0,len(positions)-1,min(max(1,limit//max(1,len(instancers))),len(positions)),dtype=int) if len(positions) else []
        matrix=np.asarray(cache.GetLocalToWorldTransform(instancer.GetPrim()))
        points=positions[select]@matrix[:3,:3]+matrix[3,:3];represented+=len(points)
        for rank,chunk in enumerate(np.array_split(points,32)):
            if len(chunk):append(chunk,float(np.min(chunk[:,2])),rank/32,color)
    groups.append(dict(name=label,start=start,count=sum(len(p) for p in parts)-start,
        actual_instances=total,represented_instances=represented,geometry='actual USD placement anchors, shown as points'))

if source_hash!=sha256_file(scene):raise RuntimeError('Scene changed during preview sampling; retry after its stage completes')
points=np.concatenate(parts) if parts else np.empty((0,8),dtype='<f4')
if not np.isfinite(points).all():raise ValueError('Nonfinite native preview geometry')
temporary=out/'assets.bin.partial';points.tofile(temporary);temporary.replace(out/'assets.bin')
atomic_json(out/'assets.json',dict(status='actual_native_geometry_sampled',at_utc=utc_now(),scene_sha256=source_hash,
    file='assets.bin',sha256=sha256_file(out/'assets.bin'),points=len(points),stride_floats=8,groups=groups,
    fields=['x','y','z','root_z','reveal_order','red','green','blue'],
    producer_sha256=sha256_file(Path(__file__)),
    scope='Construction visualization: sampled real geometry and placement anchors, symbolic colors. Not an Isaac render or substitute world asset.'))
print(dict(status='actual_native_geometry_sampled',points=len(points),groups=[(g['name'],g['actual_instances']) for g in groups]))
