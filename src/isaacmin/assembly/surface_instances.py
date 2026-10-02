"""Bounded storage of unchanged global vegetation placements."""
from pathlib import Path
import numpy as np
from pxr import UsdGeom,Sdf,Vt
from isaacmin.io import sha256_file


def author_surface_instances(stage,path,directory,record,prototypes,*,litter=False):
    directory=Path(directory);count=0
    parts=record.get('parts',[dict(path='.')])
    for part in parts:
        source=directory/part['path']/'placements.npz'
        if 'placements_sha256' in part and sha256_file(source)!=part['placements_sha256']:
            raise ValueError('Placement partition changed')
        with np.load(source) as data:
            instancer=UsdGeom.PointInstancer.Define(stage,path+('/'+part['path'] if 'parts' in record else ''))
            targets=[]
            for i,proto in enumerate(record['prototypes']):
                target=instancer.GetPath().AppendChild('Prototype'+str(i))
                prim=UsdGeom.Xform.Define(stage,target).GetPrim()
                if litter:prim.GetReferences().AddReference(proto['scene_asset'],Sdf.Path(proto['usd_prim']))
                else:prim.GetReferences().AddInternalReference(prototypes[(proto['blend_sha256'],proto['object_name'])])
                prim.SetInstanceable(True);targets.append(target)
            positions=data['positions'];size=len(positions)
            instancer.CreatePrototypesRel().SetTargets(targets)
            instancer.CreateProtoIndicesAttr().Set(Vt.IntArray.FromNumpy(data['prototype_indices']))
            instancer.CreatePositionsAttr().Set(Vt.Vec3fArray.FromNumpy(positions))
            if litter:quaternions=data['orientations_xyzw']
            else:
                instancer.CreateScalesAttr().Set(Vt.Vec3fArray.FromNumpy(data['scales']))
                quaternions=np.ascontiguousarray(data['orientations_wxyz'][:,[1,2,3,0]])
            instancer.CreateOrientationsAttr().Set(Vt.QuathArray.FromNumpy(quaternions))
            instancer.CreateIdsAttr().Set(Vt.Int64Array.FromNumpy(np.arange(count,count+size,dtype=np.int64)))
            count+=size
    if count!=record['instances']:raise ValueError('Native instance count differs from placement record')
    return count
