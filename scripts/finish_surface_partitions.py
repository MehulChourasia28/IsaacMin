"""Compose verified terrain partitions and independently check shared edges."""
from pathlib import Path
import argparse
import numpy as np
from pxr import Usd,UsdGeom,Sdf
from isaacmin.io import read_json,atomic_json,sha256_file
from isaacmin.terrain.surface_partitions import partition_windows
p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);a=p.parse_args()
r=read_json(a.request);out=Path(r['output']);terrain=read_json(Path(r['terrain'])/'terrain.json')
parts=[];edges={};seams=[]
for expected in partition_windows(terrain['grid_shape']):
    path=out/expected['name'];record=read_json(path/'partition.json');asset=path/'verified_terrain.usdc'
    if any(record[k]!=expected[k] for k in ('name','core','halo')) or sha256_file(asset)!=record['usd_sha256']:
        raise ValueError('Terrain partition identity changed')
    stage=Usd.Stage.Open(str(asset));mesh=UsdGeom.Mesh(stage.GetPrimAtPath(record['mesh_path']))
    z0,z1,x0,x1=record['core'];shape=(z1-z0+1,x1-x0+1,3)
    points=np.asarray(mesh.GetPointsAttr().Get()).reshape(shape)
    normals=np.asarray(mesh.GetNormalsAttr().Get()).reshape(shape)
    for name,pv in [('rest_position',UsdGeom.PrimvarsAPI(mesh).GetPrimvar('IsaacMinSurfaceRestPosition')),
                    ('rest_normal',UsdGeom.PrimvarsAPI(mesh).GetPrimvar('IsaacMinSurfaceRestNormal')),
                    ('points',None),('normals',None)]:
        values=points if name=='points' else normals if name=='normals' else np.asarray(pv.Get()).reshape(shape)
        for key,edge in [((name,'z',z0,x0,x1),values[0]),((name,'z',z1,x0,x1),values[-1]),
                         ((name,'x',x0,z0,z1),values[:,0]),((name,'x',x1,z0,z1),values[:,-1])]:
            if key in edges:
                if not np.array_equal(edges[key],edge):raise ValueError('Native shared-edge mismatch: '+repr(key))
                seams.append(dict(field=name,axis=key[1],coordinate=key[2],equal=True));del edges[key]
            else:edges[key]=edge.copy()
    parts.append(dict(record,asset=str(asset.relative_to(out))))
    del stage,mesh,points,normals
if sum(p['triangles'] for p in parts)!=terrain['triangles']:raise ValueError('Triangle ownership gap/overlap')
layer=Sdf.Layer.CreateNew(str(out/'outdoor_terrain.usda'))
layer.subLayerPaths=[p['asset'] for p in parts];layer.Save()
atomic_json(out/'partitions.json',dict(status='native_partitions_verified',parts=parts,
    exact_shared_edge_comparisons=seams,unique_triangles=terrain['triangles'],
    reconstruction='single shared global source, drainage, erosion and fine geometry',
    normal_storage='lossless vertex indexing of native Blender face-corner normals',
    geometry_material_density_reduction=False,qualification='not_run'))
print({'native_parts':len(parts),'triangles':terrain['triangles'],'exact_shared_edges':len(seams)})
