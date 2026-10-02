"""Compose native exported ground without duplicating shared fine geometry."""
from pathlib import Path
import os,shutil
import numpy as np
from pxr import UsdGeom,UsdShade,Sdf,Vt
from isaacmin.io import read_json,sha256_file


def attach_surface_ground(stage,terrain,scene,blender_export):
    terrain,scene,blender_export=map(Path,(terrain,scene,blender_export))
    partition_path=blender_export/'partitions.json'
    if partition_path.is_file():
        record=read_json(partition_path);active=set();count=None
        directory=scene/'terrain';directory.mkdir()
        src=blender_export/'outdoor_terrain.usda';os.link(src,directory/src.name)
        for part in record['parts']:
            source=blender_export/part['asset'];target=directory/part['asset'];target.parent.mkdir(parents=True)
            if sha256_file(source)!=part['usd_sha256']:raise ValueError('Native terrain partition changed')
            os.link(source,target);active.update(part['active_material_indices'])
            if count is not None and count!=part['material_count']:raise ValueError('Material families differ between parts')
            count=part['material_count']
        stage.GetRootLayer().subLayerPaths.insert(1,'terrain/outdoor_terrain.usda')
        for part in record['parts']:
            prim=stage.GetPrimAtPath(part['mesh_path'])
            if not prim or not prim.IsA(UsdGeom.Mesh):raise ValueError('Native terrain partition missing')
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material(stage.GetPrimAtPath('/IsaacMinMaterials/SharedOriginalPBR')))
            prim.CreateAttribute('isaacmin:surfaceScope',Sdf.ValueTypeNames.String).Set('outdoor_surface_navigation')
        return count,sorted(active)

    src=Path(blender_export)/'outdoor_terrain.usdc'
    shutil.copyfile(src,scene/src.name)
    stage.GetRootLayer().subLayerPaths.insert(1,src.name)
    ground=[p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh) and 'Terrain_FinalGround_Surface' in str(p.GetPath())]
    if len(ground)!=1:raise RuntimeError('Expected one new outdoor ground mesh')
    ground=ground[0];mesh=UsdGeom.Mesh(ground)
    points=np.load(terrain/'vertices.npy',mmap_mode='r');faces=np.load(terrain/'triangles.npy',mmap_mode='r')
    if not np.array_equal(np.asarray(mesh.GetPointsAttr().Get()),points):raise RuntimeError('Blender export changed final positions')
    if not np.array_equal(np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3),faces):raise RuntimeError('Blender export changed triangles')
    mesh.CreateSubdivisionSchemeAttr().Set('none')
    weights=np.load(terrain/'weights.npy',mmap_mode='r');pv=UsdGeom.PrimvarsAPI(ground)
    for name,file in [('IsaacMinSurfaceRestPosition','rest_positions.npy'),('IsaacMinSurfaceRestNormal','rest_normals.npy')]:
        pv.CreatePrimvar(name,Sdf.ValueTypeNames.Float3Array,UsdGeom.Tokens.vertex).Set(Vt.Vec3fArray.FromNumpy(np.load(terrain/file,mmap_mode='r')))
    pv.CreatePrimvar('IsaacMinMaterialWeights012',Sdf.ValueTypeNames.Float3Array,UsdGeom.Tokens.vertex).Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(weights[:,:3])))
    pv.CreatePrimvar('IsaacMinMaterialWeight3',Sdf.ValueTypeNames.FloatArray,UsdGeom.Tokens.vertex).Set(Vt.FloatArray.FromNumpy(np.ascontiguousarray(weights[:,3])))
    material_count=weights.shape[1]
    active_material_indices=np.flatnonzero(np.any(weights!=0,axis=0)).tolist()
    if material_count!=4:
        for index in range(material_count):
            pv.CreatePrimvar('IsaacMinSurfaceWeight'+str(index),Sdf.ValueTypeNames.FloatArray,UsdGeom.Tokens.vertex).Set(Vt.FloatArray.FromNumpy(np.ascontiguousarray(weights[:,index])))
    UsdShade.MaterialBindingAPI.Apply(ground).Bind(UsdShade.Material(stage.GetPrimAtPath('/IsaacMinMaterials/SharedOriginalPBR')))
    ground.CreateAttribute('isaacmin:surfaceScope',Sdf.ValueTypeNames.String).Set('outdoor_surface_navigation')
    ground.CreateAttribute('isaacmin:finalGroundSha256',Sdf.ValueTypeNames.String).Set(sha256_file(terrain/'vertices.npy'))
    del points,faces,weights
    return material_count,active_material_indices
