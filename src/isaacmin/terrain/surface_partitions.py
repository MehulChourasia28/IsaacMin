"""Storage partitions of a single authoritative fine surface; no reconstruction."""
from pathlib import Path
import math
import numpy as np
from isaacmin.io import read_json


def grid_triangles(nz,nx):
    # Deliberately NumPy-only: callable inside the isolated Blender Python.
    a=(np.arange(nz-1,dtype=np.int32)[:,None]*nx+np.arange(nx-1,dtype=np.int32)).ravel()
    faces=np.empty((2*len(a),3),np.int32)
    faces[0::2]=np.column_stack((a,a+nx,a+1));faces[1::2]=np.column_stack((a+1,a+nx,a+nx+1))
    return faces


def partition_windows(shape, maximum_cells=4096):
    nz,nx=shape
    return [dict(name=f'Tile_{z:05d}_{x:05d}',
        core=[z,min(z+maximum_cells,nz-1),x,min(x+maximum_cells,nx-1)],
        halo=[max(0,z-1),min(z+maximum_cells+1,nz-1),max(0,x-1),min(x+maximum_cells+1,nx-1)])
        for z in range(0,nz-1,maximum_cells) for x in range(0,nx-1,maximum_cells)]


def source_windows(bounds, maximum_m=128):
    xmin,zmin,xmax,zmax=bounds
    return [[x,z,min(x+maximum_m,xmax),min(z+maximum_m,zmax)]
        for z in np.arange(zmin,zmax,maximum_m) for x in np.arange(xmin,xmax,maximum_m)]


def source_cell_grid(bounds,full_bounds,cell,*,negative_z=False):
    """Assign every global seed cell to exactly one partition, without seams."""
    xmin,zmin,xmax,zmax=bounds;fxmin,fzmin,fxmax,fzmax=full_bounds
    if negative_z:zmin,zmax,fzmin,fzmax=-zmax,-zmin,-fzmax,-fzmin
    x0=math.floor(xmin/cell) if xmin==fxmin else math.ceil(xmin/cell)
    z0=math.floor(zmin/cell) if zmin==fzmin else math.ceil(zmin/cell)
    return np.meshgrid(np.arange(x0,math.ceil(xmax/cell)),np.arange(z0,math.ceil(zmax/cell)))


def window_array(terrain, filename, bounds):
    terrain=Path(terrain);shape=read_json(terrain/'terrain.json')['grid_shape']
    data=np.load(terrain/filename,mmap_mode='r');z0,z1,x0,x1=bounds
    trailing=data.shape[1:] if data.ndim>1 else ()
    return np.ascontiguousarray(data.reshape(*shape,*trailing)[z0:z1+1,x0:x1+1].reshape(-1,*trailing))


def native_partition(stage, terrain, partition):
    """Trim a native Blender halo export, preserving its exact smooth normals.

    The extra ring provides every incident face at interior tile boundaries.
    Only duplicate halo faces are removed; core positions/triangles are verified
    against the shared arrays, and corner normals are compacted only if exact.
    """
    from pxr import UsdGeom,Sdf,Vt
    ground=[p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh)]
    if len(ground)!=1:raise ValueError('Expected one native Blender terrain part')
    mesh=UsdGeom.Mesh(ground[0]);halo=partition['halo'];core=partition['core']
    hz0,hz1,hx0,hx1=halo;z0,z1,x0,x1=core;hz,hx=hz1-hz0+1,hx1-hx0+1
    expected=window_array(terrain,'vertices.npy',halo)
    if not np.array_equal(np.asarray(mesh.GetPointsAttr().Get()),expected):
        raise ValueError('Native Blender changed halo positions')
    indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3)
    if not np.array_equal(indices,grid_triangles(hz,hx)):
        raise ValueError('Native Blender changed halo triangles')
    cells=(np.arange(z0-hz0,z1-hz0)[:,None]*(hx-1)+np.arange(x0-hx0,x1-hx0)).ravel()
    selected=np.column_stack((cells*2,cells*2+1)).ravel()
    normal=np.asarray(mesh.GetNormalsAttr().Get())
    interpolation=mesh.GetNormalsInterpolation()
    if interpolation==UsdGeom.Tokens.faceVarying:
        corner=normal.reshape(-1,3,3)[selected].reshape(-1,3)
    elif interpolation==UsdGeom.Tokens.vertex:
        corner=normal[indices[selected].ravel()]
    else:raise ValueError('Unsupported native Blender normal interpolation')
    points=window_array(terrain,'vertices.npy',core);faces=grid_triangles(z1-z0+1,x1-x0+1)
    normals=np.empty_like(points);normals[faces.ravel()]=corner
    for begin in range(0,len(faces),200000):
        if not np.array_equal(normals[faces[begin:begin+200000].ravel()],corner[begin*3:(begin+200000)*3]):
            raise ValueError('Corner normals are not losslessly representable per vertex')
    mesh.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(points))
    mesh.CreateFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(faces),3,np.int32)))
    mesh.CreateFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(faces.ravel()))
    mesh.CreateNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normals));mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
    mesh.CreateExtentAttr().Set(Vt.Vec3fArray.FromNumpy(np.array([points.min(axis=0),points.max(axis=0)],np.float32)))
    mesh.CreateSubdivisionSchemeAttr().Set('none')
    pv=UsdGeom.PrimvarsAPI(mesh)
    for name,file in [('IsaacMinSurfaceRestPosition','rest_positions.npy'),('IsaacMinSurfaceRestNormal','rest_normals.npy')]:
        value=window_array(terrain,file,core)
        pv.CreatePrimvar(name,Sdf.ValueTypeNames.Float3Array,UsdGeom.Tokens.vertex).Set(Vt.Vec3fArray.FromNumpy(value))
    weights=window_array(terrain,'weights.npy',core)
    active=np.flatnonzero(np.any(weights!=0,axis=0)).tolist()
    if weights.shape[1]==4:
        pv.CreatePrimvar('IsaacMinMaterialWeights012',Sdf.ValueTypeNames.Float3Array,UsdGeom.Tokens.vertex).Set(Vt.Vec3fArray.FromNumpy(np.ascontiguousarray(weights[:,:3])))
        pv.CreatePrimvar('IsaacMinMaterialWeight3',Sdf.ValueTypeNames.FloatArray,UsdGeom.Tokens.vertex).Set(Vt.FloatArray.FromNumpy(np.ascontiguousarray(weights[:,3])))
    else:
        for index in active:
            pv.CreatePrimvar('IsaacMinSurfaceWeight'+str(index),Sdf.ValueTypeNames.FloatArray,UsdGeom.Tokens.vertex).Set(Vt.FloatArray.FromNumpy(np.ascontiguousarray(weights[:,index])))
    return dict(partition,mesh_path=str(mesh.GetPath()),vertices=len(points),triangles=len(faces),
        active_material_indices=active,material_count=weights.shape[1],
        position_and_triangle_comparison='exact shared-array equality',
        normal_compaction='every native face-corner normal equals indexed vertex normal exactly',
        halo_policy='one complete incident-face ring; shared geometry before partitioning',
        quality_reduction=False)
