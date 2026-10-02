"""Deterministic continuous three-patch sampling of acquired texture channels.

Shared CPU mapping for outdoor scan displacement. Target qualification is separate.
No image synthesis model, rescaling, recolouring or fake texture data is used.
"""
import numpy as np


def offsets(vertices):
    """Bounded integer hash: identical arithmetic is expressible in pinned MDL."""
    p=np.asarray(vertices,np.int64)%4093
    h=(p[...,0]*761+p[...,1]*569+121)%4093
    u=(h*h+1999*h+17)%4093
    v=(u*u+1553*u+109)%4093
    return np.stack(((u+.5)/4093,(v+.5)/4093),axis=-1).astype(np.float32)


def mapping(uv):
    uv=np.asarray(uv,np.float64)
    if uv.shape[-1]!=2 or not np.isfinite(uv).all():raise ValueError('Finite metric chart coordinates required')
    grid=np.stack((uv[...,0]-uv[...,1]/np.sqrt(3),uv[...,1]*2/np.sqrt(3)),axis=-1)
    cell=np.floor(grid).astype(np.int64);f=grid-cell;upper=f.sum(axis=-1)>1
    lo=np.stack((cell,cell+[1,0],cell+[0,1]),axis=-2)
    hi=np.stack((cell+[1,1],cell+[0,1],cell+[1,0]),axis=-2)
    vertices=np.where(upper[...,None,None],hi,lo)
    weights=np.where(upper[...,None],np.stack((f.sum(axis=-1)-1,1-f[...,0],1-f[...,1]),axis=-1),np.stack((1-f.sum(axis=-1),f[...,0],f[...,1]),axis=-1))
    weights=np.maximum(weights,0)**3
    weights/=weights.sum(axis=-1,keepdims=True)
    return uv[...,None,:]+offsets(vertices),weights,vertices


def bilinear(image,uv):
    image=np.asarray(image)
    if image.ndim not in (2,3) or not np.isfinite(image).all():raise ValueError('Finite scalar/vector source image required')
    uv=np.asarray(uv,np.float64);h,w=image.shape[:2];pixel=(uv%1)*[w,h]-.5
    lo=np.floor(pixel).astype(np.int64);f=pixel-lo
    x0=lo[...,0]%w;y0=lo[...,1]%h;x1=(x0+1)%w;y1=(y0+1)%h
    a,b=f[...,0],f[...,1]
    if image.ndim==3:a=a[...,None];b=b[...,None]
    return image[y0,x0]*(1-a)*(1-b)+image[y0,x1]*a*(1-b)+image[y1,x0]*(1-a)*b+image[y1,x1]*a*b


def sample(image,uv):
    coordinates,weights,_=mapping(uv);values=bilinear(image,coordinates)
    if np.asarray(image).ndim==3:weights=weights[...,None]
    return np.sum(values*weights,axis=-2 if np.asarray(image).ndim==3 else -1)


def displaced_height(points, normals, weights, materials, *, amplitude, material_order=None):
    """Original scan heights in the same three-patch charts as native MDL.

    Rest positions and projection normals are authored as USD primvars so
    displacement does not feed back into texture projection after export.
    """
    from pathlib import Path
    from PIL import Image
    from isaacmin.io import sha256_file
    from isaacmin.assembly.material_assignment import REQUIRED
    result = np.zeros(len(points),np.float32)
    proof=[]
    for material_index, name in enumerate(material_order or REQUIRED):
        material=next(m for m in materials if m['asset_id']==name)
        path=Path(material['height']); expected=material['original_channel_proof']['height']
        if sha256_file(path)!=expected['sha256']:
            raise ValueError('Original height channel changed')
        with path.open('rb') as stream:header=stream.read(33)
        bits=header[24]
        if bits not in (8,16):raise ValueError('Unsupported original height depth')
        with Image.open(path) as im:image=np.asarray(im,dtype=np.float32)[::-1]/((1<<bits)-1)
        if image.ndim==3:image=image[:,:,0]
        repeat=float(material['repeat_m'])
        selected=np.flatnonzero(weights[:,material_index]>0)
        for start in range(0,len(selected),100000):
            indices=selected[start:start+100000];p=points[indices];n=normals[indices]
            signs=np.where(n<0,-1.,1.);axis=np.abs(n)**4;axis/=axis.sum(axis=1,keepdims=True)
            charts=(np.column_stack((p[:,1]*signs[:,0],p[:,2])),
                    np.column_stack((-p[:,0]*signs[:,1],p[:,2])),
                    np.column_stack((p[:,0]*signs[:,2],p[:,1])))
            values=np.zeros(len(indices))
            for k,chart in enumerate(charts):values+=sample(image,chart/repeat)*axis[:,k]
            result[indices]+=(values-.5)*amplitude*weights[indices,material_index]
        proof.append(dict(asset_id=name,path=str(path),sha256=expected['sha256'],bits=bits,repeat_m=repeat))
    return result, dict(method='original_scan_three_patch_signed_triplanar',amplitude_m=amplitude,
        displacement_range_m=[float(result.min()),float(result.max())],originals=proof,
        rest_primvars=['IsaacMinSurfaceRestPosition','IsaacMinSurfaceRestNormal'],
        mesh_displacement='vertical at each final grid vertex; collision uses these displaced triangles',
        target_material_mapping_qualification='not_run',producer_sha256=sha256_file(Path(__file__)))
