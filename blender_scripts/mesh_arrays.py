"""Bulk metric geometry operations with frozen source normals for displacement."""
import hashlib, pathlib
import numpy as np

def values(collection,key,width,dtype=np.float32):
    array=np.empty(len(collection)*width,dtype=dtype);collection.foreach_get(key,array)
    return array.reshape(len(collection),width) if width>1 else array

def macro_delta(mesh,base,delta,protect,minimum,origin):
    co=values(mesh.vertices,'co',3);ox,oy,oz=origin;mx,mz=minimum
    for start in range(0,len(co),1_000_000):
        part=co[start:start+1_000_000];p=part.astype(np.float64)
        ix=np.floor(p[:,0]+ox-mx).astype(np.int64);iz=np.floor(-p[:,1]+oz-mz).astype(np.int64)
        selected=np.flatnonzero((ix>=0)&(iz>=0)&(ix<base.shape[1])&(iz<base.shape[0]))
        selected=selected[~protect[iz[selected],ix[selected]]]
        distance=abs(p[selected,2]+oy-base[iz[selected],ix[selected]])
        selected=selected[distance<1.5];distance=distance[distance<1.5]
        part[selected,2]=(p[selected,2]+delta[iz[selected],ix[selected]]*np.maximum(0,1-distance/1.5)).astype(np.float32)
    mesh.vertices.foreach_set('co',co.ravel());mesh.update()

def physical_uvs(mesh,mats,default_repeat):
    co=values(mesh.vertices,'co',3)
    normals=values(mesh.polygons,'normal',3);axis=np.argmax(abs(normals),axis=1);del normals
    material=values(mesh.polygons,'material_index',1,np.int32)
    counts=values(mesh.polygons,'loop_total',1,np.int32)
    loop_axis=np.repeat(axis,counts);repeat=np.asarray([m.get('repeat_m',default_repeat) for m in mats],float)
    if np.any((repeat<.1)|(repeat>20)):raise RuntimeError('Invalid per-material physical scale')
    loop_repeat=np.repeat(repeat[material],counts)
    indices=values(mesh.loops,'vertex_index',1,np.int32)
    uv_values=np.empty((len(indices),2),np.float32)
    for start in range(0,len(indices),1_000_000):
        end=min(start+1_000_000,len(indices));p=co[indices[start:end]].astype(np.float64)
        a=loop_axis[start:end];r=loop_repeat[start:end]
        uv_values[start:end,0]=np.where(a==0,p[:,1],p[:,0])/r
        uv_values[start:end,1]=np.where(a==2,p[:,1],p[:,2])/r
    layer=mesh.uv_layers.new(name='PhysicalUV');layer.data.foreach_set('uv',uv_values.ravel())
    mesh.polygons.foreach_set('use_smooth',np.ones(len(mesh.polygons),dtype=bool))

def soil_displacement(mesh,mats,eligible,amplitude,default_repeat,*,delta_context=None):
    import bpy
    co=values(mesh.vertices,'co',3);normal=values(mesh.vertices,'normal',3)
    ids=values(mesh.loops,'vertex_index',1,np.int32)
    per_face=values(mesh.polygons,'material_index',1,np.int32)
    counts=values(mesh.polygons,'loop_total',1,np.int32)
    per_loop=np.repeat(per_face,counts)
    lower=np.full(len(co),np.iinfo(np.int32).max,np.int32);upper=np.full(len(co),-1,np.int32)
    np.minimum.at(lower,ids,per_loop);np.maximum.at(upper,ids,per_loop)
    del ids,per_face,counts,per_loop
    results=[]
    for material_index in sorted(eligible):
        spec=mats[material_index]
        if not spec.get('height'):raise RuntimeError('Eligible soil material has no acquired displacement map')
        image=bpy.data.images.load(str(pathlib.Path(spec['height']).resolve()),check_existing=True)
        image.colorspace_settings.name='Non-Color'
        pixels=np.empty(image.size[0]*image.size[1]*4,np.float32);image.pixels.foreach_get(pixels)
        pixels=pixels.reshape(image.size[1],image.size[0],4)[:,:,0]
        if not np.isfinite(pixels).all() or np.ptp(pixels)<1e-6:raise RuntimeError('Displacement texture is empty/nonfinite')
        indices=np.flatnonzero((lower==material_index)&(upper==material_index)&(normal[:,2]>.7))
        modified=0;largest=0.;repeat=float(spec.get('repeat_m',default_repeat))
        for start in range(0,len(indices),1_000_000):
            chosen=indices[start:start+1_000_000];p=co[chosen].astype(np.float64)
            if delta_context:
                base,protect,minimum,origin=delta_context;mx,mz=minimum;ox,oy,oz=origin
                ix=np.floor(p[:,0]+ox-mx).astype(np.int64);iz=np.floor(-p[:,1]+oz-mz).astype(np.int64)
                good=(ix>=0)&(iz>=0)&(ix<base.shape[1])&(iz<base.shape[0]);k=np.flatnonzero(good)
                k=k[~protect[iz[k],ix[k]]&(abs(p[k,2]+oy-base[iz[k],ix[k]])<=2)]
                chosen=chosen[k];p=p[k]
            if not len(chosen):continue
            f=(p[:,:2]/repeat)%1;f*=np.asarray([pixels.shape[1],pixels.shape[0]])
            lo=np.floor(f).astype(np.int64);a,b=(f-lo).T
            x0,y0=lo.T;x1=(x0+1)%pixels.shape[1];y1=(y0+1)%pixels.shape[0]
            value=(1-a)*(1-b)*pixels[y0,x0]+a*(1-b)*pixels[y0,x1]+(1-a)*b*pixels[y1,x0]+a*b*pixels[y1,x1]
            displacement=(value-.5)*amplitude
            co[chosen]=(p+normal[chosen].astype(np.float64)*displacement[:,None]).astype(np.float32)
            modified+=len(chosen);largest=max(largest,float(np.max(abs(displacement))))
        results.append({'material_index':material_index,'material':spec['name'],'geometry_vertices_displaced':modified,
            'peak_to_peak_parameter_m':amplitude,'max_measured_vertex_offset_m':largest,'physical_repeat_m':repeat,
            'source_height_texture':spec['height'],'source_height_sha256':hashlib.sha256(pathlib.Path(spec['height']).read_bytes()).hexdigest(),
            'normal_policy':'source_normals_frozen_before_displacement; independent_of_vertex_edit_order',
            'amplitude_provenance':'recorded_engineering_default_pending_reference_qualification'})
    mesh.vertices.foreach_set('co',co.ravel());mesh.update();return results
