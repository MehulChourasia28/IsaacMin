"""Portable transition textures from acquired PBR masters, independent of Cycles.

Imported inside native Blender. Geometry is read only; existing material slots and
PhysicalUV are the only authored changes. Bakes stream one bounded tile at a time.
"""
from pathlib import Path
from collections import defaultdict
import hashlib
import json
import math
import struct
import zlib
import numpy as np
from .material_assignment import material_blend_weights,REQUIRED

RECIPE='source_continuous_transition_pbr_v1'


def _digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def _png16(path,values,*,srgb=False):
    """Lossless PNG16 writer; input channels already use their declared encoding."""
    values=np.asarray(values)
    if values.ndim==2:values=values[:,:,None]
    h,w,c=values.shape
    if c not in (1,3):raise ValueError('PNG output needs one or three channels')
    def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    temporary=Path(str(path)+'.partial');compress=zlib.compressobj(6)
    with temporary.open('wb') as f:
        f.write(b'\x89PNG\r\n\x1a\n');f.write(chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,16,0 if c==1 else 2,0,0,0)))
        if srgb:f.write(chunk(b'sRGB',b'\0'))
        for row in values:
            data=compress.compress(b'\0'+np.rint(np.clip(row,0,1)*65535).astype('>u2').tobytes())
            if data:f.write(chunk(b'IDAT',data))
        f.write(chunk(b'IDAT',compress.flush()));f.write(chunk(b'IEND',b''))
    temporary.replace(path)


def _linear(value):return np.where(value<=.04045,value/12.92,((value+.055)/1.055)**2.4)
def _srgb(value):return np.where(value<=.0031308,value*12.92,1.055*np.maximum(value,0)**(1/2.4)-.055)


def _sample(image,coord,repeat):
    h,w=image.shape[:2];uv=coord/repeat
    fx=(uv[:,0]%1)*w-.5;fy=(uv[:,1]%1)*h-.5
    x=np.floor(fx).astype(int);y=np.floor(fy).astype(int);a=(fx-x)[:,None];b=(fy-y)[:,None]
    return image[y%h,x%w]*(1-a)*(1-b)+image[y%h,(x+1)%w]*a*(1-b)+image[(y+1)%h,x%w]*(1-a)*b+image[(y+1)%h,(x+1)%w]*a*b


def _load_masters(materials):
    import bpy
    maps={};proof=[]
    for name in REQUIRED:
        spec=next(m for m in materials if m.get('asset_id',m.get('name'))==name)
        role_maps={}
        for role in ('base_color','roughness','normal'):
            path=Path(spec[role]).resolve();image=bpy.data.images.load(str(path),check_existing=False)
            # Non-Color requests numerical source values, avoiding implicit sRGB
            # conversion in float buffers. We decode base color explicitly once.
            image.colorspace_settings.name='Non-Color'
            pixels=np.empty(len(image.pixels),np.float32);image.pixels.foreach_get(pixels)
            pixels=pixels.reshape(image.size[1],image.size[0],image.channels)[:,:,:3]
            if role=='base_color':pixels=_linear(pixels)
            if not np.isfinite(pixels).all():raise RuntimeError('Native material master contains nonfinite pixels')
            role_maps[role]=pixels.copy();proof.append({'asset_id':name,'role':role,'path':str(path),'sha256':_digest(path),'repeat_m':spec['repeat_m'],'loaded_shape':list(pixels.shape)})
            bpy.data.images.remove(image)
        maps[name]=(spec,role_maps)
    return maps,proof


def _mix(maps,coordinates,weights):
    color=np.zeros((len(coordinates),3),np.float32);rough=np.zeros((len(coordinates),1),np.float32);normal=np.zeros((len(coordinates),3),np.float32)
    for index,name in enumerate(REQUIRED):
        chosen=weights[:,index]>1e-7
        if not chosen.any():continue
        spec,images=maps[name];coord=coordinates[chosen];factor=weights[chosen,index,None]
        color[chosen]+=_sample(images['base_color'],coord,spec['repeat_m'])*factor
        rough[chosen]+=_sample(images['roughness'],coord,spec['repeat_m'])[:,:1]*factor
        normal[chosen]+=(_sample(images['normal'],coord,spec['repeat_m'])*2-1)*factor
    length=np.linalg.norm(normal,axis=1);normal/=np.maximum(length,1e-8)[:,None]
    return _srgb(color),rough,normal*.5+.5


def apply_transition_bakes(terrain,materials,source_surface_path,minecraft_origin_xyz,output_directory,*,
                           exterior_delta_spec=None,structural_source_coordinates=None,material_factory,
                           texels_per_m=1024,tile_size_m=4,gutter_pixels=16):
    """Bake continuous weights on the actual mesh with explicit per-chart normals.

    Within a chart every master uses the SAME axis/order/handedness as PhysicalUV
    and positive physical repeat. Their tangent normals consequently share one
    basis before weighted vector normalization. No normals from different charts
    are mixed. Existing dominant-axis projection boundaries are reported, remain
    unqualified, and require native seam/UV review. No geometry is moved.
    """
    import bpy
    if texels_per_m<1024 or tile_size_m<=0 or tile_size_m>8:raise ValueError('Production transitions require at least1024texels/m and bounded<=8m tiles')
    output=Path(output_directory).resolve();output.mkdir(parents=True,exist_ok=True)
    report_path=output/'transition_bakes.json'
    if report_path.exists():raise RuntimeError('Transition evidence is immutable; changed terrain needs a fresh bake directory')
    mesh=terrain.data;count=len(mesh.vertices);face_count=len(mesh.polygons)
    counts=np.empty(face_count,np.int32);starts=np.empty(face_count,np.int32)
    mesh.polygons.foreach_get('loop_total',counts);mesh.polygons.foreach_get('loop_start',starts)
    if not np.all(counts==3) or not np.array_equal(starts,np.arange(face_count,dtype=np.int32)*3):
        raise RuntimeError('Transition bake requires authoritative contiguous triangles before chart assignment')
    triangles=np.empty(len(mesh.loops),np.int32);mesh.loops.foreach_get('vertex_index',triangles);triangles=triangles.reshape(-1,3)
    face_normals=np.empty(face_count*3,np.float32);mesh.polygons.foreach_get('normal',face_normals)
    face_axes=np.argmax(np.abs(face_normals.reshape(-1,3)),axis=1).astype(np.int8);del face_normals,counts,starts
    points=np.empty(count*3,np.float32);normals=np.empty(count*3,np.float32)
    mesh.vertices.foreach_get('co',points);mesh.vertices.foreach_get('normal',normals)
    points=points.reshape(-1,3);normals=normals.reshape(-1,3);matrix=np.asarray(terrain.matrix_world,float)
    points=points@matrix[:3,:3].T+matrix[:3,3];normals=normals@np.linalg.inv(matrix[:3,:3]);normals/=np.linalg.norm(normals,axis=1)[:,None]
    before=hashlib.sha256(memoryview(np.ascontiguousarray(points,dtype='<f8'))).hexdigest()
    with np.load(source_surface_path,allow_pickle=False) as f:source={k:f[k] for k in f.files}
    delta=None
    if exterior_delta_spec:
        with np.load(exterior_delta_spec['path'],allow_pickle=False) as f:
            delta=f[exterior_delta_spec['delta_key']].copy();delta[f[exterior_delta_spec['protection_key']].astype(bool)]=0
    weights=np.empty((count,4),np.float32)
    for start in range(0,count,250000):
        stop=min(count,start+250000)
        weights[start:stop]=material_blend_weights(points[start:stop],normals[start:stop],source,minecraft_origin_xyz,
            exterior_delta=delta,structural_source_coordinates=structural_source_coordinates)
    del normals
    # Numerical ownership travels with actual native geometry. These are data
    # primvars, not colors or shader claims; independent visible face rays can
    # interpolate the precise contribution of each source scan after export.
    three=mesh.attributes.get('IsaacMinMaterialWeights012') or mesh.attributes.new(name='IsaacMinMaterialWeights012',type='FLOAT_VECTOR',domain='POINT')
    three.data.foreach_set('vector',np.ascontiguousarray(weights[:,:3]).ravel())
    fourth=mesh.attributes.get('IsaacMinMaterialWeight3') or mesh.attributes.new(name='IsaacMinMaterialWeight3',type='FLOAT',domain='POINT')
    fourth.data.foreach_set('value',np.ascontiguousarray(weights[:,3]))
    mesh['isaacmin_material_weight_order']=json.dumps(list(REQUIRED))
    by_id={m.get('asset_id',m.get('name')):i for i,m in enumerate(materials)}
    native_slots=np.asarray([by_id[name] for name in REQUIRED],np.int32)
    uv=mesh.uv_layers.get('PhysicalUV')
    if uv is None:raise RuntimeError('Native physical UV layer is required')
    uv_values=np.empty((len(mesh.loops),2),np.float32);uv.data.foreach_get('uv',uv_values.ravel())
    material_indices=np.empty(face_count,np.int32);mesh.polygons.foreach_get('material_index',material_indices)
    epsilon=1e-5;pure=0;mixed_parts=[];key_parts=[];projection=np.array([[1,2],[0,2],[0,1]])
    # Bulk mesh reads and bounded NumPy selections replace per-RNA-face traversal.
    for start in range(0,face_count,250000):
        stop=min(face_count,start+250000);ids=np.arange(start,stop,dtype=np.int32);vertices=triangles[start:stop]
        w=weights[vertices];dominant=np.argmax(w.mean(axis=1),axis=1)
        pure_mask=np.all(np.take_along_axis(w,dominant[:,None,None],axis=2)[:,:,0]>=1-epsilon,axis=1)
        material_indices[ids[pure_mask]]=native_slots[dominant[pure_mask]];pure+=int(pure_mask.sum())
        center=points[vertices].mean(axis=1);axes=projection[face_axes[start:stop]]
        coords=np.take_along_axis(center,axes,axis=1)
        mixed=ids[~pure_mask]
        mixed_parts.append(mixed);key_parts.append(np.column_stack((face_axes[mixed],np.floor(coords[~pure_mask]/tile_size_m).astype(np.int32))))
        if pure_mask.any():
            chosen=ids[pure_mask];corner_points=points[triangles[chosen]]
            coord=np.take_along_axis(corner_points,np.repeat(projection[face_axes[chosen]][:,None,:],3,axis=1),axis=2)
            repeat=np.asarray([m['repeat_m'] for m in materials])[material_indices[chosen]]
            uv_values[(chosen[:,None]*3+np.arange(3)).ravel()]=(coord/repeat[:,None,None]).reshape(-1,2)
    mixed=np.concatenate(mixed_parts);keys=np.concatenate(key_parts);del mixed_parts,key_parts
    if len(mixed):
        order=np.lexsort((keys[:,2],keys[:,1],keys[:,0]));mixed=mixed[order];keys=keys[order]
        boundaries=np.r_[0,np.flatnonzero(np.any(np.diff(keys,axis=0),axis=1))+1,len(keys)]
        groups=[(tuple(map(int,keys[a])),mixed[a:b]) for a,b in zip(boundaries[:-1],boundaries[1:])]
    else:groups=[]
    del keys
    maps,proof=_load_masters(materials)
    report={'schema_version':1,'recipe':RECIPE,'status':'baking','qualification':'not_run','source_surface':{'path':str(Path(source_surface_path).resolve()),'sha256':_digest(source_surface_path)},
            'producer_sha256':_digest(__file__),'weight_code_sha256':_digest(Path(__file__).with_name('material_assignment.py')),
            'material_order':list(REQUIRED),'source_weight_primvars':['IsaacMinMaterialWeights012','IsaacMinMaterialWeight3'],'masters':proof,'geometry_points_sha256_before':before,'tile_size_m':tile_size_m,'texels_per_m':texels_per_m,'gutter_pixels':gutter_pixels,
            'weight_recipe':{'source_smoothing':'separable binomial[1,2,1]/4 per1m cell then bilinear cell-centered sampling','soil_slope_falloff_degrees':[30,50],
                             'source_surface_distance_falloff_m':[.35,1.25],'deep_cave_and_structural':'rock only','pure_weight_epsilon':epsilon,'raster_interpolation':'source weights evaluated at native vertices, then barycentric interpolation on unchanged native triangles'},
            'pure_triangles_using_original_masters':pure,'transition_triangles':len(mixed),'tile_count':len(groups),'tiles':[],
            'disk_uncompressed_lower_estimate_bytes':int(len(groups)*(tile_size_m*texels_per_m+2*gutter_pixels)**2*14),
            'normal_semantics':'Each chart uses the same positive physicalUV projection and orientation for every master; blend decoded tangent vectors then renormalize within that single basis. Never mix components from different chart bases.',
            'limitations':['Dominant-axis chart boundaries retain existing projection discontinuities; target UV/seam inspection still required',
                           '16pixel gutter covers near-field filtering; long-range temporal minification must still be tested',
                           'Roughness linearly blends acquired scalar fields; wetness and BRDF response remain target-unqualified']}
    report_path.write_text(json.dumps(report,indent=2))
    for key,rows in groups:
        axis,tx,ty=key;axes=((1,2),(0,2),(0,1))[axis];all_vertices=np.unique(triangles[rows]);coords=points[all_vertices][:,axes]
        lo=np.floor(coords.min(axis=0)*texels_per_m)/texels_per_m-gutter_pixels/texels_per_m
        hi=np.ceil(coords.max(axis=0)*texels_per_m)/texels_per_m+gutter_pixels/texels_per_m
        size=np.ceil((hi-lo)*texels_per_m).astype(int)
        if max(size)>9216:raise RuntimeError('A transition chart exceeds bounded native tile resolution; refine mesh before baking')
        width,height=map(int,size);color=np.zeros((height,width,3),np.float32);rough=np.ones((height,width,1),np.float32);normal=np.zeros((height,width,3),np.float32);normal[:]=[.5,.5,1]
        priority=np.full((height,width),-np.inf,np.float32);axisdepth=np.full((height,width),np.nan,np.float32)
        for polygon in rows:
            vertices=triangles[polygon];coord=points[vertices][:,axes];pixel=(coord-lo)*texels_per_m-.5
            low=np.maximum(np.floor(pixel.min(axis=0)-gutter_pixels).astype(int),0);high=np.minimum(np.ceil(pixel.max(axis=0)+gutter_pixels).astype(int),[width-1,height-1])
            x,y=np.meshgrid(np.arange(low[0],high[0]+1),np.arange(low[1],high[1]+1));x=x.ravel();y=y.ravel()
            a,b,c=pixel;denom=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
            if abs(denom)<1e-8:continue
            wa=((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/denom
            wb=((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/denom
            bary=np.column_stack((wa,wb,1-wa-wb));score=bary.min(axis=1)
            chosen=score>priority[y,x];x=x[chosen];y=y[chosen];bary=bary[chosen];score=score[chosen]
            if not len(x):continue
            existing=priority[y,x]>=-1e-5;depth=bary@points[vertices,axis]
            if np.any(existing&(score>=-1e-5)&(np.abs(axisdepth[y,x]-depth)>.02)):
                raise RuntimeError('Different stacked transition surfaces overlap one chart; split chart layers before baking')
            blend=np.clip(bary,0,1);blend/=blend.sum(axis=1,keepdims=True);blend=blend@weights[vertices]
            physical=lo+(np.column_stack((x,y))+.5)/texels_per_m
            colors,roughness,normals_out=_mix(maps,physical,blend)
            color[y,x]=colors;rough[y,x]=roughness;normal[y,x]=normals_out;priority[y,x]=score;axisdepth[y,x]=depth
        # Exact chart coordinates, including independent rectangular image sizes.
        name=f'IsaacMinTransition_{axis}_{tx}_{ty}';folder=output/name;folder.mkdir()
        spec={'name':name,'asset_id':name,'repeat_m':tile_size_m,'qualification':'not_run','transition_source_families':list(REQUIRED)}
        files=[]
        for role,array in [('base_color',color),('roughness',rough),('normal',normal)]:
            # Raster coordinates use USD/Blender lower-left UV. PNG scanlines
            # start at the upper-left; reverse rows exactly once on serialization.
            path=folder/(role+'.png');_png16(path,array[::-1],srgb=role=='base_color');spec[role]=str(path)
            files.append({'role':role,'path':str(path),'sha256':_digest(path),'bytes':path.stat().st_size})
        material=material_factory(spec);slot=len(mesh.materials);mesh.materials.append(material)
        material_indices[rows]=slot
        for start in range(0,len(rows),250000):
            selected=rows[start:start+250000];vertices=triangles[selected];coords=points[vertices][:,:,axes]
            uv_values[(selected[:,None]*3+np.arange(3)).ravel()]=((coords-lo)*texels_per_m/size).reshape(-1,2)
        record={'material':name,'axis':axis,'axes':list(axes),'world_chart_min':lo.tolist(),'resolution':size.tolist(),'faces':len(rows),'files':files,
                'populated_pixels':int(np.isfinite(priority).sum()),'normal_basis':'same PhysicalUV projected coordinate order for all actual masters',
                'source_weight_min':weights[all_vertices].min(axis=0).tolist(),'source_weight_max':weights[all_vertices].max(axis=0).tolist()}
        report['tiles'].append(record);report_path.write_text(json.dumps(report,indent=2))
        del color,rough,normal,priority,axisdepth
    # Blender4.5 USD omits subsets when a material-index attribute is constant,
    # then binds slot0. Compact unused slots so a uniform nonzero selection cannot
    # export the wrong master. Preserve the exact selected material identities.
    used_slots=np.unique(material_indices);kept=[mesh.materials[int(i)] for i in used_slots]
    remap=np.full(len(mesh.materials),-1,np.int32);remap[used_slots]=np.arange(len(used_slots),dtype=np.int32)
    report['native_material_slot_remap']={str(int(old)):int(new) for new,old in enumerate(used_slots)}
    report['native_material_names']=[m.name for m in kept]
    mesh.materials.clear()
    for mat in kept:mesh.materials.append(mat)
    mesh.polygons.foreach_set('material_index',remap[material_indices]);uv.data.foreach_set('uv',uv_values.ravel())
    mesh.update();bpy.context.view_layer.update()
    after_points=np.empty(count*3,np.float32);mesh.vertices.foreach_get('co',after_points);after_points=after_points.reshape(-1,3)@matrix[:3,:3].T+matrix[:3,3]
    after=hashlib.sha256(memoryview(np.ascontiguousarray(after_points,dtype='<f8'))).hexdigest()
    if before!=after:raise RuntimeError('Transition baking unexpectedly changed ground geometry')
    report.update(status='baked_candidate',geometry_points_sha256_after=after,total_texture_bytes=sum(f['bytes'] for t in report['tiles'] for f in t['files']))
    report_path.write_text(json.dumps(report,indent=2));return report
