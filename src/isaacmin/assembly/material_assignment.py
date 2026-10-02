"""Source-coordinate material ownership, using existing verified native PBRs."""
from collections import Counter
import hashlib
import math
from pathlib import Path
import numpy as np

RULE_VERSION='source_surface_materials_1'
REQUIRED=('brown_mud_dry','rock_face_03','leafy_grass','forest_ground_04')
FOREST=('minecraft:forest','minecraft:birch_forest','minecraft:old_growth_birch_forest')
MEADOW=('minecraft:plains','minecraft:meadow')


def _banks(height,water_height,wet,radius=2):
    result=wet.copy();h,w=height.shape
    for dz in range(-radius,radius+1):
        for dx in range(-radius,radius+1):
            if dx*dx+dz*dz>radius*radius:continue
            za,zb=max(0,-dz),min(h,h-dz);xa,xb=max(0,-dx),min(w,w-dx)
            neighbour=wet[za+dz:zb+dz,xa+dx:xb+dx]
            levels=water_height[za+dz:zb+dz,xa+dx:xb+dx]
            result[za:zb,xa:xb]|=neighbour & (np.abs(height[za:zb,xa:xb]-levels)<=1.5)
    return result


def classify_surface_materials(centers,normals,source_surface,minecraft_origin_xyz,materials,*,exterior_delta=None,route_mask=None,structural_source_coordinates=None):
    centers,normals=np.asarray(centers,float),np.asarray(normals,float)
    if centers.ndim!=2 or centers.shape[1]!=3 or normals.shape!=centers.shape or not np.isfinite(centers).all() or not np.isfinite(normals).all():
        raise ValueError('Finite world-coordinate face centers/normals are required')
    by_id={m.get('asset_id',m.get('name')):i for i,m in enumerate(materials)}
    if not set(REQUIRED)<=set(by_id):raise ValueError('Source-aware assignment requires the four declared acquired materials')
    for material in materials:
        if not math.isfinite(material['repeat_m']) or not .1<=material['repeat_m']<=20:
            raise ValueError('Material physical repeat must be verified and bounded')
    height=np.asarray(source_surface['height'],float);validity=np.asarray(source_surface['validity'],bool)
    biomes=np.asarray(source_surface['biome']);substrates=np.asarray(source_surface['substrate_id'])
    names=np.asarray(source_surface['block_names']);water_height=np.asarray(source_surface['water_height'],float)
    water_validity=np.asarray(source_surface['water_validity'],bool)
    if any(a.shape!=height.shape for a in (validity,biomes,substrates,water_height,water_validity)):
        raise ValueError('Source material fields have incompatible shapes')
    if exterior_delta is not None:
        exterior_delta=np.asarray(exterior_delta,float)
        if exterior_delta.shape!=height.shape or not np.isfinite(exterior_delta[validity]).all():raise ValueError('Exterior delta has invalid shape/values')
    expected=height+(exterior_delta if exterior_delta is not None else 0)
    # Subsurface aquifers must never strip meadow/forest cover from dry ground.
    wet=validity&water_validity&(water_height>=height)
    banks=_banks(height,water_height,wet)
    ox,oy,oz=minecraft_origin_xyz;mx,mz=source_surface['min_xz']
    ix=np.floor(centers[:,0]+ox-mx).astype(int);iz=np.floor(-centers[:,1]+oz-mz).astype(int)
    inside=(ix>=0)&(iz>=0)&(ix<height.shape[1])&(iz<height.shape[0])
    cx=np.clip(ix,0,height.shape[1]-1);cz=np.clip(iz,0,height.shape[0]-1)
    valid=inside&validity[cz,cx]
    length=np.linalg.norm(normals,axis=1);up=normals[:,2]/np.maximum(length,1e-12)
    difference=centers[:,2]+oy-expected[cz,cx]
    # Near-surface ownership excludes cave floors, deep faces and downward roofs.
    exterior=valid&(np.abs(difference)<=.8)&(up>=math.cos(math.radians(50)))
    substrate_names=names[substrates[cz,cx]].astype(str)
    organic=np.array([any(t in name for t in ('grass_block','dirt','podzol','mycelium','moss_block','mud','farmland')) for name in substrate_names])
    grass_cover=np.array([any(t in name for t in ('grass_block','podzol','mycelium','moss_block')) for name in substrate_names])
    sediment=np.array([any(t in name for t in ('sand','gravel','clay')) for name in substrate_names])
    gentle=up>=math.cos(math.radians(35));biome=biomes[cz,cx]
    indices=np.full(len(centers),by_id['rock_face_03'],np.int32)
    reasons=np.full(len(centers),'rock_deep_steep_or_exposed',dtype='<U48')
    bare=exterior&(organic|sediment)
    indices[bare]=by_id['brown_mud_dry'];reasons[bare]='bare_soil_or_sediment_candidate'
    meadow=exterior&organic&grass_cover&gentle&np.isin(biome,MEADOW)
    forest=exterior&organic&gentle&np.isin(biome,FOREST)
    indices[meadow]=by_id['leafy_grass'];reasons[meadow]='source_meadow_ground'
    indices[forest]=by_id['forest_ground_04'];reasons[forest]='source_deciduous_litter_candidate'
    bank=exterior&(organic|sediment)&banks[cz,cx]
    indices[bank]=by_id['brown_mud_dry'];reasons[bank]='exposed_water_bank_or_bed'
    if route_mask is not None:
        route_mask=np.asarray(route_mask,bool)
        if route_mask.shape!=height.shape:raise ValueError('Route mask differs from source sampling')
        trail=exterior&(organic|sediment)&route_mask[cz,cx]
        indices[trail]=by_id['brown_mud_dry'];reasons[trail]='authored_or_source_explicit_trail'
    reasons[~valid]='outside_valid_source_rock_candidate'
    structural=np.zeros(len(centers),bool)
    raw_structure=np.asarray(structural_source_coordinates if structural_source_coordinates is not None else [],dtype=float)
    if not np.isfinite(raw_structure).all() or not np.equal(raw_structure,np.floor(raw_structure)).all():
        raise ValueError('Structural source coordinates must be finite integer block min corners')
    structure_coords=raw_structure.astype(np.int64)
    if structure_coords.size:
        if structure_coords.ndim!=2 or structure_coords.shape[1]!=3:
            raise ValueError('Structural coordinates must be Minecraft block min-corner Nx3')
        source_points=np.column_stack((centers[:,0]+ox,centers[:,2]+oy,-centers[:,1]+oz))
        source_normals=np.column_stack((normals[:,0],normals[:,2],-normals[:,1]))/np.maximum(length[:,None],1e-12)
        low,high=structure_coords.min(axis=0)-.4,structure_coords.max(axis=0)+1.4
        nearby=np.flatnonzero(np.all((source_points>=low)&(source_points<=high),axis=1))
        retained={tuple(p) for p in structure_coords.tolist()}
        for face in nearby:
            structural[face]=any(tuple(np.floor(source_points[face]+offset*source_normals[face]).astype(int)) in retained for offset in (0,-.35,.35))
        indices[structural]=by_id['rock_face_03']
        reasons[structural]='retained_structural_unqualified'
    unique,count_values=np.unique(indices,return_counts=True)
    counts={materials[int(i)].get('asset_id',materials[int(i)].get('name')):int(c) for i,c in zip(unique,count_values)}
    report={'rule_version':RULE_VERSION,'qualification':'not_run','face_count':len(indices),'material_face_counts':dict(counts),
            'assignment_reason_counts':dict(zip(*[a.tolist() for a in np.unique(reasons,return_counts=True)])),'exposed_water_source_columns':int(wet.sum()),
            'aquifer_only_columns_excluded':int((water_validity&validity&~wet).sum()),
            'bank_source_columns':int(banks.sum()),'outside_valid_source_faces':int((~valid).sum()),
            'physical_repeat_m':{m.get('asset_id',m.get('name')):m['repeat_m'] for m in materials},
            'displacement_eligible_material_indices':[by_id[a] for a in ('brown_mud_dry','leafy_grass','forest_ground_04')],
            'protected_rock_material_indices':[by_id['rock_face_03']],
            'structural_ownership':{'source_block_count':len(structure_coords),'face_count':int(structural.sum()),
                                    'face_indices':np.flatnonzero(structural).tolist(),
                                    'source_coordinate_sha256':hashlib.sha256(structure_coords.astype('<i8').tobytes()).hexdigest(),
                                    'classification':'retained_structural_unqualified',
                                    'mapping':'source voxel containing face center or offset0.35m along either face-normal direction',
                                    'material_appearance':'Existing rock slot is an explicit unqualified diagnostic appearance for retained masonry; not a natural-rock ownership claim',
                                    'isaac_appearance_qualification':'not_run'},
            'unsupported_surface_biomes':dict(Counter(biome[exterior&~np.isin(biome,(*MEADOW,*FOREST))].tolist())),
            'surface_proximity_tolerance_m':.8,'vegetated_max_slope_degrees':35,'soil_max_slope_degrees':50,
            'transition_qualification':'not_run; discrete face ownership needs target seam/transition review',
            'uv_qualification':'not_run; caller must use each selected material physical repeat and test stretch in Isaac',
            'isaac_qualification':'not_run',
            'limitations':['Forest biome litter assignment lacks within-canopy decay mapping',
                           'Dry soil scan is a bank/bed candidate; target wetness appearance remains unqualified',
                           'Steep/deep rock uses existing verified rock material; lithology remains unidentified',
                           'Outside-source faces are explicit coverage-boundary candidates, never inferred source terrain']}
    return indices,report


def assign_materials(terrain,materials,source_surface_path,minecraft_origin_xyz,*,exterior_delta_spec=None,route_mask=None,structural_source_coordinates=None):
    """Batch native polygon reads; bounded classification for dense real terrain."""
    source_surface_path=Path(source_surface_path)
    with np.load(source_surface_path,allow_pickle=False) as source:surface={k:source[k] for k in source.files}
    total=len(terrain.data.polygons);centers=np.empty(total*3,np.float32);normals=np.empty(total*3,np.float32)
    terrain.data.polygons.foreach_get('center',centers);terrain.data.polygons.foreach_get('normal',normals)
    centers=centers.reshape(-1,3);normals=normals.reshape(-1,3);matrix=np.asarray(terrain.matrix_world,float)
    normal_matrix=np.linalg.inv(matrix[:3,:3]);delta=None
    if exterior_delta_spec:
        with np.load(exterior_delta_spec['path'],allow_pickle=False) as values:
            delta=values[exterior_delta_spec['delta_key']].copy();delta[values[exterior_delta_spec['protection_key']].astype(bool)]=0
    indices=np.empty(total,np.int32);structural_faces=[];material_counts=Counter();reason_counts=Counter();unsupported=Counter();outside=0;report=None
    for start in range(0,total,250000):
        end=min(total,start+250000)
        indices[start:end],part=classify_surface_materials(centers[start:end]@matrix[:3,:3].T+matrix[:3,3],normals[start:end]@normal_matrix,
            surface,minecraft_origin_xyz,materials,exterior_delta=delta,route_mask=route_mask,structural_source_coordinates=structural_source_coordinates)
        material_counts.update(part['material_face_counts']);reason_counts.update(part['assignment_reason_counts']);unsupported.update(part['unsupported_surface_biomes'])
        outside+=part['outside_valid_source_faces'];structural_faces.extend(start+i for i in part['structural_ownership']['face_indices']);report=part
    if report is None:raise ValueError('Material assignment requires actual nonempty geometry')
    terrain.data.polygons.foreach_set('material_index',indices)
    report.update(face_count=total,material_face_counts=dict(material_counts),assignment_reason_counts=dict(reason_counts),
                  unsupported_surface_biomes=dict(unsupported),outside_valid_source_faces=outside,classification_batch_faces=250000)
    report['structural_ownership'].update(face_indices=structural_faces,face_count=len(structural_faces))
    report['source_surface_sha256']=hashlib.sha256(source_surface_path.read_bytes()).hexdigest();report['minecraft_origin_xyz']=list(minecraft_origin_xyz)
    ownership=terrain.data.attributes.get('IsaacMinSourceOwnership') or terrain.data.attributes.new(name='IsaacMinSourceOwnership',type='INT',domain='FACE')
    values=np.zeros(total,dtype=np.int32);values[structural_faces]=1;ownership.data.foreach_set('value',values)
    report['structural_ownership']['blender_face_attribute']='IsaacMinSourceOwnership:0=natural_candidate,1=retained_structural_unqualified'
    report['structural_ownership']['face_index_scope']='current pre-triangulation mesh; face attribute follows native topology operations'
    return report


def _sample_bilinear(field, x, z):
    """Cell-centred source data with clamped border, including fractional weights."""
    h,w=field.shape;fx=np.clip(x-.5,0,w-1);fz=np.clip(z-.5,0,h-1)
    ix=np.floor(fx).astype(int);iz=np.floor(fz).astype(int);jx=np.minimum(ix+1,w-1);jz=np.minimum(iz+1,h-1)
    a=fx-ix;b=fz-iz
    return field[iz,ix]*(1-a)*(1-b)+field[iz,jx]*a*(1-b)+field[jz,ix]*(1-a)*b+field[jz,jx]*a*b


def _smooth_field(field,radius=1):
    """Separable binomial1m kernel; no wrapped source/world boundaries."""
    result=np.asarray(field,float)
    for axis in (0,1):
        pad=[(0,0),(0,0)];pad[axis]=(radius,radius);p=np.pad(result,pad,mode='edge')
        result=(np.take(p,range(0,result.shape[axis]),axis=axis)+2*np.take(p,range(1,result.shape[axis]+1),axis=axis)+np.take(p,range(2,result.shape[axis]+2),axis=axis))/4
    return result


def _ramp(value,lo,hi):
    value=np.clip((value-lo)/(hi-lo),0,1);return value*value*(3-2*value)


def material_blend_weights(points,normals,source_surface,minecraft_origin_xyz,*,exterior_delta=None,route_mask=None,structural_source_coordinates=None):
    """Continuous actual-source soil/rock/meadow/litter weights, never a shader.

    Order is REQUIRED. Binomial1m source smoothing and bilinear sampling remove
    block-cell jumps. A30..50deg coverage falloff and0.35..1.25m exterior proximity
    retain rock on steep/deep faces. Native baker must sample each physical PBR
    scale and convert its normals into the chosen chart basis explicitly.
    """
    p=np.asarray(points,float);n=np.asarray(normals,float)
    if p.ndim!=2 or p.shape[1]!=3 or p.shape!=n.shape or not np.isfinite(p).all() or not np.isfinite(n).all():raise ValueError('Finite world positions and normals required')
    length=np.linalg.norm(n,axis=1)
    if np.any(length<.5):raise ValueError('Surface normal magnitude is invalid')
    up=n[:,2]/length;h=np.asarray(source_surface['height'],float);valid=np.asarray(source_surface['validity'],bool)
    expected=h+(np.asarray(exterior_delta,float) if exterior_delta is not None else 0)
    if not np.isfinite(expected[valid]).all():raise ValueError('Source heights must be finite in valid coverage')
    expected=np.where(valid,expected,0)
    names=np.asarray(source_surface['block_names'])[np.asarray(source_surface['substrate_id'],int)].astype(str)
    organic=np.zeros(h.shape,bool);cover=np.zeros(h.shape,bool);sediment=np.zeros(h.shape,bool)
    for token in ('grass_block','dirt','podzol','mycelium','moss_block','mud','farmland'):organic|=np.char.find(names,token)>=0
    for token in ('grass_block','podzol','mycelium','moss_block'):cover|=np.char.find(names,token)>=0
    for token in ('sand','gravel','clay'):sediment|=np.char.find(names,token)>=0
    wet=valid&source_surface['water_validity']&(source_surface['water_height']>=h)
    banks=_banks(h,source_surface['water_height'],wet)
    ox,oy,oz=minecraft_origin_xyz;mx,mz=source_surface['min_xz'];x=p[:,0]+ox-mx;z=-p[:,1]+oz-mz
    inside=(x>=0)&(z>=0)&(x<h.shape[1])&(z<h.shape[0])
    sample=lambda field:_sample_bilinear(_smooth_field(field),x,z)
    good=inside&(_sample_bilinear(valid.astype(float),x,z)>.999)
    difference=np.abs(p[:,2]+oy-_sample_bilinear(expected,x,z))
    proximity=1-_ramp(difference,.35,1.25)
    slope=_ramp(up,math.cos(math.radians(50)),math.cos(math.radians(30)))
    soil=sample(valid&(organic|sediment))*proximity*slope*good
    bank=sample(banks);trail=sample(route_mask) if route_mask is not None else np.zeros(len(p))
    biome=np.asarray(source_surface['biome'])
    meadow=sample(valid&organic&cover&np.isin(biome,MEADOW))*(1-bank)*(1-trail)
    forest=sample(valid&organic&np.isin(biome,FOREST))*(1-bank)*(1-trail)
    vegetation=np.maximum(1,meadow+forest);meadow/=vegetation;forest/=vegetation
    weights=np.column_stack((soil*(1-meadow-forest),1-soil,soil*meadow,soil*forest))
    structure=np.asarray(structural_source_coordinates if structural_source_coordinates is not None else [],float)
    if structure.size:
        if structure.ndim!=2 or structure.shape[1]!=3 or not np.isfinite(structure).all() or not np.equal(structure,np.floor(structure)).all():raise ValueError('Finite integer structural block min corners required')
        source_points=np.column_stack((p[:,0]+ox,p[:,2]+oy,-p[:,1]+oz));source_normals=np.column_stack((n[:,0],n[:,2],-n[:,1]))/length[:,None]
        low,high=structure.min(axis=0)-.4,structure.max(axis=0)+1.4
        retained={tuple(v) for v in structure.astype(int).tolist()}
        for i in np.flatnonzero(np.all((source_points>=low)&(source_points<=high),axis=1)):
            if any(tuple(np.floor(source_points[i]+o*source_normals[i]).astype(int)) in retained for o in (0,-.35,.35)):weights[i]=[0,1,0,0]
    weights=np.clip(weights,0,1);weights/=weights.sum(axis=1,keepdims=True)
    return weights.astype(np.float32)
