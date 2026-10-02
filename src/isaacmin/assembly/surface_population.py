"""Whole-region, camera-independent outdoor ground cover on exact final triangles."""
from pathlib import Path
import math
import numpy as np
from scipy.spatial.transform import Rotation
from isaacmin.io import atomic_json,read_json,sha256_file,utc_now
from isaacmin.terrain.surface_navigation import SurfaceQuery,source_fields


def random_cells(ix, iy, channel):
    with np.errstate(over='ignore'):
        h=np.asarray(ix,np.int64).astype(np.uint64)*np.uint64(0x9e3779b97f4a7c15)
        h^=np.asarray(iy,np.int64).astype(np.uint64)*np.uint64(0xbf58476d1ce4e5b9)
        h^=np.uint64(1729+channel*104729)
        h=(h^(h>>np.uint64(30)))*np.uint64(0xbf58476d1ce4e5b9)
        h=(h^(h>>np.uint64(27)))*np.uint64(0x94d049bb133111eb)
        return ((h^(h>>np.uint64(31)))>>np.uint64(11)).astype(float)/2**53


def cluster(x,y,size,channel):
    x,y=x/size,y/size;ix,iy=np.floor(x).astype(int),np.floor(y).astype(int)
    a,b=x-ix,y-iy;a=a*a*(3-2*a);b=b*b*(3-2*b)
    return ((1-a)*(1-b)*random_cells(ix,iy,channel)+a*(1-b)*random_cells(ix+1,iy,channel)
            +(1-a)*b*random_cells(ix,iy+1,channel)+a*b*random_cells(ix+1,iy+1,channel))


def route_distances(x,y,route,*,max_distance_m=None):
    distance=np.full(np.shape(x),np.inf)
    if route is None:return distance
    points=np.asarray(route['points_world_xyz'],float)[:,:2]
    if max_distance_m is not None:
        # Exact segment distances within the requested corridor. A midpoint
        # bound excludes distant segments without changing near-route decisions.
        from scipy.spatial import cKDTree
        half_width=route['corridor_width_m']/2;cap=float(max_distance_m)+half_width
        starts,ends=points[:-1],points[1:];d=ends-starts;length=np.sum(d*d,axis=1)
        valid=length>0;starts,d,length=starts[valid],d[valid],length[valid]
        if not len(starts):return distance
        mids=starts+d*.5;tree=cKDTree(mids);radius=cap+np.sqrt(length.max())*.5
        samples=np.column_stack((np.asarray(x).ravel(),np.asarray(y).ravel()))
        result=np.full(len(samples),cap)
        for start in range(0,len(samples),100000):
            sample=samples[start:start+100000]
            near=np.flatnonzero(tree.query(sample)[0]<=radius)
            if not len(near):continue
            groups=tree.query_ball_point(sample[near],radius)
            sizes=np.fromiter((len(g) for g in groups),int,len(groups))
            rows=np.repeat(near,sizes);cols=np.concatenate(groups).astype(int)
            delta=sample[rows]-starts[cols]
            t=np.clip(np.sum(delta*d[cols],axis=1)/length[cols],0,1)
            values=np.linalg.norm(delta-d[cols]*t[:,None],axis=1)
            np.minimum.at(result,start+rows,values)
        return result.reshape(np.shape(x))-half_width
    for start,end in zip(points[:-1],points[1:]):
        d=end-start;length=np.dot(d,d)
        if length==0:continue
        t=np.clip(((x-start[0])*d[0]+(y-start[1])*d[1])/length,0,1)
        distance=np.minimum(distance,np.hypot(x-start[0]-t*d[0],y-start[1]-t*d[1]))
    return distance-route['corridor_width_m']/2


def populate_surface(workspace, terrain_directory, object_directory, output, *, route=None,_bounds=None,_context=None):
    workspace,terrain_directory,object_directory,output=map(Path,(workspace,terrain_directory,object_directory,output))
    output.mkdir(parents=True,exist_ok=False)
    if _context is None:
        q=SurfaceQuery(terrain_directory);fields=source_fields(terrain_directory/'material_fields.npz')
        canopy=np.load(object_directory/'canopy.npz')['cover_fraction']
        q.ground_file_sha256=sha256_file(terrain_directory/'vertices.npy')
    else:q,fields,canopy=_context
    from isaacmin.terrain.surface_partitions import source_windows,source_cell_grid
    full_bounds=[q.xmin,q.zmin,q.xmax,q.zmax]
    if _bounds is None and max(q.xmax-q.xmin,q.zmax-q.zmin)>512:
        parts=[];counts={};rejected={};prototypes=None
        for i,bounds in enumerate(source_windows(full_bounds)):
            directory=output/f'Part{i:04d}'
            record=populate_surface(workspace,terrain_directory,object_directory,directory,
                route=route,_bounds=bounds,_context=(q,fields,canopy))
            prototypes=record['prototypes']
            parts.append(dict(path=str(directory.relative_to(output)),instances=record['instances'],
                bounds_source_xz=bounds,placements_sha256=sha256_file(directory/'placements.npz')))
            for k,v in record['instance_counts_by_asset'].items():counts[k]=counts.get(k,0)+v
            for k,v in record['rejected'].items():rejected[k]=rejected.get(k,0)+v
            atomic_json(output/'progress.json',dict(completed_parts=len(parts),instances=sum(p['instances'] for p in parts)))
        record.update(parts=parts,instances=sum(p['instances'] for p in parts),instance_counts_by_asset=counts,
            rejected=rejected,bounds_source_xz=full_bounds,partition_policy='Unique global cells; common final surface, route, source canopy and density')
        atomic_json(output/'population.json',record);return record
    bounds=_bounds or full_bounds
    prototypes=[]
    for asset in read_json(workspace/'state/normalized_assets.json')['assets']:
        for obj in asset['objects']:
            name=obj['object_name'];aid=asset['asset_id'];layer=None
            if aid=='grass_medium_01' and any(s in name for s in ('_small_','_mid_','_large_')):layer='short'
            elif aid=='grass_medium_02' and not name.endswith('_geo'):layer='tall'
            elif aid=='dandelion_01':layer='forb'
            elif aid=='celandine_01':layer='woodland_forb'
            elif aid=='fern_02':layer='fern'
            if layer:prototypes.append(dict(obj,asset_id=aid,blend_sha256=asset['output_sha256'],layer=layer))
    if not all(any(p['layer']==k for p in prototypes) for k in ('short','tall','forb')):
        raise ValueError('Required original meadow assets unavailable')
    groups={};weights=np.sqrt([p['dimensions_m'][0]*p['dimensions_m'][1] for p in prototypes])
    for layer in ('short','tall','forb'):
        selected=np.array([i for i,p in enumerate(prototypes) if p['layer']==layer]);cdf=np.cumsum(weights[selected]);cdf/=cdf[-1]
        groups[layer]=(selected,cdf)
    cell=.12;ox,oy,oz=q.origin
    # Global source cells keep overlapping builds consistent even when their
    # local USD origins differ. World +Y points along Minecraft -Z.
    gx,gy=source_cell_grid(bounds,full_bounds,cell,negative_z=True)
    gx,gy=gx.ravel(),gy.ravel();x=(gx+random_cells(gx,gy,1))*cell-ox;y=(gy+random_cells(gx,gy,2))*cell+oz
    mx,mz=fields['min_xz'];sx=np.floor(x+ox-mx).astype(int);sz=np.floor(-y+oz-mz).astype(int)
    names=fields['block_names'][fields['substrate_id'][sz,sx]].astype(str)
    soil=np.zeros(len(x),bool)
    for token in ('grass_block','dirt','podzol','moss_block','mud'):soil|=np.char.find(names,token)>=0
    biome=fields['biome'][sz,sx]
    allowed=np.isin(biome,['minecraft:plains','minecraft:sunflower_plains','minecraft:meadow',
        'minecraft:forest','minecraft:flower_forest','minecraft:birch_forest','minecraft:old_growth_birch_forest'])
    dry=~(fields['water_validity'][sz,sx]&(fields['water_height'][sz,sx]>=fields['height'][sz,sx]))
    patch=cluster(x+ox,y-oz,7.5,30);small=cluster(x+ox,y-oz,1.3,31)
    density=(24+42*patch)*(.45+.8*small)*(1-.65*canopy[sz,sx])
    good=(soil&allowed&dry&(canopy[sz,sx]<=.65)&(random_cells(gx,gy,3)<np.minimum(.93,density*cell*cell)))
    gx,gy,x,y,patch=[a[good] for a in (gx,gy,x,y,patch)]
    distances=route_distances(x,y,route,max_distance_m=3.);keep=distances>=.6
    rejected={'trail':int((~keep).sum())}
    gx,gy,x,y,patch=[a[keep] for a in (gx,gy,x,y,patch)]
    ids=np.zeros(len(x),np.int32);draw=random_cells(gx,gy,4)
    tall=random_cells(gx,gy,5)<.06+.08*patch;forb=random_cells(gx,gy,13)<.055
    for group,mask in [('short',~tall),('tall',tall),('forb',forb)]:
        selected,cdf=groups[group];ids[mask]=selected[np.searchsorted(cdf,draw[mask])]
    scale=.85+.27*random_cells(gx,gy,6);yaw=2*np.pi*random_cells(gx,gy,7)
    # Distinct sparse woodland layer. Reusing meadow density under closed canopy
    # would produce implausible grass carpets and omit the native fern assets.
    wood_cell=.6
    wx,wy=source_cell_grid(bounds,full_bounds,wood_cell,negative_z=True)
    wx,wy=wx.ravel(),wy.ravel();px=(wx+random_cells(wx,wy,101))*wood_cell-ox;py=(wy+random_cells(wx,wy,102))*wood_cell+oz
    ix=np.floor(px+ox-mx).astype(int);iz=np.floor(-py+oz-mz).astype(int)
    canopy_here=canopy[iz,ix];biome_here=fields['biome'][iz,ix]
    woodland=np.isin(biome_here,['minecraft:forest','minecraft:flower_forest','minecraft:birch_forest','minecraft:old_growth_birch_forest',
        'minecraft:dark_forest','minecraft:taiga','minecraft:old_growth_pine_taiga','minecraft:old_growth_spruce_taiga'])
    tropical=np.isin(biome_here,['minecraft:jungle','minecraft:sparse_jungle','minecraft:bamboo_jungle'])
    names_here=fields['block_names'][fields['substrate_id'][iz,ix]].astype(str);wood_soil=np.zeros(len(px),bool)
    for token in ('grass_block','dirt','podzol','moss_block','mud'):wood_soil|=np.char.find(names_here,token)>=0
    dry_here=~(fields['water_validity'][iz,ix]&(fields['water_height'][iz,ix]>=fields['height'][iz,ix]))
    wood_eligible=(woodland|tropical)&wood_soil&dry_here&(canopy_here>=.2)
    fern_colonies=cluster(px+ox,py-oz,6.,108)
    wood_density=np.where(tropical,(.30+1.0*fern_colonies**2)*(.5+canopy_here),
        (.6+1.6*fern_colonies**2)*(.5+canopy_here))
    wood_eligible&=random_cells(wx,wy,103)<wood_density*wood_cell**2
    wood_eligible&=route_distances(px,py,route,max_distance_m=3.)>=.6
    wx,wy,px,py,canopy_here,tropical=[a[wood_eligible] for a in (wx,wy,px,py,canopy_here,tropical)]
    wood_ids=np.empty(len(px),np.int32);fern_group=np.array([i for i,p in enumerate(prototypes) if p['layer']=='fern'])
    flower_group=np.array([i for i,p in enumerate(prototypes) if p['layer']=='woodland_forb'])
    if not len(fern_group) or not len(flower_group):raise ValueError('Distinct native woodland originals missing')
    # Sparse shade grasses accompany fern colonies; the dense open meadow
    # layer remains independently excluded from closed canopy.
    is_grass=~tropical&(random_cells(wx,wy,109)<.55)
    is_fern=~is_grass&(tropical|((canopy_here>=.3)&(random_cells(wx,wy,104)<.65)))
    shade_grass=np.array([i for i,p in enumerate(prototypes) if p['layer']=='short'])
    for group,mask in [(fern_group,is_fern),(flower_group,~is_fern&~is_grass),(shade_grass,is_grass)]:
        wood_ids[mask]=group[np.minimum((random_cells(wx[mask],wy[mask],105)*len(group)).astype(int),len(group)-1)]
    gx=np.concatenate((gx,wx));gy=np.concatenate((gy,wy));x=np.concatenate((x,px));y=np.concatenate((y,py));ids=np.concatenate((ids,wood_ids))
    scale=np.concatenate((scale,.85+.27*random_cells(wx,wy,106)));yaw=np.concatenate((yaw,2*np.pi*random_cells(wx,wy,107)))
    pz=q.heights(x,y);radius=.08
    normal=np.column_stack((-(q.heights(x+radius,y)-q.heights(x-radius,y))/(2*radius),
                            -(q.heights(x,y+radius)-q.heights(x,y-radius))/(2*radius),np.ones(len(x))))
    normal/=np.linalg.norm(normal,axis=1,keepdims=True)
    axis=np.column_stack((np.cos(yaw),np.sin(yaw),np.zeros(len(x))))
    axis-=normal*np.sum(axis*normal,axis=1,keepdims=True);axis/=np.linalg.norm(axis,axis=1,keepdims=True)
    rotation=np.stack((axis,np.cross(normal,axis),normal),axis=-1)
    keep=np.isfinite(pz)&np.isfinite(normal).all(axis=1)&(normal[:,2]>=np.cos(np.deg2rad(30)))
    rejected['edge_or_slope']=int((~keep).sum())
    for index,prototype in enumerate(prototypes):
        selected=np.flatnonzero(keep&(ids==index))
        anchors=np.asarray(prototype['contact_anchors_local_m'],float)
        if not len(anchors):raise ValueError('Plant roots need measured native anchors')
        for begin in range(0,len(selected),100000):
            ind=selected[begin:begin+100000];r=rotation[ind];s=scale[ind]
            minimum=np.full(len(ind),np.inf);maximum=-minimum;total=np.zeros(len(ind));finite=np.ones(len(ind),bool)
            for anchor in anchors:
                a=np.einsum('nij,j->ni',r,anchor)*s[:,None]
                support=q.heights(x[ind]+a[:,0],y[ind]+a[:,1])-a[:,2]
                finite&=np.isfinite(support);minimum=np.minimum(minimum,support);maximum=np.maximum(maximum,support);total+=support
            accept=finite&(maximum-minimum<=.02)
            keep[ind]&=accept;pz[ind]=total/len(anchors)
    rejected['root_footprint']=len(x)-int(keep.sum())-rejected['edge_or_slope']
    quat=Rotation.from_matrix(rotation[keep]).as_quat()
    positions=np.column_stack((x[keep],y[keep],pz[keep])).astype(np.float32)
    np.savez_compressed(output/'placements.npz',positions=positions,prototype_indices=ids[keep],
        scales=np.repeat(scale[keep,None],3,axis=1).astype(np.float32),
        orientations_wxyz=np.column_stack((quat[:,3],quat[:,:3])).astype(np.float32),
        world_cells=np.column_stack((gx[keep],gy[keep])).astype(np.int32))
    record=dict(status='final_surface_rooted_candidates',at_utc=utc_now(),instances=len(positions),
        instance_counts_by_asset={aid:int(sum(np.count_nonzero(ids[keep]==i) for i,p in enumerate(prototypes) if p['asset_id']==aid)) for aid in sorted({p['asset_id'] for p in prototypes})},
        prototypes=prototypes,rejected=rejected,camera_dependent=False,geometry_reduction=False,
        placement_cell_frame='global Minecraft X,-Z; independent of local USD origin',
        ground_sha256=q.ground_file_sha256,root_max_spread_m=.02,
        biome_scope='temperate ground cover plus tropical fern colonies; original fern as a shared ecological growth form, not exact botanical identity; tropical broadleaf herbs handled separately',
        woodland_layers='Canopy-conditioned sparse shade grasses, fern colonies and forbs, plus separately supported original forest litter',
        qualification='not_run',producer_sha256=sha256_file(Path(__file__)))
    atomic_json(output/'population.json',record)
    return record
