"""Camera-independent scanned stones, supported on the final terrain mesh."""
import math
from pathlib import Path
import numpy as np
from isaacmin.io import read_json,atomic_json
from isaacmin.terrain.surface_navigation import SurfaceQuery,source_fields
from isaacmin.assembly.surface_population import random_cells,cluster,route_distances


def scatter_rocks(terrain,objects,prototypes,*,route=None):
    q=SurfaceQuery(terrain);fields=source_fields(Path(terrain)/'material_fields.npz')
    canopy=np.load(Path(objects)/'canopy.npz')['cover_fraction']
    cell=2.;ox,oy,oz=q.origin
    ix,iz=np.meshgrid(np.arange(math.floor(q.xmin/cell),math.ceil(q.xmax/cell)),
                      np.arange(math.floor(q.zmin/cell),math.ceil(q.zmax/cell)))
    ix,iz=ix.ravel(),iz.ravel()
    sx=(ix+random_cells(ix,iz,301))*cell;sz=(iz+random_cells(ix,iz,302))*cell
    x,y=sx-ox,-sz+oz
    mx,mz=fields['min_xz'];cx=np.floor(sx-mx).astype(int);cz=np.floor(sz-mz).astype(int)
    biome=fields['biome'][cz,cx]
    mountain=np.isin(biome,['minecraft:jagged_peaks','minecraft:frozen_peaks','minecraft:stony_peaks',
        'minecraft:snowy_slopes','minecraft:windswept_hills','minecraft:windswept_gravelly_hills','minecraft:grove'])
    woodland=np.isin(biome,['minecraft:forest','minecraft:flower_forest','minecraft:birch_forest',
        'minecraft:old_growth_birch_forest','minecraft:dark_forest','minecraft:taiga'])
    wet=fields['water_validity'][cz,cx]&(fields['water_height'][cz,cx]>=fields['height'][cz,cx])
    patches=cluster(sx,sz,13.,306)
    density=np.where(mountain,.018,.007)*(0.25+1.5*patches)
    frozen=np.isin(fields['block_names'][fields['substrate_id'][cz,cx]],
        ['minecraft:ice','minecraft:packed_ice','minecraft:blue_ice','minecraft:frosted_ice'])
    good=(mountain|woodland)&~wet&~frozen&(random_cells(ix,iz,303)<density*cell*cell)
    good&=(sx>=q.xmin+2)&(sx<=q.xmax-2)&(sz>=q.zmin+2)&(sz<=q.zmax-2)
    good&=route_distances(x,y,route,max_distance_m=3.)>=2.
    rows=[];rejected={};counts={}
    for k in np.flatnonzero(good):
        aid='boulder_01' if mountain[k] else 'rock_moss_set_02'
        choices=[p for p in prototypes if p['asset_id']==aid]
        if not choices:
            rejected['missing:'+aid]=rejected.get('missing:'+aid,0)+1;continue
        proto=choices[min(int(random_cells(ix[k],iz[k],304)*len(choices)),len(choices)-1)]
        dzdx=float(q.heights(x[k]+.5,y[k])-q.heights(x[k]-.5,y[k]))
        dzdy=float(q.heights(x[k],y[k]+.5)-q.heights(x[k],y[k]-.5))
        normal=np.array([-dzdx,-dzdy,1.]);normal/=np.linalg.norm(normal)
        if not np.isfinite(normal).all() or normal[2]<math.cos(math.radians(45)):
            rejected['slope']=rejected.get('slope',0)+1;continue
        angle=2*np.pi*random_cells(ix[k],iz[k],305)
        tangent=np.array([np.cos(angle),np.sin(angle),0.]);tangent-=normal*np.dot(tangent,normal)
        tangent/=np.linalg.norm(tangent);rotation=np.column_stack((tangent,np.cross(normal,tangent),normal))
        anchors=np.asarray(proto['contact_anchors_local_m'])@rotation.T
        heights=q.heights(x[k]+anchors[:,0],y[k]+anchors[:,1])-anchors[:,2]
        burial=.04*proto['dimensions_m'][2]
        if not np.isfinite(heights).all() or np.ptp(heights)+burial>min(.6,.6*proto['dimensions_m'][2]):
            rejected['basal_support']=rejected.get('basal_support',0)+1;continue
        height=float(heights.min()-burial)
        rows.append(dict(asset_id=aid,object_name=proto['object_name'],scene_asset=proto['scene_asset'],
            usd_prim=proto['usd_prim'],position_world_xyz=[float(x[k]),float(y[k]),height],
            rotation_matrix=rotation.tolist(),source_cell=[int(ix[k]),int(iz[k])],
            source_biome=str(biome[k]),maximum_sampled_basal_gap_m=-burial,
            sampled_burial_range_m=[burial,float(np.ptp(heights)+burial)],
            original_scale_preserved=True))
        counts[aid]=counts.get(aid,0)+1
    return dict(status='scanned_rock_candidates',placements=rows,counts=counts,rejected=rejected,
        camera_dependent=False,randomness_coordinate_frame='global_Minecraft_XZ_cells',
        ecology='lichen boulders on mountains; mossy stones in woodland; explicit biome lists',
        collision='decorative_only; final terrain collider remains authoritative; obstacle collision not yet qualified',
        qualification='not_run')
