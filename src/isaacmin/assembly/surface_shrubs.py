"""Original dry shrubs and tropical herbs in source-derived ecological patches."""
from pathlib import Path
import math
import numpy as np
from scipy import ndimage
from isaacmin.terrain.surface_navigation import SurfaceQuery,source_fields
from isaacmin.assembly.surface_population import random_cells,cluster,route_distances
from isaacmin.assembly.surface_canopy import embedded_woody_root
from isaacmin.io import read_json


def scatter_shrubs(terrain,prototypes,*,objects=None,route=None):
    q=SurfaceQuery(terrain);fields=source_fields(Path(terrain)/'material_fields.npz')
    cell=1.5;ox,oy,oz=q.origin
    ix,iz=np.meshgrid(np.arange(math.floor(q.xmin/cell),math.ceil(q.xmax/cell)),
        np.arange(math.floor(q.zmin/cell),math.ceil(q.zmax/cell)))
    ix,iz=ix.ravel(),iz.ravel()
    sx=(ix+random_cells(ix,iz,411))*cell;sz=(iz+random_cells(ix,iz,412))*cell
    x,y=sx-ox,-sz+oz;mx,mz=fields['min_xz']
    cx=np.floor(sx-mx).astype(int);cz=np.floor(sz-mz).astype(int)
    biome=fields['biome'][cz,cx]
    dry=np.isin(biome,['minecraft:desert','minecraft:badlands','minecraft:eroded_badlands',
        'minecraft:wooded_badlands','minecraft:savanna','minecraft:savanna_plateau','minecraft:windswept_savanna'])
    tropical=np.isin(biome,['minecraft:jungle','minecraft:sparse_jungle','minecraft:bamboo_jungle'])
    wet=fields['water_validity']&(fields['water_height']>=fields['height'])
    banks=ndimage.distance_transform_edt(~wet) if wet.any() else np.full(wet.shape,100.)
    patches=cluster(sx,sz,9.,414)
    density=.045*(.15+2.*patches**2)*(1+.8*np.exp(-banks[cz,cx]/12.))
    if tropical.any():
        if objects is None:raise ValueError('Tropical understory requires observed source canopy')
        canopy=np.load(Path(objects)/'canopy.npz')['cover_fraction']
        # Broadleaf ground herbs form colonies under broken tropical canopy.
        # These are authored ecological additions, not extracted source plants.
        shade=np.clip(canopy[cz,cx],0,1)
        colonies=cluster(sx,sz,5.,417)
        density=np.where(tropical,.55*(.08+2.5*colonies**3)*(.35+.65*shade),density)
    good=(dry|tropical)&~wet[cz,cx]&(random_cells(ix,iz,413)<np.minimum(density*cell*cell,1.))
    good&=(sx>=q.xmin+2)&(sx<=q.xmax-2)&(sz>=q.zmin+2)&(sz<=q.zmax-2)
    good&=route_distances(x,y,route,max_distance_m=3.)>=1.5
    if objects:
        from scipy.spatial import cKDTree
        trees=read_json(Path(objects)/'objects.json')['trees']
        if trees:
            roots=np.array([[t['source_x'],t['source_z']] for t in trees])
            good&=cKDTree(roots).query(np.column_stack((sx,sz)))[0]>.75
    pools={aid:sorted((p for p in prototypes if p['asset_id']==aid),key=lambda p:p['object_name'])
           for aid in ('wild_rooibos_bush','calathea_orbifolia_01')}
    for mask,aid in ((dry,'wild_rooibos_bush'),(tropical,'calathea_orbifolia_01')):
        if (good&mask).any() and not pools[aid]:raise ValueError('Original understory library unavailable: '+aid)
    rows=[];rejected={}
    for k in np.flatnonzero(good):
        dx=float(q.heights(x[k]+.5,y[k])-q.heights(x[k]-.5,y[k]))
        dy=float(q.heights(x[k],y[k]+.5)-q.heights(x[k],y[k]-.5))
        if not np.isfinite(dx+dy) or np.hypot(dx,dy)>math.tan(math.radians(30 if tropical[k] else 24)):
            rejected['slope']=rejected.get('slope',0)+1;continue
        pool=pools['calathea_orbifolia_01' if tropical[k] else 'wild_rooibos_bush']
        proto=pool[min(int(random_cells(ix[k],iz[k],415)*len(pool)),len(pool)-1)]
        yaw=360.*float(random_cells(ix[k],iz[k],416))
        support=embedded_woody_root(q,x[k],y[k],yaw,proto['contact_anchors_local_m'],proto['dimensions_m'][2],
            maximum_burial_m=proto['maximum_burial_m'],maximum_burial_fraction=proto['maximum_burial_fraction_of_height'])
        if support is None:
            rejected['root_footprint']=rejected.get('root_footprint',0)+1;continue
        height,root=support
        rows.append(dict(asset_id=proto['asset_id'],object_name=proto['object_name'],
            scene_asset=proto['scene_asset'],usd_prim=proto['usd_prim'],
            position_world_xyz=[float(x[k]),float(y[k]),height],yaw_degrees=yaw,
            source_cell=[int(ix[k]),int(iz[k])],source_biome=str(biome[k]),root_placement=root,
            original_scale_preserved=True))
    return dict(status='outdoor_understory_candidates',placements=rows,rejected=rejected,
        randomness_coordinate_frame='global Minecraft X,Z cells',camera_dependent=False,
        interpretation='Original Cape dry shrubs and tropical Calathea colonies as functional vegetation in fictional biomes; authored ecology, not exact regional botanical identity or source plant extraction',
        density_policy='Dry-shrub patches with modest bank enrichment; separate tropical colonies weighted by observed canopy. No water, trunk or steep-cliff placement',
        qualification='not_run',collision='decorative only; terrain supports ground contact')
