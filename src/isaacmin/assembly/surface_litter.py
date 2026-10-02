"""Source-canopy litter, using original meshes and measured final ground support."""
from pathlib import Path
import math
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from isaacmin.io import atomic_json, read_json, sha256_file
from isaacmin.terrain.surface_navigation import SurfaceQuery, source_fields
from isaacmin.assembly.surface_population import random_cells, cluster, route_distances


FORESTS = ('forest', 'flower_forest', 'birch_forest', 'old_growth_birch_forest',
           'dark_forest', 'taiga', 'old_growth_pine_taiga', 'old_growth_spruce_taiga',
           'grove', 'jungle', 'sparse_jungle', 'bamboo_jungle', 'swamp', 'mangrove_swamp')


def scatter_litter(terrain, objects, prototypes, output, *, route=None,_bounds=None,_context=None):
    """Stable metric cells, canopy-driven patches, and no camera-conditioned density.

    Leaves belong near observed oak/dark-oak trees. Woody debris has broader
    forest scope. No leaf-species substitution is inferred for tropical forests.
    Large debris is kept outside the entire route corridor plus its footprint.
    """
    terrain, objects, output = map(Path, (terrain, objects, output))
    output.mkdir(parents=True, exist_ok=False)
    if _context is None:
        q = SurfaceQuery(terrain); fields = source_fields(terrain/'material_fields.npz')
        q.ground_file_sha256=sha256_file(terrain/'vertices.npy')
        canopy = np.load(objects/'canopy.npz')['cover_fraction']
        trees = read_json(objects/'objects.json')['trees']
        oak = np.array([[t['source_x'], t['source_z']] for t in trees
                        if t['species'] in ('oak', 'dark_oak')], float).reshape(-1, 2)
        oak_tree = cKDTree(oak) if len(oak) else None
    else:q,fields,canopy,oak_tree=_context
    from isaacmin.terrain.surface_partitions import source_windows,source_cell_grid
    full_bounds=[q.xmin,q.zmin,q.xmax,q.zmax]
    if _bounds is None and max(q.xmax-q.xmin,q.zmax-q.zmin)>512:
        parts=[];counts={};rejected={};extrema={}
        for i,bounds in enumerate(source_windows(full_bounds)):
            directory=output/f'Part{i:04d}'
            record=scatter_litter(terrain,objects,prototypes,directory,route=route,
                _bounds=bounds,_context=(q,fields,canopy,oak_tree))
            parts.append(dict(path=str(directory.relative_to(output)),instances=record['instances'],
                bounds_source_xz=bounds,placements_sha256=sha256_file(directory/'placements.npz')))
            for k,v in record['counts'].items():counts[k]=counts.get(k,0)+v
            for k,v in record['rejected'].items():rejected[k]=rejected.get(k,0)+v
            for k,v in record['maximum_accepted_basal_spread_m'].items():extrema[k]=max(extrema.get(k,0),v)
            atomic_json(output/'progress.json',dict(completed_parts=len(parts),instances=sum(p['instances'] for p in parts)))
        record.update(parts=parts,instances=sum(p['instances'] for p in parts),counts=counts,rejected=rejected,
            maximum_accepted_basal_spread_m=extrema,partition_policy='Unique global cells; common final surface and source ecology')
        atomic_json(output/'litter.json',record);return record
    bounds=_bounds or full_bounds
    mx, mz = fields['min_xz']; ox, _, oz = q.origin
    positions=[]; rotations=[]; ids=[]; cells=[]; counts={}; rejected={}
    # Density is objects per square metre before terrain/route support rejection.
    recipes=[('dry_branches_medium_01', 1., .16, .6),
             ('dead_tree_trunk', 4., .002, 1.), ('rock_09', .65, .6, .25),
             ('LeafSet012', .22, 8., .08), ('LeafSet030', .22, 8., .08)]
    support_extrema={}
    for layer, (aid, cell, density, margin) in enumerate(recipes):
        pool=[i for i,p in enumerate(prototypes) if p['asset_id']==aid]
        if not pool:
            rejected['missing:'+aid]=1; continue
        channel=500+20*layer
        ix,iz=source_cell_grid(bounds,full_bounds,cell)
        ix,iz=ix.ravel(),iz.ravel()
        sx=(ix+random_cells(ix,iz,channel))*cell
        sz=(iz+random_cells(ix,iz,channel+1))*cell
        x,y=sx-ox,-sz+oz
        cx,cz=np.floor(sx-mx).astype(int),np.floor(sz-mz).astype(int)
        names=fields['block_names'][fields['substrate_id'][cz,cx]].astype(str)
        soil=np.zeros(len(sx),bool)
        for token in ('grass_block','dirt','podzol','moss_block','mud'):
            soil |= np.char.find(names,token)>=0
        dry=~(fields['water_validity'][cz,cx] & (fields['water_height'][cz,cx]>=fields['height'][cz,cx]))
        forest=np.isin(fields['biome'][cz,cx],['minecraft:'+b for b in FORESTS])
        cover=canopy[cz,cx]
        patch=cluster(sx,sz,5.,channel+2)
        eligible=forest & soil & dry & (cover>.1)
        leaf=aid.startswith('LeafSet')
        if leaf:
            # Explicitly scoped to source oaks; no maple/birch/jungle relabelling.
            oak_distance=oak_tree.query(np.column_stack((sx,sz)))[0] if oak_tree else np.full(len(sx),np.inf)
            eligible &= oak_distance<9.
            rate=density*(.25+1.5*cover)*(.15+1.6*patch**2)*np.clip((9.-oak_distance)/4.,0,1)
        else:
            rate=density*(.4+cover)*(.25+1.5*patch)
        eligible &= random_cells(ix,iz,channel+3)<np.minimum(.95,rate*cell*cell)
        eligible &= (sx>q.xmin+2)&(sx<q.xmax-2)&(sz>q.zmin+2)&(sz<q.zmax-2)
        selected=np.flatnonzero(eligible)
        if not len(selected):counts[aid]=0;continue
        ix,iz,x,y=[a[selected] for a in (ix,iz,x,y)]
        pick=np.minimum((random_cells(ix,iz,channel+4)*len(pool)).astype(int),len(pool)-1)
        proto_ids=np.asarray(pool)[pick]
        # A small leaf rests on the local mesh tangent; long debris uses a
        # broader gradient and is independently checked across its original base.
        radius=.08 if leaf or aid=='rock_09' else .4
        normals=np.column_stack((-(q.heights(x+radius,y)-q.heights(x-radius,y))/(2*radius),
                                 -(q.heights(x,y+radius)-q.heights(x,y-radius))/(2*radius),np.ones(len(x))))
        normals/=np.linalg.norm(normals,axis=1,keepdims=True)
        yaw=2*np.pi*random_cells(ix,iz,channel+5)
        tangent=np.column_stack((np.cos(yaw),np.sin(yaw),np.zeros(len(x))))
        tangent-=normals*np.sum(tangent*normals,axis=1,keepdims=True)
        tangent/=np.linalg.norm(tangent,axis=1,keepdims=True)
        rotation=np.stack((tangent,np.cross(normals,tangent),normals),axis=-1)
        keep=np.isfinite(normals).all(axis=1)&(normals[:,2]>=math.cos(math.radians(32)))
        route_distance=route_distances(x,y,route,max_distance_m=3.)
        height=np.zeros(len(x));accepted_spread=[]
        for index in pool:
            proto=prototypes[index]; rows=np.flatnonzero(keep&(proto_ids==index))
            if not len(rows):continue
            anchors=np.asarray(proto['contact_anchors_local_m'],float)
            if not len(anchors):raise ValueError('Litter requires measured basal anchors')
            footprint=np.linalg.norm(anchors[:,:2],axis=1).max()
            keep[rows] &= route_distance[rows]>=footprint+margin
            minimum=np.full(len(rows),np.inf);maximum=-minimum
            for anchor in anchors:
                a=np.einsum('nij,j->ni',rotation[rows],anchor)
                support=q.heights(x[rows]+a[:,0],y[rows]+a[:,1])-a[:,2]
                minimum=np.minimum(minimum,support);maximum=np.maximum(maximum,support)
            spread=maximum-minimum
            # Thin laminae may touch at their lowest point with <=4mm basal gap.
            # Volumetric twigs/stones embed their lowest basal envelope in soil.
            limit=.004 if leaf else min(.09, .45*proto['dimensions_m'][2])
            good=np.isfinite(spread)&(spread<=limit)
            keep[rows]&=good
            height[rows]=(maximum+.0003 if leaf else minimum-.001)
            accepted_spread.extend(spread[good&keep[rows]].tolist())
        rejected[aid]=int((~keep).sum());counts[aid]=int(keep.sum())
        support_extrema[aid]=max(accepted_spread,default=0.)
        positions.append(np.column_stack((x[keep],y[keep],height[keep])))
        rotations.append(Rotation.from_matrix(rotation[keep]).as_quat())
        ids.append(proto_ids[keep]);cells.append(np.column_stack((ix[keep],iz[keep])))
    p=np.concatenate(positions).astype(np.float32) if positions else np.empty((0,3),np.float32)
    r=np.concatenate(rotations).astype(np.float32) if rotations else np.empty((0,4),np.float32)
    i=np.concatenate(ids).astype(np.int32) if ids else np.empty(0,np.int32)
    c=np.concatenate(cells).astype(np.int64) if cells else np.empty((0,2),np.int64)
    np.savez_compressed(output/'placements.npz',positions=p,orientations_xyzw=r,prototype_indices=i,source_cells=c)
    record=dict(status='original_forest_litter_placed',instances=len(p),counts=counts,rejected=rejected,
        maximum_accepted_basal_spread_m=support_extrema,prototypes=prototypes,
        seed_frame='global source XZ cells; no map name, camera or local origin seed',
        leaf_scope='Brown oak scans within9m of actual source oak/dark-oak trees; no tropical leaf-species claim',
        ecology='dry forest soil, canopy-conditioned clusters, source water/ice/snow excluded',
        collision='decorative only; larger debris excluded from authored route by measured footprint',
        ground_sha256=q.ground_file_sha256,camera_dependent=False,
        qualification='not_run',producer_sha256=sha256_file(Path(__file__)))
    atomic_json(output/'litter.json',record)
    return record
