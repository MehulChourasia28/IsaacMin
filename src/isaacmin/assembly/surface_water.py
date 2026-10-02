"""Source-level water with continuous, ground-clipped shorelines."""
from pathlib import Path
import numpy as np
from scipy import ndimage
from isaacmin.terrain.surface_navigation import SurfaceQuery,source_fields,sample_raster,grid_triangles


def _positive_triangles(points,faces,values):
    """Clip linear triangles against a signed continuous shoreline field."""
    tri=points[faces];f=values[faces];positive=f>0;count=positive.sum(axis=1)
    parts=[tri[count==3]]
    for number in (1,2):
        ids=np.flatnonzero(count==number)
        if not len(ids):continue
        t=tri[ids];v=f[ids];mask=positive[ids]
        # Rotate cyclically so vertex0 is the sole inside/outside vertex.
        start=np.argmax(mask if number==1 else ~mask,axis=1)
        order=(start[:,None]+np.arange(3))%3
        t=np.take_along_axis(t,order[:,:,None],axis=1);v=np.take_along_axis(v,order,axis=1)
        a=v[:,0]/(v[:,0]-v[:,1]);b=v[:,0]/(v[:,0]-v[:,2])
        ab=t[:,0]+a[:,None]*(t[:,1]-t[:,0]);ac=t[:,0]+b[:,None]*(t[:,2]-t[:,0])
        if number==1:parts.append(np.stack((t[:,0],ab,ac),axis=1))
        else:
            parts.append(np.stack((ab,t[:,1],t[:,2]),axis=1))
            parts.append(np.stack((ab,t[:,2],ac),axis=1))
    result=np.concatenate(parts)
    # Exact grid crossings can create zero-area boundary triangles.
    area=np.linalg.norm(np.cross(result[:,1]-result[:,0],result[:,2]-result[:,0]),axis=1)/2
    return result[area>1e-10],float(area[area>1e-10].sum())


def _water_patch(q,f,bounds,level,signed,ownership):
    xmin,zmin,xmax,zmax=bounds;spacing=q.spacing;ox,oy,oz=q.origin
    sx,sz=np.meshgrid(np.arange(xmin,xmax+spacing*.5,spacing),np.arange(zmin,zmax+spacing*.5,spacing))
    boundary=sample_raster(signed,sx,sz,f['min_xz'])
    if ownership is not None:boundary=np.minimum(boundary,sample_raster(ownership,sx,sz,f['min_xz']))
    ground=q.heights(sx-ox,-sz+oz);clearance=level-oy-ground
    values=np.minimum(boundary,np.where(np.isfinite(clearance),clearance,-1.))
    points=np.stack((sx-ox,-sz+oz,np.full(sx.shape,level-oy)),axis=-1).reshape(-1,3)
    from isaacmin.assembly.planar_water import compact_planar_water
    triangles,area,_=compact_planar_water(points.reshape(*sx.shape,3),values,_positive_triangles)
    return triangles,area


def build_surface_water(terrain,*,sink=None):
    q=SurfaceQuery(terrain);f=source_fields(Path(terrain)/'material_fields.npz')
    # Use the ground's own triangle lattice. Clipping a coarser water grid
    # misses intervening banks and can put visible water under dry terrain.
    spacing=q.spacing
    wet=f['water_validity']&(f['water_height']>=f['height'])
    mx,mz=f['min_xz'];ox,oy,oz=q.origin
    xx=np.arange(wet.shape[1])[None,:]+mx+.5;zz=np.arange(wet.shape[0])[:,None]+mz+.5
    inside=(xx>=q.xmin)&(xx<q.xmax)&(zz>=q.zmin)&(zz<q.zmax)
    visible=wet&inside;pieces=[];levels=[];triangle_count=0;part_count=0
    unique_levels=np.unique(f['water_height'][visible])
    nearest_level=None
    if len(unique_levels)>1:
        _,nearest=ndimage.distance_transform_edt(~wet,return_indices=True)
        nearest_level=f['water_height'][tuple(nearest)]
    rz,rx=np.nonzero(visible)
    center_clearance=f['water_height'][rz,rx]-oy-q.heights(mx+rx+.5-ox,-mz-rz-.5+oz)
    lost=center_clearance<=.015
    if lost.any() or not np.isfinite(center_clearance).all():
        raise ValueError('Final ground covers visible source water columns: '+str(int(lost.sum())))
    for level in unique_levels:
        mask=wet&(f['water_height']==level)
        rz,rx=np.nonzero(mask&inside)
        if not len(rx):continue
        xmin=max(q.xmin,mx+rx.min()-1);xmax=min(q.xmax,mx+rx.max()+2)
        zmin=max(q.zmin,mz+rz.min()-1);zmax=min(q.zmax,mz+rz.max()+2)
        signed=ndimage.distance_transform_edt(mask)-ndimage.distance_transform_edt(~mask)
        # The source boundary is the middle of two block columns. Natural banks
        # intersect the level plane nearer the dry column centre. Allow that
        # one-column collar, then let the exact ground define the wet edge.
        signed=ndimage.gaussian_filter(signed,.35,mode='nearest')+1.05
        ownership=None
        if nearest_level is not None:
            # Shore collars may extend into dry ground, never into another
            # source water level. Partition the collar by nearest source water.
            owns=nearest_level==level
            ownership=ndimage.distance_transform_edt(owns)-ndimage.distance_transform_edt(~owns)
        from isaacmin.terrain.surface_partitions import source_windows
        bounds=[xmin,zmin,xmax,zmax]
        windows=source_windows(bounds,256) if max(xmax-xmin,zmax-zmin)>512 else [bounds]
        level_count=0;area=0.
        for window in windows:
            # A bilinear shoreline sample is a convex combination of adjacent
            # source values. If this enclosing source stencil is nonpositive,
            # no fine triangle in the patch can be wet. No surface is omitted.
            a,b,c,d=window
            ix0=max(0,int(np.floor(a-mx-.5)));ix1=min(signed.shape[1],int(np.ceil(c-mx-.5))+1)
            iz0=max(0,int(np.floor(b-mz-.5)));iz1=min(signed.shape[0],int(np.ceil(d-mz-.5))+1)
            if signed[iz0:iz1,ix0:ix1].max()<=0:continue
            triangles,part_area=_water_patch(q,f,window,level,signed,ownership)
            if not len(triangles):continue
            level_count+=len(triangles);area+=part_area;triangle_count+=len(triangles)
            if sink:
                points=triangles.reshape(-1,3).astype(np.float32)
                faces=np.arange(len(points),dtype=np.int32).reshape(-1,3)
                sink(points,faces,part_count);part_count+=1
            else:pieces.append(triangles)
        if not level_count:
            raise ValueError('A visible source water level has no retained surface: '+str(level))
        _,components=ndimage.label(mask&inside)
        levels.append(dict(source_level_m=float(level),source_columns=len(rx),source_components=int(components),
            surface_area_m2=area,triangles=level_count))
    triangles=np.concatenate(pieces) if pieces else np.empty((0,3,3))
    points=triangles.reshape(-1,3).astype(np.float32)
    faces=np.arange(len(points),dtype=np.int32).reshape(-1,3)
    return points,faces,dict(status='source_water_mesh_authored',source_water_columns=int(visible.sum()),
        levels=levels,triangles=triangle_count,spacing_m=spacing,storage_parts=part_count if sink else 1,
        method='source level planes; continuous signed shoreline clipped against final terrain',
        planar_interior='Exact flat union retriangulated with all fine perimeter vertices; original clipped shoreline triangles retained',
        source_levels_unchanged=True,ground_intersection_margin_m=0.,
        maximum_source_shoreline_collar_m=1.05,
        multilevel_ownership='nearest source wet-column partition; no collar overlap into another water level',
        source_water_centers_above_final_ground=int(len(center_clearance)),
        minimum_source_center_depth_m=float(center_clearance.min()) if len(center_clearance) else None,
        qualification='not_run',flow_and_wave_dynamics='not_implemented',
        collision='water is not supporting ground; source-derived bed remains in terrain collider')
