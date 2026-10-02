"""Deterministic dry route candidates on the final outdoor triangle surface."""
from pathlib import Path
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from isaacmin.io import read_json,sha256_file
from isaacmin.terrain.surface_navigation import SurfaceQuery,source_fields
from isaacmin.terrain.routes import route_candidate


def plan_outdoor_route(terrain,objects):
    q=SurfaceQuery(terrain);fields=source_fields(Path(terrain)/'material_fields.npz')
    ox,oy,oz=q.origin
    sx,sz=np.meshgrid(np.arange(q.xmin+.5,q.xmax,1.),np.arange(q.zmin+.5,q.zmax,1.))
    x,y=sx-ox,-sz+oz;h=q.heights(x,y)
    dx=q.heights(x+.25,y)-q.heights(x-.25,y)
    dy=q.heights(x,y+.25)-q.heights(x,y-.25)
    safe=np.isfinite(h)&np.isfinite(dx)&np.isfinite(dy)&(np.hypot(dx,dy)/.5<=.35)
    safe&=(sx>=q.xmin+6)&(sx<q.xmax-6)&(sz>=q.zmin+6)&(sz<q.zmax-6)
    ix=np.floor(sx-fields['min_xz'][0]).astype(int);iz=np.floor(sz-fields['min_xz'][1]).astype(int)
    wet=fields['water_validity']&(fields['water_height']>=fields['height'])
    banks=ndimage.distance_transform_edt(~wet) if wet.any() else np.full(wet.shape,100.)
    safe&=banks[iz,ix]>2.
    trees=read_json(Path(objects)/'objects.json')['trees']
    if trees:
        roots=np.array([[t['source_x'],t['source_z']] for t in trees])
        safe&=cKDTree(roots).query(np.column_stack((sx.ravel(),sz.ravel())))[0].reshape(safe.shape)>1.5
    labels,count=ndimage.label(safe)
    sizes=np.bincount(labels.ravel());sizes[0]=0
    largest=int(np.argmax(sizes)) if count else 0
    if count:safe&=labels==largest
    result=route_candidate(h,safe,wet[iz,ix],spacing_m=1.,
        centre_index=(h.shape[0]//2,h.shape[1]//2),origin_xz=(q.xmin+.5,q.zmin+.5),
        world_origin=q.origin,max_grade=.35)
    result.update(ground_sha256=sha256_file(Path(terrain)/'vertices.npy'),
        source='final reconstructed triangle support, observed water and source tree roots',
        world_origin_source_xyz=q.origin,source_sampling_m=1.,nearby_grade_baseline_m=.5,
        actual_traversal='not_run',navigation_stack='not_run_not_supplied',
        authored_corridor='low vegetation and decorative rocks excluded by constructor; no extracted-trail claim',
        start_policy='nearest center sample within largest four-connected dry support component',
        dry_components=count,selected_component_area_m2=int(sizes[largest]) if count else 0,
        qualification='not_run',producer_sha256=sha256_file(Path(__file__)))
    result['limitations']=['Grid samples of final triangles do not establish continuous body clearance',
        'Candidate length is planned unique coverage, not measured traversal or navigation success',
        'No trail wear or terrain geometry is authored by this planner']
    if result.get('unique_polyline_length_m',0)<2:
        result.update(status='blocked',reason='Insufficient connected dry support for a usable route')
    return result
