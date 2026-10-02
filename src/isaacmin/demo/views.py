"""Bounded camera requests over an existing real world; no reconstruction or repair."""
from pathlib import Path
import math,re
import numpy as np

from isaacmin.io import read_json,sha256_file
from isaacmin.terrain.surface_navigation import SurfaceQuery,source_fields
from .catalog import presets


VIEW_SCHEMA={'type':'object','properties':{
    'kind':{'type':'string','enum':['ground','aerial']},
    'count':{'type':'integer','minimum':1,'maximum':6},
    'heading_degrees':{'type':'number','minimum':0,'maximum':360},
    'height_m':{'type':'number','minimum':20,'maximum':600},
    'source_xz':{'type':'array','items':{'type':'number'},'minItems':2,'maxItems':2}},
    'required':['kind'],'additionalProperties':False}


def validate_view(value):
    import jsonschema
    try:jsonschema.Draft202012Validator(VIEW_SCHEMA).validate(value)
    except jsonschema.ValidationError as error:raise ValueError('Invalid view request: '+error.message) from None
    result={**value,'count':int(value.get('count',1))}
    numbers=[value[k] for k in ('heading_degrees','height_m') if k in value]+value.get('source_xz',[])
    if not all(math.isfinite(v) for v in numbers):raise ValueError('Camera controls must be finite numbers')
    if value['kind']=='ground' and 'height_m' in value:raise ValueError('Ground cameras use the recorded 0.6 m robot height')
    return result


def resolve_target(root,jobs,preset,target='preset'):
    root=Path(root)
    if not isinstance(target,str) or not re.fullmatch(r'preset|latest|job_[a-f0-9]{32}',target):
        raise ValueError('Invalid view target')
    if target=='preset':
        from .library import require_saved_preset
        require_saved_preset(root,preset)
        return Path(presets(root)[preset]['build']),'preset_'+preset
    if target=='latest':
        target=jobs.latest_conversion(preset)
        if not target:raise ValueError('No completed new conversion exists for this preset')
    job=jobs.get(target)
    if not job or job['preset']!=preset or job['action']!='convert' or job['status']!='complete':
        raise ValueError('Choose the preset or a completed conversion of this same world')
    return root/'artifacts/demo/jobs'/target/'build',target


def object_directory(root,build):
    path=build/'objects'
    if path.is_dir():return path
    # Derived material-only builds can retain their original terrain/objects.
    terrain=read_json(build/'terrain/terrain.json')
    original=Path(terrain['source_surface']['path']).parent.parent
    if original.resolve().is_relative_to(Path(root).resolve()) and (original/'outdoor_build.json').is_file():
        if read_json(original/'outdoor_build.json')['identity']['source']==read_json(build/'outdoor_build.json')['identity']['source']:
            if (original/'objects').is_dir():return original/'objects'
    return None


def plan_views(root,build,options,seed):
    options=validate_view(options);query=SurfaceQuery(build/'terrain')
    origin=query.origin;count=options['count'];span=max(query.xmax-query.xmin,query.zmax-query.zmin)
    objects=object_directory(root,build);heading=options.get('heading_degrees')
    point=options.get('source_xz')
    if point and not (query.xmin+2<=point[0]<query.xmax-2 and query.zmin+2<=point[1]<query.zmax-2):
        raise ValueError('Requested location is outside the built region; use a point at least 2 m inside its bounds')
    def direction(degrees):
        angle=math.radians(degrees);return np.array([math.sin(angle),math.cos(angle)])
    if options['kind']=='ground':
        if point:
            fields=source_fields(build/'terrain/material_fields.npz');x,z=point;wx,wy=x-origin[0],-z+origin[2]
            height=float(query.heights(wx,wy));mx,mz=fields['min_xz'];rx,rz=int(math.floor(x-mx)),int(math.floor(z-mz))
            if fields['water_validity'][rz,rx] and fields['water_height'][rz,rx]>=height+origin[1]:
                raise ValueError('Requested ground camera is in exposed water; choose an aerial or a dry bank')
            slope=math.hypot(float(query.heights(wx+.5,wy)-query.heights(wx-.5,wy)),float(query.heights(wx,wy+.5)-query.heights(wx,wy-.5)))
            if slope>math.tan(math.radians(25)):raise ValueError('Requested ground camera is on terrain steeper than 25 degrees')
            cover=0.
            if objects:
                trees=read_json(objects/'objects.json')['trees']
                if any(math.hypot(t['source_x']-x,t['source_z']-z)<1.75 for t in trees):raise ValueError('Requested ground camera intersects a tree clearance area')
                canopy=np.load(objects/'canopy.npz')['cover_fraction'];cover=float(np.clip((canopy[rz,rx]-.15)/.45,0,1));cover=cover*cover*(3-2*cover)
            ev=13.747247562465875-4.5*cover
            poses=[dict(kind='static',position=[wx,wy,height+.6],look_at=[wx,wy+8,height+.4],
                scout_height_m=.6,held_out=False,surface_scope='outdoor',source_biome=str(fields['biome'][rz,rx]),
                camera_response=dict(ev100=ev,f_number=2.8 if ev<11 else 8.,iso=100)) for _ in range(count)]
        else:
            from isaacmin.outdoor_pipeline import outdoor_camera_poses
            poses=outdoor_camera_poses(build/'terrain',objects,seed=seed,maximum=count+2)
            poses=[p for p in poses if p.get('name') not in ('source_ice_spire_ground','source_water_shore')][:count]
            if len(poses)<count:raise ValueError('Insufficient distinct supported ground locations in this region')
        for index,pose in enumerate(poses):
            if heading is not None or point:
                vector=direction((heading or 0)+index*360/count);x,y,z=pose['position']
                pose['look_at']=[x+float(vector[0])*8,y+float(vector[1])*8,z-.2]
            pose.update(name=f'requested_ground_{seed:x}_{index+1}',requested_view=True)
        return poses
    x,z=point or [(query.xmin+query.xmax)/2,(query.zmin+query.zmax)/2]
    wx,wy=x-origin[0],-z+origin[2];height=float(query.heights(wx,wy))
    altitude=float(options.get('height_m',span*.8));distance=min(span*.65,altitude*1.1)
    first=float(heading if heading is not None else seed%360);poses=[]
    for index in range(count):
        vector=direction(first+index*360/count);camera=np.array([wx,wy])-distance*vector
        support=float(query.heights(*camera))
        if not math.isfinite(support):support=float(np.max(query.height))
        poses.append(dict(kind='aerial_overview',name=f'requested_aerial_{seed:x}_{index+1}',
            position=[float(camera[0]),float(camera[1]),max(height,support)+altitude],look_at=[wx,wy,height],
            overview_only=True,held_out=False,requested_view=True,requested_height_m=altitude,
            source_bounds_xz=list(query.record['bounds_source_xz']),
            scope='user requested actual existing-world aerial; not robot-height qualification'))
    return poses


def completed_views(root,target):
    results=[]
    for path in sorted((Path(root)/'artifacts/demo/views'/target).glob('*/*/render.json')):
        report=read_json(path)
        if report.get('status')=='actual_native_capture_complete':results.append(path.parent)
    return results
