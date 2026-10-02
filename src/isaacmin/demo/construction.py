"""Publish light browser previews from completed, real construction stages."""
from pathlib import Path
import threading
import numpy as np

from isaacmin.io import read_json,atomic_json,sha256_file,utc_now
from isaacmin.security import redact

STAGES=[('source','Read Minecraft terrain'),('objects','Find trees and surface features'),
    ('terrain','Shape the natural landscape'),('route','Plan surface access'),
    ('blender','Export terrain'),('population','Place ground cover'),('assembly','Assemble trees, rocks and water'),
    ('collision','Build ground collision'),('capture','Capture ground views in Isaac'),
    ('overview','Capture aerial views in Isaac'),('delivery','Prepare your download')]


def surface_preview(build,output,final=False):
    from isaacmin.terrain.surface_navigation import source_fields
    record=read_json(build/'outdoor_build.json');source=read_json(build/'source/macro_surface.json')
    fields=source_fields(build/'source/macro_surface.npz')
    if final:
        terrain=read_json(build/'terrain/terrain.json');bounds=terrain['bounds_source_xz'];origin=terrain['origin_xyz']
        heights=np.load(build/'terrain/height.npy',mmap_mode='r')
        zs=np.linspace(0,heights.shape[0]-1,min(129,heights.shape[0]),dtype=int)
        xs=np.linspace(0,heights.shape[1]-1,min(129,heights.shape[1]),dtype=int)
        x=np.linspace(bounds[0],bounds[2],heights.shape[1])[xs]
        z=np.linspace(bounds[1],bounds[3],heights.shape[0])[zs]
        height=heights[np.ix_(zs,xs)].copy()
    else:
        extent=record['identity']['extent'];cx,cz=source['scope']['actual_center_xz']
        bounds=[cx-extent/2,cz-extent/2,cx+extent/2,cz+extent/2]
        origin=[record['identity']['center'][0],0,record['identity']['center'][1]]
        x=np.linspace(bounds[0]+.5,bounds[2]-.5,65);z=np.linspace(bounds[1]+.5,bounds[3]-.5,65)
        ix=np.clip((x-fields['min_xz'][0]).astype(int),0,fields['height'].shape[1]-1)
        iz=np.clip((z-fields['min_xz'][1]).astype(int),0,fields['height'].shape[0]-1)
        height=fields['height'][np.ix_(iz,ix)]-origin[1]
    xx,zz=np.meshgrid(x,z);shape=height.shape
    ix=np.clip((xx-fields['min_xz'][0]).astype(int),0,fields['height'].shape[1]-1)
    iz=np.clip((zz-fields['min_xz'][1]).astype(int),0,fields['height'].shape[0]-1)
    if 'validity' in fields and not fields['validity'][iz,ix].all():raise ValueError('Preview crop contains unknown source columns')
    biome=fields['biome'][iz,ix];substrate=fields['substrate'][iz,ix]
    color=np.empty((*shape,3),np.float32);color[:]=[.35,.49,.34]
    masks=[(np.char.find(biome,'badlands')>=0,[.67,.39,.26]),
        (np.char.find(biome,'desert')>=0,[.78,.64,.39]),
        (np.char.find(substrate,'stone')>=0,[.49,.51,.47]),
        ((np.char.find(substrate,'snow')>=0)|(np.char.find(substrate,'ice')>=0),[.77,.88,.91])]
    for mask,tint in masks:color[mask]=tint
    wet=fields['water_validity'][iz,ix]&(fields['water_height'][iz,ix]>=fields['height'][iz,ix])
    water=np.where(wet,fields['water_height'][iz,ix]-origin[1],np.nan)
    color[wet]=[.20,.46,.57]
    vertices=np.stack((xx-origin[0],-zz+origin[2],height),axis=-1)
    if not np.isfinite(vertices).all():raise ValueError('Preview source has uncovered/nonfinite terrain')
    name='terrain' if final else 'source'
    data=dict(status='actual_final_surface_sample' if final else 'actual_source_surface_sample',at_utc=utc_now(),
        rows=shape[0],columns=shape[1],positions=np.round(vertices.reshape(-1,3),4).tolist(),
        colors=np.round(color.reshape(-1,3),3).tolist(),water=np.round(water[np.isfinite(water)],3).tolist(),
        water_indices=np.flatnonzero(np.isfinite(water)).tolist(),bounds_source_xz=bounds,
        source_save_sha256=record['identity']['source'],
        input_manifest_sha256=sha256_file(build/('terrain/terrain.json' if final else 'source/macro_surface.json')),
        scope='Live construction diagram; sampled actual terrain, symbolic biome colors, not final Isaac material appearance')
    atomic_json(output/(name+'.json'),data)


def native_assets(root,build,output,stop):
    from isaacmin.process import run_worker
    import shutil
    worker=output/'export_construction_points.py'
    shutil.copyfile(root/'scripts/export_construction_points.py',worker)
    request=output/'assets_request.json';atomic_json(request,dict(scene=str(build/'assembled/scene/world.usda'),output=str(output)))
    def heartbeat():
        if stop.is_set():raise InterruptedError('Preview observer stopped')
    result=run_worker([str(root/'.venv/bin/python'),str(worker),'--request',str(request)],cwd=root,
        log_path=output/'assets_worker.log',timeout=180,estimated_memory_bytes=4*2**30,heartbeat=heartbeat,
        environment={'PYTHONPATH':str(root/'.tools/usd/lib/python')+':'+str(root/'src'),
            'LD_LIBRARY_PATH':str(root/'.tools/native/usr/lib/aarch64-linux-gnu')+':'+str(root/'.tools/native/usr/lib')})
    if result['exit_code']!=0:raise RuntimeError('Native preview sampling failed; world construction is unaffected')


class ConstructionWatch:
    """Optional read-only observer. Preview failure never changes world quality."""
    def __init__(self,root,job_id):
        self.root=Path(root);self.directory=self.root/'artifacts/demo/jobs'/job_id
        self.build=self.directory/'build';self.output=self.directory/'construction'
        self.stop=threading.Event();self.thread=None
    def __enter__(self):
        self.output.mkdir(parents=True,exist_ok=True)
        self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start();return self
    def __exit__(self,*_):
        self.stop.set();self.thread.join(timeout=20)
    def loop(self):
        attempted=set()
        while not self.stop.is_set():
            path=self.build/'outdoor_build.json'
            if path.is_file():
                stages=read_json(path).get('stages',{})
                for stage,name in [('source','source'),('terrain','terrain'),('collision','assets')]:
                    if self.stop.is_set():break
                    if stage not in stages or name in attempted or (self.output/(name+'.json')).is_file():continue
                    attempted.add(name)
                    try:
                        if name=='assets':native_assets(self.root,self.build,self.output,self.stop)
                        else:surface_preview(self.build,self.output,final=name=='terrain')
                    except Exception as error:
                        atomic_json(self.output/(name+'_error.json'),dict(status='preview_unavailable',at_utc=utc_now(),reason=redact(str(error)),
                            world_build_affected=False))
            self.stop.wait(2)


def progress(root,jobs,ident):
    root=Path(root);job=jobs.get(ident)
    if not job or job['action']!='convert':raise ValueError('Choose a conversion job')
    directory=root/'artifacts/demo/jobs'/ident;manifest=directory/'build/outdoor_build.json'
    record=read_json(manifest) if manifest.is_file() else {};stages=record.get('stages',{})
    completed=set(stages)
    if job['status']=='complete':completed.add('delivery')
    active=next((key for key,_ in STAGES if key not in completed),None)
    previews={};errors=[]
    for name in ('source','terrain','assets'):
        path=directory/'construction'/(name+'.json')
        if path.is_file():previews[name]='/construction/'+ident+'/'+name+'.json'
        error=directory/'construction'/(name+'_error.json')
        if error.is_file():errors.append(read_json(error))
    captured=[]
    for kind in ('preview','overview'):
        path=directory/'build/assembled'/kind/'preview_checkpoint.json'
        if path.is_file():
            report=read_json(path);captured.append(dict(kind=kind,frames=report.get('completed_frames',0),planned=report.get('planned_frames',0)))
    return dict(job=job,stages=[dict(id=key,label=label,status='complete' if key in completed else
        'running' if key==active and job['status']=='running' else 'pending') for key,label in STAGES],
        completed_stages=sum(k in completed for k,_ in STAGES),stage_count=len(STAGES),
        previews=previews,captures=captured,preview_errors=errors,
        placement_counts={name:stages.get(name,{}) for name in ('objects','population')},
        preview_scope='Completed real build data, animated as it arrives. Simplified display; final images come from Isaac.')
