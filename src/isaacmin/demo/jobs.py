"""Durable, idempotent demo queue with a single process owner."""
from pathlib import Path
import json,sqlite3,uuid,fcntl,threading,time
from isaacmin.io import read_json,atomic_json,utc_now,sha256_file,hash_object
from isaacmin.security import redact
from .catalog import presets,refresh

ACTIONS=('verify','prepare','convert','agent','views')


class Jobs:
    def __init__(self,root):
        self.root=Path(root).resolve();self.path=self.root/'state/demo.sqlite'
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.connect() as db:
            db.execute('PRAGMA journal_mode=WAL')
            db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY,operation_key TEXT UNIQUE,preset TEXT,action TEXT,status TEXT,payload TEXT,result TEXT,error TEXT,created TEXT,updated TEXT,attempts INTEGER DEFAULT 0)')
            db.execute('CREATE TABLE IF NOT EXISTS controls (operation_key TEXT PRIMARY KEY,job_id TEXT,action TEXT,result TEXT)')
    def connect(self):
        db=sqlite3.connect(self.path,timeout=15);db.row_factory=sqlite3.Row;return db
    def submit(self,preset,action,operation_key,*,prompt='',view=None,target='preset',context_job=None):
        if preset not in presets(self.root) or action not in ACTIONS:raise ValueError('Unknown preset or action')
        from .library import require_saved_preset,visibility
        if action in ('verify','prepare'):require_saved_preset(self.root,preset)
        if not isinstance(operation_key,str) or not 8<=len(operation_key)<=100:raise ValueError('Invalid operation key')
        if not isinstance(prompt,str) or len(prompt)>3000 or (action=='agent' and not prompt.strip()):raise ValueError('Invalid agent request')
        from .views import validate_view
        options=validate_view(view) if action=='views' else None
        if action!='views' and (view is not None or target!='preset'):raise ValueError('Camera controls apply only to view jobs')
        request=dict(preset=preset,action=action,prompt=redact(prompt),view=options,target=target)
        if context_job is not None:
            context=self.get(context_job) if isinstance(context_job,str) else None
            if action!='agent' or not context or context['action']!='convert' or context['preset']!=preset:
                raise ValueError('Assistant context must be a conversion of the selected world')
            request['context_job']=context_job
        with self.connect() as db:existing=db.execute('SELECT * FROM jobs WHERE operation_key=?',(operation_key,)).fetchone()
        if existing and existing['id'] in visibility(self.root)['archived_jobs']:
            raise ValueError('This operation was archived when the library was cleared; create a new request')
        if existing and 'request' in json.loads(existing['payload']):
            if json.loads(existing['payload'])['request']!=request:raise ValueError('Operation key already belongs to another request')
            return self.public(dict(existing))
        data=dict(prompt=redact(prompt),request=request)
        if context_job is not None:data['context_job']=context_job
        if action=='views':
            from .views import resolve_target,plan_views
            build,key=resolve_target(self.root,self,preset,target)
            record=read_json(build/'outdoor_build.json')
            seed=int(hash_object(dict(operation_key=operation_key,preset=preset))[:8],16)
            planned=plan_views(self.root,build,options,seed)
            capture_settings=build/'assembled/preview_request.json'
            if capture_settings.is_file():
                from isaacmin.assembly.outdoor_lighting import daylight_camera_response
                settings=read_json(capture_settings)
                planned=[daylight_camera_response(pose,settings) for pose in planned]
            data.update(view=options,target=key,build=str(build),poses=planned,
                build_manifest_sha256=sha256_file(build/'outdoor_build.json'),scene_sha256=sha256_file(Path(record['scene'])))
        payload=json.dumps(data,sort_keys=True);now=utc_now()
        with self.connect() as db:
            db.execute('INSERT OR IGNORE INTO jobs (id,operation_key,preset,action,status,payload,created,updated) VALUES (?,?,?,?,?,?,?,?)',
                ('job_'+uuid.uuid4().hex,operation_key,preset,action,'queued',payload,now,now))
            row=dict(db.execute('SELECT * FROM jobs WHERE operation_key=?',(operation_key,)).fetchone())
            saved=json.loads(row['payload'])
            same=(saved['request']==request) if 'request' in saved else (saved=={k:v for k,v in data.items() if k!='request'})
            if (row['preset'],row['action'])!=(preset,action) or not same:raise ValueError('Operation key already belongs to another request')
            return self.public(row)
    def public(self,row):
        d={k:row[k] for k in ('id','preset','action','status','error','created','updated','attempts')}
        d['result']=json.loads(row['result']) if row['result'] else None
        # Full tool traces stay in the durable checkpoint, not recursively inside
        # later list-jobs responses or the browser's polling payload.
        if isinstance(d['result'],dict):d['result'].pop('tools',None)
        if row['action']=='convert':d['build_url']='/?job='+row['id']
        if row['action']=='agent':
            payload=json.loads(row['payload']);d['prompt']=payload['prompt'];d['context_job']=payload.get('context_job')
        if row['action']=='views':
            payload=json.loads(row['payload']);d['view_request']=payload['view'];d['target']=payload['target']
            if row['status']=='running':d['progress']='Capturing requested views in Isaac Sim'
            if row['status']=='complete':d['view_url']=('/library?preset='+row['preset'] if payload['target'].startswith('preset_') else '/results/'+payload['target'])
        if row['action']=='convert' and row['status']=='complete':
            d['view_url']='/results/'+row['id']
            if (self.root/'artifacts/demo/downloads'/(row['id']+'.json')).is_file():d['download_url']='/download/'+row['id']
        if row['action']=='convert' and row['status']=='running':
            path=self.root/'artifacts/demo/jobs'/row['id']/'build/outdoor_build.json'
            stages=read_json(path).get('stages',{}) if path.is_file() else {}
            phase=next((label for key,label in (
                ('source','Reading the immutable Minecraft save'),('objects','Reading trees and surface features'),
                ('terrain','Reconstructing natural terrain'),('route','Planning surface access'),
                ('blender','Exporting terrain to USD'),('population','Placing ground cover'),
                ('assembly','Assembling the natural world'),('collision','Building ground collision'),
                ('capture','Capturing ground views in Isaac Sim'),('overview','Capturing aerial views in Isaac Sim'))
                if key not in stages),'Preparing the portable download')
            d['progress']=phase
        if row['action']=='convert':
            from .pipeline_agent import public_progress
            d['pipeline_agent']=public_progress(self.root,row['id'])
        return d
    def list(self):
        from .library import visibility
        archived=set(visibility(self.root)['archived_jobs'])
        with self.connect() as db:return [self.public(dict(r)) for r in db.execute('SELECT * FROM jobs ORDER BY created DESC') if r['id'] not in archived][:60]
    def get(self,ident):
        from .library import visibility
        if ident in visibility(self.root)['archived_jobs']:return None
        with self.connect() as db:row=db.execute('SELECT * FROM jobs WHERE id=?',(ident,)).fetchone()
        return self.public(dict(row)) if row else None
    def latest_conversion(self,preset):
        from .library import visibility
        archived=set(visibility(self.root)['archived_jobs'])
        with self.connect() as db:
            row=next((r for r in db.execute("SELECT id FROM jobs WHERE preset=? AND action='convert' AND status='complete' ORDER BY created DESC",(preset,)) if r['id'] not in archived),None)
        return row['id'] if row else None
    def worker_status(self):
        result={}
        for lane,name in [('work','demo_worker.lock'),('agent','demo_agent_worker.lock')]:
            path=self.root/'state'/name
            result[lane]='stopped'
            if not path.is_file():continue
            with path.open('r') as handle:
                try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:result[lane]='available'
                else:fcntl.flock(handle,fcntl.LOCK_UN)
        return result
    def control(self,ident,action,operation_key=None):
        if action not in ('resume','cancel'):raise ValueError('Unknown job control')
        if not isinstance(ident,str):raise ValueError('Invalid job identifier')
        from .library import visibility
        if ident in visibility(self.root)['archived_jobs']:raise ValueError('This job has been archived')
        if operation_key is not None and (not isinstance(operation_key,str) or not 8<=len(operation_key)<=100):
            raise ValueError('Invalid operation key')
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            previous=db.execute('SELECT * FROM controls WHERE operation_key=?',(operation_key,)).fetchone() if operation_key else None
            if previous:
                if (previous['job_id'],previous['action'])!=(ident,action):raise ValueError('Operation key already belongs to another control')
                return json.loads(previous['result'])
            row=db.execute('SELECT * FROM jobs WHERE id=?',(ident,)).fetchone()
            if row is None:raise ValueError('Unknown job')
            if action=='resume':
                if row['status'] not in ('interrupted','failed','cancelled'):raise ValueError('Only interrupted, failed or cancelled jobs can resume')
                if row['attempts']>=3:raise ValueError('Three job attempts exhausted; evidence retained')
                status='queued'
            else:
                if row['status']!='queued':raise ValueError('Only queued jobs can be cancelled; running native work is preserved')
                status='cancelled'
            db.execute('UPDATE jobs SET status=?,error=NULL,updated=? WHERE id=?',(status,utc_now(),ident))
            result=dict(id=ident,status=status,action=action)
            if operation_key:db.execute('INSERT INTO controls VALUES (?,?,?,?)',(operation_key,ident,action,json.dumps(result)))
            return result
    def resume(self,ident,operation_key=None):
        return self.control(ident,'resume',operation_key)
    def finish(self,ident,status,*,result=None,error=None):
        with self.connect() as db:db.execute('UPDATE jobs SET status=?,result=?,error=?,updated=? WHERE id=?',
            (status,json.dumps(redact(result)) if result else None,redact(error),utc_now(),ident))
    def execute(self,row):
        p=presets(self.root)[row['preset']];build=Path(p['build']);record=read_json(build/'outdoor_build.json')
        if row['action']=='agent':
            from .planner import run
            payload=json.loads(row['payload'])
            return run(self.root,self,row['preset'],payload['prompt'],row['id'],context_job=payload.get('context_job'))
        if row['action']=='views':
            from isaacmin.outdoor_render import render_outdoor,validate_capture_result
            payload=json.loads(row['payload']);build=Path(payload['build']);record=read_json(build/'outdoor_build.json')
            if sha256_file(build/'outdoor_build.json')!=payload['build_manifest_sha256'] or sha256_file(Path(record['scene']))!=payload['scene_sha256']:
                raise ValueError('Selected world changed after the view request; create a new request')
            directory=self.root/'artifacts/demo/views'/payload['target']/row['id']
            completed=None
            for receipt in sorted(directory.glob('*/render.json')):
                previous=read_json(receipt)
                if previous.get('status')=='actual_native_capture_complete' and previous['build_manifest_sha256']==payload['build_manifest_sha256']:
                    validate_capture_result(Path(previous['capture_result']),payload['scene_sha256'],payload['poses'])
                    completed=previous;break
            if completed is None:
                completed=render_outdoor(self.root,build,directory/('attempt_'+str(row['attempts']+1)),views='requested',poses=payload['poses'])
            refresh(self.root)
            return dict(status='ready',frames=len(payload['poses']),capture=completed['capture_result'],
                geometry_rebuild=False,qualification='not_inferred',message=f"{len(payload['poses'])} new {payload['view']['kind']} views are ready. Existing world geometry and materials preserved.")
        if row['action']=='verify':
            from isaacmin.packaging import ascii_dependency_closure
            closure=ascii_dependency_closure(Path(record['scene']))
            if closure['status']!='pass':raise ValueError('Scene dependency integrity failed')
            refresh(self.root)
            return dict(status='verified',files=len(closure['files']),message='Native scene dependencies and displayed captures match their recorded hashes.',qualification='not_inferred')
        if row['action']=='convert':
            from .construction import ConstructionWatch
            with ConstructionWatch(self.root,row['id']):return self.convert(row,p)
        from isaacmin.outdoor_packaging import package_outdoor
        scene=Path(record['scene']);identity=sha256_file(scene.parent/'native_dependency_closure.json')[:16]
        package=self.root/'packages'/('demo_'+row['preset']+'_'+identity)
        capture=self.root/p.get('overview_capture','artifacts/demo/captures/'+row['preset']+'/overview_1')
        receipt=capture/'render.json'
        if not receipt.is_file() or read_json(receipt)['status']!='actual_native_capture_complete':
            raise ValueError('The matching aerial capture is still pending; retry preparation after capture completes')
        if not (package/'package.json').is_file():package_outdoor(self.root,build,output=package,reference_capture=capture)
        from .archive import prepare_archive
        delivery=prepare_archive(self.root,package,scene,row['preset']);refresh(self.root)
        return dict(status='ready',download='/download/'+row['preset'],bytes=delivery['bytes'],message='Portable USD world and original local dependencies are ready.')
    def convert(self,row,p):
        from isaacmin.outdoor_pipeline import build_outdoor
        from isaacmin.outdoor_packaging import package_outdoor
        from .archive import prepare_archive
        from .pipeline_agent import PipelineAgent
        agent=PipelineAgent(self.root,row['id'])
        registry={x['id']:x for x in read_json(self.root/'state/source_worlds.json')['worlds']}
        source=registry[row['preset']]
        build=self.root/'artifacts/demo/jobs'/row['id']/'build';manifest=build/'outdoor_build.json'
        if not manifest.is_file() or read_json(manifest).get('status')!='geometry_world':
            build_outdoor(self.root,self.root/source['source'],extent=256,center=source['center_minecraft_xz'],
                output=build,asset_library=self.root/p['asset_library'],capture_set='full',stage_controller=agent)
        # Delivery resumes from completed native construction without
        # rebuilding it against a later save or another recipe version.
        record=read_json(manifest);scene=Path(record['scene'])
        if not scene.resolve().is_relative_to(build):raise ValueError('Conversion scene is outside its job')
        for kind in ('preview','overview'):
            capture=read_json(build/'assembled'/kind/'preview_result.json')
            if capture['status']!='actual_visual_preview_complete' or capture['scene_sha256']!=sha256_file(scene):
                raise ValueError('Conversion requires matching completed ground and aerial captures')
        package=self.root/'packages'/('demo_conversion_'+row['id'])
        delivered=agent.state['steps'].get('delivery',{}).get('status')=='complete'
        if not delivered:agent.before('delivery',record)
        if not (package/'package.json').is_file():package_outdoor(self.root,build,output=package)
        delivery=prepare_archive(self.root,package,scene,row['id'])
        receipt=dict(archive_sha256=delivery['sha256'],bytes=delivery['bytes'],scene_sha256=delivery['scene_sha256'])
        if not delivered:agent.completed('delivery',receipt)
        elif agent.state['steps']['delivery']['receipt_sha256']!=hash_object(receipt):
            raise ValueError('Completed agent delivery changed')
        return dict(status='ready',build=str(build),source=record['source_world'],
            source_save_sha256=record['identity']['source'],download='/download/'+row['id'],bytes=delivery['bytes'],
            pipeline_agent=agent.state['model'],agent_harness=agent.state.get('harness','custom'),agent_dispatches=len(agent.state['steps']),
            qualification='not_inferred',message='Agent conversion, Isaac gallery and portable download are ready.')

    def loop(self,stop,lane='work'):
        # One native job and one hosted request can progress independently.
        # Only the owner of a lane may recover its interrupted jobs.
        if lane not in ('work','agent'):raise ValueError('Unknown worker lane')
        predicate="action='agent'" if lane=='agent' else "action!='agent'"
        lock_name='demo_agent_worker.lock' if lane=='agent' else 'demo_worker.lock'
        with (self.root/'state'/lock_name).open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:return
            with self.connect() as db:db.execute("UPDATE jobs SET status='interrupted',error='Local worker stopped; resume reuses durable outputs.',updated=? WHERE status='running' AND "+predicate,(utc_now(),))
            while not stop.is_set():
                external_busy=False
                for path in (self.root/'artifacts/demo/captures').glob('*/*/worker.heartbeat.json'):
                    heartbeat=read_json(path)
                    if heartbeat.get('status')=='running' and time.time()-path.stat().st_mtime<15:
                        external_busy=True;break
                with self.connect() as db:
                    db.execute('BEGIN IMMEDIATE')
                    rows=db.execute("SELECT * FROM jobs WHERE status='queued' AND "+predicate+" ORDER BY created").fetchall()
                    row=next((r for r in rows if not(external_busy and r['action'] in ('convert','views'))),None)
                    if row:
                        row=dict(row);db.execute("UPDATE jobs SET status='running',attempts=attempts+1,updated=? WHERE id=?",(utc_now(),row['id']))
                if not row:stop.wait(1);continue
                try:self.finish(row['id'],'complete',result=self.execute(row))
                except Exception as error:self.finish(row['id'],'failed',error=str(error))
                except BaseException:
                    self.finish(row['id'],'interrupted',error='Worker interrupted; durable outputs and agent actions retained.');raise


def worker(root,lane='work'):
    import signal
    def interrupt(*_):raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM,interrupt)
    Jobs(root).loop(threading.Event(),lane)
