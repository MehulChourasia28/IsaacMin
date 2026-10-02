"""Loopback-only demo server. No workspace file browsing or remote shell API."""
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlsplit
import json,secrets,threading,mimetypes,re,html
from isaacmin.io import read_json,sha256_file
from isaacmin.security import redact,safe_path
from .catalog import load,planner_status,presets
from .jobs import Jobs


class Server(ThreadingHTTPServer):
    daemon_threads=True
    def __init__(self,root,port):
        self.root=Path(root).resolve();self.csrf=secrets.token_urlsafe(32);self.jobs=Jobs(self.root)
        self.stop=threading.Event();load(self.root)
        super().__init__(('127.0.0.1',port),Handler)
    def start_worker(self):
        import subprocess,sys
        from isaacmin.security import worker_environment
        path=self.root/'artifacts/demo/worker.log';path.parent.mkdir(parents=True,exist_ok=True)
        # The queue owner survives a browser/UI-server restart. The worker's
        # exclusive flock makes concurrent launches harmless.
        with path.open('ab') as log:
            self.worker_processes=[subprocess.Popen([sys.executable,'-m','isaacmin','demo-worker','--lane',lane],
                cwd=self.root,env=worker_environment(),stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
                for lane in ('work','agent')]
    def server_close(self):
        self.stop.set();super().server_close()


class Handler(BaseHTTPRequestHandler):
    server_version='IsaacMin'
    def log_message(self,*args):pass
    def valid_host(self):
        return self.headers.get('Host','') in {f'127.0.0.1:{self.server.server_port}',f'localhost:{self.server.server_port}'}
    def reply_headers(self,status,kind,length,**extra):
        self.send_response(status);self.send_header('Content-Type',kind);self.send_header('Content-Length',str(length))
        self.send_header('X-Content-Type-Options','nosniff');self.send_header('Referrer-Policy','no-referrer')
        self.send_header('Content-Security-Policy',"default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        for k,v in extra.items():self.send_header(k.replace('_','-'),str(v))
        self.end_headers()
    def send_json(self,data,status=200):
        body=json.dumps(redact(data)).encode();self.reply_headers(status,'application/json',len(body),Cache_Control='no-store')
        if self.command!='HEAD':self.wfile.write(body)
    def serve_file(self,path,*,attachment=None,digest=None,head=False):
        path=Path(path);root=self.server.root
        if path.is_symlink() or not path.resolve().is_relative_to(root) or not path.is_file():raise ValueError('File is unavailable')
        size=path.stat().st_size;start=0;end=size-1;status=200;extra={'Accept_Ranges':'bytes'}
        if digest:extra['ETag']='"'+digest+'"'
        value=self.headers.get('Range')
        if value:
            match=re.fullmatch(r'bytes=(\d*)-(\d*)',value)
            if not match or not any(match.groups()):self.reply_headers(416,'text/plain',0,Content_Range=f'bytes */{size}');return
            a,b=match.groups()
            if a:start=int(a);end=min(int(b),end) if b else end
            else:start=max(0,size-int(b))
            if start>=size or end<start:self.reply_headers(416,'text/plain',0,Content_Range=f'bytes */{size}');return
            status=206;extra['Content_Range']=f'bytes {start}-{end}/{size}'
        if attachment:extra['Content_Disposition']='attachment; filename="'+attachment+'"'
        self.reply_headers(status,mimetypes.guess_type(path.name)[0] or 'application/octet-stream',end-start+1,**extra)
        if head:return
        with path.open('rb') as stream:
            stream.seek(start);remaining=end-start+1
            while remaining:
                data=stream.read(min(1024*1024,remaining))
                if not data:break
                self.wfile.write(data);remaining-=len(data)
    def do_HEAD(self):self.do_GET(head=True)
    def result_source(self,job,record,build):
        path=self.server.root/'artifacts/demo/minecraft'/job['preset']/'render.json'
        if not path.is_file():return None
        report=read_json(path)
        terrain=read_json(build/'terrain/terrain.json')
        if (report.get('status')!='actual_source_capture_complete'
                or report.get('source_save_sha256')!=record['identity']['source']
                or report.get('bounds_source_xz')!=terrain['bounds_source_xz']):return None
        return report
    def result_data(self,ident):
        job=self.server.jobs.get(ident)
        if not job or job['action']!='convert' or job['status']!='complete':raise KeyError()
        root=self.server.root;build=root/'artifacts/demo/jobs'/ident/'build'
        record=read_json(build/'outdoor_build.json');digest=sha256_file(Path(record['scene']))
        from .views import completed_views
        groups=[(kind,build/'assembled'/kind) for kind in ('preview','overview')]
        groups.extend(('extra_'+p.parent.name.removeprefix('job_')+'_'+p.name.removeprefix('attempt_'),p/'capture') for p in completed_views(root,ident))
        images=[]
        for kind,directory in groups:
            report=read_json(directory/'preview_result.json')
            if report.get('status')!='actual_visual_preview_complete' or report['scene_sha256']!=digest:raise ValueError('Result capture identity differs')
            for index,frame in enumerate(report['frames']):
                pose=frame['pose'];aerial=pose['kind']=='aerial_overview'
                images.append(dict(url=f'/results/{ident}/{kind}/{index}',caption=pose.get('name') or f'Ground view {index+1}',kind='aerial' if aerial else 'ground'))
        delivery=read_json(root/'artifacts/demo/downloads'/(ident+'.json')) if job.get('download_url') else None
        source=self.result_source(job,record,build)
        minecraft=[dict(url=f'/results/{ident}/source/{index}',caption=frame['caption'],kind='source')
            for index,frame in enumerate(source['images'])] if source else []
        return dict(job=job,images=images,minecraft=minecraft,download=dict(url=job['download_url'],bytes=delivery['bytes']) if delivery else None,
            qualification=record.get('full_end_to_end_qualification','not_run'))
    def conversion_result(self,ident,kind=None,index=None,head=False):
        job=self.server.jobs.get(ident)
        if not job or job['action']!='convert' or job['status']!='complete':raise KeyError()
        build=self.server.root/'artifacts/demo/jobs'/ident/'build'
        record=read_json(build/'outdoor_build.json');digest=sha256_file(Path(record['scene']))
        if kind=='source':
            source=self.result_source(job,record,build)
            if not source or index>=len(source['images']):raise KeyError()
            frame=source['images'][index]
            path=safe_path(self.server.root/'artifacts/demo/minecraft'/job['preset'],frame['file'],must_exist=True)
            if sha256_file(path)!=frame['sha256']:raise ValueError('Source image changed')
            self.serve_file(path,digest=frame['sha256'],head=head);return
        frames=[]
        from .views import completed_views
        captures=[(capture,build/'assembled'/capture) for capture in ('preview','overview')]
        captures.extend(('extra_'+path.parent.name.removeprefix('job_')+'_'+path.name.removeprefix('attempt_'),path/'capture') for path in completed_views(self.server.root,ident))
        for capture,directory in captures:
            report=read_json(directory/'preview_result.json')
            if report['status']!='actual_visual_preview_complete' or report['scene_sha256']!=digest:
                raise ValueError('Conversion capture does not match its completed native world')
            for number,frame in enumerate(report['frames']):
                if kind==capture and index==number:
                    path=safe_path(directory,frame['rgb'],must_exist=True)
                    if sha256_file(path)!=frame['rgb_sha256']:raise ValueError('Conversion image changed')
                    self.serve_file(path,digest=frame['rgb_sha256'],head=head);return
                frames.append((capture,number,frame['pose'].get('name') or f'Ground view {number+1}'))
        if kind is not None:raise KeyError()
        title=presets(self.server.root)[job['preset']]['title']
        data=self.result_data(ident)
        download=f'<p><a class="primary" href="/download/{ident}">Download this world ↓</a></p>' if job.get('download_url') else ''
        controls=f'<div class="view-controls" data-preset="{html.escape(job["preset"])}" data-target="{ident}"></div><script src="/views.js" defer></script>'
        source=data['minecraft'][0] if data['minecraft'] else None
        # Start on an aerial image for the most useful geography comparison.
        selected=next((item for item in data['images'] if item['kind']=='aerial'),data['images'][0])
        source_html=(f'<a href="{source["url"]}" target="_blank" rel="noopener"><img id="result-source" src="{source["url"]}" alt="Minecraft source-save aerial"></a>'
            if source else '<div class="empty"><p>No matching Minecraft render is available for this saved region.</p></div>')
        thumbnails=''.join(f'<button class="thumbnail{" selected" if item==selected else ""}" data-kind="{item["kind"]}" data-url="{item["url"]}" data-caption="{html.escape(item["caption"].replace("_"," "))}" aria-pressed="{str(item==selected).lower()}" aria-label="Show {html.escape(item["caption"].replace("_"," "))}"><img loading="lazy" src="{item["url"]}" alt=""><span>{item["kind"].capitalize()}</span></button>' for item in data['images'])
        comparison=f'''<section class="workspace result-workspace" aria-label="Minecraft and Isaac Sim comparison">
<div class="comparison"><article class="view"><div class="view-label">Minecraft source <small>ACTUAL SAVE · CHUNKY</small></div><div class="image-wrap">{source_html}</div><div class="view-foot">Source-save aerial</div></article>
<article class="view"><div class="view-label">Generated Isaac Sim <small>ACTUAL SAVED CAPTURE</small></div><div class="image-wrap"><a id="result-image-link" href="{selected['url']}" target="_blank" rel="noopener"><img id="result-image" src="{selected['url']}" alt="{html.escape(selected['caption'].replace('_',' '))}"></a></div><div id="result-caption" class="view-foot" aria-live="polite">{html.escape(selected['caption'].replace('_',' '))}</div></article></div>
<div class="gallery-bar"><div class="gallery-label"><p class="eyebrow">Explore the Isaac world</p><div class="view-filters" aria-label="Filter Isaac views"><button class="selected" data-filter="all" aria-pressed="true">All views</button><button data-filter="ground" aria-pressed="false">Ground</button><button data-filter="aerial" aria-pressed="false">Aerial</button></div></div><div class="thumbnails">{thumbnails}</div><p class="subtle">Same source save and region; camera angles differ. Select an Isaac view to compare, or click either image to open it full size.</p></div></section>'''
        body=(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)} · World comparison</title><link rel="stylesheet" href="/style.css"><script src="/results.js" defer></script></head><body><main><p><a href="/library">← Saved worlds</a> · <a href="/">Create a world</a></p><h1>{html.escape(title)}</h1><p>Compare your Minecraft region with its generated Isaac world.</p>{comparison}{download}{controls}<p class="subtle">Development result; full realism, motion and contact qualification remain pending.</p></main></body></html>').encode()
        self.reply_headers(200,'text/html; charset=utf-8',len(body),Cache_Control='no-store')
        if not head:self.wfile.write(body)
    def do_GET(self,head=False):
        if not self.valid_host():self.send_json({'error':'Invalid local host'},403);return
        route=urlsplit(self.path).path;root=self.server.root
        try:
            if route=='/api/catalog':
                catalog=load(root);self.send_json(dict(presets=catalog['presets'],updated_at_utc=catalog['updated_at_utc'],csrf=self.server.csrf,planner=planner_status(root),workers=self.server.jobs.worker_status()));return
            if route=='/api/jobs':self.send_json({'jobs':self.server.jobs.list(),'workers':self.server.jobs.worker_status()});return
            construction=re.fullmatch(r'/api/jobs/(job_[a-f0-9]{32})/construction',route)
            if construction:
                from .construction import progress
                self.send_json(progress(root,self.server.jobs,construction.group(1)));return
            result_api=re.fullmatch(r'/api/results/(job_[a-f0-9]{32})',route)
            if result_api:self.send_json(self.result_data(result_api.group(1)));return
            preview=re.fullmatch(r'/construction/(job_[a-f0-9]{32})/(source\.json|terrain\.json|assets\.json|assets\.bin)',route)
            if preview:
                ident,name=preview.groups();job=self.server.jobs.get(ident)
                if not job or job['action']!='convert':raise KeyError()
                path=root/'artifacts/demo/jobs'/ident/'construction'/name
                if name=='assets.bin':
                    manifest=read_json(path.with_suffix('.json'))
                    if path.stat().st_size!=manifest['points']*manifest['stride_floats']*4:raise ValueError('Construction point data is incomplete')
                self.serve_file(path,head=head);return
            job_route=re.fullmatch(r'/api/jobs/(job_[a-f0-9]{32})',route)
            if job_route:
                job=self.server.jobs.get(job_route.group(1))
                if not job:raise KeyError()
                self.send_json(job);return
            result=re.fullmatch(r'/results/(job_[a-f0-9]{32})(?:/(source|preview|overview|extra_[a-f0-9]{32}_\d)/(\d{1,3}))?',route)
            if result:
                ident,kind,index=result.groups()
                self.conversion_result(ident,kind,int(index) if index is not None else None,head);return
            if route.startswith('/media/'):
                media=load(root)['media'].get(route.removeprefix('/media/'))
                if media is None:raise KeyError()
                self.serve_file(media['path'],digest=media['sha256'],head=head);return
            if route.startswith('/download/'):
                ident=route.removeprefix('/download/')
                if ident in presets(root):
                    item=next(p for p in load(root)['presets'] if p['id']==ident)
                    if not item['download']:raise KeyError()
                else:
                    if not re.fullmatch(r'job_[a-f0-9]{32}',ident):raise KeyError()
                    job=self.server.jobs.get(ident)
                    if not job or not job.get('download_url'):raise KeyError()
                delivery=read_json(root/'artifacts/demo/downloads'/(ident+'.json'))
                self.serve_file(delivery['archive'],attachment='IsaacMin-'+ident+'.zip',digest=delivery['sha256'],head=head);return
            static={'/':'start.html','/index.html':'start.html','/library':'index.html','/next':'start.html',
                '/start.js':'start.js','/start.css':'start.css','/construction.js':'construction.js',
                '/vendor/three.module.min.js':'vendor/three.module.min.js','/vendor/three.core.min.js':'vendor/three.core.min.js',
                '/app.js':'app.js','/views.js':'views.js','/results.js':'results.js','/style.css':'style.css'}
            if route not in static:raise KeyError()
            self.serve_file(root/'web'/static[route],head=head)
        except (KeyError,FileNotFoundError):self.send_json({'error':'Not found'},404)
        except (BrokenPipeError,ConnectionResetError):pass
        except ValueError as error:self.send_json({'error':str(error)},400)
    def do_POST(self):
        expected={'http://127.0.0.1:'+str(self.server.server_port),'http://localhost:'+str(self.server.server_port)}
        if (not self.valid_host() or self.headers.get('Origin') not in expected
                or not secrets.compare_digest(self.headers.get('X-IsaacMin-Token',''),self.server.csrf)):
            self.send_json({'error':'Local request token or origin invalid'},403);return
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<=16000 or self.headers.get('Content-Type')!='application/json':raise ValueError('Invalid JSON request')
            data=json.loads(self.rfile.read(length));route=urlsplit(self.path).path
            if not isinstance(data,dict):raise ValueError('Expected an object')
            if route=='/api/jobs':
                if not {'preset','action','operation_key'}<=set(data) or set(data)-{'preset','action','operation_key','prompt','view','target','context_job'}:raise ValueError('Unknown or missing job fields')
                self.send_json(self.server.jobs.submit(**data),202)
            elif route in ('/api/jobs/resume','/api/jobs/cancel'):
                if 'id' not in data or set(data)-{'id','operation_key'}:raise ValueError('Expected job identifier and optional operation key')
                result=self.server.jobs.control(data['id'],route.rsplit('/',1)[1],data.get('operation_key'))
                self.send_json(result,202)
            else:self.send_json({'error':'Not found'},404)
        except (ValueError,TypeError,KeyError) as error:self.send_json({'error':str(error)},400)


def serve(root,port=8765):
    with Server(root,port) as server:
        server.start_worker();print(f'IsaacMin demo: http://127.0.0.1:{server.server_port}',flush=True)
        try:server.serve_forever()
        except KeyboardInterrupt:pass
