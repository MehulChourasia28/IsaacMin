"""Local controller and HTTP contracts; no fixture claims native/model success."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
from pathlib import Path
from types import SimpleNamespace

import pytest

from isaacmin.io import atomic_json, read_json, sha256_file
from isaacmin.demo.jobs import Jobs


@pytest.fixture
def root(tmp_path):
    atomic_json(tmp_path/'state/demo_presets.json', {'presets':[
        {'id':'fixture','build':str(tmp_path/'build')},
        {'id':'other','build':str(tmp_path/'other')} ]})
    atomic_json(tmp_path/'build/outdoor_build.json', {'scene':'fixture-only'})
    return tmp_path


def test_concurrent_submission_is_one_durable_job_and_reuse_cannot_change_it(root):
    jobs=Jobs(root)
    with ThreadPoolExecutor(max_workers=6) as pool:
        results=list(pool.map(lambda _:jobs.submit('fixture','verify','same-operation'),range(12)))
    assert len({r['id'] for r in results})==1
    assert len(Jobs(root).list())==1
    with pytest.raises(ValueError,match='another request'):
        jobs.submit('other','verify','same-operation')
    assert Jobs(root).list()[0]['preset']=='fixture'


def test_worker_crash_retains_job_and_resume_has_finite_attempts(root,monkeypatch):
    jobs=Jobs(root);job=jobs.submit('fixture','verify','crash-operation')
    def interrupted(_):raise KeyboardInterrupt()
    monkeypatch.setattr(jobs,'execute',interrupted)
    for count in range(1,4):
        with pytest.raises(KeyboardInterrupt):jobs.loop(threading.Event())
        result=jobs.list()[0]
        assert (result['status'],result['attempts'])==('interrupted',count)
        if count<3:Jobs(root).resume(job['id'])
    with pytest.raises(ValueError,match='exhausted'):Jobs(root).resume(job['id'])


def test_second_worker_cannot_recover_or_execute_another_owners_job(root):
    import fcntl
    jobs=Jobs(root);job=jobs.submit('fixture','verify','owner-operation')
    jobs.finish(job['id'],'running')
    with (root/'state/demo_worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        jobs.loop(threading.Event())
    assert jobs.list()[0]['status']=='running'
    stop=threading.Event();stop.set();jobs.loop(stop)
    assert jobs.list()[0]['status']=='interrupted'


def test_worker_recovery_is_scoped_to_its_lane(root):
    jobs=Jobs(root)
    native=jobs.submit('fixture','verify','native-operation')
    agent=jobs.submit('fixture','agent','agent-operation',prompt='Fixture request')
    for job in (native,agent):jobs.finish(job['id'],'running')
    stop=threading.Event();stop.set();jobs.loop(stop,'agent')
    states={j['id']:j['status'] for j in jobs.list()}
    assert states[native['id']]=='running' and states[agent['id']]=='interrupted'
    jobs.loop(stop,'work')
    assert all(j['status']=='interrupted' for j in jobs.list())


def test_public_polling_does_not_repeat_nested_agent_tool_traces(root):
    jobs=Jobs(root);job=jobs.submit('fixture','agent','trace-operation',prompt='Fixture request')
    jobs.finish(job['id'],'complete',result={'answer':'Fixture result','tools':[{'result':{'jobs':[{'result':{'tools':[]}}]}}]})
    public=jobs.list()[0]
    assert public['result']=={'answer':'Fixture result'}
    with jobs.connect() as db:
        assert json.loads(db.execute('SELECT result FROM jobs WHERE id=?',(job['id'],)).fetchone()[0])['tools']


def test_resumed_download_rejects_corrupted_or_extra_archive_members(tmp_path):
    import zipfile
    from isaacmin.demo.archive import verify_archive
    package=tmp_path/'package';package.mkdir();(package/'world.usda').write_text('fixture scene bytes')
    manifest={'files':[dict(path='world.usda',sha256=sha256_file(package/'world.usda'))]}
    atomic_json(package/'package.json',manifest)
    for altered in ('none','payload','extra'):
        path=tmp_path/(altered+'.zip')
        with zipfile.ZipFile(path,'w') as archive:
            archive.write(package/'package.json','IsaacMin-fixture/package.json')
            archive.writestr('IsaacMin-fixture/world.usda','different scene bytes' if altered=='payload' else 'fixture scene bytes')
            if altered=='extra':archive.writestr('IsaacMin-fixture/unexpected','extra')
        if altered=='none':verify_archive(path,package,manifest,'fixture')
        else:
            with pytest.raises(ValueError,match='differs|differ'):verify_archive(path,package,manifest,'fixture')


def planner_fixture(root,monkeypatch,client):
    from functools import partial
    from isaacmin.demo import planner
    # Retained historical sessions use the original runtime on resume.
    monkeypatch.setattr(planner,'run',partial(planner.run,harness='custom_fixture'))
    atomic_json(root/'state/planner_selection.json', {'generation_options':{}})
    monkeypatch.setattr(planner,'planner_model',lambda _: 'nvidia/fixture-controller-only')
    monkeypatch.setattr(planner,'planner_status',lambda _: {'status':'available'})
    monkeypatch.setattr(planner,'load',lambda _: {'presets':[dict(id='fixture',title='Fixture',extent_m=[1,1],
        isaac=[],minecraft=[],download=None,qualification='fixture_only')]})
    monkeypatch.setattr(planner,'NIMClient',lambda *a,**kw:client)
    return planner


def tool_call(name,args,ident='fixture-call'):
    return {'role':'assistant','content':None,'tool_calls':[dict(id=ident,type='function',
        function=dict(name=name,arguments=json.dumps(args)))]}


def test_agent_crash_after_queue_insert_reuses_action_and_preserves_tool_messages(root,monkeypatch):
    calls=[]
    def chat(model,messages,**kw):
        calls.append(1)
        if len(calls)==1:return {'message':tool_call('start_job',{'preset':'fixture','action':'verify'})}
        assert messages[-2]['tool_calls'][0]['id']=='fixture-call'
        assert messages[-1]['tool_call_id']=='fixture-call'
        assert json.loads(messages[-1]['content'])['status']=='queued'
        return {'message':{'role':'assistant','content':'Fixture action queued, not qualified.'}}
    client=SimpleNamespace(chat=chat,client=SimpleNamespace(close=lambda:None))
    planner=planner_fixture(root,monkeypatch,client);jobs=Jobs(root);submit=jobs.submit
    def crash(*a,**kw):submit(*a,**kw);raise KeyboardInterrupt()
    monkeypatch.setattr(jobs,'submit',crash)
    with pytest.raises(KeyboardInterrupt):planner.run(root,jobs,'fixture','Verify this fixture','job_fixture')
    checkpoint=read_json(root/'state/demo_agent_runs/job_fixture.json')
    assert len(checkpoint['pending'])==1 and checkpoint['calls']==1
    monkeypatch.setattr(jobs,'submit',submit)
    result=planner.run(root,jobs,'fixture','Verify this fixture','job_fixture')
    assert result['status']=='complete' and len(jobs.list())==1 and len(calls)==2
    assert planner.run(root,jobs,'fixture','Verify this fixture','job_fixture')==result
    assert len(calls)==2
    with pytest.raises(ValueError,match='changed'):planner.run(root,jobs,'other','Verify this fixture','job_fixture')


def test_agent_cannot_dispatch_to_unselected_preset(root,monkeypatch):
    calls=[]
    def chat(model,messages,**kw):
        calls.append(1)
        if len(calls)==1:return {'message':tool_call('start_job',{'preset':'other','action':'convert'})}
        assert json.loads(messages[-1]['content'])['status']=='failed'
        return {'message':{'role':'assistant','content':'Fixture action rejected.'}}
    planner=planner_fixture(root,monkeypatch,SimpleNamespace(chat=chat,client=SimpleNamespace(close=lambda:None)))
    jobs=Jobs(root);planner.run(root,jobs,'fixture','Inspect fixture','job_fixture')
    assert not jobs.list()


def test_empty_provider_completion_is_not_success(root,monkeypatch):
    client=SimpleNamespace(chat=lambda *a,**kw:{'message':{'role':'assistant','content':None}},client=SimpleNamespace(close=lambda:None))
    planner=planner_fixture(root,monkeypatch,client)
    with pytest.raises(RuntimeError,match='no final answer'):planner.run(root,Jobs(root),'fixture','Inspect fixture','job_fixture')
    assert read_json(root/'state/demo_agent_runs/job_fixture.json')['status']!='complete'


@pytest.fixture
def server(root):
    from isaacmin.demo.server import Server
    media=root/'artifacts/fixture.bin';media.parent.mkdir();media.write_bytes(bytes(range(256))*8)
    atomic_json(root/'artifacts/demo/catalog.json',dict(updated_at_utc='fixture',presets=[
        {'id':'fixture','download':{'url':'/download/fixture'}}],media={
        'fixture':dict(path=str(media),sha256=sha256_file(media))}))
    atomic_json(root/'artifacts/demo/downloads/fixture.json',{'archive':str(media),'sha256':sha256_file(media)})
    web=root/'web';web.mkdir();(web/'index.html').write_text('fixture-local-server')
    (web/'start.html').write_text('fixture-clean-start')
    with Server(root,0) as app:
        thread=threading.Thread(target=app.serve_forever,daemon=True);thread.start()
        yield app
        app.shutdown();thread.join()


def request(server,path,method='GET',body=None,headers=None):
    connection=HTTPConnection('127.0.0.1',server.server_port,timeout=5)
    connection.request(method,path,body,headers or {})
    response=connection.getresponse();data=response.read();result=(response.status,dict(response.getheaders()),data)
    connection.close();return result


def test_http_range_requests_reassemble_exact_download_and_reject_invalid_ranges(server):
    full=request(server,'/download/fixture')[2];parts=[]
    for start,end in [(0,1023),(1024,2047)]:
        status,headers,data=request(server,'/download/fixture',headers={'Range':f'bytes={start}-{end}'})
        assert status==206 and headers['Content-Range']==f'bytes {start}-{end}/2048';parts.append(data)
    assert b''.join(parts)==full
    assert request(server,'/download/fixture',headers={'Range':'bytes=-8'})[2]==full[-8:]
    assert request(server,'/download/fixture',headers={'Range':'bytes=2048-'})[0]==416
    assert request(server,'/download/fixture',headers={'Range':'bytes=0-1,4-5'})[0]==416
    status,headers,body=request(server,'/download/fixture',method='HEAD')
    assert status==200 and not body and headers['Content-Length']=='2048'


def test_http_mutations_require_local_host_origin_and_token_and_paths_are_allowlisted(server):
    assert request(server,'/')[2]==b'fixture-clean-start'
    assert request(server,'/library')[2]==b'fixture-local-server'
    body=json.dumps(dict(preset='fixture',action='verify',operation_key='http-operation'))
    headers={'Content-Type':'application/json','Origin':f'http://127.0.0.1:{server.server_port}','X-IsaacMin-Token':server.csrf}
    for changed in [{'Origin':'https://untrusted.example'},{'X-IsaacMin-Token':'wrong'},{'Host':'untrusted.example'}]:
        assert request(server,'/api/jobs','POST',body,{**headers,**changed})[0]==403
    assert request(server,'/api/jobs','POST',body,headers)[0]==202
    assert request(server,'/api/jobs','POST',body,headers)[0]==202
    assert len(server.jobs.list())==1
    for path in ['/.env','/media/../../.env','/media/%2e%2e/.env','/download/unknown']:
        assert request(server,path)[0]==404
    assert request(server,'/',headers={'Host':'untrusted.example'})[0]==403


def test_job_lookup_and_controls_survive_polling_window_and_replayed_requests(server):
    job=server.jobs.submit('fixture','verify','http-control-job')
    headers={'Content-Type':'application/json','Origin':f'http://127.0.0.1:{server.server_port}','X-IsaacMin-Token':server.csrf}
    body=json.dumps(dict(id=job['id'],operation_key='http-cancel-once'))
    assert request(server,'/api/jobs/cancel','POST',body,headers)[0]==202
    assert request(server,'/api/jobs/cancel','POST',body,headers)[0]==202
    assert json.loads(request(server,'/api/jobs/'+job['id'])[2])['status']=='cancelled'
    body=json.dumps(dict(id=job['id'],operation_key='http-resume-once'))
    assert request(server,'/api/jobs/resume','POST',body,headers)[0]==202
    assert request(server,'/api/jobs/resume','POST',body,headers)[0]==202
    assert json.loads(request(server,'/api/jobs/'+job['id'])[2])['status']=='queued'
    assert request(server,'/api/jobs/job_'+'0'*32)[0]==404


def test_clear_library_removes_old_results_but_preserves_new_conversions_and_evidence(server):
    from isaacmin.demo.library import clear_saved_worlds
    from isaacmin.demo.views import resolve_target
    jobs=server.jobs;old=jobs.submit('fixture','convert','old-conversion')
    jobs.finish(old['id'],'complete')
    evidence=server.root/'artifacts/fixture.bin';before=sha256_file(evidence)
    clear_saved_worlds(server.root,jobs)
    assert jobs.list()==[] and jobs.get(old['id']) is None
    assert jobs.latest_conversion('fixture') is None
    assert request(server,'/download/fixture')[0]==404
    assert request(server,'/results/'+old['id'])[0]==404
    assert sha256_file(evidence)==before
    with pytest.raises(ValueError,match='archived'):jobs.resume(old['id'])
    with pytest.raises(ValueError,match='no saved world'):resolve_target(server.root,jobs,'fixture')
    with pytest.raises(ValueError,match='no saved world'):jobs.submit('fixture','prepare','old-preparation')
    new=jobs.submit('fixture','convert','new-conversion')
    assert jobs.get(new['id'])['status']=='queued'
    with pytest.raises(ValueError,match='Finish active'):clear_saved_worlds(server.root,jobs)


def test_saved_result_compares_only_matching_source_region_and_checks_image_bytes(server):
    registry=read_json(server.root/'state/demo_presets.json')
    registry['presets'][0]['title']='Fixture world'
    atomic_json(server.root/'state/demo_presets.json',registry)
    job=server.jobs.submit('fixture','convert','comparison-fixture')
    server.jobs.finish(job['id'],'complete')
    build=server.root/'artifacts/demo/jobs'/job['id']/'build'
    build.mkdir(parents=True);scene=build/'world.usda';scene.write_text('fixture scene')
    atomic_json(build/'outdoor_build.json',dict(scene=str(scene),identity=dict(source='fixture-save')))
    atomic_json(build/'terrain/terrain.json',dict(bounds_source_xz=[0,0,256,256]))
    for kind in ('preview','overview'):
        directory=build/'assembled'/kind;directory.mkdir(parents=True)
        (directory/'view.png').write_bytes(b'fixture-image')
        atomic_json(directory/'preview_result.json',dict(status='actual_visual_preview_complete',
            scene_sha256=sha256_file(scene),frames=[dict(rgb='view.png',rgb_sha256=sha256_file(directory/'view.png'),
                pose=dict(name=kind,kind='aerial_overview' if kind=='overview' else 'ground'))]))
    source=server.root/'artifacts/demo/minecraft/fixture';source.mkdir(parents=True)
    picture=source/'source.png';picture.write_bytes(b'fixture-source')
    receipt=dict(status='actual_source_capture_complete',source_save_sha256='fixture-save',bounds_source_xz=[0,0,256,256],
        images=[dict(file=picture.name,sha256=sha256_file(picture),caption='Fixture source')])
    atomic_json(source/'render.json',receipt)
    route='/results/'+job['id'];api='/api/results/'+job['id']
    result=json.loads(request(server,api)[2])
    assert result['minecraft'][0]['url']==route+'/source/0'
    status,_,body=request(server,route)
    assert status==200 and b'Minecraft source' in body and b'Generated Isaac Sim' in body
    assert request(server,route+'/source/0')[2]==b'fixture-source'
    assert request(server,route+'/source/1')[0]==404
    picture.write_bytes(b'changed')
    assert request(server,route+'/source/0')[0]==400
    for field,value in [('source_save_sha256','different-save'),('bounds_source_xz',[1,0,257,256])]:
        atomic_json(source/'render.json',{**receipt,field:value})
        assert json.loads(request(server,api)[2])['minecraft']==[]
        assert request(server,route+'/source/0')[0]==404
        assert b'No matching Minecraft render' in request(server,route)[2]
