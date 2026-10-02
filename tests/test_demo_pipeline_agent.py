"""Controller adversarial fixtures; these are not hosted-model or Isaac evidence."""
import json
from types import SimpleNamespace
import pytest
from isaacmin.io import atomic_json, read_json
from isaacmin.demo.pipeline_agent import PipelineAgent, next_stage, STAGES


def controller(tmp_path, responder):
    atomic_json(tmp_path/'state/planner_selection.json', dict(
        model='nvidia/fixture-only', user_authorized_nvidia_selection=True))
    def chat(model, messages, **options):
        assert options['temperature'] == 0
        schema=options['tools'][0]['function']['parameters']
        arguments={key:value['enum'][0] for key,value in schema['properties'].items()}
        name,arguments=responder(arguments)
        return dict(call_id='fixture-provider', message=dict(role='assistant', tool_calls=[
            dict(id='fixture-tool', function=dict(name=name, arguments=json.dumps(arguments)))]))
    client=SimpleNamespace(chat=chat, client=SimpleNamespace(close=lambda:None))
    return PipelineAgent(tmp_path,'fixture',client_factory=lambda *a,**kw:client)


def test_model_cannot_skip_steps_change_parameters_or_invent_tools(tmp_path):
    attempts=[]
    def response(args):
        attempts.append(1)
        if len(attempts)==1:return 'run_pipeline_stage',{**args,'stage':'delivery'}
        if len(attempts)==2:return 'run_pipeline_stage',{**args,'samples':1}
        return 'arbitrary_shell',args
    agent=controller(tmp_path,response);record=dict(identity={'source':'fixture'},stages={})
    with pytest.raises(ValueError,match='next'):agent.before('capture',record)
    assert not attempts
    with pytest.raises(RuntimeError,match='exhausted'):agent.before('source',record)
    assert len(attempts)==3 and not any(e['status']=='dispatched' for e in agent.state['events'])
    with pytest.raises(RuntimeError,match='exhausted'):agent.before('source',record)
    assert len(attempts)==3


def test_resume_reuses_dispatch_and_reconciles_real_checkpoint_after_crash(tmp_path):
    calls=[]
    def response(args):calls.append(args);return 'run_pipeline_stage',args
    agent=controller(tmp_path,response);record=dict(identity={'source':'fixture'},stages={})
    agent.before('source',record)
    agent=controller(tmp_path,response);agent.before('source',record)
    assert len(calls)==1
    # Native checkpoint committed, then the process died before agent journal update.
    record['stages']['source']={'manifest_sha256':'fixture-result'}
    agent=controller(tmp_path,response);agent.before('objects',record)
    assert len(calls)==2 and agent.state['steps']['source']['status']=='complete'
    assert next_stage({'source':{}})=='objects'
    record['stages']['source']['manifest_sha256']='tampered'
    with pytest.raises(ValueError,match='receipt changed'):agent.before('objects',record)


def test_identity_and_completed_order_cannot_change_on_resume(tmp_path):
    agent=controller(tmp_path,lambda args:('run_pipeline_stage',args))
    record=dict(identity={'source':'fixture'},stages={});agent.before('source',record)
    record['identity']['source']='different'
    with pytest.raises(ValueError,match='dependencies changed'):agent.before('source',record)
    with pytest.raises(ValueError,match='out of order'):next_stage({'source':{},'terrain':{}})
    with pytest.raises(ValueError,match='Unknown'):next_stage({'invented':{}})
    assert next_stage(dict.fromkeys(STAGES)) is None


def test_provider_failure_never_dispatches_a_stage_and_call_budget_persists(tmp_path):
    agent=controller(tmp_path,lambda _:(_ for _ in ()).throw(RuntimeError('fixture network unavailable')))
    record=dict(identity={'source':'fixture'},stages={})
    with pytest.raises(RuntimeError,match='network'):agent.before('source',record)
    saved=read_json(agent.path)
    assert saved['steps']['source']['calls']==1 and saved['status']=='blocked_at_stage'
    assert not saved['events']


def test_daylight_lift_is_bounded_shade_safe_and_idempotent():
    from isaacmin.assembly.outdoor_lighting import daylight_camera_response
    capture={'lighting_parameters':{'daylight_response':dict(maximum_lift_ev=.45,shade_ev100=10.,open_ev100=13.747)}}
    for ev in (9.,10.,12.,13.747,15.):
        pose=dict(camera_response=dict(ev100=ev,f_number=8,iso=100))
        changed=daylight_camera_response(pose,capture)
        assert 0<=ev-changed['camera_response']['ev100']<=.450001
        if ev<=10:assert changed['camera_response']['ev100']==ev
        assert daylight_camera_response(changed,capture)==changed
        assert daylight_camera_response(pose,{'lighting_parameters':{}})==pose
