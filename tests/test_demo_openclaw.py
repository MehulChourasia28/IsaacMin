"""Transport/controller fixtures only; live OpenClaw proof is under evidence/demo."""
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from isaacmin.agent.registry import Registry, Tool
from isaacmin.demo import openclaw
from isaacmin.io import atomic_json, read_json


@pytest.fixture
def harness(tmp_path, monkeypatch):
    atomic_json(tmp_path/'state/planner_selection.json', dict(
        model='nvidia/fixture-only', user_authorized_nvidia_selection=True))
    atomic_json(tmp_path/'.tools/openclaw/node_modules/openclaw/package.json',
        {'version': openclaw.VERSION})
    atomic_json(tmp_path/'.tools/openclaw/package-lock.json', {})
    plugin=tmp_path/'integrations/openclaw/index.js';plugin.parent.mkdir(parents=True)
    plugin.write_text('// fixture, never launched')
    actions=[]; registry=Registry()
    registry.register(Tool('run_pipeline_stage','Only next stage',dict(type='object',
        properties={'stage':{'enum':['source']}},required=['stage'],additionalProperties=False),
        lambda args: actions.append(args) or {'accepted':True}))
    provider=[]
    def chat(model,messages,**options):
        provider.append(options)
        return dict(call_id='fixture-provider',finish_reason='tool_calls',usage={},
            message=dict(role='assistant',content=None,tool_calls=[dict(id='fixture-call',type='function',
                function=dict(name='run_pipeline_stage',arguments=json.dumps({'stage':'source'})))]))
    monkeypatch.setattr(openclaw,'NIMClient',lambda *a,**kw:
        SimpleNamespace(chat=chat,client=SimpleNamespace(close=lambda:None)))
    return tmp_path, registry, actions, provider


def test_bridge_rejects_forged_calls_model_changes_and_unapproved_tools(harness, monkeypatch):
    root, registry, actions, provider=harness
    def embedded(cmd,**kwargs):
        env=kwargs['env'];assert 'NVIDIA_API_KEY' not in env
        config=read_json(Path(env['OPENCLAW_CONFIG_PATH']))
        assert config['tools']['allow']==['run_pipeline_stage']
        assert 'group:fs' in config['tools']['deny'] and config['tools']['toolSearch'] is False
        assert config['models']['providers']['nvidia']['apiKey']=='${ISAACMIN_BRIDGE_TOKEN}'
        with httpx.Client(base_url=env['ISAACMIN_BRIDGE_URL'],trust_env=False) as client:
            call=dict(id='fixture-call',name='run_pipeline_stage',arguments={'stage':'source'})
            assert client.post('/tool',json=call).status_code==403
            client.headers['Authorization']='Bearer '+env['ISAACMIN_BRIDGE_TOKEN']
            assert client.post('/tool',json=call).status_code==400
            body=dict(model='nvidia/fixture-only',messages=[],tools=registry.schemas(),stream=False)
            assert client.post('/v1/chat/completions',json={**body,'model':'other'}).status_code==400
            assert client.post('/v1/chat/completions',json={**body,'tools':[]}).status_code==400
            assert not provider and not actions
            assert client.post('/v1/chat/completions',json=body).status_code==200
            assert client.post('/tool',json={**call,'arguments':{'stage':'delivery'}}).status_code==400
            assert not actions
            assert client.post('/tool',json=call).json()=={'accepted':True}
            assert client.post('/tool',json=call).json()=={'accepted':True}  # same ID, no duplicate action
        return SimpleNamespace(returncode=0,stderr='',stdout=json.dumps(dict(ok=True,
            final='fixture only',provider='nvidia',model='nvidia/fixture-only')))
    monkeypatch.setattr(openclaw.subprocess,'run',embedded)
    result=openclaw.run(root,registry,[dict(role='user',content='fixture')],run_id='bounded',max_tools=1)
    assert result['status']=='complete' and len(actions)==1 and len(provider)==1
    assert openclaw.run(root,registry,[dict(role='user',content='fixture')],run_id='bounded',max_tools=1)==result
    assert len(actions)==1


def test_provider_budget_survives_process_failure_and_resume(harness,monkeypatch):
    root, registry, actions, provider=harness
    def interrupted(cmd,**kwargs):
        env=kwargs['env']
        with httpx.Client(base_url=env['ISAACMIN_BRIDGE_URL'],trust_env=False,
            headers={'Authorization':'Bearer '+env['ISAACMIN_BRIDGE_TOKEN']}) as client:
            response=client.post('/v1/chat/completions',json=dict(model='nvidia/fixture-only',
                messages=[],tools=registry.schemas(),stream=False))
            assert response.status_code==(200 if len(provider)==1 and not attempted[0] else 400)
        attempted[0]=True
        return SimpleNamespace(returncode=1,stderr='',stdout=json.dumps(dict(ok=False,error='fixture interrupted')))
    attempted=[False];monkeypatch.setattr(openclaw.subprocess,'run',interrupted)
    for _ in range(2):
        with pytest.raises(RuntimeError,match='fixture interrupted'):
            openclaw.run(root,registry,[dict(role='user',content='fixture')],run_id='resume',max_calls=1)
    assert len(provider)==1 and not actions
    assert read_json(root/'state/openclaw/runs/resume/receipt.json')['calls']==1
