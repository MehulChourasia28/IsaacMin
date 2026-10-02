"""Bounded detector development, isolated from held-out case locations and labels.

Exactly three attributed development recipes may be attempted in one ledger.
Complete held-out calibration runs only after a recipe passes every development
case. Neither protocol/schema failure nor partial category coverage can qualify.
"""
from pathlib import Path
import hashlib
import json
import fcntl
from .critic import load_dataset,decode_finding_compat,evaluate_case,CATEGORIES
from .nim import NIMClient,VISION
from .visual_workflow import _immutable,_bound,_verify,_hash
from isaacmin.assets.network import ServiceError,utcnow,digest


def _complete(cases,split):
    selected=[c for c in cases if c['split']==split]
    return bool(set(CATEGORIES)=={e['category'] for c in selected for e in c.get('expected_severe',[])} and
                any(c.get('clean_control_independently_verified') and not c.get('expected_severe') for c in selected))


def _recipe(value):
    if set(value)!={'prompt','reasoning_budget','repair_rationale','parent_failure_report'} or not isinstance(value['prompt'],str) or not value['prompt'].strip() or value['reasoning_budget'] not in (0,128,512,1024):
        raise ServiceError('invalid_detector_recipe','Explicit prompt, bounded documented reasoning budget, rationale and parent failure evidence are required')
    _verify(value['parent_failure_report'])
    return value


def _evaluate(workspace,cases,recipe,output,*,dataset_path,scope):
    client=NIMClient(workspace,deadline=900);records=[]
    protocol={'model':VISION,'prompt':recipe['prompt'],'prompt_sha256':hashlib.sha256(recipe['prompt'].encode()).hexdigest(),
              'reasoning_budget':recipe['reasoning_budget'],'max_tokens':2048,'temperature':0,'decoder':'identical_consecutive_json_v1'}
    _immutable(output/'request_protocol.json',protocol)
    for case in cases:
        path=output/'cases'/(hashlib.sha256(case['id'].encode()).hexdigest()+'.json')
        if path.exists():
            record=json.loads(path.read_text())
            if record['image_sha256']!=case['image_sha256'] or record['protocol_sha256']!=digest(output/'request_protocol.json'):
                raise ServiceError('changed_calibration_session','Persisted detector call belongs to another exact image or protocol')
        else:
            record={'id':case['id'],'split':case['split'],'image_sha256':case['image_sha256'],'protocol_sha256':digest(output/'request_protocol.json')}
            try:
                response=client.inspect_image(Path(case['image']),recipe['prompt'],max_tokens=2048,reasoning_budget=recipe['reasoning_budget'],temperature=0)
                record.update(call_id=response['call_id'],raw_final_response=response['message'].get('content',''))
                decoded=decode_finding_compat(record['raw_final_response']);record.update(**decoded,**evaluate_case(case,decoded['finding']))
            except ServiceError as exc:
                record.update(status='blocked' if exc.code in ('access_restricted','invalid_key','network','provider_outage','rate_limit','request_deadline') else 'fail',error=exc.record(),
                              failed_severe_cases_not_evaluable=len(case.get('expected_severe',[])))
            _immutable(path,record)
        records.append(record)
        if record['status']=='blocked':break
    coverage={split:sorted({e['category'] for c in cases if c['split']==split for e in c.get('expected_severe',[])}) for split in ('development','held_out')}
    controls={split:sum(c.get('clean_control_independently_verified',False) and not c.get('expected_severe') for c in cases if c['split']==split) for split in coverage}
    status='blocked' if any(c['status']=='blocked' for c in records) else 'pass' if len(records)==len(cases) and all(c['status']=='pass' for c in records) else 'fail'
    report={'schema_version':1,'created_at_utc':utcnow(),**protocol,'status':status,'scope':scope,
            'dataset_path':str(dataset_path),'dataset_sha256':digest(dataset_path),'cases':records,
            'category_coverage':coverage,'verified_clean_controls':controls,
            'severe_missed':sum(len(c.get('missed',[])) for c in records),'severe_cases_not_evaluable':sum(c.get('failed_severe_cases_not_evaluable',0) for c in records),
            'false_positives':sum(len(c.get('false_positives',[])) for c in records),'critic_usable_for_Q09':False,
            'limitations':['All severe faults and verified controls must pass unchanged criteria','Development pass alone cannot qualify the critic','Original wire/schema failures are preserved; no extraction of JSON from prose or inference of absent fields']}
    return report


def develop_detector_recipe(workspace,dataset_path,recipe,ledger_directory):
    """Run ONE repair cycle using a manifest containing development cases ONLY."""
    workspace=Path(workspace).resolve();dataset_path=Path(dataset_path).resolve();recipe=_recipe(recipe)
    data=load_dataset(dataset_path)
    if any(c['split']!='development' for c in data['cases']):raise ServiceError('heldout_development_leakage','Detector development input must exclude all held-out cases')
    if not _complete(data['cases'],'development'):raise ServiceError('incomplete_detector_development','All five known severe classes and a verified clean control are required before detector repair')
    ledger=Path(ledger_directory).resolve();ledger.mkdir(parents=True,exist_ok=True)
    with (ledger/'repair.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        identity={'model':VISION,'development_dataset':_bound(dataset_path),'maximum_repair_cycles':3,'held_out_inputs':'forbidden'}
        _immutable(ledger/'session.json',identity)
        attempts=sorted(ledger.glob('attempt_*/recipe.json'))
        # Existing identities remain immutable; new directories cannot conceal the
        # stable session's previous attempts from this bounded ledger.
        for previous in attempts:_recipe(json.loads(previous.read_text())['recipe'])
        matching=next((p for p in attempts if json.loads(p.read_text())['recipe']==recipe),None)
        if matching:
            output=matching.parent
            if (output/'report.json').exists():return json.loads((output/'report.json').read_text())
        else:
            if len(attempts)>=3:raise ServiceError('detector_repair_budget_exhausted','Three automatic development recipes exhausted; retain calibration failure')
            output=ledger/f'attempt_{len(attempts)+1:02d}'
            record={'cycle':len(attempts)+1,'recipe':recipe,'session_sha256':digest(ledger/'session.json'),
                    'previous_recipe_sha256':digest(attempts[-1]) if attempts else None,'input_policy':'development original raw RGB only; no held-out label, bbox or pose enters construction'}
            _immutable(output/'recipe.json',record)
    report=_evaluate(workspace,data['cases'],recipe,output,dataset_path=dataset_path,scope='development_detector_repair_only')
    report['recipe_record']=_bound(output/'recipe.json');_immutable(output/'report.json',report)
    return report


def calibrate_developed_recipe(workspace,dataset_path,development_report,output_directory):
    """Evaluate complete frozen development+held-out set once, then publish if pass."""
    workspace=Path(workspace).resolve();dataset_path=Path(dataset_path).resolve();data=load_dataset(dataset_path)
    development_path=Path(development_report).resolve();dev=json.loads(development_path.read_text())
    if dev.get('status')!='pass' or dev.get('scope')!='development_detector_repair_only':raise ServiceError('development_not_passing','A development recipe must pass before frozen final evaluation')
    recipe_record=json.loads(_verify(dev['recipe_record']).read_text());recipe=_recipe(recipe_record['recipe'])
    if not all(_complete(data['cases'],split) for split in ('development','held_out')):raise ServiceError('incomplete_final_calibration','Both frozen splits require every severe class and independently verified controls')
    original_dev=load_dataset(Path(dev['dataset_path']))
    expected={c['id']:c['image_sha256'] for c in original_dev['cases']}
    if expected!={c['id']:c['image_sha256'] for c in data['cases'] if c['split']=='development'}:raise ServiceError('changed_development_fold','Final evaluation must preserve the exact development fold')
    output=Path(output_directory).resolve()
    if output.exists():raise ServiceError('immutable_final_calibration','Final held-out evaluation needs a fresh explicit evidence directory')
    report=_evaluate(workspace,data['cases'],recipe,output,dataset_path=dataset_path,scope='full_actual_native_known_category_calibration')
    report['development_report']=_bound(development_path)
    report['critic_usable_for_Q09']=report['status']=='pass'
    _immutable(output/'report.json',report)
    if report['critic_usable_for_Q09']:
        _immutable(workspace/'state/qualified_visual_critic.json',{'model':VISION,'calibration_report':str((output/'report.json').relative_to(workspace)),'calibration_sha256':digest(output/'report.json'),
                   'status':'pass','scope':'actual complete known-severe-category and restricted independently clean-control calibration; not a world realism claim'})
    return report
