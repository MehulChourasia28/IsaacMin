"""Frozen, labelled actual-Isaac calibration for the configured visual critic.

This supplements independent validators. API access alone cannot qualify Q09.
"""
from pathlib import Path
import hashlib
import json
import math
import uuid

from isaacmin.assets.network import ServiceError, atomic_json, digest, utcnow
from .nim import NIMClient, VISION

CATEGORIES=('conspicuous_seam','floating_vegetation','missing_material',
            'broken_leaf_opacity','retained_voxel_steps')
PROMPT='''Inspect the attached unaltered Isaac renderer image for visible defects. The image may or may not contain deliberate faults. Report only evidence visible in pixels; neither a generic realism score nor an assumed pass is useful. Inspect every visible area for conspicuous_seam (open terrain crack), floating_vegetation (unsupported plant roots), missing_material (fallback, absent or conspicuously broken texture), broken_leaf_opacity (opaque rectangular foliage cards), retained_voxel_steps (regular block-like stair steps on terrain). Use these exact category names. Other appearance concerns may be reported as other. Sparse foliage alone is not broken_leaf_opacity. Ordinary natural ledges alone are not retained_voxel_steps. Do not infer hidden contact, scene metadata, licence, depth or motion success from RGB. Return ONLY one JSON OBJECT, never a top-level list, with keys defects, uncertainty. defects is a list of objects with exactly category, severity, confidence, bbox, description. severity is severe, moderate or minor; confidence is a number in [0,1]; bbox is normalized [left,top,right,bottom] locating the actual visible fault. Every coordinate MUST be a decimal fraction between0 and1, not pixels or a0..1000 grid. description is a concise observation. uncertainty is a concise string identifying what cannot be assessed, not an invented severe defect. Format example only, not an expected answer: {"defects":[{"category":"floating_vegetation","severity":"severe","confidence":0.9,"bbox":[0.10,0.20,0.40,0.80],"description":"Plant roots visibly hover above the supporting ground"}],"uncertainty":"Hidden surfaces cannot be assessed"}. Return an empty defects list only when no specific fault is visible. No markdown or private reasoning.'''


def validate_finding(content):
    try:
        value=json.loads(content)
        if not isinstance(value,dict) or set(value)!={'defects','uncertainty'} or not isinstance(value['defects'],list) or len(value['defects'])>40 or not isinstance(value['uncertainty'],str):
            raise ValueError()
        for defect in value['defects']:
            if not isinstance(defect,dict) or set(defect)!={'category','severity','confidence','bbox','description'}:
                raise ValueError()
            if defect['category'] not in (*CATEGORIES,'other') or defect['severity'] not in ('severe','moderate','minor'):
                raise ValueError()
            if type(defect['confidence']) not in (float,int) or not math.isfinite(defect['confidence']) or not 0<=defect['confidence']<=1:
                raise ValueError()
            box=defect['bbox']
            if not isinstance(box,list) or len(box)!=4 or not all(type(v) in (float,int) and math.isfinite(v) and 0<=v<=1 for v in box) or box[0]>=box[2] or box[1]>=box[3]:
                raise ValueError()
            if not isinstance(defect['description'],str) or len(defect['description'])>1000:
                raise ValueError()
        return value
    except (ValueError,TypeError,KeyError):
        raise ServiceError('invalid_critic_output','Visual response did not satisfy strict local finding schema') from None


def decode_finding_compat(content):
    """Consume the entire response; collapse identical complete JSON objects only.

    Every value independently satisfies the unchanged finding/coordinate schema.
    No prose stripping, field inference, unit conversion or conflict resolution.
    """
    decoder=json.JSONDecoder();offset=0;values=[];canonical=[]
    if not isinstance(content,str):raise ServiceError('invalid_critic_output','Final response must be text')
    try:
        while offset<len(content):
            while offset<len(content) and content[offset].isspace():offset+=1
            if offset==len(content):break
            value,end=decoder.raw_decode(content,offset)
            checked=validate_finding(json.dumps(value,allow_nan=False))
            values.append(checked);canonical.append(json.dumps(checked,sort_keys=True,separators=(',',':'),allow_nan=False))
            offset=end
    except (ValueError,TypeError):
        raise ServiceError('invalid_critic_output','Entire final response must contain only complete strict finding JSON objects') from None
    if not values or any(v!=canonical[0] for v in canonical[1:]):
        raise ServiceError('conflicting_critic_output','Missing or conflicting consecutive finding objects cannot be normalized')
    return {'finding':values[0],'json_value_count':len(values),'duplicate_count':len(values)-1,
            'transport_normalization':'deduplicate_identical_consecutive_json_values' if len(values)>1 else 'none',
            'wire_schema_status':'fail' if len(values)>1 else 'pass',
            'provider_schema_compliance':len(values)==1}


def replay_protocol_compatibility(workspace: Path,protocol_report: Path):
    """Offline replay of retained provider bytes; no provider request is issued."""
    workspace=Path(workspace).resolve();protocol_report=Path(protocol_report).resolve()
    original=json.loads(protocol_report.read_text())
    replay={'schema_version':1,'created_at_utc':utcnow(),'model':VISION,'source_protocol_report':str(protocol_report),
            'source_protocol_sha256':digest(protocol_report),'network_calls':0,'records':[],
            'decoder':'identical_consecutive_json_v1','coordinate_convention':'normalized_0_1',
            'scope':'Offline serialization compatibility only; critic calibration and user-map qualification not_run'}
    records=[(f"development_{a['repair_cycle']}",a) for a in original['attempts']]+[('held_out',original['held_out_result'])]
    for name,record in records:
        item={'id':name,'original_wire_status':record['status'],'call_id':record.get('call_id')}
        try:item.update(status='pass',**decode_finding_compat(record.get('raw_final_response','')))
        except ServiceError as exc:item.update(status='fail',error=exc.record())
        replay['records'].append(item)
    replay['status']='fixture_protocol_compatible' if all(r['status']=='pass' for r in replay['records']) else 'fail'
    path=protocol_report.parent/'compatibility_replay.json';atomic_json(path,replay)
    if replay['status']=='fixture_protocol_compatible':
        selected=original['attempts'][-1]
        if selected['prompt_sha256']!=original['held_out_result']['prompt_sha256']:
            raise ServiceError('protocol_mismatch','Development/held-out final prompt hashes differ')
        atomic_json(workspace/'state/vision_protocol.json',{'status':replay['status'],'model':VISION,
                    'prompt':selected['prompt'],'prompt_sha256':selected['prompt_sha256'],'reasoning_budget':selected['reasoning_budget'],
                    'decoder':replay['decoder'],'coordinate_convention':'normalized_0_1; no conversion',
                    'evidence':str(path),'evidence_sha256':digest(path),'provider_schema_compliance':False,
                    'critic_calibration':'not_run','user_map_qualification':'not_run'})
    return {**replay,'report_path':str(path)}


def _overlap(a,b):
    return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))>0


def repair_image_protocol(workspace: Path,development_image: Path,held_out_image: Path,max_repairs=3):
    """Bounded format-only development on real RGB, then one untouched held-out test."""
    if not 1<=max_repairs<=3:raise ValueError('Protocol repair budget is one to three cycles')
    workspace=Path(workspace).resolve();development_image=Path(development_image).resolve();held_out_image=Path(held_out_image).resolve()
    if digest(development_image)==digest(held_out_image):raise ServiceError('calibration_leakage','Held-out protocol image must differ')
    used=0
    for previous in (workspace/'evidence/services/vision_protocol').glob('*/protocol.json'):
        old=json.loads(previous.read_text())
        if old.get('model')==VISION and old.get('development_sha256')==digest(development_image) and old.get('held_out_sha256')==digest(held_out_image):
            used+=sum('raw_final_response' in a for a in old.get('attempts',[]))
    if used>=max_repairs:
        raise ServiceError('repair_budget_exhausted','Three-cycle schema repair budget for these frozen images is exhausted; retained bytes may be replayed without new calls')
    folder=workspace/'evidence/services/vision_protocol'/uuid.uuid4().hex;folder.mkdir(parents=True)
    report={'schema_version':1,'created_at_utc':utcnow(),'model':VISION,'scope':'schema_protocol_on_actual_Isaac_synthetic_native_fixture_only',
            'development_image':str(development_image),'development_sha256':digest(development_image),
            'held_out_image':str(held_out_image),'held_out_sha256':digest(held_out_image),
            'coordinate_convention':'normalized_0_1; no inferred or silently converted coordinates',
            'attempts':[],'status':'running','critic_calibration':'not_run','user_map_qualification':'not_run'}
    atomic_json(folder/'protocol.json',report)
    client=NIMClient(workspace,max_attempts=5,deadline=300)
    additions=['', '\nThe response will be parsed by a strict JSON validator. Start with {"defects": and include a separate string uncertainty. A category is never Uncertainty. All bbox coordinates must be at most1.0.',
               '\nMandatory output contract: one JSON object, defects array, uncertainty string. Each defect has exactly category, severity, confidence, bbox, description. Coordinates are FRACTIONS in [0,1], such as0.25. No1000-grid coordinates. Do not output a top-level array.']
    selected=None
    for cycle in range(used,max_repairs):
        prompt=PROMPT+additions[cycle];attempt={'repair_cycle':cycle+1,'prompt':prompt,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                                              'reasoning_budget':128 if cycle==0 else 512}
        try:
            response=client.inspect_image(development_image,prompt,max_tokens=2048,reasoning_budget=attempt['reasoning_budget'],temperature=0)
            attempt.update(call_id=response['call_id'],raw_final_response=response['message'].get('content',''))
            attempt['finding']=validate_finding(attempt['raw_final_response']);attempt['status']='pass'
            selected=attempt
        except ServiceError as exc:
            attempt.update(status='fail' if exc.code=='invalid_critic_output' else 'blocked',error=exc.record())
        report['attempts'].append(attempt);atomic_json(folder/'protocol.json',report)
        if selected is not None or attempt['status']=='blocked':break
    if selected is not None:
        held={'prompt_sha256':selected['prompt_sha256']}
        try:
            response=client.inspect_image(held_out_image,selected['prompt'],max_tokens=2048,reasoning_budget=selected['reasoning_budget'],temperature=0)
            held.update(call_id=response['call_id'],raw_final_response=response['message'].get('content',''))
            held['finding']=validate_finding(held['raw_final_response']);held['status']='pass'
        except ServiceError as exc:
            held.update(status='fail' if exc.code=='invalid_critic_output' else 'blocked',error=exc.record())
        report['held_out_result']=held
        report['status']='fixture_schema_verified' if held['status']=='pass' else held['status']
    else:
        report['status']=report['attempts'][-1]['status'];report['held_out_result']={'status':'not_run'}
    atomic_json(folder/'protocol.json',report)
    if report['status']=='fixture_schema_verified':
        atomic_json(workspace/'state/vision_protocol.json',{'model':VISION,'status':report['status'],
                    'prompt':selected['prompt'],'prompt_sha256':selected['prompt_sha256'],'reasoning_budget':selected['reasoning_budget'],
                    'coordinate_convention':report['coordinate_convention'],'evidence':str(folder/'protocol.json'),'evidence_sha256':digest(folder/'protocol.json'),
                    'critic_calibration':'not_run','user_map_qualification':'not_run'})
    return {**report,'report_path':str(folder/'protocol.json')}


def evaluate_case(case,finding):
    severe=[d for d in finding['defects'] if d['severity']=='severe' and d['confidence']>=.5]
    expected=case.get('expected_severe',[])
    missed=[e for e in expected if not any(d['category']==e['category'] and _overlap(d['bbox'],e['bbox']) for d in severe)]
    false_positives=[d for d in severe if d['category'] in case.get('known_absent_categories',[])]
    return {'severe_faults':len(expected),'detected':len(expected)-len(missed),'missed':missed,
            'false_positives':false_positives,'status':'fail' if missed or false_positives else 'pass',
            'uninjected_control_is_clean':bool(case.get('clean_control_independently_verified',False))}


def load_dataset(path: Path):
    path=Path(path).resolve();data=json.loads(path.read_text())
    rejection=path.parent/'independent_rejection.json'
    if rejection.exists():
        rejected=json.loads(rejection.read_text())
        if rejected.get('status')=='fail' and rejected.get('dataset_sha256')==digest(path):
            raise ServiceError('rejected_actual_capture','Independent inspection rejected this exact actual capture set; preserve it and use a repaired new capture')
    if data.get('schema_version')!=1 or data.get('source')!='actual_isaac_raw_rgb' or not isinstance(data.get('cases'),list):
        raise ServiceError('invalid_calibration_dataset','Calibration requires an actual Isaac raw capture manifest')
    seen=set();fold_images={'development':set(),'held_out':set()}
    for case in data['cases']:
        if case['id'] in seen or case['split'] not in fold_images:
            raise ServiceError('invalid_calibration_split','Case identifiers and development/held-out splits must be distinct')
        seen.add(case['id'])
        image=Path(case['image']).resolve()
        if digest(image)!=case['image_sha256']:
            raise ServiceError('changed_capture','Frozen actual renderer image changed')
        metadata=Path(case['capture_metadata']).resolve()
        if digest(metadata)!=case['capture_metadata_sha256']:
            raise ServiceError('changed_capture_metadata','Actual renderer provenance changed')
        provenance=json.loads(metadata.read_text())
        if provenance.get('renderer')!='RayTracedLighting' or provenance.get('raw_image_sha256')!=case['image_sha256'] or provenance.get('postprocessing')!='none':
            raise ServiceError('unverified_capture','Calibration image lacks matching raw Isaac capture provenance')
        if provenance.get('variant_scene_sha256'):
            variant=Path(provenance.get('variant_scene',metadata.parent/'calibration_only.usda'))
            if not variant.is_file() or digest(variant)!=provenance['variant_scene_sha256']:
                raise ServiceError('changed_calibration_variant','Captured native mutation-layer bytes changed')
        dependency_record=provenance.get('captured_scene_dependencies')
        if case.get('expected_severe') or case.get('clean_control_independently_verified'):
            if not dependency_record:
                raise ServiceError('unbound_calibration_scene','Labelled calibration image must bind actual USD content layers and textures, not only the root wrapper')
            for bound in [dependency_record['manifest'],*dependency_record['files']]:
                if digest(Path(bound['path']))!=bound['sha256']:
                    raise ServiceError('changed_calibration_scene','Labelled calibration scene dependency bytes changed')
        for item in case.get('expected_severe',[]):
            if item['category'] not in CATEGORIES or provenance.get('injected_category')!=item['category'] or not provenance.get('mutation',{}).get('changed_elements',0):
                raise ServiceError('unverified_mutation','Fault label lacks matching native USD mutation evidence')
            validation=case.get('fault_validation',{})
            label_path=Path(validation.get('path',''))
            if not label_path.is_file() or digest(label_path)!=validation.get('sha256'):
                raise ServiceError('unverified_severe_label','Visible severe labels require immutable independent actual-image observations')
            observation=json.loads(label_path.read_text())
            if observation.get('image_sha256')!=case['image_sha256'] or not observation.get('obvious_visible_severe') or observation.get('category')!=item['category']:
                raise ServiceError('unverified_severe_label','Severe observation does not match the frozen actual capture')
        if case.get('clean_control_independently_verified'):
            validation=case.get('control_validation',{})
            control_path=Path(validation.get('path',''))
            if not control_path.is_file() or digest(control_path)!=validation.get('sha256'):
                raise ServiceError('unverified_control','Clean-control claim lacks immutable independent validation')
            control=json.loads(control_path.read_text())
            if control.get('status')!='pass' or control.get('image_sha256')!=case['image_sha256'] or set(control.get('absent_categories',[]))!=set(CATEGORIES):
                raise ServiceError('unverified_control','Control validation does not establish all supported categories absent in this image')
        fold_images[case['split']].add(case['image_sha256'])
    if fold_images['development'] & fold_images['held_out']:
        raise ServiceError('calibration_leakage','Held-out images duplicate development bytes')
    return data


def capture_calibration(workspace: Path,scene: Path,baseline_capture: Path,output_directory: Path,*,frozen_poses=None,categories=CATEGORIES):
    """Run sequentially after the caller releases every other heavy native worker.

    baseline_capture is an actual capture_result.json with distinct frozen static
    development/held-out poses. Its RGB files remain untouched.
    """
    from isaacmin.process import run_worker
    workspace=Path(workspace).resolve();baseline_capture=Path(baseline_capture).resolve()
    baseline=json.loads(baseline_capture.read_text())
    frames=baseline.get('frames',[])
    choices={}
    if frozen_poses is not None:
        if set(frozen_poses)!={'development','held_out'}:
            raise ServiceError('invalid_calibration_poses','Explicit frozen development and held-out poses are both required')
        for split,pose in frozen_poses.items():
            for key in ('position','look_at'):
                if len(pose.get(key,[]))!=3 or not all(type(v) in (int,float) and math.isfinite(v) for v in pose[key]):
                    raise ServiceError('invalid_calibration_poses','Explicit calibration camera coordinates must be finite XYZ')
            choices[split]=pose
        if choices['development']['position']==choices['held_out']['position'] and choices['development']['look_at']==choices['held_out']['look_at']:
            raise ServiceError('calibration_leakage','Development and held-out calibration poses must differ')
    for split,held_out in ([] if frozen_poses is not None else [('development',False),('held_out',True)]):
        eligible=[f for f in frames if f.get('pose',{}).get('kind')=='static' and bool(f['pose'].get('held_out'))==held_out]
        if not eligible:
            raise ServiceError('missing_calibration_poses','Actual baseline lacks distinct frozen static development/held-out captures')
        chosen=eligible[0]
        image=baseline_capture.parent/chosen['rgb']
        if digest(image)!=chosen['rgb_sha256']:
            raise ServiceError('changed_capture','Original baseline RGB bytes changed')
        choices[split]=chosen['pose']
    if not categories or len(set(categories))!=len(categories) or not set(categories)<=set(CATEGORIES):
        raise ServiceError('invalid_calibration_categories','Choose distinct supported native fault classes')
    output=Path(output_directory).resolve()/uuid.uuid4().hex
    output.mkdir(parents=True)
    request={'scene':str(Path(scene).resolve()),'output':str(output),'poses':choices,'categories':['baseline',*categories],
             'baseline_capture':str(baseline_capture),'baseline_capture_sha256':digest(baseline_capture),
             'classification':'calibration_only_never_release_world'}
    atomic_json(output/'request.json',request)
    config=workspace/'state/native_tools.json'
    runtime=Path(json.loads(config.read_text())['isaac_python']) if config.exists() else Path.home()/'IsaacSim/_build/linux-aarch64/release/python.sh'
    if not runtime.is_file():
        raise ServiceError('blocked_dependency','Pinned native Isaac Python launcher unavailable')
    result=run_worker([str(runtime),str(workspace/'isaac_scripts/critic_calibration_capture.py'),'--request',str(output/'request.json')],
                      cwd=workspace,log_path=output/'worker.log',timeout=7200,environment={'LD_LIBRARY_PATH':''})
    if result['exit_code']!=0 or not (output/'dataset.json').is_file():
        raise ServiceError('calibration_capture_failed','Actual Isaac calibration capture failed; logs and partial captures retained')
    dataset=load_dataset(output/'dataset.json')
    return {'status':dataset.get('status','incomplete'),'dataset':str(output/'dataset.json'),'process':result}


def combine_calibration_datasets(dataset_paths,output_path):
    """Bind retained controls and reviewed fault cases without changing originals."""
    cases=[];inputs=[]
    for path in dataset_paths:
        path=Path(path).resolve();data=load_dataset(path);cases.extend(data['cases'])
        inputs.append({'path':str(path),'sha256':digest(path)})
    output_path=Path(output_path).resolve()
    if output_path.exists():raise ServiceError('immutable_calibration','Combined calibration manifest must use a new path')
    result={'schema_version':1,'source':'actual_isaac_raw_rgb','cases':cases,'parent_datasets':inputs,
            'created_at_utc':utcnow(),'scope':'Known category calibration only; never source-world realism qualification'}
    atomic_json(output_path,result)
    load_dataset(output_path)
    return result


def publish_fault_labels(dataset_path,observations_path,output_path):
    """Publish independently inspected severe labels; pixel delta alone is not one.

    observations is a list of exact image-bound records with id, image_sha256,
    category, bbox, obvious_visible_severe, observation and observer. The observer
    must inspect original actual RGB and native mutation evidence, independently
    of the visual critic being calibrated. Unobserved/ambiguous cases stay unlabelled.
    """
    dataset_path=Path(dataset_path).resolve();output_path=Path(output_path).resolve()
    data=load_dataset(dataset_path);observations_path=Path(observations_path).resolve()
    observations=json.loads(observations_path.read_text());by_id={c['id']:c for c in data['cases']}
    if output_path.exists():raise ServiceError('immutable_calibration','Reviewed labels must use a new output manifest')
    label_folder=output_path.parent/(output_path.stem+'_labels');label_folder.mkdir(parents=True,exist_ok=False)
    for observation in observations:
        case=by_id.get(observation.get('id'))
        if case is None or observation.get('image_sha256')!=case['image_sha256'] or not observation.get('observer') or not observation.get('observation'):
            raise ServiceError('severe_observation_provenance','Independent observation must bind an existing original frame and identify its observer')
        if not observation.get('obvious_visible_severe'):continue
        finding={'defects':[{'category':observation['category'],'severity':'severe','confidence':1,
                           'bbox':observation['bbox'],'description':observation['observation']}],'uncertainty':'Controlled known-category label only'}
        validate_finding(json.dumps(finding))
        meta=json.loads(Path(case['capture_metadata']).read_text())
        if observation['category'] not in CATEGORIES or meta.get('injected_category')!=observation['category'] or not meta.get('mutation',{}).get('changed_elements'):
            raise ServiceError('unverified_mutation','Observed category lacks corresponding native mutation')
        # Output names derive from a digest, never from unconstrained case text.
        target=label_folder/(hashlib.sha256(case['id'].encode()).hexdigest()+'.json');atomic_json(target,observation)
        case['expected_severe']=[{'category':observation['category'],'bbox':observation['bbox']}]
        case['fault_validation']={'path':str(target),'sha256':digest(target)}
    data['label_provenance']={'original_dataset':str(dataset_path),'original_dataset_sha256':digest(dataset_path),
                              'observations':str(observations_path),'observations_sha256':digest(observations_path)}
    atomic_json(output_path,data);load_dataset(output_path);return data


def calibrate_critic(workspace: Path,dataset_path: Path,output_directory: Path|None=None):
    """Real Omni requests only; never rechecks persisted authentication failures."""
    workspace=Path(workspace).resolve();dataset_path=Path(dataset_path).resolve()
    dataset=load_dataset(dataset_path)
    prompt,reasoning_budget=PROMPT,128
    protocol={}
    protocol_path=workspace/'state/vision_protocol.json'
    if protocol_path.exists():
        protocol=json.loads(protocol_path.read_text())
        if protocol.get('model')!=VISION or protocol.get('status') not in ('fixture_schema_verified','fixture_protocol_compatible') or hashlib.sha256(protocol['prompt'].encode()).hexdigest()!=protocol['prompt_sha256'] or digest(Path(protocol['evidence']))!=protocol['evidence_sha256']:
            raise ServiceError('changed_vision_protocol','Stored schema protocol evidence is invalid or changed')
        prompt,reasoning_budget=protocol['prompt'],protocol['reasoning_budget']
    output=Path(output_directory) if output_directory else workspace/'evidence/services/critic_calibration'/uuid.uuid4().hex
    output.mkdir(parents=True,exist_ok=True)
    report={'schema_version':1,'created_at_utc':utcnow(),'status':'running','model':VISION,
            'dataset_sha256':digest(dataset_path),'dataset_path':str(dataset_path),
            'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'prompt':prompt,'reasoning_budget':reasoning_budget,
            'decoder':protocol.get('decoder','strict_json_v1'),
            'cases':[],'scope':'Static RGB supported-category calibration; no motion, contact or universal realism claim'}
    atomic_json(output/'report.json',report)
    client=NIMClient(workspace,deadline=900)
    for case in dataset['cases']:
        record={'id':case['id'],'split':case['split'],'image_sha256':case['image_sha256']}
        try:
            response=client.inspect_image(Path(case['image']),prompt,max_tokens=2048,reasoning_budget=reasoning_budget,temperature=0)
            record['raw_final_response']=response['message'].get('content','')
            if protocol_path.exists() and protocol.get('decoder')=='identical_consecutive_json_v1':
                decoded=decode_finding_compat(response['message'].get('content',''))
            else:
                decoded={'finding':validate_finding(response['message'].get('content','')),'transport_normalization':'none','duplicate_count':0,'wire_schema_status':'pass'}
            finding=decoded['finding']
            record.update(call_id=response['call_id'],**decoded,**evaluate_case(case,finding))
        except ServiceError as exc:
            record.update(status='blocked' if exc.code in ('access_restricted','invalid_key','network','provider_outage','rate_limit','request_deadline') else 'fail',error=exc.record())
            report['cases'].append(record)
            atomic_json(output/'report.json',report)
            if record['status']=='blocked':
                report['status']='blocked';report['remaining_cases']='not_run';break
            continue
        report['cases'].append(record)
        atomic_json(output/'report.json',report)
    coverage={split:sorted({e['category'] for c in dataset['cases'] if c['split']==split for e in c.get('expected_severe',[])}) for split in ('development','held_out')}
    controls={split:sum(c.get('clean_control_independently_verified',False) and not c.get('expected_severe') for c in dataset['cases'] if c['split']==split) for split in coverage}
    report.update(category_coverage=coverage,verified_clean_controls=controls,
                  severe_missed=sum(len(c.get('missed',[])) for c in report['cases']),
                  false_positives=sum(len(c.get('false_positives',[])) for c in report['cases']))
    if report['status']!='blocked':
        report['status']='fail' if any(c['status']=='fail' for c in report['cases']) else 'pass' if all(set(v)==set(CATEGORIES) for v in coverage.values()) and all(controls.values()) else 'incomplete'
    report['critic_usable_for_Q09']=report['status']=='pass'
    report['limitations']=['Actual calibration does not establish universal visual judgement',
                           'Uninjected baseline is not presumed clean; independent control verification required',
                           'All obvious severe faults must be detected; no aggregate score can erase a miss']
    atomic_json(output/'report.json',report)
    capability_path=workspace/'state/nim_capabilities.json'
    if capability_path.exists():
        capabilities=json.loads(capability_path.read_text())
        availability_path=workspace/'state/nim_availability.json'
        availability=json.loads(availability_path.read_text()).get('models',{}).get(VISION,{}) if availability_path.exists() else {}
        capabilities.setdefault('current_access',{})['vision']={**availability,'actual_capture_calibration_status':report['status']}
        capabilities['critic_isaac_calibration']={'status':report['status'],'report':str(output/'report.json'),
                                                'sha256':digest(output/'report.json'),'usable_for_Q09':report['critic_usable_for_Q09']}
        atomic_json(capability_path,capabilities)
    return report
