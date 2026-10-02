"""Immutable actual-renderer RGB review, separate from construction and Q10.

Public API: freeze_review(), inspect_frozen(), construction_defects(),
record_repair(). See docs/SERVICES_API.md. Importing this module makes no calls.
"""
from collections import Counter
from pathlib import Path
import fcntl
import hashlib
import json
import os

from isaacmin.assets.network import ServiceError, digest, utcnow
from isaacmin.assets.usd_provenance import verify_scene_closure,captured_closure_hash,validated_native_renderer
from .critic import decode_finding_compat, validate_finding, CATEGORIES
from .nim import NIMClient, VISION


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _read(path):
    return json.loads(Path(path).read_text())


def _bound(path):
    path=Path(path).resolve()
    if not path.is_file():
        raise ServiceError('missing_review_input', 'A required review input does not exist')
    return {'path':str(path),'sha256':digest(path)}


def _verify(bound):
    if digest(Path(bound['path']))!=bound['sha256']:
        raise ServiceError('changed_review_input','Frozen review input bytes changed; create a new candidate review')
    return Path(bound['path'])


def _immutable(path, value):
    """Publish one complete record without overwriting a concurrent/prior result."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    encoded=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    temporary=path.parent/(path.name+'.'+str(os.getpid())+'.partial')
    with temporary.open('xb') as handle:
        handle.write(encoded);handle.flush();os.fsync(handle.fileno())
    try:
        os.link(temporary,path)
    except FileExistsError:
        if path.read_bytes()!=encoded:
            raise ServiceError('immutable_review_conflict','A frozen evidence record already exists with different bytes') from None
    finally:
        temporary.unlink()
    return _bound(path)


def freeze_review(workspace, capture_results, quality_profile, output_directory, *,
                  reference_ids=('birchwood','alps_field'), required_features=(),
                  required_material_families=(), required_asset_families=(), visibility_report=None):
    """Bind raw static captures and real licensed photographs before model calls.

    Required coverage comes from the actual scene/source inventory. Pose tags are
    intent only; observed coverage requires separately bound native measurements.
    At least24 distinct static poses, three balanced heights and both lighting
    conditions are required for complete view coverage. No image is rewritten.
    """
    workspace=Path(workspace).resolve();output=Path(output_directory).resolve()
    manifest_path=output/'frozen_review.json'
    if manifest_path.exists():
        raise ServiceError('already_frozen','Use the existing manifest to resume; a changed world needs a new review directory')
    records=[];scenes={};poses={};image_folds={'development':set(),'held_out':set()}
    files=[]
    for item in capture_results:
        path=Path(item).resolve();capture=_read(path);files.append(_bound(path))
        validated_native_renderer(capture)
        if 'synthetic' in capture.get('scope','').lower() or not capture.get('scope','').startswith('real_source'):
            raise ServiceError('not_user_map_capture','User-map review requires actual Isaac real_source captures; synthetic fixtures cannot qualify it')
        scene=_bound(capture['scene'])
        if scene['sha256']!=capture['scene_sha256']:
            raise ServiceError('changed_scene','Captured scene bytes no longer match the retained renderer record')
        dependencies=verify_scene_closure(scene['path'],captured_closure_hash(capture))
        scenes[_hash([scene['sha256'],dependencies['manifest']['sha256']])]={'root':scene,'dependencies':dependencies}
        for frame in capture.get('frames',[]):
            pose=frame.get('pose',{})
            if pose.get('kind')!='static':
                continue
            if type(pose.get('held_out')) is not bool:
                raise ServiceError('unfrozen_split','Every static pose needs a predeclared held_out boolean')
            if not all(k in frame for k in ('camera_world_matrix_row_vectors','simulation_time_s','depth_semantics')):
                raise ServiceError('missing_capture_calibration','Actual pose, timestamp and depth convention are required')
            split='held_out' if pose['held_out'] else 'development'
            rgb=_bound(path.parent/frame['rgb']);depth=_bound(path.parent/frame['depth'])
            if rgb['sha256']!=frame['rgb_sha256'] or depth['sha256']!=frame['depth_sha256']:
                raise ServiceError('changed_capture','Original RGB/depth bytes changed')
            pose_id=_hash({k:pose[k] for k in ('position','look_at')})
            if pose_id in poses and poses[pose_id]['split']!=split:
                raise ServiceError('held_out_leakage','The same pose cannot be development in one light and held out in another')
            poses.setdefault(pose_id,{'split':split,'scout_height_m':pose.get('scout_height_m'),'lighting':set()})['lighting'].add(capture.get('lighting'))
            image_folds[split].add(rgb['sha256'])
            records.append({'id':_hash([files[-1]['sha256'],frame['frame']])[:24],
                            'split':split,'pose_id':pose_id,'pose':pose,'rgb':rgb,'depth':depth,
                            'capture_result':files[-1],'frame':frame['frame'],
                            'lighting':capture.get('lighting'),'scene_sha256':scene['sha256'],
                            'scene_dependencies_sha256':dependencies['manifest']['sha256'],
                            'simulation_time_s':frame['simulation_time_s'],
                            'camera_world_matrix_row_vectors':frame['camera_world_matrix_row_vectors'],
                            'depth_semantics':frame['depth_semantics'],
                            'capture_contact_status':capture.get('status'),
                            'postprocessing':'none; original recorded PNG bytes retained'})
    if not records or len(scenes)!=1:
        raise ServiceError('review_scene_identity','One exact source scene and at least one actual static capture are required')
    if image_folds['development'] & image_folds['held_out']:
        raise ServiceError('held_out_leakage','Held-out and development captures duplicate image bytes')
    catalogue_path=workspace/'state/reference_catalogue.json';catalogue=_read(catalogue_path)
    references=[]
    for identity in reference_ids:
        reference=next((r for r in catalogue['references'] if r['asset_id']==identity),None)
        if reference is None or reference.get('licence')!='CC0-1.0' or not reference.get('not_scene_evidence') or reference.get('classification')!='photographic_hdr_panorama_provider_tonemapped':
            raise ServiceError('reference_provenance','Requested reference must be a retained, licensed actual photograph')
        _verify(reference['board']);_verify({'path':reference['original_path'],'sha256':reference['original_sha256']})
        references.append(reference)
    heights=Counter(p['scout_height_m'] for p in poses.values())
    held=sum(p['split']=='held_out' for p in poses.values());gaps=[]
    if len(poses)<24:gaps.append('fewer_than24_distinct_static_poses')
    if any(heights.get(h,0)<8 for h in (.25,.6,1.5)):gaps.append('fewer_than8_poses_at_each_scout_height')
    if held/len(poses)<.25:gaps.append('held_out_fraction_below25percent')
    if any(p['lighting']!={'diffuse','directional'} for p in poses.values()):gaps.append('not_every_static_pose_captured_in_both_lighting_conditions')
    observed={key:set() for key in ('features','material_families','asset_families')}
    planned={key:set() for key in observed}
    visibility_bound=None;visibility={}
    if visibility_report:
        from isaacmin.assets.view_coverage import validate_view_coverage
        visibility_bound,visibility=validate_view_coverage(visibility_report,records)
    else:gaps.append('missing_actual_visibility_evidence')
    for r in records:
        for key in observed:
            planned[key].update(r['pose'].get(key,[]))
        if r['pose'].get('surface_scope'):planned['features'].add(r['pose']['surface_scope'])
        measurement=visibility.get((r['capture_result']['sha256'],r['frame'],r['rgb']['sha256']))
        r['observed_coverage']=measurement
        if measurement:
            for key in observed:observed[key].update(measurement[key])
    if visibility_bound and any(r['observed_coverage'] is None for r in records):gaps.append('some_actual_frames_lack_visibility_measurements')
    required={'features':list(required_features),'material_families':list(required_material_families),'asset_families':list(required_asset_families)}
    missing={k:sorted(set(required[k])-observed[k]) for k in required}
    gaps.extend('unrepresented_'+k+':'+v for k,values in missing.items() for v in values)
    if not required_material_families:gaps.append('used_material_family_inventory_not_supplied')
    if not required_asset_families:gaps.append('used_asset_family_inventory_not_supplied')
    if not references:gaps.append('no_licensed_photographic_references')
    value={'schema_version':1,'created_at_utc':utcnow(),'source':'actual_isaac_raw_rgb',
           'renderer_protocol':_bound(Path(__file__).parents[1]/'assets/usd_provenance.py'),
           'scene':next(iter(scenes.values()))['root'],'scene_dependencies':next(iter(scenes.values()))['dependencies'],
           'quality_profile':_bound(quality_profile),
           'capture_results':files,'reference_catalogue':_bound(catalogue_path),'references':references,
           'visibility_report':visibility_bound,
           'frames':records,'coverage':{'status':'complete' if not gaps else 'incomplete',
             'distinct_static_poses':len(poses),'scout_height_counts':{str(k):v for k,v in heights.items()},
             'held_out_fraction':held/len(poses),'required':required,'missing':missing,'gaps':gaps,
             'observed':{k:sorted(v) for k,v in observed.items()},'planned_intent_only':{k:sorted(v) for k,v in planned.items()}},
           'qualification':'not_run','motion_qualification':'separate_Q10; still images cannot establish motion quality',
           'held_out_policy':'Frozen image/pose split is evaluation-only; construction_defects returns development findings exclusively'}
    _immutable(manifest_path,value)
    return {'manifest':str(manifest_path),'sha256':digest(manifest_path),'coverage':value['coverage']}


def _load_frozen(path):
    value=_read(path)
    if value.get('renderer_protocol'):_verify(value['renderer_protocol'])
    for field in ('scene','quality_profile','reference_catalogue'):_verify(value[field])
    _verify(value['scene_dependencies']['manifest'])
    for bound in value['scene_dependencies']['files']:_verify(bound)
    for bound in value['capture_results']:_verify(bound)
    for frame in value['frames']:
        _verify(frame['rgb']);_verify(frame['depth'])
    if value.get('visibility_report'):
        from isaacmin.assets.view_coverage import validate_view_coverage
        validate_view_coverage(_verify(value['visibility_report']),value['frames'])
    for reference in value['references']:
        _verify(reference['board']);_verify({'path':reference['original_path'],'sha256':reference['original_sha256']})
    return value


def _calibration(path):
    result=_read(path)
    if result.get('model')!=VISION or result.get('status')!='pass' or not result.get('critic_usable_for_Q09'):
        raise ServiceError('critic_not_calibrated','Actual injected-fault and independently clean-control calibration has not passed')
    if hashlib.sha256(result['prompt'].encode()).hexdigest()!=result['prompt_sha256']:
        raise ServiceError('calibration_changed','Calibrated prompt hash is inconsistent')
    if digest(Path(result['dataset_path']))!=result['dataset_sha256']:
        raise ServiceError('calibration_changed','Calibrated dataset manifest changed')
    from .critic import load_dataset
    dataset=load_dataset(Path(result['dataset_path']))
    expected={c['id'] for c in dataset['cases']}
    if expected!={c['id'] for c in result.get('cases',[])} or any(c.get('status')!='pass' for c in result.get('cases',[])):
        raise ServiceError('calibration_incomplete','Calibration cases do not all have successful retained evaluations')
    if any(set(result.get('category_coverage',{}).get(s,[]))!=set(CATEGORIES) or not result.get('verified_clean_controls',{}).get(s) for s in ('development','held_out')):
        raise ServiceError('calibration_incomplete','All severe categories and independently verified clean controls are required in both splits')
    return result


def _defects(frame,finding,model_record):
    repair_classes={'conspicuous_seam':'terrain_join','floating_vegetation':'asset_grounding',
                    'missing_material':'material_binding','broken_leaf_opacity':'leaf_material',
                    'retained_voxel_steps':'terrain_shape','other':'appearance_investigation'}
    return [{'defect_id':_hash([frame['id'],d]),'category':d['category'],'severity':d['severity'],
             'world_bounds':None,'world_bounds_reason':'RGB bounding box alone does not establish geometric world bounds',
             'frame_id':frame['id'],'split':frame['split'],'region':frame['scene_sha256'],
             'evidence':[frame['rgb'],model_record],'pixel_bbox_normalized':d['bbox'],
             'originating_validator':VISION,'confidence':d['confidence'],'observation':d['description'],
             'classification':'visual_suspicion_requires_reproduction_or_independent_geometry',
             'reproducibility':'not_run','proposed_repair_class':repair_classes[d['category']],
             'attempts':0,'resolution_evidence':[]} for d in finding['defects']]


REFERENCE_PROMPT='''The first image is an unaltered actual Isaac scene capture. The second is a labelled board projected from a licensed real photographic panorama at a different location. Compare ground texture scale, natural clustering, canopy/plant silhouettes, visible leaf-card artifacts, contact appearance, surface gloss and cave/exterior lighting only where the photographic conditions support comparison. Do not expect identical plants, geography, pose, exposure or lighting. Never treat the reference photograph as captured Isaac geometry or metric ground truth. Localize specific visible concerns ONLY in the FIRST image. No generic realism score. Return one JSON object with exactly defects and uncertainty. Each defect has category (conspicuous_seam,floating_vegetation,missing_material,broken_leaf_opacity,retained_voxel_steps,other), severity (severe,moderate,minor), confidence (0..1), bbox ([left,top,right,bottom] fractional0..1 on FIRST image), description. defects may be empty. uncertainty must state relevant capture/view/season limitations. This comparison is supplemental, not the calibrated fault detector.'''


def inspect_frozen(workspace, manifest_path, calibration_report, output_directory):
    """Resume exact calibrated single-image detection and supplemental photo review.

    Makes real NVIDIA requests. Call only after populated user-map captures exist.
    Authentication denials remain enforced by NIMClient. Supplemental two-image
    comparison is explicitly uncalibrated and cannot promote a quality gate.
    """
    workspace=Path(workspace).resolve();manifest_path=Path(manifest_path).resolve();output=Path(output_directory).resolve()
    frozen=_load_frozen(manifest_path);calibration=_calibration(calibration_report)
    session={'manifest':_bound(manifest_path),'calibration':_bound(calibration_report),'model':VISION,
             'prompt_sha256':calibration['prompt_sha256'],'supplemental_prompt_sha256':hashlib.sha256(REFERENCE_PROMPT.encode()).hexdigest(),
             'decoder':calibration.get('decoder','strict_json_v1')}
    _immutable(output/'session.json',session)
    client=NIMClient(workspace,deadline=900);records=[];defects=[]
    for frame in frozen['frames']:
        record_path=output/frame['split']/(frame['id']+'.json')
        if record_path.exists():
            record=_read(record_path)
            if record.get('session_sha256')!=digest(output/'session.json') or record.get('image_sha256')!=frame['rgb']['sha256']:
                raise ServiceError('changed_review_session','Retained frame result belongs to another frozen session')
        else:
            record={'schema_version':1,'created_at_utc':utcnow(),'frame_id':frame['id'],'split':frame['split'],
                    'image_sha256':frame['rgb']['sha256'],'session_sha256':digest(output/'session.json')}
            try:
                response=client.inspect_image(Path(frame['rgb']['path']),calibration['prompt'],max_tokens=2048,
                                              reasoning_budget=calibration['reasoning_budget'],temperature=0)
                record.update(call_id=response['call_id'],raw_final_response=response['message'].get('content',''))
                decoded=decode_finding_compat(record['raw_final_response']) if session['decoder']=='identical_consecutive_json_v1' else {'finding':validate_finding(record['raw_final_response']),'transport_normalization':'none','duplicate_count':0,'wire_schema_status':'pass'}
                record.update(status='inspected',**decoded)
            except ServiceError as exc:
                record.update(status='blocked' if exc.retryable or exc.code in ('access_restricted','invalid_key') else 'failed_schema',error=exc.record())
            _immutable(record_path,record)
        records.append(record)
        if record['status']!='inspected':
            break
        for defect in _defects(frame,record['finding'],_bound(record_path)):
            _immutable(output/'defects'/frame['split']/(defect['defect_id']+'.json'),defect);defects.append(defect)
    # References are inspected against development frames only. Held-out locations
    # never become construction-time photo-selection hints.
    comparisons=[]
    if len(records)==len(frozen['frames']) and all(r['status']=='inspected' for r in records):
        for condition in ('diffuse','directional'):
            candidates=[f for f in frozen['frames'] if f['split']=='development' and f['lighting']==condition]
            if not candidates:continue
            for reference in frozen['references']:
                # Source-tagged forest/meadow views are preferred; no inferred species label.
                tag='forest' if reference['asset_id']=='birchwood' else 'meadow'
                frame=next((f for f in candidates if tag in f['pose'].get('features',[])),candidates[0])
                identity=_hash([frame['id'],reference['board']['sha256']]);target=output/'reference_comparisons'/(identity+'.json')
                if target.exists():comparison=_read(target)
                else:
                    comparison={'frame_id':frame['id'],'reference_asset_id':reference['asset_id'],
                                'reference':reference['board'],'capture':frame['rgb'],'supplemental_uncalibrated':True,
                                'reference_limitations':reference['limitations'],'session_sha256':digest(output/'session.json')}
                    try:
                        response=client.inspect_images([Path(frame['rgb']['path']),Path(reference['board']['path'])],REFERENCE_PROMPT,max_tokens=2048,reasoning_budget=512,temperature=0)
                        comparison.update(call_id=response['call_id'],raw_final_response=response['message'].get('content',''))
                        comparison.update(status='inspected',**decode_finding_compat(comparison['raw_final_response']))
                    except ServiceError as exc:
                        comparison.update(status='blocked',error=exc.record())
                    _immutable(target,comparison)
                comparisons.append(comparison)
                if comparison['status']=='inspected':
                    for defect in _defects(frame,comparison['finding'],_bound(target)):
                        defect['defect_id']=_hash([identity,defect['defect_id']]);defect['classification']='supplemental_uncalibrated_reference_comparison_suspicion'
                        _immutable(output/'defects/development'/(defect['defect_id']+'.json'),defect);defects.append(defect)
                else:break
            if comparisons and comparisons[-1]['status']!='inspected':break
    held=[d for d in defects if d['split']=='held_out']
    summary={'schema_version':1,'session':_bound(output/'session.json'),'inspected_frames':sum(r['status']=='inspected' for r in records),
             'expected_frames':len(frozen['frames']),'coverage':frozen['coverage'],
             'model_observation_status':'incomplete' if len(records)!=len(frozen['frames']) or any(r['status']!='inspected' for r in records) else 'findings_require_resolution' if defects else 'no_reported_findings',
             'reference_comparisons':len(comparisons),'reference_comparison_failures':sum(c['status']!='inspected' for c in comparisons),
             'held_out_aggregate':{'findings':len(held),'severe':sum(d['severity']=='severe' for d in held)},
             'unresolved_severe':sum(d['severity']=='severe' for d in defects),
             'Q09':'not_promoted; requires complete coverage, resolved reproducible findings and independent material/geometric gates',
             'limitations':['Supplemental photograph comparisons are not calibrated fault detection',
                            'Empty model findings do not establish objective realism, contact, sensor or temporal correctness']}
    _immutable(output/'summary.json',summary)
    return {'summary_path':str(output/'summary.json'),**summary}


def construction_defects(review_directory):
    """Only development findings; never held-out pose/frame/box metadata."""
    return [_read(p) for p in sorted((Path(review_directory)/'defects/development').glob('*.json'))]


def record_repair(review_directory, defect_id, candidate_before, candidate_after, recipe_record,
                  gate_evidence, *, outcome, ledger_directory=None):
    """Append one attributable development repair; max3 per class for this review.

    No tools are executed here. Only immutable candidate bytes and actual affected
    gate reports are accepted. A held-out finding is not a repair target.
    """
    directory=Path(review_directory).resolve();matches=[d for d in construction_defects(directory) if d['defect_id']==defect_id]
    if len(matches)!=1:raise ServiceError('held_out_or_unknown_defect','Repair must name a development finding, never a held-out location')
    defect=matches[0]
    if outcome not in ('unresolved','resolved','regressed'):raise ValueError('Invalid repair outcome')
    before,after=_bound(candidate_before),_bound(candidate_after)
    if before['sha256']==after['sha256']:raise ServiceError('unchanged_repair','A repair must change candidate bytes')
    gates=[_bound(p) for p in gate_evidence]
    if not gates:raise ServiceError('missing_repair_validation','Actual affected gate reports are required')
    if outcome=='resolved' and any(_read(g['path']).get('status')!='pass' for g in gates):
        raise ServiceError('unvalidated_resolution','A resolved repair requires passing affected gate reports')
    # Sibling candidate reviews share the same budget. Callers with another
    # directory layout must supply one persistent region-scoped ledger_directory.
    folder=Path(ledger_directory).resolve() if ledger_directory else directory.parent/'repair_ledger'
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'ledger.lock').open('a') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        previous=sorted(folder.glob('*.json'));chain=None;count=0
        for p in previous:
            event=_read(p)
            if event['previous_event_sha256']!=chain:raise ServiceError('repair_history_changed','Immutable repair ledger chain is inconsistent')
            count+=event['repair_class']==defect['proposed_repair_class'];chain=digest(p)
        if count>=3:raise ServiceError('repair_budget_exhausted','Three automatic attempts for this frozen defect class are exhausted')
        event={'schema_version':1,'created_at_utc':utcnow(),'defect_id':defect_id,
               'review_directory':str(directory),
               'repair_class':defect['proposed_repair_class'],'class_attempt':count+1,
               'previous_event_sha256':chain,'candidate_before':before,'candidate_after':after,
               'recipe':_bound(recipe_record),'affected_gate_evidence':gates,'outcome':outcome,
               'qualification':'A repair does not transfer qualification; affected exact candidates require revalidation'}
        path=folder/f'{len(previous)+1:04d}.json';_immutable(path,event)
    return _bound(path)
