"""Review-integrity fixtures; no provider or renderer is mocked as actual evidence."""
import json
from pathlib import Path
import pytest

from isaacmin.assets.network import ServiceError,digest
from isaacmin.adapters.visual_workflow import freeze_review,_load_frozen,construction_defects,record_repair,_calibration


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value));return path


def setup_review(tmp_path):
    scene=tmp_path/'scene.usda';scene.write_text('fixture scene identity')
    content=tmp_path/'content.usdc';content.write_bytes(b'fixture payload')
    closure=write(tmp_path/'native_dependency_closure.json',{'status':'pass','root_sha256':digest(scene),
                  'files':[{'path':p.name,'sha256':digest(p)} for p in (scene,content)]})
    profile=write(tmp_path/'quality_profile.json',{'fixture':True})
    write(tmp_path/'state/reference_catalogue.json',{'references':[]})
    captures=[]
    for condition in ('diffuse','directional'):
        folder=tmp_path/condition;folder.mkdir();frames=[]
        for i in range(24):
            rgb=folder/f'{i}.png';rgb.write_bytes(f'fixture RGB {condition} {i}'.encode())
            depth=folder/f'{i}.npy';depth.write_bytes(f'fixture depth {i}'.encode())
            frames.append({'frame':i,'rgb':rgb.name,'rgb_sha256':digest(rgb),'depth':depth.name,'depth_sha256':digest(depth),
                           'pose':{'kind':'static','position':[i,0,1],'look_at':[i+1,0,1],'scout_height_m':(.25,.6,1.5)[i%3],'held_out':i%4==0,
                                   'features':['meadow'],'material_families':['soil'],'asset_families':['grass']},
                           'camera_world_matrix_row_vectors':[[1,0,0,0],[0,1,0,0],[0,0,1,0],[i,0,1,1]],
                           'simulation_time_s':i,'depth_semantics':'axial_metres'})
        captures.append(write(folder/'capture_result.json',{'renderer':'RayTracedLighting','scope':'real_source_test_metadata_fixture',
                           'scene':str(scene),'scene_sha256':digest(scene),'frames':frames,'lighting':condition,
                           'physx_contact_rays':{'native_dependency_closure_sha256':digest(closure)}}))
    return captures,profile


def test_freeze_binds_bytes_balanced_unique_poses_and_explicit_reference_gap(tmp_path):
    captures,profile=setup_review(tmp_path)
    result=freeze_review(tmp_path,captures,profile,tmp_path/'review',reference_ids=(),required_features=['meadow'],required_material_families=['soil'],required_asset_families=['grass'])
    assert result['coverage']['distinct_static_poses']==24
    assert result['coverage']['held_out_fraction']==.25
    assert result['coverage']['observed']=={'features':[],'material_families':[],'asset_families':[]}
    assert result['coverage']['planned_intent_only']['features']==['meadow']
    assert set(result['coverage']['gaps'])=={'missing_actual_visibility_evidence','unrepresented_features:meadow',
           'unrepresented_material_families:soil','unrepresented_asset_families:grass','no_licensed_photographic_references'}
    manifest=Path(result['manifest']);_load_frozen(manifest)
    (tmp_path/'diffuse/0.npy').write_bytes(b'changed')
    with pytest.raises(ServiceError,match='changed'):_load_frozen(manifest)
    with pytest.raises(ServiceError):freeze_review(tmp_path,captures,profile,tmp_path/'review',reference_ids=())


def test_synthetic_fixture_and_cross_lighting_split_leakage_rejected(tmp_path):
    captures,profile=setup_review(tmp_path)
    original=json.loads(captures[0].read_text());synthetic={**original,'scope':'synthetic_native_cross_runtime_only'}
    write(captures[0],synthetic)
    with pytest.raises(ServiceError,match='synthetic'):freeze_review(tmp_path,captures,profile,tmp_path/'rejected',reference_ids=())
    original['frames'][0]['pose']['held_out']=False;write(captures[0],original)
    with pytest.raises(ServiceError,match='same pose'):freeze_review(tmp_path,captures,profile,tmp_path/'leaked',reference_ids=())


def test_no_critic_promotion_without_real_complete_calibration(tmp_path):
    report=write(tmp_path/'calibration.json',{'status':'incomplete','model':'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning','critic_usable_for_Q09':False})
    with pytest.raises(ServiceError,match='has not passed'):_calibration(report)


def test_unchanged_wrapper_cannot_hide_changed_usd_payload(tmp_path):
    captures,profile=setup_review(tmp_path)
    original_wrapper=digest(tmp_path/'scene.usda')
    (tmp_path/'content.usdc').write_bytes(b'changed payload but same wrapper')
    assert digest(tmp_path/'scene.usda')==original_wrapper
    with pytest.raises(ServiceError,match='content layer'):freeze_review(tmp_path,captures,profile,tmp_path/'review',reference_ids=())


def test_heldout_hidden_repairs_bounded_across_sibling_candidates_and_chain_verified(tmp_path):
    review=tmp_path/'reviews/one'
    development={'defect_id':'dev','proposed_repair_class':'leaf_material'}
    write(review/'defects/development/dev.json',development)
    write(review/'defects/held_out/secret.json',{'defect_id':'secret','pose':'held_out_location'})
    assert construction_defects(review)==[development]
    before=write(tmp_path/'before.json',{'candidate':1});after=write(tmp_path/'after.json',{'candidate':2})
    recipe=write(tmp_path/'recipe.json',{'opacityThreshold':.5});gate=write(tmp_path/'gate.json',{'status':'pass'})
    with pytest.raises(ServiceError,match='held-out'):record_repair(review,'secret',before,after,recipe,[gate],outcome='resolved')
    for i in range(3):record_repair(review,'dev',before,after,recipe,[gate],outcome='resolved')
    sibling=tmp_path/'reviews/two';write(sibling/'defects/development/dev.json',development)
    with pytest.raises(ServiceError,match='Three automatic'):record_repair(sibling,'dev',before,after,recipe,[gate],outcome='resolved')
    event=tmp_path/'reviews/repair_ledger/0001.json';value=json.loads(event.read_text());value['outcome']='regressed';write(event,value)
    with pytest.raises(ServiceError,match='chain'):record_repair(review,'dev',before,after,recipe,[gate],outcome='resolved')


def test_resolution_needs_changed_candidate_and_passing_gates(tmp_path):
    review=tmp_path/'review';write(review/'defects/development/d.json',{'defect_id':'d','proposed_repair_class':'terrain_join'})
    before=write(tmp_path/'before.json',{'candidate':1});after=write(tmp_path/'after.json',{'candidate':2})
    failed=write(tmp_path/'failed.json',{'status':'fail'});recipe=write(tmp_path/'recipe.json',{'version':1})
    with pytest.raises(ServiceError,match='change'):record_repair(review,'d',before,before,recipe,[failed],outcome='unresolved')
    with pytest.raises(ServiceError,match='passing'):record_repair(review,'d',before,after,recipe,[failed],outcome='resolved')
