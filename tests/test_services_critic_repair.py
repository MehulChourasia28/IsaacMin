"""Ledger/isolation logic only; these tests never establish critic quality."""
import json
import pytest
from isaacmin.adapters import critic_repair as repair
from isaacmin.assets.network import ServiceError


def test_detector_budget_is_persistent_and_same_recipe_resumes_without_call(tmp_path,monkeypatch):
    dataset=tmp_path/'development.json';dataset.write_text('{}')
    parent=tmp_path/'prior_failure.json';parent.write_text('{"status":"fail"}')
    cases=[{'id':c,'split':'development','expected_severe':[{'category':c}]} for c in repair.CATEGORIES]
    cases.append({'id':'clean','split':'development','expected_severe':[],'clean_control_independently_verified':True})
    monkeypatch.setattr(repair,'load_dataset',lambda _: {'cases':cases})
    calls=[]
    def evaluate(*args,**kwargs):
        calls.append(args[2]['prompt']);return {'status':'fail','scope':'development_detector_repair_only'}
    monkeypatch.setattr(repair,'_evaluate',evaluate)
    recipes=[{'prompt':f'Explicit repair {i}','reasoning_budget':128,'repair_rationale':'Test ledger only','parent_failure_report':repair._bound(parent)} for i in range(4)]
    ledger=tmp_path/'ledger'
    for recipe in recipes[:3]:
        assert repair.develop_detector_recipe(tmp_path,dataset,recipe,ledger)['status']=='fail'
    assert len(calls)==3
    assert repair.develop_detector_recipe(tmp_path,dataset,recipes[0],ledger)['status']=='fail'
    assert len(calls)==3
    with pytest.raises(ServiceError) as error:
        repair.develop_detector_recipe(tmp_path,dataset,recipes[3],ledger)
    assert error.value.code=='detector_repair_budget_exhausted' and len(calls)==3
    assert not (tmp_path/'state/qualified_visual_critic.json').exists()


def test_held_out_case_never_enters_development_and_failed_dev_cannot_qualify(tmp_path,monkeypatch):
    dataset=tmp_path/'dataset.json';dataset.write_text('{}')
    parent=tmp_path/'prior_failure.json';parent.write_text('{"status":"fail"}')
    recipe={'prompt':'Only development','reasoning_budget':128,'repair_rationale':'Test fold isolation','parent_failure_report':repair._bound(parent)}
    monkeypatch.setattr(repair,'load_dataset',lambda _: {'cases':[{'split':'held_out'}]})
    monkeypatch.setattr(repair,'_evaluate',lambda *a,**k:pytest.fail('Hosted evaluation must not run'))
    with pytest.raises(ServiceError) as error:repair.develop_detector_recipe(tmp_path,dataset,recipe,tmp_path/'ledger')
    assert error.value.code=='heldout_development_leakage'
    development=tmp_path/'failed_development.json';development.write_text(json.dumps({'status':'fail','scope':'development_detector_repair_only'}))
    with pytest.raises(ServiceError) as error:repair.calibrate_developed_recipe(tmp_path,dataset,development,tmp_path/'final')
    assert error.value.code=='development_not_passing'
    assert not (tmp_path/'final').exists() and not (tmp_path/'state/qualified_visual_critic.json').exists()
