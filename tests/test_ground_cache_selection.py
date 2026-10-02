"""Cross-build reuse guards. Fixture files never assert Blender execution."""
import json
from pathlib import Path
import pytest

from test_workers_native_cache import bare_contract_fixture


def setup_cache(tmp_path):
    from isaacmin.assembly.native_cache import record_bare
    root,request,bare,_=bare_contract_fixture(tmp_path)
    request['source_surface']=str(root/'terrain_surface.npz')
    Path(request['source_surface']).write_bytes(b'identity fixture, not source terrain')
    request['texture_repeat_m']=1.3
    request['geometry_detail']['subdivision_method']='bilinear_double_no_limit_v1'
    destination=root/'worlds/previous/data/bare_scene';destination.parent.mkdir(parents=True)
    bare.rename(destination);record_bare(request,root,destination)
    (destination/'assembly_request.json').write_text(json.dumps(request))
    return root,request,destination


def select(root,request):
    from isaacmin.assembly.reuse import find_bare_scene
    return find_bare_scene(root,root,root/'worlds/next/data/bare_scene',
        materials=request['materials'],origin=request['minecraft_origin'],
        support_mask_path=None,exterior_delta=None,geometry_detail={'subdivision_levels':3},
        terrain_material_recipe=None,evidence=root/'selection.json')


def test_unchanged_native_cache_survives_unrelated_validator_change(tmp_path):
    root,request,cache=setup_cache(tmp_path)
    assert select(root,request)==cache
    validator=root/'src/isaacmin/assets/usd_inspection.py'
    validator.parent.mkdir(parents=True,exist_ok=True);validator.write_text('changed validator fixture')
    assert select(root,request)==cache
    report=json.loads((root/'selection.json').read_text())
    assert report['qualification']=='not_inferred'
    assert report['selected']['status']=='verified_cache_candidate'


def test_changed_source_or_native_payload_rejects_reuse(tmp_path):
    root,request,cache=setup_cache(tmp_path)
    source=Path(request['source_surface']);original=source.read_bytes()
    source.write_bytes(original+b' changed')
    assert select(root,request) is None
    source.write_bytes(original)
    (cache/'scene.blend').write_bytes(b'changed serialized ground')
    assert select(root,request) is None
    assert json.loads((root/'selection.json').read_text())['candidates'][0]['status']=='rejected'


def test_changed_native_producer_or_material_scale_rejects_reuse(tmp_path):
    root,request,cache=setup_cache(tmp_path)
    request['materials'][0]['repeat_m']=2
    assert select(root,request) is None
    request['materials'][0]['repeat_m']=1.3
    producer=root/'blender_scripts/export_scene.py'
    producer.write_bytes(producer.read_bytes()+b'changed')
    assert select(root,request) is None


def test_missing_reexport_manifest_is_completed_before_population(tmp_path, monkeypatch):
    from isaacmin.assembly import reuse, pipeline
    from isaacmin.assembly.native_cache import verify_bare
    root, request, bare, _ = bare_contract_fixture(tmp_path)
    (bare / 'assembly_request.json').write_text(json.dumps(request))
    monkeypatch.setattr(reuse, 'find_bare_scene', lambda *a, **k: root / 'previous')
    monkeypatch.setattr(pipeline, 'assemble_region', lambda *a, **k: {'status': 'success'})
    result = reuse.assemble_bare(root, root, bare, evidence=root / 'selection.json')
    assert result['status'] == 'success'
    assert verify_bare(request, root, bare)['manifest']['appearance_qualification'] == 'not_inferred_from_cache'
    original = (bare / 'bare_scene_cache.json').read_bytes()
    (bare / 'scene.blend').write_bytes(b'changed fixture bytes')
    with pytest.raises(ValueError, match='bytes changed'):
        reuse.complete_bare_manifest(root, bare)
    assert (bare / 'bare_scene_cache.json').read_bytes() == original


@pytest.mark.parametrize('invalid', ['populated', 'failed_export', 'missing_closure'])
def test_missing_manifest_cannot_promote_incomplete_or_populated_export(tmp_path, invalid):
    from isaacmin.assembly.reuse import complete_bare_manifest
    root, request, bare, export = bare_contract_fixture(tmp_path)
    if invalid == 'populated':
        request['assets'] = [{'fixture': True}]
    elif invalid == 'failed_export':
        export['status'] = 'failed'
        (bare / 'export_result.json').write_text(json.dumps(export))
    else:
        (bare / 'native_dependency_closure.json').unlink()
    (bare / 'assembly_request.json').write_text(json.dumps(request))
    with pytest.raises((ValueError, FileNotFoundError)):
        complete_bare_manifest(root, bare)
    assert not (bare / 'bare_scene_cache.json').exists()
