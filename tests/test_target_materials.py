"""Source containment and preparation-proof guards, not rendered appearance tests."""
import pytest

from isaacmin.assets.target_materials import _source,verify_prepared_asset,prepare_target_assets,PRODUCERS,RECIPE
from isaacmin.io import sha256_file


def test_absent_material_inputs_do_not_claim_preparation(tmp_path):
    result=prepare_target_assets(tmp_path)
    assert result['status']=='not_available' and result['qualification']=='not_run'


def test_original_asset_requires_exact_owned_blend_bytes(tmp_path):
    path=tmp_path/'assets/normalized/grass/normalized.blend';path.parent.mkdir(parents=True)
    path.write_bytes(b'guard-only bytes; never opened or reported as a native blend')
    asset={'asset_id':'grass','output_blend':str(path),'output_sha256':sha256_file(path)}
    assert _source(tmp_path,asset)==path
    path.write_bytes(b'changed')
    with pytest.raises(ValueError,match='changed'):_source(tmp_path,asset)
    asset['asset_id']='../escape'
    with pytest.raises(ValueError,match='identifier'):_source(tmp_path,asset)
    path=tmp_path/'outside.blend';path.write_bytes(b'outside the asset store')
    asset.update(asset_id='grass',output_blend=str(path),output_sha256=sha256_file(path))
    with pytest.raises(ValueError,match='outside'):_source(tmp_path,asset)


def test_preparation_status_and_changed_producers_cannot_establish_verification(tmp_path):
    for name in PRODUCERS:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('identity fixture')
    asset={'status':'pass','isaac_qualification':'pass'}
    assert not verify_prepared_asset(tmp_path,asset)
    asset['target_material_preparation']={'recipe':RECIPE,'producers':{name:'stale' for name in PRODUCERS}}
    assert not verify_prepared_asset(tmp_path,asset)
