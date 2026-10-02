from pathlib import Path
import json

import pytest

from isaacmin.jobs.stage_checkpoint import run_stage


def test_completed_stage_reuses_only_identical_bytes_and_inputs(tmp_path):
    calls = []
    def operation():
        calls.append(True); (tmp_path / 'result').mkdir()
        (tmp_path / 'result/mesh.bin').write_bytes(b'actual stage output')
        return {'status': 'success', 'sample': len(calls)}
    def run(identity):
        return run_stage(tmp_path, 'terrain', identity, ['result'], operation,
                         accepted=lambda r: r['status'] == 'success')
    assert run({'source': 'one'})['sample'] == 1
    assert run({'source': 'one'})['sample'] == 1
    assert len(calls) == 1
    (tmp_path / 'result/mesh.bin').write_bytes(b'changed')
    assert run({'source': 'one'})['sample'] == 2
    assert run({'source': 'two'})['sample'] == 3
    assert any(p.read_bytes() == b'changed' for p in (tmp_path / 'attempts').rglob('mesh.bin'))


def test_interrupted_partial_output_never_reused_or_erased(tmp_path):
    def interrupted():
        (tmp_path / 'scene').mkdir(); (tmp_path / 'scene/partial.usda').write_text('retained partial')
        raise InterruptedError('test interruption')
    with pytest.raises(InterruptedError):
        run_stage(tmp_path, 'export', {}, ['scene'], interrupted, accepted=lambda r: True)
    record = json.loads((tmp_path / 'stage_checkpoints/export.json').read_text())
    assert record['status'] == 'interrupted_or_failed'
    def complete():
        assert not (tmp_path / 'scene').exists()
        (tmp_path / 'scene').mkdir(); (tmp_path / 'scene/complete.usda').write_text('complete')
        return {'status': 'success'}
    run_stage(tmp_path, 'export', {}, ['scene'], complete, accepted=lambda r: True)
    assert any(p.read_text() == 'retained partial' for p in (tmp_path / 'attempts').rglob('partial.usda'))


def test_failed_atomic_staging_is_preserved_before_retry(tmp_path):
    staging=tmp_path/'scene.staging';staging.mkdir();(staging/'content').write_text('incomplete')
    def complete():
        assert not staging.exists()
        (tmp_path/'scene').mkdir();(tmp_path/'scene/world.usda').write_text('complete')
        return {'status':'success'}
    run_stage(tmp_path,'collision',{},['scene'],complete,accepted=lambda r:True,
              partial_outputs=['scene.staging'])
    assert any(p.read_text()=='incomplete' for p in (tmp_path/'attempts').rglob('content'))


def test_missing_required_output_cannot_be_cached(tmp_path):
    def incomplete():
        (tmp_path/'image.png').write_bytes(b'one output')
        return {'status':'success'}
    with pytest.raises(RuntimeError,match='required output'):
        run_stage(tmp_path,'capture',{},['image.png','depth.npy'],incomplete,accepted=lambda r:True)
    assert json.loads((tmp_path/'stage_checkpoints/capture.json').read_text())['status']!='completed'
