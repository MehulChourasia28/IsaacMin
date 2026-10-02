from pathlib import Path

import pytest

from isaacmin.io import atomic_json, sha256_file
from isaacmin.jobs.repair_budget import exhausted_repairs


def test_exhausted_source_geometry_cannot_repeat_by_changing_unrelated_code(tmp_path):
    ledger = tmp_path / 'artifacts/repair.json'
    atomic_json(ledger, {'defect_class': 'native_surface', 'status': 'exhausted_failed',
                        'maximum_repair_attempts': 3, 'attempts': [{'attempt': i} for i in (1, 2, 3)]})
    atomic_json(tmp_path / 'state/known_failures.json', {'failures': [{
        'source_snapshot_sha256': 'save-a', 'source_ir_sha256': 'region-a', 'defect_class': 'native_surface',
        'ledger': 'artifacts/repair.json', 'ledger_sha256': sha256_file(ledger)}]})
    assert exhausted_repairs(tmp_path, 'save-a', 'region-a')[0]['attempts'] == 3
    (tmp_path / 'unrelated_material.py').write_text('# changed material transport\n')
    assert exhausted_repairs(tmp_path, 'save-a', 'region-a')[0]['status'] == 'blocked_repair_budget'
    assert not exhausted_repairs(tmp_path, 'save-b', 'region-a')
    assert not exhausted_repairs(tmp_path, 'save-a', 'region-b')
    ledger.write_text('{}')
    with pytest.raises(ValueError, match='ledger changed'):
        exhausted_repairs(tmp_path, 'save-a', 'region-a')
