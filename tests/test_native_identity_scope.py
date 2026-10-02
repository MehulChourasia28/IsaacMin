"""Actual pinned OBJ-comparator fixtures for separate export proof namespaces."""
from pathlib import Path

import numpy as np
import pytest
import trimesh
from isaacmin.io import read_json, sha256_file
from test_source_native_integration import write_native_fixture

ROOT = Path(__file__).resolve().parents[1]
from isaacmin.validation import native_ground as candidate

def test_identical_bare_and_populated_meshes_keep_separate_exact_proofs(tmp_path, monkeypatch):
    monkeypatch.setattr(candidate, 'workspace_root', lambda: ROOT)
    mesh = trimesh.creation.box()
    bare = write_native_fixture(tmp_path / 'bare', mesh.vertices, mesh.faces)
    populated = write_native_fixture(tmp_path / 'populated', mesh.vertices, mesh.faces)
    assert sha256_file(bare) == sha256_file(populated)
    first = candidate.bind_native_ground(bare, tmp_path / 'evidence')
    original = first['identity_report'].read_bytes()
    second = candidate.bind_native_ground(populated, tmp_path / 'evidence')
    assert first['identity_report'] != second['identity_report']
    assert first['report']['status'] == second['report']['status'] == 'pass'
    assert first['identity_report'].read_bytes() == original
    assert first['report']['input_files'][0]['path'] == str(bare)
    assert second['report']['input_files'][0]['path'] == str(populated)
    before = second['identity_report'].stat().st_mtime_ns
    repeated = candidate.bind_native_ground(populated, tmp_path / 'evidence')
    assert repeated['identity_report'].stat().st_mtime_ns == before


def test_changed_export_still_requires_exact_native_correspondence(tmp_path, monkeypatch):
    monkeypatch.setattr(candidate, 'workspace_root', lambda: ROOT)
    mesh = trimesh.creation.box()
    obj = write_native_fixture(tmp_path / 'scene', mesh.vertices, mesh.faces)
    before = candidate.bind_native_ground(obj, tmp_path / 'evidence')
    original = before['identity_report'].read_bytes()
    lines = obj.read_text().splitlines()
    index = next(i for i, line in enumerate(lines) if line.startswith('v '))
    words = lines[index].split()
    words[1] = repr(float(words[1]) + .001)
    lines[index] = ' '.join(words)
    obj.write_text('\n'.join(lines) + '\n')
    with pytest.raises(ValueError, match='does not exactly preserve'):
        candidate.bind_native_ground(obj, tmp_path / 'evidence')
    assert before['identity_report'].read_bytes() == original
