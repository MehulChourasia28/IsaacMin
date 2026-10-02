"""Real temporary-filesystem checks for reversible archival, not native evidence."""
import importlib.util
import os
from pathlib import Path

import pytest

from isaacmin.io import atomic_json


@pytest.fixture
def archive(tmp_path, monkeypatch):
    script=Path(__file__).resolve().parents[1]/'scripts/archive_development.py'
    spec=importlib.util.spec_from_file_location('archive_housekeeping',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'ROOT',tmp_path)
    monkeypatch.setattr(module,'REPORT',tmp_path/'reports/cleanup_plan.json')
    # These tests exercise filesystem safety only, independently of live workers.
    monkeypatch.setattr(module,'assert_idle',lambda:None)
    original=tmp_path/'artifacts/development/obsolete/run';original.mkdir(parents=True)
    (original/'payload').write_bytes(b'preserve these exact bytes')
    os.link(original/'payload',original/'hardlink')
    (original/'symlink').symlink_to('payload')
    data=dict(archive='other/test_archive',protected=[],moves=[dict(
        source=str(original.relative_to(tmp_path)),destination='other/test_archive/'+str(original.relative_to(tmp_path)),
        receipt=module.tree_receipt(original))])
    atomic_json(module.REPORT,data)
    return module,original,tmp_path/data['moves'][0]['destination']


def test_archive_restore_preserves_bytes_hardlinks_and_is_resumable(archive):
    module,original,moved=archive;before=module.tree_receipt(original)
    module.relocate();assert not original.exists()
    assert module.tree_receipt(moved)==before
    module.relocate()  # Replaying an already completed move is safe.
    module.relocate(restore=True)
    assert not moved.exists() and module.tree_receipt(original)==before
    assert (original/'payload').read_bytes()==b'preserve these exact bytes'
    assert (original/'payload').stat().st_ino==(original/'hardlink').stat().st_ino
    assert (original/'symlink').readlink()==Path('payload')


def test_archive_rejects_changed_input_and_restore_collision(archive):
    module,original,moved=archive
    (original/'unexpected').write_text('new work')
    with pytest.raises(ValueError,match='changed since plan'):module.relocate()
    assert original.is_dir() and not moved.exists()
    (original/'unexpected').unlink()
    module.relocate()
    original.mkdir();(original/'new-work').write_text('must survive')
    with pytest.raises(FileExistsError,match='Refusing to overwrite'):module.relocate(restore=True)
    assert (original/'new-work').read_text()=='must survive' and moved.is_dir()


def test_archive_rejects_escape_outside_cleanup_scope(archive):
    module,original,moved=archive
    protected=module.ROOT/'.env';protected.write_text('fixture only')
    atomic_json(module.REPORT,dict(archive='other/test_archive',moves=[dict(
        source='artifacts/development/../../.env',destination='other/test_archive/env',
        receipt=module.tree_receipt(protected))]))
    with pytest.raises(ValueError,match='outside explicit cleanup scope'):module.relocate()
    assert protected.read_text()=='fixture only'
