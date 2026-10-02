import io
import json
import tarfile
import zipfile

import pytest

from isaacmin.source.archive import unpack_archive
from isaacmin.source.nbt import SourceError
from isaacmin.source.snapshot import snapshot_world,verify_source_unchanged


def test_actual_archive_snapshot_and_source_identity(tmp_path):
    archive=tmp_path/"world.zip"
    with zipfile.ZipFile(archive,"w") as writer:
        writer.writestr("OneWorld/level.dat",b"level-data-bytes")
        writer.writestr("OneWorld/region/r.0.0.mca",b"region-data-bytes")
    before=archive.read_bytes()
    snapshot=snapshot_world(archive,tmp_path/"snapshots")
    assert snapshot["original_kind"]=="archive"
    assert archive.read_bytes()==before
    assert verify_source_unchanged(snapshot)["status"]=="pass"
    assert snapshot_world(archive,tmp_path/"snapshots")["save_sha256"]==snapshot["save_sha256"]
    archive.write_bytes(before+b"changed")
    assert verify_source_unchanged(snapshot)["status"]=="fail"


@pytest.mark.parametrize("name",["../escape","/absolute","C:/windows","world/../../escape","world\\escape"])
def test_zip_path_traversal_is_rejected(tmp_path,name):
    archive=tmp_path/"bad.zip"
    with zipfile.ZipFile(archive,"w") as writer:
        writer.writestr(name,b"bad")
    with pytest.raises(SourceError):unpack_archive(archive,tmp_path/"out")
    assert not (tmp_path/"escape").exists()


def test_archive_links_and_resource_budgets(tmp_path):
    archive=tmp_path/"link.tar"
    with tarfile.open(archive,"w") as writer:
        member=tarfile.TarInfo("world/link")
        member.type=tarfile.SYMTYPE
        member.linkname="/tmp/target"
        writer.addfile(member)
    with pytest.raises(SourceError,match="Links"):unpack_archive(archive,tmp_path/"links")
    archive=tmp_path/"large.zip"
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED) as writer:
        writer.writestr("world/level.dat",b"0"*10000)
    with pytest.raises(SourceError,match="budget"):unpack_archive(archive,tmp_path/"budget",max_uncompressed_bytes=32)


def test_multiple_world_archives_are_ambiguous(tmp_path):
    archive=tmp_path/"worlds.zip"
    with zipfile.ZipFile(archive,"w") as writer:
        writer.writestr("world_one/level.dat",b"one")
        writer.writestr("world_two/level.dat",b"two")
    with pytest.raises(SourceError,match="exactly one"):unpack_archive(archive,tmp_path/"out")
