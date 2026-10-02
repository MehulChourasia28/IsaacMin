"""Bounded, non-executing save-archive extraction into snapshot-owned staging."""
from __future__ import annotations

import shutil
import stat
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

from .nbt import SourceError
from .snapshot import _snapshot_directory, sha256


def _relative(name: str) -> Path:
    part=PurePosixPath(name)
    if not name or "\\" in name or part.is_absolute() or ".." in part.parts or not part.parts or ":" in part.parts[0]:
        raise SourceError("Unsafe path in save archive")
    return Path(*part.parts)


def unpack_archive(archive: Path, destination: Path, *, max_uncompressed_bytes: int=64*1024**3, max_files: int=500000) -> Path:
    archive,destination=Path(archive),Path(destination)
    if destination.exists() and any(destination.iterdir()):
        raise SourceError("Archive staging destination must be empty")
    destination.mkdir(parents=True,exist_ok=True)
    budget=min(max_uncompressed_bytes,int(shutil.disk_usage(destination).free*.75))
    total,count,seen=0,0,set()
    def entry(name,size,directory):
        nonlocal total,count
        relative=_relative(name)
        if relative in seen:
            raise SourceError("Duplicate path in save archive")
        seen.add(relative)
        count+=1
        total+=size
        if size<0 or total>budget or count>max_files:
            raise SourceError("Save archive exceeds file-count, size, or disk-reserve budget")
        target=destination/relative
        if directory:
            target.mkdir(parents=True,exist_ok=True)
        else:
            target.parent.mkdir(parents=True,exist_ok=True)
        return target
    def copy(stream,target,expected):
        actual=0
        with target.open("xb") as handle:
            while True:
                block=stream.read(min(1024*1024,expected-actual+1))
                if not block:break
                actual+=len(block)
                if actual>expected:
                    raise SourceError("Save archive entry expands beyond declared size")
                handle.write(block)
        if actual!=expected:
            raise SourceError("Truncated save archive entry")
    try:
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as reader:
                for member in reader.infolist():
                    mode=(member.external_attr>>16)&0xffff
                    if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0,stat.S_IFREG,stat.S_IFDIR)):
                        raise SourceError("Links/special files are forbidden in save archives")
                    if member.flag_bits&1:
                        raise SourceError("Encrypted save archives are unsupported")
                    target=entry(member.filename,member.file_size,member.is_dir())
                    if not member.is_dir():
                        with reader.open(member) as stream:copy(stream,target,member.file_size)
        elif tarfile.is_tarfile(archive):
            with tarfile.open(archive,"r:*") as reader:
                for member in reader:
                    if member.isdir() and member.name in (".","./"):
                        continue
                    if not member.isfile() and not member.isdir():
                        raise SourceError("Links/special files are forbidden in save archives")
                    target=entry(member.name,member.size,member.isdir())
                    if member.isfile():
                        stream=reader.extractfile(member)
                        if stream is None:raise SourceError("Unreadable save archive member")
                        with stream:copy(stream,target,member.size)
        else:
            raise SourceError("Expected an unencrypted ZIP or TAR save archive")
    except (zipfile.BadZipFile,tarfile.TarError,OSError) as exc:
        raise SourceError(f"Save archive extraction failed: {type(exc).__name__}") from exc
    roots=[p.parent for p in destination.rglob("level.dat") if p.is_file()]
    if len(roots)!=1:
        raise SourceError("Save archive must contain exactly one Java world level.dat")
    return roots[0]


def snapshot_archive(archive: Path, output: Path) -> dict:
    archive,output=Path(archive).resolve(),Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    initial_hash=sha256(archive)
    staging=Path(tempfile.mkdtemp(prefix=".archive-",dir=output))
    try:
        world=unpack_archive(archive,staging)
        if sha256(archive)!=initial_hash:
            raise SourceError("Archive changed during extraction; source snapshot rejected")
        return _snapshot_directory(world,output,source_reference={"original_path":str(archive),"original_kind":"archive",
                                   "archive_sha256":initial_hash,"archive_world_prefix":world.relative_to(staging).as_posix(),
                                   "archive_extraction":"bounded read-only ZIP/TAR; traversal, links and special entries rejected"})
    finally:
        shutil.rmtree(staging,ignore_errors=True)
