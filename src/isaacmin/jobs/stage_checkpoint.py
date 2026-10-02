"""Byte-bound stage reuse; completed work survives ordinary CLI interruption."""
from pathlib import Path
import shutil

from isaacmin.io import atomic_json, hash_object, read_json, sha256_file, utc_now


def run_stage(directory, name, identity, owned_outputs, operation, *, accepted, partial_outputs=()):
    directory = Path(directory).resolve()
    if not name or any(c not in 'abcdefghijklmnopqrstuvwxyz_0123456789' for c in name):
        raise ValueError('Expected a local stage identifier')
    owned = [Path(p) for p in owned_outputs]
    partial = [Path(p) for p in partial_outputs]
    if not owned or any(p.is_absolute() or '..' in p.parts or p == Path('.') for p in owned+partial):
        raise ValueError('Stage outputs must be bounded paths within the build')
    checkpoint = directory / 'stage_checkpoints' / (name + '.json')
    key = hash_object(identity)
    previous = read_json(checkpoint) if checkpoint.exists() else None
    if previous and previous.get('status') == 'completed' and previous.get('identity_sha256') == key:
        files = previous.get('files', [])
        valid = bool(files)
        for item in files:
            path = directory / item['path']
            if (path.is_symlink() or not path.resolve().is_relative_to(directory)
                    or not path.is_file() or path.stat().st_size != item['bytes']
                    or sha256_file(path) != item['sha256']):
                valid = False
                break
        if valid and accepted(previous['result']):
            return previous['result']
    existing = [p for p in owned+partial if (directory / p).exists()]
    if existing or previous:
        archive = directory / 'attempts' / ('stage_' + name + '_' + utc_now().replace(':', '').replace('.', ''))
        archive.mkdir(parents=True)
        for relative in existing:
            source = directory / relative
            if source.is_symlink() or not source.resolve().is_relative_to(directory):
                raise ValueError('Refuse to archive an external stage output')
            target = archive / relative; target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(source, target)
        if previous:
            atomic_json(archive / 'checkpoint.json', previous)
    record = {'schema_version': 1, 'name': name, 'status': 'running',
              'identity_sha256': key, 'identity': identity, 'started_at_utc': utc_now(),
              'owned_outputs': [str(p) for p in owned], 'partial_outputs':[str(p) for p in partial],
              'qualification': 'not_inferred'}
    atomic_json(checkpoint, record)
    try:
        result = operation()
        record['result'] = result
        if not accepted(result):
            raise RuntimeError('Stage did not complete: ' + name)
        paths = []
        for relative in owned:
            path = directory / relative
            if not path.exists():
                raise RuntimeError('Stage omitted a required output: '+str(relative))
            if path.is_dir():
                paths.extend(p for p in path.rglob('*') if p.is_file())
            elif path.is_file():
                paths.append(path)
        if not paths:
            raise RuntimeError('Stage emitted no concrete output files: ' + name)
        files = []
        for p in sorted(set(paths)):
            if p.is_symlink() or not p.resolve().is_relative_to(directory):
                raise ValueError('Stage output contains an external link')
            files.append({'path': str(p.relative_to(directory)), 'bytes': p.stat().st_size,
                          'sha256': sha256_file(p)})
        record.update(status='completed', completed_at_utc=utc_now(), files=files)
        atomic_json(checkpoint, record)
        return result
    except BaseException as exc:
        record.update(status='interrupted_or_failed', completed_at_utc=utc_now(),
                      error={'type': type(exc).__name__, 'reason': str(exc)})
        atomic_json(checkpoint, record)
        raise
