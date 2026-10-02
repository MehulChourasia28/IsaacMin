"""Standalone package controller: standard-library Python, pinned Isaac subprocesses.

python3 reproduction/reproduce_package.py --package PACKAGE --isaac-python /path/to/python.sh --output NEW_DIR
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(4*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def local(root, name):
    path = (root / name).resolve()
    if Path(name).is_absolute() or not path.is_relative_to(root) or '..' in Path(name).parts:
        raise ValueError('Package contains an escaping path')
    return path


def verify(root, records):
    if not records or len({r['path'] for r in records}) != len(records):
        raise ValueError('Empty or duplicated package inventory')
    for record in records:
        path = local(root, record['path'])
        if not path.is_file() or path.stat().st_size != record['bytes'] or digest(path) != record['sha256']:
            raise ValueError('Package payload changed: ' + record['path'])


def run(argv, output, name):
    allowed = ('PATH', 'HOME', 'USER', 'LANG', 'LC_ALL', 'DISPLAY', 'XDG_RUNTIME_DIR',
               'CUDA_VISIBLE_DEVICES', 'VK_ICD_FILENAMES')
    env = {key: os.environ[key] for key in allowed if key in os.environ}
    env.update(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', LD_LIBRARY_PATH='')
    started = time.monotonic()
    with (output / (name+'.log')).open('w') as stream:
        process = subprocess.Popen(argv, cwd=output, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            while process.poll() is None:
                available = next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines()
                                 if line.startswith('MemAvailable:'))
                if available < 24*2**30 or time.monotonic()-started > 3600:
                    raise RuntimeError('Portable probe stopped at memory reserve or time budget')
                time.sleep(.5)
            if process.returncode:
                raise RuntimeError(name+' returned a nonzero exit code; inspect retained log')
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--isaac-python', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.package.resolve(), args.output.resolve()
    if output.is_relative_to(root):
        raise ValueError('Probe output must be separate from the immutable package')
    output.mkdir(parents=True, exist_ok=True)
    scene_dir = output / 'scene'
    scene_dir.mkdir()  # Existing attempts cannot be overwritten.
    manifest = json.loads((root / 'package.json').read_text())
    package_hash = digest(root / 'package.json')
    verify(root, manifest['files'])
    closure = json.loads((root / 'dependency_closure.json').read_text())
    for record in closure['files']:
        source = local(root, record['path'])
        target = local(scene_dir, record['path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    # OBJ is diagnostic interchange data used only to bind contact-ray provenance.
    if (root / 'final_ground.obj').is_file():
        shutil.copyfile(root / 'final_ground.obj', scene_dir / 'final_ground.obj')
    probe = root / 'reproduction'
    request = json.loads((probe / 'request.json').read_text())
    request.update(scene=str(local(scene_dir, request['scene'])), output=str(output / 'capture'))
    request_path = output / 'capture_request.json'
    request_path.write_text(json.dumps(request, indent=2)+'\n')
    runtime = str(args.isaac_python.resolve())
    run([runtime, str(probe / 'capture_scene.py'), '--request', str(request_path)], output, 'capture')
    if not (output / 'capture/capture_result.json').is_file():
        raise RuntimeError('Native process did not produce current captures')
    run([runtime, str(probe / 'reopen_scene.py'), '--scene', request['scene'],
         '--output', str(output / 'reopen')], output, 'reopen')
    verify(root, manifest['files'])
    if digest(root / 'package.json') != package_hash:
        raise ValueError('Package manifest changed during reproduction')
    run([runtime, str(probe / 'compare_reproduction.py'), '--package', str(root), '--output', str(output)],
        output, 'comparison')


if __name__ == '__main__':
    main()
