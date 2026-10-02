"""Install only the locked, project-private Linux ARM64 OpenClaw runtime."""
import hashlib
import os
import platform
import shutil
import subprocess
import tarfile
import urllib.request
from pathlib import Path

from isaacmin.io import read_json
from isaacmin.security import worker_environment

root = Path(__file__).resolve().parents[1]
lock = read_json(root / 'configs/openclaw/runtime.lock.json')
if platform.system() != 'Linux' or platform.machine() not in ('aarch64', 'arm64'):
    raise SystemExit('This lock is for Linux ARM64; no system installation will be changed.')
node = lock['node']
tools = root / '.tools'
archive = tools / 'downloads' / node['url'].rsplit('/', 1)[-1]
archive.parent.mkdir(parents=True, exist_ok=True)
if not archive.is_file():
    partial = archive.with_suffix('.partial')
    with urllib.request.urlopen(node['url'], timeout=60) as response, partial.open('wb') as output:
        shutil.copyfileobj(response, output)
    if hashlib.sha256(partial.read_bytes()).hexdigest() != node['sha256']:
        raise SystemExit('Node archive integrity mismatch; retained .partial, installed nothing.')
    partial.replace(archive)
if hashlib.sha256(archive.read_bytes()).hexdigest() != node['sha256']:
    raise SystemExit('Cached Node archive integrity mismatch.')
directory = tools / archive.name.removesuffix('.tar.xz')
if not directory.is_dir():
    with tarfile.open(archive) as package:
        package.extractall(tools, filter='data')
destination = tools / 'openclaw'
destination.mkdir(exist_ok=True)
for name in ('package.json', 'package-lock.json'):
    source = root / 'configs/openclaw' / name
    if name == 'package-lock.json' and hashlib.sha256(source.read_bytes()).hexdigest() != lock['openclaw']['npm_lock_sha256']:
        raise SystemExit('OpenClaw dependency lock mismatch.')
    shutil.copyfile(source, destination / name)
env = worker_environment()
env['PATH'] = str(directory / 'bin') + os.pathsep + env.get('PATH', '')
env['npm_config_cache'] = str(tools / 'npm-cache')
subprocess.run([str(directory / 'bin/node'),
    str(directory / 'lib/node_modules/npm/bin/npm-cli.js'), 'ci',
    '--prefix', str(destination), '--ignore-scripts', '--no-audit', '--no-fund'],
    env=env, check=True)
subprocess.run([str(directory / 'bin/node'),
    str(destination / 'node_modules/openclaw/openclaw.mjs'), '--version'], env=env, check=True)
print('Private OpenClaw runtime installed; NVIDIA key and system installations unchanged.')
