#!/usr/bin/python3
"""Extract distribution-signed ARM64 packages privately; never run maintainer scripts."""
import apt
import hashlib
import json
import pathlib
import subprocess
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEST = ROOT / '.tools/native'
CACHE = ROOT / '.tools/debs'
DEST.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
cache = apt.Cache()
for package in cache:
    if package.candidate and 'esm.ubuntu.com' in (package.candidate.uri or ''):
        public = [v for v in package.versions if v.uri and 'ports.ubuntu.com' in v.uri]
        if public:
            package.candidate = public[0]
for name in ['blender', 'libopenvdb-dev', 'libopencv-dev', 'libgsl-dev',
             'libassimp-dev', 'libglm-dev', 'ocl-icd-opencl-dev',
             'opencl-clhpp-headers', 'libtbb-dev', 'libblosc-dev']:
    cache[name].mark_install(auto_fix=True, auto_inst=True, from_user=True)
records = []
for package in cache.get_changes():
    if not (package.marked_install or package.marked_upgrade):
        continue
    version = package.candidate
    if version.architecture not in ('arm64', 'all'):
        raise RuntimeError(f'Unexpected package architecture: {package.name}')
    url = version.uri.replace('http://', 'https://')
    target = CACHE / pathlib.Path(url).name
    expected = version.sha256
    if not target.exists() or hashlib.sha256(target.read_bytes()).hexdigest() != expected:
        part = target.with_suffix('.part')
        urllib.request.urlretrieve(url, part)
        if hashlib.sha256(part.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Package checksum mismatch: {package.name}')
        part.replace(target)
    subprocess.run(['dpkg-deb', '-x', str(target), str(DEST)], check=True)
    records.append(dict(name=package.name, version=version.version,
                        architecture=version.architecture, url=url,
                        sha256=expected, method='dpkg-deb-extract-no-maintainer-scripts'))
    print(package.name, version.version, flush=True)
(ROOT / 'artifacts/bootstrap/native-packages.json').write_text(json.dumps(records, indent=2)+'\n')
