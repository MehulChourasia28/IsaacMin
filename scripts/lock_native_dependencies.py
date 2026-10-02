#!/usr/bin/env python3
"""Snapshot static dependency bytes separately from changing qualification runs."""
from pathlib import Path
import hashlib,json,platform

ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):digest.update(block)
    return digest.hexdigest()
def descriptor(path):
    return {'path':str(path.relative_to(ROOT)),'sha256':sha(path),'bytes':path.stat().st_size}
lock_path=ROOT/'artifacts/bootstrap/native-build-lock.json';native=json.loads(lock_path.read_text())
binary_paths=[ROOT/p for p in ['.tools/build/native-gcc13/isaacmin_highmap','.tools/openvdb_worker',
    '.tools/blender/blender','.tools/isaacsim-6.1.0/python.sh','.tools/isaacsim-6.1.0/kit/kit',
    '.tools/isaacsim-6.1.0/VERSION']]
binary_paths+=list((ROOT/'.tools/manifold-py/lib/python3.12/site-packages').glob('manifold3d*.so'))
if len(binary_paths)!=7:raise RuntimeError('Exactly one pinned native Manifold extension is required')
runtime=json.loads((ROOT/'artifacts/bootstrap/blender/runtime_dependencies.json').read_text())
if runtime['status']!='pass' or runtime['runtime_version']!=native['isaac_runtime']['version']:
    raise RuntimeError('Pinned native runtime material dependency evidence is missing')
modules=[]
for entry in runtime['transitive_module_files']:
    path=Path(entry['path'])
    if sha(path)!=entry['sha256']:raise RuntimeError('Runtime material module changed')
    modules.append(descriptor(path))
record={'schema_version':1,'architecture':platform.machine(),'system':platform.system(),
        'orchestrator_requirements':descriptor(ROOT/'requirements.lock'),
        'native_build_lock':descriptor(lock_path),'native_binaries':[descriptor(p) for p in binary_paths],
        'runtime_material_modules':modules,'isaac_runtime_version':native['isaac_runtime']['version'],
        'blender_commit':native['sources']['blender']['commit'],
        'highmap_commit':native['sources']['HighMap']['commit'],
        'geometry_python_packages':native['geometry_python_packages'],
        'qualification_policy':'Qualification runs and artifact hashes are separate; static bytes do not imply a current scene or quality pass'}
path=ROOT/'artifacts/bootstrap/dependency-lock.json';path.write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
print(str(path));print(sha(path))
