"""Content-bound numerical workers for native precision construction."""
from pathlib import Path
import json
from isaacmin.io import atomic_json,hash_object,sha256_file
from isaacmin.process import run_worker

PRODUCERS=('scripts/native_precision_worker.py','src/isaacmin/volumes/precision_algorithms.py',
           'src/isaacmin/volumes/planar_patch.py','src/isaacmin/volumes/native_precision.py',
           'src/isaacmin/volumes/projected_patch.py','src/isaacmin/volumes/intersection_repair.py',
           'src/isaacmin/validation/global_intersections.py','scripts/native/triangle_intersection_validator.cpp',
           'src/isaacmin/volumes/plane_conditioning.py','scripts/bootstrap_precision.py',
           'scripts/native/subdivide_double.cpp','scripts/native/write_native_obj.cpp')


def run_precision(root,request,output,*,estimated_memory_bytes):
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    executable=root/'.tools/manifold-py/bin/python'
    site=root/'.tools/manifold-py/lib/python3.12/site-packages'
    runtime_files=sorted({p for pattern in ('manifold3d*.so','numpy/**/*.so','numpy/**/*.py',
        'numpy.libs/*','trimesh/**/*.py','scipy/**/*.so','scipy/**/*.py','scipy.libs/*')
        for p in site.glob(pattern) if p.is_file()})
    from isaacmin.validation.global_intersections import native_validator_files
    _,native_files=native_validator_files(root)
    build_path=root/'.tools/native_precision/build_manifest.json'
    build=json.loads(build_path.read_text())
    for worker in build['workers'].values():
        for entry in worker['files']:
            if sha256_file(Path(entry['path']))!=entry['sha256']:raise ValueError('Native precision worker dependency changed')
    identity={'native_geometry_discovery_files':native_files,'native_precision_build_sha256':sha256_file(build_path),'request':{k:v for k,v in request.items() if k!='output'},
        'producers':{name:sha256_file(root/name) for name in PRODUCERS},
        'toolchain':sha256_file(root/'artifacts/bootstrap/dependency-lock.json'),
        'python':sha256_file(executable),'runtime_files':[
            {'path':str(p.relative_to(root)),'sha256':sha256_file(p)} for p in runtime_files]}
    input_keys=('terrain_mesh',) if request['mode'] in ('coarse','subdivision') else ('vertices','triangles')
    for key in input_keys:
        if sha256_file(Path(request[key]))!=request[key+'_sha256']:raise ValueError('Precision source input changed: '+key)
    record=output/'precision_manifest.json'
    if record.exists():
        result=json.loads(record.read_text())
        if result['identity']!=identity:raise ValueError('Precision candidate inputs changed; use a new immutable directory')
        for entry in result['files']:
            if sha256_file(output/entry['path'])!=entry['sha256']:raise ValueError('Precision candidate bytes changed')
        return result
    request=dict(request,output=str(output.resolve()));path=output/'request.json';atomic_json(path,request)
    process=run_worker([str(executable),str(root/'scripts/native_precision_worker.py'),'--request',str(path)],
        cwd=root,log_path=output/'worker.log',timeout=86400,
        environment={'PYTHONPATH':str(root/'src'),'LD_LIBRARY_PATH':'','OMP_NUM_THREADS':'8'},
        estimated_memory_bytes=estimated_memory_bytes,estimated_disk_bytes=estimated_memory_bytes//4)
    if process['exit_code']!=0:raise RuntimeError('Native precision construction failed; inspect '+str(output/'worker.log'))
    for key in input_keys:
        if sha256_file(Path(request[key]))!=request[key+'_sha256']:raise ValueError('Precision source changed during execution: '+key)
    names=tuple(str(p.relative_to(output)) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='precision_manifest.json')
    result={'status':'native_precision_candidate_not_qualified','identity':identity,
        'files':[{'path':name,'sha256':sha256_file(output/name),'bytes':(output/name).stat().st_size} for name in names],
        'process':process,'request_sha256':sha256_file(path)}
    atomic_json(record,result);return result


def condition_input(root,mesh_path,translation,origin,detail,output_parent):
    request={'mode':'coarse','terrain_mesh':str(Path(mesh_path).resolve()),'terrain_mesh_sha256':sha256_file(Path(mesh_path)),
        'terrain_translation':list(translation),'minecraft_origin':list(origin),'geometry_detail':detail}
    key=hash_object({'request':request,'producers':{name:sha256_file(Path(root)/name) for name in PRODUCERS}})
    output=Path(output_parent)/key[:20]
    result=run_precision(root,request,output,estimated_memory_bytes=4*2**30)
    return output/'terrain.obj',result


def subdivide_precise(root,mesh_path,levels,output):
    request={'mode':'subdivision','terrain_mesh':str(Path(mesh_path).resolve()),
        'terrain_mesh_sha256':sha256_file(mesh_path),'subdivision_levels':int(levels)}
    return run_precision(root,request,output,estimated_memory_bytes=16*2**30)
