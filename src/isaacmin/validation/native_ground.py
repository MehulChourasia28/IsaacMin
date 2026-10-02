"""Independent exact correspondence between exported OBJ and native arrays."""
from pathlib import Path
import json
import numpy as np

from ..io import atomic_json,sha256_file,hash_object
from ..process import run_worker
from .global_intersections import file_record,_verify_record


def workspace_root():return Path(__file__).resolve().parents[3]


def evidence_root(output):
    output=Path(output).resolve()
    return next((parent for parent in (output,*output.parents) if parent.name=='evidence'),output.parent)


def has_native_ground(mesh_path):
    mesh_path=Path(mesh_path)
    return mesh_path.suffix=='.obj' and (mesh_path.parent/'native_precision/finalization.json').is_file()


def verify_records(records):
    for entry in records:_verify_record(entry)


def bind_native_ground(mesh_path,evidence_root):
    """Require every OBJ coordinate/index to equal the actual f32/i32 arrays.

    Six-decimal approximations are explicitly rejected. Native arrays alone do
    not establish export/renderer fidelity. This proof compares the entire OBJ.
    """
    root=workspace_root();mesh_path=Path(mesh_path).resolve();native=mesh_path.parent/'native_precision';meta_path=native/'finalization.json'
    metadata=json.loads(meta_path.read_text());vertices_path=native/'final_vertices.npy';triangles_path=native/'final_triangles.npy'
    for path,key in [(vertices_path,'final_vertices_sha256'),(triangles_path,'final_triangles_sha256')]:
        if sha256_file(path)!=metadata[key]:raise ValueError('Native final geometry differs from recorded finalization')
    v=np.load(vertices_path,mmap_mode='r',allow_pickle=False);f=np.load(triangles_path,mmap_mode='r',allow_pickle=False)
    if v.dtype!=np.dtype('<f4') or f.dtype!=np.dtype('<i4') or v.ndim!=2 or f.ndim!=2 or v.shape[1:]!=(3,) or f.shape[1:]!=(3,) or not v.flags.c_contiguous or not f.flags.c_contiguous:raise ValueError('Actual final native arrays require C-order f32/i32 triples')
    build_path=root/'.tools/geometry_validation/obj_compare_build_manifest.json';build=json.loads(build_path.read_text())
    dependencies=[build['source'],build['binary'],build['compiler'],*build['compile_dependencies'],*build['runtime_libraries']];verify_records(dependencies)
    mesh_hash=sha256_file(mesh_path);inputs=[file_record(path,role) for path,role in [(mesh_path,'authoritative_obj'),(vertices_path,'actual_native_vertices'),(triangles_path,'actual_native_triangles'),(meta_path,'native_finalization')]]
    producers=[file_record(__file__,'native_ground_identity_validator'),file_record(build_path,'pinned_obj_comparator')]
    identity=sha256_file(build_path)[:16]+'_'+sha256_file(Path(__file__))[:16]
    # The bare and populated exports can have identical mesh bytes at distinct
    # paths. Each proof must retain its exact input paths without colliding with
    # another copy's cache entry. Changed arrays/finalization get a new key too.
    input_identity=hash_object(inputs)
    output=Path(evidence_root)/'native_ground_identity'/mesh_hash/identity/input_identity;report_path=output/'ground_identity.json'
    if report_path.exists():
        report=json.loads(report_path.read_text());verify_records([*report['input_files'],*report['producer_files'],*report['files']])
        if report['input_files']!=inputs or report['producer_files']!=producers or report['status']!='pass':raise ValueError('Native ground identity cache is stale or failed')
        return {'vertices':v,'triangles':f,'vertices_path':vertices_path,'triangles_path':triangles_path,'identity_report':report_path,'report':report}
    if output.exists() and any(output.iterdir()):raise ValueError('Interrupted native ground proof retained; use a new evidence root')
    output.mkdir(parents=True,exist_ok=True)
    process=run_worker([build['binary']['path'],str(mesh_path),str(vertices_path),str(v.offset),str(len(v)),str(triangles_path),str(f.offset),str(len(f)),str(output/'native_comparison.json')],cwd=root,log_path=output/'worker.log',timeout=3600,estimated_memory_bytes=2*2**30,estimated_disk_bytes=2**24)
    result=json.loads((output/'native_comparison.json').read_text());success=bool(result['complete'] and process['exit_code']==0 and not process['timed_out'] and not process['resource_limited'] and result['vertices_compared']==len(v) and result['triangles_compared']==len(f) and result['coordinate_mismatch_vertices']==0 and result['index_mismatch_triangles']==0 and result['maximum_coordinate_error_m']==0)
    verify_records([*inputs,*producers,*dependencies])
    report={'schema_version':1,'kind':'IndependentNativeGroundIdentity','status':'pass' if success else 'fail','native_measurements':result,'input_files':inputs,'producer_files':producers,
        'files':[file_record(output/name,role) for name,role in [('native_comparison.json','all_original_coordinate_index_comparisons'),('worker.process.json','actual_process'),('worker.log','actual_process_log')]],
        'scope':'Every authoritative OBJ vertex/triangle equals retained native arrays; USD/renderer correspondence remains a separate required proof'}
    atomic_json(report_path,report)
    if not success:raise ValueError('Authoritative OBJ does not exactly preserve actual native coordinates/triangles')
    return {'vertices':v,'triangles':f,'vertices_path':vertices_path,'triangles_path':triangles_path,'identity_report':report_path,'report':report}


def native_bounds(vertices,transform):
    transform=np.asarray(transform,float);bounds=np.array([[np.inf]*3,[-np.inf]*3])
    for start in range(0,len(vertices),131072):
        p=vertices[start:start+131072].astype(float)@transform[:3,:3].T+transform[:3,3];bounds[0]=np.minimum(bounds[0],p.min(axis=0));bounds[1]=np.maximum(bounds[1],p.max(axis=0))
    return bounds
