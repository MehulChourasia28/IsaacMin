"""Bounded independent nearest-triangle batches on exact native geometry."""
from pathlib import Path
import json
import numpy as np
from ..io import atomic_json
from ..process import run_worker
from .global_intersections import file_record,_verify_record

CLOSEST_DTYPE=np.dtype([('triangle','<u8'),('distance2','<f8'),('point','<f8',(3,)),('normal','<f8',(3,))])


def proximity_validator_files(workspace):
    path=Path(workspace)/'.tools/geometry_validation/proximity_build_manifest.json'
    manifest=json.loads(path.read_text())
    entries=[manifest['source'],manifest['binary'],manifest['compiler'],manifest['dependency_manifest'],manifest['boost_version_header'],*manifest['runtime_libraries'],*manifest['compile_dependencies']]
    for entry in entries:_verify_record(entry)
    for entry in json.loads(Path(manifest['dependency_manifest']['path']).read_text())['packages']:_verify_record(entry)
    return manifest,[file_record(path,'native_proximity_build_manifest'),file_record(__file__,'independent_proximity_validator')]


def query_native_proximity(workspace,vertices_path,triangles_path,points,output,*,maximum_leaf_faces=2_000_000):
    """Measure global nearest points without approximate distance pruning.

    All original triangles are retained. Every point visits every bounded leaf;
    the closest actual result across leaves is retained with original face ID.
    """
    workspace,output=Path(workspace).resolve(),Path(output).resolve();points=np.asarray(points,dtype='<f8')
    if output.exists() and any(output.iterdir()):raise ValueError('Proximity evidence is immutable')
    if points.ndim!=2 or points.shape[1:]!=(3,) or not 1<=len(points)<=250_000 or not np.isfinite(points).all():raise ValueError('Finite N x 3 closest-point batch required')
    if not 1<=maximum_leaf_faces<=4_000_000:raise ValueError('Native proximity leaf bound exceeded')
    v=np.load(vertices_path,mmap_mode='r',allow_pickle=False);f=np.load(triangles_path,mmap_mode='r',allow_pickle=False)
    if v.dtype!=np.dtype('<f4') or f.dtype!=np.dtype('<i4') or v.shape[1:]!=(3,) or f.shape[1:]!=(3,) or not v.flags.c_contiguous or not f.flags.c_contiguous:raise ValueError('Proximity queries require original C-order f32/i32 arrays')
    output.mkdir(parents=True,exist_ok=True);np.save(output/'points.npy',points);q=np.load(output/'points.npy',mmap_mode='r',allow_pickle=False)
    manifest,producers=proximity_validator_files(workspace)
    inputs=[file_record(vertices_path,'candidate_vertices'),file_record(triangles_path,'candidate_triangles'),file_record(output/'points.npy','actual_query_points')]
    argv=[manifest['binary']['path'],str(Path(vertices_path).resolve()),str(v.offset),str(len(v)),str(Path(triangles_path).resolve()),str(f.offset),str(len(f)),str(output/'points.npy'),str(q.offset),str(len(q)),str(output),str(maximum_leaf_faces)]
    process=run_worker(argv,cwd=workspace,log_path=output/'worker.log',timeout=12*3600,estimated_memory_bytes=10*2**30,estimated_disk_bytes=2*2**30)
    native=json.loads((output/'native_result.json').read_text());raw_path=output/'closest_points.bin'
    if raw_path.stat().st_size!=native['closest_point_records']*CLOSEST_DTYPE.itemsize:raise ValueError('Native closest-point output truncated')
    results=np.fromfile(raw_path,dtype=CLOSEST_DTYPE)
    if len(results) and (np.any(results['triangle']>=len(f)) or not all(np.isfinite(results[name]).all() for name in ['distance2','point','normal']) or np.any(results['distance2']<0)):raise ValueError('Malformed native closest-point results')
    complete=bool(native['complete'] and process['exit_code']==0 and not process['timed_out'] and not process['resource_limited'] and native['vertices']==len(v) and native['original_triangles']==len(f) and native['queries']==len(q) and len(results)==len(q) and native['leaf_face_visits']>=len(f) and native['query_leaf_visits']==len(q)*native['completed_leaves'])
    for entry in [*inputs,*producers]:_verify_record(entry)
    proximity_validator_files(workspace)
    report={'schema_version':1,'kind':'IndependentNativeProximityQueries','status':'complete' if complete else 'incomplete','complete':complete,'query_count':len(q),'native_measurements':native,'input_files':inputs,'producer_files':producers,
        'files':[file_record(output/name,role) for name,role in [('native_result.json','native_measurements'),('closest_points.bin','original_closest_point_results'),('worker.process.json','actual_process'),('worker.log','actual_process_log'),('progress.jsonl','spatial_partition_progress')]],
        'scope':'Independent point-to-original-triangle distances; no surface or source qualification inferred'}
    atomic_json(output/'proximity_queries.json',report)
    if not complete:raise ValueError('Native proximity batch is incomplete')
    return results,report
