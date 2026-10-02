"""Memory-bounded, read-only original-triangle segment queries via pinned CGAL."""
from pathlib import Path
import json

import numpy as np

from ..io import atomic_json
from ..process import run_worker
from .global_intersections import file_record, _verify_record

HIT_DTYPE=np.dtype([('query','<u8'),('triangle','<u8'),('kind','<u8'),('first','<f8',(3,)),('second','<f8',(3,)),('normal','<f8',(3,))])


def segment_validator_files(workspace):
    path=Path(workspace)/'.tools/geometry_validation/segment_build_manifest.json'
    manifest=json.loads(path.read_text())
    entries=[manifest['source'],manifest['binary'],manifest['compiler'],manifest['dependency_manifest'],manifest['boost_version_header'],*manifest['runtime_libraries'],*manifest['compile_dependencies']]
    for entry in entries:_verify_record(entry)
    for entry in json.loads(Path(manifest['dependency_manifest']['path']).read_text())['packages']:_verify_record(entry)
    # The pinned manifest recursively binds the complete headers/libraries. Avoid
    # duplicating thousands of dependency rows in every numerical report.
    return manifest,[file_record(path,'native_segment_build_manifest'),file_record(__file__,'independent_segment_validator')]


def query_native_segments(workspace, vertices_path, triangles_path, segments, output, *, maximum_leaf_faces=2_000_000, maximum_hits=10_000_000):
    """Return all actual finite-segment intersections; never silently omit hits.

    Segments are N x 2 x 3 endpoints in the original vertex frame. Spatial cells
    retain full triangles, and the original triangle/query identity removes only
    repeated observations across cells. Distinct faces sharing a geometric edge
    remain distinct records for the caller's explicit parity/tangency policy.
    """
    workspace,output=Path(workspace).resolve(),Path(output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError('Segment evidence is immutable')
    segments=np.asarray(segments,dtype='<f8')
    if segments.ndim!=3 or segments.shape[1:]!=(2,3) or not 1<=len(segments)<=1_000_000 or not np.isfinite(segments).all() or np.any(np.all(segments[:,0]==segments[:,1],axis=1)):
        raise ValueError('Finite nondegenerate N x 2 x 3 segment endpoints required')
    if not 1<=maximum_leaf_faces<=4_000_000 or not 1<=maximum_hits<=10_000_000:raise ValueError('Native query bounds exceeded')
    v=np.load(vertices_path,mmap_mode='r',allow_pickle=False);f=np.load(triangles_path,mmap_mode='r',allow_pickle=False)
    if v.dtype!=np.dtype('<f4') or f.dtype!=np.dtype('<i4') or v.ndim!=2 or f.ndim!=2 or v.shape[1]!=3 or f.shape[1]!=3 or not v.flags.c_contiguous or not f.flags.c_contiguous:raise ValueError('Native segment queries require original C-order f32/i32 arrays')
    output.mkdir(parents=True,exist_ok=True);np.save(output/'segments.npy',segments.reshape(-1,6))
    q=np.load(output/'segments.npy',mmap_mode='r',allow_pickle=False)
    manifest,producers=segment_validator_files(workspace)
    inputs=[file_record(vertices_path,'candidate_vertices'),file_record(triangles_path,'candidate_triangles'),file_record(output/'segments.npy','actual_query_endpoints')]
    argv=[manifest['binary']['path'],str(Path(vertices_path).resolve()),str(v.offset),str(len(v)),str(Path(triangles_path).resolve()),str(f.offset),str(len(f)),str(output/'segments.npy'),str(q.offset),str(len(q)),str(output),str(maximum_leaf_faces),str(maximum_hits)]
    process=run_worker(argv,cwd=workspace,log_path=output/'worker.log',timeout=12*3600,estimated_memory_bytes=10*2**30,estimated_disk_bytes=2*2**30)
    native=json.loads((output/'native_result.json').read_text());raw_path=output/'segment_hits.bin'
    if raw_path.stat().st_size!=native['raw_intersections']*HIT_DTYPE.itemsize:raise ValueError('Native segment output truncated')
    raw=np.fromfile(raw_path,dtype=HIT_DTYPE)
    if len(raw) and (raw['query'].max()>=len(segments) or raw['triangle'].max()>=len(f) or np.any(raw['kind']>1) or not all(np.isfinite(raw[k]).all() for k in ['first','second','normal'])):raise ValueError('Malformed native intersection records')
    order=np.lexsort((raw['triangle'],raw['query']));raw=raw[order]
    unique=np.r_[True,(raw['triangle'][1:]!=raw['triangle'][:-1])|(raw['query'][1:]!=raw['query'][:-1])] if len(raw) else np.zeros(0,bool)
    if np.any(~unique):
        # The same primitive/query construction must be bit-identical whichever
        # spatial tree happens to contain it. No tolerance can hide disagreement.
        repeat=np.flatnonzero(~unique)
        for field in ['kind','first','second','normal']:
            if not np.array_equal(raw[field][repeat],raw[field][repeat-1]):raise ValueError('Repeated original query/triangle yielded different intersection constructions')
    hits=raw[unique];np.save(output/'unique_hits.npy',hits)
    complete=bool(native['complete'] and process['exit_code']==0 and not process['timed_out'] and not process['resource_limited'] and native['vertices']==len(v) and native['original_triangles']==len(f) and native['queries']==len(q) and native['leaf_face_visits']>=len(f))
    for entry in [*inputs,*producers]:_verify_record(entry)
    # Re-open the closure to verify dependencies were unchanged during execution.
    segment_validator_files(workspace)
    report={'schema_version':1,'kind':'IndependentNativeSegmentQueries','status':'complete' if complete else 'incomplete','complete':complete,
        'query_count':len(q),'unique_triangle_intersections':len(hits),'coplanar_segment_intersections':int(np.count_nonzero(hits['kind']==1)),
        'spatial_query_triangle_duplicates':int((~unique).sum()),'native_measurements':native,'input_files':inputs,'producer_files':producers,
        'files':[file_record(output/name,role) for name,role in [('native_result.json','native_measurements'),('segment_hits.bin','raw_triangle_hits'),('unique_hits.npy','identity_deduplicated_hits'),('worker.process.json','actual_process'),('worker.log','actual_process_log'),('progress.jsonl','spatial_partition_progress')]],
        'scope':'All requested finite segments versus unchanged original triangles. Query completion does not establish source fidelity, surface closure or world qualification.'}
    atomic_json(output/'segment_queries.json',report)
    return hits,report
