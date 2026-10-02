"""Independent bounded native self-intersection evidence for original arrays."""
from pathlib import Path
import json

import numpy as np

from ..io import atomic_json,sha256_file
from ..process import run_worker
from ..security import safe_path


def file_record(path,role):
    path=Path(path).resolve()
    return {'role':role,'path':str(path),'sha256':sha256_file(path),'bytes':path.stat().st_size}


def _verify_record(entry):
    path=Path(entry['path'])
    if sha256_file(path)!=entry['sha256'] or ('bytes' in entry and path.stat().st_size!=entry['bytes']):
        raise ValueError('Native validation dependency changed: '+str(path))


def native_validator_files(workspace):
    workspace=Path(workspace);manifest_path=workspace/'.tools/geometry_validation/build_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    entries=[manifest['source'],manifest['binary'],manifest['compiler'],manifest['dependency_manifest'],manifest['boost_version_header'],*manifest['runtime_libraries'],*manifest.get('compile_dependencies',[])]
    for entry in entries:_verify_record(entry)
    dependency=json.loads(Path(manifest['dependency_manifest']['path']).read_text())
    for entry in dependency['packages']:_verify_record(entry)
    files=[file_record(manifest_path,'native_build_manifest'),file_record(Path(__file__),'independent_global_validator')]
    files.extend(file_record(entry['path'],'native_dependency') for entry in entries)
    files.extend(file_record(entry['path'],'dependency_archive') for entry in dependency['packages'])
    return manifest,files


def validate_global_intersections(workspace,vertices_path,triangles_path,output,*,
                                  context_path=None,cumulative_attempt=None,maximum_leaf_faces=2_000_000,maximum_pairs=1_000_000):
    """All original triangle pairs, adaptively partitioned without geometry cuts.

    Exhaustion or an unsplittable oversized cell yields incomplete, never pass.
    A pair is emitted once by its original AABB-overlap minimum's owning cell.
    """
    workspace=Path(workspace).resolve();output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError('Global evidence is immutable')
    if not 1<=maximum_leaf_faces<=4_000_000 or not 1<=maximum_pairs<=1_000_000:raise ValueError('Unbounded native diagnostic request')
    output.mkdir(parents=True,exist_ok=True)
    v=np.load(vertices_path,mmap_mode='r',allow_pickle=False);f=np.load(triangles_path,mmap_mode='r',allow_pickle=False)
    if v.dtype!=np.dtype('<f4') or f.dtype!=np.dtype('<i4') or v.ndim!=2 or f.ndim!=2 or v.shape[1]!=3 or f.shape[1]!=3 or not v.flags.c_contiguous or not f.flags.c_contiguous:
        raise ValueError('Native checker requires unchanged C-order float32 vertices and int32 triangle triples')
    manifest,producers=native_validator_files(workspace)
    inputs=[file_record(vertices_path,'candidate_vertices'),file_record(triangles_path,'candidate_triangles')]
    if context_path is not None:inputs.append(file_record(context_path,'continuation_context'))
    argv=[manifest['binary']['path'],str(Path(vertices_path).resolve()),str(v.offset),str(len(v)),str(Path(triangles_path).resolve()),str(f.offset),str(len(f)),str(output),str(maximum_leaf_faces),str(maximum_pairs)]
    process=run_worker(argv,cwd=workspace,log_path=output/'worker.log',timeout=12*3600,estimated_memory_bytes=12*2**30,estimated_disk_bytes=2**30)
    native=json.loads((output/'native_result.json').read_text())
    path=output/'intersection_pairs.u64'
    if path.stat().st_size!=16*native['intersection_pairs']:raise ValueError('Native intersection output is truncated')
    pairs=np.fromfile(path,dtype='<u8').reshape(-1,2)
    if len(pairs) and (pairs.max()>=len(f) or np.any(pairs[:,0]>pairs[:,1]) or len(np.unique(pairs,axis=0))!=len(pairs)):
        raise ValueError('Native ownership returned invalid or duplicate original pair IDs')
    complete=bool(native['complete'] and process['exit_code']==0 and not process['timed_out'] and not process['resource_limited'] and native['original_triangles']==len(f) and native['vertices']==len(v) and native['leaf_face_visits']>=len(f))
    for entry in [*inputs,*producers]:_verify_record(entry)
    files=[file_record(output/name,role) for name,role in [('native_result.json','native_measurements'),('intersection_pairs.u64','original_triangle_intersection_pairs'),('worker.process.json','actual_process'),('worker.log','actual_process_log'),('progress.jsonl','spatial_partition_progress')]]
    report={'schema_version':1,'kind':'IndependentGlobalTriangleIntersections','status':'fail' if len(pairs) else 'pass' if complete else 'incomplete',
        'complete':complete,'candidate_vertices':len(v),'candidate_triangles':len(f),'intersection_pair_count':len(pairs),
        'degenerate_triangles':int(np.count_nonzero(pairs[:,0]==pairs[:,1])) if len(pairs) else 0,
        'cumulative_attempt':cumulative_attempt,'native_measurements':native,'input_files':inputs,'producer_files':producers,'files':files,
        'predicate':'CGAL exact-predicate triangle-soup self-intersections; legitimate original shared vertices/edges excluded, coplanar positive-area overlaps retained',
        'partition_completeness':'Full original triangles included in every touching cell. Each AABB-overlap minimum lies in one half-open owning cell; no faces or possible intersecting pairs omitted.',
        'geometry_mutations':0,'full_world_qualification':'not_run','limits':['Self-intersection freedom alone does not establish source fidelity, watertightness, native normals or renderer realism.','Input topology is original native vertex identity; attribute-split external meshes require a separately justified exact-position topology preparation.']}
    atomic_json(output/'global_intersections.json',report)
    return report
