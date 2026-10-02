"""Content-bound generation cache for the independently tested source repairs."""
from pathlib import Path
import json
from ..io import hash_object,sha256_file,atomic_json,read_json
from ..process import run_worker


def constrained_terrain(workspace:Path, ir:Path, volume:Path, *, support_mask_path=None) -> dict:
    graph=ir/'topology/source_topology_graph.json'
    if not graph.is_file():
        return {'mesh':str(volume/'terrain.obj'),'scope':'technical_fixture_without_source_topology',
                'source_preservation':'not_applicable'}
    inputs=[volume/'terrain.obj',ir/'world_ir.json',ir/'natural_occupancy.npz',ir/'terrain_surface.npz',graph,
            ir/'topology/source_topology_labels.npz',workspace/'src/isaacmin/volumes/interface_projection.py',
            workspace/'scripts/native/interface_projection_worker.py',workspace/'scripts/native/portal_air_worker.py']
    if support_mask_path:inputs.append(Path(support_mask_path))
    lock=read_json(workspace/'artifacts/bootstrap/native-build-lock.json')
    identity={'files':{str(p.resolve().relative_to(workspace)):sha256_file(p) for p in inputs},
              'geometry_python_packages':lock['geometry_python_packages']}
    key=hash_object(identity);directory=volume/'source_constraints'/key
    record_path=directory/'construction.json';mesh=directory/'portal/terrain.obj'
    if record_path.is_file():
        record=read_json(record_path)
        if record['identity']!=identity or not mesh.is_file() or sha256_file(mesh)!=record['mesh_sha256']:
            raise ValueError('Cached source repair identity/geometry changed')
        for name,digest in record['evidence_hashes'].items():
            if sha256_file(directory/name)!=digest:raise ValueError('Cached source repair evidence changed')
        return {**record,'mesh':str(mesh)}
    if directory.exists() and any(directory.iterdir()):
        raise ValueError('Incomplete constrained-volume attempt retained; inspect logs before choosing a new build identity')
    directory.mkdir(parents=True,exist_ok=True)
    request={'ir':str(ir),'mesh':str(volume/'terrain.obj'),'output':str(directory/'projected'),
             'support_mask':str(support_mask_path) if support_mask_path else None}
    request_path=directory/'projection_request.json';atomic_json(request_path,request)
    stages=[([str(workspace/'.venv/bin/python'),str(workspace/'scripts/native/interface_projection_worker.py'),
              '--request',str(request_path)],'projection_worker.log'),
            ([str(workspace/'.tools/manifold-py/bin/python'),str(workspace/'scripts/native/portal_air_worker.py'),
              '--mesh',str(directory/'projected/terrain.obj'),'--ir',str(ir),'--output',str(directory/'portal')],
             'portal_worker.log')]
    for command,log in stages:
        result=run_worker(command,cwd=workspace,log_path=directory/log,timeout=1800,
            environment={'PYTHONPATH':str(workspace/'src'),'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'8'},
            estimated_memory_bytes=8*2**30,estimated_disk_bytes=2**30)
        if result['exit_code']!=0:raise RuntimeError('Source repair worker failed; preserved '+str(directory/log))
    evidence=['projected/projection.json','projected/constraints.npz','portal/portal_repair.json',
              'portal/source_portal_air_cells.npz']
    record={'identity':identity,'key':key,'mesh_sha256':sha256_file(mesh),
            'evidence_hashes':{name:sha256_file(directory/name) for name in evidence},
            'construction_status':'generated_candidate','repair_cycle':3,
            'independent_source_validation':'not_run','appearance_qualification':'not_run'}
    atomic_json(record_path,record)
    return {**record,'mesh':str(mesh)}
