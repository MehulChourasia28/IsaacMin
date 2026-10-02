"""Subtract exact retained portal air; no inferred source cell is excavated.

Executed in the pinned private manifold-py ARM64 environment. This is the
third bounded source-interface repair candidate, not a quality certificate.
"""
from __future__ import annotations
import argparse,hashlib,json,time
from pathlib import Path
from importlib.metadata import version
import numpy as np
import trimesh
import manifold3d as manifold


def repair(mesh_path: Path, ir: Path, output: Path) -> dict:
    started=time.monotonic();output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise ValueError('Choose a new portal repair directory; retain earlier attempts')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    data=np.load(ir/'natural_occupancy.npz',allow_pickle=False)
    occupied=data['occupancy'];valid=data['validity'];minimum=data['min_xyz']
    graph_path=ir/'topology/source_topology_graph.json';graph=json.loads(graph_path.read_text())
    if graph['source_ir_sha256']!=sha(ir/'world_ir.json'):raise ValueError('Portal graph/source identity mismatch')
    cells=set();faces=0
    for portal in graph['portals']:
        for face in portal['faces']:
            axis='xyz'.index(face['normal_axis']);centre=np.asarray(face['center_xyz'],float)
            for sign in (-1,1):
                point=centre.copy();point[axis]+=.25*sign
                xyz=np.floor(point).astype(np.int64)-minimum
                y,z,x=map(int,xyz[[1,2,0]])
                if not(0<=y<occupied.shape[0] and 0<=z<occupied.shape[1] and 0<=x<occupied.shape[2]):
                    raise ValueError('Observed portal cutter extends outside known source coverage')
                if not valid[y,z,x] or occupied[y,z,x]!=0:
                    raise ValueError('Portal subtraction would remove a non-air or unknown source cell')
                cells.add((x,y,z))
            faces+=1
    source=trimesh.load(mesh_path,force='mesh',process=False)
    vertices,inverse=np.unique(source.vertices,axis=0,return_inverse=True)
    source=trimesh.Trimesh(vertices,inverse[source.faces],process=False)
    if not source.is_volume:raise ValueError('Portal repair input must be closed positive terrain')
    native=manifold.Manifold(manifold.Mesh64(np.asarray(source.vertices,dtype=np.float64),
                                          np.asarray(source.faces,dtype=np.uint64)))
    if native.status()!=manifold.Error.NoError:raise ValueError('Native manifold rejected source mesh: '+str(native.status()))
    # Every cutter is precisely one retained source air cube, transformed into
    # native X=x,Y=-z,Z=y coordinates. Their Boolean union has no internal caps.
    cubes=[manifold.Manifold.cube().translate((x-.5,-z-.5,y-.5)) for x,y,z in sorted(cells)]
    cutter=manifold.Manifold.batch_boolean(cubes,manifold.OpType.Add)
    result=manifold.Manifold.batch_boolean([native,cutter],manifold.OpType.Subtract)
    if result.status()!=manifold.Error.NoError:raise ValueError('Portal Boolean failed: '+str(result.status()))
    result_mesh=result.to_mesh64()
    terrain=trimesh.Trimesh(np.asarray(result_mesh.vert_properties)[:,:3],np.asarray(result_mesh.tri_verts),process=False)
    if not terrain.is_volume or not np.isfinite(terrain.vertices).all():raise ValueError('Portal subtraction lost positive closed terrain')
    if terrain.volume>source.volume+1e-7:raise ValueError('Subtraction unexpectedly added terrain volume')
    target=output/'terrain.obj';terrain.export(target,digits=12)
    cells_path=output/'source_portal_air_cells.npz'
    np.savez_compressed(cells_path,source_cell_xyz=np.asarray(sorted(cells),dtype=np.int32)+minimum,
                        min_xyz=minimum)
    record={'status':'generated_candidate','repair_cycle':3,'repair_class':'source_interface_geometry',
            'method':'double_precision_manifold_boolean_difference_of_exact_known_portal_air_union',
            'source_mesh_sha256':sha(mesh_path),'source_ir_sha256':sha(ir/'world_ir.json'),
            'source_graph_sha256':sha(graph_path),'source_occupancy_sha256':sha(ir/'natural_occupancy.npz'),
            'worker_sha256':sha(Path(__file__)),'manifold_version':version('manifold3d'),
            'mesh':str(target.resolve()),'mesh_sha256':sha(target),'vertices':len(terrain.vertices),
            'triangles':len(terrain.faces),'positive_closed_mesh':bool(terrain.is_volume),
            'source_portal_faces':faces,'portal_air_cells':len(cells),'source_solid_cells_removed':0,
            'source_air_cells':cells_path.name,'source_air_cells_sha256':sha(cells_path),
            'removed_rounded_surface_volume_m3':float(source.volume-terrain.volume),
            'source_frame_min_xyz':minimum.tolist(),'coordinate_frame':'Native OpenVDB voxel centres X=x,Y=-z,Z=y; source origin min_xyz+0.5',
            'independent_source_validation':'not_run','appearance_qualification':'not_run',
            'limitations':['Exact source portal margins can retain voxel-shaped transitions; no realism pass is implied',
                           'Only known source air adjoining observed portals was subtracted'],
            'elapsed_seconds':time.monotonic()-started}
    (output/'portal_repair.json').write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--mesh',required=True,type=Path)
    parser.add_argument('--ir',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();print(json.dumps(repair(args.mesh,args.ir,args.output)),flush=True)
