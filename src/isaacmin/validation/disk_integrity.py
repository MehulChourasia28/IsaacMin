"""Independent ownership/topology proof for bounded nonplanar disk retriangulation."""
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

from ..io import atomic_json
from .global_intersections import file_record,_verify_record
from .patch_integrity import _boundary
from . import patch_integrity


def _disk(faces):
    boundary,edge_errors,simple=_boundary(faces)
    edges=defaultdict(list)
    for i,face in enumerate(faces):
        for a,b in zip(face,np.roll(face,-1)):edges[tuple(sorted((int(a),int(b))))].append(i)
    neighbours=[set() for _ in faces]
    for ids in edges.values():
        for a in ids:neighbours[a].update(ids)
    seen=set();todo=[0]
    while todo:
        face=todo.pop()
        if face in seen:continue
        seen.add(face);todo.extend(neighbours[face]-seen)
    vertices=np.unique(faces);link_errors=[];boundary_vertices={v for edge in boundary for v in edge}
    for vertex in vertices:
        link=defaultdict(set)
        for face in faces[np.any(faces==vertex,axis=1)]:
            others=[int(i) for i in face if i!=vertex]
            if len(others)!=2:link_errors.append(int(vertex));break
            link[others[0]].add(others[1]);link[others[1]].add(others[0])
        if not link:continue
        visited=set();remaining=[next(iter(link))]
        while remaining:
            p=remaining.pop()
            if p in visited:continue
            visited.add(p);remaining.extend(link[p]-visited)
        degrees=Counter(map(len,link.values()))
        valid=(degrees.get(1,0)==2 and all(k in (1,2) for k in degrees)) if vertex in boundary_vertices else set(degrees)=={2}
        if not valid or len(visited)!=len(link):link_errors.append(int(vertex))
    return {'oriented_boundary':boundary,'edge_incidence_errors':len(edge_errors),'single_simple_boundary':simple,
        'face_connected':len(seen)==len(faces),'euler_characteristic':len(vertices)-len(edges)+len(faces),'nonmanifold_vertex_links':sorted(set(link_errors))}


def validate_disk_retriangulation(original_vertices,original_triangles,candidate_vertices,candidate_triangles,declared_patch_path,output,*,context_path,cumulative_attempt):
    """Exact vertex/boundary/outside-patch identity, independent of constructor.

    This deliberately does not assert surface equality for nonplanar diagonal
    changes. Global self-intersections and frozen source distances remain
    separate mandatory reports. Patch metadata supplies IDs, never pass values.
    """
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('Disk-repair evidence is immutable')
    output.mkdir(parents=True,exist_ok=True)
    paths=[original_vertices,original_triangles,candidate_vertices,candidate_triangles,declared_patch_path,context_path]
    roles=['original_vertices','original_triangles','candidate_vertices','candidate_triangles','declared_patch_ids','continuation_context']
    inputs=[file_record(path,role) for path,role in zip(paths,roles)]
    ov=np.load(original_vertices,mmap_mode='r');of=np.load(original_triangles,mmap_mode='r');nv=np.load(candidate_vertices,mmap_mode='r');nf=np.load(candidate_triangles,mmap_mode='r')
    with np.load(declared_patch_path,allow_pickle=False) as data:ids=data['face_ids']
    if ids.ndim!=1 or not 1<=len(ids)<=2048 or not np.issubdtype(ids.dtype,np.integer) or len(np.unique(ids))!=len(ids) or ids.min()<0 or ids.max()>=len(of):raise ValueError('Invalid bounded original patch face IDs')
    if ov.shape!=nv.shape or of.shape!=nf.shape or ov.dtype!=nv.dtype or of.dtype!=nf.dtype:raise ValueError('Disk repair changed original array shape or dtype')
    changed_vertices=0;changed=[]
    for start in range(0,len(ov),131072):changed_vertices+=int(np.any(ov[start:start+131072]!=nv[start:start+131072],axis=1).sum())
    for start in range(0,len(of),131072):changed.extend((np.flatnonzero(np.any(of[start:start+131072]!=nf[start:start+131072],axis=1))+start).tolist())
    old,new=np.asarray(of[ids]),np.asarray(nf[ids]);original_disk,candidate_disk=_disk(old),_disk(new)
    same_vertices=np.array_equal(np.unique(old),np.unique(new));outside=np.setdiff1d(changed,ids)
    def bad_disk(d):return bool(d['edge_incidence_errors'] or not d['single_simple_boundary'] or not d['face_connected'] or d['euler_characteristic']!=1 or d['nonmanifold_vertex_links'])
    p=np.asarray(nv[new],float);cross=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);area=np.linalg.norm(cross,axis=1)
    new_max=float(np.linalg.norm(p-np.roll(p,-1,axis=1),axis=2).max());oldp=np.asarray(ov[old],float);old_max=float(np.linalg.norm(oldp-np.roll(oldp,-1,axis=1),axis=2).max())
    failures={'changed_vertex_coordinates':changed_vertices,'unannounced_changed_faces':len(outside),'patch_vertex_set_changed':not same_vertices,
        'oriented_boundary_changed':original_disk['oriented_boundary']!=candidate_disk['oriented_boundary'],'original_not_single_manifold_disk':bad_disk(original_disk),'replacement_not_single_manifold_disk':bad_disk(candidate_disk),
        'degenerate_or_nonfinite_replacement_faces':int(np.count_nonzero(~np.isfinite(area)|(area==0))),'maximum_edge_sampling_coarsened':new_max>old_max}
    np.savez_compressed(output/'disk_measurements.npz',face_ids=ids,original_triangles=old,replacement_triangles=new,original_points=oldp,replacement_points=p,actually_changed_faces=np.asarray(changed),unannounced_changed_faces=outside)
    for entry in inputs:_verify_record(entry)
    report={'schema_version':1,'kind':'IndependentLocalDiskRepairIntegrity','status':'pass' if not any(failures.values()) else 'fail','cumulative_attempt':cumulative_attempt,
        'whole_array_vertices_compared':len(ov),'whole_array_faces_compared':len(of),'declared_patch_faces':len(ids),'actually_changed_faces':len(changed),'failures':failures,
        'original_disk':original_disk,'replacement_disk':candidate_disk,'original_maximum_patch_edge_m':old_max,'replacement_maximum_patch_edge_m':new_max,'minimum_replacement_area2_m2':float(area.min()),
        'input_files':inputs,'producer_files':[file_record(__file__,'independent_disk_validator'),file_record(patch_integrity.__file__,'independent_boundary_helper')],'files':[file_record(output/'disk_measurements.npz','original_disk_measurements')],
        'surface_equality_claim':False,'global_intersection_status':'not_run_in_this_component','source_fidelity_status':'not_run_in_this_component','world_qualification':'not_run',
        'scope':'Exact original native vertices, owned face changes, oriented boundary and manifold disk topology. Nonplanar replacement requires separate global intersections and unchanged frozen source interface/roof budgets.'}
    atomic_json(output/'disk_integrity.json',report);return report
