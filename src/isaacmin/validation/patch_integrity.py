"""Independent exact patch ownership and unchanged-array verification."""
from collections import Counter
from fractions import Fraction
from itertools import combinations
from pathlib import Path

import numpy as np

from ..io import atomic_json,sha256_file
from .global_intersections import file_record
from .surface_overlap import exact_coplanar_intersection
from . import surface_overlap


def _boundary(faces):
    directed=Counter((int(a),int(b)) for face in faces for a,b in zip(face,np.roll(face,-1)))
    undirected=Counter(tuple(sorted(edge)) for edge in directed.elements())
    invalid=[edge for edge,count in undirected.items() if count>2 or (count==2 and directed[edge]!=1)]
    border=sorted(edge for edge in directed if undirected[tuple(sorted(edge))]==1)
    incoming=Counter(b for a,b in border);outgoing=Counter(a for a,b in border)
    simple=bool(border) and incoming==outgoing and all(count==1 for count in incoming.values())
    if simple:
        nxt=dict(border);start=border[0][0];seen={start};current=nxt[start]
        while current not in seen:seen.add(current);current=nxt[current]
        simple=current==start and len(seen)==len(border)
    return border,invalid,simple


def validate_planar_patch(original_vertices,original_triangles,candidate_vertices,candidate_triangles,patch_path,output,*,context_path,cumulative_attempt):
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('Patch evidence is immutable')
    output.mkdir(parents=True,exist_ok=True)
    paths=[original_vertices,original_triangles,candidate_vertices,candidate_triangles,patch_path,context_path]
    roles=['original_vertices','original_triangles','candidate_vertices','candidate_triangles','declared_patch','continuation_context']
    inputs=[file_record(path,role) for path,role in zip(paths,roles)]
    ov=np.load(original_vertices,mmap_mode='r');of=np.load(original_triangles,mmap_mode='r')
    nv=np.load(candidate_vertices,mmap_mode='r');nf=np.load(candidate_triangles,mmap_mode='r')
    with np.load(patch_path,allow_pickle=False) as data:ids=data['face_ids'];old=data['old_triangles'];new=data['new_triangles']
    if ids.ndim!=1 or len(ids)>100000 or len(np.unique(ids))!=len(ids) or ids.min()<0 or ids.max()>=len(of):raise ValueError('Invalid bounded declared face patch')
    shape_match=ov.shape==nv.shape and of.shape==nf.shape and ov.dtype==nv.dtype and of.dtype==nf.dtype
    changed_vertices=0;changed_faces=0;outside_changed=0;changed_ids=[]
    if shape_match:
        for start in range(0,len(ov),131072):changed_vertices+=int(np.any(ov[start:start+131072]!=nv[start:start+131072],axis=1).sum())
        for start in range(0,len(of),131072):
            changed=np.flatnonzero(np.any(of[start:start+131072]!=nf[start:start+131072],axis=1))+start
            changed_faces+=len(changed);outside_changed+=int((~np.isin(changed,ids)).sum());changed_ids.extend(changed[:max(0,10000-len(changed_ids))].tolist())
    exact_old=bool(np.array_equal(of[ids],old));exact_new=bool(np.array_equal(nf[ids],new)) if shape_match else False
    ob,oi,os=_boundary(old);nb,ni,ns=_boundary(new)
    same_vertex_set=np.array_equal(np.unique(old),np.unique(new))
    overlap_rows=[];plane_errors=[];area_signs=[];area_sum_old=Fraction(0);area_sum_new=Fraction(0)
    triangles=np.asarray(nv[new],float);first=triangles[0]
    normal=np.cross(first[1]-first[0],first[2]-first[0]);axis=int(np.argmax(np.abs(normal)));project=[i for i in range(3) if i!=axis]
    for i,triangle in enumerate(triangles):
        proof=exact_coplanar_intersection(first,triangle)
        if proof['status']!='exactly_coplanar':plane_errors.append(i)
        coords=[[Fraction(float(p[j])) for j in project] for p in triangle]
        signed=sum(coords[k][0]*coords[(k+1)%3][1]-coords[k][1]*coords[(k+1)%3][0] for k in range(3));area_signs.append(1 if signed>0 else -1 if signed<0 else 0);area_sum_new+=signed
    for triangle in np.asarray(ov[old],float):
        coords=[[Fraction(float(p[j])) for j in project] for p in triangle]
        area_sum_old+=sum(coords[k][0]*coords[(k+1)%3][1]-coords[k][1]*coords[(k+1)%3][0] for k in range(3))
    for a,b in combinations(range(len(new)),2):
        proof=exact_coplanar_intersection(triangles[a],triangles[b])
        if proof['positive_area_overlap']:overlap_rows.append({'face_ids':[int(ids[a]),int(ids[b])],**proof})
    failures={'shape_or_dtype_changed':not shape_match,'changed_vertex_positions':changed_vertices,'changed_faces_outside_declared_patch':outside_changed,
        'old_patch_does_not_match_original':not exact_old,'candidate_patch_does_not_match_declared':not exact_new,'vertex_set_changed':not same_vertex_set,
        'oriented_boundary_changed':ob!=nb,'original_patch_topology_errors':len(oi),'candidate_patch_topology_errors':len(ni),
        'boundary_not_single_simple_cycle':not (os and ns),'nonplanar_or_degenerate_replacement_faces':len(plane_errors),
        'inconsistent_or_zero_exact_face_orientation':len(set(area_signs))!=1 or 0 in area_signs,'exact_oriented_area_changed':area_sum_old!=area_sum_new,
        'positive_area_overlap_pairs':len(overlap_rows)}
    np.savez_compressed(output/'patch_measurements.npz',original_face_ids=ids,original_triangles=old,replacement_triangles=new,
        original_boundary=np.asarray(ob),candidate_boundary=np.asarray(nb),triangle_points=triangles,changed_face_id_sample=np.asarray(changed_ids))
    for entry in inputs:
        if sha256_file(Path(entry['path']))!=entry['sha256']:raise ValueError('Patch input changed during verification')
    report={'schema_version':1,'kind':'IndependentPlanarPatchIntegrity','status':'pass' if not any(failures.values()) else 'fail',
        'cumulative_attempt':cumulative_attempt,'failures':failures,'original_vertices':len(ov),'candidate_vertices':len(nv),'original_triangles':len(of),'candidate_triangles':len(nf),
        'whole_array_faces_compared':len(of) if shape_match else 0,'whole_array_vertices_compared':len(ov) if shape_match else 0,
        'declared_patch_faces':len(ids),'actually_changed_faces':changed_faces,'preserved_patch_vertices':len(np.unique(new)),'preserved_oriented_boundary_edges':len(nb),
        'original_projected_signed_area2_exact':str(area_sum_old),'candidate_projected_signed_area2_exact':str(area_sum_new),'overlap_pairs':overlap_rows,
        'input_files':inputs,'producer_files':[file_record(Path(__file__),'independent_patch_validator'),file_record(Path(surface_overlap.__file__),'exact_rational_overlap_predicate')],
        'files':[file_record(output/'patch_measurements.npz','raw_patch_measurements')],
        'scope':'Exact original float32 coordinate/triangle ownership and all replacement triangle pairs; not a global intersection or source/renderer qualification'}
    atomic_json(output/'patch_integrity.json',report);return report
