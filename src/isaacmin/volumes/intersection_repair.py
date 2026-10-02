"""Bounded local construction from measured native crossing components.

A projected disk is a candidate construction, never independent qualification.
All original 3D vertices and directed boundary edges remain unchanged.
"""
from pathlib import Path
import numpy as np
from isaacmin.io import atomic_json,sha256_file
from isaacmin.validation.global_intersections import validate_global_intersections
from .projected_patch import triangulate_projected_patch


def repair_coarse_intersections(workspace,vertices,triangles,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    v=np.asarray(vertices,np.float32);f=np.asarray(triangles,np.int32).copy()
    np.save(output/'input_vertices.npy',v,allow_pickle=False)
    np.save(output/'input_triangles.npy',f,allow_pickle=False)
    discovery=validate_global_intersections(workspace,output/'input_vertices.npy',output/'input_triangles.npy',output/'discovery')
    if not discovery['complete']:raise RuntimeError('Intersection discovery incomplete')
    pairs=np.fromfile(output/'discovery/intersection_pairs.u64',np.uint64).reshape(-1,2)
    if np.any(pairs[:,0]==pairs[:,1]):raise RuntimeError('Degenerate coarse faces require precision conditioning before disk repair')
    parent={int(i):int(i) for i in np.unique(pairs)}
    def find(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for a,b in pairs:
        a,b=find(int(a)),find(int(b));parent[max(a,b)]=min(a,b)
    groups={}
    for i in sorted(parent):groups.setdefault(find(i),[]).append(i)
    records=[];claimed=set()
    for seed,group in sorted(groups.items()):
        patch=np.asarray(group,np.int64);candidates=[];rejections=[]
        for ring in range(5):
            if claimed.intersection(patch.tolist()):raise RuntimeError('Repair collars overlap; require a combined bounded patch')
            old=f[patch].copy();points=v[old].astype(float)
            maximum=float(np.linalg.norm(points-points[:,[1,2,0]],axis=2).max())
            for axis in range(3):
                try:
                    replacement,info=triangulate_projected_patch(v,old,axis)
                    if len(replacement)!=len(old) or info['maximum_edge_m']>maximum+1e-12:
                        raise ValueError('Physical edge sampling or face count would be reduced')
                    candidates.append((info['minimum_native_area2'],-axis,replacement,info))
                except ValueError as exc:rejections.append({'ring':ring,'axis':axis,'reason':str(exc)})
            if candidates:break
            selected=np.zeros(len(v),bool);selected[np.unique(old)]=True
            expanded=np.flatnonzero(selected[f].any(axis=1))
            if len(expanded)<=len(patch) or len(expanded)>4096:break
            patch=expanded
        if not candidates:
            atomic_json(output/'unrepairable_patch.json',{'seed_face':seed,'rejections':rejections})
            raise RuntimeError('No admissible projected disk for native intersection component')
        _,_,replacement,info=max(candidates,key=lambda c:(c[0],c[1]))
        # Canonical oriented triples make construction independent of traversal.
        replacement=np.asarray([np.roll(row,-int(np.argmin(row))) for row in replacement],np.int32)
        replacement=replacement[np.lexsort(replacement.T[::-1])]
        original=f[patch].copy();f[patch]=replacement;claimed.update(patch.tolist())
        path=output/f'patch_{len(records):04d}.npz'
        np.savez_compressed(path,face_ids=patch,original_triangles=original,replacement_triangles=replacement)
        records.append(dict(info,seed_face=seed,ring=ring,face_ids=patch.tolist(),patch_file_sha256=sha256_file(path),projection_rejections=rejections,
                            original_maximum_edge_m=maximum))
    np.save(output/'vertices.npy',v,allow_pickle=False);np.save(output/'triangles.npy',f,allow_pickle=False)
    final=validate_global_intersections(workspace,output/'vertices.npy',output/'triangles.npy',output/'post_construction')
    report={'kind':'NativeCrossingConstruction','status':'constructed_candidate' if final['status']=='pass' else 'failed',
            'scope':'construction discovery and candidate check; independent source/surface proof remains required',
            'input_intersections':len(pairs),'output_intersections':final['intersection_pair_count'],'patches':records,
            'vertices_moved':0,'faces_before':len(triangles),'faces_after':len(f),
            'input_vertices_sha256':sha256_file(output/'input_vertices.npy'),'input_triangles_sha256':sha256_file(output/'input_triangles.npy'),
            'output_vertices_sha256':sha256_file(output/'vertices.npy'),'output_triangles_sha256':sha256_file(output/'triangles.npy'),
            'geometry_constraints':'all original native positions, patch boundary edges, face count and maximum edge sampling preserved'}
    atomic_json(output/'construction.json',report)
    if final['status']!='pass':raise RuntimeError('Crossings remain after bounded local disk construction')
    return v,f,report
