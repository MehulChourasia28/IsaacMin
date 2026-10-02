"""Bounded native-coordinate conditioning with immutable source-plane coordinates."""
from pathlib import Path
import time
import numpy as np
from isaacmin.io import atomic_json


def condition_planes(vertices,triangles,source_normals,anchors,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    v=np.asarray(vertices,np.float32).astype(float);f=np.asarray(triangles,np.int32)
    original=v.copy();refs=np.asarray(source_normals,float).copy();anchors=np.asarray(anchors,bool)
    length=np.linalg.norm(refs,axis=1)
    if np.any(length==0) or not np.isfinite(refs).all():raise ValueError('Source face orientation is undetermined')
    refs/=length[:,None]
    p=v[f];edge=np.linalg.norm(p-p[:,[1,2,0]],axis=2).max(axis=1)
    normal=np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);orientation=np.einsum('ij,ij->i',normal,refs)
    if np.any(orientation==0):raise ValueError('Native coarse face orientation cannot be determined')
    # These directions constrain actual point movement, never authored normals.
    # The original source orientation decides the sign of the native face plane.
    refs=normal/np.linalg.norm(normal,axis=1)[:,None]*np.where(orientation<0,-1,1)[:,None]
    signed=np.einsum('ij,ij->i',normal,refs);altitude=signed/edge
    target_altitude=.0002;maximum_displacement=.0002
    bad=np.flatnonzero(altitude<target_altitude);selected=np.unique(f[bad]);mask=np.zeros(len(v),bool);mask[selected]=True
    local=np.flatnonzero(mask[f].any(axis=1));freedom=(~anchors)&mask[:,None]
    targets=np.minimum(np.maximum(signed[local],0)*.5,target_altitude*edge[local]);is_bad=np.isin(local,bad)
    targets[is_bad]=target_altitude*edge[local][is_bad];records=[];remaining=np.zeros(len(local),bool)
    for iteration in range(256):
        changes=0
        for row,target in zip(local,targets):
            ids=f[row];a,b,c=v[ids];n=refs[row];value=float(np.dot(np.cross(b-a,c-a),n))
            if value>=target:continue
            gradient=np.array([np.cross(n,c-b),np.cross(n,a-c),np.cross(n,b-a)])*freedom[ids]
            denominator=float(np.square(gradient).sum())
            if denominator<=1e-30:continue
            proposed=v[ids]+gradient*((target-value)*.9/denominator)
            delta=proposed-original[ids];size=np.linalg.norm(delta,axis=1)
            quantization_reserve=np.linalg.norm(np.spacing(original[ids].astype(np.float32)).astype(float)*freedom[ids],axis=1)
            scale=np.minimum(1,(maximum_displacement-quantization_reserve)/np.maximum(size,1e-300))
            proposed=original[ids]+delta*scale[:,None];proposed=proposed.astype(np.float32).astype(float)
            proposed=np.where(anchors[ids],original[ids],proposed)
            if np.any(proposed!=v[ids]):changes+=1;v[ids]=proposed
        q=v[f[local]];value=np.einsum('ij,ij->i',np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]),refs[local])
        remaining=value<targets*.98
        records.append({'iteration':iteration,'conditioning_targets_below_98_percent':int(remaining.sum()),'updates':changes})
        if not remaining.any() or not changes:break
    q=v[f];normal=np.cross(q[:,1]-q[:,0],q[:,2]-q[:,0]);area=np.linalg.norm(normal,axis=1)
    signed=np.einsum('ij,ij->i',normal,refs);delta=np.linalg.norm(v-original,axis=1)
    invariant=bool(np.isfinite(v).all() and delta.max(initial=0)<=maximum_displacement and
                   not np.any((v!=original)&anchors) and np.all(area>1e-12) and np.all(signed>0))
    np.savez_compressed(output/'displacement_witnesses.npz',vertex_ids=np.flatnonzero(delta),before=original[delta>0],
        after=v[delta>0],anchors=anchors[delta>0],original_bad_faces=bad,unmet_predictor_faces=local[remaining])
    np.save(output/'vertices.npy',v.astype(np.float32),allow_pickle=False);np.save(output/'triangles.npy',f,allow_pickle=False)
    report={'status':'bounded_candidate' if invariant else 'failed','method':'Point constraints preserve fixed source-grid coordinates, topology and original source-facing orientation; no vertex normals are authored',
        'maximum_per_vertex_displacement_m':maximum_displacement,'actual_max_displacement_m':float(delta.max(initial=0)),
        'moved_vertices':int(np.count_nonzero(delta)),'source_plane_coordinates_changed':int(np.count_nonzero((v!=original)&anchors)),
        'minimum_area2_m2':float(area.min()),'minimum_reference_signed_altitude_m':float((signed/edge).min()),
        'negative_reference_faces':int((signed<=0).sum()),'topology_changed':False,
        'conditioning_altitude_predictor_m':target_altitude,'conditioning_predictor_met':not bool(remaining.any()),
        'unmet_predictor_constraints':int(remaining.sum()),'predictor_is_not_qualification':True,
        'required_next_checks':'Complete global intersections, protected source geometry, actual final native normals and renderer/physics',
        'iterations':records,'elapsed_s':time.monotonic()-started}
    atomic_json(output/'conditioning.json',report)
    if not invariant:raise RuntimeError('Source-plane native conditioning violated a hard bound')
    return v.astype(np.float32),f,report
