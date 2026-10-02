"""Independent triangle-surface numerics and explicit boundary ownership.

This is one component of Q04. It never promotes a mesh report into proof of
material continuity, payload loading, or an actual Isaac crossing recording.
Input faces are retained exactly; no repair or tolerance welding is performed.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ..io import atomic_json,sha256_file
from ..security import safe_path
from . import axis_parity
from . import surface_overlap


def _read_triangles(path,identity_root=None):
    path=Path(path)
    from .native_ground import has_native_ground,bind_native_ground
    if has_native_ground(path):
        bound=bind_native_ground(path,identity_root or path.parent)
        return bound['vertices'],bound['triangles'],bound['report']['input_files']
    if path.is_dir():
        inputs=[path/"vertices.npy",path/"triangles.npy"]
        vertices=np.load(inputs[0],mmap_mode="r",allow_pickle=False)
        faces=np.load(inputs[1],mmap_mode="r",allow_pickle=False)
    elif path.suffix==".npz":
        with np.load(path,allow_pickle=False) as data:vertices=data["vertices"];faces=data["faces"]
        inputs=[path]
    else:
        mesh=trimesh.load(path,force="mesh",process=False)
        if not isinstance(mesh,trimesh.Trimesh):raise ValueError("Triangle surface input is not a mesh")
        vertices,faces=np.asarray(mesh.vertices),np.asarray(mesh.faces);inputs=[path]
    return vertices,faces,[{"path":str(p.resolve()),"sha256":sha256_file(p)} for p in inputs]


def validate_continuous_surface(mesh_path,output,*,source_ir=None,mesh_to_source=None,
                                topology_scope="closed_initial_crop",declared_seam_planes=(),
                                face_ownership_path=None,quality_profile_path=None,numerics_only=False):
    """Measure raw, float32 and exact-position topology without changing faces.

    `mesh_path` accepts OBJ, NPZ(vertices/faces), or a native diagnostic directory
    containing vertices.npy/triangles.npy. A runtime payload may have open edges
    only on explicitly declared metric seam planes. Its composed join is still
    unqualified. Closed initial crops label their outer closure as crop scope,
    never as a successful internal tile join.
    """
    started=time.monotonic();output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError("Surface evidence is immutable; use a new output directory")
    if topology_scope not in {"closed_initial_crop","runtime_payload","composed_world"}:raise ValueError("Undeclared surface topology scope")
    output.mkdir(parents=True,exist_ok=True)
    gap=.002;profile_hash=None
    if quality_profile_path is not None:
        profile=json.loads(Path(quality_profile_path).read_text());profile=profile.get("profile",profile)
        gap=float(profile["thresholds"]["seam_gap_max_m"]);profile_hash=sha256_file(Path(quality_profile_path))
    if not 0<gap<=.002:raise ValueError("Surface boundary classification cannot loosen the2mm starting budget")
    from .native_ground import has_native_ground,bind_native_ground,evidence_root,workspace_root
    native_bound=bind_native_ground(mesh_path,evidence_root(output)) if has_native_ground(mesh_path) else None
    if native_bound is not None:
        v,f,inputs=native_bound['vertices'],native_bound['triangles'],native_bound['report']['input_files']
    else:
        v,f,inputs=_read_triangles(mesh_path,evidence_root(output))
    report={"schema_version":1,"kind":"IndependentContinuousSurfaceNumerics","status":"fail",
        "full_Q04_status":"not_run","scope":topology_scope,"input_files":inputs,
        "validator_sha256":sha256_file(Path(__file__)),"classifier_sha256":sha256_file(Path(axis_parity.__file__)),
        "overlap_classifier_sha256":sha256_file(Path(surface_overlap.__file__)),
        "quality_profile_sha256":profile_hash,"surface_join_position_budget_m":gap,
        "repair_performed":False,"exact_position_welding_for_incidence_only":True,
        "crossing_recordings":"not_run","normal_material_transition_qualification":"not_run",
        "nonmatching_coplanar_overlap_test":"not_run; exact duplicate triangles and edge ownership are checked exhaustively",
        "global_self_intersection_test":"not_run",
        "shared_edge_coplanar_overlap_test":"not_run",
        "limitations":["Numerical integrity does not establish rendered realism or any actual Isaac crossing",
            "Exact duplicate-face detection does not exhaustively detect coplanar overlap with different triangulations",
            "Source crop closures are explicit finite-domain surfaces, not internal world seams",
            "Runtime payload boundary declarations do not prove neighboring geometry is present or matches"]}
    if v.ndim!=2 or v.shape[1]!=3 or f.ndim!=2 or f.shape[1]!=3 or not len(v) or not len(f):
        raise ValueError("Expected nonempty XYZ vertices and original triangle triples")
    finite=np.isfinite(v).all(axis=1)
    invalid_faces=np.flatnonzero(np.any((f<0)|(f>=len(v)),axis=1)) if np.issubdtype(f.dtype,np.integer) else np.arange(len(f))
    report.update(vertices=len(v),triangles=len(f),nonfinite_vertex_count=int((~finite).sum()),invalid_index_face_count=len(invalid_faces))
    if not finite.all() or len(invalid_faces):
        np.savez_compressed(output/"invalid_input.npz",nonfinite_vertex_indices=np.flatnonzero(~finite),invalid_index_face_indices=invalid_faces)
        report["failure"]="Nonfinite positions or invalid original face indices; no invalid rows were discarded"
        report["files"]=[{"path":"invalid_input.npz","sha256":sha256_file(output/"invalid_input.npz")}]
        atomic_json(output/"continuous_surface.json",report);return report
    if native_bound is not None and not numerics_only:
        from .global_intersections import validate_global_intersections,file_record
        global_result=validate_global_intersections(workspace_root(),native_bound['vertices_path'],native_bound['triangles_path'],output/'global_intersections')
        report['global_self_intersection_test']={'status':global_result['status'],'complete':global_result['complete'],
            'intersection_pairs':global_result['intersection_pair_count'],'evidence':file_record(output/'global_intersections/global_intersections.json','complete_global_intersections')}
        if global_result['status']!='pass':
            report['failure']='Complete original-triangle global intersection check failed or remained incomplete; no closure/orientation pass inferred'
            report['files']=[]
            atomic_json(output/'continuous_surface.json',report);return report
    # Exact equality removes OBJ attribute seams from incidence calculations.
    # Original positions and original triangle IDs remain in all diagnostics.
    if numerics_only:
        vertices=np.asarray(v);faces=np.asarray(f)
        report["exact_position_welding_for_incidence_only"]=False
    else:
        vertices,inverse=np.unique(v,axis=0,return_inverse=True)
        vertices=np.asarray(vertices,dtype=np.float64)
        faces=inverse[f]
    nface=len(faces);nvertex=len(vertices);batch=131072
    area=np.lib.format.open_memmap(output/"face_area2.npy",mode="w+",dtype=np.float64,shape=(nface,))
    f32_area=np.lib.format.open_memmap(output/"face_float32_area2.npy",mode="w+",dtype=np.float64,shape=(nface,))
    bad64=[];bad32=[];flipped32=[];raw_repeated=[];normal_underflow=[];min_altitude=float("inf")
    native_position_error=0.;max_coordinate_ulp=0.;crop_faces={};crop_band_faces={}
    source_bounds=None;transform=None;source_record=None
    if source_ir is not None:
        source_ir=Path(source_ir);source_record=json.loads((source_ir/"world_ir.json").read_text())
        for entry in source_record["files"]:
            path=safe_path(source_ir,entry["path"],must_exist=True)
            if sha256_file(path)!=entry["sha256"]:raise ValueError("Surface validator source IR payload changed")
        with np.load(source_ir/"natural_occupancy.npz",allow_pickle=False) as data:
            minimum=data["min_xyz"].astype(float);shape=np.asarray(data["occupancy"].shape)[[2,0,1]]
        source_bounds=np.asarray([minimum,minimum+shape]);transform=np.asarray(mesh_to_source,float)
        if transform.shape!=(4,4) or not np.isfinite(transform).all():raise ValueError("Explicit finite source transform required")
        report["source_ir_sha256"]=sha256_file(source_ir/"world_ir.json")
        report["mesh_to_source"]=transform.tolist();report["source_crop_bounds_xyz"]=source_bounds.tolist()
        crop_faces={f"{axis}_{side}":[] for axis in "xyz" for side in ("min","max")}
        crop_band_faces={key:0 for key in crop_faces}
    volume_origin=vertices.mean(axis=0)
    for start in range(0,nface,batch):
        stop=min(start+batch,nface);tri=vertices[faces[start:stop]].astype(np.float64)
        u,vv=tri[:,1]-tri[:,0],tri[:,2]-tri[:,0];cross=np.cross(u,vv)
        area2=np.linalg.norm(cross,axis=1);area[start:stop]=area2
        native=tri.astype(np.float32)
        cross32=np.cross(native[:,1]-native[:,0],native[:,2]-native[:,0])
        area32=np.linalg.norm(cross32.astype(float),axis=1);f32_area[start:stop]=area32
        # Preserve the existing independent mesh_integrity numerical definition.
        bad64.extend((np.flatnonzero(area2<=1e-12)+start).tolist())
        bad32.extend((np.flatnonzero(area32<=1e-12)+start).tolist())
        flipped32.extend((np.flatnonzero(np.einsum("ij,ij->i",cross,cross32)<0)+start).tolist())
        normal_underflow.extend((np.flatnonzero(np.einsum("ij,ij->i",cross32,cross32)<=np.finfo(np.float32).tiny)+start).tolist())
        repeated=np.any(np.diff(np.sort(faces[start:stop],axis=1),axis=1)==0,axis=1)
        raw_repeated.extend((np.flatnonzero(repeated)+start).tolist())
        max_edge=np.maximum.reduce((np.linalg.norm(u,axis=1),np.linalg.norm(vv,axis=1),np.linalg.norm(vv-u,axis=1)))
        altitude=np.divide(area2,max_edge,out=np.zeros_like(area2),where=max_edge>0)
        min_altitude=min(min_altitude,float(altitude.min()))
        native_position_error=max(native_position_error,float(np.max(np.linalg.norm(native.astype(float)-tri,axis=2))))
        max_coordinate_ulp=max(max_coordinate_ulp,float(np.max(np.abs(np.spacing(native)))))
        if source_bounds is not None:
            source_tri=tri@transform[:3,:3].T+transform[:3,3]
            for axis,letter in enumerate("xyz"):
                for side,side_name in enumerate(("min","max")):
                    key=f"{letter}_{side_name}";difference=np.abs(source_tri[:,:,axis]-source_bounds[side,axis])
                    crop_faces[key].extend((np.flatnonzero(np.all(difference<=gap,axis=1))+start).tolist())
                    crop_band_faces[key]+=int(np.any(difference<=1.,axis=1).sum())
    area.flush();f32_area.flush()
    if numerics_only:
        np.savez_compressed(output/"surface_anomalies.npz",float64_degenerate_faces=np.asarray(bad64,np.int64),
            float32_degenerate_faces=np.asarray(bad32,np.int64),float32_orientation_reversal_faces=np.asarray(flipped32,np.int64),
            float32_squared_normal_underflow_faces=np.asarray(normal_underflow,np.int64),repeated_index_faces=np.asarray(raw_repeated,np.int64),
            **{"source_crop_plane_faces_"+key:np.asarray(value,np.int64) for key,value in crop_faces.items()})
        failures={"float64_degenerate_triangles":len(bad64),"native_float32_degenerate_triangles":len(bad32),
            "native_float32_orientation_reversals":len(flipped32),"native_float32_squared_normal_underflow":len(normal_underflow)}
        report.update(status="fail" if any(failures.values()) else "measurements_incomplete",failures=failures,
            numerical_triangle_area2_threshold_m2=1e-12,minimum_triangle_altitude_m=min_altitude,
            native_float32_max_position_roundtrip_error_m=native_position_error,native_float32_max_coordinate_ulp_m=max_coordinate_ulp,
            minimum_float64_area2_m2=float(area.min()),minimum_float32_area2_m2=float(f32_area.min()),
            edge_incidence="not_run_in_numerics_only_mode",face_ownership="not_run_in_numerics_only_mode",
            shell_orientation="not_run_in_numerics_only_mode",elapsed_seconds=round(time.monotonic()-started,3))
        report["files"]=[{"path":p.name,"bytes":p.stat().st_size,"sha256":sha256_file(p)} for p in (output/"surface_anomalies.npz",output/"face_area2.npy",output/"face_float32_area2.npy")]
        for item in inputs:
            if sha256_file(Path(item["path"]))!=item["sha256"]:raise ValueError("Surface input changed during validation")
        atomic_json(output/"continuous_surface.json",report);return report
    owner=np.zeros(nface,np.int32);owner_names=["single_authoritative_ground_mesh"]
    ownership_status="single_mesh_scope_only"
    if face_ownership_path is not None:
        with np.load(face_ownership_path,allow_pickle=False) as data:
            owner=data["owner_id"];owner_names=data["owner_names"].tolist()
            bound=data["input_sha256"].item()
        if len(inputs)!=1 or bound!=inputs[0]["sha256"]:raise ValueError("Face ownership refers to another mesh")
        if owner.shape!=(nface,) or not np.issubdtype(owner.dtype,np.integer) or owner.min()<0 or owner.max()>=len(owner_names):
            raise ValueError("Every original face requires exactly one declared valid owner")
        ownership_status="declared_per_face_owners_verified";report["face_ownership_sha256"]=sha256_file(Path(face_ownership_path))
    # Sorting disk-backed records bounds the largest transient arrays. The
    # original input face count is unchanged and every edge is counted.
    face_dtype=np.dtype([("a","<u8"),("b","<u8"),("c","<u8"),("face","<u8")])
    canonical=np.lib.format.open_memmap(output/".canonical_faces.npy",mode="w+",dtype=face_dtype,shape=(nface,))
    for start in range(0,nface,batch):
        stop=min(nface,start+batch);sorted_faces=np.sort(faces[start:stop],axis=1)
        for j,key in enumerate(("a","b","c")):canonical[key][start:stop]=sorted_faces[:,j]
        canonical["face"][start:stop]=np.arange(start,stop)
    canonical.sort(order=["a","b","c"])
    duplicate_mask=(canonical["a"][1:]==canonical["a"][:-1])&(canonical["b"][1:]==canonical["b"][:-1])&(canonical["c"][1:]==canonical["c"][:-1])
    duplicate_indices=np.flatnonzero(duplicate_mask)+1
    duplicate_faces=np.column_stack((canonical["face"][duplicate_indices-1],canonical["face"][duplicate_indices])).astype(np.int64)
    duplicate_cross_owners=int(np.count_nonzero(owner[duplicate_faces[:,0]]!=owner[duplicate_faces[:,1]])) if len(duplicate_faces) else 0
    del canonical;(output/".canonical_faces.npy").unlink()
    edge_dtype=np.dtype([("lo","<u8"),("hi","<u8"),("direction","i1"),("face","<u8")])
    edges=np.lib.format.open_memmap(output/".edges.npy",mode="w+",dtype=edge_dtype,shape=(nface*3,))
    for start in range(0,nface,batch):
        stop=min(nface,start+batch);a=faces[start:stop];pairs=np.stack((a[:,[0,1]],a[:,[1,2]],a[:,[2,0]]),axis=1).reshape(-1,2)
        selection=slice(start*3,stop*3)
        edges["lo"][selection]=pairs.min(axis=1);edges["hi"][selection]=pairs.max(axis=1)
        edges["direction"][selection]=np.sign(pairs[:,1]-pairs[:,0]);edges["face"][selection]=np.repeat(np.arange(start,stop),3)
    edges.sort(order=["lo","hi"])
    changes=np.r_[True,(edges["lo"][1:]!=edges["lo"][:-1])|(edges["hi"][1:]!=edges["hi"][:-1])]
    starts=np.flatnonzero(changes);counts=np.diff(np.r_[starts,len(edges)])
    winding=np.add.reduceat(edges["direction"],starts,dtype=np.int64)
    boundary=np.flatnonzero(counts==1);nonmanifold=np.flatnonzero(counts>2);inconsistent=np.flatnonzero((counts==2)&(winding!=0))
    boundary_edges=np.column_stack((edges["lo"][starts[boundary]],edges["hi"][starts[boundary]])).astype(np.int64)
    declared=np.zeros(len(boundary_edges),bool);seam_counts={}
    for seam in declared_seam_planes:
        if not seam.get("join_id") or not seam.get("neighbor_owner"):raise ValueError("Open payload seam needs join identity and neighbor owner")
        axis={"x":0,"y":1,"z":2}[seam["axis"]];coordinate=float(seam["coordinate_m"])
        matched=np.all(np.abs(vertices[boundary_edges][:,:,axis]-coordinate)<=gap,axis=1)
        declared|=matched;seam_counts[seam["join_id"]]=int(matched.sum())
    illegal_boundary=len(boundary_edges) if topology_scope!="runtime_payload" else int((~declared).sum())
    manifold_groups=np.flatnonzero(counts==2);first=starts[manifold_groups]
    row=edges["face"][first].astype(np.int64);col=edges["face"][first+1].astype(np.int64)
    overlap_records=[]
    for offset in range(0,len(first),batch):
        stop=min(offset+batch,len(first));indices=first[offset:stop]
        edge_vertices=np.column_stack((edges["lo"][indices],edges["hi"][indices]))
        overlap_records.extend(surface_overlap.shared_edge_overlap_candidates(vertices,faces,row[offset:stop],col[offset:stop],edge_vertices))
    adjacency=coo_matrix((np.ones(len(row),bool),(row,col)),shape=(nface,nface)).tocsr()
    component_count,face_components=connected_components(adjacency,directed=False)
    raw_edge_groups=np.unique(np.r_[boundary,nonmanifold,inconsistent])
    anomaly_edges=np.column_stack((edges["lo"][starts[raw_edge_groups]],edges["hi"][starts[raw_edge_groups]])).astype(np.int64)
    anomaly_counts=counts[raw_edge_groups];anomaly_winding=winding[raw_edge_groups]
    del adjacency,row,col,edges;(output/".edges.npy").unlink()
    best_area=np.zeros(component_count);best_face=np.full(component_count,nface,np.int64);volume=np.zeros(component_count)
    for start in range(0,nface,batch):
        stop=min(nface,start+batch);groups=face_components[start:stop]
        np.maximum.at(best_area,groups,area[start:stop])
        tri=vertices[faces[start:stop]]-volume_origin
        signed=np.einsum("ij,ij->i",tri[:,0],np.cross(tri[:,1],tri[:,2]))/6
        volume+=np.bincount(groups,weights=signed,minlength=component_count)
    for start in range(0,nface,batch):
        stop=min(nface,start+batch);groups=face_components[start:stop];ids=np.arange(start,stop)
        np.minimum.at(best_face,groups,np.where(area[start:stop]==best_area[groups],ids,nface))
    orientation_probe={"status":"not_run","reason":"Requires closed, nondegenerate edge-manifold geometry without a detected shared-edge overlap"}
    orientation_bad=np.empty(0,np.int64);probe_positions=np.empty((0,2,3));probe_inside=np.empty((0,2),bool)
    if topology_scope!="runtime_payload" and not (len(bad64) or len(bad32) or illegal_boundary or len(nonmanifold) or len(inconsistent) or len(duplicate_faces) or len(overlap_records)):
        selected=best_face[best_area>0];tri=vertices[faces[selected]];cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normals=cross/np.linalg.norm(cross,axis=1)[:,None]
        minimum_edge=np.minimum.reduce([np.linalg.norm(tri[:,i]-tri[:,j],axis=1) for i,j in ((0,1),(1,2),(2,0))])
        offsets=np.minimum(.01,minimum_edge*.02);centres=tri.mean(axis=1)
        probe_positions=np.stack((centres+normals*offsets[:,None],centres-normals*offsets[:,None]),axis=1)
        if native_bound is not None:
            from .native_segments import query_native_segments
            from .native_source import DIRECTIONS,classify_native_parity
            point_queries=probe_positions.reshape(-1,3)
            origins=np.repeat(point_queries,len(DIRECTIONS),axis=0)
            directions=np.tile(DIRECTIONS,(len(point_queries),1))
            extent=4*np.linalg.norm(vertices.max(axis=0)-vertices.min(axis=0))+1
            hits,native_queries=query_native_segments(workspace_root(),native_bound['vertices_path'],native_bound['triangles_path'],
                np.stack([origins,origins+directions*extent],axis=1),output/'native_orientation_queries')
            classifications,classification_raw=classify_native_parity(hits,point_queries)
            if not native_queries['complete'] or classification_raw['unresolved'].any():
                raise ValueError('Closed-shell native orientation queries are incomplete or ambiguous')
            from .global_intersections import file_record
            report['native_geometry_identity']=file_record(native_bound['identity_report'],'original_OBJ_native_array_equivalence')
            report['native_orientation_query_evidence']=file_record(output/'native_orientation_queries/segment_queries.json','actual_shell_orientation_queries')
        else:
            welded_mesh=trimesh.Trimesh(vertices,faces,process=False)
            classifications,classification_raw=axis_parity.contains_axis_consensus(welded_mesh,probe_positions.reshape(-1,3))
        probe_inside=classifications.reshape(-1,2);orientation_bad=np.flatnonzero(probe_inside[:,0]|~probe_inside[:,1])
        orientation_probe={"status":"pass" if not len(orientation_bad) else "fail","shell_samples":len(selected),
            "outward_side_must_be_air":True,"inward_side_must_be_solid":True,"mismatched_shell_samples":len(orientation_bad),
            "method":"Largest-area triangle per edge-connected shell; both-side independent six-axis parity. Negative-volume inner void shells are allowed when oriented correctly."}
        if native_bound is None:del welded_mesh
    np.savez_compressed(output/"surface_anomalies.npz",float64_degenerate_faces=np.asarray(bad64,np.int64),
        float32_degenerate_faces=np.asarray(bad32,np.int64),float32_orientation_reversal_faces=np.asarray(flipped32,np.int64),
        float32_squared_normal_underflow_faces=np.asarray(normal_underflow,np.int64),repeated_position_faces=np.asarray(raw_repeated,np.int64),
        duplicate_face_pairs=duplicate_faces,boundary_edge_vertex_ids=boundary_edges,boundary_edges_declared=declared,
        incidence_anomaly_edges=anomaly_edges,incidence_anomaly_count=anomaly_counts,incidence_anomaly_winding=anomaly_winding,
        coplanar_overlap_face_pairs=np.asarray([item["face_ids"] for item in overlap_records],np.int64).reshape(-1,2),
        shell_representative_faces=best_face,shell_signed_volumes=volume,orientation_probe_positions=probe_positions,
        orientation_probe_inside=probe_inside,orientation_bad_sample_indices=orientation_bad,
        **{"source_crop_plane_faces_"+key:np.asarray(value,np.int64) for key,value in crop_faces.items()})
    failures={"float64_degenerate_triangles":len(bad64),"native_float32_degenerate_triangles":len(bad32),
        "native_float32_orientation_reversals":len(flipped32),"native_float32_squared_normal_underflow":len(normal_underflow),
        "duplicate_triangles":len(duplicate_faces),"nonmanifold_edges":len(nonmanifold),"inconsistent_winding_edges":len(inconsistent),
        "unexplained_boundary_edges":illegal_boundary,"shell_orientation_failures":len(orientation_bad)}
    failures["coplanar_shared_edge_overlaps"]=len(overlap_records)
    if topology_scope!="runtime_payload" and float(volume.sum())<=0:failures["nonpositive_total_signed_volume"]=1
    report.update(status="pass" if not any(failures.values()) else "fail",failures=failures,
        unique_exact_positions=nvertex,attribute_seam_duplicate_positions=len(v)-nvertex,
        numerical_triangle_area2_threshold_m2=1e-12,minimum_triangle_altitude_m=min_altitude,
        native_float32_max_position_roundtrip_error_m=native_position_error,native_float32_max_coordinate_ulp_m=max_coordinate_ulp,
        minimum_float64_area2_m2=float(area.min()),minimum_float32_area2_m2=float(f32_area.min()),
        edge_count=len(starts),boundary_edges=len(boundary_edges),declared_payload_seam_edge_counts=seam_counts,
        edge_connected_shells=component_count,total_signed_volume_m3=float(volume.sum()),shell_orientation=orientation_probe,
        shared_edge_coplanar_overlap_test={"status":"fail" if overlap_records else "pass_in_tested_scope",
            "scope":"Every edge with exactly two incident triangles; positive-area coplanar overlap proved by exact rational clipping of original binary coordinates",
            "tested_shared_edges":len(manifold_groups),"overlap_count":len(overlap_records),"overlaps":overlap_records,
            "global_nonadjacent_or_noncoplanar_intersections":"not_run"},
        face_ownership={"status":ownership_status,"owners":owner_names,"duplicate_faces_with_different_owners":duplicate_cross_owners,
                        "exact_geometric_duplicate_faces":len(duplicate_faces)},
        source_crop_boundary={"status":"labelled" if source_bounds is not None else "not_supplied",
            "plane_face_counts":{key:len(value) for key,value in crop_faces.items()},"one_metre_boundary_band_face_counts":crop_band_faces,
            "interpretation":"Finite source-crop closure candidates, including bottom/outer sides; not an internal tile join or proof of continuation beyond crop"},
        elapsed_seconds=round(time.monotonic()-started,3))
    report["files"]=[{"path":p.name,"bytes":p.stat().st_size,"sha256":sha256_file(p)} for p in (output/"surface_anomalies.npz",output/"face_area2.npy",output/"face_float32_area2.npy")]
    for item in inputs:
        if sha256_file(Path(item["path"]))!=item["sha256"]:raise ValueError("Surface input changed during validation")
    atomic_json(output/"continuous_surface.json",report)
    return report
