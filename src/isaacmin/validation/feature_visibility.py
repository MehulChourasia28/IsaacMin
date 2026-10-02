"""Observed cave features from actual native pixels and independent mesh rays."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..io import atomic_json, sha256_file
from ..security import safe_path
from .capture_plan import load_ground
from .sensors import camera_rays
from ..volumes.source_evidence import verify_source_topology,recheck_source_topology

METHOD = "source_feature_visibility_geometry_v1"


def bounded_first_hits(mesh, origins, directions, maximum_distance):
    """Exact independent Möller–Trumbore segment hits with bounded candidates.

    A captured depth bounds the visibility witness segment; any earlier ground
    occluder remains in the query. Queries never discard a mismatched hit.
    """
    origins, directions = np.asarray(origins,float), np.asarray(directions,float)
    maximum_distance=np.asarray(maximum_distance,float)
    if origins.shape!=directions.shape or origins.ndim!=2 or origins.shape[1]!=3 or maximum_distance.shape!=(len(origins),):
        raise ValueError("Segment query arrays have inconsistent shapes")
    if not np.isfinite(origins).all() or not np.isfinite(directions).all() or not np.isfinite(maximum_distance).all() or np.any(maximum_distance<0):
        raise ValueError("Nonfinite or negative visibility segment")
    if not np.allclose(np.linalg.norm(directions,axis=1),1.,atol=1e-8):
        raise ValueError("Visibility rays require normalized directions")
    distances=np.full(len(origins),np.nan);faces=np.full(len(origins),-1,np.int64)
    if hasattr(mesh,"queries"):
        locations,_,ids=mesh.queries.first_hits(origins,directions,maximum_distance)
        valid=ids>=0;distances[valid]=np.linalg.norm(locations[valid]-origins[valid],axis=1)
        return distances,ids
    tree=mesh.triangles_tree
    for ray,(origin,direction,length) in enumerate(zip(origins,directions,maximum_distance)):
        end=origin+direction*length
        bounds=np.r_[np.minimum(origin,end)-1e-7,np.maximum(origin,end)+1e-7]
        # Materialize at most65k triangle candidates at a time, including when a
        # dense diagonal segment intersects a large BVH bounding box.
        iterator=iter(tree.intersection(bounds))
        best=float("inf")
        while True:
            import itertools
            candidates=np.fromiter(itertools.islice(iterator,65536),dtype=np.int64)
            if not len(candidates):break
            triangles=mesh.vertices[mesh.faces[candidates]]
            edge1=triangles[:,1]-triangles[:,0];edge2=triangles[:,2]-triangles[:,0]
            cross=np.cross(np.broadcast_to(direction,edge2.shape),edge2)
            determinant=np.einsum("ij,ij->i",edge1,cross)
            valid=np.abs(determinant)>1e-12
            inverse=np.zeros_like(determinant);inverse[valid]=1/determinant[valid]
            offset=origin-triangles[:,0]
            u=np.einsum("ij,ij->i",offset,cross)*inverse
            q=np.cross(offset,edge1)
            v=(q@direction)*inverse
            distance=np.einsum("ij,ij->i",edge2,q)*inverse
            valid &= (u>=-1e-9)&(v>=-1e-9)&(u+v<=1+1e-9)&(distance>=0)&(distance<=length+1e-7)
            if valid.any():
                selected=np.flatnonzero(valid)[np.argmin(distance[valid])]
                if distance[selected]<best:
                    best=float(distance[selected]);faces[ray]=candidates[selected]
        if faces[ray]>=0:distances[ray]=best
    return distances,faces


def _source_labels(points, minimum, labels, validity, air):
    indices=np.floor(points-minimum).astype(np.int64)[:,[1,2,0]]
    inside=np.all(indices>=0,axis=1)&np.all(indices<np.asarray(labels.shape),axis=1)
    result=np.zeros(len(points),np.int32)
    y,z,x=indices[inside].T
    allowed=validity[y,z,x]&air[y,z,x]
    result[np.flatnonzero(inside)[allowed]]=labels[y[allowed],z[allowed],x[allowed]]
    return result


def collect_feature_visibility(mesh_path, ir_dir, capture_results, coordinate_frame, output_dir, *,
                               grid_stride=16, maximum_range_m=12., minimum_witness_pixels=8):
    """Measure actual visible source cavities, portals, and roof undersides.

    Sparse witnessed presence is distinct from readability/appearance approval.
    Planned pose tags are not used to identify any feature.
    """
    mesh_path,ir_dir,output_dir=Path(mesh_path),Path(ir_dir),Path(output_dir)
    if output_dir.exists():raise ValueError("Feature evidence is immutable; choose a new output directory")
    if not isinstance(grid_stride,int) or grid_stride<1 or minimum_witness_pixels<1 or not 0<maximum_range_m<=30:
        raise ValueError("Invalid bounded feature visibility sampling")
    graph,source_evidence=verify_source_topology(ir_dir)
    inputs={}
    def bind(path):
        path=Path(path).resolve();item={"path":str(path),"sha256":sha256_file(path)};inputs[str(path)]=item;return item
    for entry in source_evidence["payloads"]:bind(entry["path"])
    graph_path=ir_dir/"topology/source_topology_graph.json"
    bind(Path(__file__).parents[1]/"volumes/source_evidence.py")
    with np.load(ir_dir/"natural_occupancy.npz",allow_pickle=False) as data:
        minimum=data["min_xyz"].astype(float);validity=data["validity"];air=data["occupancy"]==0
    label_path=ir_dir/"topology/source_topology_labels.npz"
    with np.load(label_path,allow_pickle=False) as data:labels=data["cavity_labels"]
    for path in (mesh_path,ir_dir/"world_ir.json",graph_path,label_path,Path(__file__),Path(__file__).with_name("sensors.py")):
        bind(path)
    transform=np.asarray(coordinate_frame.record()["world_to_source"],float)
    def source(points):return np.asarray(points)@transform[:3,:3].T+transform[:3,3]
    features_by_label={feature["label"]:feature for feature in graph["features"]}
    output_dir.mkdir(parents=True)
    mesh=load_ground(mesh_path,output_dir/"native_ground_queries")
    jobs=[]
    rows=[]
    from ..assets.usd_provenance import captured_closure_hash,verify_scene_closure,validated_native_renderer
    from ..assets import usd_provenance
    bind(Path(usd_provenance.__file__))
    for capture_index,capture_path in enumerate(capture_results):
        capture_path=Path(capture_path);capture=json.loads(capture_path.read_text());capture_hash=bind(capture_path)["sha256"]
        validated_native_renderer(capture)
        closure_hash=captured_closure_hash(capture)
        scene=Path(capture["scene"])
        if sha256_file(scene)!=capture["scene_sha256"]:
            raise ValueError("Feature observations require the actual recorded native renderer and unchanged scene")
        closure=verify_scene_closure(scene,closure_hash)
        for item in [closure["manifest"],*closure["files"]]:
            inputs[item["path"]]=item
        near=float(capture["near_m"])
        if near<=0 or not closure_hash:raise ValueError("Native capture clipping and scene closure must be recorded")
        for frame in capture["frames"]:
            # Visibility is valid in static or moving captures. Camera kind and
            # surface_scope never supply the observed feature identity.
            bound={}
            for key in ("rgb","depth","instance_segmentation"):
                path=safe_path(capture_path.parent,frame[key],must_exist=True);bound[key]=bind(path)
                if bound[key]["sha256"]!=frame[key+"_sha256"]:raise ValueError("Native visibility sensor bytes changed")
            if frame["depth_semantics"] not in ("axial_metres","distance_to_image_plane","distance_to_image_plane_m"):
                raise ValueError("Unsupported actual depth convention")
            depth=np.load(bound["depth"]["path"],allow_pickle=False).squeeze()
            segmentation=np.load(bound["instance_segmentation"]["path"],allow_pickle=False).squeeze()
            if depth.ndim!=2 or depth.shape!=segmentation.shape:raise ValueError("Native sensor arrays are not aligned")
            ground_ids=[int(k) for k,v in frame["instance_id_to_prim_path"].items() if "Terrain_FinalGround" in str(v)]
            yy,xx=np.mgrid[grid_stride//2:depth.shape[0]:grid_stride,grid_stride//2:depth.shape[1]:grid_stride]
            pixels=np.column_stack((xx.ravel(),yy.ravel()))
            intrinsics=frame.get("intrinsics") or {k:frame[k+"_pixels"] for k in ("fx","fy","cx","cy")}
            origins,directions,cosines=camera_rays(pixels+.5,intrinsics,frame["camera_world_matrix_row_vectors"])
            native_depth=depth[pixels[:,1],pixels[:,0]]
            ranges=native_depth/cosines
            select=np.isin(segmentation[pixels[:,1],pixels[:,0]],ground_ids)&np.isfinite(ranges)&(ranges>near/cosines)&(ranges<=maximum_range_m)
            pixels,origins,directions,cosines,native_depth,ranges=(a[select] for a in (pixels,origins,directions,cosines,native_depth,ranges))
            begin=near/cosines
            jobs.append((capture_index,frame,capture_hash,closure_hash,bound,pixels,origins,directions,cosines,native_depth,ranges,begin))
    native_queries=getattr(mesh,'queries',None);offsets=np.r_[0,np.cumsum([len(job[5]) for job in jobs])]
    if native_queries is not None and offsets[-1]:
        query_origins=np.concatenate([job[6]+job[7]*job[11][:,None] for job in jobs]);query_directions=np.concatenate([job[7] for job in jobs]);query_lengths=np.concatenate([job[10]-job[11]+.02 for job in jobs])
        all_distances,all_triangles=bounded_first_hits(mesh,query_origins,query_directions,query_lengths)
    for job_index,(capture_index,frame,capture_hash,closure_hash,bound,pixels,origins,directions,cosines,native_depth,ranges,begin) in enumerate(jobs):
            if native_queries is None:distances,triangles=bounded_first_hits(mesh,origins+directions*begin[:,None],directions,ranges-begin+.02)
            elif not len(pixels):distances,triangles=np.empty(0),np.empty(0,np.int64)
            else:
                selected=slice(offsets[job_index],offsets[job_index+1]);distances=all_distances[selected].copy();triangles=all_triangles[selected]
            distances+=begin
            agrees=np.isfinite(distances)&(np.abs(distances-ranges)<=.02)
            hits=np.full((len(pixels),3),np.nan);normals=np.zeros_like(hits);cavity=np.zeros(len(pixels),np.int32)
            hits[agrees]=origins[agrees]+directions[agrees]*distances[agrees,None]
            normals[agrees]=mesh.face_normals[triangles[agrees]]
            cavity[agrees]=_source_labels(source(hits[agrees]+normals[agrees]*.05),minimum,labels,validity,air)
            observed=[]
            for label in sorted(set(cavity.tolist())-{0}):
                selected=(cavity==label)&agrees
                if selected.sum()<minimum_witness_pixels:continue
                feature=features_by_label[label]
                scopes=["cave_interior"]
                if np.count_nonzero(selected&(normals[:,2]<-.5))>=minimum_witness_pixels:scopes.append("cave_underside")
                for scope in scopes:
                    witnesses=selected if scope=="cave_interior" else selected&(normals[:,2]<-.5)
                    observed.append({"feature_id":scope,"source_feature_id":feature["id"],"status":"observed","method":METHOD,
                        "witness_pixels":int(witnesses.sum()),"witness_sample_indices":np.flatnonzero(witnesses).tolist(),
                        "range_min_max_m":[float(distances[witnesses].min()),float(distances[witnesses].max())],
                        "source_crop_truncated":feature["touches_crop_boundary"],
                        "evidence":"Native ground pixels match independent first triangle hit; outward air-side point lies in exact source covered-air component"})
            source_origins=source(origins);source_directions=directions@transform[:3,:3].T
            portal_ids=[];portal_samples=[];portal_distance=[]
            for portal in graph["portals"]:
                witnessed=np.zeros(len(pixels),bool)
                for face in portal["faces"]:
                    axis={"x":0,"y":1,"z":2}[face["normal_axis"]];centre=np.asarray(face["center_xyz"],float)
                    denominator=source_directions[:,axis]
                    eligible=agrees&(np.abs(denominator)>.1)
                    t=np.full(len(pixels),np.nan);t[eligible]=(centre[axis]-source_origins[eligible,axis])/denominator[eligible]
                    intersection=source_origins+source_directions*t[:,None]
                    other=[i for i in range(3) if i!=axis]
                    inside=eligible&(t>begin+.05)&(t<distances-.05)&np.all(np.abs(intersection[:,other]-centre[other])<.45,axis=1)
                    for sample in np.flatnonzero(inside):
                        portal_ids.append(portal["label"]);portal_samples.append(sample);portal_distance.append(t[sample])
                    witnessed|=inside
                if witnessed.sum()>=minimum_witness_pixels:
                    observed.append({"feature_id":"cave_portal","source_feature_id":portal["id"],"status":"observed","method":METHOD,
                        "witness_pixels":int(witnessed.sum()),"witness_sample_indices":np.flatnonzero(witnessed).tolist(),
                        "evidence":"Actual native ground rays pass through the source aperture interior before a matching independent final-ground hit; earlier ground occluders are included"})
            name=f"capture_{capture_index:03d}_frame_{int(frame['frame']):05d}.npz"
            np.savez_compressed(output_dir/name,pixels_uv=pixels,native_axial_depth=native_depth,independent_range=distances,
                native_range=ranges,depth_agrees=agrees,triangle_index=triangles,hit_world_xyz=hits,hit_world_normal=normals,
                source_cavity_label=cavity,portal_label=np.asarray(portal_ids),portal_sample_index=np.asarray(portal_samples),portal_range=np.asarray(portal_distance))
            bind(output_dir/name)
            rows.append({"capture_result_sha256":capture_hash,"frame":frame["frame"],"rgb_sha256":bound["rgb"]["sha256"],
                "scene_dependencies_sha256":closure_hash,"features":observed,"raw_samples":name,
                "sampled_native_ground_pixels":len(pixels),"independent_matching_depth_pixels":int(agrees.sum()),
                "rejected_depth_witnesses":int((~agrees).sum()),"planned_tags_used":False})
    if native_queries is not None:
        query_evidence=mesh.receipt();bind(query_evidence["path"])
    for item in inputs.values():
        if sha256_file(Path(item["path"]))!=item["sha256"]:raise ValueError("Feature evidence input changed during collection")
    recheck_source_topology(ir_dir,source_evidence)
    report={"schema_version":1,"status":"measured","method":METHOD,"input_files":list(inputs.values()),"frames":rows,
        "source_payload_evidence":source_evidence,
        "coordinate_frame":coordinate_frame.record(),"thresholds":{"grid_stride_pixels":grid_stride,"maximum_range_m":maximum_range_m,
            "minimum_witness_pixels":minimum_witness_pixels,"native_range_agreement_m":.02,"air_side_offset_m":.05,"portal_face_inset_source_blocks":.05},
        "qualification":"Observed source-feature presence only; readability, target appearance, complete feature coverage and navigation remain separate",
        "limits":["Sparse pixel sampling can miss visible small features; absence is an evidence gap", "Source crop-truncated components keep local identity only", "No pose intent or planned feature label supplies an observed feature"]}
    if native_queries is not None:report["native_query_evidence"]=query_evidence
    atomic_json(output_dir/"feature_visibility.json",report)
    return report
