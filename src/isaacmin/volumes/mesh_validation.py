"""Independent mesh ray checks against protected actual-source volume facts.

Uses trimesh/Rtree triangle queries, not the OpenVDB scalar field or its sign.
This verifies specific source occupancy and portal segments. Complete target
connectivity, appearance and Isaac ground contact remain separate gates.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import rtree
import trimesh
from scipy import ndimage

from isaacmin.source.nbt import SourceError
from isaacmin.source.snapshot import sha256,write_json
from isaacmin.validation import axis_parity
from .source_evidence import verify_source_topology,recheck_source_topology


def _load_mesh(path: Path, mesh_to_source) -> trimesh.Trimesh:
    path=Path(path)
    if path.suffix==".npz":
        with np.load(path,allow_pickle=False) as data:
            mesh=trimesh.Trimesh(vertices=np.asarray(data["vertices"],dtype=float),faces=np.asarray(data["faces"],dtype=np.int64),process=False)
            if mesh_to_source is None and "mesh_to_source" in data:
                mesh_to_source=data["mesh_to_source"]
    else:
        loaded=trimesh.load(path,force="mesh",process=False)
        if not isinstance(loaded,trimesh.Trimesh):
            raise SourceError("Validator input must be a triangle mesh")
        mesh=loaded
    if mesh_to_source is None:
        raise SourceError("An explicit mesh-to-source4x4 transform is required")
    transform=np.asarray(mesh_to_source,dtype=float)
    if transform.shape!=(4,4) or not np.isfinite(transform).all() or not np.allclose(transform[3],[0,0,0,1]):
        raise SourceError("Invalid mesh-to-source transform")
    if not len(mesh.vertices) or not len(mesh.faces) or not np.isfinite(mesh.vertices).all():
        raise SourceError("Mesh contains missing or non-finite geometry")
    # OBJ may split the same geometric vertex at normal/material/UV seams.
    # Exact equality welding removes those attribute seams without rounding
    # coordinates, moving vertices, or sealing a physical crack.
    vertices,inverse=np.unique(mesh.vertices,axis=0,return_inverse=True)
    mesh=trimesh.Trimesh(vertices=vertices,faces=inverse[mesh.faces],process=False)
    mesh.apply_transform(transform)
    return mesh


def _choose(mask, count, rng):
    indices=np.flatnonzero(mask)
    if not len(indices):
        return np.empty((0,3),dtype=int)
    chosen=rng.choice(indices,size=min(count,len(indices)),replace=False)
    return np.column_stack(np.unravel_index(chosen,mask.shape))


def validate_native_occupancy_fixture(mesh_path: Path, coordinate_path: Path, output: Path, spacing_m: float=1.0) -> dict:
    """Enumerate all occupied/air centres in a bounded native worker fixture.

    Both the OBJ and worker input are already in target-local X,-Z,Y axes.
    This must not apply the additional transform used for Minecraft WorldIR.
    """
    mesh_path,coordinate_path,output=Path(mesh_path),Path(coordinate_path),Path(output)
    if not 0.02<=spacing_m<=1:
        raise SourceError("Fixture voxel spacing outside supported bounds")
    occupied=np.fromfile(coordinate_path,dtype="<i4")
    if not len(occupied) or len(occupied)%3:
        raise SourceError("Native fixture coordinates must be nonempty int32 XYZ triples")
    occupied=occupied.reshape(-1,3)
    minimum,maximum=occupied.min(axis=0)-1,occupied.max(axis=0)+1
    shape=maximum-minimum+1
    if int(np.prod(shape,dtype=np.int64))>2_000_000:
        raise SourceError("Native fixture exhaustive validation exceeds bounded2M cell budget; use region sampler")
    indices=np.indices(shape).reshape(3,-1).T+minimum
    positions=indices*spacing_m
    expected_grid=np.zeros(shape,dtype=bool)
    shifted=occupied-minimum
    expected_grid[shifted[:,0],shifted[:,1],shifted[:,2]]=True
    expected=expected_grid.ravel()
    mesh=_load_mesh(mesh_path,np.eye(4))
    actual,ray_evidence=axis_parity.contains_axis_consensus(mesh,positions)
    failures=np.flatnonzero(actual!=expected)
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/"samples.npz",target_local_xyz=positions,expected_inside=expected,mesh_inside=actual,**ray_evidence)
    report={"schema_version":1,"kind":"IndependentNativeOccupancyFixtureValidation",
            "status":"pass" if not len(failures) and mesh.is_volume else "fail",
            "evidence_scope":"actual native worker compatibility fixture; not user-map geometry or appearance",
            "coordinate_frame":"identity target-local XYZ matching worker occupied_xyz_i32.bin; input integers scaled by spacing_m",
            "spacing_m":spacing_m,"mesh_sha256":sha256(mesh_path),"occupancy_file_sha256":sha256(coordinate_path),
            "sample_count":len(positions),"mismatch_count":len(failures),"mismatches":positions[failures].tolist(),
            "watertight":bool(mesh.is_watertight),"winding_consistent":bool(mesh.is_winding_consistent),
            "positive_closed_volume":bool(mesh.is_volume),"signed_volume_m3":float(mesh.volume),
            "toolchain":{"trimesh":trimesh.__version__,"rtree":rtree.__version__},
            "validator_sha256":sha256(Path(__file__)),"classifier_sha256":sha256(Path(axis_parity.__file__)),
            "axis_ambiguous_diagonal_fallbacks":len(ray_evidence["axis_fallback_indices"]),
            "samples_file":"samples.npz","samples_sha256":sha256(output/"samples.npz")}
    write_json(output/"comparison.json",report)
    return report


def validate_source_connectivity(ir_dir: Path, mesh_path: Path, output: Path, mesh_to_source=None,
                                 exterior_deformation_envelope_m: float=1.5, *, support_manifest_path: Path | None=None) -> dict:
    """Revoxelize every source column from independent mesh intersections.

    One vertical ray per x,z cell produces all solid intervals, so validating
    millions of source centres needs only thousands of accelerated ray queries.
    Both original and predicted air graphs use the same explicit known-domain
    and face connectivity. Sub-voxel aperture checks remain complementary.
    """
    if not 0<=exterior_deformation_envelope_m<=2:
        raise SourceError("Exterior deformation envelope outside frozen implementation bounds")
    ir_dir,mesh_path,output=Path(ir_dir),Path(mesh_path),Path(output)
    started=time.monotonic()
    original_graph,source_evidence=verify_source_topology(ir_dir)
    from ..validation.native_ground import has_native_ground
    if has_native_ground(mesh_path):
        from .native_source_adapter import StreamedColumnMesh
        mesh=StreamedColumnMesh(mesh_path,ir_dir,output,mesh_to_source)
    else:
        mesh=_load_mesh(mesh_path,mesh_to_source)
    with np.load(ir_dir/"natural_occupancy.npz",allow_pickle=False) as data:
        occupancy=data["occupancy"]
        validity=data["validity"]
        xmin,ymin,zmin=map(float,data["min_xyz"])
    with np.load(ir_dir/"terrain_surface.npz",allow_pickle=False) as data:
        height=data["height"]
    with np.load(ir_dir/"topology/source_topology_labels.npz",allow_pickle=False) as data:
        covered=data["covered_air"]
        cavity_labels=data["cavity_labels"]
        portal_labels=data["portal_labels"]
    ny,nz,nx=occupancy.shape
    if ny*nz*nx>100_000_000:
        raise SourceError("Connectivity validation requires a contextual patch below100M voxels")
    z,x=np.indices((nz,nx))
    origins=np.column_stack([x.ravel()+xmin+0.5,np.full(nx*nz,float(mesh.bounds[0,1])-2),z.ravel()+zmin+0.5])
    directions=np.tile([0.,1.,0.],(nx*nz,1))
    ycentres=np.arange(ny)+ymin+0.5
    predicted=np.zeros(occupancy.shape,dtype=bool)
    raw_hit_rays,raw_hit_y=[],[]
    odd_columns=[]
    vertical_barriers=[]
    roof_intervals=[]
    roof_sign_failures=[]
    duplicate_hits=0
    source_air=(occupancy==0)&validity
    natural=(occupancy==1)&validity
    from .support_validation import structural_support_for_validation
    structural,support_context=structural_support_for_validation(ir_dir,occupancy,support_manifest_path)
    supporting=natural|structural
    roof_is_structural=[]
    for start in range(0,len(origins),256):
        hits,rays,_=mesh.ray.intersects_location(origins[start:start+256],directions[start:start+256],multiple_hits=True)
        hits=np.asarray(hits).reshape((-1,3))
        order=np.lexsort((hits[:,1],rays))
        hits,rays=hits[order],rays[order]
        for ray in range(min(256,len(origins)-start)):
            values=hits[rays==ray,1]
            keep=np.r_[True,np.diff(values)>1e-7] if len(values) else np.zeros(0,dtype=bool)
            duplicate_hits+=int((~keep).sum())
            values=values[keep]
            global_ray=int(start+ray)
            rz,rx=divmod(global_ray,nx)
            raw_hit_rays.extend([global_ray]*len(values))
            raw_hit_y.extend(values.tolist())
            if len(values)%2:
                odd_columns.append({"source_xz":[rx+xmin+0.5,rz+zmin+0.5],"crossings":len(values)})
            predicted[:,rz,rx]=np.searchsorted(values,ycentres,side="right")%2==1
            roof_indices=np.flatnonzero(supporting[1:,rz,rx]&covered[:-1,rz,rx])+1
            non_natural=np.r_[np.flatnonzero(~supporting[:,rz,rx]),ny]
            for roof_index in roof_indices:
                end=int(non_natural[np.searchsorted(non_natural,roof_index,side="right")])
                source_bottom,source_top=ymin+roof_index,ymin+end
                interval=int(np.searchsorted(values,ycentres[roof_index],side="right"))
                if interval%2 and 0<interval<len(values):
                    target_bottom,target_top=float(values[interval-1]),float(values[interval])
                else:
                    target_bottom=target_top=0.0
                if target_top<=target_bottom:
                    roof_sign_failures.append({"source_xyz":[rx+xmin+0.5,float(ycentres[roof_index]),rz+zmin+0.5]})
                roof_intervals.append([rx+xmin+0.5,rz+zmin+0.5,source_bottom,source_top,target_bottom,target_top,source_top-source_bottom,target_top-target_bottom])
                roof_is_structural.append(bool(structural[roof_index,rz,rx]))
            # A mesh crossing between adjacent protected source-air centres is
            # a barrier even if a thin new cap misses both sample centres.
            for value in values:
                low=int(np.floor(value-ymin-0.5))
                if 0<=low<ny-1 and source_air[low,rz,rx] and source_air[low+1,rz,rx] and (covered[low,rz,rx] or covered[low+1,rz,rx]):
                    if value-ycentres[low]>1e-7 and ycentres[low+1]-value>1e-7:
                        vertical_barriers.append({"source_xyz":[rx+xmin+0.5,float(value),rz+zmin+0.5],"source_air_y_indices":[low,low+1]})
    known=validity&((occupancy==0)|supporting)
    structure=ndimage.generate_binary_structure(3,1)
    protected=known&((ycentres[:,None,None]<=height[None]-exterior_deformation_envelope_m)
                    |ndimage.binary_dilation(covered,structure=structure)
                    |ndimage.binary_dilation(portal_labels>0,structure=structure)|structural)
    mismatches=protected&(predicted!=supporting)
    source_labels,source_count=ndimage.label(source_air,structure)
    target_air=known&~predicted
    target_labels,target_count=ndimage.label(target_air,structure)
    anchors=source_air&covered
    anchors[-1]|=source_air[-1]
    encoded=source_labels[anchors].astype(np.int64)*(target_count+1)+target_labels[anchors]
    pairs=np.unique(encoded)
    source_to_target={}
    target_to_source={}
    lost_components=[]
    for pair in pairs:
        source_label,target_label=divmod(int(pair),target_count+1)
        if source_label==0:continue
        source_to_target.setdefault(source_label,set())
        if target_label:
            source_to_target[source_label].add(target_label)
            target_to_source.setdefault(target_label,set()).add(source_label)
    for source_label,targets in source_to_target.items():
        if not targets:lost_components.append(source_label)
    splits=[{"source_air_component":k,"target_components":sorted(v)} for k,v in source_to_target.items() if len(v)>1]
    merges=[{"target_air_component":k,"source_components":sorted(v)} for k,v in target_to_source.items() if len(v)>1]
    target_sky=set(np.unique(target_labels[-1][target_air[-1]]).tolist())-{0}
    feature_comparison=[]
    changed_sky=[]
    for feature in original_graph["features"]:
        target_ids=set(np.unique(target_labels[cavity_labels==feature["label"]]).tolist())-{0}
        has_sky=bool(target_ids&target_sky)
        changed=has_sky!=feature["connected_to_observed_sky"]
        record={"source_feature_id":feature["id"],"source_sky_connected":feature["connected_to_observed_sky"],
                "target_sky_connected":has_sky,"target_air_components":sorted(target_ids),"crop_truncated":feature["touches_crop_boundary"],"same_sky_connection":not changed}
        feature_comparison.append(record)
        if changed:changed_sky.append(record)
    output.mkdir(parents=True,exist_ok=True)
    roofs=np.asarray(roof_intervals,dtype=np.float64).reshape((-1,8))
    np.savez_compressed(output/"column_ray_reconstruction.npz",predicted_supporting_solid=predicted,
                        protected_comparison=protected,occupancy_mismatch=mismatches,source_air_labels=source_labels,target_air_labels=target_labels,
                        hit_ray_index=np.asarray(raw_hit_rays,dtype=np.int32),hit_source_y=np.asarray(raw_hit_y,dtype=np.float64),
                        min_xyz=np.asarray([xmin,ymin,zmin]),source_y_centres=ycentres,roof_vertical_intervals=roofs,
                        source_structural_support=structural,roof_base_is_source_structural=np.asarray(roof_is_structural,dtype=bool),
                        roof_interval_columns=np.asarray(["source_x","source_z","source_bottom_y","source_top_y","target_bottom_y","target_top_y","source_thickness_m","target_thickness_m"]))
    mismatch_positions=np.argwhere(mismatches)
    tangent_pairs=getattr(mesh,'tangencies',0)
    success=not (len(mismatch_positions) or odd_columns or vertical_barriers or splits or merges or lost_components or changed_sky or roof_sign_failures or tangent_pairs) and mesh.is_volume
    roof_metrics={"samples":len(roofs),"positive_thickness_and_inside_sign_preserved":not roof_sign_failures,
                  "sign_or_nonpositive_thickness_failures":roof_sign_failures,
                  "source_min_thickness_m":float(roofs[:,6].min()) if len(roofs) else None,
                  "target_min_thickness_m":float(roofs[:,7].min()) if len(roofs) else None,
                  "maximum_thickness_loss_m":float(np.maximum(roofs[:,6]-roofs[:,7],0).max()) if len(roofs) else None,
                  "maximum_absolute_thickness_change_m":float(np.abs(roofs[:,6]-roofs[:,7]).max()) if len(roofs) else None,
                  "one_metre_source_roof_samples":int((roofs[:,6]==1).sum()) if len(roofs) else 0,
                  "one_metre_roof_target_min_m":float(roofs[roofs[:,6]==1,7].min()) if len(roofs) and (roofs[:,6]==1).any() else None,
                  "thickness_deformation_qualification":"measurements retained; positive thickness is an integrity check, not approval of source-relative thinning"}
    recheck_source_topology(ir_dir,source_evidence)
    report={"schema_version":1,"kind":"IndependentMeshConnectivityComparison","status":"pass" if success else "fail",
            "source_payload_evidence":source_evidence,
            "method":"vertical triangle-ray interval parity at every source column, then6-connected target/source air graphs",
            "coordinate_frame":"Minecraft source XYZ metres","ray_count":len(origins),"classified_source_centres":int(occupancy.size),
            "protected_voxel_comparisons":int(protected.sum()),"protected_occupancy_mismatches":len(mismatch_positions),
            "mismatch_source_xyz":[[float(x+xmin+0.5),float(y+ymin+0.5),float(z+zmin+0.5)] for y,z,x in mismatch_positions[:1000]],
            "source_air_component_count":source_count,"target_air_component_count":target_count,
            "source_components_with_anchors":len(source_to_target),"source_component_splits":splits,"target_component_merges":merges,
            "lost_source_components":lost_components,"changed_sky_connections":changed_sky,"feature_comparison":feature_comparison,
            "odd_intersection_columns":odd_columns,"deduplicated_intersections":duplicate_hits,
            "vertical_source_air_barrier_intersections":vertical_barriers,"exterior_deformation_envelope_m":exterior_deformation_envelope_m,
            "roof_thickness":roof_metrics,"structural_support":{**support_context,"inside_failures":int((structural&~predicted).sum()),
                "structural_roof_intervals":sum(roof_is_structural)},
            "graph_resolution_m":1,"unknown_fluid_nonterrain_policy":"unknown/fluid/other nonterrain excluded from both graphs; explicitly retained masonry stays a protected solid in the shared known domain",
            "mesh_positive_closed_volume":bool(mesh.is_volume),"source_ir_sha256":sha256(ir_dir/"world_ir.json"),"source_topology_sha256":sha256(ir_dir/"topology/source_topology_graph.json"),
            "mesh_to_source":np.asarray(mesh_to_source).tolist() if mesh_to_source is not None else "embedded_npz",
            "mesh_sha256":sha256(mesh_path),"validator_sha256":sha256(Path(__file__)),"toolchain":{"trimesh":trimesh.__version__,"rtree":rtree.__version__},
            "limitations":["Graph describes supplied contextual crop; boundary-truncated global caves remain incomplete", "1m source-centre graph is supplemented by exact observed portal segment tests; arbitrary sub-voxel lateral obstructions require further local rays", "No Isaac render/contact or appearance qualification follows from this result"],
            "elapsed_seconds":round(time.monotonic()-started,3),"files":[{"path":"column_ray_reconstruction.npz","bytes":(output/"column_ray_reconstruction.npz").stat().st_size,"sha256":sha256(output/"column_ray_reconstruction.npz")}]}
    if hasattr(mesh,'evidence'):
        report['backend_evidence']=mesh.evidence()
        report['toolchain']={'solver':'All original triangles streamed at every source column; exact rational fallback, no slope cutoff'}
        report['vertical_tangent_ray_triangle_pairs']=tangent_pairs
        verify_mesh_backend_evidence(report)
    write_json(output/"mesh_connectivity_comparison.json",report)
    return report


def validate_source_mesh(ir_dir: Path, mesh_path: Path, output: Path, mesh_to_source=None, sample_count: int=10000, seed: int=1729,
                         *, support_manifest_path: Path | None=None) -> dict:
    if sample_count<10000:
        raise SourceError("Independent region mesh comparison needs at least10000 requested samples")
    ir_dir,mesh_path,output=Path(ir_dir),Path(mesh_path),Path(output)
    from ..validation.native_ground import has_native_ground
    if has_native_ground(mesh_path):
        from .native_source_adapter import validate_source_mesh_native
        return validate_source_mesh_native(ir_dir,mesh_path,output,mesh_to_source,sample_count,seed,support_manifest_path)
    started=time.monotonic()
    topo,source_evidence=verify_source_topology(ir_dir)
    output.mkdir(parents=True,exist_ok=True)
    mesh=_load_mesh(mesh_path,mesh_to_source)
    with np.load(ir_dir/"natural_occupancy.npz",allow_pickle=False) as source:
        occupancy=source["occupancy"]
        validity=source["validity"]
        xmin,ymin,zmin=map(float,source["min_xyz"])
    with np.load(ir_dir/"topology/source_topology_labels.npz",allow_pickle=False) as source:
        covered=source["covered_air"]
    natural=(occupancy==1)&validity
    air=(occupancy==0)&validity
    from .support_validation import structural_support_for_validation
    structural,support_context=structural_support_for_validation(ir_dir,occupancy,support_manifest_path)
    supporting=natural|structural
    connectivity=ndimage.generate_binary_structure(3,1)
    deep_natural=ndimage.binary_erosion(natural,structure=connectivity,border_value=0)
    deep_air=ndimage.binary_erosion(air,structure=connectivity,border_value=0)
    roof=np.zeros_like(air)
    floor=np.zeros_like(air)
    roof[1:]=supporting[1:]&covered[:-1]
    floor[:-1]=supporting[:-1]&covered[1:]
    rng=np.random.default_rng(seed)
    positions=[]
    expected=[]
    strata=[]
    claimed=np.zeros_like(validity)
    masks=[("solid_interior",deep_natural,True,sample_count//2),
           ("air_interior",deep_air,False,sample_count//4),
           ("covered_source_air",covered,False,sample_count//8),
           ("retained_support_roof",roof,True,sample_count//16),
           ("retained_support_floor",floor,True,sample_count//16)]
    for label,mask,inside,count in masks:
        selected=_choose(mask&~claimed,count,rng)
        if len(selected):
            claimed[selected[:,0],selected[:,1],selected[:,2]]=True
        # y,z,x ndarray coordinates become Minecraft source x,y,z centres.
        xyz=selected[:,[2,0,1]].astype(float)+[xmin+0.5,ymin+0.5,zmin+0.5]
        positions.extend(xyz.tolist())
        expected.extend([inside]*len(xyz))
        strata.extend([label]*len(xyz))
    if len(positions)<sample_count:
        selected=_choose((deep_natural|deep_air)&~claimed,sample_count-len(positions),rng)
        xyz=selected[:,[2,0,1]].astype(float)+[xmin+0.5,ymin+0.5,zmin+0.5]
        positions.extend(xyz.tolist())
        expected.extend(natural[selected[:,0],selected[:,1],selected[:,2]].tolist())
        strata.extend(["remaining_interior"]*len(xyz))
    # Every structural cell is protected, including blocks outside a randomly
    # selected natural floor/roof stratum. Keep its class explicit in evidence.
    structural_indices=np.argwhere(structural)
    structural_xyz=structural_indices[:,[2,0,1]].astype(float)+[xmin+.5,ymin+.5,zmin+.5]
    positions.extend(structural_xyz.tolist())
    expected.extend([True]*len(structural_xyz))
    strata.extend(["source_structural_masonry"]*len(structural_xyz))
    portal_segments=[]
    for portal in topo["portals"]:
        for face in portal["faces"]:
            center=np.asarray(face["center_xyz"],dtype=float)
            vector=np.zeros(3)
            vector[{"x":0,"y":1,"z":2}[face["normal_axis"]]]=face["normal_sign"]
            # Both sides of every source aperture face must remain outside
            # natural solid. Test the connecting segment for a spurious cap.
            a,b=center-vector*0.35,center+vector*0.35
            portal_segments.append((a,b,portal["id"]))
            positions.extend([a.tolist(),center.tolist(),b.tolist()])
            expected.extend([False,False,False])
            strata.extend(["portal_aperture"]*3)
    positions=np.asarray(positions,dtype=float)
    expected=np.asarray(expected,dtype=bool)
    classifications,ray_evidence=axis_parity.contains_axis_consensus(mesh,positions)
    failures=np.flatnonzero(classifications!=expected)
    portal_caps=[]
    if portal_segments:
        for start in range(0,len(portal_segments),128):
            batch=portal_segments[start:start+128]
            origins=np.asarray([entry[0] for entry in batch])
            ends=np.asarray([entry[1] for entry in batch])
            distances=np.linalg.norm(ends-origins,axis=1)
            directions=(ends-origins)/distances[:,None]
            hits,ray_ids,face_ids=mesh.ray.intersects_location(origins,directions,multiple_hits=True)
            hit_distances=np.linalg.norm(hits-origins[ray_ids],axis=1)
            blocked=(hit_distances>1e-5)&(hit_distances<distances[ray_ids]-1e-5)
            for hit,ray,face in zip(hits[blocked],ray_ids[blocked],face_ids[blocked]):
                portal_caps.append({"portal_id":batch[int(ray)][2],"source_hit_xyz":hit.tolist(),"triangle":int(face)})
    np.savez_compressed(output/"source_mesh_samples.npz",source_xyz=positions,expected_inside=expected,mesh_inside=classifications,
                        stratum=np.asarray(strata),mismatch_indices=failures,**ray_evidence)
    recheck_source_topology(ir_dir,source_evidence)
    report={"schema_version":1,"kind":"IndependentSourceMeshValidation","status":"pass" if not len(failures) and not portal_caps and mesh.is_volume and len(positions)>=sample_count else "fail",
            "source_payload_evidence":source_evidence,
            "scope":"source interior/air, cave roof/floor, and every observed source portal aperture; does not qualify exterior photorealism",
            "mesh_sha256":sha256(mesh_path),"source_ir_sha256":sha256(ir_dir/"world_ir.json"),"source_topology_sha256":sha256(ir_dir/"topology/source_topology_graph.json"),
            "validator_sha256":sha256(Path(__file__)),
            "classifier_sha256":sha256(Path(axis_parity.__file__)),
            "axis_ambiguous_diagonal_fallbacks":len(ray_evidence["axis_fallback_indices"]),
            "mesh_to_source":np.asarray(mesh_to_source).tolist() if mesh_to_source is not None else "embedded_npz",
            "toolchain":{"trimesh":trimesh.__version__,"rtree":rtree.__version__,"solver":"six opposing axis triangle-ray parity consensus via trimesh/Rtree; original diagonal fallback for any ambiguity"},
            "mesh":{"vertices":len(mesh.vertices),"triangles":len(mesh.faces),"watertight":bool(mesh.is_watertight),"winding_consistent":bool(mesh.is_winding_consistent),"positive_closed_volume":bool(mesh.is_volume),"signed_volume_m3":float(mesh.volume),"source_bounds_xyz":mesh.bounds.tolist()},
            "sample_count":len(positions),"minimum_sample_count":sample_count,"strata":dict(Counter(strata)),"mismatch_count":len(failures),
            "structural_support":{**support_context,"inside_failures":sum(strata[i]=="source_structural_masonry" for i in failures)},
            "mismatch_by_stratum":dict(Counter(strata[i] for i in failures)),"mismatches":[{"source_xyz":positions[i].tolist(),"stratum":strata[i],"expected_inside":bool(expected[i]),"mesh_inside":bool(classifications[i])} for i in failures[:500]],
            "portal_segment_count":len(portal_segments),"portal_cap_intersections":portal_caps,"seed":seed,
            "full_target_connectivity_graph":{"status":"not_run","reason":"Point and aperture segment parity do not prove every path through reconstructed cave connectivity"},
            "isaac_contact":{"status":"not_run","reason":"Independent offline triangle checks do not establish simulator contact"},
            "exterior_refinement_policy":"Deep interior samples allow surface naturalization; protected roof/floor and portal strata receive no unexplained occupancy changes",
            "elapsed_seconds":round(time.monotonic()-started,3),"files":[{"path":"source_mesh_samples.npz","bytes":(output/"source_mesh_samples.npz").stat().st_size,"sha256":sha256(output/"source_mesh_samples.npz")}]}
    write_json(output/"source_mesh_validation.json",report)
    return report


def verify_mesh_backend_evidence(report):
    """Verify optional bounded-native producers/raw records for public consumers."""
    if 'backend_evidence' not in report:
        if report.get('classifier_kind') not in (None,'legacy_trimesh_axis'):
            raise SourceError('Unknown independent geometry classifier')
        return False
    from .native_source_adapter import verify_backend_evidence
    return verify_backend_evidence(report)
