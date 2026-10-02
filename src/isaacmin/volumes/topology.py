"""Source cave connectivity, portal contacts and exact voxel roof constraints.

These are source facts at Minecraft resolution. Topology preservation by a
reconstructed mesh is a separate comparison and cannot inherit this result.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import ndimage

from isaacmin.source.snapshot import canonical_hash, sha256, write_json


CONNECTIVITY = ndimage.generate_binary_structure(3,1)


def source_topology(ir_dir: Path, output: Path | None = None) -> dict:
    ir_dir=Path(ir_dir)
    output=Path(output) if output is not None else ir_dir/"topology"
    output.mkdir(parents=True,exist_ok=True)
    ir_manifest=json.loads((ir_dir/"world_ir.json").read_text())
    source_content=ir_manifest["content_sha256"]
    with np.load(ir_dir/"natural_occupancy.npz",allow_pickle=False) as data:
        occupancy=data["occupancy"]
        validity=data["validity"]
        xmin,ymin,zmin=map(int,data["min_xyz"])
    with np.load(ir_dir/"terrain_surface.npz",allow_pickle=False) as surface:
        heights=surface["height"]
        surface_validity=surface["validity"]
    air=(occupancy==0)&validity
    natural=(occupancy==1)&validity
    covered=air&(np.arange(len(air))[:,None,None]+ymin < heights[None]-1)&surface_validity[None]
    air_labels,air_count=ndimage.label(air,CONNECTIVITY)
    # The top stored air boundary proves contact with the open sky. Crop-side
    # contacts remain truncated, since adjacent out-of-crop geometry is unknown.
    sky_ids=np.unique(air_labels[-1][air[-1]])
    sky_ids=sky_ids[sky_ids!=0]
    sky_connected=np.isin(air_labels,sky_ids)
    exterior=air&~covered&sky_connected
    labels,count=ndimage.label(covered,CONNECTIVITY)
    ext_neighbor=ndimage.binary_dilation(exterior,structure=CONNECTIVITY)
    portal_contacts=covered&ext_neighbor
    portal_labels,portal_count=ndimage.label(portal_contacts,CONNECTIVITY)
    unknown=(~validity)|(occupancy==3)
    unknown_neighbor=ndimage.binary_dilation(unknown,structure=CONNECTIVITY)
    roof_runs=np.zeros(occupancy.shape,dtype=np.uint16)
    for y in range(len(occupancy)-1,-1,-1):
        roof_runs[y]=natural[y]*(1+(roof_runs[y+1] if y+1<len(occupancy) else 0))
    roof_contact=np.zeros_like(air)
    roof_contact[:-1]=covered[:-1]&natural[1:]
    component_slices=ndimage.find_objects(labels)
    portal_slices=ndimage.find_objects(portal_labels)
    features=[]
    portals=[]
    component_ids={}
    for index,slices in enumerate(component_slices,1):
        if slices is None:
            continue
        yy,zz,xx=slices
        mask=labels[slices]==index
        bounds=[xmin+xx.start,ymin+yy.start,zmin+zz.start,xmin+xx.stop,ymin+yy.stop,zmin+zz.stop]
        stable_id="source-cavity-"+canonical_hash({"bounds_xyz":bounds,"source_content":source_content,"component":index})[:20]
        component_ids[index]=stable_id
        boundary=xx.start==0 or zz.start==0 or yy.start==0 or xx.stop==occupancy.shape[2] or zz.stop==occupancy.shape[1] or yy.stop==occupancy.shape[0]
        roofs=np.argwhere(roof_contact[slices]&mask)
        thickness=[]
        if len(roofs):
            ys,zs,xs=(roofs[:,0]+yy.start,roofs[:,1]+zz.start,roofs[:,2]+xx.start)
            thickness=roof_runs[ys+1,zs,xs].tolist()
        contact_ids=np.unique(portal_labels[slices][mask])
        contact_ids=contact_ids[contact_ids!=0].tolist()
        full_ids=np.unique(air_labels[slices][mask]).tolist()
        floor_voxels=0
        yvalues,zvalues,xvalues=np.nonzero(mask)
        gy=yvalues+yy.start
        valid_below=gy>0
        if valid_below.any():
            floor_voxels=int(natural[gy[valid_below]-1,zvalues[valid_below]+zz.start,xvalues[valid_below]+xx.start].sum())
        features.append({"id":stable_id,"label":index,"kind":"source_covered_air_component","source_evidence":"natural_occupancy.npz and exact retained block palettes",
                         "bounds_xyz_blocks":bounds,"air_voxels":int(mask.sum()),"air_volume_m3":int(mask.sum()),"full_air_component_labels":full_ids,
                         "connected_to_observed_sky":bool(np.isin(full_ids,sky_ids).any()),"portal_labels":contact_ids,
                         "touches_crop_boundary":bool(boundary),"touches_unknown_source":bool((unknown_neighbor[slices]&mask).any()),
                         "minimum_vertical_roof_thickness_m":min(thickness) if thickness else None,
                         "vertical_roof_contact_samples":len(thickness),"natural_floor_contacts":floor_voxels,
                         "topology_scope":"truncated_by_crop" if boundary else "inside_crop",
                         "navigation_clearance":"not_inferred","allowed_deformation":"pending qualified feature-specific constraints; retain topology and source roofs"})
    for index,slices in enumerate(portal_slices,1):
        if slices is None:
            continue
        yy,zz,xx=slices
        mask=portal_labels[slices]==index
        contact_cavities=np.unique(labels[slices][mask])
        contact_cavities=contact_cavities[contact_cavities!=0]
        bounds=[xmin+xx.start,ymin+yy.start,zmin+zz.start,xmin+xx.stop,ymin+yy.stop,zmin+zz.stop]
        global_positions=np.argwhere(mask)+np.asarray([yy.start,zz.start,xx.start])
        faces=[]
        for dy,dz,dx,axis,sign in ((1,0,0,"y",1),(-1,0,0,"y",-1),(0,1,0,"z",1),(0,-1,0,"z",-1),(0,0,1,"x",1),(0,0,-1,"x",-1)):
            adjacent=global_positions+np.asarray([dy,dz,dx])
            inside=np.all(adjacent>=0,axis=1)&np.all(adjacent<np.asarray(occupancy.shape),axis=1)
            current=global_positions[inside]
            adjacent=adjacent[inside]
            valid=exterior[adjacent[:,0],adjacent[:,1],adjacent[:,2]]
            for gy,gz,gx in current[valid]:
                face=[float(gx+xmin+0.5),float(gy+ymin+0.5),float(gz+zmin+0.5)]
                face[{"x":0,"y":1,"z":2}[axis]]+=sign*0.5
                faces.append({"center_xyz":face,"normal_axis":axis,"normal_sign":sign})
        portals.append({"id":"source-portal-"+canonical_hash({"bounds":bounds,"index":index,"source_content":source_content})[:20],"label":index,
                        "bounds_xyz_blocks":bounds,"cavity_ids":[component_ids[int(c)] for c in contact_cavities],
                        "interface_voxels":int(mask.sum()),"exterior_face_count":len(faces),"exterior_face_area_m2":len(faces),
                        "faces":faces,"definition":"covered source air faces adjacent to non-covered air connected to observed sky",
                        "robot_passable":"not_inferred"})
    np.savez_compressed(output/"source_topology_labels.npz",cavity_labels=labels,portal_labels=portal_labels,
                        covered_air=covered,exterior_air=exterior,min_xyz=np.asarray([xmin,ymin,zmin],dtype=np.int32))
    report={"schema_version":1,"kind":"SourceTopologyGraph","status":"source_graph_computed","source_ir_sha256":sha256(ir_dir/"world_ir.json"),
            "axis_order":"y,z,x","source_coordinates":"Minecraft metres","connectivity":"6-connected face-neighbor air cells",
            "component_count":count,"portal_count":portal_count,"all_air_component_count":air_count,"observed_sky_component_labels":sky_ids.tolist(),
            "covered_air_voxels":int(covered.sum()),"features":features,"portals":portals,
            "limitations":["Voxel-resolution source evidence, not refined-mesh validation", "Components meeting crop edges require contextual expansion before preserving full-world identity", "Diagonal corner contacts are not traversable openings", "Unknown cells are excluded and never treated as air", "Portal face area is not robot clearance"],
            "target_comparison":{"status":"not_run","reason":"Requires reconstructed native mesh, independent inside/outside rays, and protected portal correspondence"},
            "files":[{"path":"source_topology_labels.npz","bytes":(output/"source_topology_labels.npz").stat().st_size,"sha256":sha256(output/"source_topology_labels.npz")}]}
    write_json(output/"source_topology_graph.json",report)
    return report
