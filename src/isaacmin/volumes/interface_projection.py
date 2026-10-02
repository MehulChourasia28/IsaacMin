"""Constrain one shared 3D support mesh to exact source interface samples.

This is a generation operator, never an independent preservation certificate.
Native OpenVDB topology-to-levelset rounds thin features. Fixed-XY barycentric
constraints restore measured source floors, ceilings and exterior heights;
independent sign, topology, thickness and portal tests still decide validity.
"""
from __future__ import annotations
import hashlib,json,time
from pathlib import Path
import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsmr


def constrain_interfaces(mesh_path: Path, output_dir: Path, *, supporting_solid: np.ndarray,
                         covered_air: np.ndarray, min_xyz: np.ndarray,
                         exterior_height: np.ndarray, exterior_validity: np.ndarray,
                         known_source_air: np.ndarray | None = None) -> dict:
    started=time.monotonic();mesh_path=Path(mesh_path).resolve();output_dir=Path(output_dir).resolve()
    solid=np.asarray(supporting_solid);covered=np.asarray(covered_air);minimum=np.asarray(min_xyz,float)
    height=np.asarray(exterior_height,float);validity=np.asarray(exterior_validity,bool)
    if solid.dtype!=np.bool_ or covered.dtype!=np.bool_ or solid.shape!=covered.shape or solid.ndim!=3:
        raise ValueError('Supporting occupancy and covered air must be matching Boolean[y,z,x]')
    if minimum.shape!=(3,) or height.shape!=solid.shape[1:] or validity.shape!=height.shape or not np.isfinite(height[validity]).all():
        raise ValueError('Source interface frame/surface shape invalid')
    air=~solid if known_source_air is None else np.asarray(known_source_air,bool)
    if air.shape!=solid.shape or np.any(air&solid):raise ValueError('Source-air semantics invalid')
    if np.any(solid&covered):raise ValueError('Source solid/covered-air masks overlap')
    mesh=trimesh.load(mesh_path,force='mesh',process=False)
    # Weld ONLY exactly equal coordinates. No rounding, crack filling or smoothing.
    vertices,inverse=np.unique(mesh.vertices,axis=0,return_inverse=True)
    mesh=trimesh.Trimesh(vertices,inverse[mesh.faces],process=False)
    if not mesh.is_volume:raise ValueError('Projection requires closed positive-volume native mesh')
    mesh_triangles=np.asarray(mesh.triangles).copy();mesh_faces=np.asarray(mesh.faces).copy()
    ny,nz,nx=solid.shape;targets={};counts={'cave_floors':0,'roof_undersides':0,'roof_tops':0,'exterior':0}
    def add(x,z,y,role,up):
        key=(float(x),float(z),float(y));value=targets.setdefault(key,{'roles':[],'up':up})
        if value['up']!=up:raise ValueError('Source interface orientation conflict')
        value['roles'].append(role);counts[role]+=1
    fy,fz,fx=np.nonzero(solid[:-1]&covered[1:])
    for y,z,x in zip(fy,fz,fx):add(x,z,minimum[1]+y+1,'cave_floors',True)
    for z in range(nz):
        for x in range(nx):
            roofs=np.flatnonzero(solid[1:,z,x]&covered[:-1,z,x])+1
            ends=np.r_[np.flatnonzero(~solid[:,z,x]),ny]
            for y in roofs:
                end=ends[np.searchsorted(ends,y,side='right')]
                add(x,z,minimum[1]+y,'roof_undersides',False)
                add(x,z,minimum[1]+end,'roof_tops',True)
    for z,x in np.argwhere(validity):add(x,z,height[z,x],'exterior',True)
    # Dense stencils apply only where all nine source columns agree on the
    # complete supporting interval and known air on both sides. Source cliffs,
    # diagonals, crop boundaries and unknown/non-air cells are not mislabelled
    # as uniform material interiors.
    stencils=[];offsets=(-.4,-.2,0.,.2,.4)
    for z in range(1,nz-1):
        for x in range(1,nx-1):
            region=(slice(z-1,z+2),slice(x-1,x+2))
            floors=np.flatnonzero(solid[:-1,z,x]&covered[1:,z,x])+1
            for y in floors:
                if solid[y-1][region].all() and air[y][region].all():
                    stencils.append((x,z,minimum[1]+y,True,'cave_floors'))
            roofs=np.flatnonzero(solid[1:,z,x]&covered[:-1,z,x])+1
            ends=np.r_[np.flatnonzero(~solid[:,z,x]),ny]
            for y in roofs:
                end=int(ends[np.searchsorted(ends,y,side='right')])
                if end>=ny:continue
                if solid[y:end,z-1:z+2,x-1:x+2].all() and air[y-1][region].all() and air[end][region].all():
                    stencils.extend([(x,z,minimum[1]+y,False,'roof_undersides'),
                                     (x,z,minimum[1]+end,True,'roof_tops')])
            if validity[region].all() and np.all(height[region]==height[z,x]):
                stencils.append((x,z,height[z,x],True,'exterior'))
    for x,z,y,up,role in stencils:
        for dx in offsets:
            for dz in offsets:add(x+dx,z+dz,y,role,up)
    xy=sorted({(x,z) for x,z,_ in targets})
    ray_lookup={point:index for index,point in enumerate(xy)}
    origins=np.asarray([[x,-z,ny+2] for x,z in xy],float)
    locations=[];rays=[];faces=[]
    for start in range(0,len(origins),256):
        batch=origins[start:start+256]
        points,ids,triangles=mesh.ray.intersects_location(batch,np.tile([0.,0.,-1.],(len(batch),1)),multiple_hits=True)
        locations.extend(points);rays.extend(ids+start);faces.extend(triangles)
    locations=np.asarray(locations);rays=np.asarray(rays);faces=np.asarray(faces)
    order=np.argsort(rays,kind='stable');ordered_rays=rays[order]
    unique,starts,nums=np.unique(ordered_rays,return_index=True,return_counts=True)
    by_ray={int(r):order[a:a+n] for r,a,n in zip(unique,starts,nums)}
    upward=np.asarray(mesh.face_normals)[:,2]>0
    choices=[];residual=[];records=[];failures=[]
    for (x,z,expected),spec in targets.items():
        candidates=by_ray.get(ray_lookup[(x,z)],np.array([],dtype=int))
        candidates=candidates[upward[faces[candidates]]==spec['up']]
        if not len(candidates):failures.append([x,z,expected,'no_oriented_interface']);continue
        choice=candidates[np.argmin(abs(locations[candidates,2]+minimum[1]+.5-expected))]
        actual=float(locations[choice,2]+minimum[1]+.5)
        if abs(actual-expected)>.50001:failures.append([x,z,expected,actual]);continue
        choices.append(choice);residual.append(expected-actual);records.append([x,z,expected,actual])
    if failures:raise ValueError('Native interface pairing failed; refusing inferred replacement: '+str(failures[:8]))
    choices=np.asarray(choices);selected=faces[choices]
    bary=trimesh.triangles.points_to_barycentric(mesh_triangles[selected],locations[choices])
    matrix=coo_matrix((bary.ravel(),(np.repeat(np.arange(len(choices)),3),mesh_faces[selected].ravel())),
                      shape=(len(choices),len(mesh.vertices))).tocsr()
    solution=lsmr(matrix,np.asarray(residual),damp=1e-8,atol=1e-12,btol=1e-12,maxiter=3000)
    delta=solution[0];error=matrix@delta-np.asarray(residual)
    if not np.isfinite(delta).all() or np.max(abs(error))>1e-7 or np.max(abs(delta))>1:
        output_dir.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(output_dir/'failed_constraints.npz',source_targets=np.asarray(records),vertex_delta_z=delta,residual_after=error)
        failure={'status':'fail','constraints':len(choices),'max_abs_residual_m':float(np.max(abs(error))),
                 'max_abs_vertex_offset_m':float(np.max(abs(delta))),'solver_stop':int(solution[1]),
                 'iterations':int(solution[2]),'reason':'Exact bounded interface constraints not satisfied'}
        (output_dir/'projection_failure.json').write_text(json.dumps(failure,indent=2)+'\n')
        raise ValueError(str(failure))
    mesh.vertices[:,2]+=delta
    if not mesh.is_volume:raise ValueError('Projected mesh lost positive closed topology')
    output_dir.mkdir(parents=True,exist_ok=True);target=output_dir/'terrain.obj'
    mesh.export(target,digits=9)
    arrays=output_dir/'constraints.npz'
    np.savez_compressed(arrays,source_targets=np.asarray(records),vertex_delta_z=delta,residual_after=error,
                        min_xyz=minimum,axis_order=np.asarray(['x','-z','y']))
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    record={'status':'generated_candidate','method':'shared_triangle_fixed_XY_minimum_Z_displacement',
            'source_mesh_sha256':sha(mesh_path),'mesh':str(target),'mesh_sha256':sha(target),
            'constraints':len(choices),'unique_ray_columns':len(origins),'uniform_interior_stencils':len(stencils),
            'interior_offsets_m':list(offsets),'interior_rule':'All9source columns agree on complete solid interval and known air bounding it',
            'constraint_roles':counts,'pairing_failures':0,
            'max_abs_constraint_residual_m':float(np.max(abs(error))),
            'max_abs_vertex_Z_offset_m':float(np.max(abs(delta))),
            'solver':'scipy.sparse.linalg.lsmr','iterations':int(solution[2]),'stop_code':int(solution[1]),
            'vertices':len(mesh.vertices),'triangles':len(mesh.faces),'positive_closed_mesh':bool(mesh.is_volume),
            'occupancy_sha256':hashlib.sha256(np.ascontiguousarray(solid).tobytes()).hexdigest(),
            'covered_air_sha256':hashlib.sha256(np.ascontiguousarray(covered).tobytes()).hexdigest(),
            'surface_height_sha256':hashlib.sha256(np.ascontiguousarray(height).tobytes()).hexdigest(),
            'source_frame_min_xyz':minimum.tolist(),'coordinate_frame':'Native OpenVDB voxel centres X=x,Y=-z,Z=y; source origin min_xyz+0.5',
            'constraints_npz':str(arrays),'constraints_sha256':sha(arrays),
            'independent_source_validation':'not_run','appearance_qualification':'not_run',
            'elapsed_seconds':time.monotonic()-started}
    (output_dir/'projection.json').write_text(json.dumps(record,indent=2)+'\n')
    return record
