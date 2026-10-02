"""Measured native PhysX rays on stratified final surface triangles."""
import hashlib
import json
from pathlib import Path
import numpy as np
from pxr import UsdGeom,Usd


def sample_contact_rays(stage,query,terrain_paths,output,count=12000,collision_paths=None):
    output=Path(output);rng=np.random.default_rng(40739)
    origins=[];directions=[];source_faces=[];source_meshes=[]
    per_mesh=max(1,int(np.ceil(count/len(terrain_paths))))
    strata={'upward':0,'downward':0,'side':0}
    for mesh_id,path in enumerate(terrain_paths):
        prim=stage.GetPrimAtPath(path);mesh=UsdGeom.Mesh(prim)
        counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get(),dtype=np.int32)
        if not np.all(counts==3):raise RuntimeError('Contact sampler requires authoritative triangles')
        ids=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),dtype=np.int32).reshape(-1,3)
        vertices=np.asarray(mesh.GetPointsAttr().Get(),dtype=np.float64)
        chosen=np.floor(np.linspace(0,len(ids),per_mesh,endpoint=False)).astype(np.int64)
        samples=vertices[ids[chosen]]
        transform=np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
        samples=samples@transform[:3,:3]+transform[3,:3]
        normals=np.cross(samples[:,1]-samples[:,0],samples[:,2]-samples[:,0])
        lengths=np.linalg.norm(normals,axis=1);good=lengths>1e-12
        samples=samples[good];normals=normals[good]/lengths[good,None];chosen=chosen[good]
        uv=rng.random((len(samples),2));uv[uv.sum(axis=1)>1]=1-uv[uv.sum(axis=1)>1]
        points=samples[:,0]+uv[:,:1]*(samples[:,1]-samples[:,0])+uv[:,1:]*(samples[:,2]-samples[:,0])
        origins.extend(points+normals*.2);directions.extend(-normals)
        source_faces.extend(chosen);source_meshes.extend([mesh_id]*len(chosen))
        strata['upward']+=int((normals[:,2]>.5).sum())
        strata['downward']+=int((normals[:,2]<-.5).sum())
        strata['side']+=int((np.abs(normals[:,2])<=.5).sum())
    origins=np.asarray(origins);directions=np.asarray(directions)
    hit=[];distance=[];position=[];normal=[];collision=[];collider_names=[]
    for origin,direction in zip(origins,directions):
        # The independent triangle query intersects both face orientations.
        # A concave surface's normal-offset origin may lie across another wall;
        # preserve every ray and ask PhysX the same two-sided geometric question.
        result=query.raycast_closest(tuple(map(float,origin)),tuple(map(float,direction)),2.0,True)
        hit.append(bool(result.get('hit',False)))
        distance.append(float(result.get('distance',np.nan)) if hit[-1] else np.nan)
        position.append(list(result.get('position',(np.nan,)*3)))
        normal.append(list(result.get('normal',(np.nan,)*3)))
        name=str(result.get('collision',''))
        if name not in collider_names:collider_names.append(name)
        collision.append(collider_names.index(name))
    path=output/'physx_contact_rays.npz'
    np.savez_compressed(path,origins=origins,directions=directions,max_distance_m=2.0,
         hit=np.asarray(hit,dtype=bool),distance_m=np.asarray(distance),
         position=np.asarray(position),normal=np.asarray(normal),
         source_face_index=np.asarray(source_faces),source_mesh_index=np.asarray(source_meshes),
         collision_index=np.asarray(collision,dtype=np.int32))
    report={'status':'measured','path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'ray_count':len(hit),'distinct_origins':len(np.unique(origins,axis=0)),'hits':sum(hit),'strata':strata,
            'sampling':'deterministic stratification across all final triangles, including floor/roof/side normals',
            'ray_triangle_sides':'both; matches independent final-triangle intersection semantics',
            'terrain_paths':collision_paths if collision_paths is not None else terrain_paths,
            'sampled_render_meshes':terrain_paths,'collider_paths':collider_names,
            'independent_geometry_comparison':'not_run','navigation_stack':'not_run'}
    scene=Path(stage.GetRootLayer().realPath).resolve()
    ground=scene.parent/'final_ground.obj'
    report['input_root_usd_sha256']=hashlib.sha256(scene.read_bytes()).hexdigest()
    if ground.is_file():
        with ground.open('rb') as stream:report['final_ground_sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
    else:report['final_ground_sha256']=None
    closure=scene.parent/'native_dependency_closure.json'
    report['native_dependency_closure_sha256']=hashlib.sha256(closure.read_bytes()).hexdigest() if closure.is_file() else None
    (output/'physx_contact_rays.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
