"""Final native topology and exact planar-fold repair before material construction."""
from pathlib import Path
import json
import numpy as np
import bpy
from isaacmin.io import atomic_json,sha256_file
from isaacmin.volumes.native_precision import run_precision
from isaacmin.volumes.planar_patch import triangulate_patch


def _arrays(mesh):
    v=np.empty((len(mesh.vertices),3),np.float32);mesh.vertices.foreach_get('co',v.reshape(-1))
    f=np.empty((len(mesh.polygons),3),np.int32)
    if len(mesh.loops)!=f.size:raise ValueError('Authoritative precision repair requires triangles')
    mesh.loops.foreach_get('vertex_index',f.reshape(-1));return v,f


def _bad_normals(mesh):
    normals=np.empty((len(mesh.vertices),3),np.float32);mesh.vertices.foreach_get('normal',normals.reshape(-1))
    zero=np.all(normals==0,axis=1);nonfinite=~np.isfinite(normals).all(axis=1)
    return np.flatnonzero(zero|nonfinite),int(zero.sum()),int(nonfinite.sum())


def _incident(f,vertices,total_vertices):
    selected=np.zeros(total_vertices,bool);selected[vertices]=True;ids=[]
    for start in range(0,len(f),1_000_000):
        ids.append(start+np.flatnonzero(selected[f[start:start+1_000_000]].any(axis=1)))
    return np.concatenate(ids)


def repair_folded_normals(mesh,output):
    """Seeds come from current measured normals, never map-specific vertex IDs."""
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    v,f=_arrays(mesh);bad,zero,nonfinite=_bad_normals(mesh);original_bad=bad.copy();records=[]
    for vertex in bad:
        patch=_incident(f,np.asarray([vertex]),len(v));last_error=None
        # One bounded deterministic disk search, not repeated candidate tuning.
        for ring in range(9):
            try:
                replacement,info=triangulate_patch(v,f[patch]);break
            except ValueError as exc:
                last_error=str(exc)
                if 'not exactly planar' in last_error or 'no determined' in last_error:raise
                # Obtain the exact native supporting plane from a nondegenerate
                # incident triangle. Axis-aligned planes are compared exactly;
                # arbitrary binary planes use integer dot products.
                points=v[f[patch]].astype(np.float64);normals=np.cross(points[:,1]-points[:,0],points[:,2]-points[:,0])
                seed=int(np.argmax(np.linalg.norm(normals,axis=1)));a,b,c=points[seed]
                candidates=_incident(f,np.unique(f[patch]),len(v));cp=v[f[candidates]]
                normal=normals[seed];axis=int(np.argmax(abs(normal)))
                if np.count_nonzero(normal)==1:
                    planar=np.all(cp[:,:,axis]==a[axis],axis=1)
                else:
                    denominator=max(float(x).as_integer_ratio()[1] for x in np.r_[points[seed].reshape(-1),cp.reshape(-1)])
                    aa,bb,cc=[[int(float(x)*denominator) for x in point] for point in (a,b,c)]
                    u=[bb[i]-aa[i] for i in range(3)];w=[cc[i]-aa[i] for i in range(3)]
                    n=[u[1]*w[2]-u[2]*w[1],u[2]*w[0]-u[0]*w[2],u[0]*w[1]-u[1]*w[0]]
                    planar=np.asarray([all(sum(n[i]*(int(float(point[i])*denominator)-aa[i]) for i in range(3))==0 for point in tri) for tri in cp])
                expanded=candidates[planar]
                if not np.isin(patch,expanded).all() or len(expanded)<=len(patch):raise ValueError('No larger exact planar disk: '+last_error)
                patch=expanded
        else:raise ValueError('Bounded planar disk search exhausted: '+str(last_error))
        original=f[patch].copy()
        maximum=float(np.linalg.norm(v[original].astype(float)-v[original[:,[1,2,0]]].astype(float),axis=2).max())
        if len(replacement)!=len(patch) or info['maximum_edge_m']>maximum+1e-12:
            raise ValueError('Retriangulation must preserve face count and physical edge sampling')
        f[patch]=replacement;np.savez_compressed(output/f'patch_{len(records):04d}.npz',
            face_ids=patch,original_triangles=original,replacement_triangles=replacement)
        records.append(dict(info,seed_vertex=int(vertex),ring=ring,face_ids=patch.tolist()))
    if records:
        mesh.loops.foreach_set('vertex_index',f.reshape(-1));mesh.update(calc_edges=True,calc_edges_loose=True)
    observed,_=_arrays(mesh)
    if not np.array_equal(v,observed):raise RuntimeError('Planar repair changed native vertex positions')
    remaining,z,n=_bad_normals(mesh)
    report={'status':'pass' if not len(remaining) else 'fail','input_zero_normals':zero,
        'input_nonfinite_normals':nonfinite,'input_bad_vertices':original_bad.tolist(),
        'output_zero_normals':z,'output_nonfinite_normals':n,'patches':records,
        'vertex_positions_changed':0,'face_count_changed':0,'source_constraints':'unchanged native positions and oriented patch boundaries',
        'actual_normal_count':len(v),'vertices':len(v),'triangles':len(f)}
    atomic_json(output/'planar_repair.json',report)
    if len(remaining):raise RuntimeError('Actual native normal check still fails')
    return report


def finalize_mesh(terrain,root,output,*,preserve_native_topology=False):
    if preserve_native_topology:return _preserve_and_check_native(terrain,root,output)
    root=Path(root);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    v,f=_arrays(terrain.data)
    np.save(output/'input_vertices.npy',v,allow_pickle=False);np.save(output/'input_triangles.npy',f,allow_pickle=False)
    request={'mode':'final','vertices':str(output/'input_vertices.npy'),'triangles':str(output/'input_triangles.npy'),
        'vertices_sha256':sha256_file(output/'input_vertices.npy'),'triangles_sha256':sha256_file(output/'input_triangles.npy')}
    del v,f
    result=run_precision(root,request,output/'topology',estimated_memory_bytes=32*2**30)
    v=np.load(output/'topology/vertices.npy');f=np.load(output/'topology/triangles.npy')
    old=terrain.data;materials=list(old.materials)
    mesh=bpy.data.meshes.new('NativeAuthoritativeGround');mesh.vertices.add(len(v));mesh.vertices.foreach_set('co',v.reshape(-1))
    mesh.loops.add(f.size);mesh.loops.foreach_set('vertex_index',f.reshape(-1));mesh.polygons.add(len(f))
    mesh.polygons.foreach_set('loop_start',np.arange(len(f),dtype=np.int32)*3)
    mesh.polygons.foreach_set('loop_total',np.full(len(f),3,np.int32));mesh.update(calc_edges=True,calc_edges_loose=True)
    for material in materials:mesh.materials.append(material)
    terrain.data=mesh
    if old.users==0:bpy.data.meshes.remove(old)
    del v,f
    repaired=repair_folded_normals(mesh,output/'planar')
    v,f=_arrays(mesh);np.save(output/'final_vertices.npy',v,allow_pickle=False);np.save(output/'final_triangles.npy',f,allow_pickle=False)
    report={'status':'native_geometry_candidate_not_qualified','topology_manifest_sha256':sha256_file(output/'topology/precision_manifest.json'),
        'planar_repair':repaired,'final_vertices_sha256':sha256_file(output/'final_vertices.npy'),
        'final_triangles_sha256':sha256_file(output/'final_triangles.npy'),
        'blender_version':bpy.app.version_string,'blender_build_hash':bpy.app.build_hash.decode(),
        'producer_sha256':sha256_file(Path(__file__)),'independent_geometry_source_and_render_checks':'required'}
    atomic_json(output/'finalization.json',report);return report


def _preserve_and_check_native(terrain,root,output):
    from isaacmin.validation.global_intersections import validate_global_intersections
    import os
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    v,f=_arrays(terrain.data);bad,zero,nonfinite=_bad_normals(terrain.data)
    np.save(output/'final_vertices.npy',v,allow_pickle=False);np.save(output/'final_triangles.npy',f,allow_pickle=False)
    bad_faces=[];minimum=float('inf')
    for start in range(0,len(f),250000):
        p=v[f[start:start+250000]].astype(float);area=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)
        minimum=min(minimum,float(area.min()));bad_faces.extend((start+np.flatnonzero(area<=1e-12)).tolist())
    report={'status':'native_geometry_candidate_not_qualified','construction':'native triangles preserved verbatim; no topology cleanup or authored normals',
        'vertices':len(v),'triangles':len(f),'actual_normal_count':len(v),'zero_native_normals':zero,'nonfinite_native_normals':nonfinite,
        'minimum_area2_m2':minimum,'faces_below_existing_area_threshold':bad_faces,'bad_normal_vertex_ids':bad.tolist(),
        'final_vertices_sha256':sha256_file(output/'final_vertices.npy'),'final_triangles_sha256':sha256_file(output/'final_triangles.npy'),
        'blender_version':bpy.app.version_string,'blender_build_hash':bpy.app.build_hash.decode(),
        'producer_sha256':sha256_file(Path(__file__)),'independent_geometry_source_and_render_checks':'required'}
    atomic_json(output/'finalization.json',report);del v,f
    if len(bad) or bad_faces:raise RuntimeError('Actual native geometry failed normals or nondegeneracy; retained without fallback')
    global_result=validate_global_intersections(root,output/'final_vertices.npy',output/'final_triangles.npy',output/'global_intersections')
    report['global_intersections_sha256']=sha256_file(output/'global_intersections/global_intersections.json')
    report['global_intersection_pairs']=global_result['intersection_pair_count'];report['global_intersections_complete']=global_result['complete']
    if global_result['status']!='pass':
        report['status']='failed_native_intersections';atomic_json(output/'finalization.json',report)
        raise RuntimeError('Actual native triangles intersect; no silent final topology repair')
    atomic_json(output/'finalization.json',report);return report


def write_authoritative_obj(terrain,root,output,precision_finalization=None):
    from isaacmin.process import run_worker
    output=Path(output);root=Path(root);v,f=_arrays(terrain.data)
    if not np.array_equal(np.asarray(terrain.matrix_world),np.eye(4)):
        raise RuntimeError('Authoritative ground must have identity world transform')
    directory=output/'native_precision' if precision_finalization else output/'obj_geometry'
    directory.mkdir(parents=True,exist_ok=True)
    vp=directory/'final_vertices.npy';fp=directory/'final_triangles.npy'
    if precision_finalization:
        observed=np.load(vp,mmap_mode='r');indices=np.load(fp,mmap_mode='r')
        if not np.array_equal(v,observed) or not np.array_equal(f,indices):
            raise RuntimeError('Material or export stages changed validated native ground geometry')
    else:
        np.save(vp,v,allow_pickle=False);np.save(fp,f,allow_pickle=False)
    del v,f
    points=np.load(vp,mmap_mode='r');faces=np.load(fp,mmap_mode='r')
    if points.dtype!=np.float32 or faces.dtype!=np.int32 or not points.flags.c_contiguous or not faces.flags.c_contiguous:
        raise RuntimeError('Authoritative OBJ writer requires C-order native float32/int32 buffers')
    build=json.loads((root/'.tools/native_precision/build_manifest.json').read_text())
    for entry in build['workers']['write_native_obj']['files']:
        if sha256_file(Path(entry['path']))!=entry['sha256']:raise RuntimeError('Authoritative OBJ worker dependency changed')
    result=run_worker([str(root/'.tools/native_precision/write_native_obj'),str(vp),str(points.offset),str(len(points)),str(fp),str(faces.offset),str(len(faces)),str(output/'final_ground.obj')],
        cwd=root,log_path=directory/'obj_writer.log',timeout=86400,estimated_memory_bytes=2**30,
        estimated_disk_bytes=len(points)*80+len(faces)*36)
    if result['exit_code']!=0:raise RuntimeError('Exact authoritative OBJ export failed')
    report={'status':'native_array_exported','method':'17 significant decimal digits of each exact float32 coordinate; original triangle indices',
        'vertices_sha256':sha256_file(vp),'triangles_sha256':sha256_file(fp),'obj_sha256':sha256_file(output/'final_ground.obj'),
        'obj_bytes':(output/'final_ground.obj').stat().st_size,'writer_build_manifest_sha256':sha256_file(root/'.tools/native_precision/build_manifest.json'),
        'independent_roundtrip_validation':'required'}
    atomic_json(directory/'obj_export.json',report);return report
