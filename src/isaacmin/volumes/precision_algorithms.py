"""Bounded native precision construction with explicit source-frame constraints."""
from pathlib import Path
import hashlib,json,heapq,time
import numpy as np


def finalize_native_topology(request, output):
    """Reconstruct native float32 topology without a simplification operation."""
    import manifold3d as mf
    out=Path(output);out.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    points=np.load(request['vertices'],allow_pickle=False)
    faces=np.load(request['triangles'],allow_pickle=False)
    if points.dtype!=np.float32 or faces.ndim!=2 or faces.shape[1]!=3:
        raise ValueError('Native finalization requires actual float32 vertices and triangles')
    body=mf.Manifold(mf.Mesh(points,np.asarray(faces,np.uint32)))
    if body.status()!=mf.Error.NoError:raise RuntimeError('Native precision topology rejected: '+str(body.status()))
    encoded=body.to_mesh();v=np.asarray(encoded.vert_properties[:,:3],np.float32);f=np.asarray(encoded.tri_verts,np.int32)
    original_counts={'vertices':len(points),'triangles':len(faces)}
    del body,points,faces
    used=np.bincount(f.reshape(-1),minlength=len(v))>0
    unused=int((~used).sum())
    if unused:
        mapping=np.full(len(v),-1,np.int32);mapping[used]=np.arange(int(used.sum()),dtype=np.int32)
        v=v[used];f=mapping[f]
    del used
    minimum=float('inf');bad=[]
    for start in range(0,len(f),250000):
        p=v[f[start:start+250000]].astype(np.float64)
        area=np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=1)
        minimum=min(minimum,float(area.min()));bad.extend((start+np.flatnonzero(area<=1e-12)).tolist())
    np.save(out/'vertices.npy',v,allow_pickle=False);np.save(out/'triangles.npy',f,allow_pickle=False)
    np.save(out/'native_source_face_ids.npy',np.asarray(encoded.face_id,np.uint32),allow_pickle=False)
    report={'status':'success' if not bad else 'failed','input':original_counts,
        'output':{'vertices':len(v),'triangles':len(f)},'removed_unreferenced_vertices':unused,
        'minimum_area2_m2':minimum,'faces_below_existing_area_threshold':bad,
        'simplification':'not_performed','coordinate_precision':'actual_float32',
        'normals_and_self_intersections':'require_independent_native_and_geometry_checks',
        'elapsed_s':time.monotonic()-started}
    (out/'topology_result.json').write_text(json.dumps(report,indent=2)+'\n')
    if bad:raise RuntimeError('Final topology still has faces below frozen nondegeneracy threshold')
    return report


def condition_coarse(request,output):
    import trimesh,manifold3d as mf
    start=time.monotonic();out=Path(output);out.mkdir(parents=True,exist_ok=True)
    src=trimesh.load(request['terrain_mesh'],force='mesh',process=False)
    native_positions=(np.asarray(src.vertices,np.float32)+np.asarray(request['terrain_translation'],np.float32))
    source_points=np.asarray(src.vertices,float)[src.faces]
    source_normals=np.cross(source_points[:,1]-source_points[:,0],source_points[:,2]-source_points[:,0])
    encoded=mf.Manifold(mf.Mesh(native_positions,np.asarray(src.faces,np.uint32),face_id=np.arange(len(src.faces),dtype=np.uint32)))
    assert encoded.status()==mf.Error.NoError,encoded.status()
    encoded=encoded.to_mesh()
    v=np.asarray(encoded.vert_properties[:,:3],float);f=np.asarray(encoded.tri_verts,np.int64).copy();original=v.copy()
    orientation_reference=source_normals[np.asarray(encoded.face_id)].copy()
    tolerance=.0002;minimum_altitude=.0005
    origin=np.asarray(request['minecraft_origin'],float)
    grid_offset=np.array([origin[0],-origin[2],origin[1]])
    source_grid=np.asarray(src.vertices,float)+np.asarray(request['terrain_translation'],float)+grid_offset
    input_anchors=np.abs(source_grid-np.rint(source_grid))<1e-8
    anchor_lookup={}
    for point,flags in zip(native_positions,input_anchors):
        key=point.tobytes();anchor_lookup[key]=anchor_lookup.get(key,np.zeros(3,bool))|flags
    grid_anchors=np.asarray([anchor_lookup[np.asarray(point,np.float32).tobytes()] for point in v])
    live=np.ones(len(f),bool);inc=[set() for _ in range(len(v))];members={}
    for fi,face in enumerate(f):
        for vertex in face:inc[int(vertex)].add(fi)
    original_maxedge=float(np.linalg.norm(v[f]-np.roll(v[f],1,axis=1),axis=2).max())
    def neighbors(vertex):
        return {int(x) for fi in inc[vertex] for x in f[fi] if x!=vertex}
    def grid_score(vertex):
        # Retain observed source-grid intersections when a nearby roundoff vertex
        # would otherwise replace them; never snap an arbitrary point onto a grid.
        return int(grid_anchors[vertex].sum())
    def metrics(points):
        n=np.cross(points[:,1]-points[:,0],points[:,2]-points[:,0]);length=np.linalg.norm(n,axis=1)
        edge=np.linalg.norm(points-np.roll(points,1,axis=1),axis=2)
        return n,length/np.maximum(edge.max(axis=1),1e-300),edge
    
    normal,altitude,edges=metrics(v[f]);queue=[]
    for face in f[altitude<minimum_altitude]:
        for a,b in zip(face,np.roll(face,1)):
            a,b=sorted((int(a),int(b)));d=float(np.linalg.norm(v[a]-v[b]))
            if d<=tolerance:heapq.heappush(queue,(d,a,b))
    collapsed=[];seen=set();reject={}
    while queue:
        distance,a,b=heapq.heappop(queue)
        if (a,b) in seen:continue
        seen.add((a,b))
        if not inc[a] or not inc[b]:continue
        shared=inc[a]&inc[b]
        if len(shared)!=2:reject['edge_incidence']=reject.get('edge_incidence',0)+1;continue
        opposing={int(x) for fi in shared for x in f[fi] if x not in (a,b)}
        if neighbors(a)&neighbors(b)!=opposing:reject['link_condition']=reject.get('link_condition',0)+1;continue
        keep,remove=(a,b) if (grid_score(a),-a)>=(grid_score(b),-b) else (b,a)
        cluster=members.get(keep,{keep})|members.get(remove,{remove})
        move=float(np.linalg.norm(original[list(cluster)]-v[keep],axis=1).max())
        if move>tolerance:reject['cluster_envelope']=reject.get('cluster_envelope',0)+1;continue
        changed=sorted(inc[remove]-shared);new=f[changed].copy();new[new==remove]=keep
        oldn,_,_=metrics(v[f[changed]]);newn,newalt,newedge=metrics(v[new])
        source_reference=orientation_reference[changed]
        original_flip=np.einsum('ij,ij->i',oldn,source_reference)<=0
        required_reference=np.where(original_flip[:,None],source_reference,oldn)
        if np.any(np.einsum('ij,ij->i',required_reference,newn)<=0) or np.any(newalt<=0) or newedge.max(initial=0)>original_maxedge+2*tolerance:
            reject['face_orientation_or_extent']=reject.get('face_orientation_or_extent',0)+1;continue
        for fi in shared:
            live[fi]=False
            for x in f[fi]:inc[int(x)].discard(fi)
        for fi,face in zip(changed,new):
            inc[remove].discard(fi);inc[keep].add(fi);f[fi]=face
        members[keep]=cluster;members.pop(remove,None)
        collapsed.append({'keep':keep,'remove':remove,'removed_face_ids':sorted(shared),'max_original_vertex_move_m':move})
        for x in neighbors(keep):
            lo,hi=sorted((keep,x));distance=float(np.linalg.norm(v[lo]-v[hi]))
            if distance<=tolerance:heapq.heappush(queue,(distance,lo,hi))
    
    # Re-triangulate skinny local quadrilaterals; both topology and source envelope
    # are checked before every operation. No whole-surface coarsening occurs.
    flips=[];flipped_edges=set();projected=[]
    for iteration in range(8):
        ids=np.flatnonzero(live);_,alt,_=metrics(v[f[ids]])
        bad=ids[alt<minimum_altitude];changed_count=0
        for fi in bad:
            if not live[fi]:continue
            face=f[fi];pn,pa,pe=metrics(v[face[None,:]])
            if pa[0]>=minimum_altitude:continue
            longest=int(np.argmax(pe[0]));a,b=int(face[longest]),int(face[(longest-1)%3]);c=int(next(x for x in face if x not in (a,b)))
            edge=tuple(sorted((a,b)))
            if edge in flipped_edges:continue
            adjacent=(inc[a]&inc[b])-{int(fi)}
            if len(adjacent)!=1:continue
            fj=next(iter(adjacent));d=int(next(x for x in f[fj] if x not in (a,b)))
            if c==d or d in neighbors(c):continue
            old=f[[fi,fj]].copy();oldn,oldalt,oldedge=metrics(v[old]);reference=oldn.sum(axis=0)
            saved_c=v[c].copy();projection=None
            if pa[0]<=tolerance:
                edgevector=v[b]-v[a];t=float(np.dot(v[c]-v[a],edgevector)/np.dot(edgevector,edgevector))
                q=v[a]+np.clip(t,0,1)*edgevector
                fixed=grid_anchors[c]
                if 0<t<1 and np.all(np.abs(q[fixed]-v[c,fixed])<1e-10) and np.linalg.norm(original[list(members.get(c,{c}))]-q,axis=1).max()<=tolerance:
                    other=sorted(inc[c]-{int(fi),int(fj)});before_other,_,_=metrics(v[f[other]])
                    v[c]=q;after_other,other_alt,_=metrics(v[f[other]])
                    if np.any(np.einsum('ij,ij->i',before_other,after_other)<=0) or np.any(other_alt<=0):v[c]=saved_c
                    else:projection=q.copy()
            new=np.asarray([[c,d,a],[d,c,b]],np.int64);newn,newalt,newedge=metrics(v[new])
            if np.dot(newn.sum(axis=0),reference)<0:new=new[:,[0,2,1]];newn,newalt,newedge=metrics(v[new])
            if np.any(newn@reference<=0) or newalt.min()<=oldalt.min()*1.01 or newedge.max()>original_maxedge+2*tolerance:
                v[c]=saved_c;continue
            # In either triangulation, every alternative opposite vertex lies
            # within the fixed metric envelope of the two supporting planes.
            envelope=0.
            for triangles,normals in ((old,oldn),(new,newn)):
                for tri,n in zip(triangles,normals):
                    for point in (a,b,c,d):envelope=max(envelope,float(abs(np.dot(v[point]-v[tri[0]],n))/np.linalg.norm(n)))
            if envelope>tolerance:v[c]=saved_c;continue
            for row in (int(fi),int(fj)):
                for x in f[row]:inc[int(x)].discard(row)
            f[fi],f[fj]=new
            orientation_reference[fi],orientation_reference[fj]=newn
            for row in (int(fi),int(fj)):
                for x in f[row]:inc[int(x)].add(row)
            flips.append({'face_ids':[int(fi),int(fj)],'old':old.tolist(),'new':new.tolist(),'surface_plane_envelope_m':envelope})
            if projection is not None:projected.append({'vertex':c,'before':saved_c.tolist(),'after':projection.tolist(),'original_displacement_m':float(np.linalg.norm(original[c]-projection))})
            flipped_edges.add(edge);changed_count+=1
        if not changed_count:break
    
    faces=f[live];used=np.unique(faces);remap=np.full(len(v),-1,np.int64);remap[used]=np.arange(len(used))
    from .plane_conditioning import condition_planes
    from isaacmin.validation.global_intersections import validate_global_intersections
    conditioned_v,conditioned_f,plane_report=condition_planes(v[used],remap[faces],orientation_reference[live],grid_anchors[used],out/'source_planes')
    crossing_repair=validate_global_intersections(Path(__file__).resolve().parents[3],out/'source_planes/vertices.npy',
        out/'source_planes/triangles.npy',out/'native_crossings')
    if crossing_repair['status']!='pass':raise RuntimeError('Source-plane conditioning left native crossings')
    target=trimesh.Trimesh(conditioned_v,conditioned_f,process=False)
    target.export(out/'terrain.obj',digits=17)
    _,altitude,edges=metrics(target.vertices[target.faces]);_,native_alt,native_edges=metrics(target.vertices.astype(np.float32).astype(float)[target.faces])
    report={'status':'unqualified_candidate','construction_stage':'source_frame_aware_native_conditioning','method':'bounded short-edge collapse with strict closed-mesh link condition; bounded local diagonal flips',
     'max_vertex_move_m':max((r['max_original_vertex_move_m'] for r in collapsed),default=0),'surface_tolerance_m':tolerance,
     'minimum_target_altitude_m':minimum_altitude,'collapsed_edges':len(collapsed),'flipped_edges':len(flips),'rejected_collapses':reject,
     'vertices':len(target.vertices),'triangles':len(target.faces),'positive_closed_mesh':bool(target.is_volume),'consistent_winding':bool(target.is_winding_consistent),
     'minimum_altitude_m':float(altitude.min()),'float32_minimum_altitude_m':float(native_alt.min()),'remaining_thin_faces':int((altitude<minimum_altitude).sum()),
     'float32_zero_area_faces':int((native_alt==0).sum()),'max_edge_m':float(edges.max()),'original_max_edge_m':original_maxedge,
     'collapsed_edge_records':collapsed,'diagonal_flip_records':flips,'shared_edge_projection_records':projected,'elapsed_s':time.monotonic()-start,
     'source_mesh_sha256':hashlib.sha256(Path(request['terrain_mesh']).read_bytes()).hexdigest(),
     'output_sha256':hashlib.sha256((out/'terrain.obj').read_bytes()).hexdigest(),'worker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
     'independent_source_validation':'not_run','native_subdivision_validation':'not_run',
     'crossing_construction':{'status':crossing_repair['status'],'intersection_pairs':crossing_repair['intersection_pair_count']},
     'source_plane_conditioning':plane_report}
    (out/'candidate.json').write_text(json.dumps(report,indent=2)+'\n')
    report['source_plane_anchor_protocol']='derived before float32 encoding in caller coordinate frame'
    report['source_origin_xyz']=origin.tolist()
    report['geometry_detail']=request.get('geometry_detail',{})
    (out/'candidate.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def subdivide_double(request,output):
    import trimesh,subprocess
    from isaacmin.io import atomic_json,sha256_file
    from isaacmin.security import worker_environment
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    mesh=trimesh.load(request['terrain_mesh'],force='mesh',process=False)
    if not np.isfinite(mesh.vertices).all():raise ValueError('Nonfinite coarse subdivision points')
    off=out/'coarse.off'
    with off.open('w') as stream:
        stream.write(f'OFF\n{len(mesh.vertices)} {len(mesh.faces)} 0\n')
        np.savetxt(stream,mesh.vertices,fmt='%.17g')
        np.savetxt(stream,np.column_stack([np.full(len(mesh.faces),3),mesh.faces]),fmt='%d')
    levels=int(request['subdivision_levels'])
    if not 1<=levels<=3:raise ValueError('Subdivision levels outside fixed profile')
    executable=Path(__file__).resolve().parents[3]/'.tools/native_precision/subdivide_double'
    process=subprocess.run([str(executable),str(off),str(levels),str(out/'vertices.npy'),str(out/'quads.npy')],
        env=worker_environment(),check=False)
    if process.returncode:raise RuntimeError('Double bilinear worker failed')
    v=np.load(out/'vertices.npy',mmap_mode='r');q=np.load(out/'quads.npy',mmap_mode='r')
    expected=len(mesh.faces)*3*4**(levels-1)
    if q.shape!=(expected,4) or v.dtype!=np.float32 or q.dtype!=np.int32 or not np.isfinite(v).all():
        raise RuntimeError('Native subdivision output violates exact count/type/finite constraints')
    report={'status':'generated_not_qualified','subdivision_levels':levels,'scheme':'BILINEAR',
        'arithmetic':'OpenSubdiv PrimvarRefinerReal<double>; encode final native points once',
        'corresponding_native_recipe':'Blender SIMPLE with use_limit_surface=False; default sharp-limit workaround is a different interior sampling',
        'input':{'vertices':len(mesh.vertices),'polygons':len(mesh.faces),'polygon_corners':len(mesh.faces)*3},
        'output':{'vertices':len(v),'quads':len(q)},'source_mesh_sha256':sha256_file(Path(request['terrain_mesh'])),
        'vertices_sha256':sha256_file(out/'vertices.npy'),'quads_sha256':sha256_file(out/'quads.npy')}
    atomic_json(out/'subdivision.json',report);return report
