"""Bounded triangle streaming for exact source-column locations, without a BVH.

Every original triangle is visited. Bounding boxes only select possible column
centres; barycentric triangle intersections determine the actual heights. No
terrain/SDF approximation, slope cutoff, or expected-source sign selects hits.
"""
from fractions import Fraction

import numpy as np


def _rational_hit(triangle,x,z):
    p=[[Fraction(float(v)) for v in point] for point in triangle]
    x,z=Fraction(float(x)),Fraction(float(z))
    u=[p[1][i]-p[0][i] for i in range(3)];v=[p[2][i]-p[0][i] for i in range(3)]
    dx,dz=x-p[0][0],z-p[0][2]
    den=u[0]*v[2]-u[2]*v[0]
    if not den:return None
    a=(dx*v[2]-dz*v[0])/den;b=(u[0]*dz-u[2]*dx)/den
    if a<0 or b<0 or a+b>1:return None
    return float(p[0][1]+a*u[1]+b*v[1]),int(a==0 or b==0 or a+b==1)


def streamed_column_intersections(vertices,faces,mesh_to_source,minimum_xz,shape_zx,*,
                                  triangle_batch=131072,candidate_batch=131072):
    """Return every original face hit at unit-source-column centres.

    Heights/face IDs are not deduplicated here. Consumers keep their declared
    coincident-hit and parity policy. Exact rational fallback resolves edge or
    ill-conditioned projected triangles using the stored transformed doubles.
    Tangent rays contained in a vertical face are reported separately.
    """
    transform=np.asarray(mesh_to_source,float);minimum=np.asarray(minimum_xz,float)
    nz,nx=map(int,shape_zx)
    if transform.shape!=(4,4) or not np.isfinite(transform).all() or not np.array_equal(transform[3],[0,0,0,1]):
        raise ValueError('Streamed columns require an explicit finite affine source transform')
    if minimum.shape!=(2,) or not np.isfinite(minimum).all() or min(nx,nz,triangle_batch,candidate_batch)<1:
        raise ValueError('Invalid bounded source column grid')
    if np.asarray(vertices).ndim!=2 or vertices.shape[1]!=3 or np.asarray(faces).ndim!=2 or faces.shape[1]!=3:
        raise ValueError('Expected original vertex and triangle arrays')
    if not np.issubdtype(faces.dtype,np.integer):raise ValueError('Original face indices must be integers')
    ray_hits=[];y_hits=[];face_hits=[];normal_hits=[];edge_hits=[]
    tangencies=[];candidate_count=0;parallel_count=0;near_parallel_hits=0;fallbacks=0;max_batch=0
    bbox_padding=1e-8
    for start in range(0,len(faces),triangle_batch):
        indices=faces[start:start+triangle_batch]
        if np.any(indices<0) or np.any(indices>=len(vertices)):raise ValueError('Invalid original triangle index')
        tri=np.asarray(vertices[indices],float)@transform[:3,:3].T+transform[:3,3]
        if not np.isfinite(tri).all():raise ValueError('Nonfinite original triangle; no row discarded')
        normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);norm=np.linalg.norm(normals,axis=1)
        if np.any(norm==0):raise ValueError('Degenerate original triangle; no row discarded')
        low=np.ceil(tri[:,:,[0,2]].min(axis=1)-minimum-.5-bbox_padding).astype(np.int64)
        high=np.floor(tri[:,:,[0,2]].max(axis=1)-minimum-.5+bbox_padding).astype(np.int64)
        low=np.maximum(low,0);high=np.minimum(high,[nx-1,nz-1])
        width=np.maximum(high-low+1,0);counts=width[:,0]*width[:,1]
        selected=np.flatnonzero(counts)
        if not len(selected):continue
        ends=np.cumsum(counts[selected]);begins=np.r_[0,ends[:-1]]
        candidate_count+=int(ends[-1])
        for first in range(0,int(ends[-1]),candidate_batch):
            flat=np.arange(first,min(first+candidate_batch,int(ends[-1])))
            which=np.searchsorted(ends,flat,side='right');local=selected[which];offset=flat-begins[which]
            x=low[local,0]+offset%width[local,0];z=low[local,1]+offset//width[local,0]
            px=x+minimum[0]+.5;pz=z+minimum[1]+.5
            t=tri[local];u=t[:,1]-t[:,0];v=t[:,2]-t[:,0]
            den=u[:,0]*v[:,2]-u[:,2]*v[:,0]
            dx=px-t[:,0,0];dz=pz-t[:,0,2]
            an=dx*v[:,2]-dz*v[:,0];bn=u[:,0]*dz-u[:,2]*dx
            valid=den!=0;parallel_count+=int((~valid).sum());max_batch=max(max_batch,len(flat))
            # A ray lying in a vertical face has no unique face intersection.
            # Record it rather than choosing a parity result from source labels.
            parallel=np.flatnonzero(~valid)
            for i in parallel:
                if abs(an[i])+abs(bn[i])<=1e-12:
                    tangencies.append((int(z[i]*nx+x[i]),int(start+local[i])))
            a=np.zeros(len(flat));b=np.zeros(len(flat))
            a[valid]=an[valid]/den[valid];b[valid]=bn[valid]/den[valid]
            weights=np.column_stack((a,b,1-a-b))
            inside=valid&np.all(weights>=0,axis=1)&np.all(weights<=1,axis=1)
            height=t[:,0,1]+a*u[:,1]+b*v[:,1]
            edge=np.zeros(len(flat),bool)
            projection_condition=np.abs(den)/(np.linalg.norm(u,axis=1)*np.linalg.norm(v,axis=1))
            ambiguous=valid&((np.min(np.abs(weights),axis=1)<1e-12)|(projection_condition<1e-10))
            for i in np.flatnonzero(ambiguous):
                fallbacks+=1;exact=_rational_hit(t[i],px[i],pz[i]);inside[i]=exact is not None
                if exact is not None:height[i],edge[i]=exact
            chosen=np.flatnonzero(inside)
            if len(chosen):
                ray_hits.append(z[chosen]*nx+x[chosen]);y_hits.append(height[chosen]);face_hits.append(start+local[chosen]);edge_hits.append(edge[chosen])
                normalized_y=normals[local[chosen],1]/norm[local[chosen]];normal_hits.append(normalized_y)
                near_parallel_hits+=int((np.abs(normalized_y)<=1e-5).sum())
    arrays={"ray_index":np.concatenate(ray_hits) if ray_hits else np.empty(0,np.int64),
            "source_y":np.concatenate(y_hits) if y_hits else np.empty(0),
            "triangle_index":np.concatenate(face_hits) if face_hits else np.empty(0,np.int64),
            "normal_source_y":np.concatenate(normal_hits) if normal_hits else np.empty(0),
            "exact_edge_hit":np.concatenate(edge_hits) if edge_hits else np.empty(0,bool),
            "vertical_face_tangent_ray_triangle":np.asarray(tangencies,np.int64).reshape(-1,2)}
    arrays["metrics"]={"original_triangles_visited":len(faces),"requested_columns":nx*nz,
        "source_grid_minimum_xz":minimum.tolist(),"source_grid_shape_zx":[nz,nx],"source_grid_spacing_m":1.,
        "candidate_pairs":candidate_count,"maximum_candidate_batch":max_batch,"triangle_batch":triangle_batch,
        "rational_fallback_pairs":fallbacks,"parallel_candidate_pairs":parallel_count,
        "hits_below_trimesh_plane_cosine_cutoff":near_parallel_hits,"bbox_candidate_padding_m":bbox_padding,
        "intersection_acceptance_padding_m":0.,"slope_cutoff":None,"heightfield_approximation":False}
    return arrays
