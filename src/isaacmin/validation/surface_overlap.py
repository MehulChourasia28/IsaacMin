"""Exact coplanar triangle-overlap predicates for original binary coordinates.

These predicates test supplied pairs. They do not discover arbitrary triangle
pairs or claim a complete global self-intersection test.
"""
from fractions import Fraction

import numpy as np


def _sub(a,b):return [x-y for x,y in zip(a,b)]
def _cross(a,b):return [a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]
def _dot(a,b):return sum(x*y for x,y in zip(a,b))
def _cross2(a,b):return a[0]*b[1]-a[1]*b[0]


def exact_coplanar_intersection(first,second):
    """Positive area is proved with rational arithmetic, never a fit tolerance.

    Float coordinates are interpreted as their exact stored binary numbers.
    The returned projected area differs from 3D area only by the stated axis
    projection; a strictly positive projected area proves surface overlap.
    """
    first,second=np.asarray(first,float),np.asarray(second,float)
    if first.shape!=(3,3) or second.shape!=(3,3) or not np.isfinite([first,second]).all():
        raise ValueError("Overlap predicates require two finite original triangles")
    a=[[Fraction(float(v)) for v in point] for point in first]
    b=[[Fraction(float(v)) for v in point] for point in second]
    normal=_cross(_sub(a[1],a[0]),_sub(a[2],a[0]))
    if not any(normal):return {"status":"degenerate_first_triangle","positive_area_overlap":False}
    if any(_dot(normal,_sub(point,a[0])) for point in b):
        return {"status":"not_exactly_coplanar","positive_area_overlap":False}
    drop=max(range(3),key=lambda axis:abs(normal[axis]));axes=[i for i in range(3) if i!=drop]
    subject=[[point[i] for i in axes] for point in a];clip=[[point[i] for i in axes] for point in b]
    signed=_cross2(_sub(clip[1],clip[0]),_sub(clip[2],clip[0]))
    if not signed:return {"status":"degenerate_second_triangle","positive_area_overlap":False}
    orientation=1 if signed>0 else -1
    polygon=subject
    for index,p in enumerate(clip):
        q=clip[(index+1)%3];edge=_sub(q,p)
        def side(point):return orientation*_cross2(edge,_sub(point,p))
        old=polygon;polygon=[]
        if not old:break
        previous=old[-1];prev_side=side(previous)
        for current in old:
            current_side=side(current)
            if (current_side>=0)!=(prev_side>=0):
                t=prev_side/(prev_side-current_side)
                polygon.append([previous[i]+t*(current[i]-previous[i]) for i in range(2)])
            if current_side>=0:polygon.append(current)
            previous,prev_side=current,current_side
    twice=abs(sum(_cross2(polygon[i],polygon[(i+1)%len(polygon)]) for i in range(len(polygon)))) if polygon else Fraction(0)
    area=twice/2
    return {"status":"exactly_coplanar","positive_area_overlap":area>0,
            "projection_drop_axis":drop,"projected_overlap_area_m2":float(area),
            "projected_overlap_area_exact":f"{area.numerator}/{area.denominator}"}


def shared_edge_overlap_candidates(vertices,faces,first_faces,second_faces,edge_vertices):
    """Prove positive-area folds in supplied shared-edge face pairs.

    A floating broad phase only selects triangles lying on the same side of
    their shared edge. Every reported overlap then uses exact rational clipping.
    """
    first_faces,second_faces=np.asarray(first_faces),np.asarray(second_faces)
    edges=np.asarray(edge_vertices)
    if first_faces.shape!=second_faces.shape or edges.shape!=(len(first_faces),2):
        raise ValueError("Shared-edge face-pair arrays have inconsistent shapes")
    results=[]
    for start in range(0,len(edges),131072):
        stop=min(start+131072,len(edges));selected=edges[start:stop]
        fa=faces[first_faces[start:stop]];fb=faces[second_faces[start:stop]]
        mask_a=(fa!=selected[:,0,None])&(fa!=selected[:,1,None])
        mask_b=(fb!=selected[:,0,None])&(fb!=selected[:,1,None])
        valid=(mask_a.sum(axis=1)==1)&(mask_b.sum(axis=1)==1)
        ids=np.flatnonzero(valid)
        if not len(ids):continue
        c=fa[ids,np.argmax(mask_a[ids],axis=1)];d=fb[ids,np.argmax(mask_b[ids],axis=1)]
        a=vertices[selected[ids,0]].astype(float);b=vertices[selected[ids,1]].astype(float)
        e=b-a;u=np.cross(e,vertices[c]-a);v=np.cross(e,vertices[d]-a)
        same=np.einsum('ij,ij->i',u,v)>0
        # An exact zero plane determinant is required in the narrow phase.
        # The broad-phase tolerance cannot itself establish an overlap.
        scale=np.linalg.norm(u,axis=1)*np.maximum(np.linalg.norm(vertices[d]-a,axis=1),1e-300)
        near=np.abs(np.einsum('ij,ij->i',u,vertices[d]-a))<=scale*1e-12
        for local in ids[same&near]:
            index=start+int(local);left=int(first_faces[index]);right=int(second_faces[index])
            proof=exact_coplanar_intersection(vertices[faces[left]],vertices[faces[right]])
            if proof['positive_area_overlap']:
                results.append({"face_ids":[left,right],"edge_vertex_ids":edges[index].tolist(),**proof})
    return results
