"""Exact constrained disk triangulation of existing binary floating-point vertices."""
from collections import Counter
import numpy as np


def orient(a,b,c):
    return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])


def triangulate_patch(vertices, triangles):
    ids=np.unique(triangles)
    coordinates=np.asarray(vertices[ids],dtype=np.float64)
    denominator=max(float(x).as_integer_ratio()[1] for x in coordinates.reshape(-1))
    integers=[[int(round(float(x)*denominator)) for x in row] for row in coordinates]
    positions=dict(zip(ids.tolist(),integers))
    origin=integers[0];normal=None
    for tri in triangles:
        a,b,c=[positions[int(i)] for i in tri]
        u=[b[i]-a[i] for i in range(3)];w=[c[i]-a[i] for i in range(3)]
        n=[u[1]*w[2]-u[2]*w[1],u[2]*w[0]-u[0]*w[2],u[0]*w[1]-u[1]*w[0]]
        if any(n):normal=n;break
    if normal is None:raise ValueError('Patch has no determined supporting plane')
    if any(sum(normal[i]*(point[i]-origin[i]) for i in range(3)) for point in integers):
        raise ValueError('Patch is not exactly planar in native coordinates')
    drop=max(range(3),key=lambda i:abs(normal[i]));axes=[i for i in range(3) if i!=drop]
    p={key:tuple(point[i]-origin[i] for i in axes) for key,point in positions.items()}
    directed=Counter((int(a),int(b)) for tri in triangles for a,b in zip(tri,np.roll(tri,-1)))
    if any(n!=1 for n in directed.values()):raise ValueError('Repeated oriented patch edge')
    boundary=[edge for edge in directed if (edge[1],edge[0]) not in directed]
    if len(set(a for a,b in boundary))!=len(boundary) or len(set(b for a,b in boundary))!=len(boundary):
        raise ValueError('Boundary is not one manifold cycle')
    following=dict(boundary);loop=[];start=min(following);node=start
    while True:
        if node not in following:raise ValueError('Patch boundary is open')
        loop.append(node);node=following.pop(node)
        if node==start:break
    if following:raise ValueError('Patch has more than one boundary loop')
    for i,a in enumerate(loop):
        b=loop[(i+1)%len(loop)]
        for j in range(i+2,len(loop)):
            if i==0 and j==len(loop)-1:continue
            c,d=loop[j],loop[(j+1)%len(loop)]
            if all(max(p[a][k],p[b][k])>=min(p[c][k],p[d][k]) and max(p[c][k],p[d][k])>=min(p[a][k],p[b][k]) for k in range(2)):
                if orient(p[a],p[b],p[c])*orient(p[a],p[b],p[d])<=0 and orient(p[c],p[d],p[a])*orient(p[c],p[d],p[b])<=0:
                    raise ValueError('Patch boundary self-intersects')
    area=sum(p[a][0]*p[b][1]-p[a][1]*p[b][0] for a,b in zip(loop,loop[1:]+loop[:1]))
    if not area:raise ValueError('Patch has zero oriented area')
    sign=1 if area>0 else -1
    def side(a,b,c):return sign*orient(p[a],p[b],p[c])
    def on_segment(a,b,c):
        return side(a,b,c)==0 and all(min(p[a][k],p[b][k])<=p[c][k]<=max(p[a][k],p[b][k]) for k in range(2))
    def contains(tri,q):return all(side(a,b,q)>=0 for a,b in zip(tri,tri[1:]+tri[:1]))
    remaining=loop.copy();new=[]
    while len(remaining)>3:
        ears=[]
        for i,b in enumerate(remaining):
            a,c=remaining[i-1],remaining[(i+1)%len(remaining)];tri=[a,b,c]
            if side(a,b,c)<=0 or any(contains(tri,q) for q in remaining if q not in tri):continue
            longest=max(sum((p[x][k]-p[y][k])**2 for k in range(2)) for x,y in zip(tri,tri[1:]+tri[:1]))
            ears.append((side(a,b,c)/longest,-b,i,tri))
        if not ears:raise ValueError('Exact polygon has no nondegenerate constrained ear')
        _,_,index,tri=max(ears);new.append(tri);remaining.pop(index)
    if side(*remaining)<=0:raise ValueError('Final ear is degenerate')
    new.append(remaining)
    for q in sorted(set(ids.tolist())-set(loop)):
        if any(on_segment(a,b,q) for a,b in boundary):raise ValueError('Interior vertex lies on fixed patch boundary')
        matches=[i for i,tri in enumerate(new) if contains(tri,q)]
        if not matches:raise ValueError('An existing interior vertex lies outside the patch disk')
        replacements=[]
        if len(matches)==1:
            tri=new[matches[0]]
            for a,b in zip(tri,tri[1:]+tri[:1]):replacements.append([a,b,q])
        elif len(matches)==2:
            for index in matches:
                tri=new[index]
                for a,b in zip(tri,tri[1:]+tri[:1]):
                    if side(a,b,q)>0:replacements.append([a,b,q])
        else:raise ValueError('Duplicate coincident point or invalid triangulation')
        if any(side(*tri)<=0 for tri in replacements):raise ValueError('Insertion would create a degenerate face')
        new=[tri for i,tri in enumerate(new) if i not in matches]+replacements
    # Flip only unconstrained diagonals of strictly convex quads. Exact integer
    # in-circle predicates produce a constrained Delaunay triangulation.
    fixed={tuple(sorted(edge)) for edge in boundary};flips=0
    for iteration in range(max(1,len(new)**2)):
        adjacency={}
        for i,tri in enumerate(new):
            for a,b in zip(tri,tri[1:]+tri[:1]):adjacency.setdefault(tuple(sorted((a,b))),[]).append(i)
        changed=False
        for edge,adjacent in sorted(adjacency.items()):
            if edge in fixed or len(adjacent)!=2:continue
            i,j=adjacent;a,b=edge;c=next(x for x in new[i] if x not in edge);d=next(x for x in new[j] if x not in edge)
            if side(c,d,a)*side(c,d,b)>=0:continue
            if side(a,b,c)<0:a,b=b,a
            aa,bb,cc=[(p[x][0]-p[d][0],p[x][1]-p[d][1]) for x in (a,b,c)]
            det=(aa[0]**2+aa[1]**2)*(bb[0]*cc[1]-bb[1]*cc[0])-(bb[0]**2+bb[1]**2)*(aa[0]*cc[1]-aa[1]*cc[0])+(cc[0]**2+cc[1]**2)*(aa[0]*bb[1]-aa[1]*bb[0])
            if sign*det<=0:continue
            t1,t2=[c,d,a],[d,c,b]
            if side(*t1)<0:t1[1],t1[2]=t1[2],t1[1]
            if side(*t2)<0:t2[1],t2[2]=t2[2],t2[1]
            if side(*t1)<=0 or side(*t2)<=0:raise ValueError('Diagonal flip degenerates the disk')
            new[i],new[j]=t1,t2;flips+=1;changed=True;break
        if not changed:break
    else:raise ValueError('Constrained triangulation did not converge')
    result=np.asarray(new,dtype=triangles.dtype)
    assert set(result.reshape(-1))==set(ids)
    assert sum(orient(p[a],p[b],p[c]) for a,b,c in result)==area
    new_directed=Counter((int(a),int(b)) for tri in result for a,b in zip(tri,np.roll(tri,-1)))
    assert set(edge for edge in new_directed if (edge[1],edge[0]) not in new_directed)==set(boundary)
    assert all(n==1 for n in new_directed.values())
    points=np.asarray(vertices[result],dtype=np.float32)
    area2=np.linalg.norm(np.cross(points[:,1]-points[:,0],points[:,2]-points[:,0]).astype(np.float64),axis=1)
    if np.any(area2<=1e-12):raise ValueError('Native float32 face arithmetic fails existing area threshold')
    maximum_edge=float(np.linalg.norm(points.astype(float)-points[:,[1,2,0]].astype(float),axis=2).max())
    return result,{'vertices_preserved':len(ids),'boundary_edges_preserved':len(boundary),
        'original_faces':len(triangles),'replacement_faces':len(result),'delaunay_flips':flips,
        'maximum_edge_m':maximum_edge,'minimum_native_area2':float(area2.min()),
        'exact_planar_disk_area_m2':abs(area)/(2*denominator**2),'drop_axis':drop,
        'boundary_vertex_ids':loop,'vertex_positions_changed':0}
