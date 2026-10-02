"""Retriangulate flat water interiors without changing their covered surface.

Every partially wet fine cell keeps its original clipped triangles. Fully wet
rectangles use a planar fan with EVERY original perimeter vertex, so neighboring
rectangles and detailed shore cells meet without T junctions. No boundary,
water level, material coordinate or terrain/collision vertex is simplified.
"""
import numpy as np


def _integral(mask):
    return np.pad(mask.cumsum(axis=0,dtype=np.int64).cumsum(axis=1,dtype=np.int64),((1,0),(1,0)))


def compact_planar_water(points,values,clipper):
    points=np.asarray(points);values=np.asarray(values)
    nz,nx=values.shape
    if points.shape!=(nz,nx,3) or not np.isfinite(points).all() or not np.isfinite(values).all():
        raise ValueError('Finite regular water grid required')
    if not np.all(points[:,:,2]==points[0,0,2]):raise ValueError('Only an exactly horizontal plane may be compacted')
    positive=values>0
    full=positive[:-1,:-1]&positive[:-1,1:]&positive[1:,:-1]&positive[1:,1:]
    any_wet=positive[:-1,:-1]|positive[:-1,1:]|positive[1:,:-1]|positive[1:,1:]
    full_sum=_integral(full);wet_sum=_integral(any_wet)
    def count(a,z0,z1,x0,x1):return int(a[z1,x1])-int(a[z0,x1])-int(a[z1,x0])+int(a[z0,x0])
    stack=[(0,nz-1,0,nx-1)];parts=[];boundary=[];full_count=0;empty_count=0;rectangles=0
    while stack:
        z0,z1,x0,x1=stack.pop();area=(z1-z0)*(x1-x0)
        if count(wet_sum,z0,z1,x0,x1)==0:empty_count+=area;continue
        if count(full_sum,z0,z1,x0,x1)==area:
            full_count+=area;rectangles+=1
            # Source +Z is world -Y. This perimeter winds counterclockwise in
            # world XY, as do the original two fine triangles in each cell.
            # Keep acceleration-structure boxes bounded: a single kilometre
            # fan would have thousands of heavily overlapping ray-test bounds.
            # Interior2m rectangles remain EXACTLY on the original plane.
            step=max(1,int(2./abs(float(points[0,1,0]-points[0,0,0]))))
            zs=np.unique(np.r_[np.arange(z0,z1,step),z1]);xs=np.unique(np.r_[np.arange(x0,x1,step),x1])
            az,ax=np.meshgrid(zs[:-1],xs[:-1],indexing='ij');bz,bx=np.meshgrid(zs[1:],xs[1:],indexing='ij')
            interior=(az>z0)&(bz<z1)&(ax>x0)&(bx<x1)
            a,b,c,d=points[az[interior],ax[interior]],points[bz[interior],ax[interior]],points[bz[interior],bx[interior]],points[az[interior],bx[interior]]
            if len(a):parts.extend((np.stack((a,b,d),axis=1),np.stack((d,b,c),axis=1)))
            for za,zb,xa,xb in zip(az[~interior],bz[~interior],ax[~interior],bx[~interior]):
                ring=np.concatenate((points[za:zb,xa] if xa==x0 else points[za:za+1,xa],
                    points[zb,xa:xb] if zb==z1 else points[zb,xa:xa+1],
                    points[zb:za:-1,xb] if xb==x1 else points[zb:zb+1,xb],
                    points[za,xb:xa:-1] if za==z0 else points[za,xb:xb+1]))
                center=(points[za,xa]+points[zb,xb])*.5
                parts.append(np.stack((np.broadcast_to(center,ring.shape),ring,np.roll(ring,-1,axis=0)),axis=1))
            continue
        if area==1:boundary.append(z0*nx+x0);continue
        zm=(z0+z1)//2;xm=(x0+x1)//2
        zs=[z0,zm,z1] if z1-z0>1 else [z0,z1]
        xs=[x0,xm,x1] if x1-x0>1 else [x0,x1]
        stack.extend((a,b,c,d) for a,b in zip(zs[:-1],zs[1:]) for c,d in zip(xs[:-1],xs[1:]))
    if full_count!=int(full.sum()) or len(boundary)!=int((any_wet&~full).sum()):
        raise ValueError('Planar cell ownership has a gap or duplicate')
    if full_count+empty_count+len(boundary)!=(nz-1)*(nx-1):raise ValueError('Water coverage mismatch')
    if boundary:
        a=np.asarray(boundary,np.int32);faces=np.empty((2*len(a),3),np.int32)
        faces[0::2]=np.column_stack((a,a+nx,a+1));faces[1::2]=np.column_stack((a+1,a+nx,a+nx+1))
        clipped,_=clipper(points.reshape(-1,3),faces,values.ravel());parts.append(clipped)
    triangles=np.concatenate(parts) if parts else np.empty((0,3,3),points.dtype)
    cross=np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])
    if np.any(cross[:,2]<=0):raise ValueError('Water winding or degenerate fan failure')
    area=float(cross[:,2].sum()/2)
    return triangles,area,dict(original_full_cells=full_count,retained_boundary_cells=len(boundary),
        excluded_cells=empty_count,planar_rectangles=rectangles,output_triangles=len(triangles),
        plane_height=float(points[0,0,2]),level_and_boundary_changed=False,
        proof='Disjoint integer cell ownership; unchanged fine boundary clipping; every fine perimeter vertex retained')
