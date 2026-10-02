"""Native Blender final-ground rays without Python objects for every face."""
import hashlib
import time
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree


class NativeGroundQuery:
    def __init__(self, terrain, depsgraph, *, expected_vertices=None, expected_triangles=None):
        if terrain.type!='MESH' or terrain.modifiers:
            raise RuntimeError('Placement requires the authoritative applied native ground mesh')
        mesh=terrain.data
        if expected_vertices is not None and len(mesh.vertices)!=expected_vertices:
            raise RuntimeError('Native ground vertex count differs from resource-admission inventory')
        counts=np.empty(len(mesh.polygons),np.int32);mesh.polygons.foreach_get('loop_total',counts)
        triangles=int(np.maximum(counts-2,0).sum());del counts
        if expected_triangles is not None and triangles!=expected_triangles:
            raise RuntimeError('Native ground triangle count differs from resource-admission inventory')
        self.matrix=terrain.matrix_world.copy();self.inverse=self.matrix.inverted()
        self.normal_matrix=self.matrix.to_3x3().inverted().transposed()
        transform=np.asarray(self.matrix,dtype=np.float64)
        coordinates=np.empty((len(mesh.vertices),3),np.float32)
        mesh.vertices.foreach_get('co',coordinates.ravel())
        digest=hashlib.sha256();lo=float('inf');hi=-float('inf')
        for start in range(0,len(coordinates),250000):
            world=coordinates[start:start+250000].astype(np.float64)@transform[:3,:3].T+transform[:3,3]
            if not np.isfinite(world).all():raise RuntimeError('Nonfinite final-ground coordinates')
            digest.update(np.ascontiguousarray(world,dtype='<f4').tobytes())
            lo=min(lo,float(world[:,2].min()));hi=max(hi,float(world[:,2].max()))
        del coordinates
        self.top=hi+1.;self.bottom=lo-1.;self.vertices_sha256=digest.hexdigest()
        started=time.monotonic()
        # Native C++ copies numeric coordinate/triangle buffers directly. Building
        # Python Vector/tuple lists for tens of millions of faces is unnecessary.
        self.bvh=BVHTree.FromObject(terrain,depsgraph,deform=False,cage=False,epsilon=0.0)
        if self.bvh is None:raise RuntimeError('Native final-ground BVH construction failed')
        self.evidence={'method':'Blender_BVHTree.FromObject_on_applied_authoritative_mesh',
            'vertices':len(mesh.vertices),'triangles':triangles,'build_seconds':time.monotonic()-started,
            'geometry_reduction':False,'ray_coordinate_space':'object; inverse world ray and inverse-transpose normal transform',
            'vertices_sha256':self.vertices_sha256,
            'vertex_hash_semantics':'ordered world coordinates rounded to little-endian float32; complete serialized geometry additionally bound by scene SHA',
            'numeric_hash_chunk_vertices':250000,'python_per_vertex_or_face_objects':False}

    def support(self,x,y,hint=None):
        z=self.top if hint is None else hint+.15
        if z<=self.bottom:return None
        origin=self.inverse@Vector((x,y,z));end=self.inverse@Vector((x,y,self.bottom))
        direction=end-origin;distance=direction.length
        if not distance:return None
        co,normal,face,_=self.bvh.ray_cast(origin,direction/distance,distance)
        if co is None:return None
        co=self.matrix@co;normal=(self.normal_matrix@normal).normalized()
        if normal.z<=0:return None
        return {'z':co.z,'normal':list(normal),'surface_id':'Terrain_FinalGround','mesh_sha256':self.vertices_sha256}
