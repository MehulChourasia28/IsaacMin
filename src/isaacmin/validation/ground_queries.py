"""Finite batched queries against exact native ground, with retained evidence."""
from pathlib import Path
import numpy as np
from ..io import atomic_json
from .native_ground import bind_native_ground,evidence_root,workspace_root,native_bounds
from .native_segments import query_native_segments
from .native_proximity import query_native_proximity
from .native_source import DIRECTIONS,classify_native_parity
from .global_intersections import file_record


class GroundQueries:
    def __init__(self,mesh_path,output):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.bound=bind_native_ground(mesh_path,evidence_root(self.output));self.bounds=native_bounds(self.bound['vertices'],np.eye(4));self.records=[];self.serial=0
    def _directory(self,kind):
        self.serial+=1;return self.output/('%03d_%s'%(self.serial,kind))
    def segments(self,starts,ends):
        starts,ends=np.asarray(starts,float).reshape(-1,3),np.asarray(ends,float).reshape(-1,3)
        if len(starts)!=len(ends):raise ValueError('Segment batch endpoint counts differ')
        if not len(starts):return np.empty(0,dtype=__import__('isaacmin.validation.native_segments',fromlist=['HIT_DTYPE']).HIT_DTYPE)
        chunks=[]
        for offset in range(0,len(starts),500_000):
            output=self._directory('segments');hits,report=query_native_segments(workspace_root(),self.bound['vertices_path'],self.bound['triangles_path'],np.stack([starts[offset:offset+500_000],ends[offset:offset+500_000]],axis=1),output)
            if not report['complete']:raise ValueError('Ground segment batch incomplete')
            hits=hits.copy();hits['query']+=offset;chunks.append(hits);self.records.append(file_record(output/'segment_queries.json','actual_segment_batch'))
        return np.concatenate(chunks)
    def first_hits(self,origins,directions,max_distances=None):
        origins,directions=np.asarray(origins,float).reshape(-1,3),np.asarray(directions,float).reshape(-1,3)
        if directions.shape!=origins.shape or not np.isfinite([origins,directions]).all() or np.any(np.linalg.norm(directions,axis=1)==0):raise ValueError('Finite nonzero rays required')
        directions=directions/np.linalg.norm(directions,axis=1)[:,None]
        if max_distances is None:
            distances=np.linalg.norm(origins-self.bounds.mean(axis=0),axis=1)+2*np.linalg.norm(np.diff(self.bounds,axis=0))+1
        else:distances=np.broadcast_to(np.asarray(max_distances,float),(len(origins),))
        if not np.isfinite(distances).all() or np.any(distances<=0):raise ValueError('Finite positive ray lengths required')
        hits=self.segments(origins,origins+directions*distances[:,None]);locations=np.full_like(origins,np.nan);normals=np.full_like(origins,np.nan);triangles=np.full(len(origins),-1,np.int64);observed=np.full(len(origins),np.inf)
        for record in hits:
            q=int(record['query']);point=record['first'];distance=float(np.dot(point-origins[q],directions[q]))
            if record['kind']:
                alternative=record['second'];second=float(np.dot(alternative-origins[q],directions[q]))
                if second<distance:point,distance=alternative,second
            if distance<observed[q] or (distance==observed[q] and int(record['triangle'])<triangles[q]):
                observed[q]=distance;locations[q]=point;triangles[q]=record['triangle'];normals[q]=record['normal']/np.linalg.norm(record['normal'])
        return locations,normals,triangles
    def contains(self,points,*,reject_ambiguous=False):
        points=np.asarray(points,float).reshape(-1,3)
        if not len(points):return np.empty(0,bool)
        origins=np.repeat(points,len(DIRECTIONS),axis=0);directions=np.tile(DIRECTIONS,(len(points),1))
        lengths=np.linalg.norm(origins-self.bounds.mean(axis=0),axis=1)+2*np.linalg.norm(np.diff(self.bounds,axis=0))+1
        hits=self.segments(origins,origins+directions*lengths[:,None]);inside,raw=classify_native_parity(hits,points)
        directory=self._directory('parity');directory.mkdir()
        np.savez_compressed(directory/'classifications.npz',points=points,inside=inside,**raw)
        self.records.append(file_record(directory/'classifications.npz','actual_point_classifications'))
        if reject_ambiguous:return inside|raw['unresolved']
        if raw['unresolved'].any():raise ValueError('Ground point parity remained ambiguous; no guessed air labels')
        return inside
    def closest(self,points):
        points=np.asarray(points,float).reshape(-1,3);results=[]
        for offset in range(0,len(points),250_000):
            output=self._directory('proximity');values,report=query_native_proximity(workspace_root(),self.bound['vertices_path'],self.bound['triangles_path'],points[offset:offset+250_000],output)
            results.append(values);self.records.append(file_record(output/'proximity_queries.json','actual_proximity_batch'))
        if not results:return np.empty((0,3)),np.empty(0),np.empty(0,int)
        values=np.concatenate(results);return values['point'],np.sqrt(values['distance2']),values['triangle'].astype(np.int64)
    def unobstructed(self,starts,ends,epsilon=1e-5):
        starts,ends=np.asarray(starts,float).reshape(-1,3),np.asarray(ends,float).reshape(-1,3);lengths=np.linalg.norm(ends-starts,axis=1)
        hits=self.segments(starts,ends);result=np.ones(len(starts),bool)
        for row in hits:
            i=int(row['query']);distance=min(np.linalg.norm(row['first']-starts[i]),np.linalg.norm(row['second']-starts[i]))
            if distance<lengths[i]-epsilon:result[i]=False
        return result
    def receipt(self):
        from . import native_ground,native_segments,native_proximity,native_source
        result={'kind':'IndependentNativeGroundQueryBatches','native_geometry':file_record(self.bound['identity_report'],'exact_OBJ_native_identity'),'batches':self.records,'producer_files':[file_record(p,'ground_query_producer') for p in [__file__,native_ground.__file__,native_segments.__file__,native_proximity.__file__,native_source.__file__]],'scope':'Actual batched final-ground queries; no rendered appearance qualification'}
        atomic_json(self.output/'ground_queries.json',result);return file_record(self.output/'ground_queries.json','independent_native_ground_queries')


class _FaceNormals:
    def __init__(self,mesh):self.mesh=mesh
    def __getitem__(self,indices):
        triangles=np.asarray(self.mesh.vertices[self.mesh.faces[indices]],float)
        normals=np.cross(triangles[...,1,:]-triangles[...,0,:],triangles[...,2,:]-triangles[...,0,:])
        return normals/np.linalg.norm(normals,axis=-1)[...,None]


class NativeGroundMesh:
    """Small read-only facade for batched ray consumers; no global Python BVH."""
    def __init__(self,path,output):
        self.queries=GroundQueries(path,output);self.vertices=self.queries.bound['vertices'];self.faces=self.queries.bound['triangles'];self.bounds=self.queries.bounds;self.ray=self;self.face_normals=_FaceNormals(self)
    def intersects_location(self,origins,directions,multiple_hits=True):
        origins,directions=np.asarray(origins,float).reshape(-1,3),np.asarray(directions,float).reshape(-1,3)
        if not len(origins):return np.empty((0,3)),np.empty(0,int),np.empty(0,int)
        if not multiple_hits:
            locations,_,triangles=self.queries.first_hits(origins,directions);ids=np.flatnonzero(triangles>=0);return locations[ids],ids,triangles[ids]
        directions=directions/np.linalg.norm(directions,axis=1)[:,None];lengths=np.linalg.norm(origins-self.bounds.mean(axis=0),axis=1)+2*np.linalg.norm(np.diff(self.bounds,axis=0))+1
        hits=self.queries.segments(origins,origins+directions*lengths[:,None])
        if np.any(hits['kind']):raise ValueError('Coplanar ray segment has no unique point intersection')
        return hits['first'],hits['query'].astype(int),hits['triangle'].astype(int)
    def contains(self,points):return self.queries.contains(points)
    def receipt(self):return self.queries.receipt()
