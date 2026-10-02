"""Independent source strata and full portal stencils using bounded native rays.

Geometry closure, global intersections and protected column interface distances
are separate required checks. This module never promotes a world on ray parity.
"""
from collections import Counter
from pathlib import Path
import numpy as np
from scipy import ndimage

from ..io import atomic_json
from ..volumes.source_evidence import verify_source_topology,recheck_source_topology
from ..volumes.support_validation import structural_support_for_validation
from .native_segments import query_native_segments
from .global_intersections import file_record

AXIS_DIRECTIONS=np.array([[1.,0,0],[-1.,0,0],[0,0,1.],[0,0,-1.],[0,1.,0],[0,-1.,0]])
DIAGONAL=np.array([1.,.3713906763541037,.5291134919573865]);DIAGONAL/=np.linalg.norm(DIAGONAL)
DIRECTIONS=np.vstack([AXIS_DIRECTIONS,DIAGONAL,-DIAGONAL])
COINCIDENCE_M=1e-7


def source_sample_plan(ir_dir, *, support_manifest_path=None, sample_count=10000, seed=1729):
    """Deterministic unchanged source strata plus all original portal samples."""
    if not 10000<=sample_count<=100000:raise ValueError('Source proof requires at least 10000 bounded deterministic samples')
    ir_dir=Path(ir_dir);graph,evidence=verify_source_topology(ir_dir)
    with np.load(ir_dir/'natural_occupancy.npz',allow_pickle=False) as data:occupancy,valid,minimum=data['occupancy'],data['validity'],data['min_xyz']
    with np.load(ir_dir/'topology/source_topology_labels.npz',allow_pickle=False) as data:covered=data['covered_air']
    natural=(occupancy==1)&valid;air=(occupancy==0)&valid
    structural,support=structural_support_for_validation(ir_dir,occupancy,support_manifest_path)
    solid=natural|structural;structure=ndimage.generate_binary_structure(3,1)
    deep_solid=ndimage.binary_erosion(natural,structure=structure,border_value=0);deep_air=ndimage.binary_erosion(air,structure=structure,border_value=0)
    roof=np.zeros_like(air);roof[1:]=solid[1:]&covered[:-1]
    floor=np.zeros_like(air);floor[:-1]=solid[:-1]&covered[1:]
    masks=[('solid_interior',deep_solid,True,sample_count//2),('air_interior',deep_air,False,sample_count//4),('covered_source_air',covered,False,sample_count//8),('retained_support_roof',roof,True,sample_count//16),('retained_support_floor',floor,True,sample_count//16)]
    rng=np.random.default_rng(seed);claimed=np.zeros_like(valid);positions=[];expected=[];strata=[]
    def select(mask,count):
        choices=np.flatnonzero(mask);chosen=rng.choice(choices,size=min(count,len(choices)),replace=False) if len(choices) else np.empty(0,int)
        return np.column_stack(np.unravel_index(chosen,mask.shape))
    def append(selected,label,inside):
        positions.extend((selected[:,[2,0,1]]+minimum+.5).tolist());expected.extend([inside]*len(selected));strata.extend([label]*len(selected))
    for label,mask,inside,count in masks:
        selected=select(mask&~claimed,count)
        if len(selected):claimed[selected[:,0],selected[:,1],selected[:,2]]=True
        append(selected,label,inside)
    if len(positions)<sample_count:
        selected=select((deep_solid|deep_air)&~claimed,sample_count-len(positions));positions.extend((selected[:,[2,0,1]]+minimum+.5).tolist());expected.extend(natural[selected[:,0],selected[:,1],selected[:,2]].tolist());strata.extend(['remaining_interior']*len(selected))
    append(np.argwhere(structural),'source_structural_masonry',True)
    portal_centres=[];portal_normals=[];portal_owner=[];portal_context=[];source_faces=[]
    def known_air(point):
        x,y,z=np.floor(point).astype(int)-minimum
        return bool(0<=y<air.shape[0] and 0<=z<air.shape[1] and 0<=x<air.shape[2] and air[y,z,x])
    for portal in graph['portals']:
        for face in portal['faces']:
            centre=np.asarray(face['center_xyz'],float);axis={'x':0,'y':1,'z':2}[face['normal_axis']];normal=np.zeros(3);normal[axis]=face['normal_sign'];tangents=[i for i in range(3) if i!=axis]
            positions.extend([(centre-.35*normal).tolist(),centre.tolist(),(centre+.35*normal).tolist()]);expected.extend([False]*3);strata.extend(['portal_aperture']*3)
            face_id=len(source_faces);source_faces.append({'portal_id':portal['id'],'source_face':face})
            uniform=all(known_air(centre+da*np.eye(3)[tangents[0]]+db*np.eye(3)[tangents[1]]+side*.35*normal) for da in (-1,0,1) for db in (-1,0,1) for side in (-1,1))
            for da in (-.4,-.2,0.,.2,.4):
                for db in (-.4,-.2,0.,.2,.4):
                    point=centre.copy();point[tangents]+=[da,db]
                    if not all(known_air(point+side*.35*normal) for side in (-1,0,1)):raise ValueError('Portal graph does not identify an entirely known-air source stencil')
                    portal_centres.append(point);portal_normals.append(normal);portal_owner.append(face_id);portal_context.append(uniform)
    return {'positions':np.asarray(positions,float),'expected':np.asarray(expected,bool),'strata':np.asarray(strata),
        'portal_centres':np.asarray(portal_centres,float).reshape(-1,3),'portal_normals':np.asarray(portal_normals,float).reshape(-1,3),
        'portal_owner':np.asarray(portal_owner,int),'portal_uniform_interior':np.asarray(portal_context,bool),'source_faces':source_faces,
        'source_evidence':evidence,'support':support,'source_minimum':minimum,'seed':seed,'sample_count':sample_count}


def grouped_crossings(hits,origins,directions):
    """Retain coplanar and mixed-orientation ambiguities; never vote them away."""
    count=len(origins);values=[[] for _ in range(count)];ambiguity=np.zeros(count,bool);surface_origin=np.zeros(count,bool)
    order=np.argsort(hits['query'],kind='stable');hits=hits[order]
    query_ids,starts,counts=np.unique(hits['query'],return_index=True,return_counts=True)
    for q,start,count in zip(query_ids,starts,counts):
        q=int(q);rows=hits[start:start+count];distance=(rows['first']-origins[q])@directions[q];end_distance=(rows['second']-origins[q])@directions[q]
        segment=rows['kind']==1
        ambiguity[q]=bool(np.any(segment));surface_origin[q]=bool(np.any(np.minimum(distance,end_distance)<=COINCIDENCE_M))
        order=np.argsort(distance);distance=distance[order];normals=rows['normal'][order]
        norm=np.linalg.norm(normals,axis=1);orientation=(normals@directions[q])/np.maximum(norm,1e-300)
        cuts=np.r_[0,np.flatnonzero(np.diff(distance)>COINCIDENCE_M)+1,len(distance)]
        for start,stop in zip(cuts[:-1],cuts[1:]):
            positive=np.any(orientation[start:stop]>1e-12);negative=np.any(orientation[start:stop]<-1e-12)
            if positive==negative:ambiguity[q]=True
            values[q].append(float(distance[start]))
    return values,ambiguity,surface_origin


def classify_native_parity(hits,points,directions=DIRECTIONS):
    points=np.asarray(points,float);directions=np.asarray(directions,float)
    origins=np.repeat(points,len(directions),axis=0);vectors=np.tile(directions,(len(points),1))
    crossings,ambiguous,on_surface=grouped_crossings(hits,origins,vectors)
    counts=np.asarray([len(x) for x in crossings],np.int32).reshape(len(points),len(directions))
    ambiguous=(ambiguous|on_surface).reshape(counts.shape);parity=counts%2==1
    axis_disagree=np.any(parity[:,:6]!=parity[:,:1],axis=1)
    fallback=axis_disagree|ambiguous[:,:6].any(axis=1)
    diagonal_bad=ambiguous[:,6:].any(axis=1)|(parity[:,6]!=parity[:,7])
    unresolved=fallback&diagonal_bad;inside=np.where(fallback,parity[:,6],parity[:,0])
    return inside,{'crossing_counts':counts,'ambiguous_rays':ambiguous,'axis_disagreement':axis_disagree,'diagonal_fallback':fallback,'unresolved':unresolved}


def validate_native_source_samples(workspace,ir_dir,vertices_path,triangles_path,output,mesh_to_source,*,support_manifest_path=None,sample_count=10000,seed=1729):
    output=Path(output);ir_dir=Path(ir_dir)
    if output.exists() and any(output.iterdir()):raise ValueError('Source evidence is immutable')
    output.mkdir(parents=True,exist_ok=True);transform=np.asarray(mesh_to_source,float)
    if transform.shape!=(4,4) or not np.isfinite(transform).all() or not np.array_equal(transform[3],[0,0,0,1]) or not np.allclose(transform[:3,:3].T@transform[:3,:3],np.eye(3),atol=1e-12):raise ValueError('Explicit rigid metre-preserving source transform required')
    inverse=np.linalg.inv(transform);plan=source_sample_plan(ir_dir,support_manifest_path=support_manifest_path,sample_count=sample_count,seed=seed)
    v=np.load(vertices_path,mmap_mode='r');bounds=np.array([[np.inf]*3,[-np.inf]*3])
    for start in range(0,len(v),131072):
        part=v[start:start+131072].astype(float)@transform[:3,:3].T+transform[:3,3];bounds[0]=np.minimum(bounds[0],part.min(axis=0));bounds[1]=np.maximum(bounds[1],part.max(axis=0))
    points=plan['positions'];reach=4*np.linalg.norm(bounds[1]-bounds[0])+2*np.linalg.norm(points-(bounds[0]+bounds[1])/2,axis=1).max()+1
    point_origins=np.repeat(points,len(DIRECTIONS),axis=0);point_dirs=np.tile(DIRECTIONS,(len(points),1));point_segments=np.stack([point_origins,point_origins+point_dirs*reach],axis=1)
    portal_origins=plan['portal_centres']-.35*plan['portal_normals'];portal_segments=np.stack([portal_origins,portal_origins+plan['portal_normals']*reach],axis=1)
    portal_points=np.stack([portal_origins,plan['portal_centres'],portal_origins+.7*plan['portal_normals']],axis=1).reshape(-1,3)
    diagonal_origins=np.repeat(portal_points,2,axis=0);diagonal_directions=np.tile(DIRECTIONS[6:],(len(portal_points),1))
    diagonal_segments=np.stack([diagonal_origins,diagonal_origins+diagonal_directions*reach],axis=1)
    segments=np.concatenate([point_segments,portal_segments,diagonal_segments]);mesh_segments=segments@inverse[:3,:3].T+inverse[:3,3]
    hits,query_report=query_native_segments(workspace,vertices_path,triangles_path,mesh_segments,output/'native_queries')
    source_hits=hits.copy()
    for field in ('first','second'):source_hits[field]=hits[field]@transform[:3,:3].T+transform[:3,3]
    source_hits['normal']=hits['normal']@transform[:3,:3].T
    point_hit=source_hits[source_hits['query']<len(point_segments)]
    inside,evidence=classify_native_parity(point_hit,points)
    mismatch=np.flatnonzero(inside!=plan['expected']);unresolved=np.flatnonzero(evidence['unresolved'])
    portal_stop=len(point_segments)+len(portal_segments)
    portal_hit=source_hits[(source_hits['query']>=len(point_segments))&(source_hits['query']<portal_stop)].copy();portal_hit['query']-=len(point_segments)
    crossings,ambiguous,on_surface=grouped_crossings(portal_hit,portal_origins,plan['portal_normals'])
    portal_crossings=np.zeros(len(portal_origins),np.int32);portal_inside=np.zeros((len(portal_origins),3),bool)
    for index,values in enumerate(crossings):
        values=np.asarray(values);portal_crossings[index]=np.count_nonzero((values>COINCIDENCE_M)&(values<.7-COINCIDENCE_M));portal_inside[index]=[np.count_nonzero(values>d+COINCIDENCE_M)%2 for d in (0.,.35,.7)]
    # A coplanar overlap in the actual aperture is an obstruction even when an
    # endpoint-only count would happen to miss its interior.
    for hit in portal_hit[portal_hit['kind']==1]:
        q=int(hit['query']);ends=np.array([hit['first'],hit['second']]);distances=(ends-portal_origins[q])@plan['portal_normals'][q]
        if distances.max()>COINCIDENCE_M and distances.min()<.7-COINCIDENCE_M:portal_crossings[q]+=1
    diagonal_hit=source_hits[source_hits['query']>=portal_stop].copy();diagonal_hit['query']-=portal_stop
    diagonal_crossings,diagonal_ambiguous,diagonal_surface=grouped_crossings(diagonal_hit,diagonal_origins,diagonal_directions)
    diagonal_parity=(np.asarray([len(x) for x in diagonal_crossings])%2==1).reshape(-1,3,2)
    diagonal_bad=(diagonal_ambiguous|diagonal_surface).reshape(-1,3,2).any(axis=(1,2))|np.any(diagonal_parity[:,:,0]!=diagonal_parity[:,:,1],axis=1)
    normal_ambiguous=ambiguous|on_surface
    normal_disagree=np.any(portal_inside!=diagonal_parity[:,:,0],axis=1)&~normal_ambiguous
    portal_unresolved=diagonal_bad|normal_disagree
    portal_inside[normal_ambiguous]=diagonal_parity[normal_ambiguous,:,0]
    raw=output/'source_samples.npz';np.savez_compressed(raw,source_xyz=points,expected_inside=plan['expected'],mesh_inside=inside,stratum=plan['strata'],mismatch_indices=mismatch,**evidence,
        portal_centres=plan['portal_centres'],portal_normals=plan['portal_normals'],portal_owner=plan['portal_owner'],portal_uniform_interior=plan['portal_uniform_interior'],portal_inside=portal_inside,portal_crossings=portal_crossings,portal_unresolved=portal_unresolved,portal_diagonal_parity=diagonal_parity,portal_diagonal_ambiguous=diagonal_bad,portal_normal_ambiguous=normal_ambiguous)
    recheck_source_topology(ir_dir,plan['source_evidence'])
    complete=query_report['complete'];point_pass=complete and len(points)>=sample_count and not len(mismatch) and not len(unresolved)
    portal_pass=complete and not portal_crossings.any() and not portal_inside.any() and not portal_unresolved.any()
    from . import native_segments
    from ..volumes import source_evidence,support_validation
    report={'schema_version':1,'kind':'IndependentNativeSourceSampleComparison','status':'pass' if point_pass and portal_pass else 'fail',
        'point_parity_status':'pass' if point_pass else 'fail','portal_aperture_status':'pass' if portal_pass else 'fail','minimum_sample_count':sample_count,'sample_count':len(points),'strata':dict(Counter(plan['strata'])),'seed':seed,
        'point_mismatches':len(mismatch),'unresolved_point_queries':len(unresolved),'diagonal_fallbacks':int(evidence['diagonal_fallback'].sum()),
        'portal_faces':len(plan['source_faces']),'portal_stencils':len(portal_origins),'portal_crossing_samples':int(np.count_nonzero(portal_crossings)),'portal_inside_samples':int(portal_inside.any(axis=1).sum()),'portal_unresolved_samples':int(portal_unresolved.sum()),'portal_normal_ambiguity_diagonal_fallbacks':int(normal_ambiguous.sum()),
        'source_payload_evidence':plan['source_evidence'],'structural_support':plan['support'],'mesh_to_source':transform.tolist(),'source_bounds':bounds.tolist(),
        'numerical_coincidence_m':COINCIDENCE_M,'source_faces':plan['source_faces'],'input_files':[file_record(vertices_path,'candidate_vertices'),file_record(triangles_path,'candidate_triangles'),file_record(ir_dir/'world_ir.json','source_ir')],
        'producer_files':[file_record(path,role) for path,role in [(Path(__file__),'independent_source_ray_validator'),(native_segments.__file__,'native_segment_wrapper'),(source_evidence.__file__,'source_payload_verifier'),(support_validation.__file__,'source_structural_verifier')]],
        'files':[file_record(raw,'source_sample_measurements'),file_record(output/'native_queries/segment_queries.json','all_native_query_evidence')],
        'scope':'Six-axis source point parity with opposite independent diagonal fallback; every source portal 5x5 aperture stencil. No geometry closure, roof-thickness, global intersections, nearest-surface clearance or Isaac qualification follows from this component alone.'}
    atomic_json(output/'source_samples.json',report);return report
