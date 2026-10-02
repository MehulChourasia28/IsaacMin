"""Production schema adapters for bounded independent native source measurements."""
from pathlib import Path
import json
import shutil
import numpy as np

from ..io import atomic_json,sha256_file
from ..validation.native_ground import bind_native_ground,evidence_root,workspace_root,native_bounds,verify_records
from ..validation.global_intersections import file_record
from ..validation import native_source,native_segments,native_ground,streaming_columns,continuous_surface,evidence_closure


def _surface(mesh_path,ir_dir,output,transform,bound):
    root=evidence_root(output);path=root/'continuous_surface/surface/continuous_surface.json'
    if not path.is_file():
        path=Path(bound['identity_report']).parent/'closed_surface/continuous_surface.json'
        if not path.exists():
            continuous_surface.validate_continuous_surface(mesh_path,path.parent,source_ir=ir_dir,mesh_to_source=transform)
    report=json.loads(path.read_text())
    if report.get('validator_sha256')!=sha256_file(Path(continuous_surface.__file__)) or not any(item['sha256']==sha256_file(mesh_path) for item in report.get('input_files',[])):
        raise ValueError('Closed-surface evidence belongs to different geometry or code')
    verify_records(report['input_files'])
    for entry in report.get('files',[]):verify_records([dict(entry,path=str(path.parent/entry['path']))])
    for key in ('native_geometry_identity','native_orientation_query_evidence'):
        if key in report:verify_records([report[key]])
    global_record=report.get('global_self_intersection_test')
    if isinstance(global_record,dict) and 'evidence' in global_record:verify_records([global_record['evidence']])
    good=bool(report.get('status')=='pass' and report.get('scope')=='closed_initial_crop' and report.get('boundary_edges')==0 and report.get('total_signed_volume_m3',0)>0 and report.get('shell_orientation',{}).get('status')=='pass')
    return good,report,file_record(path,'independent_closed_surface')


def backend_evidence(bound,surface_record,*,query_report=None,column_report=None):
    producers=[file_record(module.__file__,role) for module,role in [(native_source,'native_source_classifier'),(native_segments,'native_segment_wrapper'),(native_ground,'native_OBJ_identity'),(streaming_columns,'all_triangle_column_solver'),(continuous_surface,'closed_surface_validator'),(evidence_closure,'recursive_evidence_verifier')]]
    producers.append(file_record(__file__,'production_source_adapter'))
    records=[file_record(bound['identity_report'],'native_ground_identity'),surface_record]
    if query_report is not None:records.append(file_record(query_report,'source_point_and_portal_queries'))
    if column_report is not None:records.append(file_record(column_report,'all_original_column_intersections'))
    return {'kind':'bounded_native_source_v1','producer_files':producers,'evidence_files':records}


def verify_backend_evidence(report):
    evidence=report.get('backend_evidence')
    if evidence is None:return False
    if evidence.get('kind')!='bounded_native_source_v1':raise ValueError('Unsupported independent source backend identity')
    if report.get('kind')=='IndependentSourceMeshValidation' and (report.get('classifier_kind')!='bounded_native_segments' or report.get('classifier_sha256')!=sha256_file(Path(native_source.__file__))):
        raise ValueError('Native source classifier identity changed')
    required={str(Path(module.__file__).resolve()) for module in (native_source,native_segments,native_ground,streaming_columns,continuous_surface,evidence_closure)}|{str(Path(__file__).resolve())}
    producers=evidence['producer_files']
    if {entry['path'] for entry in producers}!=required:raise ValueError('Incomplete bounded source producer closure')
    expected_roles={'native_ground_identity','independent_closed_surface'}
    if report.get('kind')=='IndependentSourceMeshValidation':
        expected_roles.add('source_point_and_portal_queries')
    elif report.get('kind')=='IndependentMeshConnectivityComparison':
        expected_roles.add('all_original_column_intersections')
    else:
        raise ValueError('Unsupported native source report kind')
    records=evidence['evidence_files']
    if len(records)!=len(expected_roles) or {entry['role'] for entry in records}!=expected_roles:
        raise ValueError('Incomplete bounded source measurement closure')
    verify_records([*producers,*evidence['evidence_files']])
    # Native build manifests recursively bind all original compile and runtime
    # dependencies; neither an edited JSON pass flag nor a copied path suffices.
    native_segments.segment_validator_files(workspace_root())
    for entry in evidence['evidence_files']:
        path=Path(entry['path']);data=json.loads(path.read_text())
        evidence_closure.verify_json_evidence(path)
        for key in ('input_files','producer_files'):
            verify_records(data.get(key,[]))
        for item in data.get('files',[]):
            item=dict(item);item['path']=str(Path(item['path']) if Path(item['path']).is_absolute() else path.parent/item['path']);verify_records([item])
        if entry['role']=='native_ground_identity' and data.get('status')!='pass':raise ValueError('Native OBJ correspondence did not pass')
    return True


def validate_source_mesh_native(ir_dir,mesh_path,output,mesh_to_source,sample_count,seed,support_manifest_path):
    from . import mesh_validation
    ir_dir,mesh_path,output=Path(ir_dir),Path(mesh_path),Path(output);output.mkdir(parents=True,exist_ok=True)
    bound=bind_native_ground(mesh_path,evidence_root(output));volume,surface,surface_record=_surface(mesh_path,ir_dir,output,mesh_to_source,bound)
    result=native_source.validate_native_source_samples(workspace_root(),ir_dir,bound['vertices_path'],bound['triangles_path'],output/'native_backend',mesh_to_source,support_manifest_path=support_manifest_path,sample_count=sample_count,seed=seed)
    shutil.copyfile(output/'native_backend/source_samples.npz',output/'source_mesh_samples.npz')
    with np.load(output/'source_mesh_samples.npz') as d:
        positions=d['source_xyz'];inside=d['mesh_inside'];expected=d['expected_inside'];strata=d['stratum'];failures=d['mismatch_indices'];portal_failed=np.flatnonzero(d['portal_crossings']|d['portal_inside'].any(axis=1)|d['portal_unresolved'])
        caps=[{'portal_sample':int(i),'source_xyz':d['portal_centres'][i].tolist(),'crossings':int(d['portal_crossings'][i]),'inside':d['portal_inside'][i].tolist(),'unresolved':bool(d['portal_unresolved'][i])} for i in portal_failed]
    report={'schema_version':1,'kind':'IndependentSourceMeshValidation','status':'pass' if result['status']=='pass' and volume else 'fail',
        'source_payload_evidence':result['source_payload_evidence'],'scope':result['scope'],'mesh_sha256':sha256_file(mesh_path),'source_ir_sha256':sha256_file(ir_dir/'world_ir.json'),'source_topology_sha256':sha256_file(ir_dir/'topology/source_topology_graph.json'),
        'validator_sha256':sha256_file(Path(mesh_validation.__file__)),'classifier_kind':'bounded_native_segments','classifier_sha256':sha256_file(Path(native_source.__file__)),'axis_ambiguous_diagonal_fallbacks':result['diagonal_fallbacks'],'mesh_to_source':np.asarray(mesh_to_source).tolist(),
        'toolchain':{'solver':'Bounded CGAL exact-predicate original-triangle segments; six opposing axes plus independent opposite diagonal ambiguity resolution'},
        'mesh':{'vertices':len(bound['vertices']),'triangles':len(bound['triangles']),'watertight':surface.get('boundary_edges')==0,'winding_consistent':surface.get('failures',{}).get('inconsistent_winding_edges')==0,'positive_closed_volume':volume,'signed_volume_m3':surface.get('total_signed_volume_m3'),'source_bounds_xyz':result['source_bounds']},
        'sample_count':len(positions),'minimum_sample_count':sample_count,'strata':result['strata'],'mismatch_count':len(failures),'unresolved_point_queries':result['unresolved_point_queries'],'structural_support':{**result['structural_support'],'inside_failures':sum(strata[i]=='source_structural_masonry' for i in failures)},'mismatches':[{'source_xyz':positions[i].tolist(),'stratum':str(strata[i]),'expected_inside':bool(expected[i]),'mesh_inside':bool(inside[i])} for i in failures[:500]],
        'portal_segment_count':result['portal_faces'],'portal_stencil_count':result['portal_stencils'],'portal_cap_intersections':caps,'seed':seed,'full_target_connectivity_graph':{'status':'not_run'},'isaac_contact':{'status':'not_run'},
        'backend_evidence':backend_evidence(bound,surface_record,query_report=output/'native_backend/source_samples.json'),
        'files':[{'path':'source_mesh_samples.npz','sha256':sha256_file(output/'source_mesh_samples.npz'),'bytes':(output/'source_mesh_samples.npz').stat().st_size}]}
    verify_backend_evidence(report);atomic_json(output/'source_mesh_validation.json',report);return report


class StreamedColumnMesh:
    """Small adapter retaining existing protected-voxel/graph/roof comparisons."""
    def __init__(self,mesh_path,ir_dir,output,transform):
        self.bound=bind_native_ground(mesh_path,evidence_root(output));self.vertices=self.bound['vertices'];self.faces=self.bound['triangles']
        self.bounds=native_bounds(self.vertices,transform)
        self.is_volume,self.surface,self.surface_record=_surface(mesh_path,ir_dir,output,transform,self.bound)
        with np.load(Path(ir_dir)/'natural_occupancy.npz') as data:self.minimum=data['min_xyz'];ny,self.nz,self.nx=data['occupancy'].shape
        data=streaming_columns.streamed_column_intersections(self.vertices,self.faces,transform,self.minimum[[0,2]],(self.nz,self.nx));self.metrics=data.pop('metrics')
        output=Path(output);output.mkdir(parents=True,exist_ok=True);self.raw_path=output/'all_original_column_hits.npz';np.savez_compressed(self.raw_path,**data)
        order=np.lexsort((data['source_y'],data['ray_index']));self.ray_index=data['ray_index'][order];self.y=data['source_y'][order];self.triangle=data['triangle_index'][order]
        self.tangencies=len(data['vertical_face_tangent_ray_triangle']);self.ray=self
        self.column_report=output/'all_original_column_hits.json';atomic_json(self.column_report,{'kind':'IndependentAllOriginalColumnIntersections','metrics':self.metrics,'tangent_ray_triangle_pairs':self.tangencies,'input_files':self.bound['report']['input_files'],'producer_files':[file_record(streaming_columns.__file__,'all_triangle_solver')],'files':[file_record(self.raw_path,'every_original_column_hit')]})
    def intersects_location(self,origins,directions,multiple_hits=True):
        origins=np.asarray(origins,float);directions=np.asarray(directions,float)
        if not multiple_hits or not np.array_equal(directions,np.tile([0.,1.,0.],(len(origins),1))):raise ValueError('Streamed graph adapter accepts only complete positive-source-Y column rays')
        grid=origins[:,[0,2]]-self.minimum[[0,2]]-.5;indices=np.rint(grid).astype(int)
        if not np.allclose(grid,indices,rtol=0,atol=1e-10) or np.any(indices<0) or np.any(indices>=[self.nx,self.nz]):raise ValueError('Source rays do not match original unit column centers')
        ids=indices[:,1]*self.nx+indices[:,0];rows=[];rays=[];triangles=[]
        for local,global_ray in enumerate(ids):
            start=np.searchsorted(self.ray_index,global_ray,side='left');stop=np.searchsorted(self.ray_index,global_ray,side='right');values=self.y[start:stop];valid=values>=origins[local,1]
            rows.extend(np.column_stack([np.full(valid.sum(),origins[local,0]),values[valid],np.full(valid.sum(),origins[local,2])]).tolist());rays.extend([local]*int(valid.sum()));triangles.extend(self.triangle[start:stop][valid].tolist())
        return np.asarray(rows,float).reshape(-1,3),np.asarray(rays,int),np.asarray(triangles,int)
    def evidence(self):return backend_evidence(self.bound,self.surface_record,column_report=self.column_report)
