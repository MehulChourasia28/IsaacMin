"""Measured view coverage from raw Isaac labels and native USD material faces.

Pose intent is never observed coverage. This reports presence and close-view
sampling only: neither proves an acceptable material or a realistic specimen.
"""
from pathlib import Path
from collections import Counter
import json
import math
import numpy as np
from .network import ServiceError,digest,atomic_json,utcnow
from .usd_provenance import verify_scene_closure,captured_closure_hash,validated_native_renderer

METHOD='native_segmentation_and_material_face_rays_v1'


def _bound(path):
    path=Path(path).resolve();return {'path':str(path),'sha256':digest(path)}


def _verify(record):
    path=Path(record['path'])
    if digest(path)!=record['sha256']:raise ServiceError('changed_visibility_evidence','Measured visibility evidence bytes changed')
    return path


def _pixels(mask,depth,fx,fy,cx,cy):
    y,x=np.nonzero(mask)
    if not len(x):return None
    z=depth[y,x];ranges=z*np.sqrt(1+((x+.5-cx)/fx)**2+((y+.5-cy)/fy)**2)
    ranges=ranges[np.isfinite(ranges)&(ranges>0)]
    return {'pixels':int(len(x)),'bbox_pixels':[int(x.min()),int(y.min()),int(x.max())+1,int(y.max())+1],
            'range_median_m':float(np.median(ranges)) if len(ranges) else None,
            'range_min_m':float(ranges.min()) if len(ranges) else None}


def _close_pixels(measured,thresholds):
    if not measured or measured['range_median_m'] is None:return False
    l,t,r,b=measured['bbox_pixels']
    return bool(measured['pixels']>=thresholds['asset_min_pixels'] and min(r-l,b-t)>=thresholds['asset_min_short_axis_pixels']
                and max(r-l,b-t)>=thresholds['asset_min_long_axis_pixels'] and measured['range_median_m']<=thresholds['asset_max_range_m'])


def _proofs(report):
    for field in ('collector','collector_dependencies','capture_results','source_manifests'):
        value=report.get(field,[])
        if isinstance(value,dict):value=[value]
        for item in value:_verify(item)
    _verify(report['scene_dependencies']['manifest'])
    for item in report['scene_dependencies']['files']:_verify(item)
    for frame in report['frames']:
        for field in ('rgb','depth','segmentation'):_verify(frame[field])
    if report.get('native_ground_queries'):
        from isaacmin.validation.evidence_closure import verify_json_evidence
        verify_json_evidence(_verify(report['native_ground_queries']))
        verify_json_evidence(_verify(report['native_USD_ground_identity']))
    return report


def validate_view_coverage(path,frames):
    """Join only exact captured-frame identities; recheck every measurement input."""
    path=Path(path).resolve();report=_proofs(json.loads(path.read_text()))
    if report.get('schema_version')!=1 or report.get('method')!=METHOD or report.get('status')!='measured':
        raise ServiceError('invalid_visibility_report','Coverage needs native segmentation and face-ray measurements')
    expected={(f['capture_result']['sha256'],f['frame'],f['rgb']['sha256']):f for f in frames}
    joined={}
    for measurement in report['frames']:
        key=(measurement['capture_result_sha256'],measurement['frame'],measurement['rgb']['sha256'])
        if key not in expected:continue
        if key in joined:raise ServiceError('duplicate_visibility_frame','An actual captured frame was measured more than once')
        target=expected[key]
        if measurement['scene_dependencies_sha256']!=target['scene_dependencies_sha256'] or measurement['depth']['sha256']!=target['depth']['sha256']:
            raise ServiceError('visibility_scene_identity','Measured visibility uses different native content or depth')
        # Derive aggregate lists from actual measurements, not caller labels.
        assets={a['family'] for a in measurement['assets'] if a.get('close_view') and a.get('identity_source')=='native_instance_attribute_and_placement_manifest'}
        materials={m['family'] for m in measurement['materials'] if m.get('close_view') and m.get('method') in ('native_ground_face_binding_and_matching_sensor_depth','native_source_weight_primvars_and_matching_sensor_depth')}
        features={f['feature_id'] for f in measurement.get('features',[]) if f.get('status')=='observed' and f.get('method') in ('native_water_segmentation','source_feature_visibility_geometry_v1')}
        joined[key]={'features':sorted(features),'material_families':sorted(materials),'asset_families':sorted(assets),
                     'measurement_index':measurement['frame'],'measurement_report':_bound(path)}
    return _bound(path),joined


def collect_view_coverage(workspace,scene,capture_results,output_path,*,ecology_manifest=None,feature_evidence=None,grid_stride=12,frame_kinds=('static',)):
    """Read-only post-render measurement; no Isaac launch or provider request.

    Native mesh instances are matched to placement asset IDs. Ground material
    bindings are sampled only where the raw sensor labels ground and agrees with
    independent native triangle rays. Features require separate geometric proof;
    supplied pose.features/surface_scope never enter this collector.
    """
    from pxr import Usd,UsdGeom,UsdShade
    import trimesh
    workspace=Path(workspace).resolve();scene=Path(scene).resolve();output_path=Path(output_path).resolve()
    if output_path.exists():raise ServiceError('immutable_visibility_report','Use a new path for changed captures')
    if type(grid_stride)!=int or grid_stride<1:raise ValueError('Positive integer stride required')
    closure=verify_scene_closure(scene,digest(scene.parent/'native_dependency_closure.json'))
    stage=Usd.Stage.Open(str(scene));xf=UsdGeom.XformCache()
    if str(UsdGeom.GetStageUpAxis(stage))!='Z' or UsdGeom.GetStageMetersPerUnit(stage)!=1:
        raise ServiceError('visibility_coordinate_frame','Native metric Z-up stage required')
    requests={};manifests=[]
    if ecology_manifest:
        placements=json.loads(Path(ecology_manifest).read_text());manifests.append(_bound(ecology_manifest))
        requests={str(r['id']):r for r in placements['exporter_assets']}
    # Use actual material asset IDs as family names; do not invent ecological labels.
    ground_parts=[];face_families=[];ground_paths=[];weight_parts=[]
    from isaacmin.validation.native_ground import has_native_ground
    native_support=has_native_ground(scene.parent/'final_ground.obj')
    if native_support:
        from .native_support import exported_native_ground
        ground,_,native_prim=exported_native_ground(stage,scene.parent/'final_ground.obj',
            output_path.parent/'coverage_native_support')
    from isaacmin.assembly.material_assignment import REQUIRED
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh) or 'Terrain_FinalGround' not in str(prim.GetPath()):continue
        mesh=UsdGeom.Mesh(prim);counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get(),int)
        if not np.all(counts==3):raise ServiceError('visibility_ground_topology','Final native ground must be triangulated')
        if native_support:
            vertices=ground.vertices
            indices=ground.faces
        else:
            matrix=np.asarray(xf.GetLocalToWorldTransform(prim),float);vertices=np.asarray(mesh.GetPointsAttr().Get(),float)
            vertices=vertices@matrix[:3,:3]+matrix[3,:3];indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),int).reshape(-1,3)
        binding=UsdShade.MaterialBindingAPI(prim);material,_=binding.ComputeBoundMaterial()
        names=np.full(len(counts),str(material.GetPrim().GetName()) if material else 'UNBOUND',object)
        for subset in binding.GetMaterialBindSubsets():
            mat,_=UsdShade.MaterialBindingAPI(subset.GetPrim()).ComputeBoundMaterial()
            names[np.asarray(subset.GetIndicesAttr().Get(),int)]=str(mat.GetPrim().GetName()) if mat else 'UNBOUND'
        if native_support:
            face_families=names
        else:
            ground_parts.append(trimesh.Trimesh(vertices,indices,process=False));face_families.extend(names)
        ground_paths.append(str(prim.GetPath()))
        primvars=UsdGeom.PrimvarsAPI(prim);three=primvars.GetPrimvar('IsaacMinMaterialWeights012');fourth=primvars.GetPrimvar('IsaacMinMaterialWeight3')
        if three or fourth:
            orders=[json.loads(a.Get()) for a in prim.GetAttributes() if str(a.GetName()).endswith('isaacmin_material_weight_order')]
            explicit=prim.GetAttribute('isaacmin:materialWeightOrder')
            if explicit and explicit.HasValue():orders.append(list(explicit.Get()))
            if not orders or any(order!=list(REQUIRED) for order in orders) or not three or not fourth or three.GetInterpolation()!='vertex' or fourth.GetInterpolation()!='vertex':
                raise ServiceError('native_weight_semantics','Actual baked-family weight order and vertex interpolation must be explicit')
            weights=np.column_stack((np.asarray(three.ComputeFlattened(),np.float32),np.asarray(fourth.ComputeFlattened(),np.float32)))
            if weights.shape!=(len(vertices),4) or not np.isfinite(weights).all() or np.any(weights<0) or not np.allclose(weights.sum(axis=1),1,atol=1e-5):
                raise ServiceError('native_weight_values','Actual material-family contributions must form a finite normalized partition')
            weight_parts.append(weights)
        else:weight_parts.append(np.full((len(vertices),4),np.nan,np.float32))
    if not ground_paths:raise ServiceError('visibility_ground_missing','Native final ground required')
    if not native_support:
        ground=trimesh.util.concatenate(ground_parts)
    face_families=np.asarray(face_families,object)
    ground_weights=weight_parts[0] if len(weight_parts)==1 else np.concatenate(weight_parts)
    thresholds={'asset_min_pixels':256,'asset_min_short_axis_pixels':32,'asset_min_long_axis_pixels':96,'asset_max_range_m':5.,
                'ground_max_range_m':3.,'ground_min_ray_samples':16,'native_sensor_depth_tolerance_m':.02,'grid_stride_pixels':grid_stride,
                'source_material_min_contribution':.10}
    feature_records={}
    if feature_evidence:
        proof=json.loads(Path(feature_evidence).read_text());manifests.append(_bound(feature_evidence))
        if proof.get('method')!='source_feature_visibility_geometry_v1':raise ServiceError('feature_visibility_method','Independent source/native feature visibility geometry required')
        for bound in proof.get('input_files',[]):_verify(bound);manifests.append(bound)
        if not proof.get('input_files'):raise ServiceError('feature_visibility_inputs','Feature geometry inputs must be bound')
        for row in proof['frames']:
            key=(row['capture_result_sha256'],row['frame'],row['rgb_sha256'])
            if key in feature_records:raise ServiceError('duplicate_feature_visibility','Duplicate source feature measurement')
            feature_records[key]=row
    # Batch the unchanged per-frame rays before querying the full native mesh.
    # One bounded native acceleration structure serves many held-out views;
    # no sampled points, labels or depth tolerances are removed.
    native_hits={}
    if native_support:
        starts=[];directions_all=[];segments=[];offset=0
        for result in capture_results:
            result=Path(result).resolve();capture=json.loads(result.read_text())
            for frame in capture['frames']:
                if frame.get('pose',{}).get('kind') not in frame_kinds:continue
                ids=np.squeeze(np.load(result.parent/frame['instance_segmentation'],allow_pickle=False))
                depth=np.squeeze(np.load(result.parent/frame['depth'],allow_pickle=False))
                if (digest(result.parent/frame['instance_segmentation'])!=frame['instance_segmentation_sha256']
                        or digest(result.parent/frame['depth'])!=frame['depth_sha256']):
                    raise ServiceError('changed_visibility_sensor','Raw batch input changed')
                ground_ids=[int(key) for key,path in frame['instance_id_to_prim_path'].items() if path in ground_paths]
                h,w=ids.shape;yy,xx=np.mgrid[grid_stride//2:h:grid_stride,grid_stride//2:w:grid_stride];x=xx.ravel();y=yy.ravel()
                selected=np.isin(ids[y,x],ground_ids)&np.isfinite(depth[y,x])&(depth[y,x]>0)
                x,y=x[selected],y[selected]
                fx,fy,cx,cy=(float(frame[k]) for k in ('fx_pixels','fy_pixels','cx_pixels','cy_pixels'))
                camera=np.asarray(frame['camera_world_matrix_row_vectors'],float)
                local=np.column_stack(((x+.5-cx)/fx,(cy-y-.5)/fy,-np.ones(len(x))))
                directions=local@camera[:3,:3];directions/=np.linalg.norm(local,axis=1)[:,None]
                starts.append(np.repeat(camera[3,:3][None,:],len(x),axis=0));directions_all.append(directions)
                segments.append((str(result),frame['frame'],offset,offset+len(x)));offset+=len(x)
        if offset:
            locations,_,triangles=ground.queries.first_hits(np.concatenate(starts),np.concatenate(directions_all))
            for file,index,begin,end in segments:
                faces=triangles[begin:end];rays=np.flatnonzero(faces>=0)
                native_hits[file,index]=(locations[begin:end][rays],rays,faces[rays])
        else:
            for file,index,_,_ in segments:native_hits[file,index]=(np.empty((0,3)),np.array([],int),np.array([],int))
        del starts,directions_all
    result_frames=[];capture_files=[]
    for result in capture_results:
        result=Path(result).resolve();capture=json.loads(result.read_text());capture_files.append(_bound(result))
        validated_native_renderer(capture)
        if capture['scene_sha256']!=digest(scene) or captured_closure_hash(capture)!=closure['manifest']['sha256']:
            raise ServiceError('visibility_capture_identity','Actual capture does not match inspected native USD closure')
        for frame in capture['frames']:
            if frame.get('pose',{}).get('kind') not in frame_kinds:continue
            bounds={}
            for name,key in [('rgb','rgb'),('depth','depth'),('segmentation','instance_segmentation')]:
                if key not in frame:raise ServiceError('missing_visibility_sensor','Actual instance segmentation is required for observed coverage')
                bounds[name]=_bound(result.parent/frame[key])
                if bounds[name]['sha256']!=frame[key+'_sha256']:raise ServiceError('changed_visibility_sensor','Raw sensor capture changed')
            ids=np.squeeze(np.load(bounds['segmentation']['path'],allow_pickle=False));depth=np.squeeze(np.load(bounds['depth']['path'],allow_pickle=False))
            if ids.ndim!=2 or depth.shape!=ids.shape:raise ServiceError('visibility_sensor_shape','Aligned two-dimensional depth and native labels required')
            if frame.get('depth_semantics') not in ('distance_to_image_plane','axial_metres','distance_to_image_plane_m'):
                raise ServiceError('visibility_depth_convention','Actual axial metric depth convention required')
            h,w=ids.shape;fx,fy,cx,cy=(float(frame[k]) for k in ('fx_pixels','fy_pixels','cx_pixels','cy_pixels'))
            camera=np.asarray(frame['camera_world_matrix_row_vectors'],float);origin=camera[3,:3]
            by_instance={};ground_ids=[];water_ids=[];unknown=[]
            for label,path in frame['instance_id_to_prim_path'].items():
                if not isinstance(path,str):raise ServiceError('visibility_label_schema','Native label path must be explicit string')
                if path in ground_paths:ground_ids.append(int(label));continue
                if 'SourceSurfaceWater' in path:water_ids.append(int(label));continue
                prim=stage.GetPrimAtPath(path)
                while prim and not prim.GetAttribute('isaacmin:isaacmin_instance_id').Get():prim=prim.GetParent()
                identity=str(prim.GetAttribute('isaacmin:isaacmin_instance_id').Get()) if prim else None
                if identity in requests:by_instance.setdefault(identity,[]).append(int(label))
                elif np.any(ids==int(label)):unknown.append({'label':int(label),'path':path})
            assets=[]
            for identity,labels in by_instance.items():
                measured=_pixels(np.isin(ids,labels),depth,fx,fy,cx,cy)
                if measured:assets.append({'instance_id':identity,'family':requests[identity]['asset_id'],'identity_source':'native_instance_attribute_and_placement_manifest',
                                           **measured,'close_view':_close_pixels(measured,thresholds)})
            yy,xx=np.mgrid[grid_stride//2:h:grid_stride,grid_stride//2:w:grid_stride];x=xx.ravel();y=yy.ravel()
            select=np.isin(ids[y,x],ground_ids)&np.isfinite(depth[y,x])&(depth[y,x]>0)
            x,y=x[select],y[select];local=np.column_stack(((x+.5-cx)/fx,(cy-y-.5)/fy,-np.ones(len(x))))
            factors=np.linalg.norm(local,axis=1);directions=local@camera[:3,:3];directions/=factors[:,None]
            if native_support:
                hits,rays,triangles=native_hits[str(result),frame['frame']]
            else:
                hits,rays,triangles=ground.ray.intersects_location(np.repeat(origin[None,:],len(x),axis=0),directions,multiple_hits=False) if len(x) else (np.empty((0,3)),np.array([],int),np.array([],int))
            distances=np.linalg.norm(hits-origin,axis=1);errors=np.abs(distances-depth[y[rays],x[rays]]*factors[rays]);agree=errors<=thresholds['native_sensor_depth_tolerance_m']
            materials=[]
            for family in sorted(set(face_families[triangles].tolist())):
                selected=(face_families[triangles]==family)&agree;near=selected&(distances<=thresholds['ground_max_range_m'])
                materials.append({'family':family,'method':'native_ground_face_binding_and_matching_sensor_depth','samples':int(selected.sum()),'near_samples':int(near.sum()),
                                  'close_view':bool(near.sum()>=thresholds['ground_min_ray_samples']),'range_min_m':float(distances[selected].min()) if selected.any() else None,
                                  'sampled_native_face_ids':sorted(set(map(int,triangles[selected]))),'range_max_near_m':float(distances[near].max()) if near.any() else None})
            if len(hits):
                bary=trimesh.triangles.points_to_barycentric(ground.vertices[ground.faces[triangles]],hits)
                contributions=np.einsum('ij,ijk->ik',bary,ground_weights[ground.faces[triangles]])
                for index,family in enumerate(REQUIRED):
                    selected=agree&np.isfinite(contributions[:,index])&(contributions[:,index]>=thresholds['source_material_min_contribution'])
                    near=selected&(distances<=thresholds['ground_max_range_m'])
                    if selected.any():materials.append({'family':family,'method':'native_source_weight_primvars_and_matching_sensor_depth',
                        'samples':int(selected.sum()),'near_samples':int(near.sum()),'close_view':bool(near.sum()>=thresholds['ground_min_ray_samples']),
                        'minimum_contribution_required':thresholds['source_material_min_contribution'],
                        'minimum_observed_contribution':float(contributions[selected,index].min()),
                        'maximum_observed_contribution':float(contributions[selected,index].max()),
                        'sampled_native_face_ids':sorted(set(map(int,triangles[selected]))),
                        'qualification':'visible texture contribution only; no ingredient-list inference or appearance pass'})
            features=[]
            water=_pixels(np.isin(ids,water_ids),depth,fx,fy,cx,cy)
            if water and water['pixels']>=256:features.append({'feature_id':'surface_water','status':'observed','method':'native_water_segmentation',**water})
            feature=feature_records.get((capture_files[-1]['sha256'],frame['frame'],bounds['rgb']['sha256']))
            if feature:
                if feature['scene_dependencies_sha256']!=closure['manifest']['sha256']:raise ServiceError('feature_scene_identity','Feature evidence uses different native content')
                for item in feature['features']:
                    if item.get('status')=='observed' and item.get('method')=='source_feature_visibility_geometry_v1':features.append(item)
            result_frames.append({'capture_result_sha256':capture_files[-1]['sha256'],'frame':frame['frame'],**bounds,
                                  'scene_dependencies_sha256':closure['manifest']['sha256'],'assets':assets,'materials':materials,'features':features,
                                  'unknown_visible_labels':unknown,'ground_rays':len(x),'ground_native_hits':len(rays),'ground_sensor_agree':int(agree.sum()),
                                  'ground_sensor_error_max_m':float(errors.max()) if len(errors) else None,
                                  'frame_kind':frame.get('pose',{}).get('kind'),'planned_pose_metadata_not_used_for_observation':True})
    verify_scene_closure(scene,closure['manifest']['sha256'])
    used_materials=set(face_families.tolist())
    for index,family in enumerate(REQUIRED):
        if np.any(np.isfinite(ground_weights[:,index])&(ground_weights[:,index]>=thresholds['source_material_min_contribution'])):used_materials.add(family)
    report={'schema_version':1,'created_at_utc':utcnow(),'status':'measured','method':METHOD,'collector':_bound(__file__),
            'collector_dependencies':[_bound(Path(__file__).with_name('usd_provenance.py'))],
            'scene_dependencies':closure,'capture_results':capture_files,'source_manifests':manifests,'thresholds':thresholds,'frames':result_frames,
            'used_material_families':sorted(used_materials),'used_asset_families':sorted({r['asset_id'] for r in requests.values()}),
            'selected_frame_kinds':list(frame_kinds),
            'inventory_scope':'All native final-ground face bindings and all declared exported-instance asset identities, including families absent from every captured frame. Source scan families additionally require native vertex contributions>=10percent.',
            'qualification':'visibility_and_close_view_sampling_only; target appearance and readability remain unqualified',
            'limitations':['Sparse ground ray sampling can miss a small material patch; missing evidence remains a gap',
                           'Close-view pixel/range thresholds are recorded engineering defaults, not species identification or realism scores',
                           'Feature inventory is never inferred from pose intent; cave/surface scope needs independent source-feature geometry evidence']}
    if native_support:
        report['native_ground_queries']=ground.receipt()
        report['native_USD_ground_identity']=ground.usd_identity
        report['collector_dependencies'].append(_bound(Path(__file__).with_name('native_support.py')))
    atomic_json(output_path,report);return report
