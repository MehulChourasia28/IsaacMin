"""Independent exported-instance ecology and root contact measurements.

Run in the pinned OpenUSD/NumPy environment after releasing heavy builders. Source
fields are recomputed from actual IR, and support comes from native USD triangles;
the generator's contact/rule success flags are never used as measurements.
"""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib
import json
import math
import numpy as np
import trimesh
from scipy.spatial import cKDTree

from isaacmin.assembly.ecology import RULES
from .network import ServiceError,digest,atomic_json,utcnow
from .usd_inspection import validate_contact_report
from .usd_provenance import verify_scene_closure


def _json(path):return json.loads(Path(path).read_text())


def _source_fields(ir,runoff_path,route_path):
    meta=_json(ir/'world_ir.json')
    with np.load(ir/'terrain_surface.npz',allow_pickle=False) as archive:data={k:archive[k] for k in archive.files}
    shape=data['height'].shape;leaf=np.zeros(shape,float);roots=set();chunks=[]
    xmin,zmin=map(int,data['min_xz'])
    for path in sorted((ir/'terrain_volume').glob('*.npz')):
        chunks.append({'path':str(path),'sha256':digest(path)})
        with np.load(path,allow_pickle=False) as c:
            palette=json.loads(str(c['palette_json']));cx,_,cz=map(int,c['min_xyz'])
            ids=[i for i,p in enumerate(palette) if p['Name'].endswith('_leaves')]
            if ids:leaf[cz-zmin:cz-zmin+16,cx-xmin:cx-xmin+16]=np.any(np.isin(c['block_id'],ids),axis=(0,1))
            for species,name in [('white_birch','minecraft:birch_log'),('oak','minecraft:oak_log')]:
                indices=[i for i,p in enumerate(palette) if p['Name']==name]
                if not indices:continue
                for si,ly,lz,lx in np.argwhere(np.isin(c['block_id'],indices)):
                    y=int(c['section_y'][si])*16+int(ly);iz=int(cz+lz-zmin);ix=int(cx+lx-xmin)
                    if 0<=iz<shape[0] and 0<=ix<shape[1] and 0<=y-float(data['height'][iz,ix])<=2:
                        roots.add((species,int(cx+lx),int(cz+lz)))
    # Summed-area calculation independent of the generator's81 shifted arrays.
    padded=np.pad(leaf,4);integral=np.pad(padded,((1,0),(1,0))).cumsum(0).cumsum(1)
    canopy=(integral[9:,9:]-integral[:-9,9:]-integral[9:,:-9]+integral[:-9,:-9])/81
    moisture=.4+.2*canopy
    if runoff_path:
        with np.load(runoff_path,allow_pickle=False) as runoff_file:runoff=runoff_file['runoff'].copy()
        if runoff.shape!=shape:raise ServiceError('ecology_field_shape','Actual runoff grid differs from source')
        runoff=np.log1p(np.maximum(runoff,0));moisture+=.3*runoff/max(1e-8,float(np.percentile(runoff,95)))
    return meta,data,canopy,np.clip(moisture,0,1),roots,chunks


def _route_distance(xy,route):
    if route is None:return None
    points=np.asarray(route['points_world_xyz'],float)[:,:2];distance=float('inf')
    for start,end in zip(points[:-1],points[1:]):
        delta=end-start;denom=float(delta@delta)
        if denom==0:continue
        t=float(np.clip((xy-start)@delta/denom,0,1))
        distance=min(distance,float(np.linalg.norm(xy-start-t*delta))-float(route['corridor_width_m'])/2)
    return distance


def _prototype_catalogue(workspace):
    index={};proof=[]
    normalized=workspace/'state/normalized_assets.json';proof.append({'path':str(normalized),'sha256':digest(normalized)})
    for asset in _json(normalized)['assets']:
        if digest(Path(asset['output_blend']))!=asset['output_sha256']:raise ServiceError('changed_native_asset','Normalized native asset changed')
        for obj in asset['objects']:
            index[(asset['output_sha256'],obj['object_name'])]={'asset_id':asset['asset_id'],'anchors':obj['contact_anchors_local_m'],
                                                             'dimensions_m':obj['dimensions_m'],'species':None,'age':'unidentified provider specimen',
                                                             'anchor_owner':True}
    selected=workspace/'state/procedural_assets.json'
    if not selected.is_file():
        return index,proof
    proof.append({'path':str(selected),'sha256':digest(selected)})
    for candidate in _json(selected)['assets']:
        species=candidate['species']
        path=Path(candidate.get('generation',str(Path(candidate['blend']).parent/'generation.json')))
        if candidate.get('generation_sha256') and digest(path)!=candidate['generation_sha256']:
            raise ServiceError('changed_native_asset','Selected original canopy generation changed')
        for entry in candidate['files']:
            if digest(Path(entry['path']))!=entry['sha256']:
                raise ServiceError('changed_native_asset','Selected original canopy dependency changed')
            proof.append(entry)
        asset=_json(path);proof.append({'path':str(path),'sha256':digest(path)})
        if digest(Path(asset['output_blend']))!=asset['output_sha256']:raise ServiceError('changed_native_asset','Procedural native asset changed')
        for variant in asset['variants']:
            for name in variant['objects']:
                index[(asset['output_sha256'],name)]={'asset_id':asset['asset_id'],'anchors':variant['contact_anchors_local_m'],
                  'dimensions_m':(np.asarray(variant['bounds_m'][1])-variant['bounds_m'][0]).tolist(),
                  'species':species,'age':'procedural '+variant.get('form','unidentified')+' form; biological age not known',
                  'anchor_owner':name.endswith('_trunk')}
    return index,proof


def collect_ecology(workspace,scene,ecology_manifest,ir_directory,output_directory,*,route_path=None,runoff_path=None):
    """Measure every exported placement exactly once; no simulator/provider call.

    Output: contact_report.json (strict validator input), ecology_report.json,
    summary.json. Missing identities/anchors/source fields block rather than
    silently reducing coverage. Candidate fit is compared with post-export data.
    """
    from pxr import Usd,UsdGeom,UsdPhysics
    workspace=Path(workspace).resolve();scene=Path(scene).resolve();ecology_manifest=Path(ecology_manifest).resolve()
    ir=Path(ir_directory).resolve();output=Path(output_directory).resolve();output.mkdir(parents=True,exist_ok=True)
    source,fields,canopy,moisture,source_roots,chunks=_source_fields(ir,runoff_path,route_path)
    if route_path and Path(route_path).suffix!='.json':raise ServiceError('route_measurement_unsupported','Independent metric route validation needs the actual explicit corridor polyline JSON')
    route=_json(route_path) if route_path else None
    placements=_json(ecology_manifest);requests={p['id']:p for p in placements['exporter_assets']}
    if len(requests)!=len(placements['exporter_assets']) or not requests:raise ServiceError('placement_identity','Every expected placed instance needs one distinct identity')
    candidates={p['instance_id']:p for p in placements.get('instances',[])}
    prototypes,prototype_proof=_prototype_catalogue(workspace)
    scene_hash=digest(scene);closure=verify_scene_closure(scene,digest(scene.parent/'native_dependency_closure.json'))
    stage=Usd.Stage.Open(str(scene));xf=UsdGeom.XformCache();native=defaultdict(list);ground_parts=[];unowned_meshes=[]
    if UsdGeom.GetStageMetersPerUnit(stage)!=1 or str(UsdGeom.GetStageUpAxis(stage))!='Z':
        raise ServiceError('native_coordinate_frame','Independent metric ecology expects the declared metre-scale Z-up export')
    from isaacmin.validation.native_ground import has_native_ground
    native_support=has_native_ground(scene.parent/'final_ground.obj')
    from .collision_scope import collision_only_paths
    collision_only=collision_only_paths(stage)
    for prim in stage.Traverse():
        if str(prim.GetPath()) in collision_only:continue
        identity=prim.GetAttribute('isaacmin:isaacmin_instance_id').Get()
        if identity:
            native[str(identity)].append(prim)
        if prim.IsA(UsdGeom.Mesh) and 'Terrain_FinalGround' not in str(prim.GetPath()) and 'SourceSurfaceWater' not in str(prim.GetPath()):
            parent=prim
            while parent and not parent.GetAttribute('isaacmin:isaacmin_instance_id').Get():parent=parent.GetParent()
            if not parent:unowned_meshes.append(str(prim.GetPath()))
        if prim.IsA(UsdGeom.Mesh) and 'Terrain_FinalGround' in str(prim.GetPath()):
            if native_support:
                continue
            mesh=UsdGeom.Mesh(prim);matrix=np.asarray(xf.GetLocalToWorldTransform(prim),float)
            points=np.asarray(mesh.GetPointsAttr().Get(),float);points=points@matrix[:3,:3]+matrix[3,:3]
            counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get(),int)
            if not np.all(counts==3):raise ServiceError('ground_not_triangulated','Independent post-USD ground must use actual final triangles')
            ground_parts.append(trimesh.Trimesh(points,np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),int).reshape(-1,3),process=False))
    if set(native)!=set(requests) or unowned_meshes:
        report={'status':'fail','missing_exported_ids':sorted(set(requests)-set(native)),'unexpected_exported_ids':sorted(set(native)-set(requests)),
                'unowned_asset_meshes':unowned_meshes,'scene_sha256':digest(scene),'ecology_manifest_sha256':digest(ecology_manifest)}
        atomic_json(output/'identity_failure.json',report)
        raise ServiceError('exported_instance_coverage','Exported native instance identities differ from expected placements; retained identity_failure.json')
    if native_support:
        from .native_support import exported_native_ground
        ground,ground_digest,_=exported_native_ground(stage,scene.parent/'final_ground.obj',output/'native_support')
    else:
        if len(ground_parts)!=1:raise ServiceError('ground_ownership','Exactly one final supporting native terrain is required for this collector')
        ground=ground_parts[0];hasher=hashlib.sha256()
        hasher.update(memoryview(np.ascontiguousarray(ground.vertices,dtype='<f8')))
        hasher.update(memoryview(np.ascontiguousarray(ground.faces,dtype='<i8')));ground_digest=hasher.hexdigest()
    entries=[];all_anchors=[];root_rows=[];rules={r.asset_id:r for r in RULES}
    origin=np.asarray(source['coordinate_frame']['source_origin_xyz'],float);minimum=np.asarray(fields['min_xz'],int)
    for identity,objects in native.items():
        request=requests[identity];source_objects=[];matrices=[];visibility=[];colliders=[];prototype_paths=[]
        for prim in objects:
            key=(str(prim.GetAttribute('isaacmin:isaacmin_source_blend_sha256').Get()),str(prim.GetAttribute('isaacmin:isaacmin_source_object').Get()))
            if key not in prototypes:raise ServiceError('unbound_exported_asset','Actual native identity does not match normalized/generated asset bytes')
            source_objects.append(prototypes[key]);matrix=np.asarray(xf.GetLocalToWorldTransform(prim),float);matrices.append(matrix)
            visibility.append(str(UsdGeom.Imageable(prim).ComputeVisibility()))
            if prim.IsInstance():prototype_paths.append(str(prim.GetPrototype().GetPath()))
            for child in Usd.PrimRange(prim,Usd.TraverseInstanceProxies()):
                if child.IsA(UsdGeom.Mesh):
                    colliders.append({'path':str(child.GetPath()),'collision_enabled':bool(child.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(child).GetCollisionEnabledAttr().Get()),
                                      'purpose':str(UsdGeom.Imageable(child).ComputePurpose()),'visibility':str(UsdGeom.Imageable(child).ComputeVisibility())})
        owners=[i for i,p in enumerate(source_objects) if p['anchor_owner']]
        if len(owners)!=1:raise ServiceError('ambiguous_root_owner','One measured root/footprint owner per logical instance is required')
        chosen=owners[0];prototype=source_objects[chosen];matrix=matrices[chosen]
        anchors=np.asarray(prototype['anchors'],float);anchors=anchors@matrix[:3,:3]+matrix[3,:3]
        start=len(all_anchors);all_anchors.extend(anchors.tolist())
        position=matrix[3,:3];scale=np.linalg.svd(matrix[:3,:3],compute_uv=False)
        violations=[];expected=np.asarray(request['world_transform'],float).T
        if any(not np.allclose(m,expected,atol=1e-5,rtol=0) for m in matrices):violations.append('exported_transform_differs_from_frozen_placement')
        if any(v!='inherited' for v in visibility) or any(c['visibility']!='inherited' or c['purpose'] not in ('default','render') for c in colliders):violations.append('native_instance_hidden_or_nonrender_purpose')
        ix=int(math.floor(position[0]+origin[0]-minimum[0]));iz=int(math.floor(-position[1]+origin[2]-minimum[1]))
        inside=0<=iz<fields['height'].shape[0] and 0<=ix<fields['height'].shape[1]
        evidence={'valid':bool(inside and fields['validity'][iz,ix])}
        if not evidence['valid']:violations.append('invalid_source_coverage')
        else:
            name=str(fields['block_names'][fields['substrate_id'][iz,ix]])
            substrate='soil' if any(t in name for t in ('dirt','grass_block','podzol','mud','mycelium','moss_block')) else 'sediment' if any(t in name for t in ('sand','gravel','clay')) else 'rock'
            water=bool(fields['water_validity'][iz,ix] and fields['water_height'][iz,ix]>=fields['height'][iz,ix])
            # Evaluate the authored corridor field at its declared source-cell
            # center, and additionally measure exact exported instance clearance.
            cell_xy=np.array([minimum[0]+ix+.5-origin[0],-(minimum[1]+iz+.5)+origin[2]])
            evidence.update(source_biome=str(fields['biome'][iz,ix]),source_substrate=name,substrate=substrate,
                            exposed_surface_water=water,moisture=float(moisture[iz,ix]),canopy=float(canopy[iz,ix]),
                            route_cell_clearance_m=_route_distance(cell_xy,route),route_exact_clearance_m=_route_distance(position[:2],route),
                            source_xz=[int(minimum[0]+ix),int(minimum[1]+iz)])
            if water:violations.append('submerged_source_surface')
        guild=candidates.get(identity,{}).get('guild','source_canopy' if prototype['species'] else 'unidentified')
        burial=float(request.get('burial_depth_m',0))
        if burial and guild not in ('deadwood','mossy_boulder'):violations.append('burial_not_permitted_for_planted_roots')
        if not math.isfinite(burial) or burial<0 or burial>min(.25,max(prototype['dimensions_m'])/4):violations.append('burial_outside_declared_bounds')
        entry={'instance_id':identity,'asset_id':prototype['asset_id'],'guild':guild,'species':prototype['species'] or prototype['asset_id'],
               'biological_age':prototype['age'],'usd_paths':[str(p.GetPath()) for p in objects],'native_prototypes':prototype_paths,
               'world_position':position.tolist(),'world_transform_row_vectors':matrix.tolist(),'measured_scale_axes':scale.tolist(),
               'dimensions_m_after_scale':(np.asarray(prototype['dimensions_m'])*scale).tolist(),
               'anchor_start':start,'anchors_measured':len(anchors),'field_evidence':evidence,'violations':violations,
               'collider_and_visibility_scope':colliders,'burial_depth_m':burial,
               'contact_kind':'explicitly_buried_nonliving_footprint' if burial else 'resting_nonliving_footprint' if guild in ('deadwood','mossy_boulder') else 'planted_roots',
               'native_appearance_qualification':'not_run'}
        entries.append(entry)
    anchors=np.asarray(all_anchors,float);distances=np.full(len(anchors),np.nan);normal_z=np.full(len(anchors),np.nan)
    query_batch=len(anchors) if native_support else 2048
    for start in range(0,len(anchors),max(1,query_batch)):
        part=anchors[start:start+query_batch];ray_origin=part.copy();ray_origin[:,2]+=.5
        directions=np.tile([0.,0.,-1.],(len(part),1))
        points,rays,triangles=ground.ray.intersects_location(ray_origin,directions,multiple_hits=False)
        distances[start+rays]=part[rays,2]-points[:,2];normal_z[start+rays]=ground.face_normals[triangles][:,2]
    for entry in entries:
        begin=entry.pop('anchor_start');end=begin+entry['anchors_measured'];offset=distances[begin:end];normals=normal_z[begin:end]
        unsupported=int((~np.isfinite(offset)|(normals<=0)).sum());finite=offset[np.isfinite(offset)]
        # Unknown support is explicit; finite metric maximum alone cannot pass it.
        maximum=float(np.max(np.abs(finite))) if len(finite) else 0.
        entry.update(unsupported_anchors=unsupported,max_abs_offset_m=maximum,
                     signed_offset_min_m=float(finite.min()) if len(finite) else None,
                     signed_offset_max_m=float(finite.max()) if len(finite) else None)
        rule=rules.get(entry['asset_id']);e=entry['field_evidence'];v=entry['violations']
        if unsupported:v.append('unsupported_native_root_anchor')
        burial=entry['burial_depth_m']
        if len(finite) and np.max(np.abs(finite+burial))>.02:v.append('native_root_or_declared_burial_offset')
        slope=float(np.degrees(np.arccos(np.clip(np.nanmean(normals),-1,1)))) if np.isfinite(normals).any() else None
        e['measured_support_slope_degrees']=slope
        if e['valid'] and rule:
            if e['source_biome'].removeprefix('minecraft:') not in rule.biomes:v.append('biome_mismatch')
            if e['substrate'] not in rule.substrates:v.append('substrate_mismatch')
            if slope is None or slope>rule.max_slope:v.append('slope_limit')
            for key,bounds in [('moisture',rule.moisture),('canopy',rule.canopy)]:
                if not bounds[0]<=e[key]<=bounds[1]:v.append(key+'_outside_recipe')
            if not all(rule.scale[0]-1e-6<=s<=rule.scale[1]+1e-6 for s in entry['measured_scale_axes']):v.append('physical_scale_outside_recipe')
            if e['route_exact_clearance_m'] is not None and e['route_exact_clearance_m']<.45:v.append('exact_trail_clearance')
        elif e['valid'] and entry['guild']=='source_canopy':
            if (entry['species'],*e['source_xz']) not in source_roots:v.append('canopy_species_without_actual_source_root')
            if e['substrate']!='soil':v.append('canopy_nonsoil_substrate')
            if max(abs(s-1) for s in entry['measured_scale_axes'])>1e-6:v.append('source_canopy_scale_changed')
            if e['route_exact_clearance_m'] is not None and e['route_exact_clearance_m']<2:v.append('canopy_trail_clearance')
        else:v.append('missing_ecological_recipe')
        if entry['asset_id']=='celandine_01' and placements.get('season') not in ('spring','late_spring'):v.append('season_mismatch')
    spacing_groups=defaultdict(list)
    for i,entry in enumerate(entries):
        if entry['asset_id'] in rules:spacing_groups[entry['guild'] if entry['guild'] in ('deadwood','mossy_boulder') else 'ground_flora'].append(i)
    nearest={};spacing_pairs=[]
    for group,indices in spacing_groups.items():
        xy=np.asarray([entries[i]['world_position'][:2] for i in indices]);tree=cKDTree(xy)
        if len(xy)>1:
            near=tree.query(xy,k=2)[0][:,1];nearest[group]={'min_m':float(near.min()),'median_m':float(np.median(near)),'p95_m':float(np.percentile(near,95))}
        radius=max(rules[entries[i]['asset_id']].minimum_spacing_m for i in indices)
        for a,b in tree.query_pairs(radius):
            ia,ib=indices[a],indices[b];limit=max(rules[entries[i]['asset_id']].minimum_spacing_m for i in (ia,ib))
            if np.linalg.norm(xy[a]-xy[b])<limit-1e-6:
                entries[ia]['violations'].append('spacing');entries[ib]['violations'].append('spacing');spacing_pairs.append([entries[ia]['instance_id'],entries[ib]['instance_id']])
    valid_area=float(np.sum(fields['validity']))
    if digest(scene)!=scene_hash:raise ServiceError('changed_measured_scene','Native scene changed during independent measurements')
    verify_scene_closure(scene,closure['manifest']['sha256'])
    contact={'schema_version':1,'created_at_utc':utcnow(),'measurement':'independent_exported_usd_anchors_against_final_ground',
             'scene_sha256':digest(scene),'ground_sha256':ground_digest,'ground_hash_encoding':'world float64 vertices followed by int64 triangle indices, little endian',
             'native_scene_dependencies':closure,
             'ecology_manifest_sha256':digest(ecology_manifest),'instances':[{k:e[k] for k in ('instance_id','usd_paths','anchors_measured','max_abs_offset_m','unsupported_anchors','contact_kind','burial_depth_m')} for e in entries]}
    # Intentional burial receives its own declared-footprint test and cannot be
    # accidentally counted as a planted-root2cm pass by the generic validator.
    contact['declared_burial_instances']=[e['instance_id'] for e in entries if e['burial_depth_m']]
    if native_support:
        contact['native_USD_ground_identity']=ground.usd_identity
        contact['native_query_evidence']=ground.receipt()
    contact['validation']=validate_contact_report(contact,expected_instance_ids=requests,scene_sha256=digest(scene),ground_sha256=ground_digest)
    atomic_json(output/'contact_report.json',contact)
    bins=Counter((int(math.floor(e['world_position'][0]/8)),int(math.floor(e['world_position'][1]/8))) for e in entries)
    counts=np.asarray(list(bins.values()),float)
    report={'schema_version':1,'created_at_utc':utcnow(),'status':'fail' if any(e['violations'] for e in entries) else 'incomplete' if route is None else 'pass',
            'scope':'Independent native instance/source-rule and root-anchor measurements only; target appearance remains pending',
            'scene_sha256':digest(scene),'ecology_manifest_sha256':digest(ecology_manifest),
            'native_scene_dependencies':closure,
            'source_ir_manifest_sha256':digest(ir/'world_ir.json'),'source_surface_sha256':digest(ir/'terrain_surface.npz'),
            'source_volume_files':chunks,'prototype_provenance':prototype_proof,
            'runoff_sha256':digest(Path(runoff_path)) if runoff_path else None,'route_sha256':digest(Path(route_path)) if route_path else None,
            'source_water_rule':'Only water_height>=bare_ground height is surface water; underground aquifers excluded',
            'instances':entries,'hard_rule_violations':dict(Counter(v for e in entries for v in e['violations'])),
            'species_counts':dict(Counter(e['species'] for e in entries)),'biological_age_distribution':dict(Counter(e['biological_age'] for e in entries)),
            'asset_counts':dict(Counter(e['asset_id'] for e in entries)),'valid_source_area_m2':valid_area,
            'density_per_m2':{a:n/valid_area for a,n in Counter(e['asset_id'] for e in entries).items()},
            'nearest_neighbour_by_layer':nearest,'spacing_violating_pairs':spacing_pairs,
            'occupied8m_cell_count':len(bins),'occupied_cell_count_variance_over_mean':float(counts.var()/counts.mean()) if len(counts) else None,
            'clustering_limit':'Occupied-cell statistic only; does not establish natural ecological clustering or include habitat-ineligible empty cells',
            'contact_report':{'path':str(output/'contact_report.json'),'sha256':digest(output/'contact_report.json')},
            'missing_measurements':['Full mesh penetration beyond measured base anchors','Biological age of provider grass/fern specimens',
                                    'Target closeup/backlight/motion qualification','Actual foliage RGB/depth/segmentation inclusion',
                                    *(['Exact trail clearance: route not supplied'] if route is None else [])],
            'inference_limits':['Moisture is the declared canopy/runoff proxy, not measured soil hydrology',
                                'Canopy source support does not establish procedural crown realism or age diversity'],
            'Q06':'partial_native_contact_only','Q07':'native_rule_component_only'}
    atomic_json(output/'ecology_report.json',report)
    summary={'status':report['status'],'instances':len(entries),'contact_status':contact['validation']['status'],
             'hard_rule_violations':report['hard_rule_violations'],'contact_report':str(output/'contact_report.json'),
             'ecology_report':str(output/'ecology_report.json'),'final_target_qualification':'not_run'}
    atomic_json(output/'summary.json',summary);return summary
