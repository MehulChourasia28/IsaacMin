"""Read-only exported USD material/collider inspection, independent of Blender.

Run with the pinned OpenUSD Python bindings. Structural success is not Q06
appearance qualification. No simulator, renderer or external service is launched.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import re

import numpy as np

from .network import ServiceError, digest, atomic_json, utcnow


def _value(value):
    if value is None or isinstance(value,(str,int,float,bool)):return value
    if hasattr(value,'path'):return {'authored':value.path,'resolved':value.resolvedPath}
    try:return [_value(v) for v in value]
    except TypeError:return str(value)


def _asset_path(attribute,asset):
    from pxr import Sdf
    if not asset:return None
    if asset.resolvedPath:return Path(asset.resolvedPath).resolve()
    stack=attribute.GetPropertyStack()
    if not stack:return None
    path=Path(Sdf.ComputeAssetPathRelativeToLayer(stack[0].layer,asset.path))
    return path.resolve() if path.is_file() else None


def _inspect_mdl(material,shader,scene,errors,textures,runtime_modules):
    """Inspect actual authored MDL context, closed originals and metric metadata."""
    from pxr import UsdShade
    item={'render_context':'mdl','required_weight_primvars':[],'status':'structural_only','qualification':'not_run'}
    attr=shader.GetPrim().GetAttribute('info:mdl:sourceAsset');asset=shader.GetSourceAsset('mdl');module=_asset_path(attr,asset)
    identifier=shader.GetSourceAssetSubIdentifier('mdl')
    if identifier=='IsaacMinThinLeaf':
        return _inspect_leaf_mdl(material,shader,scene,errors,textures,module,asset)
    if shader.GetImplementationSourceAttr().Get()!=UsdShade.Tokens.sourceAsset or not identifier:
        errors.append({'category':'invalid_mdl_implementation','material':str(material.GetPath())})
    if module is None:
        declared=next((r for r in runtime_modules if r.get('module')==(asset.path if asset else None)),None)
        if declared and Path(declared['path']).is_file() and digest(Path(declared['path']))==declared['sha256']:
            item.update(runtime_module=declared,dependency_scope='Explicit pinned native runtime module; separate target shader validation required')
        else:errors.append({'category':'unresolved_mdl_module','material':str(material.GetPath()),'asset':_value(asset)})
        return item
    item['module']={'path':str(module),'sha256':digest(module),'identifier':identifier}
    if not module.is_relative_to(scene.parent) or Path(asset.path).is_absolute():
        errors.append({'category':'nonportable_mdl_module','path':str(module)})
    prim=material.GetPrim();manifest_attr=prim.GetAttribute('isaacmin:manifest');manifest=_asset_path(manifest_attr,manifest_attr.Get()) if manifest_attr else None
    if manifest is None:
        errors.append({'category':'missing_mdl_material_manifest','material':str(material.GetPath())});return item
    try:
        record=json.loads(manifest.read_text())
        from isaacmin.assembly.material_mdl import RECIPE,WEIGHT_NAMES,png_dimensions
        from isaacmin.assembly.material_assignment import REQUIRED
        if record['recipe']!=RECIPE or record['module_sha256']!=digest(module) or record['source_asset_subidentifier']!=identifier:
            raise ValueError('MDL material recipe/module identity mismatch')
        declared=list(prim.GetAttribute('isaacmin:materialFamilies').Get() or [])
        repeats=np.asarray(prim.GetAttribute('isaacmin:physicalRepeatMetres').Get(),float)
        if declared!=list(REQUIRED) or record['material_order']!=list(REQUIRED) or repeats.shape!=(4,) or not np.isfinite(repeats).all() or np.any(repeats<=0):
            raise ValueError('MDL source-family order or physical scales invalid')
        slots={(f['asset_id'],f['role']) for f in record['textures']}
        if len(record['textures'])!=12 or slots!={(n,r) for n in REQUIRED for r in ('base_color','roughness','normal')}:
            raise ValueError('MDL needs all twelve original source channels exactly once')
        template=Path(__file__).resolve().parents[3]/'recipes/materials/shared_original.mdl.in'
        if digest(template)!=record['template_sha256']:raise ValueError('MDL template is not the current pinned implementation')
        reconstructed=template.read_text()
        expected=set();raw=module.read_text();paths=re.findall(r'texture_2d\("([^"\n]+)"\s*,\s*tex::gamma_(srgb|linear)\)',raw)
        actual={(str((module.parent/p).resolve()),g) for p,g in paths}
        for f in record['textures']:
            path=(manifest.parent/f['path']).resolve();gamma='srgb' if f['role']=='base_color' else 'linear'
            if not path.is_relative_to(scene.parent) or not path.is_file() or digest(path)!=f['sha256']:
                raise ValueError('MDL original texture changed, missing or outside package')
            if png_dimensions(path)!=(f['width'],f['height']) or min(f['width'],f['height'])<4096 or min(f['width'],f['height'])/f['repeat_m']<1024:
                raise ValueError('Original source density is below the shared-material contract')
            if abs(float(f['repeat_m'])-repeats[list(REQUIRED).index(f['asset_id'])])>1e-5:
                raise ValueError('Authored MDL scale and manifest disagree')
            expected.add((str(path),gamma));textures[str(path)]={'path':str(path),'sha256':f['sha256'],'bytes':path.stat().st_size,
                'inside_package':True,'source_color_space':f['colour_space'],'asset_id':f['asset_id'],'role':f['role'],'repeat_m':f['repeat_m']}
            index=list(REQUIRED).index(f['asset_id']);reconstructed=reconstructed.replace('@@'+str(index)+'_'+f['role']+'@@','./'+f['path'])
            reconstructed=reconstructed.replace('@@'+str(index)+'_repeat_m@@',repr(float(f['repeat_m'])))
        if actual!=expected or len(paths)!=12:raise ValueError('Actual MDL texture constructors and declared channels differ')
        if reconstructed!=raw:raise ValueError('Actual MDL coordinate, normal, color or scale implementation differs from pinned template')
        dependency=prim.GetAttribute('isaacmin:scanDependencies');listed={str(_asset_path(dependency,a)) for a in (dependency.Get() or [])}
        if listed!={p for p,_ in expected}:raise ValueError('Native MDL scan dependency closure is incomplete')
        item.update(manifest={'path':str(manifest),'sha256':digest(manifest)},source_families=declared,physical_repeat_m=repeats.tolist(),
                    required_weight_primvars=list(WEIGHT_NAMES),projection=record['projection'],normal_semantics=record['normal_semantics'])
    except (ValueError,KeyError,TypeError,OSError) as e:
        errors.append({'category':'invalid_mdl_material_contract','material':str(material.GetPath()),'reason':str(e)})
    return item


def _inspect_leaf_mdl(material,shader,scene,errors,textures,module,asset):
    """Verify closed original leaf channels; compiled appearance is separate."""
    from pxr import UsdShade
    item={'render_context':'mdl','kind':'original_tree_leaf','status':'structural_only',
          'required_weight_primvars':[],'required_uv_primvars':['st'],'qualification':'not_run'}
    try:
        if module is None or not module.is_relative_to(scene.parent) or Path(asset.path).is_absolute():
            raise ValueError('Leaf MDL module is missing or outside its portable scene')
        attr=material.GetPrim().GetAttribute('isaacmin:leafManifest')
        path=_asset_path(attr,attr.Get()) if attr else None
        if path is None or not path.is_relative_to(scene.parent):
            raise ValueError('Leaf MDL manifest is missing or outside its portable scene')
        record=json.loads(path.read_text())
        matches=[row for row in record['materials'] if row['material']==str(material.GetPath())]
        if len(matches)!=1:raise ValueError('Leaf material identity is not uniquely recorded')
        declared=matches[0]
        root=Path(__file__).resolve().parents[3]
        if (record['recipe']!='original_textures_thin_leaf_v1' or record['alpha_threshold']!=.5
                or record['template_sha256']!=digest(root/'recipes/materials/thin_leaf.mdl.in')
                or record['producer_sha256']!=digest(root/'src/isaacmin/assembly/foliage_mdl.py')
                or declared['module_sha256']!=digest(module)
                or (scene.parent/declared['module']).resolve()!=module):
            raise ValueError('Leaf module, template, cutout or original constructor differs')
        reflection,transmission=record['tissue_reflection_fraction'],record['tissue_transmission_fraction']
        if not 0<=transmission<=1 or reflection!=1-transmission:
            raise ValueError('Leaf tissue mixture is not normalized')
        roles={row['role'] for row in declared['channels']}
        if not {'base_color','opacity'}<=roles or roles-{'base_color','opacity','roughness','normal'}:
            raise ValueError('Leaf channel roles are incomplete or unsupported')
        raw=module.read_text()
        actual={(str((module.parent/p).resolve()),gamma)
                for p,gamma in re.findall(r'texture_2d\("([^"\n]+)"\s*,\s*tex::gamma_(srgb|linear)\)',raw)}
        expected=set()
        for channel in declared['channels']:
            texture=(scene.parent/channel['path']).resolve()
            original=(scene.parent/channel['source']).resolve()
            gamma='srgb' if channel['role']=='base_color' else 'linear'
            if (not texture.is_relative_to(scene.parent) or not original.is_relative_to(scene.parent)
                    or digest(texture)!=channel['sha256'] or digest(original)!=channel['sha256']
                    or channel['source_color_space']!=('sRGB' if gamma=='srgb' else 'raw')
                    or channel['uv_primvar']!='st' or channel['wrap']!='repeat'):
                raise ValueError('Original leaf texture bytes, UVs or color semantics differ')
            expected.add((str(texture),gamma))
            textures[str(texture)]={'path':str(texture),'sha256':channel['sha256'],
                'bytes':texture.stat().st_size,'inside_package':True,
                'source_color_space':channel['source_color_space'],'role':channel['role']}
        if actual!=expected:raise ValueError('Actual leaf MDL texture constructors differ from original channels')
        dependency=material.GetPrim().GetAttribute('isaacmin:leafTextures')
        if {str(_asset_path(dependency,a)) for a in (dependency.Get() or [])}!={p for p,_ in expected}:
            raise ValueError('Original leaf texture dependency closure is incomplete')
        preview=material.GetSurfaceOutput().GetConnectedSource()
        if not preview or UsdShade.Shader(preview[0]).GetIdAttr().Get()!='UsdPreviewSurface':
            raise ValueError('Original exported leaf graph was not retained')
        item.update(module={'path':str(module),'sha256':digest(module),'identifier':'IsaacMinThinLeaf'},
            manifest={'path':str(path),'sha256':digest(path)},alpha_threshold=.5,
            tissue_transmission_fraction=transmission,normal_model=declared['normal_model'],
            optical_parameters=record['optical_parameters'])
    except (ValueError,KeyError,TypeError,OSError) as exc:
        errors.append({'category':'invalid_leaf_mdl_contract','material':str(material.GetPath()),'reason':str(exc)})
    return item


def inspect_usd(scene, output_path, *, expected_cutout_materials=(), physical_materials=(), runtime_modules=()):
    """Inspect native bindings, shader nodes, texture bytes and sensor/collider scope.

    expected_cutout_materials is an explicit list of actual material names or USD
    paths (not a name heuristic); e.g. fern_02. Physical repeats are recorded from
    the generator contract but UV stretch/physical scale still need measurement.
    """
    from pxr import Usd,UsdGeom,UsdShade,UsdPhysics,UsdUtils,Sdf
    scene=Path(scene).resolve();scene_hash=digest(scene);stage=Usd.Stage.Open(str(scene))
    if not stage:raise ServiceError('usd_open_failed','Native exported USD could not be opened')
    prims=list(stage.Traverse())
    for prototype in stage.GetPrototypes():prims.extend(Usd.PrimRange(prototype))
    errors=[];warnings=[];materials={};physics_materials={};textures={};meshes=[];instances=[];placement_identities=[]
    from .collision_scope import collision_only_paths
    try:collision_only=collision_only_paths(stage)
    except ValueError as exc:
        collision_only=set();errors.append({'category':'invalid_collision_only_ownership','reason':str(exc)})
    explicit_cutouts=set(expected_cutout_materials);found_cutouts=set()
    supported={'UsdPreviewSurface','UsdUVTexture','UsdTransform2d','UsdPrimvarReader_float2'}
    for prim in prims:
        identity={str(a.GetName()):_value(a.Get()) for a in prim.GetAttributes() if 'isaacmin' in str(a.GetName())}
        if identity and not prim.IsInPrototype():placement_identities.append({'path':str(prim.GetPath()),'identity':identity})
        if prim.IsInstance():
            instances.append({'path':str(prim.GetPath()),'prototype':str(prim.GetPrototype().GetPath()),
                              'identity':{str(a.GetName()):_value(a.Get()) for a in prim.GetAttributes() if 'isaacmin' in str(a.GetName())}})
        if not prim.IsA(UsdShade.Material):continue
        material=UsdShade.Material(prim);material_path=str(prim.GetPath())
        # Isaac leaves the friction/restitution materials of removed probes in
        # the saved physics layer. The API schema, not the prim name, identifies
        # them. An authored but broken surface output still needs inspection.
        if prim.HasAPI(UsdPhysics.MaterialAPI) and not any(
                output.GetAttr().GetPropertyStack() for output in material.GetSurfaceOutputs()):
            physics_materials[material_path]={'path':material_path,'kind':'physics_only',
                'properties':{str(a.GetName()):_value(a.Get()) for a in prim.GetAttributes()
                              if str(a.GetName()).startswith('physics:')}}
            continue
        cutout=prim.GetName() in explicit_cutouts or material_path in explicit_cutouts
        if cutout:found_cutouts.update({prim.GetName(),material_path}&explicit_cutouts)
        mdl_output=material.GetSurfaceOutput('mdl')
        context='mdl' if mdl_output and mdl_output.GetAttr().HasAuthoredConnections() else 'universal'
        shader,_,_=material.ComputeSurfaceSource('mdl') if context=='mdl' else material.ComputeSurfaceSource()
        item={'path':material_path,'surface_shader':str(shader.GetPath()) if shader else None,
              'declared_cutout':cutout,'nodes':[],'channels':{},'required_uv_primvars':[]}
        if not shader:
            errors.append({'category':'missing_surface_shader','material':material_path})
        if shader and context=='mdl':item['mdl']=_inspect_mdl(material,shader,scene,errors,textures,runtime_modules)
        pending=[shader.GetPrim()] if shader and context!='mdl' else [];seen=set();uv_names=set(item.get('mdl',{}).get('required_uv_primvars',[]))
        if item.get('mdl',{}).get('kind')=='original_tree_leaf':
            retained,_,_=material.ComputeSurfaceSource()
            if retained:pending=[retained.GetPrim()]
        while pending:
            node=pending.pop();node_path=str(node.GetPath())
            if node_path in seen:continue
            seen.add(node_path);node_shader=UsdShade.Shader(node)
            if not node_shader:
                errors.append({'category':'invalid_shader_connection','path':node_path});continue
            node_id=node_shader.GetIdAttr().Get()
            node_record={'path':node_path,'id':node_id,'inputs':{}}
            if node_id not in supported:
                warnings.append({'category':'shader_requires_target_support','path':node_path,'id':node_id})
            for inp in node_shader.GetInputs():
                connections=inp.GetAttr().GetConnections();data={'value':_value(inp.Get()),'connections':[str(c) for c in connections]}
                node_record['inputs'][inp.GetBaseName()]=data
                for connection in connections:
                    source=stage.GetPrimAtPath(connection.GetPrimPath())
                    if not source or not stage.GetPropertyAtPath(connection):
                        errors.append({'category':'broken_shader_connection','connection':str(connection)})
                    elif source.IsA(UsdShade.Shader):pending.append(source)
                if node_id=='UsdPreviewSurface':item['channels'][inp.GetBaseName()]=data
            if node_id=='UsdUVTexture':
                asset=node_shader.GetInput('file').Get()
                resolved=Path(asset.resolvedPath) if asset and asset.resolvedPath else None
                if resolved is None or not resolved.is_file():
                    errors.append({'category':'missing_texture','shader':node_path,'asset':_value(asset)})
                else:
                    bound={'path':str(resolved),'sha256':digest(resolved),'bytes':resolved.stat().st_size,
                           'inside_package':resolved.is_relative_to(scene.parent),
                           'source_color_space':node_shader.GetInput('sourceColorSpace').Get() or 'auto'}
                    textures[str(resolved)]=bound;node_record['texture']=bound
                    if not bound['inside_package']:errors.append({'category':'nonportable_texture','shader':node_path,'path':str(resolved)})
            if node_id=='UsdPrimvarReader_float2':
                name=node_shader.GetInput('varname').Get()
                if name:uv_names.add(str(name))
                else:errors.append({'category':'missing_uv_primvar_name','shader':node_path})
            item['nodes'].append(node_record)
        item['required_uv_primvars']=sorted(uv_names)
        if cutout:
            opacity=item['channels'].get('opacity',{});threshold=item['channels'].get('opacityThreshold',{}).get('value',0)
            if not opacity.get('connections'):
                errors.append({'category':'missing_cutout_texture_connection','material':material_path})
            if threshold is None or not 0<float(threshold)<=1:
                errors.append({'category':'missing_cutout_threshold','material':material_path,
                               'reason':'Threshold0 permits transparent-surface lighting; botanical mask requires explicit cutout'})
        normal_connections=item['channels'].get('normal',{}).get('connections',[])
        for connected in normal_connections:
            normal=UsdShade.Shader(stage.GetPrimAtPath(Sdf.Path(connected).GetPrimPath()))
            if normal and normal.GetIdAttr().Get()=='UsdUVTexture' and normal.GetInput('sourceColorSpace').Get()!='raw':
                errors.append({'category':'normal_map_not_raw','material':material_path})
        materials[material_path]=item
    for missing in sorted(explicit_cutouts-found_cutouts):errors.append({'category':'declared_cutout_material_missing','material':missing})
    # Physics-only materials are valid for purpose="physics". They cannot
    # satisfy a rendering binding, including inherited and collection bindings,
    # mesh subsets, instance roots, or renderer-specific material purposes.
    if physics_materials:
        for prim in prims:
            if not (prim.IsA(UsdGeom.Imageable) or prim.IsA(UsdGeom.Subset)):continue
            binding=UsdShade.MaterialBindingAPI(prim)
            for purpose in (UsdShade.Tokens.allPurpose,UsdShade.Tokens.full,UsdShade.Tokens.preview):
                bound,relation=binding.ComputeBoundMaterial(materialPurpose=purpose)
                if bound and str(bound.GetPath()) in physics_materials:
                    errors.append({'category':'physics_material_used_for_rendering',
                        'prim':str(prim.GetPath()),'material':str(bound.GetPath()),
                        'material_purpose':str(purpose),'binding':str(relation.GetPath())})
    for prim in prims:
        if not prim.IsA(UsdGeom.Mesh):continue
        if str(prim.GetPath()) in collision_only:
            meshes.append({'path':str(prim.GetPath()),'is_prototype':False,
                'faces':len(UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get() or []),
                'materials':[],'collision_only':True,'render_source':str(prim.GetRelationship('isaacmin:renderSource').GetTargets()[0]),
                'visibility':'invisible','purpose':'guide','collision_api':True,'collision_enabled':True})
            continue
        mesh=UsdGeom.Mesh(prim);binding=UsdShade.MaterialBindingAPI(prim);base,_=binding.ComputeBoundMaterial()
        bound=set([str(base.GetPath())]) if base else set();subsets=[]
        face_count=len(mesh.GetFaceVertexCountsAttr().Get() or [])
        covered=np.zeros(face_count,dtype=np.bool_)
        for subset in binding.GetMaterialBindSubsets():
            mat,_=UsdShade.MaterialBindingAPI(subset.GetPrim()).ComputeBoundMaterial()
            indices=np.asarray(subset.GetIndicesAttr().Get() or [],dtype=np.int64)
            valid=(indices>=0)&(indices<face_count)
            if not valid.all():errors.append({'category':'invalid_material_subset_index','mesh':str(prim.GetPath())})
            bounded=indices[valid]
            if covered[bounded].any() or len(np.unique(bounded))!=len(bounded):errors.append({'category':'overlapping_material_subsets','mesh':str(prim.GetPath())})
            covered[bounded]=True
            subsets.append({'path':str(subset.GetPath()),'faces':len(indices),'material':str(mat.GetPath()) if mat else None})
            if mat:bound.add(str(mat.GetPath()))
            else:errors.append({'category':'unbound_material_subset','path':str(subset.GetPath())})
        if not base and not covered.all():errors.append({'category':'unbound_mesh_faces','mesh':str(prim.GetPath()),'faces':int(np.count_nonzero(~covered))})
        primvars=UsdGeom.PrimvarsAPI(prim);uvs={}
        for name in sorted({name for mat in bound for name in materials.get(mat,{}).get('mdl',{}).get('required_weight_primvars',[])}):
            weight=primvars.GetPrimvar(name)
            if not weight or weight.GetInterpolation() not in ('vertex','varying') or weight.IsIndexed() or len(weight.Get() or [])!=len(mesh.GetPointsAttr().Get() or []):
                errors.append({'category':'missing_or_invalid_mdl_weights','mesh':str(prim.GetPath()),'primvar':name})
        if any(materials.get(mat,{}).get('mdl',{}).get('required_weight_primvars') for mat in bound):
            try:
                from isaacmin.assembly.material_mdl import validate_weights,WEIGHT_NAMES,CHUNK
                from isaacmin.assembly.material_assignment import REQUIRED
                weights=np.column_stack((np.asarray(primvars.GetPrimvar(WEIGHT_NAMES[0]).Get()),np.asarray(primvars.GetPrimvar(WEIGHT_NAMES[1]).Get())))
                proof=validate_weights(weights)
                if list(prim.GetAttribute('isaacmin:materialWeightOrder').Get() or [])!=list(REQUIRED):
                    raise ValueError('Native source-weight family order changed')
                attribute=prim.GetAttribute('isaacmin:sourceWeightsManifest')
                path=_asset_path(attribute,attribute.Get()) if attribute else None
                if path is None or not path.is_relative_to(scene.parent):raise ValueError('Source-weight proof is not package-local')
                record=json.loads(path.read_text())
                if record['material_order']!=list(REQUIRED) or record['source_weights']!=proof:
                    raise ValueError('Exported source weights differ from the bound geometry producer')
                points=np.asarray(mesh.GetPointsAttr().Get(),dtype=np.float32)
                matrix=np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),float)
                measured=hashlib.sha256()
                for start in range(0,len(points),CHUNK):
                    world=points[start:start+CHUNK].astype(float)@matrix[:3,:3]+matrix[3,:3]
                    measured.update(np.ascontiguousarray(world,dtype='<f4').tobytes())
                if measured.hexdigest()!=record['world_points_float32_sha256']:
                    raise ValueError('Exported world geometry differs from the bound source-weight producer')
            except (KeyError,OSError,json.JSONDecodeError) as e:errors.append({'category':'missing_mdl_source_weight_proof','mesh':str(prim.GetPath()),'reason':str(e)})
            except (ValueError,TypeError,RuntimeError) as e:errors.append({'category':'invalid_mdl_weight_partition','mesh':str(prim.GetPath()),'reason':str(e)})
        for name in sorted({uv for mat in bound for uv in materials.get(mat,{}).get('required_uv_primvars',[])}):
            uv=primvars.FindPrimvarWithInheritance(name)
            if not uv or not uv.HasValue():errors.append({'category':'missing_mesh_uv','mesh':str(prim.GetPath()),'primvar':name})
            else:uvs[name]={'interpolation':str(uv.GetInterpolation()),'indexed':uv.IsIndexed(),'value_count':len(uv.Get() or [])}
        collision=prim.HasAPI(UsdPhysics.CollisionAPI)
        item={'path':str(prim.GetPath()),'is_prototype':prim.IsInPrototype(),'faces':face_count,
              'materials':sorted(bound),'subsets':subsets,'uv_primvars':uvs,
              'double_sided':mesh.GetDoubleSidedAttr().Get(),
              'visibility':str(UsdGeom.Imageable(prim).ComputeVisibility()),
              'purpose':str(UsdGeom.Imageable(prim).ComputePurpose()),
              'collision_api':collision,'collision_enabled':UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() if collision else False,
              'identity':{str(a.GetName()):_value(a.Get()) for a in prim.GetAttributes() if 'isaacmin' in str(a.GetName())}}
        meshes.append(item)
    layers,external,unresolved=UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(scene)))
    verified_runtime={r['module'] for r in runtime_modules if Path(r['path']).is_file() and digest(Path(r['path']))==r['sha256']}
    unresolved=[p for p in unresolved if str(p) not in verified_runtime]
    if unresolved:errors.append({'category':'unresolved_native_dependencies','paths':list(unresolved)})
    if digest(scene)!=scene_hash:raise ServiceError('changed_inspected_scene','Scene changed during read-only inspection')
    report={'schema_version':1,'created_at_utc':utcnow(),'status':'fail' if errors else 'pass',
            'validator_sha256':digest(Path(__file__)),
            'collision_scope_validator_sha256':digest(Path(__file__).with_name('collision_scope.py')),
            'evidence_level':'native_exported_structure_inspected','scene':str(scene),'scene_sha256':scene_hash,
            'openusd_version':list(Usd.GetVersion()),'units_m_per_unit':UsdGeom.GetStageMetersPerUnit(stage),
            'declared_pinned_runtime_modules':[r for r in runtime_modules if r['module'] in verified_runtime],
            'up_axis':str(UsdGeom.GetStageUpAxis(stage)),
            'mesh_count':len(meshes),'native_instance_count':len(instances),'prototype_count':len(stage.GetPrototypes()),
            'materials':list(materials.values()),'physics_materials':list(physics_materials.values()),
            'textures':list(textures.values()),'meshes':meshes,'instances':instances,'placement_identities':placement_identities,
            'physical_material_contract':[{'asset_id':m.get('asset_id',m.get('name')),'repeat_m':m.get('repeat_m')} for m in physical_materials],
            'errors':errors,'warnings':warnings,
            'collision_scope':{'meshes_with_enabled_collision':sum(bool(m['collision_enabled']) for m in meshes),
                               'non_ground_colliders':'Inspect listed mesh APIs; missing decoration colliders are not inferred'},
            'root_contact':{'status':'not_run','reason':'USD transform presence does not measure anchors against independent final ground'},
            'sensor_scope':{'status':'not_run','reason':'Visibility/purpose are authored metadata; actual depth/segmentation inclusion requires Isaac captures'},
            'Q06':'not_run','Q07':'not_run',
            'missing_measurements':['Actual Isaac closeup/backlight cutouts, transmission and material response',
                                    'UV metric stretch and physical texture repetition measured on exported surfaces',
                                    'Independent post-USD contact offset for every asset instance and full-mesh penetration',
                                    'Actual sensor depth/segmentation inclusion, including explicit transparent foliage exceptions',
                                    'Exported ecological instance rule audit and species/age/size/density/clustering distributions']}
    atomic_json(Path(output_path),report)
    return report


def validate_contact_report(report, *, expected_instance_ids, scene_sha256, ground_sha256,
                            tolerance_m=.02, minimum_fraction=.995):
    """Strict portable report schema for independent post-USD anchor measurements.

    Each instance: instance_id, usd_paths, anchors_measured, max_abs_offset_m,
    unsupported_anchors. Full-mesh penetration and native dynamics are separate.
    """
    import math
    required={'instance_id','usd_paths','anchors_measured','max_abs_offset_m','unsupported_anchors'}
    if report.get('scene_sha256')!=scene_sha256 or report.get('ground_sha256')!=ground_sha256 or report.get('measurement')!='independent_exported_usd_anchors_against_final_ground':
        raise ServiceError('contact_provenance','Contact report does not bind independent native USD/final-ground geometry')
    entries=report.get('instances',[]);ids=[r.get('instance_id') for r in entries]
    if len(set(ids))!=len(ids) or set(ids)!=set(expected_instance_ids) or not ids:
        raise ServiceError('contact_coverage','Independent contact report must measure every exact exported instance once')
    good=0
    for entry in entries:
        if not required<=set(entry) or not entry['usd_paths'] or type(entry['anchors_measured']) is not int or entry['anchors_measured']<1 or type(entry['unsupported_anchors']) is not int or not 0<=entry['unsupported_anchors']<=entry['anchors_measured']:
            raise ServiceError('contact_schema','Invalid or empty native instance anchor record')
        offset=entry['max_abs_offset_m']
        if type(offset) not in (int,float) or not math.isfinite(offset) or offset<0:
            raise ServiceError('contact_schema','Native contact offset must be a finite nonnegative metric distance')
        good+=offset<=tolerance_m and entry['unsupported_anchors']==0
    return {'status':'pass' if good/len(entries)>=minimum_fraction else 'fail','instances':len(entries),
            'within_tolerance':good,'fraction':good/len(entries),'tolerance_m':tolerance_m,
            'minimum_fraction':minimum_fraction,'full_mesh_penetration':'not_run','native_dynamic_contact':'separate_Q05'}
