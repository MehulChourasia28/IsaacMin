"""Background Blender worker. Imported .blend files never execute scripts."""
import argparse
import hashlib
import json
import math
import pathlib
import re
import sys
import bpy
from mathutils import Vector, Matrix
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from mesh_arrays import macro_delta, physical_uvs, soil_displacement

parser = argparse.ArgumentParser()
parser.add_argument('--request', required=True)
args = parser.parse_args(sys.argv[sys.argv.index('--')+1:])
request = json.loads(pathlib.Path(args.request).read_text())
output = pathlib.Path(request['output']).resolve()
output.mkdir(parents=True, exist_ok=True)
if not bpy.app.build_options.usd:
    raise RuntimeError('This Blender build lacks native USD export')
if request.get('terrain_mesh_sha256') and hashlib.sha256(pathlib.Path(request['terrain_mesh']).read_bytes()).hexdigest()!=request['terrain_mesh_sha256']:
    raise RuntimeError('Authoritative source terrain mesh changed before Blender import')
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.preferences.filepaths.use_scripts_auto_execute=False
scene = bpy.context.scene
scene.unit_settings.system = 'METRIC'
scene.unit_settings.scale_length = 1.0
material_recipe=request.get('terrain_material_recipe')
foliage_recipe=request.get('foliage_material_recipe')
if foliage_recipe:
    from isaacmin.assembly.foliage_recipe import validate_foliage_recipe
    validate_foliage_recipe(pathlib.Path(__file__).resolve().parents[1],foliage_recipe)
if material_recipe:
    from isaacmin.assembly.material_recipe import validate_material_recipe
    validate_material_recipe(pathlib.Path(__file__).resolve().parents[1],material_recipe,request['materials'])


def material(spec):
    mat = bpy.data.materials.new(spec['name'])
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get('Principled BSDF')
    bsdf.inputs['Roughness'].default_value = 0.75
    for role, socket in [('base_color','Base Color'),('roughness','Roughness'),
                         ('metallic','Metallic'),('opacity','Alpha')]:
        if role not in spec: continue
        tex = mat.node_tree.nodes.new('ShaderNodeTexImage')
        tex.image = bpy.data.images.load(str(pathlib.Path(spec[role]).resolve()), check_existing=True)
        tex.image.colorspace_settings.name = 'sRGB' if role == 'base_color' else 'Non-Color'
        mat.node_tree.links.new(tex.outputs['Color'], bsdf.inputs[socket])
    if 'normal' in spec:
        tex = mat.node_tree.nodes.new('ShaderNodeTexImage')
        tex.image = bpy.data.images.load(str(pathlib.Path(spec['normal']).resolve()), check_existing=True)
        tex.image.colorspace_settings.name = 'Non-Color'
        normal = mat.node_tree.nodes.new('ShaderNodeNormalMap')
        mat.node_tree.links.new(tex.outputs['Color'],normal.inputs['Color'])
        mat.node_tree.links.new(normal.outputs['Normal'],bsdf.inputs['Normal'])
    mat.diffuse_color = (0.35,0.3,0.22,1)
    return mat


bare_reuse=request.get('bare_scene_directory')
cache_record=None
if bare_reuse:
    import shutil
    import numpy as np
    from isaacmin.assembly.native_cache import verify_bare,terrain_payload,sha,copy_native_precision_evidence
    root=pathlib.Path(__file__).resolve().parents[1]
    cache_record=verify_bare(request,root,bare_reuse)
    copied_precision_evidence=copy_native_precision_evidence(bare_reuse,output,cache_record)
    bare_directory=pathlib.Path(cache_record['directory'])
    bpy.ops.wm.open_mainfile(filepath=str(bare_directory/'scene.blend'),load_ui=False,use_scripts=False)
    bpy.context.preferences.filepaths.use_scripts_auto_execute=False
    scene=bpy.context.scene
    if scene.unit_settings.system!='METRIC' or scene.unit_settings.scale_length!=1:
        raise RuntimeError('Cached native scene coordinate units changed')
    if sorted(o.name for o in scene.objects)!=sorted(cache_record['manifest']['expected_object_names']):
        raise RuntimeError('Cached bare scene contains unexpected objects')
    terrain=scene.objects.get('Terrain_FinalGround')
    if terrain is None or terrain.type!='MESH' or terrain.modifiers or not np.array_equal(np.asarray(terrain.matrix_world),np.eye(4)):
        raise RuntimeError('Cached final terrain transform or modifiers changed')
    if terrain_payload(terrain.data)!=cache_record['manifest']['terrain_payload']:
        raise RuntimeError('Cached terrain points, normals, indices, UV or material graph changed')
    prior=cache_record['export']
    mats=request['materials'];texture_scale=prior['physical_texture_repeat_m']
    assignment=prior['material_assignment'];subdivision_levels=prior['subdivision_levels']
    subdivision_precision=prior.get('subdivision_precision')
    exact_obj_export=prior.get('exact_obj_export')
    subdivision_before=prior['subdivision_input_counts'];displacement_info=prior['physical_displacement']
    precision_finalization=prior.get('native_precision_finalization')
    transition_bakes=prior['transition_bakes'];water_cells=prior['source_exterior_water_cells']
    if material_recipe:
        shutil.copyfile(bare_directory/'source_weights.json',output/'source_weights.json')
    water_provenance=prior['source_water_interfaces'];water=scene.objects.get('SourceSurfaceWater')
    water_faces=bool(water);water_mesh=water.data if water else None
    if water and terrain_payload(water_mesh)!=cache_record['manifest']['water_payload']:
        raise RuntimeError('Cached saved-fluid geometry, UV or material graph changed')
    if (output/'final_ground.obj').exists():raise RuntimeError('Do not overwrite existing final ground during cache reuse')
    shutil.copyfile(bare_directory/'final_ground.obj',output/'final_ground.obj')
    if sha(output/'final_ground.obj')!=cache_record['manifest']['ground_sha256']:
        raise RuntimeError('Cached authoritative ground bytes changed during copy')
else:
    detail=request.get('geometry_detail',{})
    subdivision_levels=int(detail.get('subdivision_levels',0))
    subdivision_precision=None
    if detail.get('subdivision_method')=='bilinear_double_no_limit_v1' and subdivision_levels:
        if any(request.get('terrain_translation',[0,0,0])):
            raise RuntimeError('Precise subdivision requires already normalized world coordinates')
        from precise_subdivision import construct
        terrain,subdivision_before,subdivision_precision=construct(request,output/'precise_subdivision',pathlib.Path(__file__).resolve().parents[1])
    else:
        bpy.ops.wm.obj_import(filepath=str(pathlib.Path(request['terrain_mesh']).resolve()),
                              forward_axis='Y', up_axis='Z')
        terrain = bpy.context.selected_objects[0]
        terrain.name = 'Terrain_FinalGround'
        terrain.location = request.get('terrain_translation', [0,0,0])
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        detail = request.get('geometry_detail', {})
        subdivision_levels = int(detail.get('subdivision_levels',0))
        if not 0<=subdivision_levels<=3: raise RuntimeError('Geometry subdivision outside bounded profile')
        subdivision_before={'vertices':len(terrain.data.vertices),'polygons':len(terrain.data.polygons),
                            'polygon_corners':len(terrain.data.loops)}
        if subdivision_levels:
            if not bpy.app.build_options.opensubdiv:
                raise RuntimeError('Requested physical geometry sampling requires an OpenSubdiv-enabled native Blender')
            modifier=terrain.modifiers.new('PhysicalSurfaceSampling','SUBSURF')
            modifier.subdivision_type='SIMPLE'
            modifier.levels=subdivision_levels;modifier.render_levels=subdivision_levels
            bpy.context.view_layer.objects.active=terrain
            applied=bpy.ops.object.modifier_apply(modifier=modifier.name)
            expected_faces=subdivision_before['polygon_corners']*(4**(subdivision_levels-1))
            if applied!={'FINISHED'} or len(terrain.data.vertices)<=subdivision_before['vertices'] or len(terrain.data.polygons)<expected_faces:
                raise RuntimeError('Native SIMPLE subdivision did not create requested physical support sampling')

    delta_spec = request.get('exterior_delta')
    if delta_spec:
        import numpy as np
        arrays = np.load(delta_spec['path'],allow_pickle=False)
        base, delta = arrays[delta_spec['source_height_key']], arrays[delta_spec['delta_key']]
        protect = arrays[delta_spec['protection_key']].astype(bool)
        min_x,min_z = delta_spec['source_min_xz']
        ox,oy,oz = request['minecraft_origin']
        naturalization_path = request.get('exterior_naturalization')
        if naturalization_path:
            naturalization_path = pathlib.Path(naturalization_path)
            naturalization = json.loads(naturalization_path.read_text())
            from isaacmin.io import sha256_file
            expected_inputs = {entry['path']:entry['sha256'] for entry in naturalization['inputs']}
            mesh_entry = next(entry for entry in naturalization['files'] if entry['path']=='terrain.obj')
            if (naturalization.get('status') != 'candidate_requires_independent_validation'
                    or naturalization.get('refinement_applied_to_geometry') is not True
                    or naturalization.get('fixed_native_coordinates_unchanged') is not True
                    or naturalization.get('origin') != request['minecraft_origin']
                    or sha256_file(pathlib.Path(request['terrain_mesh'])) != mesh_entry['sha256']
                    or expected_inputs.get(str(pathlib.Path(delta_spec['path']).resolve())) != sha256_file(pathlib.Path(delta_spec['path']))
                    or expected_inputs.get(str(pathlib.Path(request['source_surface']).resolve())) != sha256_file(pathlib.Path(request['source_surface']))):
                raise RuntimeError('Preapplied exterior geometry lacks matching construction evidence')
            # Geometry already contains the refined field. The original delta
            # remains the material/displacement context below; never zero it.
        else:
            macro_delta(terrain.data,base,delta,protect,(min_x,min_z),(ox,oy,oz))

    mats = request.get('materials',[])
    if not mats:
        raise RuntimeError('Production USD requires acquired PBR materials')
    for spec in mats: terrain.data.materials.append(material(spec))
    texture_scale = float(request.get('texture_repeat_m', 2.0))
    if not 0.1<=texture_scale<=20: raise RuntimeError('Invalid physical material scale')
    # Source-aware semantic ownership precedes physical UV scale and displacement.
    assignment={'mode':'technical_fixture_slope_only','displacement_eligible_material_indices':[0],
                'qualification':'not_run'}
    material_ids={m.get('asset_id',m.get('name')) for m in mats}
    if request.get('source_surface') and {'brown_mud_dry','rock_face_03','leafy_grass','forest_ground_04'}<=material_ids:
        from isaacmin.assembly.material_assignment import assign_materials
        structures=[]
        if request.get('support_manifest'):
            support=json.loads(pathlib.Path(request['support_manifest']).read_text())
            structures=[b['source_xyz_min_corner'] for b in support['structural_blocks']]
        assignment=assign_materials(terrain,mats,request['source_surface'],request['minecraft_origin'],
                                   exterior_delta_spec=delta_spec,structural_source_coordinates=structures)
    else:
        for polygon in terrain.data.polygons:
            polygon.material_index=1 if len(mats)>1 and abs(polygon.normal.z)<0.7 else 0
    physical_uvs(terrain.data,mats,texture_scale)

    # Displace real vertices only where every adjacent face agrees on one eligible
    # soil material. Mixed boundaries, rock and protected portals retain their mesh.
    displacement_info=[]
    amplitude=float(detail.get('soil_displacement_peak_to_peak_m',0))
    if amplitude:
        if not 0<amplitude<=0.04:
            raise RuntimeError('Physical soil displacement amplitude outside bounded metric range')
        context=(base,protect,(min_x,min_z),(ox,oy,oz)) if delta_spec else None
        displacement_info=soil_displacement(terrain.data,mats,set(assignment['displacement_eligible_material_indices']),
            amplitude,texture_scale,delta_context=context)

    # Author a single triangulation once, so the OBJ validator, USD render and
    # PhysX all consume exactly the same final support triangles.
    triangulation=terrain.modifiers.new('AuthoritativeTriangles','TRIANGULATE')
    triangulation.quad_method='FIXED';triangulation.ngon_method='CLIP'
    bpy.context.view_layer.objects.active=terrain
    bpy.ops.object.modifier_apply(modifier=triangulation.name)
    precision_finalization=None
    if request.get('native_precision_input'):
        from precision_mesh import finalize_mesh
        precision_finalization=finalize_mesh(terrain,pathlib.Path(__file__).resolve().parents[1],output/'native_precision',preserve_native_topology=True)
        # Final topology owns the material fields and corner tangent basis.
        # Recompute these on the authoritative triangles after exact repair;
        # physical displacement above is not repeated.
        if request.get('source_surface') and {'brown_mud_dry','rock_face_03','leafy_grass','forest_ground_04'}<=material_ids:
            assignment=assign_materials(terrain,mats,request['source_surface'],request['minecraft_origin'],
                exterior_delta_spec=delta_spec,structural_source_coordinates=structures)
        else:
            for polygon in terrain.data.polygons:
                polygon.material_index=1 if len(mats)>1 and abs(polygon.normal.z)<0.7 else 0
        physical_uvs(terrain.data,mats,texture_scale)
    transition_bakes={'status':'not_applicable','scope':'technical_fixture_without_source_material_semantics'}
    if request.get('source_surface') and {'brown_mud_dry','rock_face_03','leafy_grass','forest_ground_04'}<=material_ids:
        if material_recipe:
            from isaacmin.assembly.material_mdl import write_source_weights
            weights=write_source_weights(terrain,request['source_surface'],request['minecraft_origin'],output,
                exterior_delta_spec=delta_spec,structural_source_coordinates=structures)
            transition_bakes={'status':'shared_original_scan_candidate','recipe':material_recipe['mode'],
                'source_weights_manifest':'source_weights.json','source_weights':weights['source_weights'],
                'qualification':'not_run','geometry_unchanged':True}
        else:
            from isaacmin.assembly.material_bake import apply_transition_bakes
            transition_bakes=apply_transition_bakes(terrain,mats,request['source_surface'],request['minecraft_origin'],
                output/'transition_bakes',exterior_delta_spec=delta_spec,structural_source_coordinates=structures,
                material_factory=material,texels_per_m=1024,tile_size_m=4)
    from precision_mesh import write_authoritative_obj
    exact_obj_export=write_authoritative_obj(terrain,pathlib.Path(__file__).resolve().parents[1],output,precision_finalization)

    water_cells=0
    water_provenance={'mode':'technical_surface_fixture','scope_triangle_counts':{}}
    water_vertices=[];water_faces=[]
    if request.get('source_water_mesh'):
        water_path=pathlib.Path(request['source_water_mesh'])
        if hashlib.sha256(water_path.read_bytes()).hexdigest()!=request['source_water_mesh_sha256']:
            raise RuntimeError('Source water interface bytes changed before export')
        water_source=json.loads(water_path.read_text())
        water_vertices=water_source['vertices'];water_faces=water_source['triangles']
        for key in ('triangle_source_cell_xyz','triangle_scope','triangle_surface_kind','triangle_body_id'):
            if len(water_source[key])!=len(water_faces):raise RuntimeError('Water face ownership is incomplete')
        exterior_cells={tuple(cell) for cell,scope in zip(water_source['triangle_source_cell_xyz'],water_source['triangle_scope']) if scope=='exterior'}
        aquifer_cells={tuple(cell) for cell,scope in zip(water_source['triangle_source_cell_xyz'],water_source['triangle_scope']) if scope=='aquifer'}
        water_cells=len(exterior_cells)
        water_provenance={'mode':water_source.get('source_data_kind','exact_saved_fluid_interfaces'),'source_file':water_path.name,
            'source_sha256':request['source_water_mesh_sha256'],'construction_status':water_source['status'],
            'source_water_cells':water_source['source_water_cells'],
            'rendered_exterior_source_cells':len(exterior_cells),'rendered_aquifer_source_cells':len(aquifer_cells),
            'scope_triangle_counts':water_source['scope_triangle_counts'],
            'surface_triangle_counts':water_source['surface_triangle_counts'],
            'unknown_adjacent_faces':len(water_source['unknown_adjacent_faces']),
            'uncertain_corner_neighbors':len(water_source['uncertain_corner_neighbors']),
            'surface_convention':water_source['surface_convention'],'qualification':'not_run'}
    elif request.get('source_surface'):
        import numpy as np
        surface=np.load(request['source_surface'],allow_pickle=False)
        if 'water_validity' in surface:
            wet=(surface['water_validity'] & surface['validity'] &
                 (surface['water_height']>surface['height']+1e-5))
            mx,mz=surface['min_xz'];ox,oy,oz=request['minecraft_origin']
            vertices=[];faces=[];indices={}
            for iz,ix in np.argwhere(wet):
                level=float(surface['water_height'][iz,ix])
                face=[]
                for dx,dz in ((0,0),(0,1),(1,1),(1,0)):
                    key=(int(ix)+dx,int(iz)+dz,level)
                    if key not in indices:
                        indices[key]=len(vertices)
                        vertices.append((float(mx)+key[0]-ox,-float(mz)-key[1]+oz,level-oy))
                    face.append(indices[key])
                faces.append(face)
            water_cells=len(faces)
            water_vertices=vertices;water_faces=faces
    if water_faces:
        water_mesh=bpy.data.meshes.new('SourceWaterSurfaceMesh')
        water_mesh.from_pydata(water_vertices,[],water_faces);water_mesh.update()
        water=bpy.data.objects.new('SourceSurfaceWater',water_mesh)
        scene.collection.objects.link(water)
        water['isaacmin_source_water_sha256']=request.get('source_water_mesh_sha256') or 'technical_surface_fixture'
        water_material=bpy.data.materials.new('Water_RequiresIsaacTransmission')
        water_material.use_nodes=True
        shader=water_material.node_tree.nodes.get('Principled BSDF')
        shader.inputs['Base Color'].default_value=(0.82,0.92,0.96,1)
        shader.inputs['Roughness'].default_value=0.06
        shader.inputs['IOR'].default_value=1.333
        shader.inputs['Transmission Weight'].default_value=1
        water_mesh.materials.append(water_material)

placed = []
placement_identities=[]
cutout_materials=[]
prototype_cache = {}
source_blend_hashes = {}
for asset in request.get('assets',[]):
    source=pathlib.Path(asset['blend']).resolve()
    if source not in source_blend_hashes:
        digest=hashlib.sha256()
        with source.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
        source_blend_hashes[source]=digest.hexdigest()
    key=(str(source),tuple(asset.get('objects',[])))
    if key not in prototype_cache:
        with bpy.data.libraries.load(str(source), link=False) as (data_from,data_to):
            names=asset.get('objects') or data_from.objects
            data_to.objects=[n for n in names if n in data_from.objects]
        prototype_cache[key]=[o for o in data_to.objects if o is not None and o.type=='MESH']
        if asset.get('alpha_mode')=='cutout':
            for prototype in prototype_cache[key]:
                for slot in prototype.material_slots:
                    mat=slot.material
                    if not mat or not mat.use_nodes or mat.name in cutout_materials:continue
                    for node in list(mat.node_tree.nodes):
                        if node.type!='BSDF_PRINCIPLED' or not node.inputs['Alpha'].is_linked:continue
                        alpha=node.inputs['Alpha'];link=alpha.links[0];source_socket=link.from_socket
                        # Blender's native USD writer recognizes ROUND as cutout
                        # semantics and authors UsdPreviewSurface opacityThreshold=.5.
                        rounding=mat.node_tree.nodes.new('ShaderNodeMath')
                        rounding.operation='ROUND';rounding.label='Declared botanical cutout threshold 0.5'
                        mat.node_tree.links.remove(link)
                        mat.node_tree.links.new(source_socket,rounding.inputs[0])
                        mat.node_tree.links.new(rounding.outputs[0],alpha)
                        cutout_materials.append(mat.name)

    objects=[o.copy() for o in prototype_cache[key]]
    if not objects: raise RuntimeError(f'No meshes in asset {asset["id"]}')
    # Each request is an explicit measured placement, not an unqualified random scatter.
    for obj,prototype in zip(objects,prototype_cache[key]):
        scene.collection.objects.link(obj)
        if 'world_transform' in asset:
            obj.matrix_world = Matrix(asset['world_transform']) @ obj.matrix_world
        else:
            obj.location += Vector(asset['position'])
            obj.rotation_euler.z += float(asset.get('yaw',0))
            obj.scale *= float(asset.get('scale',1))
        obj['isaacmin_instance_id']=str(asset.get('instance_id',asset['id']))
        obj['isaacmin_source_blend_sha256']=source_blend_hashes[source]
        obj['isaacmin_source_object']=str(prototype.name)
        placement_identities.append({'object_name':obj.name,'instance_id':obj['isaacmin_instance_id'],
                                     'source_blend_sha256':obj['isaacmin_source_blend_sha256']})
        placed.append(obj.name)

texture_files=[]
for image in bpy.data.images:
    if image.source=='FILE' and (image.filepath or image.packed_file):
        source=pathlib.Path(bpy.path.abspath(image.filepath or image.name))
        if not source.is_file() and not image.packed_file:
            raise RuntimeError(f'Missing imported texture {image.name}')
        # Preserve the provider's original encoded bytes. Re-saving through
        # Blender would silently change precision/colour/alpha in packed maps.
        packed=image.packed_file
        was_packed=bool(packed)
        data=bytes(packed.data) if packed else source.read_bytes()
        digest=hashlib.sha256(data).hexdigest()
        suffix=source.suffix.lower()
        if data.startswith(b'\x89PNG\r\n\x1a\n'):suffix='.png'
        elif data.startswith(b'\xff\xd8\xff'):suffix='.jpg'
        elif data.startswith(b'v/1\x01'):suffix='.exr'
        elif data.startswith((b'II*\x00',b'MM\x00*')):suffix='.tif'
        elif data.startswith((b'#?RADIANCE',b'#?RGBE')):suffix='.hdr'
        target=output/'textures'/(digest+suffix)
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Content-addressed texture collision')
        if not target.exists():target.write_bytes(data)
        if packed:image.unpack(method='REMOVE')
        image.filepath=str(target)
        texture_files.append({'image_name':image.name,'path':str(target.relative_to(output)),
                              'sha256':digest,'packed_source':was_packed})

bpy.ops.wm.save_as_mainfile(filepath=str(output/'scene.blend'))
settings=dict(filepath=str(output/'content.usdc'),selected_objects_only=False,
              export_materials=True,export_textures=True,relative_paths=True,
              overwrite_textures=True,export_normals=True,export_uvmaps=True,
              generate_preview_surface=True,root_prim_path='/World',
              use_instancing=True,export_custom_properties=True,custom_properties_namespace='isaacmin')
available=bpy.ops.wm.usd_export.get_rna_type().properties.keys()
settings={k:v for k,v in settings.items() if k in available}
result=bpy.ops.wm.usd_export(**settings)
if result!={'FINISHED'} or not (output/'content.usdc').is_file():
    raise RuntimeError('Blender native USD export did not complete')
from pxr import Usd,UsdGeom,UsdUtils,Sdf
(output/'world.usda').write_text('#usda 1.0\n(\n defaultPrim = "World"\n metersPerUnit = 1\n upAxis = "Z"\n subLayers = [@content.usdc@]\n)\n')
native_stage=Usd.Stage.Open(str(output/'world.usda'))
if not native_stage:raise RuntimeError('Native USD failed to reopen export')
native_ground=[UsdGeom.Mesh(p) for p in native_stage.Traverse()
               if p.IsA(UsdGeom.Mesh) and 'Terrain_FinalGround' in str(p.GetPath())]
if len(native_ground)!=1:raise RuntimeError('Native USD ground ownership changed')
ground_mesh=native_ground[0]
native_points=ground_mesh.GetPointsAttr().Get()
native_counts=ground_mesh.GetFaceVertexCountsAttr().Get()
if len(native_points)!=len(terrain.data.vertices) or len(native_counts)!=len(terrain.data.polygons) or any(c!=3 for c in native_counts):
    raise RuntimeError('Native USD topology differs from authoritative triangulated ground')
import numpy as np
expected_points=np.empty(len(terrain.data.vertices)*3,dtype=np.float32)
terrain.data.vertices.foreach_get('co',expected_points)
expected_indices=np.empty(len(terrain.data.loops),dtype=np.int32)
terrain.data.loops.foreach_get('vertex_index',expected_indices)
if not np.array_equal(np.asarray(native_points).reshape(-1),expected_points) or not np.array_equal(np.asarray(ground_mesh.GetFaceVertexIndicesAttr().Get()),expected_indices):
    raise RuntimeError('Native USD changed final support vertex/index values')
shared_material=None
if material_recipe:
    from isaacmin.assembly.material_mdl import prepare_library,bind_shared_material
    root=pathlib.Path(__file__).resolve().parents[1]
    prepare_library(root,root/material_recipe['candidate_manifest'],output/'shared_material')
    shared_material=bind_shared_material(output/'world.usda',output/'shared_material/material_manifest.json',
        output/'source_weights.json',ground_prim_paths=[str(ground_mesh.GetPath())])
leaf_material=None
if foliage_recipe:
    from pxr import Tf
    from isaacmin.assembly.foliage_mdl import bind_thin_leaves
    leaf_names=[Tf.MakeValidIdentifier(name) for name in cutout_materials
                if re.fullmatch(r'[A-Za-z0-9_]+_Leaf(?:\.[0-9]{3})?',name)]
    if leaf_names:
        leaf_material=bind_thin_leaves(pathlib.Path(__file__).resolve().parents[1],output/'world.usda',
            leaf_names,transmission_fraction=foliage_recipe['tissue_transmission_fraction'])
if water_faces:
    native_water=[UsdGeom.Mesh(p) for p in native_stage.Traverse()
                  if p.IsA(UsdGeom.Mesh) and 'SourceSurfaceWater' in str(p.GetPath())]
    if len(native_water)!=1:raise RuntimeError('Native USD water ownership changed')
    expected_water_points=np.empty(len(water_mesh.vertices)*3,dtype=np.float32)
    water_mesh.vertices.foreach_get('co',expected_water_points)
    expected_water_indices=np.empty(len(water_mesh.loops),dtype=np.int32)
    water_mesh.loops.foreach_get('vertex_index',expected_water_indices)
    if not np.array_equal(np.asarray(native_water[0].GetPointsAttr().Get()).reshape(-1),expected_water_points):
        raise RuntimeError('Native USD changed saved fluid surface vertices')
    if not np.array_equal(np.asarray(native_water[0].GetFaceVertexIndicesAttr().Get()),expected_water_indices):
        raise RuntimeError('Native USD changed saved fluid surface topology')
    water_provenance.update(native_usd_geometry_verified=True,vertices=len(water_mesh.vertices),
                            polygons=len(water_mesh.polygons),usd_prim=str(native_water[0].GetPath()))
from isaacmin.assembly.credits import attach_credits
native_credits=attach_credits(pathlib.Path(__file__).resolve().parents[1],output/'world.usda',request.get('assets',[]))
def file_sha(path):
    digest=hashlib.sha256()
    with pathlib.Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()
layers,external,unresolved=UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(output/'world.usda')))
closure=[];absolute=[]
for filename in sorted(set([l.realPath for l in layers]+list(external))):
    path=pathlib.Path(filename).resolve()
    if not path.is_relative_to(output):absolute.append(str(path));continue
    closure.append({'path':str(path.relative_to(output)),'sha256':file_sha(path),'bytes':path.stat().st_size})
for layer in layers:
    for path in layer.GetExternalReferences():
        if pathlib.Path(path).is_absolute():absolute.append(path)
closure_record={'status':'pass' if not unresolved and not absolute else 'fail',
        'root_asset':'world.usda','root_sha256':file_sha(output/'world.usda'),
        'files':closure,'unresolved_paths':list(unresolved),'absolute_asset_paths':absolute,
        'method':'UsdUtils.ComputeAllDependencies','toolchain':{'blender':bpy.app.version_string,
            'build_hash':bpy.app.build_hash.decode(),'usd':list(Usd.GetVersion())}}
(output/'native_dependency_closure.json').write_text(json.dumps(closure_record,indent=2)+'\n')
if closure_record['status']!='pass':raise RuntimeError('USD dependency closure is not portable')
from isaacmin.assembly.native_cache import terrain_payload,record_bare
native_terrain_payload=terrain_payload(terrain.data)
native_water_payload=terrain_payload(water_mesh) if water_faces else None
if cache_record and native_terrain_payload!=cache_record['manifest']['terrain_payload']:
    raise RuntimeError('Population altered cached final terrain/material payload')
if cache_record and native_water_payload!=cache_record['manifest']['water_payload']:
    raise RuntimeError('Population altered cached saved-fluid payload')
(output/'export_result.json').write_text(json.dumps({
    'status':'success','evidence_level':'external_tool_verified',
    'blender_version':bpy.app.version_string,'native_usd_export':True,
    'source_terrain_mesh':request['terrain_mesh'],
    'exterior_naturalization':request.get('exterior_naturalization'),
    'terrain_vertices':len(terrain.data.vertices),'terrain_polygons':len(terrain.data.polygons),
    'final_ground_obj':str(output/'final_ground.obj'),
    'usd_sha256':file_sha(output/'world.usda'),
    'final_ground_sha256':file_sha(output/'final_ground.obj'),'exact_obj_export':exact_obj_export,
    'native_usd_geometry_verified':True,
    'native_terrain_payload':native_terrain_payload,'native_water_payload':native_water_payload,
    'bare_scene_reuse':{'manifest_sha256':cache_record['manifest_sha256'],'ground_sha256':cache_record['manifest']['ground_sha256'],'actual_payload_equality':True} if cache_record else None,
    'native_usd_terrain_vertices':len(native_points),'native_usd_terrain_triangles':len(native_counts),
    'native_instance_count':sum(p.IsInstance() for p in native_stage.Traverse()),
    'material_assignment':assignment,'material_count':len(terrain.data.materials),
    'source_material_count':len(mats),'transition_bakes':transition_bakes,'shared_original_material':shared_material,
    'tree_leaf_material':leaf_material,
    'portable_attribution_files':native_credits,
    'physical_texture_repeat_m':texture_scale,
    'material_repeat_m':{m['name']:float(m.get('repeat_m',texture_scale)) for m in mats},
    'texture_files':texture_files,
    'declared_cutout_materials':cutout_materials,'instance_identity_properties':placement_identities,'placed_object_names':placed,'up_axis':'Z','meters_per_unit':1,
    'subdivision_levels':subdivision_levels,'subdivision_input_counts':subdivision_before,'subdivision_precision':subdivision_precision,
    'subdivision_output_counts':{'vertices':len(terrain.data.vertices),'triangles':len(terrain.data.polygons)},'physical_displacement':displacement_info,
    'source_exterior_water_cells':water_cells,
    'source_water_interfaces':water_provenance,
    'water_sensor_policy':'transparent_refractive_geometry_requires_explicit_depth_exception',
    'surface_ownership':'single_full_3d_volume_no_heightfield_caps',
    'isaac_material_qualification':'not_run','appearance_qualification':'not_run',
    'native_precision_finalization':precision_finalization,
    'collision_policy':'apply_triangle_mesh_to_exact_final_Terrain_FinalGround_in_Isaac'
},indent=2)+'\n')

if not request.get('assets') and not cache_record:
    record_bare(request,pathlib.Path(__file__).resolve().parents[1],output)
