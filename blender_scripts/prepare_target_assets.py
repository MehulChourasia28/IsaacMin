"""Prepare provider material groups for native USD export; preserve original masters."""
from pathlib import Path
import json,sys,hashlib
import bpy
import numpy as np

request=json.loads(Path(sys.argv[sys.argv.index('--')+1]).read_text())
output=Path(request['output']);output.mkdir(parents=True,exist_ok=True)
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def geometry(objects):
    records={}
    for obj in objects:
        mesh=obj.data;rows={}
        for name,collection,field,width,dtype in [
            ('positions',mesh.vertices,'co',3,np.float32),
            ('normals',mesh.vertices,'normal',3,np.float32),
            ('smooth_faces',mesh.polygons,'use_smooth',1,np.bool_),
            ('loop_vertices',mesh.loops,'vertex_index',1,np.int32),
            ('face_starts',mesh.polygons,'loop_start',1,np.int32),
            ('face_lengths',mesh.polygons,'loop_total',1,np.int32),
            ('face_materials',mesh.polygons,'material_index',1,np.int32)]:
            values=np.empty(len(collection)*width,dtype);collection.foreach_get(field,values)
            rows[name]=hashlib.sha256(values.tobytes()).hexdigest()
        for uv in mesh.uv_layers:
            values=np.empty(len(uv.data)*2,np.float32);uv.data.foreach_get('uv',values)
            rows['uv:'+uv.name]=hashlib.sha256(values.tobytes()).hexdigest()
        rows['world_transform']=[list(row) for row in obj.matrix_world]
        records[obj.name]=rows
    return records

assets=json.loads(Path(request['manifest']).read_text())['assets'];results=[]
for asset in assets:
    if asset['asset_id'] not in request['asset_ids']:continue
    source=Path(asset['output_blend']);assert sha(source)==asset['output_sha256']
    directory=output/asset['asset_id'];directory.mkdir()
    bpy.ops.wm.open_mainfile(filepath=str(source),load_ui=False,use_scripts=False)
    objects=[o for o in bpy.data.objects if o.type=='MESH']
    before=geometry(objects)
    images={image.name:hashlib.sha256(bytes(image.packed_file.data)).hexdigest()
            for image in bpy.data.images if image.packed_file}
    mats={slot.material for obj in objects for slot in obj.material_slots if slot.material}
    graph_before=[];operations=[];colour_rebindings=[]
    for mat in sorted(mats,key=lambda m:m.name):
        if not mat.use_nodes:continue
        tree=mat.node_tree
        graph_before.append({'material':mat.name,'nodes':[
            {'name':node.name,'type':node.bl_idname,
             'group':node.node_tree.name if node.type=='GROUP' and node.node_tree else None,
             'inputs':{i.name:{'default':str(i.default_value) if hasattr(i,'default_value') else None,
                                'links':[link.from_node.name+':'+link.from_socket.name for link in i.links]}
                       for i in node.inputs}}
            for node in tree.nodes]})
        obj=next(o for o in objects if any(s.material==mat for s in o.material_slots))
        bpy.context.view_layer.objects.active=obj
        obj.active_material_index=next(i for i,s in enumerate(obj.material_slots) if s.material==mat)
        area=bpy.context.screen.areas[0];previous=area.type;area.type='NODE_EDITOR'
        space=area.spaces.active;space.tree_type='ShaderNodeTree';space.shader_type='OBJECT';space.pin=True;space.path.start(tree)
        try:
            for iteration in range(32):
                groups=[node for node in tree.nodes if node.type=='GROUP']
                if not groups:break
                for node in tree.nodes:node.select=False
                groups[0].select=True;tree.nodes.active=groups[0]
                operation={'material':mat.name,'group':groups[0].node_tree.name,'node':groups[0].name}
                with bpy.context.temp_override(area=area,region=next(r for r in area.regions if r.type=='WINDOW')):result=bpy.ops.node.group_ungroup()
                operation['result']=list(result);operations.append(operation)
                if result!={'FINISHED'}:raise RuntimeError('Native group ungroup failed')
            if any(n.type=='GROUP' for n in tree.nodes):raise RuntimeError('Nested group bound exhausted')
        finally:area.type=previous
        # USD cannot represent the provider's optional colour-adjustment/mask
        # graph faithfully. Select the original photographed non-dry albedo for
        # the declared late-spring candidate; never export its selection mask as
        # a colour map. Keep normals, roughness and opacity from the same scan.
        original_colour=bpy.data.images.get(asset['asset_id']+'_diff.png')
        if original_colour is None or original_colour.colorspace_settings.name!='sRGB':
            raise RuntimeError('Original photographed albedo identity is missing')
        colour_nodes=[n for n in tree.nodes if n.type=='TEX_IMAGE' and n.image==original_colour]
        if len(colour_nodes)!=1:raise RuntimeError('Ambiguous original albedo node in provider material')
        for node in list(tree.nodes):
            if node.type!='BSDF_PRINCIPLED':continue
            socket=node.inputs['Base Color']
            previous=[link.from_node.name+':'+link.from_socket.name for link in socket.links]
            for link in list(socket.links):tree.links.remove(link)
            tree.links.new(colour_nodes[0].outputs['Color'],socket)
            colour_rebindings.append({'material':mat.name,'from':previous,
                'original_image':original_colour.name,'sha256':images[original_colour.name],
                'recipe':'original non-dry photographic albedo; provider tint/dry/mask controls bypassed explicitly'})
    assert geometry(objects)==before,'Material operation changed native geometry or UVs'
    assert images=={i.name:hashlib.sha256(bytes(i.packed_file.data)).hexdigest()
                   for i in bpy.data.images if i.packed_file},'Original texture bytes changed'
    target=directory/'normalized.blend';bpy.ops.wm.save_as_mainfile(filepath=str(target))
    # The production exporter adds this same explicit alpha cutout. Keep the
    # prepared blend unrounded so it receives exactly one ROUND on population.
    cutouts=[]
    for mat in mats:
        if not mat.use_nodes:continue
        for node in list(mat.node_tree.nodes):
            if node.type!='BSDF_PRINCIPLED' or not node.inputs['Alpha'].is_linked:continue
            alpha=node.inputs['Alpha'];link=alpha.links[0];socket=link.from_socket
            rounding=mat.node_tree.nodes.new('ShaderNodeMath');rounding.operation='ROUND'
            mat.node_tree.links.remove(link);mat.node_tree.links.new(socket,rounding.inputs[0])
            mat.node_tree.links.new(rounding.outputs[0],alpha);cutouts.append(mat.name)
    usd=directory/'asset.usda'
    bpy.ops.wm.usd_export(filepath=str(usd),export_materials=True,export_textures=True,
                          relative_paths=True,generate_preview_surface=True)
    from pxr import Usd,UsdGeom,UsdShade,UsdUtils,Sdf
    stage=Usd.Stage.Open(str(usd));bound=set()
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            binding=UsdShade.MaterialBindingAPI(prim);base,_=binding.ComputeBoundMaterial()
            if base:bound.add(str(base.GetPath()))
            for subset in binding.GetMaterialBindSubsets():
                mat,_=UsdShade.MaterialBindingAPI(subset.GetPrim()).ComputeBoundMaterial()
                if not mat:raise RuntimeError('Native USD material subset is unbound')
                bound.add(str(mat.GetPath()))
            if not base and not binding.GetMaterialBindSubsets():raise RuntimeError('Native USD mesh has no material')
    shader_records=[]
    for name in sorted(bound):
        material=UsdShade.Material(stage.GetPrimAtPath(name));shader,_,_=material.ComputeSurfaceSource()
        if not shader or shader.GetIdAttr().Get()!='UsdPreviewSurface':raise RuntimeError('Missing exported native surface: '+name)
        channels={}
        for channel in ('diffuseColor','roughness','normal','opacity'):
            socket=shader.GetInput(channel);connection=socket.GetConnectedSource() if socket else None
            channels[channel]={'value':str(socket.Get()) if socket else None,'texture':None}
            if connection:
                texture=UsdShade.Shader(connection[0])
                if texture.GetIdAttr().Get()!='UsdUVTexture':raise RuntimeError('Unsupported native channel graph: '+name+':'+channel)
                reference=texture.GetInput('file').Get();file=Path(reference.resolvedPath)
                source_suffix={'diffuseColor':'diff','roughness':'rough','normal':'nor_gl','opacity':'alpha'}[channel]
                expected_image=asset['asset_id']+'_'+source_suffix+'.png'
                if not file.is_relative_to(directory) or sha(file)!=images.get(expected_image):
                    raise RuntimeError('USD channel does not retain its exact original packed role: '+channel)
                channels[channel]['texture']={'path':str(file.relative_to(output)),'sha256':sha(file),
                    'color_space':texture.GetInput('sourceColorSpace').Get(),'source_output':str(connection[1])}
                expected='sRGB' if channel=='diffuseColor' else 'raw'
                if channels[channel]['texture']['color_space']!=expected:raise RuntimeError('Native source color-space mismatch')
        threshold=shader.GetInput('opacityThreshold').Get() if shader.GetInput('opacityThreshold') else None
        if channels['opacity']['texture'] and threshold!=.5:raise RuntimeError('Native botanical alpha cutout is missing')
        if not all(channels[c]['texture'] for c in ('diffuseColor','roughness','normal')):raise RuntimeError('Material preparation lost an original PBR channel')
        shader_records.append({'material':name,'channels':channels,'opacity_threshold':threshold})
    layers,files,missing=UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(usd)))
    if missing:raise RuntimeError('Prepared native material dependencies unresolved')
    paths={Path(layer.realPath) for layer in layers}|{Path(file) for file in files}
    if any(not file.is_relative_to(output) for file in paths):raise RuntimeError('Native asset probe dependency escapes preparation output')
    closure=[{'path':str(file.relative_to(output)),'sha256':sha(file)} for file in sorted(paths)]
    result={'asset_id':asset['asset_id'],'source':str(source),'source_sha256':sha(source),
        'output_blend_relative':str(target.relative_to(output)),'output_sha256':sha(target),'material_graph_before':graph_before,
        'group_operations':operations,'geometry_and_uvs_equal':True,'original_packed_images':images,
        'native_geometry_before_and_after':before,'native_USD_relative':str(usd.relative_to(output)),'native_USD_sha256':sha(usd),
        'native_exported_materials':shader_records,'native_dependency_closure':closure,'declared_cutouts':cutouts,
        'colour_rebindings':colour_rebindings,
        'materials':[{'name':m.name,'node_types':sorted({n.bl_idname for n in m.node_tree.nodes}) if m.use_nodes else []}
                     for m in sorted(mats,key=lambda m:m.name)],
        'appearance_qualification':'not_run','executed_producer_sha256':sha(__file__)}
    write(directory/'flattening.json',result);results.append(result)
write(output/'results.json',{'status':'native_group_flattening_complete','recipe':'native_provider_group_flatten_v1','assets':results,
                            'qualification':'not_inferred'})
