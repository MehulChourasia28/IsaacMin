"""Export explicitly selected whole provider trees, preserving highest-detail meshes.

Downloaded files are data. Run with --factory-startup --disable-autoexec. Materials
are translated separately from geometry; the translation remains a candidate
until inspected in Isaac. Helpers and lower LODs are never selected implicitly.
"""
from pathlib import Path
import hashlib
import json
import os
import sys
import bpy
import numpy as np

request = json.loads(Path(sys.argv[sys.argv.index('--') + 1]).read_text())
root = Path(request['workspace'])
sys.path.insert(0, str(root / 'src'))
from isaacmin.io import atomic_json, sha256_file, utc_now

out = Path(request['output'])
out.mkdir(parents=True, exist_ok=False)
records = []
for spec in request['assets']:
    card = json.loads(Path(spec['card']).read_text())
    source = next(f for f in card['files'] if f['path'].endswith('.blend'))
    if sha256_file(Path(source['path'])) != source['sha256']:
        raise RuntimeError('Provider master changed')
    directory = out / card['asset_id']
    directory.mkdir()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.filepaths.use_scripts_auto_execute = False
    bpy.context.scene.unit_settings.system = 'METRIC'
    bpy.context.scene.unit_settings.scale_length = 1.
    with bpy.data.libraries.load(source['path'], link=False) as (available, loaded):
        if not set(spec['objects']).issubset(available.objects):
            raise RuntimeError('Explicit whole-tree selection unavailable')
        loaded.objects = spec['objects']
    geometry = {}
    for obj in loaded.objects:
        normals_only = (spec.get('allow_normal_modifiers', False)
                        and all(m.type == 'WEIGHTED_NORMAL' for m in obj.modifiers))
        if obj.type != 'MESH' or (obj.modifiers and not normals_only) or obj.parent:
            raise RuntimeError('Expected baked full-detail provider mesh')
        bpy.context.scene.collection.objects.link(obj)
        obj.hide_set(False)
        obj.hide_render = False
        obj.select_set(True)
        bpy.context.view_layer.update()
        mesh = obj.data
        points = np.empty((len(mesh.vertices), 3), np.float32)
        mesh.vertices.foreach_get('co', points.ravel())
        indices = np.empty(len(mesh.loops), np.int32)
        mesh.loops.foreach_get('vertex_index', indices)
        counts = np.empty(len(mesh.polygons), np.int32)
        mesh.polygons.foreach_get('loop_total', counts)
        ids = np.empty(len(mesh.polygons), np.int32)
        mesh.polygons.foreach_get('material_index', ids)
        uv = mesh.uv_layers.get('UVMap')
        if uv is None:
            attribute = mesh.attributes.get('UVMap')
            if attribute and attribute.data_type == 'FLOAT_VECTOR' and attribute.domain in ('POINT', 'CORNER'):
                values = np.empty((len(attribute.data), 3), np.float32)
                attribute.data.foreach_get('vector', values.ravel())
                source_uv = values[:, :2] if attribute.domain == 'CORNER' else values[indices, :2]
                # Geometry Nodes masters can store UVMap as a vector attribute.
                # Convert that exact corner field into an ordinary USD UV layer.
                uv = mesh.uv_layers.new(name='IsaacMinExportUV')
                uv.data.foreach_set('uv', np.ascontiguousarray(source_uv).ravel())
            else:
                raise RuntimeError('Original provider UVMap cannot be exported: ' + obj.name + ': ' +
                    repr([(a.name,a.data_type,a.domain) for a in mesh.attributes]))
        mesh.uv_layers.active = uv
        uv.active_render = True
        uvs = np.empty((len(mesh.loops), 2), np.float32)
        uv.data.foreach_get('uv', uvs.ravel())
        # Put the whole original tree's lowest root on local Z=0. Preserve the
        # mesh bytes and express the anchor adjustment only in its transform.
        original_transform = [list(row) for row in obj.matrix_world]
        basis = np.array([list(row) for row in obj.matrix_world.to_3x3()], np.float64)
        if not np.isfinite(basis).all() or np.linalg.det(basis) <= 0:
            raise RuntimeError('Original object needs explicit handedness repair')
        metric_points = points @ basis.T
        min_z = float(metric_points[:, 2].min())
        obj.location = (0., 0., -min_z)
        obj['source_blend_sha256'] = source['sha256']
        obj['source_object'] = obj.name
        attrs = [{'name': a.name, 'type': a.data_type, 'domain': a.domain}
                 for a in mesh.attributes]
        colors = mesh.color_attributes.get('Col')
        mask = None
        mask_domain = None
        if colors:
            values = np.empty((len(colors.data), 4), np.float32)
            colors.data.foreach_get('color', values.ravel())
            mask = values[:,0].copy() if spec.get('blend_mask_channel')=='red' else values[:, :3] @ np.array([.2126, .7152, .0722], np.float32)
            mask_domain = colors.domain
        band = metric_points[metric_points[:, 2] <= min_z + .02].copy()
        band[:, 2] -= min_z
        anchors = band[np.linspace(0, len(band) - 1, min(32, len(band))).astype(int)]
        geometry[obj.name] = dict(points=points, indices=indices, counts=counts,
            material_ids=ids, uvs=uvs, mask=mask, mask_domain=mask_domain,
            material_names=[m.name if m else None for m in mesh.materials],
            attributes=attrs, original_world_matrix=original_transform,
            native_normal_modifiers=[dict(name=m.name, type=m.type) for m in obj.modifiers],
            dimensions_m=(metric_points.max(axis=0) - metric_points.min(axis=0)).tolist(),
            original_metric_basis=basis.tolist(),
            contact_anchors_local_m=anchors.tolist(), root_offset_z_m=-min_z)
    bpy.context.view_layer.objects.active = loaded.objects[0]
    usd = directory / 'library.usdc'
    result = bpy.ops.wm.usd_export(filepath=str(usd), root_prim_path='/Library',
        selected_objects_only=True, export_materials=False, export_normals=True,
        export_uvmaps=True, export_custom_properties=True, custom_properties_namespace='isaacmin')
    if result != {'FINISHED'}:
        raise RuntimeError('Native Blender export failed')
    prototype_records = []
    for name, arrays in geometry.items():
        recipes = {}
        for slot, material_name in enumerate(arrays['material_names']):
            if material_name is None:
                if np.any(arrays['material_ids'] == slot):
                    raise RuntimeError('Original whole tree has faces using an empty material slot')
                continue
            overrides=spec.get('material_overrides',{})
            if material_name in overrides:
                if not spec.get('material_override_reason'):
                    raise RuntimeError('An authored native material needs its explicit translation policy')
                recipes[material_name]=dict(overrides[material_name])
                continue
            tree = bpy.data.materials[material_name].node_tree
            mapping = tree.nodes.get('Mapping')
            box_mapping = tree.nodes.get('Mapping.001')
            principled = next((n for n in tree.nodes if n.type == 'BSDF_PRINCIPLED'),None)
            group=None;group_recipe={}
            if principled is None and spec.get('derive_atlas_cutout_from_graph'):
                groups=[n for n in tree.nodes if n.type=='GROUP']
                if len(groups)!=1:raise RuntimeError('Provider group needs explicit translation')
                group=groups[0]
                inner=group.node_tree;output_node=next(n for n in inner.nodes if n.type=='GROUP_OUTPUT' and n.is_active_output)
                mix=output_node.inputs[0].links[0].from_node
                if mix.type!='MIX_SHADER':raise RuntimeError('Only original alpha/translucency Mix group is implemented')
                principled=mix.inputs[1].links[0].from_node
                translucent=mix.inputs[2].links[0].from_node
                factor=mix.inputs[0].links[0].from_node
                if (principled.type!='BSDF_PRINCIPLED' or translucent.type!='BSDF_TRANSLUCENT'
                    or factor.type!='MATH' or factor.operation!='MULTIPLY'
                    or {factor.inputs[i].links[0].from_socket.name for i in (0,1)}!={'Alpha','Translucency'}):
                    raise RuntimeError('Unrecognised original leaf-group algebra')
                for socket,expected in [('Hue',.5),('Saturation',1.),('Wetness',0.)]:
                    if group.inputs[socket].is_linked or group.inputs[socket].default_value!=expected:
                        raise RuntimeError('Untranslated leaf group colour/wetness adjustment')
                if group.inputs['Translucency'].is_linked or group.inputs['Value'].is_linked:
                    raise RuntimeError('Textured leaf-group parameters need a separate translation')
                group_recipe=dict(leaf_shader='original_alpha_mix',
                    transmission_factor=float(group.inputs['Translucency'].default_value),
                    albedo_value_scale=float(group.inputs['Value'].default_value))
            if principled is None:raise RuntimeError('Unsupported original provider surface shader')
            def optical_scalar(name):
                socket=principled.inputs[name]
                if not socket.is_linked:return float(socket.default_value)
                if group and len(socket.links)==1 and socket.links[0].from_node.type=='GROUP_INPUT':
                    outer=group.inputs[socket.links[0].from_socket.name]
                    if not outer.is_linked:return float(outer.default_value)
                raise RuntimeError('Provider needs textured optical parameter translation')
            for socket in ('IOR', 'Specular IOR Level'):
                optical_scalar(socket)
            for node in (mapping, box_mapping):
                if node and (any(node.inputs['Location'].default_value) or any(node.inputs['Rotation'].default_value)):
                    raise RuntimeError('Untranslated provider mapping transform')
            recipes[material_name] = dict(
                uv_scale=list(mapping.inputs['Scale'].default_value[:2]) if mapping else [1., 1.],
                box_scale=list(box_mapping.inputs['Scale'].default_value) if box_mapping else [1.4, 1.4, .7],
                ior=optical_scalar('IOR'),specular_level=optical_scalar('Specular IOR Level'),**group_recipe)
            if spec.get('texture_prefix'):
                recipes[material_name]['texture_prefix'] = spec['texture_prefix']
                uv_mapping=mapping or box_mapping
                if uv_mapping:recipes[material_name]['uv_scale']=list(uv_mapping.inputs['Scale'].default_value[:2])
            if spec.get('derive_atlas_cutout_from_graph'):
                # Only textures upstream of the active material output count;
                # unconnected provider preview nodes must not change opacity.
                outputs=[n for n in tree.nodes if n.type=='OUTPUT_MATERIAL' and n.is_active_output]
                reachable=set();pending=list(outputs)
                while pending:
                    node=pending.pop()
                    if node in reachable:continue
                    reachable.add(node)
                    pending.extend(link.from_node for socket in node.inputs for link in socket.links)
                alpha=any(n.type=='TEX_IMAGE' and n.image
                    and ('_alpha_' in Path(n.image.filepath).name or '_alpha.' in n.image.name)
                    for n in reachable)
                alpha_socket=group.inputs['Alpha'] if group else principled.inputs['Alpha']
                if alpha_socket.is_linked and not alpha:
                    raise RuntimeError('Connected original alpha was not resolved to its source map')
                translucent=bool(group) or any(n.type=='BSDF_TRANSLUCENT' for n in reachable)
                recipes[material_name].update(cutout=alpha,thin_leaf=alpha and translucent)
        array_path = directory / (name + '.npz')
        np.savez(array_path, **{k:v for k,v in arrays.items() if isinstance(v, np.ndarray)})
        contact = {}
        if spec.get('placement_class') in ('embedded_rock', 'embedded_woody_root'):
            # A rock's curved upper rim is not a flat root cut. Anchor its
            # volumetric basal portion; burying the entire silhouette would
            # incorrectly push an almost complete rock below the ground.
            vertices = arrays['points'] @ np.asarray(arrays['original_metric_basis'], np.float64).T
            vertices[:, 2] += arrays['root_offset_z_m']
            rock = spec['placement_class'] == 'embedded_rock'
            band_height = .2 * arrays['dimensions_m'][2] if rock else min(.6, .1 * arrays['dimensions_m'][2])
            vertices = vertices[vertices[:, 2] <= band_height]
            cells = np.floor(vertices[:, :2] / .025).astype(np.int32)
            order = np.lexsort((vertices[:, 2], cells[:, 1], cells[:, 0]))
            cells, vertices = cells[order], vertices[order]
            first = np.r_[True, np.any(cells[1:] != cells[:-1], axis=1)]
            arrays['contact_anchors_local_m'] = vertices[first].tolist()
            contact = dict(placement_class=spec['placement_class'], maximum_burial_m=.6 if rock else 1.,
                           maximum_burial_fraction_of_height=.6 if rock else .1,
                           root_anchor_policy='original basal lower envelope at2.5cm; band height recorded',
                           contact_envelope_height_m=band_height)
        prototype_records.append(dict(object_name=name, array_file=str(array_path),
            material_recipes=recipes, source_blend_sha256=source['sha256'],
            vertices=len(arrays['points']), polygons=len(arrays['counts']),
            triangles=int(np.sum(arrays['counts'] - 2)),
            source_array_sha256={k:hashlib.sha256(arrays[k].tobytes()).hexdigest()
                                for k in ('points', 'indices', 'counts', 'uvs')},
            **{k:v for k,v in arrays.items() if not isinstance(v, np.ndarray) and k != 'mask'},
            **contact))
    record = dict(asset_id=card['asset_id'], original_card=str(Path(spec['card']).resolve()),
        original_card_sha256=sha256_file(Path(spec['card'])), licence=card['licence'],
        usd=str(usd), raw_usd_sha256=sha256_file(usd), prototypes=prototype_records,
        qualification='not_run', native_material_authoring='next_separate_USD_worker',
        material_translation_policy=spec.get('material_override_reason','original inspected shader translation'),
        blend_mask_channel=spec.get('blend_mask_channel','luminance'))
    atomic_json(directory / 'geometry_export.json', record)
    records.append(record)
    print({'asset':card['asset_id'], 'whole_original_prototypes':len(prototype_records)}, flush=True)
atomic_json(out / 'geometry_library.json', dict(status='native_conifer_candidates', assets=records,
    at_utc=utc_now(), blender_version=bpy.app.version_string,
    producer_sha256=sha256_file(Path(__file__)), qualification='not_run'))
