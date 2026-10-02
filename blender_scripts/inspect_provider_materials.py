"""Read downloaded Blender material data without loading objects or scripts."""
from pathlib import Path
import json
import sys
import bpy

request = json.loads(Path(sys.argv[sys.argv.index('--') + 1]).read_text())


def graph(tree, seen=None):
    seen = set() if seen is None else seen
    if tree.name in seen:
        return {'reference': tree.name}
    seen.add(tree.name)
    records = []
    for node in tree.nodes:
        row = {'name': node.name, 'type': node.bl_idname, 'inputs': {}}
        row['properties'] = {name: getattr(node, name) for name in
            ('attribute_name', 'uv_map', 'operation', 'vector_type', 'projection',
             'projection_blend', 'layer_name', 'blend_type', 'distribution')
            if hasattr(node, name)}
        row['input_links'] = [{'index': i, 'name': socket.name,
            'links': [link.from_node.name + ':' + link.from_socket.name for link in socket.links]}
            for i, socket in enumerate(node.inputs)]
        for socket in node.inputs:
            value = getattr(socket, 'default_value', None)
            try:
                value = list(value)
            except TypeError:
                pass
            row['inputs'][socket.name] = {'default': value,
                'links': [link.from_node.name + ':' + link.from_socket.name for link in socket.links]}
        if node.type == 'TEX_IMAGE' and node.image:
            row['image'] = {'name': node.image.name, 'path': node.image.filepath,
                            'color_space': node.image.colorspace_settings.name}
        if node.type == 'GROUP' and node.node_tree:
            row['group'] = graph(node.node_tree, seen)
        records.append(row)
    return {'name': tree.name, 'nodes': records}


records = []
for asset in request['assets']:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    with bpy.data.libraries.load(asset['blend'], link=False) as (available, loaded):
        loaded.materials = available.materials
    records.append({'asset_id': asset['asset_id'], 'materials': [
        {'name': material.name, 'graph': graph(material.node_tree)}
        for material in loaded.materials if material and material.use_nodes]})
Path(request['output']).write_text(json.dumps({'blender_version': bpy.app.version_string,
    'assets': records, 'scripts_enabled': False}, indent=2) + '\n')
