"""Read original provider node data; never execute scripts from a blend file."""
from pathlib import Path
import json,sys
import bpy
request=json.loads(Path(sys.argv[sys.argv.index('--')+1]).read_text())
def value(x):
    if isinstance(x,(bool,int,float,str)):return x
    try:return list(x)
    except TypeError:return str(type(x).__name__)
def graph(tree,depth=0):
    result=[]
    for node in tree.nodes:
        row=dict(name=node.name,type=node.type,inputs={s.name:dict(linked=s.is_linked,
            value=value(s.default_value) if hasattr(s,'default_value') else None,
            links=[dict(node=l.from_node.name,socket=l.from_socket.name) for l in s.links]) for s in node.inputs})
        row['input_sockets']=[dict(index=i,name=s.name,value=value(s.default_value) if hasattr(s,'default_value') else None,
            links=[dict(node=l.from_node.name,socket=l.from_socket.name) for l in s.links]) for i,s in enumerate(node.inputs)]
        if hasattr(node,'operation'):row['operation']=node.operation
        if hasattr(node,'blend_type'):row['blend_type']=node.blend_type
        if node.type=='TEX_IMAGE' and node.image:row['image']=dict(name=node.image.name,path=node.image.filepath)
        if node.type=='GROUP' and depth<3:row['group']=graph(node.node_tree,depth+1)
        result.append(row)
    return result
output=[]
for spec in request['assets']:
    card=json.loads(Path(spec['card']).read_text());source=next(f for f in card['files'] if f['path'].endswith('.blend'))
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.preferences.filepaths.use_scripts_auto_execute=False
    with bpy.data.libraries.load(source['path'],link=False) as (available,loaded):loaded.objects=spec['objects']
    names=sorted({m.name for obj in loaded.objects for m in obj.data.materials if m})
    output.append(dict(asset_id=card['asset_id'],materials={n:graph(bpy.data.materials[n].node_tree) for n in names}))
Path(request['graph_output']).write_text(json.dumps(output,indent=2)+'\n')
