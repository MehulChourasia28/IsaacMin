"""Isolate custom MDL translation with original-map USD Preview Surface."""
from pathlib import Path
import argparse
import os
import shutil
from pxr import Usd, UsdShade, Sdf, Gf
from isaacmin.io import read_json,atomic_json,sha256_file
from isaacmin.assembly.usd_instances import _closure
from isaacmin.assembly.provider_pbr import principled_specular_ior

parser=argparse.ArgumentParser();parser.add_argument('--request',required=True)
r=read_json(Path(parser.parse_args().request));source=Path(r['source']);out=Path(r['output'])
out.mkdir(parents=True,exist_ok=False)
for row in read_json(source/'native_dependency_closure.json')['files']:
    p=source/row['path'];q=out/row['path']
    if p.is_symlink() or not p.resolve().is_relative_to(source) or sha256_file(p)!=row['sha256']:
        raise ValueError('Native dependency changed')
    q.parent.mkdir(parents=True,exist_ok=True)
    if p.suffix in ('.mdl','.usda','.json') or ('canopy_library' in p.parts and p.suffix=='.usdc'):
        shutil.copyfile(p,q)
    else:os.link(p,q)
changed=[]
for usd in (out/'canopy_library').rglob('library.usdc'):
    stage=Usd.Stage.Open(str(usd));aid=usd.parent.name
    for prim in list(stage.Traverse()):
        if not prim.IsA(UsdShade.Material) or not prim.GetName().endswith('_twig'):continue
        mat=UsdShade.Material(prim);base=prim.GetPath();stem=prim.GetName()
        prim.RemoveProperty('outputs:mdl:surface')
        shader=UsdShade.Shader.Define(stage,base.AppendChild('PreviewSurface'))
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.8)
        shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(0.)
        shader.CreateInput('ior',Sdf.ValueTypeNames.Float).Set(principled_specular_ior(1.45,.01))
        shader.CreateInput('opacityThreshold',Sdf.ValueTypeNames.Float).Set(0.)
        reader=UsdShade.Shader.Define(stage,base.AppendChild('UV'))
        reader.CreateIdAttr('UsdPrimvarReader_float2');reader.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
        reader.CreateOutput('result',Sdf.ValueTypeNames.Float2)
        for role,input_name,gamma in [('diff','diffuseColor','sRGB'),('rough','roughness','raw'),('alpha','opacity','raw'),('nor_gl','normal','raw')]:
            file='./textures/'+stem+'_'+role+'_4k.png'
            if not (usd.parent/file).is_file():raise ValueError('Original texture missing')
            tex=UsdShade.Shader.Define(stage,base.AppendChild('Texture_'+role));tex.CreateIdAttr('UsdUVTexture')
            tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(file))
            tex.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set(gamma)
            tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(),'result')
            vector=role in ('diff','nor_gl');output='rgb' if vector else 'r'
            tex.CreateOutput(output,Sdf.ValueTypeNames.Float3 if vector else Sdf.ValueTypeNames.Float)
            if role=='nor_gl':
                tex.CreateInput('scale',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2,2,2,1))
                tex.CreateInput('bias',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1,-1,-1,0))
            shader.CreateInput(input_name,Sdf.ValueTypeNames.Normal3f if role=='nor_gl' else
                Sdf.ValueTypeNames.Color3f if vector else Sdf.ValueTypeNames.Float).ConnectToSource(tex.ConnectableAPI(),output)
        shader.CreateOutput('surface',Sdf.ValueTypeNames.Token)
        mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
        changed.append(str(base))
    stage.GetRootLayer().Save()
atomic_json(out/'native_dependency_closure.json',_closure(out/'world.usda'))
atomic_json(out.parent/'comparison.json',dict(status='diagnostic_authored',source_scene=str(source/'world.usda'),
    changed_materials=changed,geometry_placements_original_textures_lighting_unchanged=True,
    purpose='isolate custom MDL shader from original texture/UV and native geometry import',
    limitation='USD Preview Surface lacks thin-leaf diffuse transmission; diagnostic only',qualification='not_run'))
