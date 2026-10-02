"""An immutable same-world comparison restoring inspected provider optics."""
from pathlib import Path
import argparse
import os
import shutil
from isaacmin.io import read_json, atomic_json, sha256_file
from isaacmin.assembly.usd_instances import _closure
from isaacmin.assembly.provider_pbr import principled_specular_ior

parser=argparse.ArgumentParser(); parser.add_argument('--request',required=True)
r=read_json(Path(parser.parse_args().request))
source=Path(r['source']); out=Path(r['output']); details=Path(r['original_material_inspection'])
out.mkdir(parents=True,exist_ok=False)
for row in read_json(source/'native_dependency_closure.json')['files']:
    p=source/row['path']; q=out/row['path']
    if p.is_symlink() or not p.resolve().is_relative_to(source) or sha256_file(p)!=row['sha256']:
        raise ValueError('Native dependency changed')
    q.parent.mkdir(parents=True,exist_ok=True)
    if p.suffix in ('.mdl','.usda','.json'):shutil.copyfile(p,q)
    else:os.link(p,q)
original={}
for asset in read_json(details)['assets']:
    for material in asset['materials']:
        node=next(n for n in material['graph']['nodes'] if n['type']=='ShaderNodeBsdfPrincipled')
        fields=[node['inputs'][k] for k in ('IOR','Specular IOR Level')]
        if any(f['links'] for f in fields):raise ValueError('Textured original optical control')
        original[material['name']]=tuple(f['default'] for f in fields)
changed=[]
for p in (out/'canopy_library').rglob('*.mdl'):
    if p.stem not in original:continue
    ior,level=original[p.stem]; eta=principled_specular_ior(ior,level)
    old=p.read_text()
    new=old.replace('ior:1.5,weight:1.0','ior:'+repr(eta)+',weight:1.0').replace('ior:color(1.5)','ior:color('+repr(ior)+')')
    if old==new:raise ValueError('Expected original optical mismatch')
    p.write_text(new)
    changed.append(dict(module=str(p.relative_to(out)), original_ior=ior,
                        original_specular_level=level, translated_specular_ior=eta))
if not changed:raise ValueError('No original materials found')
atomic_json(out/'native_dependency_closure.json',_closure(out/'world.usda'))
atomic_json(out.parent/'comparison.json',dict(status='candidate_authored', source_scene=str(source/'world.usda'),
    original_material_inspection=str(details), original_material_inspection_sha256=sha256_file(details),
    changed_modules=changed, change='restore original Principled dielectric and specular IOR adjustment',
    source_formula='Blender intern/cycles/kernel/svm/closure.h:386-395',
    geometry_placements_original_textures_lighting_unchanged=True, qualification='not_run'))
