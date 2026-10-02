"""Make an immutable actual-scene candidate retaining original filtered alpha."""
from pathlib import Path
import argparse
import os
import shutil
from isaacmin.io import read_json,atomic_json,sha256_file
from isaacmin.assembly.usd_instances import _closure
parser=argparse.ArgumentParser();parser.add_argument('--request',required=True)
a=parser.parse_args();r=read_json(Path(a.request));source=Path(r['source']);out=Path(r['output'])
out.mkdir(parents=True,exist_ok=False)
for row in read_json(source/'native_dependency_closure.json')['files']:
    p=source/row['path'];q=out/row['path']
    if p.is_symlink() or not p.resolve().is_relative_to(source) or sha256_file(p)!=row['sha256']:
        raise ValueError('Native dependency changed')
    q.parent.mkdir(parents=True,exist_ok=True)
    if p.suffix in ('.mdl','.usda','.json'):shutil.copyfile(p,q)
    else:os.link(p,q)
changed=[]
for p in (out/'canopy_library').rglob('*_twig.mdl'):
    old=p.read_text();new=old.replace('cutout_opacity:alpha >= 0.5 ? 1.0 : 0.0','cutout_opacity:math::clamp(alpha,0.0,1.0)')
    if old==new:raise ValueError('Expected explicit source opacity mismatch')
    p.write_text(new);changed.append(str(p.relative_to(out)))
if not changed:raise ValueError('No original needle materials found')
atomic_json(out/'native_dependency_closure.json',_closure(out/'world.usda'))
atomic_json(out.parent/'comparison.json',dict(status='candidate_authored',source_scene=str(source/'world.usda'),
    changed_modules=changed,change='preserve original filtered opacity instead of added hard half-threshold',
    geometry_placements_original_textures_lighting_unchanged=True,qualification='not_run'))
