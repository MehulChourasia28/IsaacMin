"""Preserve native geometry while correcting foliage sides and root envelopes."""
from pathlib import Path
import argparse
import os
import shutil
import numpy as np
from pxr import Usd, UsdGeom
from isaacmin.io import read_json, atomic_json, sha256_file, utc_now
from isaacmin.assembly.usd_instances import _closure
from isaacmin.assembly.provider_pbr import provider_material

parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True)
args=parser.parse_args();request=read_json(args.request)
root=Path(request['workspace']);source=Path(request['source']);out=Path(request['output'])
out.mkdir(parents=True,exist_ok=False);library=read_json(source)
optics={}
if request.get('original_material_inspection'):
    for asset in read_json(Path(request['original_material_inspection']))['assets']:
        for material in asset['materials']:
            node=next(n for n in material['graph']['nodes'] if n['type']=='ShaderNodeBsdfPrincipled')
            ior=node['inputs']['IOR'];level=node['inputs']['Specular IOR Level']
            if ior['links'] or level['links']:raise ValueError('Textured original optics need explicit translation')
            optics[material['name']]=dict(ior=ior['default'],specular_level=level['default'])
for asset in library['assets']:
    old=Path(asset['usd']);directory=out/asset['asset_id'];directory.mkdir()
    if sha256_file(old)!=asset['usd_sha256']:raise ValueError('Native library changed')
    closure=read_json(old.parent/'native_dependency_closure.json')
    for item in closure['files']:
        p=old.parent/item['path'];q=directory/item['path']
        if p.is_symlink() or not p.resolve().is_relative_to(old.parent) or sha256_file(p)!=item['sha256']:
            raise ValueError('Native dependency mismatch')
        q.parent.mkdir(parents=True,exist_ok=True)
        if p.suffix=='.mdl':continue  # All modules are regenerated from original maps.
        if p.suffix in ('.usdc','.usda','.json'):shutil.copyfile(p,q)
        else:os.link(p,q)
    usd=directory/old.name;stage=Usd.Stage.Open(str(usd));card=read_json(Path(asset['original_card']))
    for proto in asset['prototypes']:
        top=stage.GetPrimAtPath(proto['usd_prim'])
        mesh=UsdGeom.Mesh(next(p for p in Usd.PrimRange(top) if p.IsA(UsdGeom.Mesh)))
        mesh.CreateDoubleSidedAttr().Set(True)
        arrays=np.load(proto['array_file']);points=arrays['points']
        if not np.array_equal(np.asarray(mesh.GetPointsAttr().Get()),points):
            raise RuntimeError('Repair must preserve original vertex positions')
        if asset['asset_id']=='pine_tree_01':
            local=points.astype(np.float64);local[:,2]+=proto['root_offset_z_m']
            base=local[local[:,2]<=.6]
            # The scan includes broad, raised root/soil flares. The old lowest
            # 2cm band missed those outer surfaces and could float on a slope.
            cells=np.floor(base[:,:2]/.025).astype(np.int32)
            order=np.lexsort((base[:,2],cells[:,1],cells[:,0]));cells=cells[order];base=base[order]
            first=np.r_[True,np.any(cells[1:]!=cells[:-1],axis=1)]
            envelope=base[first]
            proto['contact_anchors_local_m']=envelope.tolist()
            proto['root_anchor_policy']='2.5cm cells over complete lowest60cm root/soil flare; minimum original vertex per occupied cell'
            proto['maximum_burial_fraction_of_height']=request.get('maximum_burial_fraction_of_height',.05)
            proto['maximum_burial_m']=1.0
        material_records=[]
        for name,recipe in proto['material_recipes'].items():
            recipe.update(optics.get(name,{}))
            _,receipt=provider_material(root,stage,top.GetPath(),directory,card,name,**recipe)
            material_records.append(receipt)
        proto['materials']=material_records
        proto['native_foliage_sides']='double_sided; original mesh and original cutout unchanged'
        proto['UV_lookup']='explicit original st primvar'
        arrays.close()
    stage.GetRootLayer().Save()
    atomic_json(directory/'native_dependency_closure.json',_closure(usd))
    asset.update(usd=str(usd),usd_sha256=sha256_file(usd),repair_parent=str(old))
    atomic_json(directory/'export.json',asset)
library.update(updated_at_utc=utc_now(),qualification='not_run',
    repair_parent_sha256=sha256_file(source),repair_producer_sha256=sha256_file(Path(__file__)))
if request.get('original_material_inspection'):
    library['original_material_inspection_sha256']=sha256_file(Path(request['original_material_inspection']))
atomic_json(out/'library.json',library)
print({'native_library_ready':True,'original_geometry_changed':False})
