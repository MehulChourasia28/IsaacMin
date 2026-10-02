"""Independent PIL decode of native bake smoke; does not fabricate Isaac evidence."""
import argparse,json,numpy as np
from pathlib import Path
from PIL import Image
from isaacmin.assembly.material_bake import _mix
from isaacmin.assembly.material_assignment import material_blend_weights,REQUIRED
from isaacmin.assets.network import atomic_json,digest
p=argparse.ArgumentParser();p.add_argument('directory');args=p.parse_args();root=Path.cwd();out=Path(args.directory).resolve()
report=json.loads((out/'bakes/transition_bakes.json').read_text());tile=report['tiles'][0]
materials=json.loads((root/'state/terrain_materials.json').read_text())['materials'];maps={}
for name in REQUIRED:
 spec=next(m for m in materials if m['asset_id']==name);images={}
 for role in ('base_color','normal','roughness'):
  a=np.asarray(Image.open(spec[role]).convert('RGB'),np.float32)[::-1]/255
  if role=='base_color':a=np.where(a<=.04045,a/12.92,((a+.055)/1.055)**2.4)
  images[role]=a
 maps[name]=(spec,images)
with np.load(out/'fixture_source.npz') as data:s={k:data[k] for k in data.files}
coord=np.array([[.501,.501],[.249,.249],[.75,.25]])
x=np.floor((coord[:,0]-tile['world_chart_min'][0])*1024).astype(int);y=np.floor((coord[:,1]-tile['world_chart_min'][1])*1024).astype(int)
coord=np.column_stack((x+.5,y+.5))/1024+tile['world_chart_min']
vertex_weights=material_blend_weights(np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]]),np.tile([0,0,1],(4,1)),s,[0,0,0])
weights=[]
for px,py in coord:
 weights.append(np.array([1-px,px-py,py])@vertex_weights[[0,1,2]] if px>=py else np.array([1-py,px,py-px])@vertex_weights[[0,2,3]])
weights=np.asarray(weights);expected=_mix(maps,coord,weights);errors={}
for (role,c),values in zip([('base_color',3),('roughness',1),('normal',3)],expected):
 f=next(f for f in tile['files'] if f['role']==role);image=Image.open(f['path'])
 if c==1:im=np.asarray(image,np.float32)[::-1,:,None]/65535
 else:im=np.asarray(image.convert('RGB'),np.float32)[::-1]/255
 observed=im[y,x,:c];errors[role]=float(np.max(np.abs(observed-values[:,:c])))
r={'scope':'actual_native_blender_export_of_controlled_unit_geometry_using_real_licensed_PBRs','geometry_unchanged':report['geometry_points_sha256_before']==report['geometry_points_sha256_after'],'independent_source_sample_errors':errors,'tolerance_encoded_8bit_comparison':2/255,'status':'pass' if max(errors.values())<2/255 else 'fail','native_target_isaac':'not_run','weights':'independently computed source vertex weights interpolated with actual triangle barycentrics','bake_manifest_sha256':digest(out/'bakes/transition_bakes.json')}
atomic_json(out/'independent_check.json',r);print(r)
