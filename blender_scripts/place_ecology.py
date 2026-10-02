"""Ground source-informed ecological candidates against actual assembled geometry."""
import bpy
import hashlib
import json
import math
import sys
from pathlib import Path
import numpy as np

request=json.loads(Path(sys.argv[sys.argv.index('--')+1]).read_text())
workspace=Path(request['workspace']).resolve()
sys.path.insert(0,str(workspace/'src'))
sys.path.insert(0,str(workspace/'blender_scripts'))
from isaacmin.assembly.ecology import scatter_candidates,ground_prototype,PolylineClearance
from ground_queries import NativeGroundQuery
from isaacmin.assets.canopy import load_canopy_candidates

scene_path=Path(request['scene_blend']).resolve()
with scene_path.open('rb') as scene_stream:
    input_scene_sha256=hashlib.file_digest(scene_stream,'sha256').hexdigest()
bpy.ops.wm.open_mainfile(filepath=str(scene_path),use_scripts=False,load_ui=False)
terrain=bpy.data.objects.get('Terrain_FinalGround')
if terrain is None or terrain.type!='MESH':raise RuntimeError('Actual final assembled terrain mesh is required')
depsgraph=bpy.context.evaluated_depsgraph_get()
inventory=request.get('resource_inventory',{})
ground_query=NativeGroundQuery(terrain,depsgraph,expected_vertices=inventory.get('vertices'),expected_triangles=inventory.get('triangles'))
mesh_sha=ground_query.vertices_sha256
ir=Path(request['ir_directory']);manifest=json.loads((ir/'world_ir.json').read_text())
ox,oy,oz=manifest['coordinate_frame']['source_origin_xyz']
xmin,zmin,xmax,zmax=manifest['scope']['bounds_blocks_xz']
surface=np.load(ir/'terrain_surface.npz',allow_pickle=False)
height=surface['height'];valid=surface['validity'];biome=surface['biome'];water=surface['water_height'];water_valid=surface['water_validity']
substrate=surface['substrate_id'];names=surface['block_names']
canopy=np.zeros_like(height,dtype=float)
tree_columns={}
for chunk_path in (ir/'terrain_volume').glob('*.npz'):
    chunk=np.load(chunk_path,allow_pickle=False);palette=json.loads(str(chunk['palette_json']))
    cx,_,cz=chunk['min_xyz'];rx,rz=int(cx)-xmin,int(cz)-zmin
    ids=[i for i,p in enumerate(palette) if p['Name'].endswith('_leaves')]
    if ids:
        leaves=np.isin(chunk['block_id'],ids).any(axis=(0,1))
        canopy[rz:rz+16,rx:rx+16]=leaves
    for species,block_name in [('white_birch','minecraft:birch_log'),('oak','minecraft:oak_log')]:
        ids=[i for i,p in enumerate(palette) if p['Name']==block_name]
        if not ids:continue
        indices=np.argwhere(np.isin(chunk['block_id'],ids))
        for si,ly,lz,lx in indices:
            key=(species,int(cx)+int(lx),int(cz)+int(lz));block_y=int(chunk['section_y'][si])*16+int(ly)
            tree_columns[key]=min(tree_columns.get(key,10**6),block_y)
# Local canopy closure from actual source crown occupancy, not from biome alone.
pad=np.pad(canopy,4,mode='constant');filtered=np.zeros_like(canopy)
for dz in range(9):
    for dx in range(9):filtered+=pad[dz:dz+canopy.shape[0],dx:dx+canopy.shape[1]]/81
canopy=filtered
route=np.zeros_like(height,dtype=bool)
route_distance=np.full(height.shape,1000.,dtype=float)
route_path=request.get('route_path')
route_probe=None
if route_path:
    if Path(route_path).suffix=='.json':
        route_file=json.loads(Path(route_path).read_text())
        points=np.asarray(route_file['points_world_xyz'],dtype=float)
        if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():raise RuntimeError('Invalid authored route coordinates')
        iz,ix=np.indices(height.shape)
        world_xy=np.stack((ix+xmin+.5-ox,-(iz+zmin+.5)+oz),axis=-1)
        width=float(route_file['corridor_width_m'])
        route_probe=PolylineClearance(points,width)
        for a,b in zip(points[:-1,:2],points[1:,:2]):
            ab=b-a;denom=float(np.dot(ab,ab))
            if denom==0:continue
            t=np.clip(np.sum((world_xy-a)*ab,axis=-1)/denom,0,1)
            distance=np.linalg.norm(world_xy-(a+t[...,None]*ab),axis=-1)-width/2
            route_distance=np.minimum(route_distance,distance)
        route=route_distance<=0
    else:
        route_file=np.load(route_path,allow_pickle=False)
        key=request.get('route_mask_key','corridor_mask')
        if key not in route_file:raise RuntimeError('Explicit route mask key unavailable')
        route=route_file[key].astype(bool)
        if route.shape!=height.shape:raise RuntimeError('Route mask shape mismatch')
        route_distance=np.where(route,0,1000)
runoff=None
if request.get('runoff_path'):
    runoff_file=np.load(request['runoff_path'],allow_pickle=False)
    runoff=runoff_file[request.get('runoff_key','runoff')]
    if runoff.shape!=height.shape:raise RuntimeError('Runoff shape mismatch')
    runoff=np.log1p(np.maximum(runoff,0));runoff/=max(1e-8,float(np.percentile(runoff,95)))


def support(x,y,hint):
    return ground_query.support(x,y,hint)


def fields(x,y):
    ix,iz=math.floor(x+ox-xmin),math.floor(-y+oz-zmin)
    if not (0<=ix<height.shape[1] and 0<=iz<height.shape[0]):return {'valid':False}
    ground=support(x,y,None)
    if ground is None:return {'valid':False}
    name=str(names[substrate[iz,ix]])
    category='soil' if any(s in name for s in ('dirt','grass_block','podzol','mud','mycelium','moss_block')) else 'sediment' if any(s in name for s in ('sand','gravel','clay')) else 'rock'
    moisture=float(np.clip(.4+.2*canopy[iz,ix]+(.3*runoff[iz,ix] if runoff is not None else 0),0,1))
    return {'valid':bool(valid[iz,ix]),'biome':str(biome[iz,ix]),'substrate':category,
            'slope_degrees':math.degrees(math.acos(np.clip(ground['normal'][2],-1,1))),
            'moisture':moisture,'canopy':float(canopy[iz,ix]),
            'water_depth_m':max(.001,float(water[iz,ix]-oy-ground['z'])) if water_valid[iz,ix] and water[iz,ix]>=height[iz,ix] else 0,
            'route_distance_m':route_probe(x,y) if route_probe is not None else float(route_distance[iz,ix]),
            'moisture_provenance':'inferred canopy/runoff proxy; not hydrological ground truth',
            'canopy_provenance':'source leaf occupancy9m neighbourhood; final canopy placement unqualified'}


catalogue=json.loads((workspace/'state/normalized_assets.json').read_text())['assets']
result=scatter_candidates([xmin-ox,-zmax+oz,xmax-ox,-zmin+oz],fields,support,catalogue,
                          seed=request.get('seed',1729),require_qualified=False,cell_size_m=.5)
result['supporting_geometry_sha256']=mesh_sha
result['ground_query']=ground_query.evidence
result['resource_inventory']=inventory
with scene_path.open('rb') as scene_stream:
    result['supporting_scene_sha256']=hashlib.file_digest(scene_stream,'sha256').hexdigest()
if result['supporting_scene_sha256']!=input_scene_sha256:
    raise RuntimeError('Supporting scene changed during native ecological placement')
result['source_ir_sha256']=manifest['content_sha256']
result['route_status']='source-derived or authored corridor mask used' if route_path else 'not_supplied; no trail access claim'
result['field_defaults']={'season':'late_spring','moisture':'0.4 +0.2 canopy +0.3 normalized runoff when present','canopy':'actual source leaves local9m mean'}
result['exporter_assets']=[{'id':i['instance_id'],'blend':i['blend_path'],'objects':[i['object_name']],
                          'asset_id':i['asset_id'],'alpha_mode':'opaque' if i['guild'] in ('deadwood','mossy_boulder') else 'cutout',
                          'world_transform':i['world_transform'],'position':[i['world_transform'][j][3] for j in range(3)],
                          'scale':i['scale'],'yaw':0} for i in result['instances']]
tree_reports=[]
occupied_roots=set()
canopy_library=load_canopy_candidates(workspace)
for (species,mx,mz),base_y in sorted(tree_columns.items()):
    ix,iz=mx-xmin,mz-zmin
    if not (0<=ix<height.shape[1] and 0<=iz<height.shape[0]) or not 0<=base_y-height[iz,ix]<=2:
        continue
    if any((species,mx+dx,mz+dz) in occupied_roots for dx in (-1,0,1) for dz in (-1,0,1)):
        continue
    occupied_roots.add((species,mx,mz))
    candidates=canopy_library.get(species,[])
    if not candidates:
        tree_reports.append({'source_root_xz':[mx,mz],'species':species,'status':'missing_generated_asset'})
        continue
    identity=hashlib.sha256(f'source_tree:{species}:{mx}:{mz}:1729'.encode()).hexdigest()
    asset,variant=candidates[int(identity[:8],16)%len(candidates)]
    if not variant.get('contact_anchors_local_m'):
        tree_reports.append({'source_root_xz':[mx,mz],'species':species,'status':'missing_measured_root_anchors'})
        continue
    x,y=mx+.5-ox,-mz-.5+oz
    sample=fields(x,y)
    if not sample.get('valid') or sample.get('water_depth_m',1)>0 or sample.get('route_distance_m',0)<2:
        tree_reports.append({'source_root_xz':[mx,mz],'species':species,'status':'source_water_route_or_validity_conflict'})
        continue
    # Preserve trunk gravity alignment; reject incompatible root footprint rather
    # than tipping an entire tree to match a steep ground normal.
    def upright_support(ax,ay,hint):
        result=support(ax,ay,hint)
        if result:result['normal']=[0,0,1]
        return result
    grounded,failure=ground_prototype(variant,x,y,1.0,int(identity[8:16],16)/2**32*2*math.pi,upright_support)
    if failure:
        tree_reports.append({'source_root_xz':[mx,mz],'species':species,'status':failure})
        continue
    result['exporter_assets'].append({'id':identity[:24],'blend':asset['output_blend'],'objects':variant['objects'],
                                     'asset_id':asset['asset_id'],'alpha_mode':'cutout',
                                     'attribution_files':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                                         for p in sorted(Path(asset['maps_manifest']).parent.iterdir())
                                         if p.is_file() and (p.name=='CREDITS.txt' or 'licence' in p.name.lower())],
                                     'world_transform':grounded['world_transform'],'position':[grounded['world_transform'][j][3] for j in range(3)],'scale':1,'yaw':0})
    tree_reports.append({'source_root_xz':[mx,mz],'species':species,'status':'candidate_placed','provenance':'source-observed root position, synthesized botanical geometry',
                         'generator_sha256':asset.get('output_sha256'),'variant_seed':variant['seed'],**grounded})
result['canopy_source_roots']=tree_reports
result['canopy_quality']='not_run; failed/unavailable roots remain explicit ecological coverage gaps'
out=Path(request['output_report']);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2))
