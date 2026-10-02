"""Reproducible candidate preparation from acquired, checksum-verified masters."""
from pathlib import Path
import json

from .network import ServiceError, atomic_json, digest, utcnow
from .normalize import normalize_asset

UNDERSTORY_IDS = ('grass_medium_01','grass_medium_02','dandelion_01','celandine_01',
                  'fern_02','dead_tree_trunk','rock_moss_set_01')
MATERIAL_IDS = ('brown_mud_dry','rock_face_03','leafy_grass','forest_ground_04','rock_boulder_dry')


def prepare_lighting(workspace: Path):
    from isaacmin.process import run_worker
    from isaacmin.adapters.workers import native_environment
    workspace=Path(workspace).resolve()
    catalogue=json.loads((workspace/'state/asset_catalogue.json').read_text())['assets']
    images=[{'asset_id':a['asset_id'],'path':f['path']} for a in catalogue if a['asset_id'] in
            ('kloofendal_48d_partly_cloudy_puresky','kloofendal_overcast_puresky') for f in a['files'] if f.get('role')=='hdri']
    if len(images)!=2:raise ServiceError('missing_lighting_master','Acquire both prescribed Pure Sky HDRIs before native decode')
    directory=workspace/'evidence/services/lighting';directory.mkdir(parents=True,exist_ok=True)
    output=directory/'native_hdri_decode.json'
    old=json.loads(output.read_text()).get('images',[]) if output.exists() else []
    if len(old)==2 and all(any(d['asset_id']==i['asset_id'] and d['sha256']==digest(Path(i['path'])) for d in old) for i in images):
        return {'status':'external_tool_verified','cached':True,'report':str(output)}
    candidates=[workspace/'.tools/blender/bin/blender',workspace/'.tools/blender/blender']
    blender=next((p for p in candidates if p.is_file()),None)
    if blender is None:raise ServiceError('blocked_dependency','Installed native Blender with scripts/data is required')
    atomic_json(directory/'request.json',{'images':images,'output':str(output)})
    result=run_worker([str(blender),'-b','--factory-startup','--disable-autoexec','--python-use-system-env',
                      '--python-exit-code','1','--python',str(workspace/'blender_scripts/inspect_hdri.py'),'--',str(directory/'request.json')],
                      cwd=workspace,log_path=directory/'decode.log',timeout=300,environment=native_environment(workspace))
    if result['exit_code']!=0 or not output.exists():raise ServiceError('lighting_decode','Native lighting decode failed; logs retained')
    return {'status':'external_tool_verified','report':str(output),'process':result}


def procedural_manifest(workspace: Path):
    workspace=Path(workspace).resolve()
    report={'schema_version':1,'created_at_utc':utcnow(),'assets':[],
            'qualification':'not_run','geometry_sharing':'one mesh prototype per object and species; instance placement requires USD prototype verification'}
    for species in ('white_birch','oak'):
        folder=workspace/'assets/procedural'/species
        source=folder/'generation.json'
        record=json.loads(source.read_text())
        blend=Path(record['output_blend'])
        if digest(blend)!=record['output_sha256']:
            raise ServiceError('procedural_corruption','Generated canopy bytes differ from native provenance')
        paths=[blend,source,folder/'request.json',folder/'maps/maps_manifest.json',
               workspace/'blender_scripts/generate_birch.py',workspace/'scripts/generate_birch_textures.py',workspace/'scripts/prepare_oak_maps.py']
        paths+=list((folder/'maps').glob('*.png'))+list((folder/'maps').glob('leaf_profiles.json'))
        if (folder/'repair_evidence.json').exists():paths.append(folder/'repair_evidence.json')
        report['assets'].append({'asset_id':record['asset_id'],'species':species,'blend':str(blend),
               'variant_count':len(record['variants']),'objects':[v['objects'] for v in record['variants']],
               'files':[{'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(set(paths))],
               'generator_execution_script_sha256':record.get('generator_script_sha256'),
               'code_binding':'current reproduction source; original execution script hash was not captured' if not record.get('generator_script_sha256') else 'recorded at original execution',
               'isaac_qualification':'not_run','appearance_status':'unresolved_sparse_or_clustered_crown',
               'repair_budget_exhausted':species=='white_birch'})
    atomic_json(workspace/'state/procedural_assets.json',report)
    return report


def prepare_materials(workspace: Path):
    workspace=Path(workspace).resolve()
    catalogue=json.loads((workspace/'state/asset_catalogue.json').read_text())['assets']
    by_id={a['asset_id']:a for a in catalogue}
    materials=[]
    for asset_id in MATERIAL_IDS:
        if asset_id not in by_id:
            raise ServiceError('missing_material_master',f'Required acquired terrain material is missing: {asset_id}')
        asset=by_id[asset_id]
        dimensions=asset.get('dimensions_m')
        if not dimensions or len(dimensions)<2 or not .1<=float(dimensions[0])<=20:
            raise ServiceError('material_scale','Provider dimensions require explicit physical-scale calibration')
        material={'asset_id':asset_id,'name':asset_id,'repeat_m':float(dimensions[0]),
                  'texture_scale_source':'Poly Haven millimetre dimensions converted to metres',
                  'base_color_space':'sRGB','data_color_space':'linear','normal_convention':'OpenGL',
                  'displacement_amplitude_m':None,'displacement_status':'not_calibrated; scene geometry amplitude has separate evidence',
                  'qualification':'not_run'}
        for role,provider_role in [('base_color','Diffuse'),('roughness','Rough'),('normal','nor_gl'),('height','Displacement')]:
            matching=[f for f in asset['files'] if f.get('role')==provider_role]
            if len(matching)!=1 or digest(Path(matching[0]['path']))!=matching[0]['sha256']:
                raise ServiceError('material_provenance','Required material role is ambiguous, absent or corrupt')
            material[role]=matching[0]['path']
        materials.append(material)
    report={'schema_version':1,'materials':materials,'qualification':'not_run',
            'surface_assignments':{'0':MATERIAL_IDS[0],'1':MATERIAL_IDS[1]},
            'limitation':'Material candidates; actual per-face ecological assignment and transition qualification belong to the assembled scene'}
    atomic_json(workspace/'state/terrain_materials.json',report)
    return report


def prepare_native_catalogue(workspace: Path, blender: Path, environment=None):
    workspace=Path(workspace).resolve()
    by_id={a['asset_id']:a for a in json.loads((workspace/'state/asset_catalogue.json').read_text())['assets']}
    code_hash=digest(workspace/'blender_scripts/normalize_assets.py')
    report={'schema_version':1,'created_at_utc':utcnow(),'status':'running','assets':[],'failures':[]}
    output=workspace/'state/normalized_assets.json'
    for asset_id in UNDERSTORY_IDS:
        if asset_id not in by_id:
            report['failures'].append({'asset_id':asset_id,'code':'missing_master'})
            continue
        asset=by_id[asset_id]
        manifest=workspace/'assets/normalized'/asset_id/'normalization.json'
        candidate=json.loads(manifest.read_text()) if manifest.exists() else None
        blend_master=next(f for f in asset['files'] if f.get('role')=='blend')
        reusable=(candidate and candidate.get('normalizer_sha256')==code_hash
                  and candidate.get('source_sha256')==blend_master['sha256']
                  and Path(candidate['output_blend']).is_file()
                  and digest(Path(candidate['output_blend']))==candidate.get('output_sha256')
                  and all(digest(Path(f['path']))==f['sha256'] for f in asset['files']))
        try:
            if not reusable:
                candidate=normalize_asset(workspace,asset,blender,environment)
            report['assets'].append(candidate)
        except ServiceError as exc:
            report['failures'].append({'asset_id':asset_id,**exc.record()})
        # A partial catalogue has an explicit running state and never target pass.
        atomic_json(output,report)
    report['status']='external_tool_verified' if not report['failures'] else 'partial'
    report['isaac_qualification']='not_run'
    atomic_json(output,report)
    return report
