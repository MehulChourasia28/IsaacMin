"""Prepare grouped provider shaders before they reach the production exporter.

This is a native structural prerequisite. It never issues appearance qualification
or replaces original texture bytes, normalized geometry, anchors, or master assets.
"""
from pathlib import Path
import re
import shutil
import uuid

from isaacmin.io import atomic_json,read_json,sha256_file,hash_object,utc_now
from isaacmin.process import run_worker
from isaacmin.security import safe_path

RECIPE='native_provider_group_flatten_v1'
PRODUCERS=('src/isaacmin/assets/target_materials.py','blender_scripts/prepare_target_assets.py')


def _source(root,asset):
    if not re.fullmatch(r'[a-z0-9_]+',asset['asset_id']):
        raise ValueError('Expected a bounded provider asset identifier')
    path=Path(asset['output_blend']).resolve()
    if not path.is_relative_to(root/'assets') or path.suffix!='.blend' or sha256_file(path)!=asset['output_sha256']:
        raise ValueError('Original normalized asset is missing, outside the asset store, or changed')
    return path


def verify_prepared_asset(root,asset):
    """Check every native proof and image dependency; no pass from a flag alone."""
    root=Path(root).resolve();entry=asset.get('target_material_preparation')
    if not entry:return False
    expected={name:sha256_file(root/name) for name in PRODUCERS}
    if entry.get('recipe')!=RECIPE or entry.get('producers')!=expected:
        return False
    original=entry['input_asset'];_source(root,original);_source(root,asset)
    proof=safe_path(root,entry['report']['path'],must_exist=True)
    if sha256_file(proof)!=entry['report']['sha256']:
        raise ValueError('Prepared material proof changed')
    result=read_json(proof)
    if (result['source_sha256']!=original['output_sha256']
            or result['output_sha256']!=asset['output_sha256']
            or result.get('geometry_and_uvs_equal') is not True
            or result.get('executed_producer_sha256')!=expected[PRODUCERS[1]]):
        raise ValueError('Native shader preparation lacks unchanged source/geometry evidence')
    base=proof.parent.parent
    if (base/result['output_blend_relative']).resolve()!=Path(asset['output_blend']).resolve():
        raise ValueError('Prepared asset output path differs from actual native report')
    for item in result['native_dependency_closure']:
        path=safe_path(base,item['path'],must_exist=True)
        if sha256_file(path)!=item['sha256']:raise ValueError('Prepared USD material dependency changed')
    process=safe_path(root,entry['process']['path'],must_exist=True)
    if sha256_file(process)!=entry['process']['sha256']:
        raise ValueError('Native material worker receipt changed')
    execution=read_json(process)
    if execution.get('exit_code')!=0 or execution.get('timed_out') or execution.get('resource_limited'):
        raise ValueError('Native material worker did not complete')
    return True


def prepare_target_assets(workspace):
    """Automatically flatten provider groups using Blender's native operation.

    Unchanged already-prepared assets are verified and reused. Changed code or
    source creates a new derivative, with previous manifests retained for replay.
    """
    root=Path(workspace).resolve();manifest_path=root/'state/normalized_assets.json'
    if not manifest_path.is_file():return {'status':'not_available','qualification':'not_run'}
    manifest=read_json(manifest_path);before=sha256_file(manifest_path);pending=[]
    for asset in manifest['assets']:
        _source(root,asset)
        if verify_prepared_asset(root,asset):continue
        original=asset.get('target_material_preparation',{}).get('input_asset',asset)
        if any('ShaderNodeGroup' in material.get('node_types',[]) for material in original.get('materials',[])):
            _source(root,original);pending.append(original)
    if not pending:return {'status':'verified_or_no_group_conversion_needed','qualification':'not_inferred'}
    producers={name:sha256_file(root/name) for name in PRODUCERS}
    identity={'recipe':RECIPE,'producers':producers,
        'native_blender_sha256':sha256_file(root/'.tools/blender/blender'),
        'input_asset_records':{a['asset_id']:hash_object(a) for a in pending}}
    key=hash_object(identity);destination=root/'assets/target_materials'/key
    if destination.exists():
        # Recover a crash between immutable promotion and catalogue selection.
        # Every output and the original process receipt is rechecked below.
        staging=destination
        if read_json(staging/'identity.json')!=identity:
            raise ValueError('Existing material preparation identity differs')
        process=read_json(staging/'worker.process.json')
    else:
        staging=destination.with_name('.'+key+'.staging.'+uuid.uuid4().hex);staging.mkdir(parents=True)
        atomic_json(staging/'identity.json',identity)
        atomic_json(staging/'input_assets.json',{'assets':pending})
        request=staging/'request.json'
        atomic_json(request,{'manifest':str(staging/'input_assets.json'),'output':str(staging/'assets'),
                             'asset_ids':[a['asset_id'] for a in pending]})
        from isaacmin.adapters.workers import native_environment
        process=run_worker([str(root/'.tools/blender/blender'),'--background','--factory-startup',
            '--disable-autoexec','--python-use-system-env','--python-exit-code','1','--python',
            str(root/'blender_scripts/prepare_target_assets.py'),'--',str(request)],cwd=root,
            log_path=staging/'worker.log',timeout=1800,environment=native_environment(root),
            estimated_memory_bytes=2*2**30,estimated_disk_bytes=2*2**30)
    if process['exit_code'] or not (staging/'assets/results.json').is_file():
        raise RuntimeError('Native plant-material preparation failed; preserved '+str(staging/'worker.log'))
    results=read_json(staging/'assets/results.json')
    if results.get('status')!='native_group_flattening_complete' or results.get('recipe')!=RECIPE:
        raise ValueError('Incomplete native provider-material preparation')
    if {a['asset_id'] for a in results['assets']}!={a['asset_id'] for a in pending}:
        raise ValueError('Native preparation omitted a requested provider asset')
    for item in results['assets']:
        original=next(a for a in pending if a['asset_id']==item['asset_id'])
        if item['source_sha256']!=original['output_sha256'] or item.get('geometry_and_uvs_equal') is not True:
            raise ValueError('Native group conversion changed original asset geometry')
        if sha256_file(staging/'assets'/item['output_blend_relative'])!=item['output_sha256']:
            raise ValueError('Prepared asset bytes differ from native result')
    if sha256_file(manifest_path)!=before:raise ValueError('Normalized catalogue changed during preparation')
    # Every producer/source is checked again before promoting an actual result.
    for a in pending:_source(root,a)
    if producers!={name:sha256_file(root/name) for name in PRODUCERS}:
        raise ValueError('Native material producer changed during execution')
    if staging!=destination:staging.rename(destination)
    prepared={}
    for item in results['assets']:
        original=next(a for a in pending if a['asset_id']==item['asset_id'])
        proof=destination/'assets'/item['asset_id']/'flattening.json'
        receipt=destination/'worker.process.json'
        asset=dict(original,output_blend=str(destination/'assets'/item['output_blend_relative']),
            output_sha256=item['output_sha256'],materials=item['materials'],isaac_qualification='not_run')
        asset['target_material_preparation']={'recipe':RECIPE,'producers':producers,
            'input_asset':original,'report':{'path':str(proof.relative_to(root)),'sha256':sha256_file(proof)},
            'process':{'path':str(receipt.relative_to(root)),'sha256':sha256_file(receipt)},
            'qualification':'native_export_structure_only; actual Isaac appearance still required'}
        if not verify_prepared_asset(root,asset):raise ValueError('Promoted material preparation did not verify')
        prepared[asset['asset_id']]=asset
    history=root/'state/history/normalized_assets'/(before+'.json');history.parent.mkdir(parents=True,exist_ok=True)
    if not history.exists():shutil.copy2(manifest_path,history)
    updated=dict(manifest,assets=[prepared.get(a['asset_id'],a) for a in manifest['assets']])
    updated['target_material_preparation']={'at_utc':utc_now(),'recipe':RECIPE,
        'original_manifest_sha256':before,'evidence':str(destination.relative_to(root)),
        'qualification':'not_inferred'}
    atomic_json(manifest_path,updated)
    return {'status':'native_group_materials_prepared','assets':sorted(prepared),
            'evidence':str(destination.relative_to(root)),'qualification':'not_inferred'}
