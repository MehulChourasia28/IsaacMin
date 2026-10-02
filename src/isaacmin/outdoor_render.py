"""Render completed immutable outdoor scenes without rebuilding their terrain."""
from pathlib import Path
import shutil
from isaacmin.io import read_json,atomic_json,sha256_file,utc_now
from isaacmin.process import run_worker


def validate_capture_result(path,scene_sha256,poses):
    """Validate requested native frames before publishing or reusing completion."""
    from isaacmin.security import safe_path
    path=Path(path);report=read_json(path)
    if report.get('status')!='actual_visual_preview_complete' or report.get('scene_sha256')!=scene_sha256:
        raise ValueError('Requested capture does not match the selected native scene')
    frames=report.get('frames',[])
    if [frame.get('pose') for frame in frames]!=poses:raise ValueError('Requested capture camera poses differ')
    for frame in frames:
        for sensor in ('rgb','depth','instance_segmentation'):
            if sha256_file(safe_path(path.parent,frame[sensor],must_exist=True))!=frame[sensor+'_sha256']:
                raise ValueError('Requested capture sensor bytes changed: '+sensor)
    return report


def render_outdoor(workspace,build,output,*,views='full',poses=None):
    from isaacmin.outdoor_pipeline import outdoor_overview_poses
    workspace,build,output=map(lambda p:Path(p).resolve(),(workspace,build,output))
    record=read_json(build/'outdoor_build.json')
    if record.get('status')!='geometry_world':raise ValueError('Complete native construction before rendering')
    if output.exists():raise ValueError('Preserve earlier capture; choose a fresh output')
    if not output.is_relative_to(workspace) or output.is_relative_to(Path(record['scene']).parent):
        raise ValueError('Capture needs a separate workspace evidence directory')
    if views not in ('full','overview','focus','requested'):raise ValueError('Unknown capture view set')
    if (views=='requested')!=(poses is not None):raise ValueError('Requested captures require explicit validated poses')
    request=read_json(build/'assembled/preview_request.json')
    if request['scene']!=record['scene']:raise ValueError('Saved camera request belongs to another scene')
    if views=='requested':
        if not 1<=len(poses)<=6:raise ValueError('Request between one and six views')
        from isaacmin.assembly.outdoor_lighting import daylight_camera_response
        request['poses']=[daylight_camera_response(pose,request) for pose in poses]
    elif views in ('full','overview'):
        overview=outdoor_overview_poses(build/'terrain')
        request['poses']=overview+(request['poses'] if views=='full' else [])
    else:
        poses=request['poses'];shore=next((p for p in poses if p.get('name')=='source_water_shore'),poses[-1])
        request['poses']=[poses[0],shore] if shore!=poses[0] else [poses[0]]
    output.mkdir(parents=True);producer=output/'producer_sources';producer.mkdir()
    hashes={}
    for name in ('capture_outdoor_visual.py','material_log.py','visible_instance_labels.py'):
        source=workspace/'isaac_scripts'/name
        target=producer/name;shutil.copyfile(source,target)
        hashes[name]=sha256_file(target)
    request.update(output=str(output/'capture'),scope='actual completed outdoor world; native visual evidence, not physics or navigation qualification')
    atomic_json(output/'request.json',request)
    evidence=dict(at_utc=utc_now(),status='running',build=str(build),build_manifest_sha256=sha256_file(build/'outdoor_build.json'),
        scene=record['scene'],views=views,poses=len(request['poses']),producer_sha256=hashes,
        geometry_rebuild=False,render_quality_changed=False,qualification='not_run')
    atomic_json(output/'render.json',evidence)
    try:
        result=run_worker([str(workspace/'.tools/isaacsim-6.1.0/python.sh'),str(producer/'capture_outdoor_visual.py'),
            '--request',str(output/'request.json')],cwd=workspace,log_path=output/'worker.log',
            timeout=max(3600,1200+len(request['poses'])*1200),estimated_memory_bytes=70*2**30,environment={'LD_LIBRARY_PATH':''})
    except BaseException as error:
        from isaacmin.security import redact
        evidence.update(status='failed',error_type=type(error).__name__,reason=redact(str(error)),
            native_process_receipt=str(output/'worker.process.json'))
        atomic_json(output/'render.json',evidence)
        raise
    capture=output/'capture/preview_result.json'
    validation_error=None
    if views=='requested' and result['exit_code']==0 and capture.is_file():
        try:validate_capture_result(capture,sha256_file(Path(record['scene'])),request['poses'])
        except (ValueError,KeyError,FileNotFoundError) as error:validation_error=str(error)
    evidence.update(status='actual_native_capture_complete' if result['exit_code']==0 and capture.is_file() else 'failed',
        exit_code=result['exit_code'],elapsed_seconds=result['elapsed_seconds'],capture_result=str(capture),
        actual_navigation_stack='not_run_not_supplied')
    if validation_error:evidence.update(status='failed',reason=validation_error)
    atomic_json(output/'render.json',evidence)
    if evidence['status']=='failed':raise RuntimeError('Native capture failed; inspect '+str(output/'worker.log'))
    return evidence
