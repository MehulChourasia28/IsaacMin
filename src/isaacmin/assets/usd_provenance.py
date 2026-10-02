"""Bind the actual USD payload, including content layers and texture bytes."""
from pathlib import Path
import json
from .network import ServiceError,digest,safe_relative


def verify_scene_closure(scene,expected_closure_sha256):
    scene=Path(scene).resolve();manifest=scene.parent/'native_dependency_closure.json'
    if not expected_closure_sha256 or not manifest.is_file() or digest(manifest)!=expected_closure_sha256:
        raise ServiceError('scene_closure_changed','Captured native dependency closure is missing or changed')
    closure=json.loads(manifest.read_text())
    if closure.get('status')!='pass' or closure.get('root_sha256')!=digest(scene):
        raise ServiceError('scene_closure_changed','Native dependency closure does not bind the current root layer')
    files=[]
    for bound in closure.get('files',[]):
        path=(scene.parent/safe_relative(bound['path'])).resolve()
        if not path.is_relative_to(scene.parent) or not path.is_file() or digest(path)!=bound['sha256']:
            raise ServiceError('scene_payload_changed','An exported content layer or texture differs from captured dependency bytes')
        files.append({'path':str(path),'sha256':bound['sha256']})
    if not files or not any(Path(f['path'])==scene for f in files):
        raise ServiceError('scene_closure_incomplete','Closure must include the actual root USD and all native dependencies')
    return {'manifest':{'path':str(manifest),'sha256':expected_closure_sha256},'files':files}


def captured_closure_hash(capture):
    return (capture.get('native_dependency_closure_sha256') or capture.get('scene_dependency_closure_sha256')
            or capture.get('physx_contact_rays',{}).get('native_dependency_closure_sha256'))


def validated_native_renderer(capture):
    """Accept retained RTX records or the measured explicit PathTracing recipe.

    Merely changing a renderer label cannot qualify a new capture protocol.
    This checks declared native readback; sensor and scene evidence are separate.
    """
    renderer=capture.get('renderer')
    if renderer=='RayTracedLighting':return renderer
    expected={
        '/rtx/rendermode':'PathTracing','/rtx/post/aa/op':0,
        '/rtx/pathtracing/clampSpp':64,'/rtx/pathtracing/spp':64,
        '/rtx/pathtracing/totalSpp':1024,'/rtx/pathtracing/maxBounces':12,
        '/rtx/pathtracing/maxSpecularAndTransmissionBounces':12,
        '/rtx/pathtracing/maxVolumeBounces':4,
        '/rtx/pathtracing/adaptiveSampling/enabled':False,
        '/rtx/pathtracing/optixDenoiser/enabled':False,
        '/rtx/post/motionblur/enabled':False}
    recorded=capture.get('native_renderer_settings',{})
    frames=capture.get('frames',[])
    if (renderer!='PathTracing' or capture.get('renderer_recipe')!='pathtracing_1024'
            or capture.get('path_tracing_sample_budget')!=1024 or not frames
            or any(recorded.get(k)!=v or type(recorded.get(k)) is not type(v) for k,v in expected.items())
            or any(f.get('path_tracing_sample_budget')!=1024 for f in frames)):
        raise ServiceError('unsupported_native_renderer_protocol','Renderer needs a supported actual native setting readback and complete frame sample budgets')
    return renderer
