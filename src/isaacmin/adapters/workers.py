"""Native workers exchange bounded files across isolated runtime boundaries."""
from __future__ import annotations

import hashlib
import json
import time
import shutil
from pathlib import Path
import numpy as np
from isaacmin.process import run_worker
from isaacmin.security import worker_environment


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def native_environment(root: Path | None = None) -> dict[str, str]:
    root = Path(root or project_root()).resolve()
    env = worker_environment()
    env['LD_LIBRARY_PATH'] = ':'.join(str(root / p) for p in
        ('.tools/tbb/lib', '.tools/oiio/lib', '.tools/ocio/lib', '.tools/minizip/lib',
         '.tools/native/usr/lib/aarch64-linux-gnu', '.tools/native/usr/lib', '.tools/usd/lib'))
    env['OMP_NUM_THREADS'] = '8'
    env['QT_QPA_PLATFORM'] = 'offscreen'
    env['PYTHONPATH'] = str(root/'.tools/blender-py/lib/python3.12/site-packages')+':'+str(root/'.tools/usd/lib/python')+':'+str(root/'src')
    return env


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def _run(command: list[str], cwd: Path, timeout: float = 900, *,
         estimated_memory_bytes: int | None = None, estimated_disk_bytes: int = 0) -> dict:
    if not Path(command[0]).is_file():
        return {'status': 'blocked_dependency', 'reason': f'Missing worker: {command[0]}'}
    result = run_worker(command, cwd=cwd, log_path=cwd/'worker.log',
                        timeout=timeout, environment=native_environment(),
                        estimated_memory_bytes=estimated_memory_bytes,estimated_disk_bytes=estimated_disk_bytes)
    result['status'] = 'success' if result['exit_code'] == 0 else 'failed'
    result['stdout'] = (cwd/'worker.log').read_text()[-16000:]
    result['executable_sha256'] = _sha(Path(command[0]))
    return result


def refine_heightfield(heights: np.ndarray, output_dir: str | Path, *,
                      protection: np.ndarray | None = None,
                      erosion_m: float = 0.12, sample_spacing_m: float = 1.0,
                      max_lowering_m: float = 0.20,
                      global_runoff: np.ndarray | None = None,
                      global_runoff_normalized: bool = False,
                      erodibility_field: np.ndarray | None = None,
                      lowering_limit_field: np.ndarray | None = None) -> dict:
    """Run real HighMap drainage-driven erosion on a complete contextual region.

    z,x array values and bedrock are metres; no elevation renormalization occurs.
    This produces a technical refinement artifact, never a realism qualification.
    """
    z = np.asarray(heights, dtype='<f4')
    if z.ndim != 2 or min(z.shape) < 3 or not np.isfinite(z).all():
        raise ValueError('heights must be finite z,x raster with >=3 samples/axis')
    if not 0 < erosion_m <= 2 or not 0 < sample_spacing_m <= 2 or not 0 < max_lowering_m <= 2:
        raise ValueError('refinement parameters outside qualified worker bounds')
    protect = np.zeros(z.shape, bool) if protection is None else np.asarray(protection, bool)
    if protect.shape != z.shape:
        raise ValueError('protection shape mismatch')
    output = Path(output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    lowering = np.full(z.shape,max_lowering_m,dtype='<f4') if lowering_limit_field is None else np.asarray(lowering_limit_field,dtype='<f4')
    erodibility = np.ones(z.shape,dtype='<f4') if erodibility_field is None else np.asarray(erodibility_field,dtype='<f4').copy()
    if lowering.shape!=z.shape or not np.isfinite(lowering).all() or np.any(lowering<0) or np.any(lowering>max_lowering_m):
        raise ValueError('Local lowering limits must be finite and within the native maximum')
    if erodibility.shape!=z.shape or not np.isfinite(erodibility).all() or np.any(erodibility<0) or np.any(erodibility>1):
        raise ValueError('Local erodibility must be finite and in [0,1]')
    bedrock = z - np.where(protect, 0, lowering).astype('<f4')
    erodibility[protect]=0
    for name, array in [('input', z), ('bedrock', bedrock), ('erodibility', erodibility)]:
        array.tofile(output/f'{name}.f32')
    worker = project_root()/'.tools/build/native-gcc13/isaacmin_highmap'
    command = [str(worker), str(z.shape[1]), str(z.shape[0]),
                   str(output/'input.f32'), str(output/'bedrock.f32'),
                   str(output/'erodibility.f32'), str(output/'refined'),
                   str(erosion_m), str(0.2*sample_spacing_m)]
    if global_runoff is not None:
        runoff = np.asarray(global_runoff, dtype='<f4')
        if not global_runoff_normalized or runoff.shape != z.shape or not np.isfinite(runoff).all() or np.any(runoff<0) or np.any(runoff>1):
            raise ValueError('global_runoff must be normalized ONCE over complete region to [0,1], with global_runoff_normalized=True')
        runoff.tofile(output/'global_runoff.f32')
        command.append(str(output/'global_runoff.f32'))
    result = _run(command, output,estimated_memory_bytes=max(2**30,z.nbytes*32),
                  estimated_disk_bytes=z.nbytes*20)
    if result['status'] == 'success':
        arrays = {name: np.fromfile(output/f'refined_{name}.f32', dtype='<f4').reshape(z.shape)
                  for name in ('height', 'runoff', 'erosion')}
        if any(not np.isfinite(a).all() for a in arrays.values()):
            raise RuntimeError('HighMap emitted non-finite values')
        delta = arrays['height'] - z
        if np.any(arrays['height'] < bedrock-1e-5) or np.any(delta[protect] != 0):
            raise RuntimeError('HighMap violated bedrock/portal constraints')
        np.savez_compressed(output/'refinement.npz', **arrays, delta=delta,
                            protection=protect, sample_spacing_m=sample_spacing_m)
        result.update(backend='HighMap.imported_flow_stream_power.cpu' if global_runoff is not None else 'HighMap.hydraulic_stream.cpu', evidence_level='external_tool_verified',
                      arrays=str(output/'refinement.npz'), array_sha256=_sha(output/'refinement.npz'),
                      shape=list(z.shape), axis_order=['z', 'x'], units='metres',
                      protected_samples=int(protect.sum()),
                      changed_samples=int(np.count_nonzero(delta)),
                      delta_min_m=float(delta.min()), delta_max_m=float(delta.max()),
                      runoff_max=float(arrays['runoff'].max()),
                      float32_elevation_ulp_m=float(np.spacing(np.abs(z).max())),
                      scope='imported_global_drainage' if global_runoff is not None else 'complete_supplied_context_shared_drainage',
                      global_drainage_sha256=_sha(output/'global_runoff.f32') if global_runoff is not None else None,
                      erodibility_sha256=_sha(output/'erodibility.f32'),bedrock_sha256=_sha(output/'bedrock.f32'),
                      appearance_qualification='not_run')
    (output/'worker_result.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def reconstruct_occupancy(occupancy: np.ndarray, output_dir: str | Path, *,
                          voxel_size_m: float = 1.0) -> dict:
    """Convert local bool [y,z,x] occupancy through sparse OpenVDB into OBJ/VDB.

    Caller must preserve unknown/missing labels and apply the recorded local frame.
    No independent heightfield is layered over this all-surface ownership mesh.
    """
    if occupancy.dtype != np.bool_ or occupancy.ndim != 3 or not occupancy.any():
        raise ValueError('Expected nonempty Boolean occupancy [y,z,x]')
    output = Path(output_dir).resolve(); output.mkdir(parents=True, exist_ok=True)
    # OpenVDB uses target local X,-Z,Y; indices are cell centres, origin supplied separately.
    y, z, x = np.nonzero(occupancy)
    coordinates = np.column_stack((x, -z, y)).astype('<i4')
    coordinates.tofile(output/'occupied_xyz_i32.bin')
    worker = project_root()/'.tools/openvdb_worker'
    result = _run([str(worker), str(output/'occupied_xyz_i32.bin'),
                   str(output/'terrain'), str(voxel_size_m)], output, 1800,
                  estimated_memory_bytes=max(2**30,len(coordinates)*512),
                  estimated_disk_bytes=max(2**30,len(coordinates)*128))
    if result['status'] == 'success':
        metrics = json.loads(result['stdout'].splitlines()[-1])
        result.update(metrics, evidence_level='external_tool_verified',
                      occupancy_sha256=hashlib.sha256(np.ascontiguousarray(occupancy).tobytes()).hexdigest(),
                      occupancy_shape=list(occupancy.shape),
                      mesh=str(output/'terrain.obj'), mesh_sha256=_sha(output/'terrain.obj'),
                      volume=str(output/'terrain.vdb'), volume_sha256=_sha(output/'terrain.vdb'),
                      local_cell_center_transform='X=x*s,Y=-z*s,Z=y*s',
                      surface_ownership='all_terrain_surfaces_single_volume',
                      appearance_qualification='not_run',
                      portal_graph_comparison='not_run')
    (output/'worker_result.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


def capture_scene(scene: str | Path, output_dir: str | Path, *, poses: list[dict],
                  contact_probes: list[dict], hdri: str | Path | None = None,
                  scope: str = 'real_source_technical_integration',
                  resolution: tuple[int,int] = (1280,720),
                  lighting: str = 'directional', lighting_parameters: dict | None = None,
                  timeout: float | None = None, physics_scene_name: str = 'world_physics.usda',
                  renderer_recipe: str = 'legacy_rtx_8') -> dict:
    root=project_root();output=Path(output_dir).resolve();output.mkdir(parents=True,exist_ok=True)
    isaac=Path('/home/mehulchourasia/IsaacSim/_build/linux-aarch64/release/python.sh')
    config=root/'state/native_tools.json'
    if config.is_file(): isaac=Path(json.loads(config.read_text())['isaac_python'])
    request={'scene':str(Path(scene).resolve()),'output':str(output),'poses':poses,
             'contact_probes':contact_probes,'resolution':list(resolution),'scope':scope,
             'hdri':str(Path(hdri).resolve()) if hdri else None,'lighting':lighting,
             'lighting_parameters':lighting_parameters or {}}
    request['physics_scene_name']=physics_scene_name
    request['renderer_recipe']=renderer_recipe
    if lighting not in ('directional','diffuse'):raise ValueError('Unknown lighting condition')
    # Actual Spark PathTracing captures took about 38 s/frame on the small
    # compatibility scene. Keep the fixed 1024-sample recipe for large scenes;
    # allow slower completion instead of killing a valid render at 20 s/frame.
    frame_budget = 180 if renderer_recipe == 'pathtracing_1024' else 20
    capture_timeout = timeout if timeout is not None else max(1800, 1200 + len(poses) * frame_budget)
    if capture_timeout <= 0:
        raise ValueError('Capture timeout must be positive')
    request['timeout_seconds'] = capture_timeout
    path=output/'capture_request.json';path.write_text(json.dumps(request,indent=2)+'\n')
    # Kit may exit during SimulationApp.close even if Python raised; never read
    # an earlier successful artifact as evidence for the current invocation.
    for name in ('capture_result.json','capture_failure.json','reopen_result.json','reopen_failure.json','capture_frames.jsonl'):
        old=output/name
        if old.exists():old.rename(output/(old.stem+'.previous.'+str(time.time_ns())+'.json'))
    # Isaac owns its own libraries; never pass Blender/HighMap LD_LIBRARY_PATH to Kit.
    result=run_worker([str(isaac),str(root/'isaac_scripts/capture_scene.py'),'--request',str(path)],
                      cwd=root,log_path=output/'worker.log',
                      timeout=capture_timeout,
                      environment={'LD_LIBRARY_PATH':''})
    if result['exit_code']==0 and (output/'capture_result.json').is_file():
        result.update(json.loads((output/'capture_result.json').read_text()))
        # Encode only actual continuous-motion frames; static scout jumps must
        # not be presented as a navigable route or synthesized motion.
        moving=[f for f in result.get('frames',[]) if f.get('pose',{}).get('kind')=='motion']
        encoder=shutil.which('ffmpeg')
        if moving and encoder:
            sequence=output/'motion_frames.ffconcat'
            sequence.write_text('ffconcat version 1.0\n'+''.join(
                "file '"+f['rgb']+"'\nduration 0.0333333333333333\n" for f in moving))
            encoded=run_worker([encoder,'-y','-hide_banner','-loglevel','warning','-safe','0',
                '-i',str(sequence),'-an','-c:v','libx264','-crf','16','-pix_fmt','yuv420p',
                '-r','30','-frames:v',str(len(moving)),'-movflags','+faststart',str(output/'motion.mp4')],
                cwd=output,log_path=output/'ffmpeg.log',timeout=max(180,len(moving)*2),
                environment={'LD_LIBRARY_PATH':''})
            result['motion_video']={'status':'pass' if encoded['exit_code']==0 else 'fail',
                'path':str(output/'motion.mp4'),'source':'actual_native_Isaac_RGB_frames',
                'frames':len(moving),'playback_hz':30,'real_time_claim':False}
            if encoded['exit_code']==0:result['motion_video']['sha256']=_sha(output/'motion.mp4')
        physics_scene=result.get('physics_scene')
        if physics_scene:
            reopened=run_worker([str(isaac),str(root/'isaac_scripts/reopen_scene.py'),
                '--scene',physics_scene,'--output',str(output)],cwd=root,
                log_path=output/'reopen.log',timeout=600,environment={'LD_LIBRARY_PATH':''})
            if reopened['exit_code']==0 and (output/'reopen_result.json').is_file():
                result['native_reopen']=json.loads((output/'reopen_result.json').read_text())
                if result['native_reopen']['status']!='pass':result['status']='fail'
            else:result.update(status='failed',native_reopen={'status':'failed'})
    else:
        result['status']='failed'
        result['reason']='Isaac did not produce current render/contact evidence; inspect worker.log; sandbox GPU isolation may require authorized host execution'
    return result


def probe_workers(root: str | Path | None = None) -> dict:
    """Read actual worker evidence; do not turn installation into a passing probe."""
    root = Path(root or project_root()).resolve()
    paths = {'highmap': 'artifacts/bootstrap/highmap/worker_result.json',
             'openvdb': 'artifacts/bootstrap/openvdb/worker_result.json',
             'blender': 'artifacts/bootstrap/blender/result.json',
             'isaac': 'artifacts/bootstrap/isaac/result.json'}
    result = {}
    for name, relative in paths.items():
        path = root/relative
        result[name] = json.loads(path.read_text()) if path.is_file() else {'status': 'not_run'}
    for name,relative in [('openvdb_independent','artifacts/bootstrap/openvdb_independent/comparison.json'),
                          ('native_exchange','artifacts/bootstrap/cross_runtime.json')]:
        path=root/relative
        result[name]=json.loads(path.read_text()) if path.is_file() else {'status':'not_run'}
    if (root/'artifacts/bootstrap/openvdb/terrain.obj').is_file():
        if result['openvdb_independent'].get('mesh_sha256') != _sha(root/'artifacts/bootstrap/openvdb/terrain.obj'):
            result['openvdb_independent']={'status':'not_run','reason':'independent comparison is stale for current mesh'}
    for name,path in [('openvdb',root/'.tools/openvdb_worker'),
                      ('highmap',root/'.tools/build/native-gcc13/isaacmin_highmap')]:
        if not path.is_file():
            result[name]={'status':'not_run','reason':'required native worker binary missing'}
        elif result[name].get('executable_sha256')!=_sha(path):
            result[name]={'status':'not_run','reason':'worker binary changed; numerical probe must be repeated'}
    provenance=result['native_exchange'].get('chain_provenance',[])
    if not provenance:
        result['native_exchange']={'status':'not_run','reason':'native exchange predates content-bound evidence protocol'}
    else:
        for item in provenance:
            path=Path(item['path'])
            if not path.is_absolute():path=root/path
            if not path.is_file() or _sha(path)!=item['sha256']:
                result['native_exchange']={'status':'not_run','reason':'native exchange dependency changed or missing','path':str(path)}
                break
    result['cross_runtime_qualified'] = all(result[n].get('status') in ('success','pass')
        for n in [*paths,'openvdb_independent','native_exchange'])
    return result
