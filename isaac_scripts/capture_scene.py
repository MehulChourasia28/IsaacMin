"""Fresh Isaac RTX RGB/axial depth/contact harness; no user controller is implied."""
import argparse
import hashlib
import json
import pathlib
import shutil
import time
import traceback

parser=argparse.ArgumentParser()
parser.add_argument('--request',required=True)
args,_=parser.parse_known_args()
request=json.loads(pathlib.Path(args.request).read_text())
output=pathlib.Path(request['output']).resolve();output.mkdir(parents=True,exist_ok=True)
execution_started=time.perf_counter()
def phase(name,**fields):
    with (output/'phase_timings.jsonl').open('a') as stream:
        stream.write(json.dumps({'phase':name,'wall_elapsed_s':time.perf_counter()-execution_started,**fields})+'\n')
        stream.flush()
def file_sha(path):
    with pathlib.Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()
phase('application_start')
renderer_recipe=request.get('renderer_recipe','legacy_rtx_8')
if renderer_recipe not in ('legacy_rtx_8','full_resolution_rtx_64','native_resolution_rtx_128',
                           'pathtracing_1024'):
    raise ValueError('Unsupported frozen renderer recipe')
path_tracing=renderer_recipe=='pathtracing_1024'
renderer='PathTracing' if path_tracing else 'RayTracedLighting'
from isaacsim import SimulationApp
app_config={'headless':True,'renderer':renderer,'width':1280,'height':720,
            'disable_viewport_updates':True,
            'extra_args':[f'--/log/file={output}/kit.log','--/app/window/enabled=false',
                          '--/rtx/post/dlss/execMode=2']}
if path_tracing:
    # Physics settling does not produce evidence frames. Configure full sample
    # accumulation below, immediately before the measured capture sequence.
    app_config.update(samples_per_pixel_per_frame=1,denoiser=False,anti_aliasing=0,
                      max_bounces=12,max_specular_transmission_bounces=12,max_volume_bounces=4)
app=SimulationApp(app_config)
phase('application_ready')
try:
    from material_log import MaterialLogGuard
    material_guard=MaterialLogGuard(output/'kit.log',output/'native_material_diagnostics.json')
    import numpy as np
    import omni.usd
    import omni.physx
    import carb.settings
    import omni.replicator.core as rep
    from PIL import Image
    from pxr import Usd,UsdGeom,UsdLux,UsdShade,UsdPhysics,PhysxSchema,Gf,Sdf
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid
    renderer_settings=carb.settings.get_settings()
    renderer_keys=('/rtx/post/aa/op','/rtx/directLighting/sampledLighting/samplesPerPixel',
                   '/rtx/reflections/sampledLighting/samplesPerPixel','/rtx/indirectDiffuse/fetchSampleCount')
    if renderer_recipe=='full_resolution_rtx_64':
        # Values are documented by the pinned RTX settings extension. DLAA
        # renders native resolution; higher real samples trade speed for quality.
        for key,value in zip(renderer_keys,(4,8,8,4)):
            if renderer_settings.get(key) is None:
                raise RuntimeError('Pinned renderer lacks required quality setting '+key)
            renderer_settings.set_int(key,value)
            if renderer_settings.get(key)!=value:
                raise RuntimeError('Native renderer did not accept '+key)
    elif renderer_recipe=='native_resolution_rtx_128':
        # The current RTPT runtime does not expose the legacy reflection-SPP
        # control. Preserve its lighting model and improve actual resolution and
        # subframe sampling, using only a control observed on this exact host.
        key='/rtx/post/aa/op'
        if renderer_settings.get(key) is None:
            raise RuntimeError('Pinned renderer lacks native-resolution AA selection')
        renderer_settings.set_int(key,4)
        if renderer_settings.get(key)!=4:
            raise RuntimeError('Native renderer did not accept DLAA')
    native_renderer_settings={key:renderer_settings.get(key) for key in renderer_keys}
    if renderer_settings.get('/rtx-transient/dlssg/enabled'):
        raise RuntimeError('Generated display frames cannot be used for capture evidence')
    rt_subframes=1 if path_tracing else 128 if renderer_recipe=='native_resolution_rtx_128' else 64 if renderer_recipe=='full_resolution_rtx_64' else 8
    scene=pathlib.Path(request['scene']).resolve()
    input_closure_path=scene.parent/('native_'+scene.stem+'_dependency_closure.json' if scene.stem.startswith('world_physics_')
                                    else 'native_physics_dependency_closure.json' if scene.name=='world_physics.usda'
                                    else 'native_dependency_closure.json')
    if not input_closure_path.is_file():raise RuntimeError('Native scene dependency closure is required')
    input_closure=json.loads(input_closure_path.read_text())
    input_closure_sha256=hashlib.sha256(input_closure_path.read_bytes()).hexdigest()
    if input_closure.get('status')!='pass' or input_closure['root_sha256']!=file_sha(scene):
        raise RuntimeError('Input native scene closure is stale or unqualified')
    for entry in input_closure['files']:
        dependency=(scene.parent/entry['path']).resolve()
        if not dependency.is_relative_to(scene.parent) or file_sha(dependency)!=entry['sha256']:
            raise RuntimeError('Input scene dependency changed before capture')

    phase('input_closure_verified')
    if not omni.usd.get_context().open_stage(str(scene)):
        raise RuntimeError('Isaac failed to open USD scene')
    stage=omni.usd.get_context().get_stage()
    preserve_authored=bool(request.get('preserve_authored_scene',False))
    if preserve_authored:
        # These are the preceding measurement harness, never scene geometry.
        # Recreate the original camera/product paths instead of adding a second
        # render graph with inherited history. The saved package is not edited.
        for scope in ('/Replicator','/Render'):
            stage.RemovePrim(scope)
    for _ in range(30): app.update()
    material_guard.check()
    phase('scene_opened')
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage,1)
    world=World(stage_units_in_meters=1.0,physics_dt=1/120,rendering_dt=1/30)
    # This GPU PhysX configuration is separately tested on the pinned ARM64 runtime.
    world.get_physics_context().enable_gpu_dynamics(True)
    world.get_physics_context().set_broadphase_type('GPU')
    from ground_collision import configure_ground_collision
    ground_collision=configure_ground_collision(stage,preserve_authored=preserve_authored)
    terrain=ground_collision['render_ground_meshes']
    phase('exact_colliders_ready',parts=len(ground_collision['collision_meshes']))
    water_prims=[p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh) and 'SourceSurfaceWater' in str(p.GetPath())]
    if water_prims and not preserve_authored:
        water_material=UsdShade.Material.Define(stage,'/IsaacMinMaterials/Water')
        shader=UsdShade.Shader.Define(stage,'/IsaacMinMaterials/Water/Shader')
        shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
        shader.SetSourceAsset(Sdf.AssetPath('OmniSurface.mdl'),'mdl')
        shader.SetSourceAssetSubIdentifier('OmniSurface','mdl')
        shader.CreateOutput('out',Sdf.ValueTypeNames.Token)
        for name,value in [('diffuse_reflection_weight',0.0),('specular_reflection_weight',1.0),
                           ('specular_reflection_roughness',0.06),('specular_reflection_ior',1.333),
                           ('specular_transmission_weight',1.0)]:
            shader.CreateInput(name,Sdf.ValueTypeNames.Float).Set(value)
        shader.CreateInput('enable_specular_transmission',Sdf.ValueTypeNames.Bool).Set(True)
        shader.CreateInput('thin_walled',Sdf.ValueTypeNames.Bool).Set(False)
        water_material.CreateSurfaceOutput('mdl').ConnectToSource(shader.ConnectableAPI(),'out')
        for prim in water_prims:UsdShade.MaterialBindingAPI.Apply(prim).Bind(water_material)
    lighting_parameters=request.get('lighting_parameters',{})
    camera_recipe=lighting_parameters.get('camera_default',{})
    camera_settings={}
    if camera_recipe:
        settings=carb.settings.get_settings()
        authored={'/rtx/post/histogram/enabled':bool(camera_recipe.get('auto_exposure',False)),
                  '/rtx/post/motionblur/enabled':bool(camera_recipe.get('motion_blur',False)),
                  '/rtx/post/tonemap/filmIso':float(camera_recipe['iso']),
                  '/rtx/post/tonemap/fNumber':float(camera_recipe['f_number']),
                  '/rtx/post/tonemap/exposureTime':float(camera_recipe['shutter_seconds']),
                  '/rtx/post/tonemap/whitepoint':list(UsdLux.BlackbodyTemperatureAsRgb(
                      float(camera_recipe.get('white_balance_kelvin',5500))))}
        for key,value in authored.items():
            if settings.get(key) is None:raise RuntimeError('Pinned renderer lacks requested camera setting '+key)
            settings.set(key,value)
            camera_settings[key]=settings.get(key)
    if preserve_authored:
        if not stage.GetPrimAtPath('/IsaacMinLighting/Dome'):
            raise RuntimeError('Packaged scene has no authored illumination')
    else:
        light=UsdLux.DomeLight.Define(stage,'/IsaacMinLighting/Dome')
        light.CreateIntensityAttr(float(lighting_parameters.get('dome_intensity',800)))
        light.CreateExposureAttr(float(lighting_parameters.get('dome_exposure',0)))
        UsdGeom.Xformable(light).AddRotateZOp().Set(float(lighting_parameters.get('dome_rotation_z_degrees',0)))
    if request.get('hdri') and not preserve_authored:
        hdri_source=pathlib.Path(request['hdri']).resolve()
        hdri_name=hashlib.sha256(hdri_source.read_bytes()).hexdigest()[:16]+hdri_source.suffix
        hdri_target=scene.parent/'textures'/hdri_name
        hdri_target.parent.mkdir(parents=True,exist_ok=True)
        if not hdri_target.is_file():shutil.copyfile(hdri_source,hdri_target)
        light.CreateTextureFileAttr(Sdf.AssetPath('textures/'+hdri_name))
    if not preserve_authored:
        sun=UsdLux.DistantLight.Define(stage,'/IsaacMinLighting/Sun')
        sun.CreateIntensityAttr(float(lighting_parameters.get('sun_intensity',
                                     0 if request.get('hdri') else 1500 if request.get('lighting','directional')=='directional' else 0)))
        sun.CreateAngleAttr(0.53)
        UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-45,20,25))
    camera=rep.create.camera(position=(0,0,10),look_at=(0,1,10),
                             focal_length=18,horizontal_aperture=36,
                             clipping_range=(0.02,5000))
    resolution=tuple(request.get('resolution',[1280,720]))
    product=rep.create.render_product(camera,resolution)
    rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach(product)
    depth=rep.AnnotatorRegistry.get_annotator('distance_to_image_plane');depth.attach(product)
    segmentation=rep.AnnotatorRegistry.get_annotator('instance_id_segmentation',
                                                    init_params={'colorize':False})
    segmentation.attach(product)
    probe_specs=request.get('contact_probes',[])
    probes=[]
    for index,spec in enumerate(probe_specs):
        position=np.asarray(spec['position'],dtype=float)
        probes.append(world.scene.add(DynamicCuboid(prim_path=f'/IsaacMinContacts/P{index}',
                      name=f'probe{index}',position=position,
                      scale=np.array([0.2,0.2,0.2]),mass=1.0)))
    # Let native USD change notifications and mesh cooking complete before
    # dropping a body. CPU stepping alone can outrun asynchronous mesh cooking.
    for _ in range(30):app.update()
    omni.physx.get_physx_interface().force_load_physics_from_usd()
    world.reset()
    for _ in range(240): world.step(render=True)
    material_guard.check()
    phase('physics_settled')
    query=omni.physx.get_physx_scene_query_interface()
    contacts=[]
    for index,(probe,spec) in enumerate(zip(probes,probe_specs)):
        p,orientation=probe.get_world_pose()
        expected=float(spec['ground_z'])+0.1
        error=abs(float(p[2])-expected)
        contact_ray=query.raycast_closest(tuple(spec['position']),(0,0,-1),5000)
        contacts.append({'id':index,'settled_position':p.tolist(),
                         'settled_orientation_wxyz':orientation.tolist(),
                         'linear_velocity_mps':probe.get_linear_velocity().tolist(),
                         'angular_velocity_radps':probe.get_angular_velocity().tolist(),
                         'probe_shape':'cube','edge_length_m':0.2,
                         'expected_center_z':expected,'support_error_m':error,
                         'native_raycast':{'hit':bool(contact_ray.get('hit',False)),
                              'distance':float(contact_ray.get('distance',0)),
                              'position':list(contact_ray.get('position',(0,0,0))),
                              'collision':str(contact_ray.get('collision',''))},
                         'status':'pass' if error<=0.02 else 'fail'})
        probe.set_visibility(False)
    # Remove diagnostic bodies and colliders before captures and the exported runtime world.
    stage.RemovePrim('/IsaacMinContacts')
    for _ in range(2):world.step(render=False)
    from contact_rays import sample_contact_rays
    contact_rays=sample_contact_rays(stage,query,terrain,output,
                                    collision_paths=ground_collision['collision_meshes'])
    phase('contact_rays_measured',rays=contact_rays['ray_count'],hits=contact_rays['hits'])
    if path_tracing:
        # These controls exist in the pinned SimulationApp and RTX settings
        # extension. Replicator accumulates totalSpp/spp PT subframes at a held
        # simulation time; RT subframes are a separate control. No renderer RNG
        # determinism is implied by the fixed sample budget.
        pt_controls={'/rtx/rendermode':'PathTracing','/rtx/post/aa/op':0,
            '/rtx/pathtracing/clampSpp':64,'/rtx/pathtracing/spp':64,
            '/rtx/pathtracing/totalSpp':1024,'/rtx/pathtracing/maxBounces':12,
            '/rtx/pathtracing/maxSpecularAndTransmissionBounces':12,
            '/rtx/pathtracing/maxVolumeBounces':4,
            '/rtx/pathtracing/adaptiveSampling/enabled':False,
            '/rtx/pathtracing/optixDenoiser/enabled':False,
            '/rtx/post/motionblur/enabled':False}
        for key,value in pt_controls.items():
            if renderer_settings.get(key) is None:
                raise RuntimeError('Pinned renderer lacks required path tracing setting '+key)
            renderer_settings.set(key,value)
            if renderer_settings.get(key)!=value:
                raise RuntimeError('Native renderer did not accept '+key)
        native_renderer_settings={key:renderer_settings.get(key) for key in pt_controls}
        if renderer_settings.get('/omni/replicator/pathTracedMotionBlurSubSamples'):
            raise RuntimeError('Path tracing motion blur override would change sample accumulation')
    captures=[]
    started=time.monotonic()
    for index,pose in enumerate(request['poses']):
        frame_started=time.perf_counter()
        with camera:
            rep.modify.pose(position=pose['position'],look_at=pose['look_at'])
        before_physics_time=float(world.current_time)
        for _ in range(4):world.step(render=False)
        render_started=time.perf_counter()
        rep.orchestrator.step(rt_subframes=rt_subframes,pause_timeline=False,delta_time=0.0)
        render_finished=time.perf_counter()
        material_guard.check()
        observed_physics_delta=float(world.current_time)-before_physics_time
        rgba=rgb.get_data(); axial=depth.get_data()
        segments=segmentation.get_data()
        readback_finished=time.perf_counter()
        valid=np.isfinite(axial)&(axial>0)
        if rgba.shape[:2] != (resolution[1],resolution[0]) or rgba.std()<2 or valid.sum()<1000:
            raise RuntimeError(f'Invalid RTX capture at pose {index}')
        image_path=output/f'rgb_{index:05d}.png'
        depth_path=output/f'depth_{index:05d}.npy'
        Image.fromarray(rgba).save(image_path);np.save(depth_path,axial)
        segment_path=output/f'instance_{index:05d}.npy'
        np.save(segment_path,segments['data'])
        camera_prim=next(p for camera_root in camera.get_output_prims()['prims']
                         for p in Usd.PrimRange(camera_root) if p.IsA(UsdGeom.Camera))
        actual_camera=UsdGeom.Camera(camera_prim)
        matrix=UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        aperture=float(actual_camera.GetHorizontalApertureAttr().Get())
        focal=float(actual_camera.GetFocalLengthAttr().Get())
        captures.append({'frame':index,'rgb':image_path.name,'depth':depth_path.name,
                         'pose':pose,'simulation_time_s':float(world.current_time),
                         'requested_physics_steps':4,'physics_step_s':1/120,
                         'actual_rt_subframes':rt_subframes,
                         'path_tracing_sample_budget':1024 if path_tracing else None,
                         'path_tracing_subframes_per_capture':16 if path_tracing else None,
                         'observed_physics_delta_s':observed_physics_delta,
                         'wall_elapsed_s':time.monotonic()-started,
                         'physics_backend':'PhysX GPU dynamics with GPU broadphase',
            'depth_semantics':'axial_metres','valid_depth_pixels':int(valid.sum()),
                         'camera_prim_path':str(camera_prim.GetPath()),
                         'camera_world_matrix_row_vectors':[[float(matrix[r][c]) for c in range(4)] for r in range(4)],
                         'camera_axes':'USD camera looks along local -Z, up +Y; matrix row vectors',
                         'focal_length_mm':focal,'horizontal_aperture_mm':aperture,
                         'fx_pixels':focal/aperture*resolution[0],
                         'fy_pixels':focal/aperture*resolution[0],
                         'cx_pixels':resolution[0]/2,'cy_pixels':resolution[1]/2,
                         'instance_segmentation':segment_path.name,
                         'instance_segmentation_sha256':hashlib.sha256(segment_path.read_bytes()).hexdigest(),
                         'instance_id_to_prim_path':segments.get('info',{}).get('idToLabels',{}),
                         'rgb_sha256':hashlib.sha256(image_path.read_bytes()).hexdigest(),
                         'depth_sha256':hashlib.sha256(depth_path.read_bytes()).hexdigest()})
        with (output/'capture_frames.jsonl').open('a') as checkpoint:
            checkpoint.write(json.dumps(captures[-1])+'\n')
        checkpoint_data={'status':'running','completed_frames':index+1,
                         'planned_frames':len(request['poses']),
                         'scene_sha256':hashlib.sha256(scene.read_bytes()).hexdigest(),
                         'simulation_time_s':float(world.current_time),
                         'latest_rgb_sha256':captures[-1]['rgb_sha256']}
        temporary=output/'capture_checkpoint.tmp'
        temporary.write_text(json.dumps(checkpoint_data,indent=2)+'\n')
        temporary.replace(output/'capture_checkpoint.json')
        phase('frame_complete',frame=index,physics_and_pose_seconds=render_started-frame_started,
              render_seconds=render_finished-render_started,readback_seconds=readback_finished-render_finished,
              save_seconds=time.perf_counter()-readback_finished)
    # Persist the actual native stage including exact final mesh collision, lighting and camera.
    physics_name=request.get('physics_scene_name','world_physics.usda')
    if pathlib.Path(physics_name).name!=physics_name or not physics_name.endswith('.usda'):
        raise ValueError('Expected local named physics USD layer')
    physics_scene=scene if preserve_authored else scene.parent/physics_name
    if not preserve_authored:stage.GetRootLayer().Export(str(physics_scene))
    if hashlib.sha256(input_closure_path.read_bytes()).hexdigest()!=input_closure_sha256:
        raise RuntimeError('Input closure changed during capture')
    for entry in input_closure['files']:
        if file_sha(scene.parent/entry['path'])!=entry['sha256']:
            raise RuntimeError('Input scene dependency changed during capture')
    material_import=material_guard.check()
    report={'status':'pass' if contacts and captures and all(c['status']=='pass' for c in contacts) else 'fail',
            'native_material_import':material_import,
            'scope':request.get('scope','real_source_technical_integration'),
            'authored_scene_preserved':preserve_authored,
            'scene':str(scene),'scene_sha256':hashlib.sha256(scene.read_bytes()).hexdigest(),
            'input_dependency_closure_sha256':input_closure_sha256,'input_dependency_closure':input_closure,
            'physics_scene':str(physics_scene),'physics_scene_sha256':hashlib.sha256(physics_scene.read_bytes()).hexdigest(),
            'collision_meshes':ground_collision['collision_meshes'],
            'render_ground_meshes':terrain,'ground_collision':ground_collision,
            'collision_representation':'exact_static_triangle_mesh',
            'headless_unused_viewport_updates':False,
            'water_prims':[str(p.GetPath()) for p in water_prims],
            'runtime_material_modules':['OmniSurface.mdl'] if water_prims else [],
            'water_depth_exception':'Transparent water may return water surface or transmitted opaque bed; do not compare to opaque-ground axial rays without segmentation',
            'contacts':contacts,'frames':captures,'resolution':list(resolution),
            'physx_contact_rays':contact_rays,
            'horizontal_fov_degrees':90,'near_m':0.02,'far_m':5000,
            'renderer':renderer,'lighting':request.get('lighting','directional'),
            'renderer_recipe':renderer_recipe,'native_renderer_settings':native_renderer_settings,
            'actual_rt_subframes':rt_subframes,'generated_display_frames_used':False,
            'path_tracing_sample_budget':1024 if path_tracing else None,
            'path_tracing_rng_reproducibility':'not_qualified' if path_tracing else 'not_applicable',
            'lighting_parameters':lighting_parameters,
            'native_camera_settings':camera_settings,
            'physics_backend':'PhysX GPU dynamics with GPU broadphase',
            'depth_semantics':'axial_metres','actual_navigation_stack':'not_run',
            'appearance_qualification':'not_run','real_time_claim':False,
            'observed_wall_frames_per_second':len(captures)/(time.monotonic()-started)}
    (output/'capture_result.json').write_text(json.dumps(report,indent=2)+'\n')
    # A final frame is not completion until the scene closure and contact
    # result above have been checked. Persist that distinction for resumptions.
    checkpoint_data={'status':'complete' if report['status']=='pass' else 'failed',
                     'completed_frames':len(captures),'planned_frames':len(request['poses']),
                     'scene_sha256':report['scene_sha256'],
                     'capture_result_sha256':hashlib.sha256((output/'capture_result.json').read_bytes()).hexdigest()}
    temporary=output/'capture_checkpoint.tmp'
    temporary.write_text(json.dumps(checkpoint_data,indent=2)+'\n')
    temporary.replace(output/'capture_checkpoint.json')
except BaseException as error:
    failure={'status':'failed','reason':str(error),'traceback':traceback.format_exc()}
    (output/'capture_failure.json').write_text(json.dumps(failure,indent=2)+'\n')
    print(traceback.format_exc(),flush=True)
    raise
finally:
    app.close()
