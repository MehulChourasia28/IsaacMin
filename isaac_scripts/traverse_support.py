"""Bounded force-driven dynamic footprint probe; never a user navigation test."""
import argparse,hashlib,json,pathlib,time,traceback
parser=argparse.ArgumentParser();parser.add_argument('--request',required=True)
args,_=parser.parse_known_args();request=json.loads(pathlib.Path(args.request).read_text())
output=pathlib.Path(request['output']).resolve();output.mkdir(parents=True,exist_ok=True)
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'renderer':'RayTracedLighting',
    'extra_args':[f'--/log/file={output}/traversal_kit.log']})
try:
    import numpy as np,omni.usd,omni.physx
    from pxr import UsdGeom,UsdPhysics,PhysxSchema,Gf
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid
    from isaacsim.core.api.materials import PhysicsMaterial
    from isaacsim.core.prims import RigidPrim
    scene=pathlib.Path(request['scene']).resolve()
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    dependencies=request['scene_dependency_files']
    content_sha=hashlib.sha256(json.dumps(dependencies,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    if content_sha!=request['scene_content_sha256']:raise RuntimeError('Requested dependency manifest identity is invalid')
    if sha(pathlib.Path(__file__))!=request['native_script_sha256']:raise RuntimeError('Requested native worker identity is stale')
    if sha(scene.parent/'final_ground.obj')!=request['final_ground_sha256']:raise RuntimeError('Requested authoritative mesh identity is stale')
    def verify_requested_dependencies():
        for entry in dependencies:
            path=(scene.parent/entry['path']).resolve()
            if not path.is_relative_to(scene.parent) or not path.is_file() or path.stat().st_size!=entry['bytes'] or sha(path)!=entry['sha256']:
                raise RuntimeError('Requested scene dependency changed: '+entry['path'])
    verify_requested_dependencies()
    closure_path=scene.parent/('native_physics_dependency_closure.json' if scene.name=='world_physics.usda' else 'native_dependency_closure.json')
    closure=json.loads(closure_path.read_text());closure_sha256=sha(closure_path)
    if closure.get('status')!='pass' or closure['root_sha256']!=sha(scene):raise RuntimeError('Input native closure is invalid/stale')
    for entry in closure['files']:
        path=(scene.parent/entry['path']).resolve()
        if not path.is_relative_to(scene.parent) or sha(path)!=entry['sha256']:raise RuntimeError('Scene dependency changed before traversal')
    route=np.asarray(request['route_ground_xyz'],dtype=float)
    if route.ndim!=2 or route.shape[1]!=3 or len(route)<2 or not np.isfinite(route).all():
        raise RuntimeError('Traversal requires finite explicit supporting route XYZ')
    speed=float(request.get('speed_mps',.35));duration=float(request.get('maximum_simulation_seconds',120))
    if not .05<=speed<=.75 or not 2<=duration<=1800:raise RuntimeError('Traversal speed/duration outside bounded profile')
    dimensions=np.asarray(request.get('footprint_xyz_m',[.35,.25,.2]),float)
    if dimensions.shape!=(3,) or np.any(dimensions<=0) or np.any(dimensions>1):raise RuntimeError('Invalid probe footprint')
    if not omni.usd.get_context().open_stage(str(scene)):raise RuntimeError('Cannot open actual scene')
    for _ in range(30):app.update()
    stage=omni.usd.get_context().get_stage()
    if UsdGeom.GetStageUpAxis(stage)!='Z' or abs(UsdGeom.GetStageMetersPerUnit(stage)-1)>1e-12:
        raise RuntimeError('Traversal requires measured metre-scale Z-up world')
    from ground_collision import configure_ground_collision
    ground_collision=configure_ground_collision(stage,preserve_authored=True)
    terrain=ground_collision['collision_meshes']
    world=World(stage_units_in_meters=1,physics_dt=1/120,rendering_dt=1/30)
    world.get_physics_context().enable_gpu_dynamics(True);world.get_physics_context().set_broadphase_type('GPU')
    # Pin a physically ordinary contact material explicitly: the legacy
    # DynamicCuboid default uses static=0.2 and dynamic=1.0, an unsuitable
    # undocumented dependency for a reproducible support diagnostic.
    probe_material=PhysicsMaterial('/IsaacMinTraversal/ProbeMaterial',static_friction=.6,dynamic_friction=.5,restitution=0.)
    body=world.scene.add(DynamicCuboid('/IsaacMinTraversal/Probe',name='support_probe',
        position=route[0]+[0,0,dimensions[2]/2+.15],scale=dimensions,mass=1,physics_material=probe_material))
    view=world.scene.add(RigidPrim('/IsaacMinTraversal/Probe',name='measured_support_probe',
        reset_xform_properties=False,track_contact_forces=True,contact_filter_prim_paths_expr=terrain,max_contact_count=64))
    world.reset()
    for _ in range(240):world.step(render=False)
    query=omni.physx.get_physx_scene_query_interface();target=1;samples=[];waypoints=[0];failure=None
    local=np.array([[x,y,-dimensions[2]/2] for x in [-dimensions[0]/2,0,dimensions[0]/2] for y in [-dimensions[1]/2,0,dimensions[1]/2]])
    last_progress=0.;closest=float('inf');started=time.monotonic()
    for step in range(int(duration*120)):
        position,orientation=body.get_world_pose();velocity=body.get_linear_velocity()
        difference=route[target,:2]-position[:2];distance=float(np.linalg.norm(difference))
        if distance<.12:
            waypoints.append(target);target+=1;closest=float('inf');last_progress=step/120
            if target==len(route):break
            difference=route[target,:2]-position[:2];distance=float(np.linalg.norm(difference))
        if distance<closest-.02:closest=distance;last_progress=step/120
        desired=difference/max(distance,1e-9)*min(speed,2*distance)
        # Explicit bounded horizontal drive compensates static friction.
        # Without it, slowing near a waypoint can stall before the acceptance radius.
        drive=difference/max(distance,1e-9)*5.9
        force=np.r_[np.clip((desired-velocity[:2])*30+drive,-12,12),0.0]
        view.apply_forces(force[None,:])
        world.step(render=False)
        if step%4==0:
            position,orientation=body.get_world_pose();velocity=body.get_linear_velocity()
            # Direct double-precision quaternion rotation avoids the precision
            # loss of converting a float32 quaternion through Gf axis-angle.
            q=np.asarray(orientation,dtype=np.float64);q/=np.linalg.norm(q)
            vector=np.broadcast_to(q[1:],local.shape)
            cross=2*np.cross(vector,local)
            corners=local+q[0]*cross+np.cross(vector,cross)+position
            hits=[]
            for corner in corners:
                values=[]
                def callback(hit):
                    # raycast_all returns RaycastHit objects; only the closest
                    # query uses dictionaries in this pinned PhysX interface.
                    if str(hit.collision) in terrain:
                        values.append({'collision':str(hit.collision),'position':list(hit.position),
                                       'distance':float(hit.distance)})
                    return True
                query.raycast_all(tuple(map(float,corner+[0,0,.25])),(0.,0.,-1.),2.0,callback)
                hit=min(values,key=lambda h:h['distance']) if values else None
                hits.append({'hit':bool(hit),'position':list(hit['position']) if hit else None,
                             'distance_m':float(hit['distance']) if hit else None,
                             'collision':str(hit.get('collision','')) if hit else None})
            contact_force=np.asarray(view.get_net_contact_forces(dt=1/120))[0]
            detail=view.get_contact_force_data(dt=1/120)
            contact_counts=np.asarray(detail[4]);contact_starts=np.asarray(detail[5])
            contact_indices=[int(start+i) for start,count in zip(contact_starts.ravel(),contact_counts.ravel()) for i in range(int(count))]
            contact_details={key:np.asarray(value)[contact_indices].tolist() for key,value in zip(
                ['normal_force_n','position','normal','separation_m'],detail[:4])}
            sample={'step':step,'simulation_time_s':float(world.current_time),'route_target_index':target,
                'position':position.tolist(),'orientation_wxyz':orientation.tolist(),
                'linear_velocity_mps':velocity.tolist(),'angular_velocity_radps':body.get_angular_velocity().tolist(),
                'applied_force_n':force.tolist(),
                'measured_net_contact_force_n':contact_force.tolist(),
                'measured_contact_details':contact_details,
                'footprint_bottom_samples_world':corners.tolist(),'terrain_support_rays':hits}
            samples.append(sample)
            with (output/'traversal_samples.jsonl').open('a') as stream:stream.write(json.dumps(sample)+'\n')
            if not np.isfinite(position).all() or position[2]<route[target,2]-1:
                failure='probe_fell_below_authored_support';break
            if step/120-last_progress>10:failure='no_progress_for_10_seconds';break
    if target<len(route) and failure is None:failure='bounded_duration_exhausted'
    sample_path=output/'traversal_samples.jsonl'
    verify_requested_dependencies()
    for entry in closure['files']:
        if sha(scene.parent/entry['path'])!=entry['sha256']:raise RuntimeError('Input scene dependency changed during traversal')
    record={'status':'pass' if target==len(route) and samples else 'fail','failure':failure,
        'scene':str(scene),'scene_sha256':sha(scene),'terrain_paths':terrain,
        'scene_dependency_files':dependencies,'scene_content_sha256':content_sha,'surface_scope':request['surface_scope'],
        'scene_dependency_closure':closure,'scene_dependency_closure_sha256':closure_sha256,'final_ground_sha256':sha(scene.parent/'final_ground.obj'),
        'request_sha256':sha(pathlib.Path(args.request)),'samples':str(sample_path),'samples_sha256':sha(sample_path),
        'sample_count':len(samples),'completed_waypoint_indices':waypoints,'requested_waypoints':len(route),
        'physics_step_s':1/120,'render_policy':'Physics-only explicit stepping after initialization; no synthetic visual evidence',
        'initial_placement_only':True,'height_teleports_after_initialization':0,
        'controller':'bounded horizontal force velocity controller; gravity/contact determine vertical motion',
        'footprint_xyz_m':dimensions.tolist(),'requested_speed_mps':speed,'max_horizontal_force_per_axis_n':12,'friction_feedforward_n':5.9,
        'probe_physics_material':{'static_friction':.6,'dynamic_friction':.5,'restitution':0.},
        'actual_navigation_stack':'not_supplied_not_tested','independent_contact_comparison':'not_run',
        'scope':request.get('scope','explicit_bounded_support_route_only'),'wall_seconds':time.monotonic()-started}
    (output/'traversal_result.json').write_text(json.dumps(record,indent=2)+'\n')
except BaseException:
    (output/'traversal_failure.txt').write_text(traceback.format_exc());print(traceback.format_exc(),flush=True)
    raise
finally:app.close()
