"""Actual PhysX drops and independent vertical terrain rays on an outdoor world.

This is a simulator integration measurement. It is not a robot navigation stack,
an appearance qualification, or evidence of long-distance route traversal.
"""
import argparse
import hashlib
import json
from pathlib import Path
import traceback

parser = argparse.ArgumentParser()
parser.add_argument('--request', required=True)
args, _ = parser.parse_known_args()
request_path = Path(args.request).resolve()
request = json.loads(request_path.read_text())
scene, output = Path(request['scene']).resolve(), Path(request['output']).resolve()
output.mkdir(parents=True, exist_ok=False)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


closure_path = scene.parent / 'native_dependency_closure.json'
closure = json.loads(closure_path.read_text())
for entry in closure['files']:
    p = (scene.parent / entry['path']).resolve()
    if not p.is_relative_to(scene.parent) or sha(p) != entry['sha256']:
        raise RuntimeError('Input scene dependency changed')
from isaacsim import SimulationApp
app = SimulationApp({'headless': True, 'disable_viewport_updates': True,
    'extra_args': [f'--/log/file={output}/kit.log']})
try:
    import numpy as np
    import omni.usd
    import omni.physx
    from pxr import Usd, UsdGeom
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid
    from isaacsim.core.api.materials import PhysicsMaterial
    from isaacsim.core.prims import RigidPrim
    from ground_collision import configure_ground_collision
    if not omni.usd.get_context().open_stage(str(scene)):
        raise RuntimeError('Native scene open failed')
    stage = omni.usd.get_context().get_stage()
    collision = configure_ground_collision(stage, preserve_authored=True)
    if len(collision['render_ground_meshes']) != 1:
        raise RuntimeError('This measurement expects one complete outdoor region')
    prim = stage.GetPrimAtPath(collision['render_ground_meshes'][0])
    mesh = UsdGeom.Mesh(prim)
    if not np.all(np.asarray(mesh.GetFaceVertexCountsAttr().Get()) == 3):
        raise RuntimeError('Authoritative ground is not triangulated')
    vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32).reshape(-1, 3)
    transform = np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    vertices = vertices @ transform[:3, :3] + transform[3, :3]
    shape = tuple(request['grid_shape'])
    grid = vertices.reshape(*shape, 3)
    xs, minus_ys = grid[0, :, 0], -grid[:, 0, 1]
    if not (np.all(np.diff(xs) > 0) and np.all(np.diff(minus_ys) > 0)):
        raise RuntimeError('Outdoor lattice orientation differs from the declared contract')

    def vertical_ground(points):
        """Interpolate actual rendered USD triangles, not the constructor heights."""
        x, my = points[:, 0], -points[:, 1]
        ix = np.clip(np.searchsorted(xs, x, side='right')-1, 0, len(xs)-2)
        iy = np.clip(np.searchsorted(minus_ys, my, side='right')-1, 0, len(minus_ys)-2)
        a = (x-xs[ix])/(xs[ix+1]-xs[ix])
        b = (my-minus_ys[iy])/(minus_ys[iy+1]-minus_ys[iy])
        selected = 2*(iy*(shape[1]-1)+ix)+(a+b>1).astype(int)
        tri = vertices[faces[selected]]
        normals = np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0])
        height = tri[:, 0, 2]-(normals[:, 0]*(x-tri[:, 0, 0])+normals[:, 1]*(points[:, 1]-tri[:, 0, 1]))/normals[:, 2]
        valid = (x>=xs[0])&(x<=xs[-1])&(my>=minus_ys[0])&(my<=minus_ys[-1])
        return np.where(valid, height, np.nan)

    world = World(stage_units_in_meters=1, physics_dt=1/120, rendering_dt=1/30)
    world.get_physics_context().enable_gpu_dynamics(True)
    world.get_physics_context().set_broadphase_type('GPU')
    material = PhysicsMaterial('/OutdoorProbe/Material', static_friction=.6, dynamic_friction=.5, restitution=0.)
    probes = []
    for i, point in enumerate(request['probe_ground_xyz']):
        probes.append(world.scene.add(DynamicCuboid('/OutdoorProbe/P'+str(i), name='probe'+str(i),
            position=np.asarray(point)+[0., 0., .4], scale=np.array([.2, .2, .2]), mass=1., physics_material=material)))
    view = world.scene.add(RigidPrim('/OutdoorProbe/P*', name='probe_contacts',
        reset_xform_properties=False, track_contact_forces=True,
        contact_filter_prim_paths_expr=collision['collision_meshes'], max_contact_count=256))
    for _ in range(30): app.update()
    omni.physx.get_physx_interface().force_load_physics_from_usd()
    world.reset()
    for _ in range(360): world.step(render=False)
    force = np.asarray(view.get_net_contact_forces(dt=1/120))
    axis = np.linspace(-.1, .1, 17)
    surface = []
    for fixed in range(3):
        other = [i for i in range(3) if i != fixed]
        for side in (-.1, .1):
            a, b = np.meshgrid(axis, axis)
            p = np.zeros((a.size, 3)); p[:, fixed] = side
            p[:, other[0]], p[:, other[1]] = a.ravel(), b.ravel(); surface.append(p)
    surface = np.unique(np.concatenate(surface), axis=0)
    drops = []
    for i, body in enumerate(probes):
        position, orientation = body.get_world_pose()
        q = np.asarray(orientation, float); q /= np.linalg.norm(q)
        v = np.broadcast_to(q[1:], surface.shape); cross = 2*np.cross(v, surface)
        points = surface+q[0]*cross+np.cross(v, cross)+position
        delta = points[:, 2]-vertical_ground(points)
        gap = float(np.min(np.abs(delta)))
        penetration = float(np.max(np.maximum(-delta, 0)))
        linear, angular = body.get_linear_velocity(), body.get_angular_velocity()
        displacement = float(np.linalg.norm(position[:2]-np.asarray(request['probe_ground_xyz'][i])[:2]))
        passed = bool(np.isfinite(delta).all() and gap<=.02 and penetration<=.02 and
            np.linalg.norm(linear)<=.03 and np.linalg.norm(angular)<=.05 and displacement<=.5 and force[i, 2]>0)
        drops.append(dict(id=i, position=position.tolist(), orientation_wxyz=orientation.tolist(),
            linear_velocity_mps=linear.tolist(), angular_velocity_radps=angular.tolist(),
            measured_contact_force_n=force[i].tolist(), maximum_vertical_penetration_m=penetration,
            sampled_surface_support_gap_m=gap, horizontal_displacement_m=displacement,
            surface_samples=len(points), status='pass' if passed else 'fail'))
    stage.RemovePrim('/OutdoorProbe')
    for _ in range(4): world.step(render=False)
    query = omni.physx.get_physx_scene_query_interface()
    rng = np.random.default_rng(41279)
    ids = np.floor(np.linspace(0, len(faces), 12000, endpoint=False)).astype(int)
    tri = vertices[faces[ids]]
    uv = rng.random((len(ids), 2)); uv[uv.sum(axis=1)>1] = 1-uv[uv.sum(axis=1)>1]
    expected = tri[:, 0]+uv[:, :1]*(tri[:, 1]-tri[:, 0])+uv[:, 1:]*(tri[:, 2]-tri[:, 0])
    measured = np.full(expected.shape, np.nan); hit_paths = []
    for i, point in enumerate(expected):
        hit = query.raycast_closest(tuple(map(float, point+[0, 0, .5])), (0., 0., -1.), 1., True)
        hit_paths.append(str(hit.get('collision', '')))
        if hit.get('hit') and hit_paths[-1] in collision['collision_meshes']:
            measured[i] = hit['position']
    errors = np.linalg.norm(measured-expected, axis=1)
    np.savez_compressed(output/'vertical_ray_samples.npz', expected_rendered_points=expected,
        actual_PhysX_points=measured, error_m=errors, rendered_face_indices=ids)
    rays_pass = bool(np.isfinite(errors).all() and np.max(errors)<=.02)
    record = dict(status='pass' if rays_pass and all(d['status']=='pass' for d in drops) else 'fail',
        scene=str(scene), scene_sha256=sha(scene), closure_sha256=sha(closure_path),
        ray_count=len(ids), ray_status='pass' if rays_pass else 'fail',
        missing_or_wrong_collider_rays=int(np.sum(~np.isfinite(errors))),
        maximum_ray_error_m=float(np.max(errors)) if np.isfinite(errors).all() else None,
        drops=drops, physics_dt_s=1/120, gravity_steps=360,
        ray_method='Actual rendered USD triangle points vs native PhysX vertical hits; one above-ground surface',
        probe_method='Measured native body pose, velocity and force; oriented cube-face samples vs actual USD triangles',
        request_sha256=sha(request_path), producer_sha256=sha(__file__),
        scope='sampled regional ground integration only', full_contact_coverage='not_inferred',
        visual_qualification='not_run_by_this_worker', navigation_stack='not_run_not_supplied')
    (output/'ground_probe_result.json').write_text(json.dumps(record, indent=2)+'\n')
    print({k:record[k] for k in ('status','ray_count','maximum_ray_error_m')}, flush=True)
except BaseException:
    (output/'failure.txt').write_text(traceback.format_exc())
    raise
finally:
    app.close()
