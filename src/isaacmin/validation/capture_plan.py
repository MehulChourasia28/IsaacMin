"""Independent evaluation views and support probes against final triangle geometry."""
from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import trimesh

from ..io import atomic_json, sha256_file
from ..terrain.routes import route_candidate


def load_ground(path: Path, query_output: Path | None = None):
    from .native_ground import has_native_ground
    if has_native_ground(path):
        if query_output is None:
            raise ValueError('Native ground queries require an explicit immutable evidence directory')
        from .ground_queries import NativeGroundMesh
        return NativeGroundMesh(path, query_output)
    scene = trimesh.load(path, force="scene", process=False)
    mesh = scene.to_geometry()
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError("Final ground OBJ has no triangle geometry")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError("Final ground geometry is nonfinite")
    return mesh


def ground_support(mesh, xy, *, from_z=None):
    xy = np.atleast_2d(np.asarray(xy, float))
    z = float(mesh.bounds[1, 2] + 2) if from_z is None else float(from_z)
    origins = np.column_stack((xy, np.full(len(xy), z)))
    dirs = np.tile([0., 0., -1.], (len(xy), 1))
    locations, rays, triangles = mesh.ray.intersects_location(origins, dirs, multiple_hits=False)
    height = np.full(len(xy), np.nan)
    normals = np.full((len(xy), 3), np.nan)
    height[rays] = locations[:, 2]
    normals[rays] = mesh.face_normals[triangles]
    return height, normals


def continuous_motion(route_points, mesh, *, hz=30, speed_m_s=.75, length_limit_m=40.):
    """A slow ground camera, timed stops, a stationary look-around and reverse travel.

    This is a scripted evaluation camera, never a claimed robot traversal. Path
    smoothing is limited to orientation; positions remain on the dry source path.
    """
    route_points = np.asarray(route_points, float)
    if len(route_points) < 2 or hz <= 0 or speed_m_s <= 0:
        raise ValueError("Motion needs a supported route, positive speed and cadence")
    distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(route_points[:, :2], axis=0), axis=1))]
    unique = np.r_[True, np.diff(distance) > 1e-8]
    distance, route_points = distance[unique], route_points[unique]
    length = min(float(distance[-1]), length_limit_m)
    if length < 1:
        raise ValueError("Continuous source route is shorter than one metre")
    # Cosine acceleration/deceleration over one second. Integrate speed first;
    # increasing sample count instead of speed preserves the declared maximum.
    duration = length / speed_m_s + 1.
    # One extra integration interval makes the sampled cosine ramp span the
    # route even when exact-duration floating arithmetic undershoots by an ULP.
    # Rescaling then only lowers speed; no speed or quality limit is relaxed.
    frames = int(math.ceil(duration * hz)) + 2
    t = np.arange(frames) / hz
    ramp = np.minimum(np.minimum(t, t[-1]-t), 1.)
    velocity = speed_m_s * .5 * (1.-np.cos(np.pi * np.clip(ramp, 0, 1)))
    travel = np.r_[0., np.cumsum((velocity[1:]+velocity[:-1])*.5/hz)]
    if travel[-1] < length:
        raise ValueError("Motion timing failed to span the requested route")
    travel *= length / travel[-1]
    forward = np.column_stack([np.interp(travel, distance, route_points[:, a]) for a in range(3)])
    # Look along a two-metre chord to avoid ninety-degree one-frame camera snaps
    # at source-cell corners. The path itself is not allowed to cut corners.
    before = np.column_stack([np.interp(np.maximum(0., travel-1.), distance, route_points[:, a]) for a in (0, 1)])
    after = np.column_stack([np.interp(np.minimum(distance[-1], travel+1.), distance, route_points[:, a]) for a in (0, 1)])
    tangent = after-before
    yaw = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
    yaw += .15*np.sin(2*np.pi*travel/max(length, 1.))
    stop_frames = hz
    look_frames = 4*hz
    look_t = np.linspace(0., 1., look_frames)
    # A full look-around returns to the original heading before driving backwards.
    look_yaw = yaw[-1] + 2*np.pi*(.5-.5*np.cos(np.pi*look_t))
    points = np.concatenate([np.repeat(forward[:1], stop_frames, axis=0), forward,
        np.repeat(forward[-1:], stop_frames+look_frames+stop_frames, axis=0),
        forward[-2::-1], np.repeat(forward[:1], stop_frames, axis=0)])
    headings = np.r_[np.full(stop_frames, yaw[0]), yaw,
        np.full(stop_frames, yaw[-1]), look_yaw, np.full(stop_frames, yaw[-1]+2*np.pi),
        yaw[-2::-1]+2*np.pi, np.full(stop_frames, yaw[0]+2*np.pi)]
    kinds = (["stop"]*stop_frames + ["forward"]*len(forward) + ["stop"]*stop_frames
             + ["turn"]*look_frames + ["stop"]*stop_frames + ["reverse"]*(len(forward)-1)
             + ["stop"]*stop_frames)
    ground_z, _ = ground_support(mesh, points[:, :2])
    if not np.isfinite(ground_z).all():
        raise ValueError("Final surface does not support every continuous route position")
    points[:, 2] = ground_z
    # The measured three-dimensional speed is retained; ramps may increase it
    # above the declared horizontal speed. Both are engineering observations.
    speeds = np.linalg.norm(np.diff(points, axis=0), axis=1)*hz
    poses = []
    for i, (point, heading, kind) in enumerate(zip(points, headings, kinds)):
        eye = point + [0, 0, .6]
        look = eye + [8*math.cos(heading), 8*math.sin(heading), -.3]
        poses.append({"position": eye.tolist(), "look_at": look.tolist(), "kind": "motion",
                      "motion_phase": kind, "frame": i, "design_time_s": i/hz})
    return poses, {"unique_horizontal_distance_target_m": length,
                   "maximum_horizontal_speed_m_s": speed_m_s,
                   "maximum_measured_3d_speed_m_s": float(speeds.max()),
                   "design_hz": hz, "design_duration_s": (len(poses)-1)/hz,
                   "return_and_revisit": True, "positions": "dry source path, final-mesh support",
                   "motion_kind": "scripted ground camera; no navigation stack or vehicle dynamics"}


def create_capture_plan(mesh_path: Path, source_surface: Path, output: Path, *, origin=(-1065.38, 0, 688.07),
                        width=1280, height=720, speed_m_s=.75, motion_length_m=40.) -> dict:
    mesh = load_ground(mesh_path, Path(output).parent/'capture_ground_queries')
    s = np.load(source_surface, allow_pickle=False)
    h = s["height"]
    wet = s["water_validity"] & (s["water_height"] >= h)
    spacing = float(s["sample_spacing_m"])
    zz, xx = np.indices(h.shape)
    sample_xy = np.column_stack((s["min_xz"][0]+xx.ravel()*spacing+.5-origin[0],
                                 -(s["min_xz"][1]+zz.ravel()*spacing+.5-origin[2])))
    support_height, _ = ground_support(mesh, sample_xy)
    support_height = support_height.reshape(h.shape)
    route = route_candidate(support_height, s["validity"] & np.isfinite(support_height), wet, spacing_m=spacing,
                            centre_index=(h.shape[0]//2, h.shape[1]//2),
                            origin_xz=tuple(s["min_xz"]+.5), world_origin=origin, max_grade=.35)
    route_points = np.asarray(route.get("points_world_xyz", []))
    if len(route_points) < 8:
        raise ValueError("Not enough source-supported route for independent motion views")
    # Use interior path points, avoiding a camera at a crop boundary.
    lo, hi = mesh.bounds
    inside = ((route_points[:, 0] > lo[0]+4) & (route_points[:, 0] < hi[0]-4)
              & (route_points[:, 1] > lo[1]+4) & (route_points[:, 1] < hi[1]-4))
    route_points = route_points[inside]
    if len(route_points) < 8:
        raise ValueError("Insufficient interior route")
    # Select a continuous segment, not disjoint camera teleportation through excluded water.
    gaps = np.where(np.linalg.norm(np.diff(route_points[:, :2], axis=0), axis=1) > spacing*1.5)[0]+1
    segments = np.split(route_points, gaps)
    route_points = max(segments, key=len)
    motion, timing = continuous_motion(route_points, mesh, speed_m_s=speed_m_s, length_limit_m=motion_length_m)
    support_distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(route_points[:, :2], axis=0), axis=1))]
    support_length = min(8., float(support_distance[-1]))
    support_t = np.linspace(0., support_length, int(np.ceil(support_length/.2))+1)
    support_xy = np.column_stack([np.interp(support_t, support_distance, route_points[:, axis]) for axis in (0, 1)])
    support_z, _ = ground_support(mesh, support_xy)
    if not np.isfinite(support_z).all():
        raise ValueError("Dynamic support diagnostic route has missing final ground")
    support_routes = [{"id": "exterior_initial_route", "surface_scope": "exterior",
                       "points_world_xyz": np.column_stack((support_xy, support_z)).tolist(),
                       "planned_distance_m": support_length,
                       "status": "planned_not_physically_traversed",
                       "provenance": "First eight metres of the authored dry-ground scenario route"}]
    rng = np.random.default_rng(93497)
    dry = np.argwhere(s["validity"] & ~wet)
    picks = dry[rng.choice(len(dry), size=min(len(dry), 300), replace=False)]
    xy = np.column_stack((s["min_xz"][0]+picks[:, 1]+.5-origin[0],
                          -(s["min_xz"][1]+picks[:, 0]+.5-origin[2])))
    z, normals = ground_support(mesh, xy)
    okay = np.flatnonzero(np.isfinite(z) & (normals[:, 2] > .9))
    if len(okay) < 24:
        raise ValueError("Not enough diverse supporting scout positions")
    statics, contacts = [], []
    for i, selected in enumerate(okay[:24]):
        scout_height = (.25, .6, 1.5)[i % 3]
        eye = [float(xy[selected, 0]), float(xy[selected, 1]), float(z[selected]+scout_height)]
        angle = float(rng.uniform(0, 2*math.pi))
        look = [eye[0]+8*math.cos(angle), eye[1]+8*math.sin(angle), eye[2]-.2]
        statics.append({"position": eye, "look_at": look, "kind": "static", "scout_height_m": scout_height,
                        "held_out": i % 4 == 0})
        if i < 8:
            contacts.append({"position": [eye[0], eye[1], float(z[selected]+1)],
                             "ground_z": float(z[selected]), "support_normal": normals[selected].tolist(),
                             "surface_scope": "exterior"})
    from .drop_plan import plan_exterior_drops
    contacts, missing_contacts = plan_exterior_drops(mesh, contacts, s, origin=origin)
    result = {"status": "planned", "final_ground_sha256": sha256_file(mesh_path),
              "source_surface_sha256": sha256_file(source_surface), "resolution": [width, height],
              "poses_static": statics, "poses_motion": motion, "contact_probes": contacts,
              "missing_exterior_contact_probes": missing_contacts,
              "support_routes": support_routes,
              "unique_motion_distance_target_m": timing["unique_horizontal_distance_target_m"],
              "motion_timing": timing, "route_points_world_xyz": route_points.tolist(),
              "route_height_source": "independent vertical queries of final ground triangles",
              "held_out_static_fraction": .25,
              "required_lighting": ["diffuse", "directional"], "navigation_stack": "not_run_not_supplied",
              "limitations": ["Cave interior/portal scout and contact poses must be added from source topology",
                              "No pose plan establishes actual RGB, depth, contact or temporal quality"]}
    if hasattr(mesh, 'receipt'):
        result['native_query_evidence'] = mesh.receipt()
    atomic_json(output, result)
    return result
