"""Plan source-backed cave observations against the actual final ground mesh.

These are camera and contact *plans*. They neither imply simulator execution
nor certify robot access to a selected cave location.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import trimesh

from ..contracts.coordinates import CoordinateFrame
from ..io import atomic_json, sha256_file
from .capture_plan import load_ground
from . import axis_parity


def create_cave_view_plan(mesh_path: Path, ir_dir: Path, frame: CoordinateFrame,
                          output: Path | None = None, *, max_interior_views: int = 12,
                          max_portal_views: int = 8, candidate_budget: int = 256,
                          support_manifest_path: Path | None = None) -> dict:
    """Return world-space poses and support probes, using final mesh in world XYZ.

    Candidates come from exact source air immediately over natural cave floor.
    Final triangle rays determine floor/ceiling elevation and support normal.
    Nearest-triangle distance and known-source-air tests constrain each camera.
    Portal looks additionally require an unobstructed final-mesh sight segment.
    """
    if min(max_interior_views, max_portal_views) < 0 or candidate_budget < 1:
        raise ValueError("Cave planning requires nonnegative view limits and a positive candidate budget")
    mesh_path, ir_dir = Path(mesh_path), Path(ir_dir)
    from ..volumes.source_evidence import verify_source_topology,recheck_source_topology
    graph, source_evidence = verify_source_topology(ir_dir)
    graph_path = ir_dir / "topology/source_topology_graph.json"
    native_output = Path(output).parent/'cave_ground_queries' if output is not None else None
    mesh = load_ground(mesh_path, native_output)
    native = getattr(mesh, 'queries', None)
    if native is None:
        vertices, inverse = np.unique(mesh.vertices, axis=0, return_inverse=True)
        mesh = trimesh.Trimesh(vertices=vertices, faces=inverse[mesh.faces], process=False)
        volume = mesh.is_volume
    else:
        from ..volumes.native_source_adapter import _surface
        volume, _, _ = _surface(mesh_path, ir_dir, native_output, frame.record()['world_to_source'], native.bound)
    if not volume:
        raise ValueError("Cave planning requires a closed, consistently outward final ground volume")
    with np.load(ir_dir / "natural_occupancy.npz", allow_pickle=False) as source:
        occupancy, validity = source["occupancy"], source["validity"]
        minimum = source["min_xyz"].astype(float)
    with np.load(ir_dir / "topology/source_topology_labels.npz", allow_pickle=False) as topology:
        covered, labels = topology["covered_air"], topology["cavity_labels"]
    from ..volumes.support_validation import structural_support_for_validation
    structural, support_context = structural_support_for_validation(ir_dir, occupancy, support_manifest_path)
    supporting = (occupancy == 1) | structural
    if covered.shape != occupancy.shape or labels.shape != occupancy.shape:
        raise ValueError("Source topology shape differs from its source volume")
    scale = frame.metres_per_block
    shape = np.asarray(occupancy.shape)
    source_to_world = np.asarray(frame.record()["source_to_world"])
    world_to_source = np.asarray(frame.record()["world_to_source"])

    def contains(points):
        return native.contains(points,reject_ambiguous=True) if native is not None else axis_parity.contains_axis_consensus(mesh, points)[0]

    def transform(points, matrix):
        points = np.atleast_2d(np.asarray(points, dtype=float))
        return points @ matrix[:3, :3].T + matrix[:3, 3]

    def known_air(world_points):
        points = transform(world_points, world_to_source)
        indices = np.floor(points - minimum).astype(int)[:, [1, 2, 0]]
        inside = np.all(indices >= 0, axis=1) & np.all(indices < shape, axis=1)
        result = np.zeros(len(indices), dtype=bool)
        y, z, x = indices[inside].T
        result[inside] = validity[y, z, x] & (occupancy[y, z, x] == 0)
        return result

    def source_clear(points, radius):
        points = np.atleast_2d(np.asarray(points, dtype=float))
        source_points = transform(points, world_to_source)
        # Require the whole conservative camera bounding cube to contain known
        # air; six axial probes alone can miss an unknown diagonal neighbor.
        for source_point in source_points:
            low = np.floor(source_point - minimum - radius/scale).astype(int)[[1, 2, 0]]
            high = np.floor(source_point - minimum + radius/scale).astype(int)[[1, 2, 0]]
            if np.any(low < 0) or np.any(high >= shape):
                return False, 0.0
            selection = tuple(slice(a, b+1) for a, b in zip(low, high))
            if not validity[selection].all() or np.any(occupancy[selection] != 0):
                return False, 0.0
        return True, None

    clear_cache = {}
    def clear_key(point, radius):return (*np.asarray(point,float).reshape(3).tolist(),float(radius))
    def prepare_clearance(requests):
        pending = []
        for point, radius in requests:
            key = clear_key(point, radius)
            if key in clear_cache:continue
            if not source_clear(point, radius)[0]:clear_cache[key]=(False,0.)
            else:clear_cache[key]=None;pending.append((key,np.asarray(point,float),radius))
        if not pending:return
        points=np.asarray([item[1] for item in pending]);inside=contains(points)
        if native is None:_,distances,_=trimesh.proximity.closest_point(mesh,points)
        else:_,distances,_=native.closest(points)
        for (key,point,radius),solid,distance in zip(pending,inside,distances):
            clear_cache[key]=(bool(not solid and distance>=radius),float(distance))
    def clear_points(points, radius):
        points=np.atleast_2d(np.asarray(points,float));values=[clear_cache[clear_key(point,radius)] for point in points]
        return all(item[0] for item in values),min(item[1] for item in values)

    sight_cache = {}
    def sight_key(start,end):return tuple(np.r_[start,end].tolist())
    def source_sight(start,end):
        distance=float(np.linalg.norm(end-start))
        if distance<.2:return False
        steps=max(2,int(np.ceil(distance/(.2*scale))))
        return bool(known_air(np.linspace(start,end,steps)).all())
    def prepare_sights(pairs):
        pending=[]
        for start,end in pairs:
            key=sight_key(start,end)
            if key in sight_cache:continue
            if not source_sight(start,end):sight_cache[key]=False
            else:sight_cache[key]=None;pending.append((key,np.asarray(start),np.asarray(end)))
        if not pending:return
        starts=np.asarray([p[1] for p in pending]);ends=np.asarray([p[2] for p in pending]);lengths=np.linalg.norm(ends-starts,axis=1)
        if native is not None:clear=native.unobstructed(starts,ends)
        else:
            hits,rays,_=mesh.ray.intersects_location(starts,(ends-starts)/lengths[:,None],multiple_hits=True)
            clear=np.ones(len(starts),bool)
            if len(hits):clear[rays[np.linalg.norm(hits-starts[rays],axis=1)<lengths[rays]-1e-5]]=False
        for (key,_,_),okay in zip(pending,clear):sight_cache[key]=bool(okay)

    def unobstructed(start,end):return sight_cache[sight_key(start,end)]

    floor = np.zeros_like(covered)
    floor[1:] = covered[1:] & validity[1:] & (occupancy[1:] == 0) & validity[:-1] & supporting[:-1]
    candidates = np.argwhere(floor)
    candidate_xyz = candidates[:, [2, 0, 1]].astype(float) + minimum + .5
    feature_by_label = {int(feature["label"]): feature for feature in graph["features"]}
    candidate_labels = labels[tuple(candidates.T)] if len(candidates) else np.empty(0, dtype=int)
    chosen = []
    # Round-robin across cavity sizes preserves diversity rather than letting
    # one large connected cave use the entire candidate budget.
    groups = []
    for feature in sorted(graph["features"], key=lambda item: (-item["air_voxels"], item["id"])):
        indices = np.flatnonzero(candidate_labels == feature["label"])
        if len(indices):
            groups.append(indices[np.linspace(0, len(indices)-1, min(16, len(indices)), dtype=int)].tolist())
    # Reserve half the candidate budget for observed portal neighborhoods.
    portal_candidates = {}
    for portal in graph["portals"]:
        faces = np.asarray([face["center_xyz"] for face in portal["faces"]], dtype=float)
        if not len(faces) or not len(candidates):
            portal_candidates[portal["id"]] = []
            continue
        # Bounding-box distance avoids missing a long entrance near an end.
        low, high = faces.min(axis=0), faces.max(axis=0)
        delta = np.maximum(np.maximum(low - candidate_xyz, candidate_xyz - high), 0)
        distance = np.linalg.norm(delta, axis=1)
        near = np.flatnonzero(distance <= 4 / scale)
        near = near[np.argsort(distance[near], kind="stable")[:8]].tolist()
        portal_candidates[portal["id"]] = near
    for group in portal_candidates.values():
        for index in group:
            if index not in chosen and len(chosen) < candidate_budget // 2:
                chosen.append(index)
    for position in range(16):
        for group in groups:
            if position < len(group) and group[position] not in chosen and len(chosen) < candidate_budget:
                chosen.append(group[position])
    base = transform(candidate_xyz[chosen], source_to_world) if chosen else np.empty((0, 3))
    rejected = Counter()
    supports = {}
    if len(base):
        inside = contains(base)
        starts = base.copy()
        hits, rays, triangles = mesh.ray.intersects_location(starts, np.tile([0., 0., -1.], (len(base), 1)), multiple_hits=False)
        for hit, ray, triangle in zip(hits, rays, triangles):
            original = chosen[int(ray)]
            normal = mesh.face_normals[triangle]
            source_floor_y = candidate_xyz[original, 1] - .5
            ground_source = transform(hit, world_to_source)[0]
            if inside[ray]:
                rejected["source_air_center_inside_final_ground"] += 1
                continue
            if abs(ground_source[1] - source_floor_y) > .35:
                rejected["no_nearby_corresponding_source_floor"] += 1
                continue
            if normal[2] < .9:
                rejected["support_too_steep"] += 1
                continue
            ceiling = None
            feature = feature_by_label[int(candidate_labels[original])]
            cy, cz, cx = candidates[original]
            supports[original] = {"support_world_xyz": hit.tolist(), "support_normal": normal.tolist(),
                                  "support_source_class": "structural_masonry" if structural[cy-1, cz, cx] else "natural_terrain",
                                  "source_floor_xyz": [float(candidate_xyz[original, 0]), float(source_floor_y), float(candidate_xyz[original, 2])],
                                  "ceiling_world_z": ceiling, "vertical_clearance_m": ceiling-float(hit[2]) if ceiling is not None else None,
                                  "source_feature_id": feature["id"], "source_label": int(feature["label"]),
                                  "crop_truncated": feature["touches_crop_boundary"],
                                  "source_minimum_roof_thickness_m": feature["minimum_vertical_roof_thickness_m"]}
        rejected["no_downward_final_mesh_hit"] = len(base) - len(rays)

    if supports:
        support_values=list(supports.values());up=np.asarray([item['support_world_xyz'] for item in support_values])+[0,0,.02]
        hits,rays,_=mesh.ray.intersects_location(up,np.tile([0.,0.,1.],(len(up),1)),multiple_hits=False)
        for hit,ray in zip(hits,rays):
            support_values[ray]['ceiling_world_z']=float(hit[2]);support_values[ray]['vertical_clearance_m']=float(hit[2]-support_values[ray]['support_world_xyz'][2])
    clearance_requests=[]
    for support in supports.values():
        ground=np.asarray(support['support_world_xyz'])
        clearance_requests.extend((ground+[0,0,h],.12) for h in (.25,.6,1.5))
        clearance_requests.append((ground+[0,0,.75],float(np.sqrt(3)*.1)))
    prepare_clearance(clearance_requests)
    # A camera-support point alone does not establish a dropped cube footprint.
    # Measure a 40cm square around every candidate against the actual final mesh.
    footprint_offsets=np.asarray([(x,y,0.) for x in np.linspace(-.2,.2,9)
                                 for y in np.linspace(-.2,.2,9)])
    support_values=list(supports.values())
    if support_values:
        centers=np.asarray([s['support_world_xyz'] for s in support_values])
        starts=(centers[:,None,:]+footprint_offsets[None,:,:]+[0,0,.5]).reshape(-1,3)
        hits,rays,triangles=mesh.ray.intersects_location(starts,np.tile([0.,0.,-1.],(len(starts),1)),multiple_hits=False)
        floors=np.full(len(starts),np.nan);nz=np.full(len(starts),np.nan)
        floors[rays]=hits[:,2];nz[rays]=mesh.face_normals[triangles][:,2]
        for support,heights,normals in zip(support_values,floors.reshape(-1,81),nz.reshape(-1,81)):
            complete=bool(np.isfinite(heights).all())
            spread=float(np.ptp(heights)) if complete else None
            same_floor=complete and bool(np.max(np.abs(heights-support['support_world_xyz'][2]))<=.02)
            support['probe_footprint']={'sample_count':81,'width_m':.4,'all_samples_hit':complete,
                'height_spread_m':spread,'same_source_floor':same_floor,
                'suitable':bool(same_floor and spread<=.02 and np.all(normals>=.9)),
                'status':'planned_not_simulated'}

    def camera_at(support, heights):
        ground = np.asarray(support["support_world_xyz"])
        for scout_height in heights:
            eye = ground + [0, 0, scout_height]
            okay, distance = clear_points(eye, .12)
            if okay:
                return eye, scout_height, distance
        return None

    angles=np.arange(16)*np.pi/8
    look_directions=np.column_stack((np.cos(angles),np.sin(angles),np.zeros(16)))
    eye_points={}
    for support in supports.values():
        for h in (.25,.6,1.5):
            eye=np.asarray(support['support_world_xyz'])+[0,0,h]
            if clear_points(eye,.12)[0]:eye_points[tuple(eye)]=eye
    look_distances={};sight_pairs=[]
    if eye_points:
        eyes=np.asarray(list(eye_points.values()));origins=np.repeat(eyes,16,axis=0);directions=np.tile(look_directions,(len(eyes),1))
        hits,rays,_=mesh.ray.intersects_location(origins,directions,multiple_hits=False)
        distances=np.full(len(origins),8.)
        distances[rays]=np.linalg.norm(hits-origins[rays],axis=1)
        for eye,values in zip(eyes,distances.reshape(-1,16)):
            look_distances[tuple(eye)]=values
            for ray in range(16):
                length=min(4.,values[ray]*.8)
                if length>=1.:sight_pairs.append((eye,eye+look_directions[ray]*length))
    for portal in graph['portals']:
        faces_world=transform([face['center_xyz'] for face in portal['faces']],source_to_world)
        for index in portal_candidates[portal['id']]:
            if index not in supports:continue
            camera=camera_at(supports[index],(.6,1.5,.25))
            if camera is None:continue
            eye=camera[0];order=np.argsort(np.linalg.norm(faces_world-eye,axis=1),kind='stable')[:32]
            sight_pairs.extend((eye,faces_world[i]) for i in order)
    for support in supports.values():
        ground=np.asarray(support['support_world_xyz']);sight_pairs.append((ground+[0,0,.75],ground+[0,0,.11]))
    prepare_sights(sight_pairs)
    def interior_look(eye):
        distances=look_distances[tuple(eye)]
        for ray in np.argsort(-distances,kind='stable'):
            length=min(4.,distances[ray]*.8)
            if length>=1.:
                look=eye+look_directions[ray]*length
                if unobstructed(eye,look):return look,float(length)
        return None

    statics, contacts = [], []
    represented = set()
    # One view per distinct cavity first, then fill with spatially diverse views.
    ordered = sorted(supports, key=lambda index: (-feature_by_label[int(candidate_labels[index])]["air_voxels"], index))
    for unique_pass in (True, False):
        for index in ordered:
            if len([pose for pose in statics if pose["surface_scope"] == "cave_interior"]) >= max_interior_views:
                break
            support = supports[index]
            if unique_pass and support["source_feature_id"] in represented:
                continue
            if any(np.linalg.norm(np.asarray(pose["support_world_xyz"]) - support["support_world_xyz"]) < 2 for pose in statics):
                continue
            height_options = ((.25, .6, 1.5), (.6, 1.5, .25), (1.5, .6, .25))[len(statics) % 3]
            camera = camera_at(support, height_options)
            if camera is None:
                rejected["camera_clearance_or_unknown_source"] += 1
                continue
            eye, height, distance = camera
            sight = interior_look(eye)
            if sight is None:
                rejected["no_known_air_view_segment"] += 1
                continue
            look, length = sight
            statics.append({**support, "position": eye.tolist(), "look_at": look.tolist(), "kind": "static",
                            "surface_scope": "cave_interior", "scout_height_m": height, "camera_nearest_triangle_m": distance,
                            "look_segment_clear_m": length, "held_out": len(statics) % 4 == 0})
            represented.add(support["source_feature_id"])

    represented_portals = set()
    for portal in graph["portals"]:
        if len(represented_portals) >= max_portal_views:
            break
        faces = np.asarray([face["center_xyz"] for face in portal["faces"]], dtype=float)
        if not len(faces):
            continue
        faces_world = transform(faces, source_to_world)
        for index in portal_candidates[portal["id"]]:
            if index not in supports:
                continue
            support = supports[index]
            camera = camera_at(support, (.6, 1.5, .25))
            if camera is None:
                continue
            eye, height, distance = camera
            face_order = np.argsort(np.linalg.norm(faces_world-eye, axis=1), kind="stable")
            visible = next((faces_world[i] for i in face_order[:32] if unobstructed(eye, faces_world[i])), None)
            if visible is None:
                continue
            statics.append({**support, "position": eye.tolist(), "look_at": visible.tolist(), "kind": "static",
                            "surface_scope": "cave_portal", "source_portal_id": portal["id"], "scout_height_m": height,
                            "camera_nearest_triangle_m": distance, "held_out": len(statics) % 4 == 0})
            represented_portals.add(portal["id"])
            break
    contact_ids = set();missing_contacts=[]
    for pose in statics:
        candidates=sorted((s for s in supports.values() if s['source_feature_id']==pose['source_feature_id']
            and np.linalg.norm(np.asarray(s['support_world_xyz'])-pose['support_world_xyz'])<=4.),
            key=lambda s:np.linalg.norm(np.asarray(s['support_world_xyz'])-pose['support_world_xyz']))
        chosen_support=None
        for support in candidates:
            if not support['probe_footprint']['suitable']:continue
            support_key=tuple(support['support_world_xyz']);point=np.asarray(support_key)+[0,0,.75]
            okay,clearance=clear_points(point,float(np.sqrt(3)*.1))
            if not okay or not unobstructed(point,np.asarray(support_key)+[0,0,.11]):continue
            if pose['surface_scope']=='cave_portal':
                portal=next(p for p in graph['portals'] if p['id']==pose['source_portal_id'])
                faces_world=transform([face['center_xyz'] for face in portal['faces']],source_to_world)
                if np.min(np.linalg.norm(faces_world-np.asarray(support_key),axis=1))>4.:continue
            if support_key in contact_ids:
                chosen_support=support;break
            if any(np.linalg.norm(np.asarray(c['support_world_xyz'])-support_key)<.5 for c in contacts):continue
            chosen_support=support;break
        if chosen_support is None:
            missing_contacts.append({'source_feature_id':pose['source_feature_id'],
                'source_portal_id':pose.get('source_portal_id'),'surface_scope':pose['surface_scope'],
                'camera_support_world_xyz':pose['support_world_xyz'],
                'reason':'No clear, stable 40cm final-mesh footprint within4m; not simulated'})
            continue
        if support_key not in contact_ids:
            contacts.append({"position": point.tolist(), "ground_z": float(support_key[2]),
                             "support_world_xyz": list(support_key), "support_normal": chosen_support["support_normal"],
                             "surface_scope": pose["surface_scope"], "source_feature_id": pose["source_feature_id"],
                             "support_source_class": chosen_support["support_source_class"],
                             "footprint_measurement":chosen_support['probe_footprint'],
                             "camera_support_world_xyz":pose['support_world_xyz'],
                             "probe_shape": "cuboid", "half_extents_m": [.1, .1, .1], "initial_clearance_m": clearance,
                             "status": "planned_not_simulated"})
            contact_ids.add(support_key)
    # Short local support routes are source-backed physical test plans. Their
    # dense sampled footprint checks do not imply a user's navigation stack or
    # a path from the exterior into this cave has been run.
    support_routes, missing_routes = [], []
    route_features = set()
    route_poses=list(statics)
    # A camera that sees an entrance may stand on a narrow ledge. Search other
    # already verified nearby supports for a physical diagnostic route instead
    # of treating camera suitability as robot-footprint suitability.
    for portal in graph["portals"]:
        if portal["id"] not in represented_portals:
            continue
        faces_world=transform([face["center_xyz"] for face in portal["faces"]],source_to_world)
        for support in supports.values():
            point=np.asarray(support["support_world_xyz"])
            distances=np.linalg.norm(faces_world-point,axis=1)
            nearest=int(np.argmin(distances))
            if distances[nearest]<=4.:
                route_poses.append({**support,"surface_scope":"cave_portal","source_portal_id":portal["id"],
                    "look_at":faces_world[nearest].tolist(),"portal_support_distance_m":float(distances[nearest])})
    from .cave_routes import plan_cave_support_routes
    support_routes,missing_routes=plan_cave_support_routes(route_poses,mesh,native,contains,transform,world_to_source,minimum,shape,validity,occupancy)
    route_features={(route['original_surface_scope'],route.get('source_portal_id') or route['source_feature_id']) for route in support_routes}
    missing_routes=list({(route["surface_scope"],route.get("source_portal_id") or route["source_feature_id"]):route
        for route in missing_routes if (route["surface_scope"],route.get("source_portal_id") or route["source_feature_id"]) not in route_features}.values())
    missing = [{"source_feature_id": feature["id"], "kind": "cave_interior",
                "reason": "No selected final-mesh-supported clear camera within bounded scout budget"}
               for feature in graph["features"] if feature["id"] not in represented]
    missing += [{"source_feature_id": portal["id"], "kind": "cave_portal",
                 "reason": "No selected clear view from a source natural floor within4m; shafts and occluded or small openings may lack this pose"}
                for portal in graph["portals"] if portal["id"] not in represented_portals]
    result = {"schema_version": 1, "kind": "CaveViewPlan", "status": "planned" if statics else "no_clear_supported_views",
              "final_ground_sha256": sha256_file(mesh_path), "source_ir_sha256": sha256_file(ir_dir / "world_ir.json"),
              "source_topology_sha256": sha256_file(graph_path), "coordinate_frame": frame.record(),
              "structural_support": support_context, "source_payload_evidence": source_evidence,
              "poses_static": statics, "contact_probes": contacts, "support_routes": support_routes,
              "missing_contact_probes":missing_contacts,
              "missing_support_routes": missing_routes,
              "planner_sha256": sha256_file(Path(__file__)), "point_classifier_sha256": sha256_file(Path(axis_parity.__file__)),
              "available_features": {"cave_ids": sorted(represented), "portal_ids": sorted(represented_portals)},
              "missing_representative_features": missing, "candidate_source_floors": len(candidates), "candidates_tested": len(chosen),
              "actual_final_mesh_supports": len(supports), "rejected_candidates": dict(rejected),
              "method": "source covered-air natural floors; independent final triangle floor/ceiling rays, camera signed air and nearest-triangle clearance, sight segments",
              "constraints": {"camera_radius_m": .12, "contact_cube_half_extent_m": .1, "contact_circumscribed_radius_m": float(np.sqrt(3)*.1), "floor_source_correspondence_tolerance_blocks": .35, "minimum_up_normal": .9, "candidate_budget": candidate_budget},
              "limitations": ["Plans require actual Isaac capture/contact execution", "Scout teleportation does not demonstrate traversable cave access", "Finite candidate budget can miss suitable locations", "Source crop truncation remains explicit; no complete-world cave identity is asserted"]}
    recheck_source_topology(ir_dir,source_evidence)
    if native is not None:
        result["native_query_evidence"]=native.receipt()
        from . import native_source
        result["point_classifier_sha256"]=sha256_file(Path(native_source.__file__))
    if output is not None:
        atomic_json(Path(output), result)
    return result
