"""Exact, bounded triangle colliders for large multi-level terrain.

Visible ground remains untouched. Separate guide meshes own disjoint source
triangle ranges; all local positions, winding and world transforms are retained.
PhysX and independent ray/contact results are still required after construction.
"""
from pathlib import Path
import hashlib
import json
import os

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, Vt

PART = 'isaacmin:groundCollisionPart'
PARTS = 'isaacmin:exactCollisionParts'


def file_sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def render_ground(stage):
    return [p for p in stage.Traverse()
            if p.IsA(UsdGeom.Mesh) and 'Terrain_FinalGround' in str(p.GetPath())
            and not p.GetAttribute(PART).Get()]


def ground_colliders(stage):
    paths = []
    for source in render_ground(stage):
        targets = source.GetRelationship(PARTS).GetTargets()
        paths.extend(map(str, targets)) if targets else paths.append(str(source.GetPath()))
    return paths


def _arrays(prim):
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get(), dtype=np.int32)
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=np.int32)
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or not np.all(counts == 3) or len(indices) != 3 * len(counts)):
        raise ValueError('Exact collision requires finite final triangle geometry')
    if indices.size and (indices.min() < 0 or indices.max() >= len(points)):
        raise ValueError('Collision source triangle index out of range')
    return points, indices.reshape(-1, 3)


def verify_exact_parts(stage):
    """Remeasure all saved part triangles, including winding and transforms."""
    records = []
    for source in render_ground(stage):
        targets = source.GetRelationship(PARTS).GetTargets()
        if not targets:
            continue
        if source.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(source).GetCollisionEnabledAttr().Get():
            raise ValueError('Duplicate collision ownership on rendered ground')
        points, faces = _arrays(source)
        world = UsdGeom.Xformable(source).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        stop = 0
        for path in targets:
            part = stage.GetPrimAtPath(path)
            if not part or not part.GetAttribute(PART).Get():
                raise ValueError('Missing explicitly bound collision part')
            if part.GetRelationship('isaacmin:renderSource').GetTargets() != [source.GetPath()]:
                raise ValueError('Collision part lost its reciprocal source ownership')
            if UsdGeom.Mesh(part).GetOrientationAttr().Get() != UsdGeom.Mesh(source).GetOrientationAttr().Get():
                raise ValueError('Collision orientation differs from rendered source')
            start = part.GetAttribute('isaacmin:sourceFaceStart').Get()
            end = part.GetAttribute('isaacmin:sourceFaceStop').Get()
            if start != stop or not start < end <= len(faces):
                raise ValueError('Collision parts have missing or duplicate triangle ownership')
            local, triangles = _arrays(part)
            if len(triangles) != end - start:
                raise ValueError('Collision triangle range differs from saved geometry')
            if UsdGeom.Xformable(part).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) != world:
                raise ValueError('Collision world transform differs from rendered surface')
            for begin in range(0, len(triangles), 65536):
                count = min(65536, len(triangles) - begin)
                if not np.array_equal(local[triangles[begin:begin+count]], points[faces[start+begin:start+begin+count]]):
                    raise ValueError('Collision changed a source vertex or triangle winding')
            if (UsdGeom.Imageable(part).ComputePurpose() != 'guide'
                    or UsdGeom.Imageable(part).ComputeVisibility() != 'invisible'):
                raise ValueError('Duplicate collision geometry must be a non-rendered guide')
            collision = UsdPhysics.CollisionAPI(part)
            if (not collision or not collision.GetCollisionEnabledAttr().Get()
                    or UsdPhysics.MeshCollisionAPI(part).GetApproximationAttr().Get() != 'none'):
                raise ValueError('Exact triangle collision is not enabled on a part')
            stop = end
        if stop != len(faces):
            raise ValueError('Collision does not cover every final ground triangle')
        records.append({'source': str(source.GetPath()), 'triangles': len(faces), 'parts': len(targets),
                        'all_triangle_positions_and_winding_equal': True})
    owned = {path for source in render_ground(stage)
             for path in source.GetRelationship(PARTS).GetTargets()}
    marked = {p.GetPath() for p in stage.Traverse() if p.GetAttribute(PART).Get()}
    if owned != marked:
        raise ValueError('Unowned collision part or missing ownership marker')
    return records


def configure_ground_collision(stage, *, preserve_authored=False, maximum_faces=750000):
    if type(maximum_faces) is not int or not 1 <= maximum_faces <= 1000000:
        raise ValueError('Collision partition size must remain bounded')
    sources = render_ground(stage)
    if not sources:
        raise ValueError('No final rendered ground')
    for source in sources:
        if source.GetRelationship(PARTS).GetTargets():
            continue
        points, faces = _arrays(source)
        if len(faces) <= maximum_faces:
            if preserve_authored:
                collision = UsdPhysics.CollisionAPI(source)
                if (not collision or not collision.GetCollisionEnabledAttr().Get()
                        or UsdPhysics.MeshCollisionAPI(source).GetApproximationAttr().Get() != 'none'):
                    raise ValueError('Saved small ground lacks its collider')
            else:
                UsdPhysics.CollisionAPI.Apply(source).CreateCollisionEnabledAttr(True)
                UsdPhysics.MeshCollisionAPI.Apply(source).CreateApproximationAttr('none')
            continue
        if preserve_authored:
            raise ValueError('Saved large ground lacks exact collision partitions')
        identity = hashlib.sha256()
        identity.update(memoryview(points).cast('B')); identity.update(memoryview(faces).cast('B'))
        world = UsdGeom.Xformable(source).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        identity.update(str(world).encode()); identity.update(str(source.GetPath()).encode())
        identity.update(str(maximum_faces).encode())
        name = 'ground_collision_' + identity.hexdigest()[:24] + '.usdc'
        directory = Path(stage.GetRootLayer().realPath).parent
        target = directory / name
        if target.exists():
            raise ValueError('Preserve previous collision output; use a fresh scene directory')
        staging = target.with_name(target.stem + '.staging.usdc')
        if staging.exists():
            raise ValueError('Preserve incomplete collision staging')
        parts_stage = Usd.Stage.CreateNew(str(staging))
        scope = '/IsaacMinCollision/G' + identity.hexdigest()[:16]
        targets = []
        for index, start in enumerate(range(0, len(faces), maximum_faces)):
            stop = min(start + maximum_faces, len(faces))
            used, remapped = np.unique(faces[start:stop], return_inverse=True)
            local = np.ascontiguousarray(points[used])
            triangles = np.ascontiguousarray(remapped, dtype=np.int32).reshape(-1, 3)
            path = scope + '/P' + str(index)
            part = UsdGeom.Mesh.Define(parts_stage, path)
            part.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(local))
            part.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(triangles), 3, np.int32)))
            part.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(triangles.ravel()))
            part.CreateSubdivisionSchemeAttr('none')
            part.CreateOrientationAttr(UsdGeom.Mesh(source).GetOrientationAttr().Get())
            part.CreateExtentAttr([Gf.Vec3f(*map(float, local.min(axis=0))),
                                   Gf.Vec3f(*map(float, local.max(axis=0)))])
            part.CreatePurposeAttr('guide'); part.CreateVisibilityAttr('invisible')
            part.AddTransformOp().Set(world)
            prim = part.GetPrim()
            prim.CreateAttribute(PART, Sdf.ValueTypeNames.Bool, custom=True).Set(True)
            prim.CreateAttribute('isaacmin:sourceFaceStart', Sdf.ValueTypeNames.Int64, custom=True).Set(start)
            prim.CreateAttribute('isaacmin:sourceFaceStop', Sdf.ValueTypeNames.Int64, custom=True).Set(stop)
            prim.CreateRelationship('isaacmin:renderSource', custom=True).SetTargets([source.GetPath()])
            UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
            UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr('none')
            targets.append(Sdf.Path(path))
        UsdPhysics.CollisionAPI.Apply(parts_stage.OverridePrim(source.GetPath())).CreateCollisionEnabledAttr(False)
        parts_stage.GetPrimAtPath(source.GetPath()).CreateRelationship(PARTS, custom=True).SetTargets(targets)
        parts_stage.GetRootLayer().Save()
        del parts_stage
        os.replace(staging, target)
        stage.GetRootLayer().subLayerPaths.insert(0, name)
        UsdPhysics.CollisionAPI.Apply(source).CreateCollisionEnabledAttr(False)
    comparisons = verify_exact_parts(stage)
    colliders = ground_colliders(stage)
    # Optional target schema is available only inside the pinned Isaac runtime.
    try:
        from pxr import PhysxSchema
    except ImportError:
        PhysxSchema = None
    for path in colliders:
        prim = stage.GetPrimAtPath(path)
        if PhysxSchema and not preserve_authored:
            collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            collision.CreateContactOffsetAttr(0.002); collision.CreateRestOffsetAttr(0)
    return {'render_ground_meshes': [str(p.GetPath()) for p in sources],
            'collision_meshes': colliders, 'partition_comparisons': comparisons,
            'maximum_faces_per_part': maximum_faces, 'geometry_simplification': False,
            'native_PhysX_contact_qualification': 'not_run'}
