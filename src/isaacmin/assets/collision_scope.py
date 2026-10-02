"""Recognize explicitly owned, non-rendering terrain collision geometry.

This structural check permits material-free physics meshes only when both USD
relationships agree. Exact triangle equality and native contacts are separate
required measurements; an arbitrary hidden mesh never receives this exemption.
"""


def collision_only_paths(stage):
    from pxr import UsdGeom, UsdPhysics

    owned = set()
    marked = set()
    for prim in stage.Traverse():
        if prim.GetAttribute('isaacmin:groundCollisionPart').Get():
            marked.add(str(prim.GetPath()))
        targets = prim.GetRelationship('isaacmin:exactCollisionParts').GetTargets()
        if not targets:
            continue
        if (not prim.IsA(UsdGeom.Mesh) or 'Terrain_FinalGround' not in str(prim.GetPath())
                or prim.GetAttribute('isaacmin:groundCollisionPart').Get()
                or (prim.HasAPI(UsdPhysics.CollisionAPI)
                    and UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())):
            raise ValueError('Invalid rendered source or duplicate collision ownership')
        stop = 0
        for target in targets:
            part = stage.GetPrimAtPath(target)
            if (not part or not part.IsA(UsdGeom.Mesh) or str(target) in owned
                    or not part.GetAttribute('isaacmin:groundCollisionPart').Get()
                    or part.GetRelationship('isaacmin:renderSource').GetTargets() != [prim.GetPath()]):
                raise ValueError('Collision part is missing, duplicated or lacks reciprocal ownership')
            mesh = UsdGeom.Mesh(part)
            start = part.GetAttribute('isaacmin:sourceFaceStart').Get()
            end = part.GetAttribute('isaacmin:sourceFaceStop').Get()
            if (type(start) is not int or type(end) is not int or start != stop or end <= start
                    or end-start != len(mesh.GetFaceVertexCountsAttr().Get() or [])):
                raise ValueError('Collision part ranges are incomplete or overlapping')
            if (UsdGeom.Imageable(part).ComputeVisibility() != 'invisible'
                    or UsdGeom.Imageable(part).ComputePurpose() != 'guide'
                    or not part.HasAPI(UsdPhysics.CollisionAPI)
                    or not UsdPhysics.CollisionAPI(part).GetCollisionEnabledAttr().Get()
                    or not part.HasAPI(UsdPhysics.MeshCollisionAPI)
                    or UsdPhysics.MeshCollisionAPI(part).GetApproximationAttr().Get() != 'none'
                    or mesh.GetOrientationAttr().Get() != UsdGeom.Mesh(prim).GetOrientationAttr().Get()):
                raise ValueError('Collision part does not have exact, invisible guide semantics')
            owned.add(str(target)); stop = end
        if stop != len(UsdGeom.Mesh(prim).GetFaceVertexCountsAttr().Get() or []):
            raise ValueError('Collision ownership does not cover every source face')
    if marked != owned:
        raise ValueError('Unowned collision-only mesh marker')
    return owned
