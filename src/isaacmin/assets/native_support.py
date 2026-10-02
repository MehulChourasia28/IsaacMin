"""Independently bind actual USD support to exact retained native triangles."""
import hashlib
from pathlib import Path
import numpy as np


def exported_native_ground(stage, mesh_path, output):
    """Full USD coordinate/index equality, followed by bounded native queries.

    This compares every authored USD ground value; using the earlier OBJ alone
    would not establish that the export still provides the same support.
    """
    from pxr import UsdGeom
    from isaacmin.validation.ground_queries import NativeGroundMesh
    from isaacmin.validation.global_intersections import file_record
    from isaacmin.io import atomic_json
    ground = NativeGroundMesh(Path(mesh_path), Path(output))
    prims = [p for p in stage.Traverse()
             if p.IsA(UsdGeom.Mesh) and 'Terrain_FinalGround' in str(p.GetPath())]
    if len(prims) != 1:
        raise ValueError('Exactly one native final USD ground owner is required')
    prim = prims[0]
    matrix = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim), float)
    if not np.array_equal(matrix, np.eye(4)):
        raise ValueError('Final native USD ground has an unexpected world transform')
    mesh = UsdGeom.Mesh(prim)
    points = np.asarray(mesh.GetPointsAttr().Get())
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    if points.shape != ground.vertices.shape or len(counts) != len(ground.faces) or indices.shape != (ground.faces.size,):
        raise ValueError('Native USD and retained final support topology sizes differ')
    digest = hashlib.sha256()
    for start in range(0, len(points), 131072):
        values = points[start:start+131072]
        if not np.array_equal(values, ground.vertices[start:start+131072]):
            raise ValueError('Actual USD ground coordinates differ from exact final support')
        digest.update(memoryview(np.ascontiguousarray(values, dtype='<f8')))
    triangles = indices.reshape(-1, 3)
    for start in range(0, len(triangles), 131072):
        values = triangles[start:start+131072]
        if not np.all(counts[start:start+131072] == 3) or not np.array_equal(values, ground.faces[start:start+131072]):
            raise ValueError('Actual USD ground indices differ from exact final support')
        digest.update(memoryview(np.ascontiguousarray(values, dtype='<i8')))
    identity = {'status': 'pass', 'scope': 'Every actual USD coordinate and index equals retained final native geometry',
        'usd_prim': str(prim.GetPath()), 'vertices_compared': len(points), 'triangles_compared': len(triangles),
        'world_float64_int64_ground_sha256': digest.hexdigest(),
        'native_ground_identity': file_record(ground.queries.bound['identity_report'], 'exact_OBJ_native_identity'),
        'producer': file_record(__file__, 'exported_support_identity_validator')}
    path = Path(output) / 'usd_ground_identity.json'
    atomic_json(path, identity)
    ground.usd_identity = file_record(path, 'all_actual_USD_support_coordinates_and_indices')
    return ground, digest.hexdigest(), prim
