"""Real CPU OpenUSD/CGAL fixture checks; never actual-map appearance evidence."""
from pathlib import Path
import numpy as np
import pytest
import trimesh

from test_source_native_integration import write_native_fixture

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / '.tools/geometry_validation/obj_compare_build_manifest.json').exists(),
                   reason='Pinned native exact OBJ comparator unavailable')
def test_native_queries_require_every_actual_usd_support_value_to_match(tmp_path):
    pytest.importorskip('pxr.Usd')
    from pxr import Usd, UsdGeom
    from isaacmin.assets.native_support import exported_native_ground
    mesh = trimesh.creation.box()
    mesh.apply_translation((0, 0, .123451))
    obj = write_native_fixture(tmp_path / 'scene', mesh.vertices, mesh.faces)
    vertices = np.asarray(mesh.vertices, np.float32)
    scene = obj.parent / 'world.usda'
    stage = Usd.Stage.CreateNew(str(scene))
    usd = UsdGeom.Mesh.Define(stage, '/World/Terrain_FinalGround')
    usd.CreatePointsAttr(vertices.tolist())
    usd.CreateFaceVertexCountsAttr([3] * len(mesh.faces))
    usd.CreateFaceVertexIndicesAttr(mesh.faces.ravel().tolist())
    stage.GetRootLayer().Save()
    ground, digest, _ = exported_native_ground(stage, obj, tmp_path / 'evidence/native_support')
    points, rays, _ = ground.ray.intersects_location([[0, 0, 2]], [[0, 0, -1]], multiple_hits=False)
    assert rays.tolist() == [0]
    # The query computes a double-precision line intersection; its coordinate
    # can differ by one double ULP. USD/array identity itself remains exact.
    assert points[0, 2] == pytest.approx(float(vertices[:, 2].max()), rel=0, abs=1e-12)
    assert digest and ground.usd_identity['sha256']
    changed = vertices.copy()
    changed[0, 2] += .05
    usd.GetPointsAttr().Set(changed.tolist())
    stage.GetRootLayer().Save()
    with pytest.raises(ValueError, match='USD ground coordinates differ'):
        exported_native_ground(stage, obj, tmp_path / 'evidence/rejected')
