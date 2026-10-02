"""Exact collision partition properties using actual native CPU OpenUSD."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def module():
    pytest.importorskip('pxr.Usd')
    path = Path(__file__).resolve().parents[1] / 'isaac_scripts/ground_collision.py'
    spec = importlib.util.spec_from_file_location('exact_ground_collision', path)
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result)
    return result


def scene(tmp_path):
    from pxr import Usd, UsdGeom, Gf
    stage = Usd.Stage.CreateNew(str(tmp_path / 'world.usda'))
    parent = UsdGeom.Xform.Define(stage, '/World/Terrain_FinalGround')
    parent.AddTranslateOp().Set((-14.125, 3.875, 27.5))
    parent.AddRotateZOp().Set(18.0)
    mesh = UsdGeom.Mesh.Define(stage, '/World/Terrain_FinalGround/Surface')
    # Three stacked floors/ceilings and opposing windings remain distinct.
    points = [(0, 0, z) for z in (0, 2, 4)] + [(1, 0, z) for z in (0, 2, 4)] + [(0, 1, z) for z in (0, 2, 4)]
    mesh.CreatePointsAttr(points); mesh.CreateFaceVertexCountsAttr([3] * 6)
    mesh.CreateFaceVertexIndicesAttr([0, 3, 6, 1, 7, 4, 2, 5, 8, 0, 6, 3, 1, 4, 7, 2, 8, 5])
    mesh.CreateSubdivisionSchemeAttr('none'); stage.GetRootLayer().Save()
    return stage


def test_every_triangle_and_world_transform_retained_across_saved_parts(tmp_path):
    helper = module(); stage = scene(tmp_path)
    source = helper.render_ground(stage)[0]
    before = tuple(a.copy() for a in helper._arrays(source))
    report = helper.configure_ground_collision(stage, maximum_faces=2)
    assert len(report['collision_meshes']) == 3
    assert report['partition_comparisons'][0]['triangles'] == 6
    assert all(np.array_equal(a, b) for a, b in zip(before, helper._arrays(source)))
    stage.GetRootLayer().Save()
    from pxr import Usd
    reopened = Usd.Stage.Open(str(tmp_path / 'world.usda'))
    again = helper.configure_ground_collision(reopened, preserve_authored=True, maximum_faces=2)
    assert again['partition_comparisons'] == report['partition_comparisons']


def test_changed_part_cannot_pass_preserved_collision_check(tmp_path):
    helper = module(); stage = scene(tmp_path)
    report = helper.configure_ground_collision(stage, maximum_faces=2)
    from pxr import UsdGeom
    part = UsdGeom.Mesh(stage.GetPrimAtPath(report['collision_meshes'][1]))
    points = part.GetPointsAttr().Get(); points[0] = points[0] + (0.0, 0.0, 0.01)
    part.GetPointsAttr().Set(points)
    with pytest.raises(ValueError, match='source vertex|winding'):
        helper.configure_ground_collision(stage, preserve_authored=True, maximum_faces=2)


def test_missing_collision_range_rejected(tmp_path):
    helper = module(); stage = scene(tmp_path)
    report = helper.configure_ground_collision(stage, maximum_faces=2)
    helper.render_ground(stage)[0].GetRelationship(helper.PARTS).SetTargets(report['collision_meshes'][:2])
    with pytest.raises(ValueError, match='every final ground triangle'):
        helper.verify_exact_parts(stage)


def test_guide_exemption_requires_reciprocal_ownership_and_orientation(tmp_path):
    helper = module(); stage = scene(tmp_path)
    report = helper.configure_ground_collision(stage, maximum_faces=2)
    from isaacmin.assets.collision_scope import collision_only_paths
    from pxr import UsdGeom
    assert collision_only_paths(stage) == set(report['collision_meshes'])
    part = stage.GetPrimAtPath(report['collision_meshes'][0])
    UsdGeom.Mesh(part).CreateOrientationAttr('leftHanded')
    with pytest.raises(ValueError, match='orientation'):
        helper.verify_exact_parts(stage)
    with pytest.raises(ValueError, match='semantics'):
        collision_only_paths(stage)
    UsdGeom.Mesh(part).CreateOrientationAttr('rightHanded')
    part.GetRelationship('isaacmin:renderSource').SetTargets([])
    with pytest.raises(ValueError, match='reciprocal'):
        collision_only_paths(stage)


def test_unowned_hidden_mesh_cannot_escape_material_inspection(tmp_path):
    helper = module(); stage = scene(tmp_path)
    from pxr import UsdGeom, Sdf
    from isaacmin.assets.collision_scope import collision_only_paths
    from isaacmin.assets.usd_inspection import inspect_usd
    helper.configure_ground_collision(stage, maximum_faces=2)
    hidden = UsdGeom.Mesh.Define(stage, '/UnownedHidden')
    hidden.CreatePointsAttr([(0,0,0),(1,0,0),(0,1,0)])
    hidden.CreateFaceVertexCountsAttr([3]); hidden.CreateFaceVertexIndicesAttr([0,1,2])
    hidden.CreateVisibilityAttr('invisible'); hidden.CreatePurposeAttr('guide')
    stage.GetRootLayer().Save()
    report = inspect_usd(tmp_path/'world.usda', tmp_path/'inspection.json')
    assert any(e.get('mesh') == '/UnownedHidden' and e['category'] == 'unbound_mesh_faces' for e in report['errors'])
    assert not any(e.get('mesh','').startswith('/IsaacMinCollision/') for e in report['errors'])
    hidden.GetPrim().CreateAttribute(helper.PART,Sdf.ValueTypeNames.Bool).Set(True)
    with pytest.raises(ValueError, match='Unowned'):
        collision_only_paths(stage)
    with pytest.raises(ValueError, match='Unowned'):
        helper.verify_exact_parts(stage)
