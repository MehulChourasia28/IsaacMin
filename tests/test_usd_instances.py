"""Native CPU USD tests; these do not establish rendered world quality."""
import json
from pathlib import Path

import pytest

from isaacmin.assembly.usd_instances import _closure, share_scene
from isaacmin.io import atomic_json, sha256_file


def scene_fixture(directory):
    Usd = pytest.importorskip('pxr.Usd')
    from pxr import Gf, Sdf, UsdGeom, UsdShade
    directory.mkdir()
    content = Usd.Stage.CreateNew(str(directory / 'content.usdc'))
    content.DefinePrim('/World', 'Xform')
    for name, color in [('Green', (0.1, 0.3, 0.04)), ('Brown', (0.2, 0.1, 0.04))]:
        material = UsdShade.Material.Define(content, '/World/Materials/' + name)
        shader = UsdShade.Shader.Define(content, str(material.GetPath()) + '/Shader')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
        shader.CreateInput('opacityThreshold', Sdf.ValueTypeNames.Float).Set(0.5)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    for i, name in enumerate(('A', 'B', 'ChangedPoint', 'ChangedBinding', 'ChangedNormal')):
        parent = UsdGeom.Xform.Define(content, '/World/' + name)
        parent.AddTranslateOp().Set((i * 2, 3, 5))
        for key, value in [('source_blend_sha256', 'a' * 64), ('source_object', 'Fern'), ('instance_id', name)]:
            parent.GetPrim().CreateAttribute('isaacmin:isaacmin_' + key, Sdf.ValueTypeNames.String, custom=True).Set(value)
        mesh = UsdGeom.Mesh.Define(content, str(parent.GetPath()) + '/Geometry')
        mesh.CreatePointsAttr([(0, 0, 0), (1.1 if name == 'ChangedPoint' else 1, 0, 0), (0, 1, 0)])
        mesh.CreateFaceVertexCountsAttr([3]); mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
        mesh.CreateSubdivisionSchemeAttr('none')
        mesh.CreateNormalsAttr([(0, 0, -1 if name == 'ChangedNormal' else 1)] * 3)
        mesh.SetNormalsInterpolation('vertex')
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, 'vertex').Set([(0, 0), (1, 0), (0, 1)])
        material = UsdShade.Material.Get(content, '/World/Materials/' + ('Brown' if name == 'ChangedBinding' else 'Green'))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    content.GetRootLayer().Save()
    root = Usd.Stage.CreateNew(str(directory / 'world.usda'))
    root.GetRootLayer().subLayerPaths = ['content.usdc']
    root.SetDefaultPrim(root.GetPrimAtPath('/World'))
    UsdGeom.SetStageUpAxis(root, 'Z'); UsdGeom.SetStageMetersPerUnit(root, 1)
    root.GetRootLayer().Save()
    scene = directory / 'world.usda'
    atomic_json(directory / 'native_dependency_closure.json', _closure(scene))
    return scene


def test_native_sharing_preserves_scene_and_separates_changed_render_properties(tmp_path):
    source = scene_fixture(tmp_path / 'source')
    original = {p.name: sha256_file(p) for p in source.parent.iterdir()}
    result = share_scene(source, tmp_path / 'shared', tmp_path / 'proof.json')
    from pxr import Usd, UsdGeom, UsdShade
    stage = Usd.Stage.Open(result['scene'])
    assert result['native_instances'] == 5
    assert result['native_prototypes'] == 4
    assert stage.GetPrimAtPath('/World/A').GetPrototype() == stage.GetPrimAtPath('/World/B').GetPrototype()
    for name in ('ChangedPoint', 'ChangedBinding', 'ChangedNormal'):
        assert stage.GetPrimAtPath('/World/A').GetPrototype() != stage.GetPrimAtPath('/World/' + name).GetPrototype()
    mesh = stage.GetPrimAtPath('/World/B/Geometry')
    assert mesh.IsInstanceProxy()
    assert UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()[0].GetPath() == '/World/Materials/Green'
    assert tuple(UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(Usd.TimeCode.Default()).ExtractTranslation()) == (2, 3, 5)
    assert {p.name: sha256_file(p) for p in source.parent.iterdir()} == original
    assert not (tmp_path / 'shared.staging').exists()
    assert json.loads((tmp_path / 'shared/native_dependency_closure.json').read_text())['status'] == 'pass'


def test_changed_input_closure_rejected_before_output(tmp_path):
    source = scene_fixture(tmp_path / 'source')
    source.write_text(source.read_text() + '\n# changed after closure\n')
    with pytest.raises(Exception, match='closure|current root'):
        share_scene(source, tmp_path / 'shared', tmp_path / 'proof.json')
    assert not (tmp_path / 'shared').exists()
    assert not (tmp_path / 'shared.staging').exists()


def test_existing_candidate_never_overwritten(tmp_path):
    source = scene_fixture(tmp_path / 'source')
    (tmp_path / 'shared').mkdir()
    marker = tmp_path / 'shared/preserve.txt'; marker.write_text('original')
    with pytest.raises(ValueError, match='fresh destination'):
        share_scene(source, tmp_path / 'shared', tmp_path / 'proof.json')
    assert marker.read_text() == 'original'
