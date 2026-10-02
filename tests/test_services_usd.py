"""Native USD structural and independent contact schema fixtures, never Q06 proof."""
import pytest
from isaacmin.assets.usd_inspection import inspect_usd,validate_contact_report
from isaacmin.assets.network import ServiceError


def test_contact_report_requires_every_exported_instance_and_measured_offsets():
    report={'scene_sha256':'scene','ground_sha256':'ground','measurement':'independent_exported_usd_anchors_against_final_ground',
            'instances':[{'instance_id':'a','usd_paths':['/a'],'anchors_measured':4,'max_abs_offset_m':.01,'unsupported_anchors':0}]}
    args={'expected_instance_ids':['a'],'scene_sha256':'scene','ground_sha256':'ground'}
    assert validate_contact_report(report,**args)['status']=='pass'
    report['instances'][0]['unsupported_anchors']=1
    assert validate_contact_report(report,**args)['status']=='fail'
    args['expected_instance_ids']=['a','b']
    with pytest.raises(ServiceError,match='every exact'):validate_contact_report(report,**args)


def test_native_inspector_detects_missing_cutout_threshold_and_uv(tmp_path):
    pytest.importorskip('pxr.Usd')
    from pxr import Usd,UsdShade,UsdGeom,Sdf,Gf
    from PIL import Image
    image=tmp_path/'mask.png';Image.new('RGB',(4,4),'white').save(image)
    path=tmp_path/'fixture.usda';stage=Usd.Stage.CreateNew(str(path))
    mesh=UsdGeom.Mesh.Define(stage,'/Leaf');mesh.GetPointsAttr().Set([Gf.Vec3f(0,0,0),Gf.Vec3f(1,0,0),Gf.Vec3f(0,1,0)])
    mesh.GetFaceVertexCountsAttr().Set([3]);mesh.GetFaceVertexIndicesAttr().Set([0,1,2])
    mat=UsdShade.Material.Define(stage,'/fern_02');shader=UsdShade.Shader.Define(stage,'/fern_02/Surface');shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateOutput('surface',Sdf.ValueTypeNames.Token);mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    tex=UsdShade.Shader.Define(stage,'/fern_02/Mask');tex.CreateIdAttr('UsdUVTexture');tex.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('./mask.png'));tex.CreateOutput('r',Sdf.ValueTypeNames.Float)
    shader.CreateInput('opacity',Sdf.ValueTypeNames.Float).ConnectToSource(tex.ConnectableAPI(),'r')
    uv=UsdShade.Shader.Define(stage,'/fern_02/UV');uv.CreateIdAttr('UsdPrimvarReader_float2');uv.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st');uv.CreateOutput('result',Sdf.ValueTypeNames.Float2)
    tex.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(),'result')
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat);stage.GetRootLayer().Save()
    result=inspect_usd(path,tmp_path/'report.json',expected_cutout_materials=['fern_02'])
    assert {e['category'] for e in result['errors']}>={'missing_cutout_threshold','missing_mesh_uv'}
    shader.CreateInput('opacityThreshold',Sdf.ValueTypeNames.Float).Set(.5)
    primvar=UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.faceVarying);primvar.Set([Gf.Vec2f(0,0),Gf.Vec2f(1,0),Gf.Vec2f(0,1)])
    stage.GetRootLayer().Save();result=inspect_usd(path,tmp_path/'fixed.json',expected_cutout_materials=['fern_02'])
    assert result['status']=='pass' and result['Q06']=='not_run'


def physics_material_scene(tmp_path):
    pytest.importorskip('pxr.Usd')
    from pxr import Usd,UsdGeom,UsdShade,UsdPhysics,Sdf,Gf
    path=tmp_path/'physics_materials.usda';stage=Usd.Stage.CreateNew(str(path))
    mesh=UsdGeom.Mesh.Define(stage,'/Ground')
    mesh.GetPointsAttr().Set([Gf.Vec3f(0,0,0),Gf.Vec3f(1,0,0),Gf.Vec3f(0,1,0)])
    mesh.GetFaceVertexCountsAttr().Set([3]);mesh.GetFaceVertexIndicesAttr().Set([0,1,2])
    visual=UsdShade.Material.Define(stage,'/Visual')
    shader=UsdShade.Shader.Define(stage,'/Visual/Surface');shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateOutput('surface',Sdf.ValueTypeNames.Token)
    visual.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    binding=UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim());binding.Bind(visual)
    physics=UsdShade.Material.Define(stage,'/ArbitraryName')
    UsdPhysics.MaterialAPI.Apply(physics.GetPrim()).CreateStaticFrictionAttr(.8)
    binding.Bind(physics,materialPurpose='physics')
    return path,stage,binding,physics


def test_physics_only_material_does_not_require_surface_shader(tmp_path):
    path,stage,binding,physics=physics_material_scene(tmp_path)
    stage.GetRootLayer().Save();result=inspect_usd(path,tmp_path/'report.json')
    assert result['status']=='pass' and result['Q06']=='not_run'
    assert result['physics_materials'][0]['path']=='/ArbitraryName'
    assert result['materials'][0]['path']=='/Visual'


@pytest.mark.parametrize('purpose',['','full','preview'])
def test_physics_material_cannot_replace_visual_binding(tmp_path,purpose):
    path,stage,binding,physics=physics_material_scene(tmp_path)
    binding.Bind(physics,materialPurpose=purpose)
    stage.GetRootLayer().Save();result=inspect_usd(path,tmp_path/'report.json')
    assert result['status']=='fail'
    assert any(e['category']=='physics_material_used_for_rendering' and e['material_purpose']==purpose
               for e in result['errors'])


def test_physics_api_does_not_exempt_broken_visual_surface(tmp_path):
    path,stage,binding,physics=physics_material_scene(tmp_path)
    physics.CreateSurfaceOutput('mdl').GetAttr().SetConnections(['/Missing.outputs:surface'])
    stage.GetRootLayer().Save();result=inspect_usd(path,tmp_path/'report.json')
    assert result['status']=='fail'
    assert {'category':'missing_surface_shader','material':'/ArbitraryName'} in result['errors']
