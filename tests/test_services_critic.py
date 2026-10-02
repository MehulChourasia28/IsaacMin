"""Algorithmic critic checks; these fixtures cannot establish real Isaac quality."""
import importlib.util
import json
from pathlib import Path
import pytest

from isaacmin.adapters.critic import evaluate_case,validate_finding,decode_finding_compat
from isaacmin.assets.network import ServiceError


def finding(category='floating_vegetation',bbox=None):
    return {'defects':[{'category':category,'severity':'severe','confidence':.9,
             'bbox':bbox or [.1,.2,.4,.6],'description':'Visible unsupported plant base'}],
             'uncertainty':'Contact outside image cannot be assessed'}


def test_critic_does_not_pass_wrong_category_or_location():
    case={'expected_severe':[{'category':'floating_vegetation','bbox':[.1,.2,.4,.6]}]}
    assert evaluate_case(case,finding())['status']=='pass'
    assert evaluate_case(case,finding('missing_material'))['status']=='fail'
    assert evaluate_case(case,finding(bbox=[.7,.7,.9,.9]))['status']=='fail'
    assert evaluate_case(case,{'defects':[],'uncertainty':'Cannot see'})['status']=='fail'


def test_severe_false_positive_on_known_clean_control_fails():
    result=evaluate_case({'known_absent_categories':['floating_vegetation']},finding())
    assert result['status']=='fail' and result['false_positives']
    assert not result['uninjected_control_is_clean']


def test_critic_schema_rejects_scores_bad_boxes_and_nonfinite_confidence():
    assert validate_finding(json.dumps(finding()))==finding()
    for value in ({'realism_score':10},finding(bbox=[0,0,1.1,1])):
        with pytest.raises(ServiceError):validate_finding(json.dumps(value))
    value=finding();value['defects'][0]['confidence']=float('nan')
    with pytest.raises(ServiceError):validate_finding(json.dumps(value))


def test_compat_decoder_preserves_all_findings_and_declares_duplicate_transport():
    original=finding();wire=json.dumps(original)
    result=decode_finding_compat(wire+'\n\n'+wire)
    assert result['finding']==original
    assert result['json_value_count']==2 and result['duplicate_count']==1
    assert result['transport_normalization']=='deduplicate_identical_consecutive_json_values'
    assert result['wire_schema_status']=='fail' and result['provider_schema_compliance'] is False


def test_compat_decoder_rejects_conflicts_prose_missing_fields_and_unit_changes():
    valid=json.dumps(finding())
    other=finding();other['defects'][0]['severity']='minor'
    for wire in [valid+'\n'+json.dumps(other),'Here is JSON: '+valid,valid+' trailing prose',
                 valid+'\n'+json.dumps({'defects':[]}),valid+'\n'+json.dumps(finding(bbox=[0,200,400,600]))]:
        with pytest.raises(ServiceError):decode_finding_compat(wire)


def test_native_usd_mutations_are_isolated_and_change_real_geometry(tmp_path):
    pytest.importorskip('pxr.Usd')
    from pxr import Usd,UsdGeom,UsdShade,Gf,Sdf
    script=Path(__file__).resolve().parents[1]/'isaac_scripts/calibration_mutations.py'
    spec=importlib.util.spec_from_file_location('calibration_mutations',script)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    source=tmp_path/'fixture.usda';stage=Usd.Stage.CreateNew(str(source))
    mesh=UsdGeom.Mesh.Define(stage,'/Terrain_FinalGround')
    points=[Gf.Vec3f(x*.5,y*.5,.03*x) for y in range(25) for x in range(-12,13)]
    faces=[]
    for y in range(24):
        for x in range(24):
            a=y*25+x;faces.extend([a,a+1,a+26,a,a+26,a+25])
    mesh.GetPointsAttr().Set(points);mesh.GetFaceVertexCountsAttr().Set([3]*(len(faces)//3));mesh.GetFaceVertexIndicesAttr().Set(faces)
    ground_material=UsdShade.Material.Define(stage,'/GroundMaterial')
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(ground_material)
    pv=UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.faceVarying)
    pv.Set([Gf.Vec2f(0,0)]*len(faces))
    subset=UsdGeom.Subset.Define(stage,'/Terrain_FinalGround/MaterialSubset');subset.CreateElementTypeAttr('face');subset.CreateIndicesAttr(list(range(len(faces)//3)))
    plant=UsdGeom.Mesh.Define(stage,'/GrassMedium')
    plant.GetPointsAttr().Set([Gf.Vec3f(0,3,0),Gf.Vec3f(.1,3,1),Gf.Vec3f(-.1,3,1)])
    plant.GetFaceVertexCountsAttr().Set([3]);plant.GetFaceVertexIndicesAttr().Set([0,1,2])
    shader=UsdShade.Shader.Define(stage,'/LeafMaterial/Shader');shader.CreateIdAttr('UsdPreviewSurface')
    mask=UsdShade.Shader.Define(stage,'/LeafMaterial/Mask');mask.CreateIdAttr('UsdUVTexture');mask.CreateOutput('r',Sdf.ValueTypeNames.Float)
    shader.CreateInput('opacity',Sdf.ValueTypeNames.Float).ConnectToSource(mask.ConnectableAPI(),'r')
    shader.CreateInput('opacityThreshold',Sdf.ValueTypeNames.Float).Set(.5)
    stage.GetRootLayer().Save();original=source.read_bytes()
    pose={'position':[0,0,.6],'look_at':[0,8,.4]}
    for category in ('conspicuous_seam','retained_voxel_steps','floating_vegetation','missing_material','broken_leaf_opacity'):
        variant=tmp_path/(category+'.usda');layer=Sdf.Layer.CreateNew(str(variant));layer.subLayerPaths=[str(source)];layer.Save()
        working=Usd.Stage.Open(str(variant));mutation=module.inject(working,category,pose)
        assert mutation['changed_elements']>0
        working.GetRootLayer().Save()
        assert source.read_bytes()==original
        if category=='broken_leaf_opacity':
            changed=UsdShade.Shader.Get(working,'/LeafMaterial/Shader')
            assert changed.GetInput('opacity').HasConnectedSource()
            assert changed.GetInput('opacityThreshold').Get()==0
        if category=='retained_voxel_steps':
            changed=UsdGeom.Mesh.Get(working,'/IsaacMinCalibrationFault/RetainedVoxelTerraces')
            assert len(changed.GetFaceVertexCountsAttr().Get())==20
        if category=='conspicuous_seam':
            changed=UsdGeom.Mesh.Get(working,'/Terrain_FinalGround')
            assert sum(changed.GetFaceVertexCountsAttr().Get())==len(changed.GetFaceVertexIndicesAttr().Get())
            assert len(UsdGeom.PrimvarsAPI(changed).GetPrimvar('st').Get())==len(changed.GetFaceVertexIndicesAttr().Get())
            assert max(UsdGeom.Subset.Get(working,'/Terrain_FinalGround/MaterialSubset').GetIndicesAttr().Get())<len(changed.GetFaceVertexCountsAttr().Get())
