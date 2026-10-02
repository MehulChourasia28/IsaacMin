"""Local native-USD/math fixtures test evidence handling, never simulate a pass."""
import json
from pathlib import Path
import numpy as np
import pytest
from isaacmin.assets.network import digest,ServiceError
from isaacmin.assets.view_coverage import collect_view_coverage,validate_view_coverage


@pytest.mark.parametrize('with_weights',[False,True])
@pytest.mark.parametrize('native_support',[False,True])
def test_native_material_rays_and_segmentation_override_planned_tags(tmp_path,with_weights,native_support):
    pytest.importorskip('pxr.Usd')
    from pxr import Usd,UsdGeom,UsdShade,Gf,Sdf
    if native_support:
        root=Path(__file__).resolve().parents[1]
        if not (root/'.tools/geometry_validation/obj_compare_build_manifest.json').is_file():
            pytest.skip('Pinned native geometry comparator unavailable')
        from test_source_native_integration import write_native_fixture
        tmp_path=tmp_path/'native_case'
        write_native_fixture(tmp_path,[[-3,-3,0],[3,-3,0],[3,3,0],[-3,3,0]],[[0,1,2],[0,2,3]])
    scene=tmp_path/'scene.usda';stage=Usd.Stage.CreateNew(str(scene));UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(stage,1)
    mesh=UsdGeom.Mesh.Define(stage,'/Terrain_FinalGround')
    mesh.GetPointsAttr().Set([Gf.Vec3f(-3,-3,0),Gf.Vec3f(3,-3,0),Gf.Vec3f(3,3,0),Gf.Vec3f(-3,3,0)])
    mesh.GetFaceVertexCountsAttr().Set([3,3]);mesh.GetFaceVertexIndicesAttr().Set([0,1,2,0,2,3])
    material=UsdShade.Material.Define(stage,'/Materials/ActualSoil');UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material)
    if with_weights:
        var=UsdGeom.PrimvarsAPI(mesh)
        var.CreatePrimvar('IsaacMinMaterialWeights012',Sdf.ValueTypeNames.Float3Array,UsdGeom.Tokens.vertex).Set([Gf.Vec3f(0,.09,.8)]*4)
        var.CreatePrimvar('IsaacMinMaterialWeight3',Sdf.ValueTypeNames.FloatArray,UsdGeom.Tokens.vertex).Set([.11]*4)
        mesh.GetPrim().CreateAttribute('isaacmin:isaacmin_material_weight_order',Sdf.ValueTypeNames.String,custom=True).Set(json.dumps(['brown_mud_dry','rock_face_03','leafy_grass','forest_ground_04']))
    asset=UsdGeom.Xform.Define(stage,'/Plant').GetPrim();asset.CreateAttribute('isaacmin:isaacmin_instance_id',Sdf.ValueTypeNames.String).Set('plant1')
    UsdGeom.Mesh.Define(stage,'/Plant/Leaf');stage.GetRootLayer().Save()
    closure=tmp_path/'native_dependency_closure.json';closure.write_text(json.dumps({'status':'pass','root_sha256':digest(scene),'files':[{'path':scene.name,'sha256':digest(scene)}]}))
    ecology=tmp_path/'ecology.json';ecology.write_text(json.dumps({'exporter_assets':[{'id':'plant1','asset_id':'actual_grass'}]}))
    # Controlled numerical sensor arrays, explicitly local unit fixtures, not renderer evidence.
    ids=np.ones((200,200),np.uint32);ids[40:150,40:100]=2;depth=np.ones((200,200),np.float32)
    np.save(tmp_path/'depth.npy',depth);np.save(tmp_path/'seg.npy',ids);(tmp_path/'rgb.png').write_bytes(b'test only')
    matrix=np.eye(4);matrix[3,2]=1
    frame={'frame':7,'pose':{'kind':'static','features':['invented_cave'],'asset_families':['invented_tree']},
           'rgb':'rgb.png','rgb_sha256':digest(tmp_path/'rgb.png'),'depth':'depth.npy','depth_sha256':digest(tmp_path/'depth.npy'),
           'instance_segmentation':'seg.npy','instance_segmentation_sha256':digest(tmp_path/'seg.npy'),
           'instance_id_to_prim_path':{'1':'/Terrain_FinalGround','2':'/Plant/Leaf'},'depth_semantics':'axial_metres',
           'fx_pixels':100,'fy_pixels':100,'cx_pixels':100,'cy_pixels':100,'camera_world_matrix_row_vectors':matrix.tolist()}
    capture=tmp_path/'capture.json';capture.write_text(json.dumps({'renderer':'RayTracedLighting','scope':'metadata_unit_fixture',
          'scene_sha256':digest(scene),'native_dependency_closure_sha256':digest(closure),'frames':[frame]}))
    output=tmp_path/'visibility.json';report=collect_view_coverage(tmp_path,scene,[capture],output,ecology_manifest=ecology)
    measured=report['frames'][0]
    assert measured['assets'][0]['family']=='actual_grass' and measured['assets'][0]['close_view']
    assert measured['materials'][0]['family']=='ActualSoil' and measured['materials'][0]['close_view']
    assert measured['features']==[] and measured['ground_sensor_agree']==measured['ground_rays']
    if native_support:
        assert report['native_ground_queries'] and report['native_USD_ground_identity']
    record={'capture_result':{'sha256':digest(capture)},'frame':7,'rgb':{'sha256':digest(tmp_path/'rgb.png')},
            'depth':{'sha256':digest(tmp_path/'depth.npy')},'scene_dependencies_sha256':digest(closure)}
    _,joined=validate_view_coverage(output,[record]);value=next(iter(joined.values()))
    expected=['ActualSoil','forest_ground_04','leafy_grass'] if with_weights else ['ActualSoil']
    assert value['features']==[] and value['asset_families']==['actual_grass'] and value['material_families']==expected
    assert report['used_material_families']==expected
    assert 'rock_face_03' not in value['material_families']  #9percent is an ingredient, insufficient coverage.
    (tmp_path/'seg.npy').write_bytes(b'changed')
    with pytest.raises(ServiceError,match='changed'):validate_view_coverage(output,[record])
