"""Actual USD fixture geometry verifies the collector, never user-map quality."""
import json
from pathlib import Path
import numpy as np
import pytest

from isaacmin.assets.network import digest


def test_post_usd_collector_remeasures_root_and_transform_independently(tmp_path):
    pytest.importorskip('pxr.Usd')
    from pxr import Usd,UsdGeom,Sdf,Gf
    from isaacmin.assets.ecology_validation import collect_ecology
    ir=tmp_path/'ir';ir.mkdir();(ir/'terrain_volume').mkdir()
    (ir/'world_ir.json').write_text(json.dumps({'coordinate_frame':{'source_origin_xyz':[0,0,0]}}))
    np.savez(ir/'terrain_surface.npz',height=np.zeros((4,4)),validity=np.ones((4,4),bool),
             biome=np.full((4,4),'minecraft:plains'),water_height=np.zeros((4,4)),water_validity=np.zeros((4,4),bool),
             substrate_id=np.zeros((4,4),int),block_names=np.array(['minecraft:grass_block']),min_xz=np.array([0,0]))
    native=tmp_path/'source.blend';native.write_bytes(b'identity-only fixture, not a Blender qualification')
    state=tmp_path/'state';state.mkdir()
    normalized={'assets':[{'asset_id':'grass_medium_01','output_blend':str(native),'output_sha256':digest(native),
                           'objects':[{'object_name':'Grass','contact_anchors_local_m':[[0,0,0],[.05,0,0]],'dimensions_m':[.1,.1,.5]}]}]}
    (state/'normalized_assets.json').write_text(json.dumps(normalized))
    path=tmp_path/'scene.usda';stage=Usd.Stage.CreateNew(str(path));UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z);ground=UsdGeom.Mesh.Define(stage,'/Terrain_FinalGround')
    ground.GetPointsAttr().Set([Gf.Vec3f(0,-4,0),Gf.Vec3f(4,-4,0),Gf.Vec3f(4,0,0),Gf.Vec3f(0,0,0)])
    ground.GetFaceVertexCountsAttr().Set([3,3]);ground.GetFaceVertexIndicesAttr().Set([0,1,2,0,2,3])
    requests=[];ops=[]
    for i,x in enumerate((.5,2.5)):
        prim=UsdGeom.Xform.Define(stage,f'/Plant{i}').GetPrim();op=UsdGeom.Xformable(prim).AddTranslateOp();op.Set((x,-.5,0));ops.append(op)
        for key,value in [('instance_id',str(i)),('source_blend_sha256',digest(native)),('source_object','Grass')]:
            prim.CreateAttribute('isaacmin:isaacmin_'+key,Sdf.ValueTypeNames.String,custom=True).Set(value)
        blade=UsdGeom.Mesh.Define(stage,f'/Plant{i}/Mesh');blade.GetPointsAttr().Set([Gf.Vec3f(0,0,0),Gf.Vec3f(.05,0,0),Gf.Vec3f(0,0,.5)])
        blade.GetFaceVertexCountsAttr().Set([3]);blade.GetFaceVertexIndicesAttr().Set([0,1,2])
        matrix=np.eye(4);matrix[:3,3]=[x,-.5,0]
        requests.append({'id':str(i),'world_transform':matrix.tolist()})
    stage.GetRootLayer().Save()
    closure=tmp_path/'native_dependency_closure.json'
    def bind_scene():closure.write_text(json.dumps({'status':'pass','root_sha256':digest(path),'files':[{'path':path.name,'sha256':digest(path)}]}))
    bind_scene()
    ecology=tmp_path/'ecology.json';ecology.write_text(json.dumps({'exporter_assets':requests,'instances':[{'instance_id':str(i),'guild':'meadow_grass'} for i in range(2)],'season':'late_spring'}))
    route=tmp_path/'route.json';route.write_text(json.dumps({'points_world_xyz':[[0,-3,0],[4,-3,0]],'corridor_width_m':.4}))
    result=collect_ecology(tmp_path,path,ecology,ir,tmp_path/'measured',route_path=route)
    assert result['status']=='pass' and result['contact_status']=='pass' and result['instances']==2
    measured=json.loads((tmp_path/'measured/contact_report.json').read_text())
    assert all(e['max_abs_offset_m']==0 for e in measured['instances'])
    ops[1].Set((2.5,-.5,.1));stage.GetRootLayer().Save()
    bind_scene()
    result=collect_ecology(tmp_path,path,ecology,ir,tmp_path/'changed',route_path=route)
    assert result['status']=='fail' and result['contact_status']=='fail'
    assert result['hard_rule_violations']['exported_transform_differs_from_frozen_placement']==1
    assert result['hard_rule_violations']['native_root_or_declared_burial_offset']==1
