"""CPU USD/material contract fixtures; no renderer or appearance claims."""
from pathlib import Path
import hashlib,json
import numpy as np
import pytest
from PIL import Image

from isaacmin.assembly.material_mdl import validate_weights,prepare_library,bind_shared_material,digest,REQUIRED,WEIGHT_NAMES
from isaacmin.assets.usd_inspection import inspect_usd

ROOT=Path(__file__).resolve().parents[1]


def candidate(tmp_path):
    image=tmp_path/'fixture_only.png';Image.new('RGB',(4096,4096),(128,128,255)).save(image)
    materials=[]
    for name in REQUIRED:
        materials.append({'asset_id':name,'repeat_m':2.,'licence':'CC0-1.0','source_page':'fixture-only-not-provider-evidence',
                         **{role:str(image) for role in ('base_color','roughness','normal')},
                         'original_channel_proof':{role:{'sha256':digest(image),'width':4096,'height':4096} for role in ('base_color','roughness','normal')}})
    p=tmp_path/'candidate.json';p.write_text(json.dumps({'materials':materials}));return p


def make_stage(tmp_path,weights=None):
    pytest.importorskip('pxr.Usd')
    from pxr import Usd,UsdGeom,Sdf,Gf
    path=tmp_path/'world.usda';stage=Usd.Stage.CreateNew(str(path));UsdGeom.SetStageMetersPerUnit(stage,1);UsdGeom.SetStageUpAxis(stage,'Z')
    mesh=UsdGeom.Mesh.Define(stage,'/World/Terrain_FinalGround');points=np.array([[0,0,0],[1,0,0],[0,1,0]],dtype=np.float32)
    mesh.CreatePointsAttr([Gf.Vec3f(*map(float,v)) for v in points]);mesh.CreateFaceVertexCountsAttr([3]);mesh.CreateFaceVertexIndicesAttr([0,1,2]);mesh.CreateSubdivisionSchemeAttr('none')
    w=np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1]],dtype=np.float32) if weights is None else weights
    pv=UsdGeom.PrimvarsAPI(mesh);pv.CreatePrimvar(WEIGHT_NAMES[0],Sdf.ValueTypeNames.Float3Array,'vertex').Set([Gf.Vec3f(*map(float,row[:3])) for row in w])
    pv.CreatePrimvar(WEIGHT_NAMES[1],Sdf.ValueTypeNames.FloatArray,'vertex').Set(w[:,3].tolist());stage.GetRootLayer().Save()
    proof={'material_order':list(REQUIRED),'source_weights':validate_weights(w),'world_points_float32_sha256':hashlib.sha256(points.tobytes()).hexdigest()}
    q=tmp_path/'source_weights.json';q.write_text(json.dumps(proof));return path,q,stage,mesh


@pytest.mark.parametrize('bad',[np.zeros((3,4)),np.full((2,4),np.nan),np.array([[-.01,.51,.5,0]]),np.array([[1.,1.,0,0]]),np.ones((2,3))])
def test_shared_weights_never_fill_missing_or_invalid_rows(bad):
    with pytest.raises(ValueError):validate_weights(bad)


def test_original_source_density_rejects_forged_dimensions(tmp_path):
    p=candidate(tmp_path);j=json.loads(p.read_text());Image.new('RGB',(8,8),'white').save(j['materials'][0]['base_color'])
    for m in j['materials']:
        for proof in m['original_channel_proof'].values():proof['sha256']=digest(m['base_color'])
    p.write_text(json.dumps(j))
    with pytest.raises(ValueError,match='verified original'):prepare_library(ROOT,p,tmp_path/'library')


def test_native_mdl_context_checks_all_weights_and_actual_shader_implementation(tmp_path):
    scene,weights,stage,mesh=make_stage(tmp_path);p=candidate(tmp_path);prepare_library(ROOT,p,tmp_path/'library')
    manifest=tmp_path/'library/material_manifest.json';before=np.asarray(mesh.GetPointsAttr().Get()).copy()
    result=bind_shared_material(scene,manifest,weights)
    assert result['qualification']=='not_run' and np.array_equal(before,np.asarray(mesh.GetPointsAttr().Get()))
    report=inspect_usd(scene,tmp_path/'inspect.json')
    assert report['status']=='pass' and report['Q06']=='not_run'
    mdl=next(m['mdl'] for m in report['materials'] if 'mdl' in m)
    assert mdl['source_families']==list(REQUIRED) and mdl['required_weight_primvars']==list(WEIGHT_NAMES)
    module=tmp_path/'library/IsaacMinScans.mdl';module.write_text(module.read_text().replace('float3(4.0)','float3(2.0)'))
    j=json.loads(manifest.read_text());j['module_sha256']=digest(module);manifest.write_text(json.dumps(j))
    report=inspect_usd(scene,tmp_path/'tampered.json')
    assert report['status']=='fail'
    assert any(e['category']=='invalid_mdl_material_contract' and 'pinned template' in e['reason'] for e in report['errors'])


def test_changed_exported_weight_fails_before_binding(tmp_path):
    scene,weights,stage,mesh=make_stage(tmp_path);prepare_library(ROOT,candidate(tmp_path),tmp_path/'library')
    j=json.loads(weights.read_text());j['source_weights']['sha256']='0'*64;weights.write_text(json.dumps(j));before=digest(scene)
    with pytest.raises(ValueError,match='source weights differ'):bind_shared_material(scene,tmp_path/'library/material_manifest.json',weights)
    assert digest(scene)==before


def test_readonly_inspector_rechecks_bound_world_geometry_and_source_weights(tmp_path):
    scene,weights,stage,mesh=make_stage(tmp_path);prepare_library(ROOT,candidate(tmp_path),tmp_path/'library')
    from pxr import Gf
    bind_shared_material(scene,tmp_path/'library/material_manifest.json',weights)
    points=list(mesh.GetPointsAttr().Get());points[1]=Gf.Vec3f(1.1,0,0)
    mesh.GetPointsAttr().Set(points);stage.GetRootLayer().Save()
    report=inspect_usd(scene,tmp_path/'changed_geometry.json')
    assert report['status']=='fail'
    assert any(e['category']=='invalid_mdl_weight_partition' for e in report['errors'])


def test_runtime_mdl_is_explicit_and_hash_bound_not_silently_unresolved(tmp_path):
    scene,weights,stage,mesh=make_stage(tmp_path)
    from pxr import Sdf, UsdShade
    material=UsdShade.Material.Define(stage,'/World/RuntimeMaterial')
    shader=UsdShade.Shader.Define(stage,'/World/RuntimeMaterial/Shader')
    shader.SetSourceAsset(Sdf.AssetPath('FixtureBuiltin.mdl'),'mdl')
    shader.SetSourceAssetSubIdentifier('FixtureBuiltin','mdl')
    shader.CreateOutput('out',Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput('mdl').ConnectToSource(shader.ConnectableAPI(),'out')
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material);stage.GetRootLayer().Save()
    runtime=tmp_path/'explicit_runtime_file.mdl';runtime.write_text('mdl 1.7; // unit fixture only\n')
    assert inspect_usd(scene,tmp_path/'undeclared.json')['status']=='fail'
    declared=[{'module':'FixtureBuiltin.mdl','path':str(runtime),'sha256':digest(runtime)}]
    report=inspect_usd(scene,tmp_path/'declared.json',runtime_modules=declared)
    assert report['status']=='pass' and report['Q06']=='not_run'
    runtime.write_text('changed runtime bytes')
    assert inspect_usd(scene,tmp_path/'changed_runtime.json',runtime_modules=declared)['status']=='fail'
