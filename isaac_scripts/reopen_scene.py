"""Clean native Isaac process reopening the delivered physics layer and assets."""
import argparse
import hashlib
import json
import pathlib
import re
import sys

parser=argparse.ArgumentParser()
parser.add_argument('--scene',required=True)
parser.add_argument('--output',required=True)
args,_=parser.parse_known_args()
scene=pathlib.Path(args.scene).resolve();output=pathlib.Path(args.output).resolve()
output.mkdir(parents=True,exist_ok=True)
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'renderer':'RayTracedLighting','disable_viewport_updates':True,
                   'extra_args':[f'--/log/file={output}/reopen_kit.log']})
try:
    import omni.usd
    import carb.tokens
    from pxr import Usd,UsdUtils,Sdf,UsdGeom,UsdPhysics,UsdShade
    if not omni.usd.get_context().open_stage(str(scene)):
        raise RuntimeError('Clean native process could not open delivered physics stage')
    for _ in range(30):app.update()
    stage=omni.usd.get_context().get_stage()
    layers,assets,unresolved=UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(scene)))
    unresolved=list(unresolved)
    def sha(path):
        with path.open('rb') as stream:
            return hashlib.file_digest(stream,'sha256').hexdigest()
    files=[]
    for filename in sorted(set([str(l.realPath) for l in layers]+list(map(str,assets)))):
        path=pathlib.Path(filename)
        if path.is_file():files.append({'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size})
        else:unresolved.append(filename)
    kit=pathlib.Path(carb.tokens.get_tokens_interface().resolve('${kit}'))
    module_roots=[kit/'mdl/core/Base',kit/'mdl/core/mdl',kit/'mdl/core',kit/'mdl']
    native_modules=[];resolved_builtin=[];pending=[];direct_modules=[]
    for prim in stage.Traverse():
        if prim.IsA(UsdShade.Shader):
            value=UsdShade.Shader(prim).GetSourceAsset('mdl')
            if value and value.path:
                name=value.path
                path=next((p/name for p in module_roots if (p/name).is_file()),None)
                if path:
                    pending.append(path);resolved_builtin.append(name)
                    direct_modules.append({'asset':name,'resolved_path':str(path.resolve()),'sha256':sha(path)})
    visited=set();standard_modules=set();missing_module_imports=[]
    mdl_standard={'anno','base','limits','state','tex','math','df','debug','nvidia','scene'}
    while pending:
        path=pending.pop().resolve()
        if path in visited:continue
        visited.add(path)
        native_modules.append({'path':str(path),'sha256':sha(path),'bytes':path.stat().st_size})
        for imported in re.findall(r'\bimport\s+::([A-Za-z_][A-Za-z_0-9:]*)',path.read_text()):
            parts=imported.rstrip(':').split('::')
            if parts[0] in mdl_standard:
                standard_modules.add(parts[0]);continue
            relative=pathlib.Path(*parts).with_suffix('.mdl')
            found=next((p/relative for p in module_roots if (p/relative).is_file()),None)
            if found:pending.append(found)
            else:missing_module_imports.append(imported)
    unresolved=sorted(set(str(p) for p in unresolved if str(p) not in resolved_builtin))
    from ground_collision import ground_colliders,verify_exact_parts
    exact_parts=verify_exact_parts(stage)
    collider_paths=set(ground_colliders(stage))
    ground=[];diagnostics=[]
    for prim in stage.Traverse():
        if 'IsaacMinContacts' in str(prim.GetPath()):diagnostics.append(str(prim.GetPath()))
        if str(prim.GetPath()) in collider_paths:
            collision=UsdPhysics.CollisionAPI(prim)
            ground.append({'path':str(prim.GetPath()),'collision_api':bool(collision),
                'enabled':bool(collision.GetCollisionEnabledAttr().Get()) if collision else False,
                'approximation':UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get()})
    version=kit.parent/'VERSION'
    if not version.is_file() or not version.read_text().strip():
        raise RuntimeError('Cannot establish pinned Isaac runtime VERSION')
    record={'status':'pass' if ground and all(p['enabled'] and p['approximation']=='none' for p in ground)
            and not unresolved and not diagnostics and not missing_module_imports else 'fail',
            'scene':str(scene),'scene_sha256':sha(scene),'native_fresh_process':True,
            'root_layer_reopened':str(stage.GetRootLayer().realPath),
            'assets':files,'unresolved_assets':unresolved,'ground_colliders':ground,
            'exact_ground_parts':exact_parts,
            'diagnostic_bodies':diagnostics,'native_runtime_modules':native_modules,
            'resolved_builtin_asset_names':resolved_builtin,
            'mdl_standard_library_modules':sorted(standard_modules),
            'unresolved_mdl_imports':missing_module_imports,
            'native_isaac_version':version.read_text().strip() if version.is_file() else 'not_available',
            'portability':'Scene textures are bundled; built-in MDL modules require this pinned Isaac runtime.',
            'appearance_qualification':'not_run'}
    (output/'reopen_result.json').write_text(json.dumps(record,indent=2)+'\n')
    for module in direct_modules:module['runtime_version']=record['native_isaac_version']
    runtime={'status':record['status'],'runtime_version':record['native_isaac_version'],
             'runtime_build':record['native_isaac_version'],'modules':direct_modules,
             'transitive_module_files':native_modules,
             'native_resolution_evidence':{'fresh_native_process':True,
                  'scene_sha256':record['scene_sha256'],'evidence':str(output/'reopen_result.json'),
                  'native_module_search_roots':list(map(str,module_roots))}}
    runtime_name='runtime_dependencies.json' if scene.name=='world_physics.usda' else 'runtime_dependencies_'+scene.stem+'.json'
    (scene.parent/runtime_name).write_text(json.dumps(runtime,indent=2)+'\n')
    bundled=[];absolute=[]
    for item in files:
        path=pathlib.Path(item['path']).resolve()
        if path.is_relative_to(scene.parent):
            bundled.append(dict(item,path=str(path.relative_to(scene.parent))))
        elif path not in visited:absolute.append(str(path))
    for layer in layers:
        for dependency in layer.GetExternalReferences():
            if pathlib.Path(dependency).is_absolute():absolute.append(dependency)
    closure={'status':'pass' if record['status']=='pass' and not absolute else 'fail',
        'root_asset':scene.name,'root_sha256':record['scene_sha256'],
        'files':bundled,'unresolved_paths':unresolved,'absolute_asset_paths':sorted(set(absolute)),
        'method':'UsdUtils.ComputeAllDependencies','toolchain':{'isaac':record['native_isaac_version'],'usd':list(Usd.GetVersion())},
        'runtime_dependencies_manifest':runtime_name,
        'runtime_dependencies_sha256':sha(scene.parent/runtime_name),
        'verified_runtime_asset_names':resolved_builtin}
    closure_name='native_physics_dependency_closure.json' if scene.name=='world_physics.usda' else 'native_'+scene.stem+'_dependency_closure.json'
    (scene.parent/closure_name).write_text(json.dumps(closure,indent=2)+'\n')

except BaseException:
    import traceback
    (output/'reopen_failure.json').write_text(traceback.format_exc())
    print(traceback.format_exc(),flush=True)
    raise
finally:
    app.close()
