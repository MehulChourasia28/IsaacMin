"""Real Isaac raw captures of a baseline and explicitly degraded disposable USDs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

parser=argparse.ArgumentParser();parser.add_argument('--request',required=True)
args,_=parser.parse_known_args();request=json.loads(Path(args.request).read_text())
output=Path(request['output']).resolve();output.mkdir(parents=True,exist_ok=True)
source=Path(request['scene']).resolve();source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
closure_path=source.parent/'native_dependency_closure.json'
closure=json.loads(closure_path.read_text())
dependencies={'manifest':{'path':str(closure_path),'sha256':hashlib.sha256(closure_path.read_bytes()).hexdigest()},'files':[]}
if closure.get('status')!='pass' or closure['root_sha256']!=source_hash:raise RuntimeError('Calibration source dependency closure does not bind this stage')
for bound in closure['files']:
    path=(source.parent/bound['path']).resolve()
    if not path.is_relative_to(source.parent) or hashlib.sha256(path.read_bytes()).hexdigest()!=bound['sha256']:raise RuntimeError('Calibration source content layer/texture changed')
    dependencies['files'].append({'path':str(path),'sha256':bound['sha256']})
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'renderer':'RayTracedLighting','width':1280,'height':720,
                   'extra_args':[f'--/log/file={output}/kit.log','--/app/window/enabled=false','--/rtx/post/dlss/execMode=2']})
try:
    import numpy as np
    import omni.usd
    import omni.replicator.core as rep
    import carb
    from PIL import Image
    from pxr import Sdf,Usd,UsdGeom,UsdLux,Gf
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    from calibration_mutations import inject
    categories=request.get('categories',['baseline','conspicuous_seam','floating_vegetation','missing_material','broken_leaf_opacity','retained_voxel_steps'])
    if categories[0]!='baseline' or len(set(categories))!=len(categories) or not set(categories)<={'baseline','conspicuous_seam','floating_vegetation','missing_material','broken_leaf_opacity','retained_voxel_steps'}:
        raise RuntimeError('Unsupported frozen calibration category selection')
    dataset={'schema_version':1,'source':'actual_isaac_raw_rgb','scene':str(source),'scene_sha256':source_hash,
             'cases':[],'limitations':['Uninjected baseline is not presumed independently clean',
                                      'Pixel change is only observability evidence; no universal visual judgement']}
    for split,pose in request['poses'].items():
        baseline=None
        for category in categories:
            folder=output/(split+'_'+category);folder.mkdir(parents=True,exist_ok=True)
            variant=folder/'calibration_only.usda'
            layer=Sdf.Layer.CreateNew(str(variant));layer.subLayerPaths=[str(source)]
            layer.customLayerData={'isaacmin_scope':'deliberately_degraded_critic_calibration_only'};layer.Save()
            if not omni.usd.get_context().open_stage(str(variant)):raise RuntimeError('Native calibration stage failed to open')
            for _ in range(30):app.update()
            stage=omni.usd.get_context().get_stage()
            UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
            UsdGeom.SetStageMetersPerUnit(stage,1.)
            mutation=inject(stage,category,pose)
            if not stage.GetPrimAtPath('/IsaacMinLighting/Dome'):
                dome=UsdLux.DomeLight.Define(stage,'/IsaacMinLighting/Dome');dome.CreateIntensityAttr(800)
                sun=UsdLux.DistantLight.Define(stage,'/IsaacMinLighting/Sun');sun.CreateIntensityAttr(1500)
                UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-45,20,25))
            camera=rep.create.camera(position=pose['position'],look_at=pose['look_at'],look_at_up_axis=(0,0,1),focal_length=18,horizontal_aperture=36,clipping_range=(.02,5000))
            product=rep.create.render_product(camera,(1280,720))
            rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach(product)
            depth=rep.AnnotatorRegistry.get_annotator('distance_to_image_plane');depth.attach(product)
            for _ in range(20):app.update()
            rep.orchestrator.step(rt_subframes=16,pause_timeline=True)
            pixels=rgb.get_data();axial=depth.get_data()
            if pixels.shape[:2]!=(720,1280) or pixels.std()<2:raise RuntimeError('Invalid actual Isaac calibration RGB')
            camera_prim=next(p for root in camera.get_output_prims()['prims'] for p in Usd.PrimRange(root) if p.IsA(UsdGeom.Camera))
            camera_matrix=np.asarray(UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),float)
            forward=np.asarray(pose['look_at'],float)-pose['position'];forward/=np.linalg.norm(forward)
            if abs(camera_matrix[0,2])>1e-5 or camera_matrix[1,2]<=0 or np.dot(-camera_matrix[2,:3],forward)<.99999 or not np.allclose(camera_matrix[3,:3],pose['position'],atol=1e-5):
                raise RuntimeError('Actual calibration camera matrix differs from frozen Z-up pose or contains unintended roll')
            image=folder/'raw_rgb.png';depth_path=folder/'raw_axial_depth.npy'
            Image.fromarray(pixels).save(image);np.save(depth_path,axial)
            if baseline is None:baseline=pixels[:,:,:3].astype(float)
            delta=np.mean(np.abs(pixels[:,:,:3].astype(float)-baseline),axis=2)>32
            changed=int(delta.sum());ys,xs=np.nonzero(delta)
            box=[float(xs.min()/1280),float(ys.min()/720),float((xs.max()+1)/1280),float((ys.max()+1)/720)] if changed else [0,0,1,1]
            obvious=category!='baseline' and mutation['changed_elements']>0 and changed/(1280*720)>=.02
            stage.GetRootLayer().Save()
            metadata={'renderer':'RayTracedLighting','postprocessing':'none','pose':pose,'resolution':[1280,720],
                      'actual_camera_world_matrix_row_vectors':camera_matrix.tolist(),'camera_up_policy':'explicit world +Z, zero roll',
                      'camera_focal_length_mm':float(UsdGeom.Camera(camera_prim).GetFocalLengthAttr().Get()),
                      'camera_horizontal_aperture_mm':float(UsdGeom.Camera(camera_prim).GetHorizontalApertureAttr().Get()),
                      'raw_image_sha256':hashlib.sha256(image.read_bytes()).hexdigest(),
                      'raw_depth_sha256':hashlib.sha256(depth_path.read_bytes()).hexdigest(),'depth_semantics':'axial_metres',
                      'source_scene_sha256':source_hash,'variant_scene_sha256':hashlib.sha256(variant.read_bytes()).hexdigest(),
                      'captured_scene_dependencies':dependencies,
                      'injected_category':category,'mutation':mutation,'changed_pixel_count':changed,
                      'large_pixel_change_bbox':box,'observable_mutation_threshold_fraction':.02,
                      'producer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      'mutation_code_sha256':hashlib.sha256((Path(__file__).parent/'calibration_mutations.py').read_bytes()).hexdigest(),
                      'dlss_mode':carb.settings.get_settings().get('/rtx/post/dlss/execMode'),
                      'appearance_qualification':'not_run'}
            meta_path=folder/'capture_metadata.json';meta_path.write_text(json.dumps(metadata,indent=2))
            dataset['cases'].append({'id':split+'_'+category,'split':split,'image':str(image),'image_sha256':metadata['raw_image_sha256'],
                      'capture_metadata':str(meta_path),'capture_metadata_sha256':hashlib.sha256(meta_path.read_bytes()).hexdigest(),
                      'expected_severe':[],
                      'candidate_expected_severe':[{'category':category,'bbox':box}] if obvious else [],
                      'known_absent_categories':[],'clean_control_independently_verified':False,
                      'mutation_observability':'visible_candidate' if obvious else 'uninjected_control' if category=='baseline' else 'insufficient_observable_mutation',
                      'label_limitation':'Mutation and pixel change identify a fault candidate only. Independent actual-image inspection must establish visible category, severity and bbox before expected_severe is published; baseline cleanliness is separate.'})
            (output/'dataset.json').write_text(json.dumps(dataset,indent=2))
            rgb.detach(product);depth.detach(product);product.destroy()
    if hashlib.sha256(source.read_bytes()).hexdigest()!=source_hash:raise RuntimeError('Immutable source stage changed during calibration')
    for bound in [dependencies['manifest'],*dependencies['files']]:
        if hashlib.sha256(Path(bound['path']).read_bytes()).hexdigest()!=bound['sha256']:raise RuntimeError('Calibration source dependency changed during capture')
    dataset['status']='captured_unqualified';dataset['source_unchanged']=True
    (output/'dataset.json').write_text(json.dumps(dataset,indent=2))
finally:
    app.close()
