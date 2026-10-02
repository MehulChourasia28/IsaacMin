"""Static portable RGB/depth comparison; no physical or realism pass implied."""
import argparse
import json
import sys
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
from PIL import Image
from compare_reproduction import THRESHOLDS, digest, payload, label_codes


def compare(package,output):
    before=json.loads((package/'reproduction/reference.json').read_text())
    after=json.loads((output/'capture/preview_result.json').read_text())
    selected=json.loads((output/'selected_views.json').read_text())
    errors=[];results=[]
    if before['scene_sha256']!=after['scene_sha256']:errors.append('Scene bytes differ')
    if (before['renderer_recipe']!=after['renderer_recipe'] or
            before['native_renderer_settings']!=after['native_renderer_settings']):errors.append('Renderer controls differ')
    if len(after['frames'])!=len(selected):errors.append('Replay frame count differs')
    for index,b in zip(selected,after['frames']):
        a=before['frames'][index];arrays={}
        for name,root,frame in (('a',package/'reproduction/reference',a),('b',output/'capture',b)):
            for role in ('rgb','depth','instance_segmentation'):
                p=payload(root,frame[role],frame[role+'_sha256'])
                arrays[name,role]=(np.asarray(Image.open(p).convert('RGB'),np.int16)
                    if role=='rgb' else np.load(p,allow_pickle=False))
        rgb=np.abs(arrays['a','rgb']-arrays['b','rgb'])
        da,db=arrays['a','depth'],arrays['b','depth'];va=np.isfinite(da)&(da>0);vb=np.isfinite(db)&(db>0)
        valid=va&vb
        if not valid.any():raise ValueError('No valid native depth')
        delta=np.abs(da[valid].astype(float)-db[valid].astype(float));mapping={}
        labels_a=label_codes(arrays['a','instance_segmentation'].squeeze(),a,mapping)
        labels_b=label_codes(arrays['b','instance_segmentation'].squeeze(),b,mapping)
        m=dict(reference_view=index,rgb_mean_codes=float(rgb.mean()),rgb_p999_codes=float(np.quantile(rgb,.999)),
            depth_p999_m=float(np.quantile(delta,.999)),depth_max_m=float(delta.max()),
            depth_valid_masks_equal=bool(np.array_equal(va,vb)),semantic_labels_equal=bool(np.array_equal(labels_a,labels_b)),
            camera_matrix_max_difference=float(np.max(np.abs(np.array(a['camera_world_matrix_row_vectors'])-b['camera_world_matrix_row_vectors']))),
            intrinsics_equal=all(a[k]==b[k] for k in ('fx_pixels','fy_pixels','cx_pixels','cy_pixels')),
            camera_response_equal=a['per_frame_native_camera_settings']==b['per_frame_native_camera_settings'],
            pose_equal=a['pose']==b['pose'])
        good=(m['rgb_mean_codes']<=THRESHOLDS['rgb_mean_codes_max'] and m['rgb_p999_codes']<=THRESHOLDS['rgb_p999_codes_max']
            and m['depth_p999_m']<=THRESHOLDS['depth_p999_m_max'] and m['depth_max_m']<=THRESHOLDS['depth_max_m']
            and m['depth_valid_masks_equal'] and m['semantic_labels_equal']
            and m['camera_matrix_max_difference']<=THRESHOLDS['camera_matrix_abs_max']
            and m['intrinsics_equal'] and m['camera_response_equal'] and m['pose_equal'])
        m['status']='pass' if good else 'fail';results.append(m)
        if not good:errors.append('Static view differs: '+str(index))
    result=dict(status='fail' if errors else 'static_component_pass',frames=results,errors=errors,
        thresholds=THRESHOLDS,all_reference_views_replayed=selected==list(range(len(before['frames']))),
        package_manifest_sha256=digest(package/'package.json'),package_unchanged=True,
        source_save_referenced=False,generation_modules_imported=False,
        full_portable_qualification='incomplete; physical and held-out coverage absent',
        appearance='not_qualified',motion='not_run',navigation_stack='not_run_not_supplied')
    (output/'reproduction_comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--package',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=compare(args.package.resolve(),args.output.resolve())
    raise SystemExit(2 if result['status']=='fail' else 0)
