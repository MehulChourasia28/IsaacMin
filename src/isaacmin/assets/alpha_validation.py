"""Independent USD UV/mask rays versus actual Isaac instance segmentation.

This bounded validator measures one explicit opaque botanical cutout family.
It uses retained raw sensor output; no synthetic image can satisfy this check.
"""
from pathlib import Path
import json
import numpy as np
from PIL import Image
import trimesh

from .network import ServiceError, digest, atomic_json, utcnow
from .usd_provenance import verify_scene_closure,captured_closure_hash,validated_native_renderer


def _triangles(mesh, matrix):
    points=np.asarray(mesh.GetPointsAttr().Get(),float)
    points=points@matrix[:3,:3]+matrix[3,:3]
    counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get(),int)
    indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),int)
    from pxr import UsdGeom
    uv=UsdGeom.PrimvarsAPI(mesh).FindPrimvarWithInheritance('st')
    if not uv or not uv.HasValue():raise ServiceError('alpha_uv','Cutout mesh lacks native st UV')
    values=np.asarray(uv.ComputeFlattened(),float)
    if uv.GetInterpolation()=='vertex':values=values[indices]
    elif uv.GetInterpolation()!='faceVarying':raise ServiceError('alpha_uv','Explicit vertex or face-varying UV required')
    faces=[];tex=[];offset=0
    for count in counts:
        if count not in (3,4):raise ServiceError('alpha_topology','Only native triangular/quad cutout faces supported by this independent test')
        for j in range(1,count-1):
            corners=[offset,offset+j,offset+j+1]
            faces.append(indices[corners]);tex.append(values[corners])
        offset+=count
    return trimesh.Trimesh(points,np.asarray(faces),process=False),np.asarray(tex)


def _sample_mask(image,uv):
    # USD st origin lower-left; image file array origin upper-left. Bilinear texels.
    h,w=image.shape;x=(uv[:,0]%1)*w-.5;y=(1-uv[:,1]%1)*h-.5
    x0=np.floor(x).astype(int);y0=np.floor(y).astype(int);a=x-x0;b=y-y0
    return (image[y0%h,x0%w]*(1-a)*(1-b)+image[y0%h,(x0+1)%w]*a*(1-b)
            +image[(y0+1)%h,x0%w]*(1-a)*b+image[(y0+1)%h,(x0+1)%w]*a*b)


def validate_alpha(scene, capture_result, output, *, material_name='fern_02', frames=(12,13), stride=3,
                   minimum_clear_samples=200, maximum_mismatch_fraction=.03):
    """Check clear card-background and opaque-leaf samples away from mask edges.

    Multiple fronds are ray traced: transparent foreground may expose a different
    opaque leaf. Ground occlusion is independently traced as well. A conservative
    3% trigger allows sensor/AA boundary differences; no texture/threshold edits.
    """
    from pxr import Usd,UsdGeom,UsdShade
    scene=Path(scene).resolve();capture_result=Path(capture_result).resolve();capture=json.loads(capture_result.read_text())
    validated_native_renderer(capture)
    if digest(scene)!=capture['scene_sha256']:
        raise ServiceError('alpha_capture_provenance','Actual capture must match the exact inspected native scene')
    closure=verify_scene_closure(scene,captured_closure_hash(capture))
    stage=Usd.Stage.Open(str(scene));xf=UsdGeom.XformCache();plants=[];grounds=[];uv_arrays=[];paths=[];mask_path=None
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):continue
        mesh=UsdGeom.Mesh(prim);mat,_=UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        matrix=np.asarray(xf.GetLocalToWorldTransform(prim),float)
        if 'Terrain_FinalGround' in str(prim.GetPath()):
            ground,_=_triangles(mesh,matrix);grounds.append(ground)
        if not mat or mat.GetPrim().GetName()!=material_name:continue
        shader,_,_=mat.ComputeSurfaceSource();opacity=shader.GetInput('opacity')
        source=opacity.GetConnectedSource()
        if not source:raise ServiceError('alpha_connection','Actual cutout opacity has no direct texture source')
        texture=UsdShade.Shader(source[0].GetPrim())
        if texture.GetIdAttr().Get()!='UsdUVTexture' or str(source[1]) not in ('r','a'):
            raise ServiceError('alpha_connection','Independent mask test supports direct single-channel UV texture only')
        asset=texture.GetInput('file').Get();path=Path(asset.resolvedPath)
        if mask_path and mask_path!=path:raise ServiceError('alpha_mask','Selected family uses different opacity textures')
        mask_path=path;channel=str(source[1])
        # Verify the authored transform, rather than silently assuming arbitrary UV scale.
        coordinate=texture.GetInput('st').GetConnectedSource()
        if coordinate:
            transform=UsdShade.Shader(coordinate[0].GetPrim())
            if transform.GetIdAttr().Get()=='UsdTransform2d':
                if tuple(transform.GetInput('scale').Get())!=(1,1) or tuple(transform.GetInput('translation').Get())!=(0,0) or transform.GetInput('rotation').Get()!=0:
                    raise ServiceError('alpha_uv_transform','Nonidentity texture transform must be handled explicitly')
        plant,uv=_triangles(mesh,matrix);plants.append(plant);uv_arrays.append(uv);paths.append(str(prim.GetPath()))
    if not plants or not grounds or mask_path is None:raise ServiceError('alpha_coverage','Declared botanical material and final ground are required')
    plant=trimesh.util.concatenate(plants);ground=trimesh.util.concatenate(grounds);uvs=np.concatenate(uv_arrays)
    with Image.open(mask_path) as image:
        values=np.asarray(image.convert('RGBA'),float)/255
        mask=values[:,:,3 if channel=='a' else 0]
    measurements=[]
    for number in frames:
        frame=next(f for f in capture['frames'] if f['frame']==number)
        segmentation=capture_result.parent/frame['instance_segmentation']
        rgb=capture_result.parent/frame['rgb']
        if digest(segmentation)!=frame['instance_segmentation_sha256'] or digest(rgb)!=frame['rgb_sha256']:
            raise ServiceError('alpha_changed_capture','Actual segmentation or RGB bytes changed')
        ids=np.load(segmentation,allow_pickle=False);ids=np.squeeze(ids);h,w=ids.shape
        camera=np.asarray(frame['camera_world_matrix_row_vectors'],float);origin=camera[3,:3]
        view=(plant.vertices-origin)@camera[:3,:3].T
        visible=view[:,2]<-.02
        projected=np.column_stack((frame['cx_pixels']+frame['fx_pixels']*view[visible,0]/-view[visible,2],
                                   frame['cy_pixels']-frame['fy_pixels']*view[visible,1]/-view[visible,2]))
        lo=np.maximum(np.floor(projected.min(axis=0)).astype(int),0);hi=np.minimum(np.ceil(projected.max(axis=0)).astype(int),[w-1,h-1])
        xx,yy=np.meshgrid(np.arange(lo[0],hi[0]+1,stride),np.arange(lo[1],hi[1]+1,stride));x=xx.ravel();y=yy.ravel()
        rays=np.column_stack(((x+.5-frame['cx_pixels'])/frame['fx_pixels'],(frame['cy_pixels']-y-.5)/frame['fy_pixels'],-np.ones(len(x))))@camera[:3,:3]
        rays/=np.linalg.norm(rays,axis=1)[:,None];origins=np.repeat(origin[None,:],len(x),axis=0)
        locations,ray_ids,triangles=plant.ray.intersects_location(origins,rays,multiple_hits=True)
        bary=trimesh.triangles.points_to_barycentric(plant.triangles[triangles],locations)
        uv=np.einsum('ij,ijk->ik',bary,uvs[triangles]);opacity=_sample_mask(mask,uv)
        distances=np.linalg.norm(locations-origin,axis=1)
        ground_locations,ground_rays,_=ground.ray.intersects_location(origins,rays,multiple_hits=False)
        ground_distance=np.full(len(x),np.inf);ground_distance[ground_rays]=np.linalg.norm(ground_locations-origin,axis=1)
        front=distances<ground_distance[ray_ids]-.002
        ray_ids,opacity,distances=ray_ids[front],opacity[front],distances[front]
        nearest_opaque=np.full(len(x),np.inf);np.minimum.at(nearest_opaque,ray_ids[opacity>.95],distances[opacity>.95])
        nonclear=np.zeros(len(x),bool);nonclear[ray_ids[opacity>.05]]=True
        hit=np.zeros(len(x),bool);hit[ray_ids]=True
        expected_clear=hit&~nonclear
        # Exclude uncertain nearer edge pixels even if another opaque leaf is behind.
        ambiguous=np.zeros(len(x),bool);ambiguous[ray_ids[(opacity>=.05)&(opacity<=.95)&(distances<nearest_opaque[ray_ids])]]=True
        expected_leaf=np.isfinite(nearest_opaque)&~ambiguous
        plant_ids=[int(k) for k,v in frame['instance_id_to_prim_path'].items() if (v if isinstance(v,str) else str(v)) in paths]
        if not plant_ids:raise ServiceError('alpha_segmentation_identity','Actual instance labels do not identify native botanical geometry')
        observed=np.isin(ids[y,x],plant_ids)
        clear_errors=int(np.sum(expected_clear&observed));leaf_errors=int(np.sum(expected_leaf&~observed))
        clear_count=int(expected_clear.sum());leaf_count=int(expected_leaf.sum())
        status='incomplete' if min(clear_count,leaf_count)<minimum_clear_samples else 'fail' if clear_errors/clear_count>maximum_mismatch_fraction or leaf_errors/leaf_count>maximum_mismatch_fraction else 'pass'
        measurements.append({'frame':number,'rgb_sha256':frame['rgb_sha256'],'segmentation_sha256':frame['instance_segmentation_sha256'],
                             'sampled_rays':len(x),'clear_card_samples':clear_count,'clear_background_wrongly_fern':clear_errors,
                             'clear_mismatch_fraction':clear_errors/max(clear_count,1),'opaque_leaf_samples':leaf_count,
                             'opaque_leaf_missing':leaf_errors,'opaque_mismatch_fraction':leaf_errors/max(leaf_count,1),'status':status})
    report={'schema_version':1,'created_at_utc':utcnow(),'status':'pass' if all(m['status']=='pass' for m in measurements) else 'fail' if any(m['status']=='fail' for m in measurements) else 'incomplete',
            'scope':capture['scope'],'scene_sha256':digest(scene),'capture_result_sha256':digest(capture_result),
            'validator_files':[{'path':str(p.resolve()),'sha256':digest(p)} for p in (Path(__file__),Path(__file__).with_name('usd_provenance.py'))],
            'mask':{'path':str(mask_path),'sha256':digest(mask_path)},'native_plant_paths':paths,'material_name':material_name,
            'captured_scene_dependencies':closure,
            'method':'Independent native USD triangles/UV + unchanged opacity-map bilinear sampling + multiple-hit rays and final-ground occlusion versus retained actual Isaac instance segmentation',
            'frames':measurements,'maximum_mismatch_fraction':maximum_mismatch_fraction,'minimum_samples_each_class':minimum_clear_samples,
            'Q06':'partial_alpha_only; no whole-world appearance qualification','limitations':['Antialiasing/mask edge pixels excluded','Only declared material, poses and current texture scale covered','Backlight translucency and motion shimmer need separate target evidence']}
    atomic_json(Path(output),report);return report
