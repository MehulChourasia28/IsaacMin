"""Explicit defects authored only into disposable native USD calibration layers."""
import numpy as np
from pxr import Usd, UsdGeom, UsdShade, Gf, Sdf


def _terrain(stage):
    return [UsdGeom.Mesh(p) for p in stage.Traverse() if p.IsA(UsdGeom.Mesh) and 'Terrain_FinalGround' in str(p.GetPath())]


def _world_points(mesh):
    matrix=UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.asarray([tuple(matrix.Transform(Gf.Vec3d(*p))) for p in mesh.GetPointsAttr().Get()]),matrix


def _delete_faces(mesh,keep):
    counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get(),int)
    corners=np.repeat(keep,counts)
    indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),int)
    remap=np.cumsum(keep)-1
    for subset in UsdGeom.Subset.GetAllGeomSubsets(mesh):
        old=np.asarray(subset.GetIndicesAttr().Get(),int)
        subset.GetIndicesAttr().Set(remap[old[keep[old]]].tolist())
    for pv in UsdGeom.PrimvarsAPI(mesh).GetPrimvars():
        interpolation=pv.GetInterpolation()
        mask=corners if interpolation==UsdGeom.Tokens.faceVarying else keep if interpolation==UsdGeom.Tokens.uniform else None
        if mask is None:continue
        if pv.IsIndexed():
            data=list(pv.GetIndices())
            if len(data)==len(mask):pv.SetIndices([v for v,k in zip(data,mask) if k])
        else:
            data=pv.Get()
            if data is not None and len(data)==len(mask):pv.Set([v for v,k in zip(data,mask) if k])
    normals=mesh.GetNormalsAttr().Get()
    interpolation=mesh.GetNormalsInterpolation()
    mask=corners if interpolation==UsdGeom.Tokens.faceVarying else keep if interpolation==UsdGeom.Tokens.uniform else None
    if normals is not None and mask is not None and len(normals)==len(mask):
        mesh.GetNormalsAttr().Set([v for v,k in zip(normals,mask) if k])
    mesh.GetFaceVertexCountsAttr().Set(counts[keep].tolist())
    mesh.GetFaceVertexIndicesAttr().Set(indices[corners].tolist())


def inject(stage,category,pose):
    eye=np.asarray(pose['position'],float);look=np.asarray(pose['look_at'],float)
    direction=look[:2]-eye[:2];direction/=max(np.linalg.norm(direction),1e-9)
    center=eye[:2]+direction*3
    across=np.array([-direction[1],direction[0]])
    result={'category':category,'changed_elements':0,'prims':[],
            'classification':'deliberately_degraded_calibration_only_never_release_world'}
    if category in ('conspicuous_seam',):
        for mesh in _terrain(stage):
            points,matrix=_world_points(mesh)
            if category=='conspicuous_seam':
                counts=np.asarray(mesh.GetFaceVertexCountsAttr().Get(),int)
                indices=np.asarray(mesh.GetFaceVertexIndicesAttr().Get(),int)
                ends=np.cumsum(counts);starts=ends-counts
                centers=np.array([points[indices[a:b]].mean(axis=0) for a,b in zip(starts,ends)])
                delta=centers[:,:2]-center
                remove=(np.abs(delta@direction)<.55)&(np.abs(delta@across)<4)&(centers[:,2]>eye[2]-5)
                if remove.any():_delete_faces(mesh,~remove)
                changed=int(remove.sum())
                result['mutation_parameters']={'open_gap_width_m':1.1,'open_gap_length_m':8,'faces_deleted':True}
            else:
                select=(np.linalg.norm(points[:,:2]-center,axis=1)<10)&(points[:,2]>eye[2]-5)
                altered=points.copy();altered[select]=np.round(altered[select]/.65)*.65
                changed=int((np.linalg.norm(altered-points,axis=1)>1e-5).sum())
                inverse=matrix.GetInverse()
                mesh.GetPointsAttr().Set([Gf.Vec3f(*inverse.Transform(Gf.Vec3d(*p))) for p in altered])
                result['mutation_parameters']={'quantization_m':.65,'radius_m':10,'geometry_mutation':True}
            result['changed_elements']+=changed
            if changed:result['prims'].append(str(mesh.GetPath()))
    elif category=='retained_voxel_steps':
        # Controlled native geometric staircase, deliberately exaggerated after
        # the first quantization mutation was not visibly severe. This belongs
        # exclusively to calibration; it is not a source-world reconstruction.
        ground=_terrain(stage)[0];material,_=UsdShade.MaterialBindingAPI(ground.GetPrim()).ComputeBoundMaterial()
        first=eye[:2]+direction*1.2;half_width=3.;rise=.35;tread=.8;steps=10;base=eye[2]-.7
        vertices=[];faces=[];tex=[]
        def quad(points):
            begin=len(vertices);vertices.extend(points);faces.extend([begin,begin+1,begin+2,begin+3])
            for p in points:tex.append(Gf.Vec2f(float(p[0])/1.3,float(p[1])/1.3))
        for i in range(steps):
            near=first+direction*i*tread;far=near+direction*tread;level=base+i*rise;below=base+(i-1)*rise
            a=near-across*half_width;b=near+across*half_width;c=far+across*half_width;d=far-across*half_width
            quad([(a[0],a[1],level),(b[0],b[1],level),(c[0],c[1],level),(d[0],d[1],level)])
            quad([(a[0],a[1],below),(b[0],b[1],below),(b[0],b[1],level),(a[0],a[1],level)])
        injected=UsdGeom.Mesh.Define(stage,'/IsaacMinCalibrationFault/RetainedVoxelTerraces')
        injected.GetPointsAttr().Set([Gf.Vec3f(*p) for p in vertices]);injected.GetFaceVertexCountsAttr().Set([4]*(len(faces)//4));injected.GetFaceVertexIndicesAttr().Set(faces)
        injected.GetDoubleSidedAttr().Set(True);injected.GetSubdivisionSchemeAttr().Set('none')
        UsdGeom.PrimvarsAPI(injected).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.faceVarying).Set(tex)
        UsdShade.MaterialBindingAPI.Apply(injected.GetPrim()).Bind(material)
        result['prims']=[str(injected.GetPath())];result['changed_elements']=len(vertices)
        result['mutation_parameters']={'controlled_native_terraces':True,'step_rise_m':rise,'step_tread_m':tread,'step_count':steps,'width_m':half_width*2,
                                       'appearance':'unchanged actual source-fixture terrain material','scope':'calibration-only deliberately exaggerated regular terrain steps'}
    elif category=='floating_vegetation':
        candidates={}
        bbox=UsdGeom.BBoxCache(Usd.TimeCode.Default(),['default','render'],useExtentsHint=False)
        for prim in Usd.PrimRange(stage.GetPseudoRoot(),Usd.TraverseInstanceProxies()):
            name=str(prim.GetPath()).lower()
            if not prim.IsA(UsdGeom.Mesh) or not any(s in name for s in ('grass','fern','dandelion','celandine')):continue
            target=prim
            while target.IsInstanceProxy():target=target.GetParent()
            if not UsdGeom.Xformable(target):continue
            bound=bbox.ComputeWorldBound(prim).ComputeAlignedRange()
            location=np.array(bound.GetMidpoint())
            relative=location[:2]-eye[:2]
            if np.linalg.norm(relative)<14 and relative@direction>.5:
                candidates[str(target.GetPath())]=target
        for path,prim in sorted(candidates.items())[:80]:
            xform=UsdGeom.Xformable(prim);local=xform.GetLocalTransformation()
            parent=UsdGeom.Xformable(prim.GetParent()).ComputeLocalToWorldTransform(Usd.TimeCode.Default()) if prim.GetParent().IsA(UsdGeom.Xformable) else Gf.Matrix4d(1)
            local.SetTranslateOnly(local.ExtractTranslation()+parent.GetInverse().TransformDir(Gf.Vec3d(0,0,1.0)))
            xform.ClearXformOpOrder();xform.AddTransformOp(opSuffix='InjectedFloatingRoot').Set(local)
            result['prims'].append(path)
        result['changed_elements']=len(result['prims']);result['mutation_parameters']={'world_z_lift_m':1.0}
    elif category=='missing_material':
        material=UsdShade.Material.Define(stage,'/IsaacMinCalibrationFault/MissingMaterial')
        shader=UsdShade.Shader.Define(stage,'/IsaacMinCalibrationFault/MissingMaterial/Shader')
        shader.CreateIdAttr('UsdPreviewSurface')
        shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1,0,1))
        shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(1)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
        for mesh in _terrain(stage):
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(material,bindingStrength=UsdShade.Tokens.strongerThanDescendants)
            result['prims'].append(str(mesh.GetPath()))
        result['changed_elements']=len(result['prims']);result['mutation_parameters']={'terrain_binding':'explicit magenta missing-material diagnostic'}
    elif category=='broken_leaf_opacity':
        for prim in stage.Traverse():
            if not prim.IsA(UsdShade.Shader):continue
            shader=UsdShade.Shader(prim)
            opacity=shader.GetInput('opacity')
            if shader.GetIdAttr().Get()!='UsdPreviewSurface' or not opacity or not opacity.HasConnectedSource():continue
            threshold=shader.CreateInput('opacityThreshold',Sdf.ValueTypeNames.Float)
            threshold.Set(0.0);result['prims'].append(str(threshold.GetAttr().GetPath()))
        result['changed_elements']=len(result['prims']);result['mutation_parameters']={'original_real_mask_connection_preserved':True,'opacityThreshold':0,
                'reproduction':'exact native UsdPreviewSurface cutout-threshold omission previously produced independently measured opaque card regions'}
    elif category!='baseline':
        raise ValueError('Unsupported bounded calibration mutation')
    return result
