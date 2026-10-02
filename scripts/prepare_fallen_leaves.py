"""Derive fine curved leaf meshes from original CC0 photographic atlas channels.

Scale, curl and optical constants are explicit construction assumptions. This
is photograph-derived geometry, not an upstream scanned whole-leaf 3D model.
The original colour, normal, roughness, displacement and opacity bytes remain
unchanged and portable; native appearance still needs actual Isaac inspection.
"""
from pathlib import Path
import argparse
import os
import numpy as np
from PIL import Image
from scipy import ndimage
from pxr import Usd, UsdGeom, UsdShade, Sdf, Vt
from isaacmin.io import read_json, atomic_json, sha256_file, utc_now
from isaacmin.assembly.usd_instances import _closure


def prepare(card_path, out):
    card=read_json(card_path);aid=card['asset_id'];directory=out/aid
    directory.mkdir(parents=True,exist_ok=False)
    maps={};proof=[]
    for role in ('Color','NormalGL','Roughness','Opacity','Displacement'):
        entries=[f for f in card['extracted_files'] if f['path'].endswith('_'+role+'.png')]
        if len(entries)!=1:raise ValueError('Required original atlas map missing: '+role)
        entry=entries[0];source=Path(card['extracted_directory'])/entry['path']
        if source.is_symlink() or sha256_file(source)!=entry['sha256']:raise ValueError('Original atlas changed')
        target=directory/'textures'/source.name;target.parent.mkdir(exist_ok=True)
        os.link(source,target);maps[role]=target
        proof.append(dict(role=role,path=str(target.relative_to(directory)),sha256=entry['sha256']))
    opacity=np.asarray(Image.open(maps['Opacity']).convert('L'))
    height_image=np.asarray(Image.open(maps['Displacement']),float)
    height_image/=65535. if height_image.max()>255 else 255.
    if height_image.ndim==3:height_image=height_image[:,:,0]
    labels,_=ndimage.label(opacity>127)
    slices=ndimage.find_objects(labels);counts=np.bincount(labels.ravel())
    components=[(i+1,s) for i,s in enumerate(slices) if s and counts[i+1]>=1024]
    if not components:raise ValueError('No complete leaf silhouettes resolved')
    usd=directory/'library.usdc';stage=Usd.Stage.CreateNew(str(usd))
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage,'/Library').GetPrim())
    UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z);UsdGeom.SetStageMetersPerUnit(stage,1.)
    def texture(role,gamma):return 'texture_2d("./textures/'+maps[role].name+'",tex::gamma_'+gamma+')'
    mdl='''mdl 1.7;
import ::df::*; import ::state::*; import ::math::*; import ::tex::*; import ::scene::*;
export material IsaacMinFallenLeaf() = let {
    float3 uvw = state::texture_coordinate(0);
    float2 uv = scene::data_lookup_float2("st",float2(uvw.x,uvw.y));
    color c = tex::lookup_color(@@color@@,uv);
    float r = tex::lookup_float(@@rough@@,uv);
    float a = tex::lookup_float(@@alpha@@,uv);
    float3 n = math::normalize(state::normal());
    float3 tu = state::texture_tangent_u(0); float3 tv = state::texture_tangent_v(0);
    float3 t = math::normalize(tu-n*math::dot(n,tu));
    float3 b = math::normalize(tv-n*math::dot(n,tv)-t*math::dot(t,tv));
    float3 nt = tex::lookup_float3(@@normal@@,uv)*2.0-1.0;
    float3 normal = math::normalize(nt.x*t+nt.y*b+nt.z*n);
    bsdf body = df::normalized_mix(df::bsdf_component[](
        df::bsdf_component(0.92,df::diffuse_reflection_bsdf(tint:c)),
        df::bsdf_component(0.08,df::diffuse_transmission_bsdf(tint:c))));
    material_surface surface = material_surface(scattering:df::fresnel_layer(
        ior:1.45,weight:1.0,layer:df::microfacet_ggx_smith_bsdf(
        roughness_u:r*r,roughness_v:r*r,tint:color(1.0),mode:df::scatter_reflect),
        base:body,normal:normal));
} in material(thin_walled:true,ior:color(1.45),surface:surface,backface:surface,
    geometry:material_geometry(normal:normal,cutout_opacity:math::clamp(a,0.0,1.0)));
'''
    for key,role,gamma in [('color','Color','srgb'),('rough','Roughness','linear'),
                           ('alpha','Opacity','linear'),('normal','NormalGL','linear')]:
        mdl=mdl.replace('@@'+key+'@@',texture(role,gamma))
    (directory/'fallen_leaf.mdl').write_text(mdl)
    h,w=opacity.shape;prototypes=[]
    # Metric atlas scale follows a declared18cm median leaf length, retaining
    # the relative sizes of all specimens within each original photograph.
    median_length=np.median([max(s[0].stop-s[0].start,s[1].stop-s[1].start) for _,s in components])
    metres_per_pixel=.18/median_length
    for index,(label,slices_) in enumerate(components):
        y0=max(0,slices_[0].start-4);y1=min(h-1,slices_[0].stop+4)
        x0=max(0,slices_[1].start-4);x1=min(w-1,slices_[1].stop+4)
        # <=4 original pixels per mesh edge: submillimetre silhouettes and
        # true curvature/parallax, with no generated texture or flat cards.
        xx=np.unique(np.r_[np.arange(x0,x1,4),x1]);yy=np.unique(np.r_[np.arange(y0,y1,4),y1])
        gx,gy=np.meshgrid(xx,yy);nx=len(xx);ny=len(yy)
        px=(gx-(x0+x1)*.5)*metres_per_pixel
        py=((y0+y1)*.5-gy)*metres_per_pixel
        u=(gx-(x0+x1)*.5)/max(1.,(x1-x0)*.5)
        v=(gy-(y0+y1)*.5)/max(1.,(y1-y0)*.5)
        curl=(.0015+.00025*(index%5))*v*v+(.0005+.0002*(index%3))*u*u
        relief=(height_image[gy,gx]-.5)*.001
        z=curl+relief;z-=z.min()
        vertices=np.column_stack((px.ravel(),py.ravel(),z.ravel())).astype(np.float32)
        st=np.column_stack(((gx.ravel()+.5)/w,1.-(gy.ravel()+.5)/h)).astype(np.float32)
        valid=labels[gy,gx]==label
        quads=np.flatnonzero((valid[:-1,:-1]|valid[:-1,1:]|valid[1:,:-1]|valid[1:,1:]).ravel())
        qy,qx=np.divmod(quads,nx-1);a=qy*nx+qx
        triangles=np.concatenate((np.column_stack((a,a+nx,a+1)),np.column_stack((a+1,a+nx,a+nx+1)))).astype(np.int32)
        used,inverse=np.unique(triangles,return_inverse=True)
        vertices=vertices[used];st=st[used];triangles=inverse.reshape(-1,3).astype(np.int32)
        name=aid+'_leaf_'+str(index).zfill(2);path='/Library/'+name
        mesh=UsdGeom.Mesh.Define(stage,path+'/Mesh');UsdGeom.Xform.Define(stage,path)
        # A leaf is referenced as a subtree. Its material binding must remain
        # inside that same subtree to survive native USD reference composition.
        mat=UsdShade.Material.Define(stage,path+'/Looks/Leaf')
        shader=UsdShade.Shader.Define(stage,mat.GetPath().AppendChild('Shader'))
        shader.CreateImplementationSourceAttr().Set(UsdShade.Tokens.sourceAsset)
        shader.SetSourceAsset(Sdf.AssetPath('./fallen_leaf.mdl'),'mdl')
        shader.SetSourceAssetSubIdentifier('IsaacMinFallenLeaf','mdl')
        shader.CreateOutput('out',Sdf.ValueTypeNames.Token)
        mat.CreateSurfaceOutput('mdl').ConnectToSource(shader.ConnectableAPI(),'out')
        # USD cannot discover paths embedded inside MDL source. Declare every
        # original atlas channel so portable dependency closure copies it too.
        mat.GetPrim().CreateAttribute('isaacmin:originalTextures',Sdf.ValueTypeNames.AssetArray).Set(
            [Sdf.AssetPath('./'+row['path']) for row in proof])
        mesh.CreatePointsAttr().Set(Vt.Vec3fArray.FromNumpy(vertices))
        mesh.CreateFaceVertexCountsAttr().Set(Vt.IntArray.FromNumpy(np.full(len(triangles),3,np.int32)))
        mesh.CreateFaceVertexIndicesAttr().Set(Vt.IntArray.FromNumpy(triangles.ravel()))
        mesh.CreateSubdivisionSchemeAttr().Set('none');mesh.CreateDoubleSidedAttr().Set(True)
        normal=np.zeros_like(vertices)
        tri_normal=np.cross(vertices[triangles[:,1]]-vertices[triangles[:,0]],vertices[triangles[:,2]]-vertices[triangles[:,0]])
        for corner in range(3):np.add.at(normal,triangles[:,corner],tri_normal)
        length=np.linalg.norm(normal,axis=1)
        if np.any(length==0):raise ValueError('Leaf has zero-area incident geometry')
        normal/=length[:,None]
        mesh.CreateNormalsAttr().Set(Vt.Vec3fArray.FromNumpy(normal));mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)
        UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.faceVarying).Set(Vt.Vec2fArray.FromNumpy(st[triangles.ravel()]))
        UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mat)
        # The whole lower lamina is sampled, including the lowest contact point.
        cells=np.floor(vertices[:,:2]/.004).astype(np.int32)
        order=np.lexsort((vertices[:,2],cells[:,1],cells[:,0]));c=cells[order];vtx=vertices[order]
        first=np.r_[True,np.any(c[1:]!=c[:-1],axis=1)]
        anchors=vtx[first]
        np.savez(directory/(name+'.npz'),points=vertices,indices=triangles,uvs=st,normals=normal)
        if not np.array_equal(np.asarray(mesh.GetPointsAttr().Get()),vertices):raise ValueError('Native leaf points changed')
        prototypes.append(dict(object_name=name,usd_prim=path,vertices=len(vertices),triangles=len(triangles),
            dimensions_m=(vertices.max(axis=0)-vertices.min(axis=0)).tolist(),
            contact_anchors_local_m=anchors.tolist(),geometry_verified_exact=True,source_uv_verified_exact=True,
            verification_scope='Authored native USD equals saved photograph-derived mesh arrays; no upstream whole-mesh claim',
            geometry_provenance='Original opacity silhouette plus original displacement with explicitly authored curl',
            material_provenance=proof,original_atlas_component=label))
    stage.GetRootLayer().Save()
    # Fresh native reopen checks byte-exact positions and UV coordinates.
    reopened=Usd.Stage.Open(str(usd))
    for p in prototypes:
        arrays=np.load(directory/(p['object_name']+'.npz'));mesh=UsdGeom.Mesh(reopened.GetPrimAtPath(p['usd_prim']+'/Mesh'))
        if not np.array_equal(np.asarray(mesh.GetPointsAttr().Get()),arrays['points']):raise ValueError('Leaf reopen position mismatch')
        if not np.array_equal(np.asarray(UsdGeom.PrimvarsAPI(mesh).GetPrimvar('st').Get()),arrays['uvs'][arrays['indices'].ravel()]):raise ValueError('Leaf reopen UV mismatch')
        reference=Usd.Stage.CreateInMemory()
        top=UsdGeom.Xform.Define(reference,'/Leaf').GetPrim()
        top.GetReferences().AddReference(str(usd),p['usd_prim'])
        material,_=UsdShade.MaterialBindingAPI(reference.GetPrimAtPath('/Leaf/Mesh')).ComputeBoundMaterial()
        if not material:raise ValueError('Leaf material binding lost through native subtree reference')
    closure=_closure(usd)
    dependencies={row['path']:row['sha256'] for row in closure['files']}
    if any(dependencies.get(row['path'])!=row['sha256'] for row in proof):
        raise ValueError('Portable leaf closure omits an original atlas channel')
    atomic_json(directory/'native_dependency_closure.json',closure)
    return dict(asset_id=aid,usd=str(usd),usd_sha256=sha256_file(usd),prototypes=prototypes,
        licence=card['licence'],source_page=card['source_page'],original_card=str(card_path),
        original_card_sha256=sha256_file(card_path),qualification='not_run',
        assumptions=dict(median_leaf_length_m=.18,metres_per_pixel=metres_per_pixel,
            curl_m='1.5–3mm quadratic edges, fixed per scanned specimen',scan_relief_range_m=.001,
            optical_ior=1.45,diffuse_transmission_fraction=.08,physical_values='inferred, not provider-measured'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('cards',nargs='+',type=Path)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    assets=[prepare(card.resolve(),out) for card in a.cards]
    atomic_json(out/'library.json',dict(at_utc=utc_now(),status='photographic_leaf_mesh_candidates',assets=assets,
        producer_sha256=sha256_file(Path(__file__)),qualification='not_run'))
    print({asset['asset_id']:len(asset['prototypes']) for asset in assets},flush=True)
