"""Original scan materials and source ownership; explicit candidate opt-in only.

The Blender half writes source-weight attributes after final geometry. The USD
half binds a closed MDL library only after exported points and weights are
verified. Neither path moves geometry or fills an invalid normal.
"""
from pathlib import Path
import hashlib,json,math,os,shutil,struct,uuid
import numpy as np

from .material_assignment import material_blend_weights, REQUIRED

RECIPE='source_shared_original_pbr_v1'
WEIGHT_NAMES=('IsaacMinMaterialWeights012','IsaacMinMaterialWeight3')
CHUNK=250000


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def _publish(path,record):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.partial.'+uuid.uuid4().hex)
    try:
        with temporary.open('x') as f:
            json.dump(record,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
        os.link(temporary,path)  # Atomic exclusive publication; never overwrite.
    finally:
        temporary.unlink(missing_ok=True)


def validate_weights(weights):
    a=np.asarray(weights)
    if a.ndim!=2 or a.shape[1]!=4 or len(a)==0:
        raise ValueError('Source weights require one nonempty four-component row per native vertex')
    for start in range(0,len(a),CHUNK):
        chunk=a[start:start+CHUNK]
        if not np.isfinite(chunk).all() or np.any(chunk<0) or np.any(chunk>1) or np.any(abs(chunk.sum(axis=1)-1)>2e-6):
            raise ValueError('Native source weights must be finite nonnegative partitions of unity')
    return {'vertices':len(a),'sha256':hashlib.sha256(memoryview(np.ascontiguousarray(a,dtype='<f4'))).hexdigest()}


def png_dimensions(path):
    with Path(path).open('rb') as f:head=f.read(33)
    if len(head)!=33 or head[:8]!=b'\x89PNG\r\n\x1a\n' or head[12:16]!=b'IHDR':
        raise ValueError('Original material must be a PNG with an intact IHDR header')
    return struct.unpack('>II',head[16:24])


def write_source_weights(terrain,source_surface_path,minecraft_origin_xyz,output_directory,*,
                         exterior_delta_spec=None,structural_source_coordinates=None):
    """Native Blender call after authoritative triangulation/displacement."""
    mesh=terrain.data;count=len(mesh.vertices);output=Path(output_directory).resolve()
    destination=output/'source_weights.json'
    if destination.exists():raise RuntimeError('Source-weight evidence is immutable; use a fresh output directory')
    source_path=Path(source_surface_path).resolve();source_hash=digest(source_path)
    with np.load(source_path,allow_pickle=False) as f:source={k:f[k] for k in f.files}
    delta=None;delta_proof=None
    if exterior_delta_spec:
        path=Path(exterior_delta_spec['path']).resolve();delta_proof={'path':str(path),'sha256':digest(path),'spec':exterior_delta_spec}
        with np.load(path,allow_pickle=False) as f:
            delta=f[exterior_delta_spec['delta_key']].copy();delta[f[exterior_delta_spec['protection_key']].astype(bool)]=0
    matrix=np.asarray(terrain.matrix_world,dtype=np.float64)
    if not np.isfinite(matrix).all() or np.linalg.det(matrix[:3,:3])<=0:
        raise ValueError('Terrain source-weight export requires a finite nonsingular orientation-preserving transform')
    points=np.empty((count,3),np.float32);normals=np.empty_like(points)
    mesh.vertices.foreach_get('co',points.ravel());mesh.vertices.foreach_get('normal',normals.ravel())
    weights=np.empty((count,4),np.float32);world_hash=hashlib.sha256();normal_matrix=np.linalg.inv(matrix[:3,:3])
    for start in range(0,count,CHUNK):
        stop=min(count,start+CHUNK);p=points[start:stop].astype(np.float64)@matrix[:3,:3].T+matrix[:3,3]
        n=normals[start:stop].astype(np.float64)@normal_matrix;length=np.linalg.norm(n,axis=1)
        if not np.isfinite(p).all() or not np.isfinite(n).all() or np.any(length<=0):
            raise ValueError('Final native terrain contains nonfinite positions or zero/invalid vertex normals')
        n/=length[:,None];world_hash.update(np.ascontiguousarray(p,dtype='<f4').tobytes())
        weights[start:stop]=material_blend_weights(p,n,source,minecraft_origin_xyz,exterior_delta=delta,
                                                   structural_source_coordinates=structural_source_coordinates)
    proof=validate_weights(weights)
    if digest(source_path)!=source_hash or (delta_proof and digest(delta_proof['path'])!=delta_proof['sha256']):
        raise RuntimeError('Source material inputs changed during weight computation')
    before=hashlib.sha256(memoryview(points)).hexdigest()
    # Blender may reallocate CustomData when another attribute is added. An RNA
    # reference retained across that allocation can silently address `position`.
    # The native four-vertex reproduction is retained under evidence/services.
    for name,kind in zip(WEIGHT_NAMES,('FLOAT_VECTOR','FLOAT')):
        if not mesh.attributes.get(name):mesh.attributes.new(name=name,type=kind,domain='POINT')
    first=mesh.attributes[WEIGHT_NAMES[0]];fourth=mesh.attributes[WEIGHT_NAMES[1]]
    first.data.foreach_set('vector',np.ascontiguousarray(weights[:,:3]).ravel());fourth.data.foreach_set('value',weights[:,3].copy())
    mesh['isaacmin_material_weight_order']=json.dumps(list(REQUIRED));mesh['isaacmin_material_recipe']=RECIPE
    after=np.empty_like(points);mesh.vertices.foreach_get('co',after.ravel())
    if hashlib.sha256(memoryview(after)).hexdigest()!=before:raise RuntimeError('Source-weight authoring changed native positions')
    report={'schema_version':1,'status':'source_attributes_authored','qualification':'not_run','recipe':RECIPE,
            'native_points_sha256':before,'world_points_float32_sha256':world_hash.hexdigest(),
            'source_weights':proof,'source_weight_primvars':list(WEIGHT_NAMES),'material_order':list(REQUIRED),
            'source_surface':{'path':str(source_path),'sha256':source_hash},'exterior_delta':delta_proof,
            'minecraft_origin_xyz':list(minecraft_origin_xyz),'structural_source_coordinates':structural_source_coordinates or [],
            'structure_appearance':'Retained ownership; rock candidate only, masonry appearance unqualified',
            'weight_code_sha256':digest(Path(__file__).with_name('material_assignment.py')),'producer_sha256':digest(__file__),
            'geometry_policy':'No geometry, normals, topology, UVs, displacement, collision or sample density changed'}
    _publish(destination,report);return report


def prepare_library(workspace,candidate_manifest,output_directory):
    """Copy exact original files and emit one dependency-closed, metric MDL module."""
    workspace=Path(workspace).resolve();source=Path(candidate_manifest).resolve();output=Path(output_directory).resolve()
    if output.exists():raise RuntimeError('Material library output must be a new immutable directory')
    candidate=json.loads(source.read_text());materials=candidate['materials']
    if set(m['asset_id'] for m in materials)!=set(REQUIRED) or len(materials)!=4:
        raise ValueError('Shared material must contain exactly the declared four source families')
    template=workspace/'recipes/materials/shared_original.mdl.in';module=template.read_text();textures=[]
    output.mkdir(parents=True);(output/'textures').mkdir()
    for index,name in enumerate(REQUIRED):
        spec=next(m for m in materials if m['asset_id']==name);repeat=float(spec['repeat_m'])
        if not math.isfinite(repeat) or not .1<=repeat<=20 or spec.get('licence')!='CC0-1.0':
            raise ValueError('Original material needs verified physical scale and CC0 source provenance')
        module=module.replace('@@'+str(index)+'_repeat_m@@',repr(repeat))
        for role in ('base_color','roughness','normal'):
            file=Path(spec[role]).resolve();proof=spec['original_channel_proof'][role];file_hash=digest(file)
            if file_hash!=proof['sha256'] or png_dimensions(file)!=(proof['width'],proof['height']) or min(proof['width'],proof['height'])<4096 or min(proof['width'],proof['height'])/repeat<1024:
                raise ValueError('Shared material requires verified original >=4K maps and >=1024 source texels/metre')
            relative='textures/'+file_hash+'.png';target=output/relative
            if not target.exists():shutil.copy2(file,target)
            if digest(target)!=file_hash:raise RuntimeError('Copied original material bytes changed')
            module=module.replace('@@'+str(index)+'_'+role+'@@','./'+relative)
            textures.append({'asset_id':name,'role':role,'path':relative,'sha256':file_hash,'bytes':target.stat().st_size,
                             'original_path':str(file),'repeat_m':repeat,'width':proof['width'],'height':proof['height'],
                             'source_texels_per_m':min(proof['width'],proof['height'])/repeat,
                             'colour_space':'sRGB' if role=='base_color' else 'raw','source_page':spec['source_page'],'licence':spec['licence']})
    if '@@' in module:raise RuntimeError('Incomplete MDL material template')
    module_path=output/'IsaacMinScans.mdl';module_path.write_text(module)
    record={'schema_version':1,'status':'library_authored','qualification':'not_run','recipe':RECIPE,'material_order':list(REQUIRED),
            'module':'IsaacMinScans.mdl','module_sha256':digest(module_path),'source_asset_subidentifier':'IsaacMinScans','textures':textures,
            'source_candidate':{'path':str(source),'sha256':digest(source)},'template_sha256':digest(template),'producer_sha256':digest(__file__),
            'projection':'Metric world-position right-handed signed triplanar charts, abs(normalized world-space state::normal())^4 weights',
            'normal_semantics':'The input is the exported shading normal, not state::geometry_normal(). Reorient each decoded OpenGL tangent normal through its chart basis before blending world-space deltas; native renderer may adapt reflection normals',
            'originals_only':True,'runtime':'Pinned Isaac native MDL standard modules; target qualification remains separate'}
    _publish(output/'material_manifest.json',record);return record


def bind_shared_material(stage_path,material_manifest,source_weights_manifest,*,ground_prim_paths=None):
    """Author native USD binding only after exact exported weight/point checks.

    The caller must recompute the complete native dependency closure afterward.
    Authored overrides reside in the stage root layer, keeping the export payload
    bytes intact. This is not final visual qualification.
    """
    from pxr import Usd,UsdGeom,UsdShade,Sdf,Gf
    scene=Path(stage_path).resolve();manifest=Path(material_manifest).resolve();library=manifest.parent
    if not library.is_relative_to(scene.parent):raise ValueError('Shared material library must be inside the scene package')
    if not Path(source_weights_manifest).resolve().is_relative_to(scene.parent):
        raise ValueError('Source-weight manifest must be in package before binding')
    data=json.loads(manifest.read_text());source=json.loads(Path(source_weights_manifest).read_text())
    if data['material_order']!=list(REQUIRED) or source['material_order']!=list(REQUIRED):raise ValueError('Material family order changed')
    module=library/data['module']
    if digest(module)!=data['module_sha256']:raise ValueError('MDL module changed')
    for f in data['textures']:
        if digest(library/f['path'])!=f['sha256']:raise ValueError('Original scan dependency changed')
    stage=Usd.Stage.Open(str(scene))
    if not stage:raise ValueError('Cannot open actual exported USD stage')
    if abs(UsdGeom.GetStageMetersPerUnit(stage)-1)>1e-12:raise ValueError('Metric shared material requires metresPerUnit=1')
    ground=[stage.GetPrimAtPath(p) for p in ground_prim_paths] if ground_prim_paths else [p for p in stage.Traverse() if p.IsA(UsdGeom.Mesh) and 'Terrain_FinalGround' in str(p.GetPath())]
    if len(ground)!=1 or not ground[0].IsA(UsdGeom.Mesh):raise ValueError('Expected exactly one authoritative terrain mesh')
    prim=ground[0];mesh=UsdGeom.Mesh(prim);count=len(mesh.GetPointsAttr().Get())
    pv=UsdGeom.PrimvarsAPI(prim);arrays=[]
    for name,components in zip(WEIGHT_NAMES,(3,1)):
        attr=pv.GetPrimvar(name)
        if not attr or attr.GetInterpolation() not in ('vertex','varying') or attr.IsIndexed():
            raise ValueError('Source weights require unindexed per-vertex native attributes')
        a=np.asarray(attr.Get(),dtype=np.float32).reshape(-1,components)
        if len(a)!=count:raise ValueError('Native weight and point counts disagree')
        arrays.append(a)
    weights=np.concatenate(arrays,axis=1);proof=validate_weights(weights)
    if proof!=source['source_weights']:raise ValueError('Native USD source weights differ from actual Blender output')
    matrix=np.asarray(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()),float)
    points=np.asarray(mesh.GetPointsAttr().Get(),dtype=np.float32);h=hashlib.sha256()
    for start in range(0,len(points),CHUNK):
        world=points[start:start+CHUNK].astype(float)@matrix[:3,:3]+matrix[3,:3]
        h.update(np.ascontiguousarray(world,dtype='<f4').tobytes())
    if h.hexdigest()!=source['world_points_float32_sha256']:raise ValueError('Native USD points differ from source-weight geometry')
    material=UsdShade.Material.Define(stage,'/IsaacMinMaterials/SharedOriginalPBR');shader=UsdShade.Shader.Define(stage,material.GetPath().AppendChild('Shader'))
    shader.CreateImplementationSourceAttr().Set(UsdShade.Tokens.sourceAsset)
    relative=lambda p:Sdf.AssetPath('./'+Path(os.path.relpath(p,scene.parent)).as_posix())
    shader.SetSourceAsset(relative(module),'mdl');shader.SetSourceAssetSubIdentifier(data['source_asset_subidentifier'],'mdl')
    shader.CreateOutput('out',Sdf.ValueTypeNames.Token);material.CreateSurfaceOutput('mdl').ConnectToSource(shader.ConnectableAPI(),'out')
    mp=material.GetPrim();mp.CreateAttribute('isaacmin:scanDependencies',Sdf.ValueTypeNames.AssetArray,custom=True).Set([relative(library/f['path']) for f in data['textures']])
    mp.CreateAttribute('isaacmin:materialFamilies',Sdf.ValueTypeNames.StringArray,custom=True).Set(list(REQUIRED))
    mp.CreateAttribute('isaacmin:physicalRepeatMetres',Sdf.ValueTypeNames.FloatArray,custom=True).Set([next(f['repeat_m'] for f in data['textures'] if f['asset_id']==n) for n in REQUIRED])
    mp.CreateAttribute('isaacmin:manifest',Sdf.ValueTypeNames.Asset,custom=True).Set(relative(manifest))
    mp.CreateAttribute('isaacmin:recipe',Sdf.ValueTypeNames.String,custom=True).Set(RECIPE)
    binding=UsdShade.MaterialBindingAPI.Apply(prim);binding.Bind(material)
    # Existing per-face natural/structural ownership stays in provenance. Every
    # face uses the same continuous shader and its independently verified weights.
    for subset in binding.GetMaterialBindSubsets():UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(material)
    prim.CreateAttribute('isaacmin:materialWeightOrder',Sdf.ValueTypeNames.StringArray,custom=True).Set(list(REQUIRED))
    prim.CreateAttribute('isaacmin:sourceWeightsManifest',Sdf.ValueTypeNames.Asset,custom=True).Set(relative(Path(source_weights_manifest).resolve()))
    stage.GetRootLayer().Save()
    return {'status':'native_shared_material_bound','qualification':'not_run','scene_sha256':digest(scene),'ground_prim':str(prim.GetPath()),
            'material_prim':str(material.GetPath()),'verified_source_weights':proof,'world_points_float32_sha256':h.hexdigest(),
            'material_manifest_sha256':digest(manifest),'source_weights_manifest_sha256':digest(source_weights_manifest),
            'dependency_closure':'Caller must recompute complete native closure after this authoring step'}
