"""Immutable native terrain reuse; cache identity never implies qualification."""
from __future__ import annotations
import hashlib,json,os,shutil,tempfile
from pathlib import Path

PRODUCERS=(
    'src/isaacmin/terrain/naturalize_exterior.py',
    'blender_scripts/export_scene.py','blender_scripts/mesh_arrays.py',
    'src/isaacmin/assembly/material_assignment.py','src/isaacmin/assembly/material_bake.py',
    'src/isaacmin/assembly/water.py','src/isaacmin/assembly/pipeline.py',
    'src/isaacmin/assembly/native_cache.py',
    'src/isaacmin/assembly/material_mdl.py','src/isaacmin/assembly/material_recipe.py',
    'recipes/materials/shared_original.mdl.in',
    'src/isaacmin/assembly/foliage_mdl.py','src/isaacmin/assembly/foliage_recipe.py',
    'recipes/materials/thin_leaf.mdl.in',
    'src/isaacmin/assembly/credits.py','ASSET_CREDITS.md',
    'blender_scripts/precision_mesh.py','scripts/native_precision_worker.py',
    'src/isaacmin/volumes/native_precision.py','src/isaacmin/volumes/precision_algorithms.py',
    'src/isaacmin/volumes/planar_patch.py',
    'src/isaacmin/volumes/projected_patch.py','src/isaacmin/volumes/intersection_repair.py',
    'src/isaacmin/validation/global_intersections.py',
    'scripts/native/triangle_intersection_validator.cpp',
    'blender_scripts/precise_subdivision.py','src/isaacmin/volumes/plane_conditioning.py',
    'scripts/bootstrap_precision.py','scripts/native/subdivide_double.cpp','scripts/native/write_native_obj.cpp',
    '.tools/native_precision/build_manifest.json','src/isaacmin/process.py',
    'artifacts/bootstrap/dependency-lock.json',
)
ROLES=('base_color','roughness','normal','height','metallic','opacity')
PRECISION_FILES=('native_precision/final_vertices.npy','native_precision/final_triangles.npy',
                 'native_precision/finalization.json')

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()

def canonical(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def terrain_payload(mesh):
    """Hash actual native arrays and material graph without directory dependence."""
    import numpy as np
    import bpy
    result={'vertices':len(mesh.vertices),'triangles':len(mesh.polygons),'arrays':{},'materials':[],
            'uv_active_index':mesh.uv_layers.active_index,'uv_active_render':[u.name for u in mesh.uv_layers if u.active_render]}
    def array(name,collection,field,width,dtype):
        values=np.empty(len(collection)*width,dtype=dtype);collection.foreach_get(field,values)
        result['arrays'][name]=hashlib.sha256(memoryview(values)).hexdigest()
    array('positions',mesh.vertices,'co',3,np.float32)
    array('vertex_normals',mesh.vertices,'normal',3,np.float32)
    array('triangle_indices',mesh.loops,'vertex_index',1,np.int32)
    array('face_material_indices',mesh.polygons,'material_index',1,np.int32)
    array('face_smooth_flags',mesh.polygons,'use_smooth',1,np.bool_)
    for uv in mesh.uv_layers:array('uv:'+uv.name,uv.data,'uv',2,np.float32)
    for name,field,width,dtype in [('IsaacMinMaterialWeights012','vector',3,np.float32),
        ('IsaacMinMaterialWeight3','value',1,np.float32),('IsaacMinSourceOwnership','value',1,np.int32)]:
        attribute=mesh.attributes.get(name)
        if attribute:array(name,attribute.data,field,width,dtype)
    def value(x):
        if x is None or isinstance(x,(str,bool,int,float)):return x
        try:return [value(y) for y in x]
        except (TypeError,RecursionError):return None
    for material in mesh.materials:
        if not material or not material.use_nodes:raise ValueError('Cache needs explicit native material nodes')
        nodes=[]
        for node in material.node_tree.nodes:
            props={}
            for prop in node.bl_rna.properties:
                name=prop.identifier
                if prop.is_readonly or name in {'name','label','location','width','height','dimensions','parent','color','select','show_options','show_preview','show_texture','hide'}:continue
                if prop.type in {'BOOLEAN','INT','FLOAT','STRING','ENUM'}:props[name]=value(getattr(node,name))
            entry={'name':node.name,'type':node.bl_idname,'properties':props,
                   'inputs':[{'identifier':socket.identifier,'value':value(socket.default_value) if hasattr(socket,'default_value') else None} for socket in node.inputs]}
            if node.type=='GROUP':raise ValueError('Ground node groups require an explicit cache graph protocol')
            image=getattr(node,'image',None)
            if image:
                digest=hashlib.sha256(bytes(image.packed_file.data)).hexdigest() if image.packed_file else sha(bpy.path.abspath(image.filepath))
                entry['image']={'sha256':digest,'color_space':image.colorspace_settings.name,'alpha_mode':image.alpha_mode}
            nodes.append(entry)
        result['materials'].append({'name':material.name,'nodes':nodes,
            'links':[[l.from_node.name,l.from_socket.identifier,l.to_node.name,l.to_socket.identifier] for l in material.node_tree.links]})
    return {'sha256':canonical(result),'value':result}

def input_identity(request,root):
    """Only population/output paths differ between first and second export."""
    root=Path(root).resolve()
    identity={key:request.get(key) for key in (
        'terrain_translation','minecraft_origin','texture_repeat_m','geometry_detail',
        'terrain_material_recipe')}
    files={key:{'sha256':sha(request[key]),'bytes':Path(request[key]).stat().st_size}
           for key in ('terrain_mesh','source_surface','support_manifest','source_water_mesh','exterior_naturalization') if request.get(key)}
    delta=request.get('exterior_delta')
    identity['exterior_delta']=None if not delta else {k:v for k,v in delta.items() if k!='path'}
    if delta:files['exterior_delta']={'sha256':sha(delta['path']),'bytes':Path(delta['path']).stat().st_size}
    identity['files']=files
    identity['materials']=[]
    for spec in request['materials']:
        copied={k:v for k,v in spec.items() if k not in ROLES}
        copied['maps']={role:{'sha256':sha(spec[role]),'bytes':Path(spec[role]).stat().st_size}
                        for role in ROLES if spec.get(role)}
        identity['materials'].append(copied)
    identity['producers']={name:sha(root/name) for name in PRODUCERS}
    identity['native_blender_sha256']=sha(root/'.tools/blender/blender')
    return {'sha256':canonical(identity),'value':identity}

def _owned(directory,relative):
    directory=Path(directory).resolve();path=directory/relative
    if path.is_symlink() or not path.resolve().is_relative_to(directory):
        raise ValueError('Cache entry escapes its immutable directory')
    if not path.is_file():raise ValueError('Cache entry missing: '+str(relative))
    return path

def _precision_evidence(directory,export,recorded_files=None):
    """Bind the declared finalization to exact array files, without qualifying it."""
    declared=export.get('native_precision_finalization')
    if declared is None:
        return {'status':'not_declared','files':[],'qualification':'not_inferred'}
    if not isinstance(declared,dict):raise ValueError('Native precision finalization must be an explicit record')
    tracked=None if recorded_files is None else {item['path']:item for item in recorded_files}
    files=[]
    for name in PRECISION_FILES:
        if tracked is not None and name not in tracked:
            raise ValueError('Bare cache predates recorded native precision evidence: '+name)
        path=_owned(directory,name)
        item={'path':name,'sha256':sha(path),'bytes':path.stat().st_size}
        if tracked is not None and item!=tracked[name]:
            raise ValueError('Native precision cache bytes changed: '+name)
        files.append(item)
    finalization=json.loads(_owned(directory,PRECISION_FILES[2]).read_text())
    if finalization!=declared:
        raise ValueError('Native precision finalization differs from export declaration')
    for key,item in zip(('final_vertices_sha256','final_triangles_sha256'),files[:2]):
        if finalization.get(key)!=item['sha256']:
            raise ValueError('Native precision declared array hash mismatch: '+item['path'])
    return {'status':'verified_original_bytes','files':files,'qualification':'not_inferred'}

def record_bare(request,root,output):
    """Call only after native USD point/index equality and closure verification."""
    output=Path(output).resolve()
    if request.get('assets'):raise ValueError('Only a bare terrain/water scene can become the cache source')
    record=output/'bare_scene_cache.json'
    if record.exists():raise ValueError('Bare cache manifest is immutable')
    export=json.loads((output/'export_result.json').read_text())
    closure=json.loads((output/'native_dependency_closure.json').read_text())
    if export.get('status')!='success' or export.get('native_usd_geometry_verified') is not True or closure.get('status')!='pass':
        raise ValueError('Bare cache requires measured native export and portable closure')
    if export.get('placed_object_names'):raise ValueError('A populated scene is not a bare cache')
    names=set(item['path'] for item in closure['files'])|set(item['path'] for item in export['texture_files'])|{
        'scene.blend','final_ground.obj','export_result.json','native_dependency_closure.json'}
    precision=_precision_evidence(output,export)
    names.update(item['path'] for item in precision['files'])
    files=[]
    for name in sorted(names):
        path=_owned(output,name);files.append({'path':name,'sha256':sha(path),'bytes':path.stat().st_size})
    result={'schema_version':2,'status':'native_geometry_cache_only','input_identity':input_identity(request,root),
            'files':files,'ground_sha256':export['final_ground_sha256'],
            'native_precision_evidence':precision,
            'expected_object_names':['Terrain_FinalGround']+(['SourceSurfaceWater'] if export.get('source_water_interfaces',{}).get('polygons') else []),
            'native_geometry':{k:export[k] for k in ('native_usd_terrain_vertices','native_usd_terrain_triangles')},
            'terrain_payload':export['native_terrain_payload'],
            'water_payload':export.get('native_water_payload'),
            'source_coordinate_frame':{'origin':request['minecraft_origin'],'terrain_translation':request['terrain_translation']},
            'appearance_qualification':'not_inferred_from_cache'}
    record.write_text(json.dumps(result,indent=2)+'\n');return result

def verify_bare(request,root,directory):
    directory=Path(directory).resolve();path=directory/'bare_scene_cache.json'
    record=json.loads(path.read_text())
    if record['input_identity']!=input_identity(request,root):raise ValueError('Bare geometry/material producer or inputs changed')
    for item in record['files']:
        file=_owned(directory,item['path'])
        if file.stat().st_size!=item['bytes'] or sha(file)!=item['sha256']:
            raise ValueError('Bare cache bytes changed: '+item['path'])
    export=json.loads((directory/'export_result.json').read_text())
    precision=_precision_evidence(directory,export,record['files'])
    if 'native_precision_evidence' in record and record['native_precision_evidence']!=precision:
        raise ValueError('Bare cache native precision evidence record changed')
    return {'directory':str(directory),'manifest_sha256':sha(path),'manifest':record,
            'export':export,'native_precision_evidence':precision}

def copy_native_precision_evidence(bare_directory,output,cache_record):
    """Copy only the three verified precision files; never reconstruct old evidence.

    ``cache_record`` is the result of ``verify_bare``. Source and destination bytes
    are checked again here, since copying may occur after a long native operation.
    """
    bare_directory=Path(bare_directory).resolve();output=Path(output).resolve()
    if str(bare_directory)!=cache_record['directory']:
        raise ValueError('Native precision copy source differs from verified cache')
    if output==bare_directory or output.is_relative_to(bare_directory):
        raise ValueError('Native precision copy must not modify the bare cache')
    manifest_path=_owned(bare_directory,'bare_scene_cache.json')
    if sha(manifest_path)!=cache_record['manifest_sha256']:
        raise ValueError('Bare cache manifest changed before precision copy')
    manifest=json.loads(manifest_path.read_text())
    if manifest!=cache_record['manifest']:
        raise ValueError('Native precision copy received a changed cache record')
    export_path=_owned(bare_directory,'export_result.json')
    export_entries=[i for i in manifest['files'] if i['path']=='export_result.json']
    if len(export_entries)!=1 or sha(export_path)!=export_entries[0]['sha256'] or export_path.stat().st_size!=export_entries[0]['bytes']:
        raise ValueError('Bare export declaration changed before precision copy')
    export=json.loads(export_path.read_text())
    if export!=cache_record['export']:
        raise ValueError('Native precision export differs from verified cache')
    precision=_precision_evidence(bare_directory,export,manifest['files'])
    if 'native_precision_evidence' in manifest and manifest['native_precision_evidence']!=precision:
        raise ValueError('Native precision evidence differs from cache declaration')
    if precision['status']=='not_declared':return precision
    destination=output/'native_precision'
    if destination.exists() or destination.is_symlink():raise ValueError('Native precision destination is immutable')
    output.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.native-precision-',dir=output))
    try:
        for item in precision['files']:
            source=_owned(bare_directory,item['path']);target=staging/Path(item['path']).name
            shutil.copyfile(source,target)
            if target.stat().st_size!=item['bytes'] or sha(target)!=item['sha256']:
                raise ValueError('Native precision bytes changed during copy: '+item['path'])
        if destination.exists() or destination.is_symlink():raise ValueError('Native precision destination appeared during copy')
        staging.rename(destination)
    finally:
        if staging.exists():shutil.rmtree(staging)
    return {'status':'copied_verified_bytes','files':precision['files'],'qualification':'not_inferred'}

def deduplicate_generated_textures(output,store):
    """Call AFTER Blender exits. Preserve every encoded byte and in-scene path.

    Store files and successful output links are read-only. Never link provider
    masters or an active Blender destination. Cross-device links use an explicit
    verified copy; portability never depends on the store remaining present.
    """
    output=Path(output).resolve();store=Path(store).resolve();store.mkdir(parents=True,exist_ok=True)
    if store==output or store.is_relative_to(output):raise ValueError('Texture store must be outside portable scene')
    files=sorted({p for d in ('textures','transition_bakes','shared_material') for p in (output/d).rglob('*')
                  if p.is_file() and p.suffix.lower() in {'.png','.jpg','.jpeg','.exr','.tif','.tiff','.hdr'}})
    result=[]
    for path in files:
        _owned(output,str(path.relative_to(output)));digest=sha(path);size=path.stat().st_size
        target=store/digest[:2]/(digest+path.suffix.lower());target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            fd,name=tempfile.mkstemp(prefix='.incoming-',dir=target.parent)
            try:
                with os.fdopen(fd,'wb') as dst,path.open('rb') as src:
                    shutil.copyfileobj(src,dst,1024*1024);dst.flush();os.fsync(dst.fileno())
                if sha(name)!=digest:raise ValueError('Texture copy changed encoded bytes')
                os.chmod(name,0o444)
                try:os.link(name,target)
                except FileExistsError:pass
            finally:Path(name).unlink(missing_ok=True)
        if target.is_symlink() or target.stat().st_size!=size or sha(target)!=digest:
            raise ValueError('Texture content-store collision')
        mode='hardlink' if target.stat().st_dev==path.stat().st_dev else 'verified_copy_cross_device'
        if mode=='hardlink' and target.stat().st_ino!=path.stat().st_ino:
            fd,name=tempfile.mkstemp(prefix='.relink-',dir=path.parent);os.close(fd);Path(name).unlink()
            try:os.link(target,name);os.replace(name,path)
            finally:Path(name).unlink(missing_ok=True)
        if sha(path)!=digest:raise ValueError('Generated texture changed during deduplication')
        result.append({'path':str(path.relative_to(output)),'sha256':digest,'bytes':size,'storage_mode':mode})
    return {'status':'byte_preserving_storage_only','files':result,'unique_encoded_bytes':sum({i['sha256']:i['bytes'] for i in result}.values()),
            'portable_asset_paths_unchanged':True,'renderer_qualification':'not_inferred'}
