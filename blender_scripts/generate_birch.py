"""Pinned Sapling white-birch branching with real serrated curved leaf geometry."""
import ast
import bpy
import hashlib
import json
import math
import sys
import numpy as np
from pathlib import Path
from mathutils import Vector

request_path=Path(sys.argv[sys.argv.index('--')+1]).resolve()
request=json.loads(request_path.read_text())
def file_sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
execution_script_sha256=file_sha(__file__)
input_proof=None
if request.get('input_manifest'):
    input_proof=json.loads(Path(request['input_manifest']).read_text())
    if input_proof['execution_script_sha256']!=execution_script_sha256:
        raise RuntimeError('Frozen canopy execution script changed')
    for item in input_proof['files']:
        if file_sha(item['path'])!=item['sha256']:raise RuntimeError('Frozen canopy input changed')
species=request.get('species','white_birch')
if species not in ('white_birch','oak'):raise RuntimeError('Unsupported procedural species recipe')
folder=Path(request['output']).resolve();folder.mkdir(parents=True,exist_ok=True)
if (folder/'generation.json').exists():raise RuntimeError('Canopy generation is immutable; use a new candidate directory')
addon=Path(request['sapling_addon']).resolve()
sys.path.insert(0,str(addon.parent))
import add_curve_sapling
add_curve_sapling.register()
preset=addon/'presets/white_birch.py'
settings=ast.literal_eval('\n'.join(line for line in preset.read_text().splitlines() if line and not line.startswith('#')))
bpy.ops.wm.read_factory_settings(use_empty=True)
scene=bpy.context.scene
scene.unit_settings.system='METRIC';scene.unit_settings.scale_length=1
maps=Path(request['maps']).resolve()
material_specs=json.loads((maps/'materials.json').read_text())['materials'] if (maps/'materials.json').exists() else None


def channel(prefix,role,required=False):
    if material_specs is None:
        path=maps/(prefix+'_'+role+'.png')
    else:
        relative=material_specs[prefix].get(role)
        if relative is None:
            if required:raise RuntimeError('Missing original material channel: '+prefix+'/'+role)
            return None
        path=(maps/relative).resolve()
        if not path.is_relative_to(maps):raise RuntimeError('Material channel escapes the immutable map directory')
    if not path.is_file():
        if required:raise RuntimeError('Missing original material file: '+prefix+'/'+role)
        return None
    return path


def material(name, prefix, leaf=False):
    mat=bpy.data.materials.new(name);mat.use_nodes=True;mat.use_backface_culling=False
    bsdf=mat.node_tree.nodes.get('Principled BSDF')
    for role,socket in [('base_color','Base Color'),('roughness','Roughness')]:
        path=channel(prefix,role,required=role=='base_color' or material_specs is None)
        if path is None:
            roughness=float(material_specs[prefix]['roughness_value'])
            if not 0<roughness<=1:raise RuntimeError('Invalid explicitly inferred leaf roughness')
            bsdf.inputs[socket].default_value=roughness
            mat['isaacmin_roughness_provenance']='inferred botanical candidate; not a measured texture'
            continue
        tex=mat.node_tree.nodes.new('ShaderNodeTexImage');tex.image=bpy.data.images.load(str(path),check_existing=True)
        tex.image.colorspace_settings.name='sRGB' if role=='base_color' else 'Non-Color';tex.image.pack()
        mat.node_tree.links.new(tex.outputs['Color'],bsdf.inputs[socket])
    path=channel(prefix,'normal',required=material_specs is None)
    if path is not None:
        tex=mat.node_tree.nodes.new('ShaderNodeTexImage');tex.image=bpy.data.images.load(str(path),check_existing=True)
        tex.image.colorspace_settings.name='Non-Color';tex.image.pack()
        normal=mat.node_tree.nodes.new('ShaderNodeNormalMap');mat.node_tree.links.new(tex.outputs['Color'],normal.inputs['Color']);mat.node_tree.links.new(normal.outputs['Normal'],bsdf.inputs['Normal'])
    if leaf:
        bsdf.inputs['Subsurface Weight'].default_value=.06
        bsdf.inputs['Subsurface Radius'].default_value=(.001,.002,.0006)
        bsdf.inputs['Transmission Weight'].default_value=.08
        bsdf.inputs['IOR'].default_value=1.42
        opacity=channel(prefix,'opacity')
        if opacity is not None:
            tex=mat.node_tree.nodes.new('ShaderNodeTexImage');tex.image=bpy.data.images.load(str(opacity),check_existing=True)
            tex.image.colorspace_settings.name='Non-Color';tex.image.pack()
            mat.node_tree.links.new(tex.outputs['Color'],bsdf.inputs['Alpha'])
    return mat


prefix='oak' if species=='oak' else 'birch'
bark=material(species+'_Bark',prefix+'_bark')
twig_bark=material('Procedural_WhiteBirch_YoungTwigs','birch_twig') if species=='white_birch' else bark
photo_profiles=json.loads((maps/'leaf_profiles.json').read_text()) if (maps/'leaf_profiles.json').exists() else None
if photo_profiles and any('material_prefix' in profile for profile in photo_profiles):
    material_prefixes=list(dict.fromkeys(profile['material_prefix'] for profile in photo_profiles))
    for profile in photo_profiles:
        if profile['material_index']!=material_prefixes.index(profile['material_prefix']):
            raise RuntimeError('Photographed leaf profile and material indices disagree')
    leaves_mats=[material(species+'_'+str(i)+'_Leaf',name,True) for i,name in enumerate(material_prefixes)]
else:
    leaves_mats=[material(species+'_Leaf',prefix+'_leaf',True)]
records=[]
recipe=request.get('recipe','legacy_sapling_candidate')
if recipe not in ('legacy_sapling_candidate','deciduous_crown_v2'):
    raise RuntimeError('Unknown bounded botanical construction recipe')
for specification in request.get('variants',[[1729,10],[1730,14],[1731,18]]):
    if isinstance(specification,dict):
        seed,scale=specification['seed'],specification['height_m']
        form=specification['form']
    else:
        seed,scale=specification;form='mature'
    if form not in ('young','mature','open_grown') or not 4<=scale<=22:
        raise RuntimeError('Botanical size/form outside the recorded temperate candidate recipe')
    prior=set(bpy.data.objects)
    params=dict(settings)
    params.update(seed=seed,scale=scale,scaleV=scale*.1,showLeaves=True,leafShape='rect',leafScale=.075,
                  leafScaleX=.70,leafScaleV=.25,leaves=int(180*(scale/10)**2),bevel=True,bevelRes=3,resU=3,
                  levels=4,branches=(0,50,25,6),curveRes=(12,9,5,3),
                  useArm=False,armAnim=False,leafAnim=False,makeMesh=False)
    if species=='oak':
        params.update(shape='2',baseSize=.25,baseSplits=2,ratio=.025,rootFlare=1.4,
                      length=(1,.44,.42,.32),downAngle=(0,55,55,45),attractUp=(0,.15,.2,0),
                      curve=(0,20,25,0),leafScale=.12,leaves=int(65*(scale/10)**2),
                      customShape=(.5,1,.45,.5),splitAngle=(25,30,20,0))
    if recipe=='deciduous_crown_v2':
        # Separate branching architectures, not scaled copies. More leaf-bearing
        # twigs distribute actual leaves through the crown instead of increasing
        # overlapping leaves on the same sparse terminal shoots.
        params.update(baseSize_s=.35,leafDist='4',leafScaleT=.12,leafRotateV=18,
                      horzLeaves=False,leafDownAngle=45,leafDownAngleV=-25,
                      segSplits=(.12,.08,.04,0),curveV=(40,65,70,25),
                      branches=(0,52,24,14),shapeS='4',
                      curveRes=(12,9,6,4),lengthV=(.04,.22,.28,.20))
        if species=='oak':
            params.update(baseSplits=1,splitHeight=.28,baseSize=.22,ratio=.022,
                          length=(1,.38,.54,.60),leaves=145,leafScale=.115,
                          curve=(0,25,15,8),attractUp=(0,.1,.05,0))
        else:
            params.update(baseSplits=0,baseSize=.28,ratio=.014,
                          shape='7',length=(1,.29,.48,.60),leaves=125,
                          leafScale=.07,leafScaleX=.72,curve=(0,-22,-18,-10),
                          attractUp=(0,-.35,-.35,-.2),minRadius=.0009)
        if form=='young':
            params.update(baseSplits=0,branches=(0,38,20,12),baseSize=.19,
                          ratio=.013,leaves=100,curveV=(30,55,65,20))
        elif form=='open_grown':
            params.update(branches=(0,58,28,16),baseSize=.16,
                          length=(1,.43 if species=='oak' else .34,.55,.62),
                          baseSplits=2 if species=='oak' else 0)
    bpy.ops.curve.tree_add(**params)
    created=set(bpy.data.objects)-prior
    trunk=next(o for o in created if o.type=='CURVE')
    leaf_obj=next(o for o in created if o.type=='MESH')
    trunk.name=f'{species}_{seed}_trunk';leaf_obj.name=f'{species}_{seed}_leaves'
    trunk.data.materials.append(bark);trunk.data.materials.append(twig_bark)
    for spline in trunk.data.splines:
        max_radius=max((p.radius for p in spline.bezier_points),default=0)
        spline.material_index=1 if max_radius<.03 else 0
    verts=[];faces=[];uvs=[];face_materials=[]
    # Sapling's rect leaf frames are geometry input only. Replace every quad with
    # a tapered, serrated surface whose folds and edges remain visible in depth.
    original=leaf_obj.data
    for leaf_index,poly in enumerate(original.polygons):
        if len(poly.vertices)!=4: raise RuntimeError('Unexpected Sapling leaf frame schema')
        a,b,c,d=[original.vertices[i].co.copy() for i in poly.vertices]
        base=(a+d)*.5;longitudinal=(b+c-a-d)*.5;across=a-d
        normal=across.cross(longitudinal).normalized()
        start=len(verts);steps=len(photo_profiles[leaf_index%len(photo_profiles)]['rows'])-1 if photo_profiles else 22
        for j in range(steps+1):
            t=j/steps
            # Broad basal shoulder, pointed terminal apex, alternating double-serrate margin.
            envelope=(math.sin(math.pi*t)**.82)*(1-.38*t)
            serration=1.0 if j%2==0 else .89
            width=envelope*serration*.5
            for k in (-1,0,1):
                fold=(.0015*math.sin(math.pi*t)*(1-abs(k)) + .0025*math.sin(t*math.pi*1.2))
                if photo_profiles:
                    profile=photo_profiles[leaf_index%len(photo_profiles)]['rows'][j]['vertices'][k+1]
                    verts.append(base+longitudinal*t+across.normalized()*(profile[0]*longitudinal.length)+normal*fold)
                    uvs.append((profile[1],profile[2]))
                else:
                    verts.append(base+longitudinal*t+across*(k*width)+normal*fold)
                    uvs.append((.5+k*width,t))
        for j in range(steps):
            for k in range(2):
                aidx=start+j*3+k
                if j==0:
                    faces.append((aidx,aidx+4,aidx+3))
                elif j==steps-1:
                    faces.append((aidx,aidx+1,aidx+3))
                else:
                    faces.append((aidx,aidx+1,aidx+4,aidx+3))
                face_materials.append(photo_profiles[leaf_index%len(photo_profiles)].get('material_index',0) if photo_profiles else 0)
    mesh=bpy.data.meshes.new(leaf_obj.name+'_serrated_geometry');mesh.from_pydata(verts,[],faces);mesh.update()
    layer=mesh.uv_layers.new(name='BotanicalLeafUV')
    for p in mesh.polygons:
        p.use_smooth=True
        for li in p.loop_indices:layer.data[li].uv=uvs[mesh.loops[li].vertex_index]
    leaf_obj.data=mesh
    for leaf_material in leaves_mats:mesh.materials.append(leaf_material)
    mesh.polygons.foreach_set('material_index',np.asarray(face_materials,dtype=np.int32))
    # The native mesh owns these arrays now. Do not retain tens of millions of
    # Python vectors while the trunk is converted or another variant is built.
    del verts,faces,uvs,face_materials
    bpy.ops.object.select_all(action='DESELECT');trunk.select_set(True);bpy.context.view_layer.objects.active=trunk
    if hasattr(trunk.data,'use_uv_as_generated'):trunk.data.use_uv_as_generated=True
    bpy.ops.object.convert(target='MESH');trunk=bpy.context.object
    if not trunk.data.uv_layers:
        if recipe=='deciduous_crown_v2':
            raise RuntimeError('Metric bark needs the actual native bevel-ring parameterization')
        # Projection records physical cylinder circumference and height, with no baked lighting.
        layer=trunk.data.uv_layers.new(name='PhysicalBarkUV')
        for p in trunk.data.polygons:
            for li in p.loop_indices:
                co=trunk.data.vertices[trunk.data.loops[li].vertex_index].co
                layer.data[li].uv=(math.atan2(co.y,co.x)/(2*math.pi),co.z/2)
    bark_uv_evidence=None
    if recipe=='deciduous_crown_v2':
        from isaacmin.assets.bark_uv import metric_bark_uv
        import isaacmin.assets.bark_uv as bark_uv_module
        positions=np.empty((len(trunk.data.vertices),3),np.float32)
        edge_indices=np.empty((len(trunk.data.edges),2),np.int32)
        loop_indices=np.empty(len(trunk.data.loops),np.int32)
        native_uv=np.empty((len(trunk.data.loops),2),np.float32)
        trunk.data.vertices.foreach_get('co',positions.ravel())
        trunk.data.edges.foreach_get('vertices',edge_indices.ravel())
        trunk.data.loops.foreach_get('vertex_index',loop_indices)
        trunk.data.uv_layers.active.data.foreach_get('uv',native_uv.ravel())
        metric_uv,bark_uv_evidence=metric_bark_uv(positions,edge_indices,loop_indices,native_uv,
                                               request.get('bark_repeat_m',[.6,.6]))
        trunk.data.uv_layers.active.data.foreach_set('uv',metric_uv.ravel())
        trunk.data.uv_layers.active.name='MetricBarkUV'
        bark_uv_evidence['producer_sha256']=file_sha(bark_uv_module.__file__)
        del positions,edge_indices,loop_indices,native_uv,metric_uv
    for p in trunk.data.polygons:p.use_smooth=True
    leaf_obj.parent=None
    # Keep the set as two material-bearing meshes with a shared root frame.
    objs=[trunk,leaf_obj]
    lo=min(v.co.z for o in objs for v in o.data.vertices)
    for o in objs:
        for v in o.data.vertices:v.co.z-=lo
    bpy.context.view_layer.update()
    bottom_band=[list(v.co) for v in trunk.data.vertices if v.co.z<.005]
    stride=max(1,len(bottom_band)//32)
    finite=True
    for obj in objs:
        coords=np.empty((len(obj.data.vertices),3),np.float32)
        obj.data.vertices.foreach_get('co',coords.ravel())
        finite=finite and bool(np.isfinite(coords).all())
    if not finite or len(bottom_band)<3:raise RuntimeError('Canopy needs finite geometry and measured root anchors')
    records.append({'seed':seed,'height_parameter_m':scale,'form':form,'recipe':recipe,
                    'bark_uv':bark_uv_evidence,
                    'recipe_parameters':params,'objects':[o.name for o in objs],
                    'vertices':sum(len(o.data.vertices) for o in objs),'polygons':sum(len(o.data.polygons) for o in objs),
                    'leaf_count':len(original.polygons),'leaf_area_m2':sum(p.area for p in mesh.polygons),
                    'leaf_geometry':'Original photographed leaf opacity silhouette with curved midrib' if photo_profiles else '22 longitudinal strips, paired curved surfaces, serrated margins',
                    'bounds_m':[[min(v.co[i] for o in objs for v in o.data.vertices) for i in range(3)],
                                [max(v.co[i] for o in objs for v in o.data.vertices) for i in range(3)]],
                    'contact_anchor_local_m':[0,0,0], 'contact_anchors_local_m':bottom_band[::stride][:32],
                    'numeric_geometry':{'finite':finite,'all_meshes_have_uv':all(bool(o.data.uv_layers) for o in objs),
                                        'native_anchor_count':len(bottom_band[::stride][:32])},
                    'contact_qualification':'not_run'})
for text in list(bpy.data.texts):bpy.data.texts.remove(text)
blend=folder/(species+'_variants.blend')
bpy.ops.wm.save_as_mainfile(filepath=str(blend),compress=True)
report={'schema_version':1,'asset_id':'procedural_'+species,'classification':'procedurally_generated_candidate_geometry',
        'generator_script_sha256':execution_script_sha256,
        'request_sha256':file_sha(request_path),'input_manifest':request.get('input_manifest'),
        'input_manifest_sha256':file_sha(request['input_manifest']) if input_proof else None,
        'recipe':recipe,'output_sha256':file_sha(blend),
        'recipe_species':species,'recipe_parameters':params,
        'generator':'Blender Sapling Tree Gen','generator_version':add_curve_sapling.bl_info['version'],
        'generator_licence':'GPL-2.0-or-later','generator_source':'https://projects.blender.org/blender/blender-addons/src/branch/blender-v4.0-release/add_curve_sapling',
        'source_hashes':{str(p.relative_to(addon)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [addon/'__init__.py',addon/'utils.py',preset]},
        'blender_version':bpy.app.version_string,'output_blend':str(blend),'variants':records,
        'maps_manifest':str(maps/'maps_manifest.json'),'isaac_qualification':'not_run','status':'external_tool_verified',
        'limitations':['Unqualified botanical appearance; compare against licensed actual deciduous-woodland photographs',
                       'Thin-leaf transmission needs explicit Isaac material translation and backlight test',
                       'Root flare is Sapling geometry; close-ground bark peeling and exposed roots require further refinement',
                       'Species branching/leaf-scale parameters are inferred botanical candidates, not reconstructed source-tree anatomy']}
if execution_script_sha256!=file_sha(__file__):raise RuntimeError('Canopy execution script changed while running')
if input_proof:
    for item in input_proof['files']:
        if file_sha(item['path'])!=item['sha256']:raise RuntimeError('Canopy input changed while running')
(folder/'generation.json').write_text(json.dumps(report,indent=2))
