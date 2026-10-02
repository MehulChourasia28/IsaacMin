"""Deterministic above-ground conversion entry point, distinct evidence per stage.

Uses the same outdoor constructor for every extent. The initial supported asset
library covers temperate vegetation; other recipe families report their gaps.
The optional UI agent dispatches the fixed stages; the CLI also works offline.
"""
from pathlib import Path
import hashlib,json,math
import numpy as np
from isaacmin.io import atomic_json,read_json,sha256_file,utc_now
from isaacmin.process import run_worker
from isaacmin.adapters.workers import native_environment
from isaacmin.source.snapshot import snapshot_world
from isaacmin.source.macro import extract_macro_surface
from isaacmin.source.surface_objects import extract_surface_objects
from isaacmin.terrain.surface_navigation import build_surface_terrain,SurfaceQuery,source_fields,SurfaceRecipe
from isaacmin.terrain.biomes import biome_inventory
from isaacmin.assembly.surface_population import populate_surface


def source_center(workspace,world,override=None,*,metadata_world=None):
    workspace,world=Path(workspace).resolve(),Path(world).resolve()
    def checked(value):
        center=tuple(map(float,value))
        if len(center)!=2 or not all(math.isfinite(v) for v in center):
            raise ValueError('Source center requires two finite Minecraft X,Z coordinates')
        return center
    if override is not None:return checked(override),'command_line'
    # Presets are project input data, never coordinate branches in construction.
    # A replacement save at the same path must not inherit another world's center.
    registry=workspace/'state/source_worlds.json'
    if registry.is_file():
        entries=read_json(registry)['worlds']
        matches=[entry for entry in entries
            if (workspace/entry['source']).resolve()==world]
        if len(matches)>1:raise ValueError('Duplicate source path in world registry')
        if matches:
            entry=matches[0]
            if sha256_file(Path(metadata_world or world)/'level.dat')==entry['level_dat_sha256']:
                return checked(entry['center_minecraft_xz']),'registered_user_preference:'+entry['id']
    saved=workspace/'state/resolved_project.json'
    if saved.is_file():
        project=read_json(saved)
        expected=project.get('preferred_source_metadata_sha256')
        metadata_matches=expected is None or sha256_file(Path(metadata_world or world)/'level.dat')==expected
        if Path(project['source_world']).resolve()==world.resolve() and metadata_matches:
            return checked(project['centre_minecraft_xz']),'saved_user_preference_for_this_world'
    from isaacmin.source.nbt import load
    data=load(Path(metadata_world or world)/'level.dat');data=data.get('Data',data)
    spawn=data.get('spawn',{}).get('pos',[data.get('SpawnX',0),data.get('SpawnY',0),data.get('SpawnZ',0)])
    return checked((spawn[0],spawn[2])),'level.dat_spawn'


def outdoor_camera_poses(terrain,objects=None,*,seed=70101,maximum=4):
    """Deterministic coverage cameras from actual terrain, not old-world poses."""
    query=SurfaceQuery(terrain);fields=source_fields(Path(terrain)/'material_fields.npz')
    canopy=np.load(Path(objects)/'canopy.npz')['cover_fraction'] if objects else np.zeros(fields['height'].shape)
    ox,oy,oz=query.origin;mx,mz=fields['min_xz'];zz,xx=np.indices(fields['height'].shape)
    x,z=xx+mx+.5,zz+mz+.5
    dry=~(fields['water_validity']&(fields['water_height']>=fields['height']))
    inside=(x>=query.xmin+16)&(x<query.xmax-16)&(z>=query.zmin+16)&(z<query.zmax-16)&dry
    candidates=np.argwhere(inside)
    if not len(candidates):raise ValueError('No covered dry outdoor camera positions')
    rng=np.random.default_rng(seed);rng.shuffle(candidates);poses=[]
    # Assess all coverage positions before choosing a representative from each
    # biome. A steep first sample must not accidentally omit an entire biome.
    rz,rx=candidates.T;wx=x[rz,rx]-ox;wy=-z[rz,rx]+oz
    heights=query.heights(wx,wy)
    slopes=np.hypot(query.heights(wx+.5,wy)-query.heights(wx-.5,wy),query.heights(wx,wy+.5)-query.heights(wx,wy-.5))
    suitable=np.isfinite(heights)&(slopes<=math.tan(math.radians(25)))
    if objects:
        from scipy.spatial import cKDTree
        trees=read_json(Path(objects)/'objects.json')['trees']
        if trees:
            tree_xy=np.array([[t['source_x']-ox,-t['source_z']+oz] for t in trees])
            distances=cKDTree(tree_xy).query(np.column_stack((wx,wy)))[0]
            suitable&=distances>=1.75
    candidates=candidates[suitable]
    if not len(candidates):raise ValueError('No supported camera positions clear of observed trunks')
    def priority(name):
        short=name.removeprefix('minecraft:')
        return (0 if any(t in short for t in ('peaks','snowy_slopes','desert','badlands')) else
                1 if short in ('grove','taiga','swamp','jungle','beach') else 2,name)
    first=[]
    observed,counts=np.unique(fields['biome'][candidates[:,0],candidates[:,1]],return_counts=True)
    coverage=dict(zip(observed,counts))
    for biome in sorted(coverage,key=lambda name:(-coverage[name],priority(name))):
        pool=candidates[fields['biome'][candidates[:,0],candidates[:,1]]==biome]
        first.append(pool[0].tolist())
    peak_mask=inside.copy();peak_mask[:]=False
    peak_mask[candidates[:,0],candidates[:,1]]=True
    peak=np.unravel_index(np.argmax(np.where(peak_mask,fields['height'],-np.inf)),peak_mask.shape)
    peak_xy=np.array([x[peak]-ox,-z[peak]+oz])
    used=set()
    for rz,rx in first+candidates[:50].tolist():
        if (rz,rx) in used:continue
        used.add((rz,rx))
        wx=float(x[rz,rx]-ox);wy=float(-z[rz,rx]+oz);height=float(query.heights(wx,wy))
        slope=np.hypot(query.heights(wx+.5,wy)-query.heights(wx-.5,wy),query.heights(wx,wy+.5)-query.heights(wx,wy-.5))
        if not np.isfinite(height) or slope>math.tan(math.radians(25)):continue
        angle=float(rng.uniform(0,2*math.pi));direction=np.array([math.cos(angle),math.sin(angle)])
        if fields['height'][peak]-fields['height'][rz,rx]>12:
            delta=peak_xy-np.array([wx,wy]);direction=delta/np.linalg.norm(delta)
        cover=float(np.clip((canopy[rz,rx]-.15)/.45,0,1));cover=cover*cover*(3-2*cover)
        ev=13.747247562465875-4.5*cover
        poses.append(dict(kind='static',position=[wx,wy,height+.6],
            look_at=[wx+direction[0]*8,wy+direction[1]*8,height+.4],
            scout_height_m=.6,held_out=False,surface_scope='outdoor',source_biome=str(fields['biome'][rz,rx]),
            camera_response=dict(ev100=ev,f_number=2.8 if ev<11 else 8.,iso=100,
                policy='recorded camera response to source canopy shade; no scene light or bitmap alteration',
                source_canopy_cover=float(canopy[rz,rx]))))
        if len(poses)==maximum:break
    if not poses:raise ValueError('No valid ground-supported outdoor cameras')
    # Water is a required source feature. Biome representatives alone can all
    # face inland, leaving its shoreline and optical material uninspected.
    from scipy import ndimage
    wet=~dry
    if wet.any():
        distance,nearest=ndimage.distance_transform_edt(~wet,return_indices=True)
        rz,rx=candidates.T;nz,nx=nearest[:,rz,rx]
        ground=query.heights(x[rz,rx]-ox,-z[rz,rx]+oz)
        level=fields['water_height'][nz,nx]-oy
        bank=(distance[rz,rx]>=1.5)&(distance[rz,rx]<=5.)&(ground>=level)&(ground<=level+2.)
        ids=np.flatnonzero(bank)
        if len(ids):
            bz,bx=rz[ids],rx[ids]
            direction=np.column_stack((nx[ids]-bx,nz[ids]-bz)).astype(float)
            direction/=np.linalg.norm(direction,axis=1,keepdims=True)
            steps=np.arange(2.,26.,2.)
            cx=np.rint(bx[:,None]+direction[:,0,None]*steps).astype(int)
            cz=np.rint(bz[:,None]+direction[:,1,None]*steps).astype(int)
            covered=(cx>=0)&(cz>=0)&(cx<wet.shape[1])&(cz<wet.shape[0])
            cx=np.clip(cx,0,wet.shape[1]-1);cz=np.clip(cz,0,wet.shape[0]-1)
            coverage=(wet[cz,cx]&covered).sum(axis=1)
            pick=int(np.argmax(coverage));i=ids[pick];rz0,rx0=int(rz[i]),int(rx[i])
            wx,wy=float(x[rz0,rx0]-ox),float(-z[rz0,rx0]+oz)
            dx,dz=direction[pick];cover=float(np.clip(canopy[rz0,rx0],0,1))
            shade=float(np.clip((cover-.15)/.45,0,1));shade=shade*shade*(3-2*shade)
            ev=13.747247562465875-4.5*shade
            poses.append(dict(kind='static',name='source_water_shore',
                position=[wx,wy,float(ground[i])+.6],look_at=[wx+dx*12,wy-dz*12,float(level[i])+.4],
                scout_height_m=.6,held_out=False,surface_scope='outdoor_water',
                source_biome=str(fields['biome'][rz0,rx0]),
                selection='supported dry bank with greatest sampled water coverage within24m; no image-score selection',
                camera_response=dict(ev100=ev,f_number=2.8 if ev<11 else 8.,iso=100,
                    source_canopy_cover=cover,policy='same recorded source-canopy response')))
    from isaacmin.outdoor_feature_views import ice_spire_ground_pose
    ice_detail=ice_spire_ground_pose(terrain)
    if ice_detail is not None:poses.insert(0,ice_detail)
    return poses


def outdoor_overview_poses(terrain):
    """Aerial progress views derived from bounds, separate from robot evidence."""
    query=SurfaceQuery(terrain)
    width=query.xmax-query.xmin;depth=query.zmax-query.zmin
    span=max(width,depth)
    center=np.array([(query.xmin+query.xmax)/2-query.origin[0],
        -(query.zmin+query.zmax)/2+query.origin[2],float(np.median(query.height))])
    ceiling=float(np.max(query.height))+32.
    # A small horizontal offset avoids the look-at/up-axis singularity of an
    # exactly vertical camera. All frames retain the full native render recipe.
    views=[('high_overview',0.,-.12,1.08),
           ('oblique_southwest',-.8,-.8,.65),
           ('oblique_northeast',.8,.65,.60)]
    return [dict(kind='aerial_overview',name=name,
        position=[center[0]+dx*span,center[1]+dy*span,ceiling+dz*span],
        look_at=center.tolist(),held_out=False,overview_only=True,
        scope='actual current USD region; not full save coverage or robot-height evidence',
        region_extent_m=[width,depth],source_bounds_xz=list(query.record['bounds_source_xz']))
        for name,dx,dy,dz in views]


def build_outdoor(workspace,world=None,*,extent=256,center=None,output=None,capture=True,asset_library=None,capture_set='full',stage_controller=None):
    workspace=Path(workspace).resolve()
    if capture_set not in ('full','focus'):raise ValueError('Unknown development capture set')
    if stage_controller and (not capture or capture_set!='full'):
        raise ValueError('Agent conversions require the complete capture plan')
    project=read_json(workspace/'state/resolved_project.json') if (workspace/'state/resolved_project.json').is_file() else {}
    world=Path(world or project.get('source_world',workspace/'IsaacMin-MinecraftWorld')).resolve()
    if extent<32 or extent>2048 or extent%16:raise ValueError('Extent must be a multiple of16 in [32,2048]')
    # Archives and directories enter the same immutable source path. Read spawn
    # metadata only after extraction, while matching saved preferences against
    # the original user input path rather than the content-addressed snapshot.
    snapshot=snapshot_world(world,workspace/'work/snapshots')
    metadata_world=Path(snapshot['snapshot_path'])
    centre,centre_origin=source_center(workspace,world,center,metadata_world=metadata_world)
    # Every crop from a save shares this anchor. A --center change moves mesh
    # coordinates, not texture phases; large absolute Minecraft coordinates do
    # not enter the renderer's float32 material arithmetic directly.
    material_centre,_=source_center(workspace,world,metadata_world=metadata_world)
    material_origin=(material_centre[0],0.,material_centre[1])
    selection=workspace/'state/default_outdoor_assets.json'
    selected=read_json(selection)['manifest'] if selection.is_file() else 'state/surface_asset_library.json'
    library_path=Path(asset_library).resolve() if asset_library else workspace/selected
    if not library_path.is_file():raise ValueError('Native outdoor asset library has not been prepared')
    library=read_json(library_path)
    asset_dependencies={'scene':sha256_file(workspace/library['scene']/'native_dependency_closure.json'),
        'normalized_asset_registry':sha256_file(workspace/'state/normalized_assets.json'),
        'ground_materials':sha256_file(workspace/library['material_manifest']),
        'capture_configuration':sha256_file(workspace/library['capture_template'])}
    for path in library.get('canopy_libraries',[])+library.get('scenery_libraries',[])+library.get('shrub_libraries',[])+library.get('litter_libraries',[]):
        asset_dependencies[path]=sha256_file(workspace/path)
    if library.get('canopy_rules'):
        asset_dependencies['canopy_rules']=sha256_file(workspace/library['canopy_rules'])
    identity=dict(source=snapshot['save_sha256'],center=centre,extent=extent,recipe=SurfaceRecipe().__dict__,
        material_origin_source_xyz=material_origin,
        asset_library_sha256=sha256_file(library_path),asset_dependencies=asset_dependencies,
        scope='user_authorized_outdoor_no_structures',
        producer_sha256={name:sha256_file(workspace/name) for name in (
            'src/isaacmin/outdoor_pipeline.py','src/isaacmin/terrain/surface_navigation.py',
            'src/isaacmin/outdoor_routes.py','src/isaacmin/terrain/routes.py',
            'src/isaacmin/outdoor_feature_views.py',
            'src/isaacmin/terrain/surface_sampling.py','src/isaacmin/assembly/surface_population.py',
            'src/isaacmin/terrain/surface_partitions.py',
            'src/isaacmin/terrain/biomes.py','src/isaacmin/assembly/surface_materials.py',
            'src/isaacmin/assembly/frozen_surface.py',
            'src/isaacmin/assembly/surface_scenery.py','src/isaacmin/assembly/surface_canopy.py',
            'src/isaacmin/assembly/surface_shrubs.py',
            'src/isaacmin/assembly/surface_litter.py',
            'src/isaacmin/assembly/surface_ground.py',
            'src/isaacmin/assembly/surface_instances.py',
            'src/isaacmin/assembly/surface_water.py',
            'src/isaacmin/assembly/planar_water.py',
            'src/isaacmin/assembly/outdoor_lighting.py','src/isaacmin/assembly/living_grass.py',
            'src/isaacmin/source/macro.py','src/isaacmin/source/anvil.py',
            'src/isaacmin/source/nbt.py','src/isaacmin/source/semantics.py',
            'src/isaacmin/source/snapshot.py',
            'src/isaacmin/adapters/workers.py','src/isaacmin/terrain/baseline.py',
            'src/isaacmin/source/surface_objects.py','scripts/assemble_surface_preview.py',
            'blender_scripts/export_surface.py','scripts/finalize_surface_collision.py',
            'scripts/finalize_surface_partition.py','scripts/finish_surface_partitions.py',
            'isaac_scripts/material_log.py',
            'isaac_scripts/visible_instance_labels.py',
            'isaac_scripts/capture_outdoor_visual.py','recipes/materials/surface_stochastic.mdl.in',
            'recipes/materials/clear_water.mdl')})
    digest=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    output=Path(output).resolve() if output else workspace/'worlds'/('outdoor_'+digest[:16])
    output.mkdir(parents=True,exist_ok=True);manifest_path=output/'outdoor_build.json'
    if manifest_path.exists():
        record=read_json(manifest_path)
        if record['identity']!=json.loads(json.dumps(identity)):raise ValueError('Build dependencies changed; use a new output')
    else:
        record=dict(schema_version=1,identity=identity,created_at_utc=utc_now(),stages={},
            source_world=str(world),center_provenance=centre_origin,qualification='not_run',
            structures='excluded_by_user',underground='excluded_by_user',navigation_stack='not_run_not_supplied')
        atomic_json(manifest_path,record)
    # Retain the exact constructors beside each new candidate. This is source
    # provenance, not a claim that source code alone reproduces native rendering.
    import shutil
    producer_directory=output/'producer_sources'
    for name,expected in identity['producer_sha256'].items():
        frozen=producer_directory/name
        if not frozen.exists():
            frozen.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(workspace/name,frozen)
        if sha256_file(frozen)!=expected:raise ValueError('Frozen build producer changed: '+name)
    def pending(stage):
        if stage in record['stages']:return False
        if stage_controller:stage_controller.before(stage,record)
        return True
    def checkpoint(stage,receipt):
        record['stages'][stage]=receipt;record['updated_at_utc']=utc_now();atomic_json(manifest_path,record)
        if stage_controller:stage_controller.completed(stage,receipt)
    source=output/'source';objects=output/'objects';terrain=output/'terrain';blender=output/'blender';population=output/'population';scene=output/'assembled'
    if pending('source'):
        result=extract_macro_surface(Path(snapshot['snapshot_path']),source,centre,extent=extent+2*SurfaceRecipe().context_m,spacing=1,snapshot_sha256=snapshot['save_sha256'])
        checkpoint('source',dict(status=result['status'],manifest_sha256=sha256_file(source/'macro_surface.json')))
    fields=source_fields(source/'macro_surface.npz')
    atomic_json(output/'biome_coverage.json',biome_inventory(fields['biome'],fields['validity'],
        available_materials=library.get('ground_materials',[])))
    if pending('objects'):
        result=extract_surface_objects(Path(snapshot['snapshot_path']),source/'macro_surface.npz',objects)
        checkpoint('objects',dict(trees=len(result['trees']),manifest_sha256=sha256_file(objects/'objects.json')))
    if pending('terrain'):
        result=build_surface_terrain(workspace,source/'macro_surface.npz',terrain,(centre[0],0.,centre[1]),
            material_manifest=workspace/library['material_manifest'],material_origin=material_origin)
        checkpoint('terrain',dict(vertices=result['vertices'],triangles=result['triangles'],manifest_sha256=sha256_file(terrain/'terrain.json')))
    if pending('route'):
        from isaacmin.outdoor_routes import plan_outdoor_route
        result=plan_outdoor_route(terrain,objects)
        atomic_json(output/'route_candidate.json',result)
        checkpoint('route',dict(status=result['status'],candidate_length_m=result.get('unique_polyline_length_m',0.),
            manifest_sha256=sha256_file(output/'route_candidate.json'),actual_traversal='not_run'))
    route=read_json(output/'route_candidate.json')
    if route['status']!='candidate':route=None
    def worker(script,request_path,log,memory,*,blender_worker=False,isaac=False):
        if blender_worker:
            argv=[str(workspace/'.tools/blender/blender'),'--background','--factory-startup','--disable-autoexec','--python-exit-code','1','--python',str(workspace/script),'--',str(request_path)]
            env=native_environment(workspace)
        elif isaac:
            argv=[str(workspace/'.tools/isaacsim-6.1.0/python.sh'),str(workspace/script),'--request',str(request_path)];env={'LD_LIBRARY_PATH':''}
        else:
            argv=[str(workspace/'.venv/bin/python'),str(workspace/script),'--request',str(request_path)]
            env={'PYTHONPATH':str(workspace/'.tools/usd/lib/python')+':'+str(workspace/'src'),'LD_LIBRARY_PATH':str(workspace/'.tools/native/usr/lib/aarch64-linux-gnu')+':'+str(workspace/'.tools/native/usr/lib')}
        result=run_worker(argv,cwd=workspace,log_path=log,timeout=3600,estimated_memory_bytes=memory*2**30,environment=env)
        if result['exit_code']!=0:raise RuntimeError('Native stage failed; retained evidence: '+str(log))
        return dict(exit_code=0,receipt=str(log.with_suffix('.process.json')))
    if pending('blender'):
        blender.mkdir(exist_ok=True);request=blender/'request.json';atomic_json(request,dict(workspace=str(workspace),terrain=str(terrain),output=str(blender)))
        if extent>512:
            from isaacmin.terrain.surface_partitions import partition_windows
            for part in partition_windows(read_json(terrain/'terrain.json')['grid_shape']):
                directory=blender/part['name'];directory.mkdir(exist_ok=True)
                part_request=directory/'request.json'
                atomic_json(part_request,dict(workspace=str(workspace),terrain=str(terrain),output=str(directory),partition=part))
                key='blender:'+part['name']
                if key not in record['stages']:
                    worker('blender_scripts/export_surface.py',part_request,directory/'worker.log',16,blender_worker=True)
                    result=worker('scripts/finalize_surface_partition.py',part_request,directory/'finalize.log',16)
                    checkpoint(key,result)
            checkpoint('blender',worker('scripts/finish_surface_partitions.py',request,blender/'finish.log',16))
        else:
            checkpoint('blender',worker('blender_scripts/export_surface.py',request,blender/'worker.log',16,blender_worker=True))
    if pending('population'):
        result=populate_surface(workspace,terrain,objects,population,route=route)
        checkpoint('population',dict(instances=result['instances'],manifest_sha256=sha256_file(population/'population.json')))
    if pending('assembly'):
        scene.mkdir(exist_ok=True);request=scene/'request.json';atomic_json(request,dict(workspace=str(workspace),
            library_scene=str(workspace/library['scene']),terrain=str(terrain),output=str(scene),
            blender_export=str(blender),population=str(population),objects=str(objects),
            canopy_libraries=library.get('canopy_libraries',[]),
            canopy_rules=library.get('canopy_rules'),
            scenery_libraries=library.get('scenery_libraries',[]),
            shrub_libraries=library.get('shrub_libraries',[]),
            litter_libraries=library.get('litter_libraries',[]),
            native_denoising=library.get('development_native_denoising',False),
            capture_template=str(workspace/library['capture_template']),poses=outdoor_camera_poses(terrain,objects)))
        payload=read_json(request);payload['route']=route;atomic_json(request,payload)
        checkpoint('assembly',worker('scripts/assemble_surface_preview.py',request,scene/'worker.log',20))
    if pending('collision'):
        result=run_worker([str(workspace/'.venv/bin/python'),str(workspace/'scripts/finalize_surface_collision.py'),
            '--scene',str(scene/'scene/world.usda'),'--evidence',str(scene/'exact_collision.json')],cwd=workspace,
            log_path=scene/'collision.log',timeout=1800,estimated_memory_bytes=12*2**30,
            environment={'PYTHONPATH':str(workspace/'.tools/usd/lib/python')+':'+str(workspace/'src'),
                'LD_LIBRARY_PATH':str(workspace/'.tools/native/usr/lib/aarch64-linux-gnu')+':'+str(workspace/'.tools/native/usr/lib')})
        if result['exit_code']!=0:raise RuntimeError('Native collision assembly failed')
        checkpoint('collision',dict(exact_triangle_evidence=str(scene/'exact_collision.json'),PhysX_contact='not_run'))
    if capture and capture_set=='focus' and 'focus_capture' not in record['stages']:
        focused=read_json(scene/'preview_request.json');poses=focused['poses']
        # Prioritise distinctive foliage/ice in the reduced development set.
        # Full final coverage retains every authored pose; no map-name branches.
        distinctive={'minecraft:jungle','minecraft:sparse_jungle','minecraft:bamboo_jungle',
            'minecraft:frozen_ocean','minecraft:deep_frozen_ocean','minecraft:ice_spikes','minecraft:frozen_peaks',
            'minecraft:forest','minecraft:flower_forest','minecraft:birch_forest','minecraft:old_growth_birch_forest',
            'minecraft:dark_forest','minecraft:taiga','minecraft:old_growth_pine_taiga','minecraft:old_growth_spruce_taiga'}
        first=next((p for p in poses if p.get('source_biome') in distinctive),poses[0])
        shore=next((p for p in poses if p.get('name')=='source_water_shore'),poses[-1])
        focused.update(output=str(scene/'focus'),poses=[first,shore] if shore!=first else [first],
            scope='two targeted development views; full capture coverage not_run; render quality unchanged')
        atomic_json(scene/'focus_request.json',focused)
        checkpoint('focus_capture',worker(library['capture_worker'],scene/'focus_request.json',scene/'focus.log',64,isaac=True))
    if capture and capture_set=='full' and pending('capture'):
        checkpoint('capture',worker(library['capture_worker'],scene/'preview_request.json',scene/'preview.log',64,isaac=True))
    if capture and capture_set=='full' and pending('overview'):
        overview=read_json(scene/'preview_request.json')
        overview.update(output=str(scene/'overview'),poses=outdoor_overview_poses(terrain),
            scope='aerial_progress_views_of_built_region_only; not robot-height qualification')
        atomic_json(scene/'overview_request.json',overview)
        checkpoint('overview',worker(library['capture_worker'],scene/'overview_request.json',scene/'overview.log',64,isaac=True))
    record.update(status='geometry_world',scene=str(scene/'scene/world.usda'),qualification='not_run',
        requested_capture_set=capture_set if capture else 'none',
        biome_qualification='not_run; see biome_coverage.json',actual_navigation_stack='not_run_not_supplied',
        full_end_to_end_qualification='incomplete; development captures only')
    atomic_json(manifest_path,record)
    return dict(status='geometry_world',scene=record['scene'],report=str(manifest_path),qualification='not_run')
