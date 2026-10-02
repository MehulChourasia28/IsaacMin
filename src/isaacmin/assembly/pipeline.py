"""Production assembly boundary; keeps source volumes authoritative and unqualified."""
from __future__ import annotations
import json
import hashlib
from pathlib import Path
import numpy as np
from isaacmin.adapters.workers import project_root, _run, reconstruct_occupancy, _sha


def assemble_region(ir_directory: str | Path, output_dir: str | Path, *,
                    materials: list[dict], origin=(-1065.38, 0.0, 688.07),
                    assets: list[dict] | None = None,
                    exterior_delta: dict | None = None,
                    volume_directory: str | Path | None = None,
                    geometry_detail: dict | None = None,
                    support_mask_path: str | Path | None = None,
                    source_water_mesh_path: str | Path | None = None,
                    bare_scene_directory: str | Path | None = None,
                    terrain_material_recipe: dict | None = None,
                    foliage_material_recipe: dict | None = None,
                    source_fidelity_budget_path: str | Path | None = None) -> dict:
    """Build the same complete 3D terrain for both technical smoke and real regions.

    Unknown/missing coverage is rejected. Nonterrain structures/water remain explicit
    exclusions for this construction stage, never silently promoted into a final world.
    """
    root=project_root(); ir=Path(ir_directory).resolve(); output=Path(output_dir).resolve()
    if terrain_material_recipe is not None:
        from .material_recipe import validate_material_recipe
        validate_material_recipe(root,terrain_material_recipe,materials)
    if foliage_material_recipe is not None:
        from .foliage_recipe import validate_foliage_recipe
        validate_foliage_recipe(root,foliage_material_recipe)
    output.mkdir(parents=True, exist_ok=True)
    data=np.load(ir/'natural_occupancy.npz',allow_pickle=False)
    occupancy=data['occupancy']; validity=data['validity']
    if not validity.all() or np.any(occupancy==3):
        raise ValueError('Missing/unknown cells require explicit surface coverage boundaries')
    occupied=occupancy==1
    support_manifest=None
    if support_mask_path is not None:
        support_path=Path(support_mask_path).resolve()
        support=np.load(support_path,allow_pickle=False)
        support_manifest=json.loads((support_path.parent/'support_manifest.json').read_text())
        if support_manifest['source_ir_sha256']!=_sha(ir/'world_ir.json'):
            raise ValueError('Structural support was derived from a different source IR')
        support_file=next(p for p in support_manifest['files'] if p['path']==support_path.name)
        if support_file['sha256']!=_sha(support_path):raise ValueError('Structural support content hash mismatch')
        candidate=support['occupancy'];structure=support['structural_occupancy']
        if candidate.dtype!=np.bool_ or candidate.shape!=occupancy.shape or structure.dtype!=np.bool_ or structure.shape!=occupancy.shape:
            raise ValueError('Structural support Boolean occupancy shape mismatch')
        if not support['validity'].all() or not np.array_equal(support['min_xyz'],data['min_xyz']):
            raise ValueError('Structural support validity/frame differs from source')
        if not np.array_equal(candidate,occupied|structure) or np.any(structure&(occupancy!=2)):
            raise ValueError('Structural support must preserve all natural ground and add only identified nonterrain cells')
        occupied=candidate
    volume_dir=Path(volume_directory).resolve() if volume_directory else output/'volume'
    record=volume_dir/'worker_result.json'
    if not record.is_file():
        volume=reconstruct_occupancy(occupied,volume_dir)
    else:
        volume=json.loads(record.read_text())
        if volume.get('appearance_qualification')!='not_run':
            raise ValueError('Unexpected volume provenance')
        if volume.get('occupancy_sha256') != hashlib.sha256(occupied.tobytes()).hexdigest():
            raise ValueError('Cached volume source occupancy changed; rebuild required')
        if volume.get('executable_sha256') != _sha(root/'.tools/openvdb_worker'):
            raise ValueError('Cached volume worker changed; rebuild required')
        for name,key in [('terrain.obj','mesh_sha256'),('terrain.vdb','volume_sha256')]:
            if _sha(volume_dir/name)!=volume[key]: raise ValueError('Volume content hash mismatch')
    if volume.get('status')!='success': return volume
    from isaacmin.volumes.construction import constrained_terrain
    construction=constrained_terrain(root,ir,volume_dir,support_mask_path=support_mask_path)
    terrain_path=Path(construction['mesh'])
    # Source array indices denote whole blocks; VDB grid centres are at integer indices.
    geometry_detail=dict(geometry_detail or {})
    if int(geometry_detail.get('subdivision_levels',0)):
        geometry_detail.setdefault('subdivision_method','bilinear_double_no_limit_v1')
    mx,my,mz=[float(v)+0.5 for v in data['min_xyz']]
    ox,oy,oz=origin
    terrain_translation=[mx-ox,-mz+oz,my-oy]
    precision_input=None
    if int((geometry_detail or {}).get('subdivision_levels',0)):
        from isaacmin.volumes.native_precision import condition_input
        terrain_path,precision_input=condition_input(root,terrain_path,terrain_translation,origin,
            geometry_detail,volume_dir/'native_precision_input')
        # The precision worker encodes the exact eventual native world frame.
        terrain_translation=[0,0,0]
    naturalization_record = None
    if source_fidelity_budget_path is not None:
        if exterior_delta is None or precision_input is None:
            raise ValueError('Exterior naturalization requires actual refinement and normalized native precision input')
        from isaacmin.terrain.naturalize_exterior import construct_cached
        terrain_path, naturalization_record = construct_cached(
            terrain_path, ir/'terrain_surface.npz', Path(exterior_delta['path']),
            Path(source_fidelity_budget_path), origin, volume_dir/'exterior_naturalization')
    water_path=None
    source_meta=json.loads((ir/'world_ir.json').read_text()) if (ir/'world_ir.json').is_file() else {}
    if source_water_mesh_path is not None:
        supplied=Path(source_water_mesh_path).resolve()
        water_path=output/'source_water_interfaces.json'
        water_path.write_bytes(supplied.read_bytes())
    elif source_meta.get('terrain_volume',{}).get('chunk_records'):
        from .water import source_water_from_ir
        water=source_water_from_ir(ir,origin=origin,support_mask_path=support_mask_path)
        water_path=output/'source_water_interfaces.json'
        water_path.write_text(json.dumps(water,separators=(',',':'))+'\n')
    request={'output':str(output),'terrain_mesh':str(terrain_path),
             'terrain_mesh_sha256':_sha(terrain_path),'source_interface_construction':construction,
             'terrain_translation':terrain_translation,'native_precision_input':precision_input,
             'exterior_naturalization':str(naturalization_record) if naturalization_record else None,
             'minecraft_origin':list(origin),'materials':materials,
             'texture_repeat_m':float(materials[0].get('repeat_m',2)),
             'assets':assets or [],'exterior_delta':exterior_delta,
             'geometry_detail':geometry_detail or {},
             'terrain_material_recipe':terrain_material_recipe,
             'foliage_material_recipe':foliage_material_recipe,
             'support_manifest':str(Path(support_mask_path).resolve().parent/'support_manifest.json') if support_mask_path else None,
             'source_water_mesh':str(water_path) if water_path else None,
             'source_water_mesh_sha256':_sha(water_path) if water_path else None,
             'source_surface':str(ir/'terrain_surface.npz')}
    if bare_scene_directory is not None:
        request['bare_scene_directory']=str(Path(bare_scene_directory).resolve())
        from .native_cache import verify_bare
        verify_bare(request,root,bare_scene_directory)
    request_path=output/'assembly_request.json'
    request_path.write_text(json.dumps(request,indent=2)+'\n')
    candidates=[root/p for p in ('.tools/blender/bin/blender','.tools/blender/blender',
                                 '.tools/build/blender/bin/blender')]
    blender=next((p for p in candidates if p.is_file()),candidates[0])
    unique_asset_paths={Path(a['blend']).resolve() for a in (assets or [])}
    subdivision=int((geometry_detail or {}).get('subdivision_levels',0))
    # Actual51M-triangle OpenSubdiv export sampled40.3GiB RSS; the previous
    # factor10 estimated17GiB. Reserve a conservative full geometry/UV/USD peak.
    estimated_memory=2**30+terrain_path.stat().st_size*30*(4**subdivision)
    estimated_memory+=sum(p.stat().st_size*12 for p in unique_asset_paths)
    estimated_disk=estimated_memory
    if bare_scene_directory is not None:
        from .cached_population import reuse_memory_estimate
        reuse_resources=reuse_memory_estimate(
            (Path(bare_scene_directory)/'scene.blend').stat().st_size,
            sum(p.stat().st_size for p in unique_asset_paths))
        estimated_memory=reuse_resources['estimated_memory_bytes']
        estimated_disk=reuse_resources['estimated_disk_bytes']
        (output/'resource_plan.json').write_text(json.dumps(reuse_resources,indent=2)+'\n')
    result=_run([str(blender),'--background','--factory-startup','--disable-autoexec',
                 '--python-use-system-env',
                 '--python-exit-code','1','--python',str(root/'blender_scripts/export_scene.py'),
                 '--','--request',str(request_path)],output,86400,
                 estimated_memory_bytes=estimated_memory,estimated_disk_bytes=estimated_disk)
    if result['status']=='success':
        from .native_cache import deduplicate_generated_textures
        texture_storage=deduplicate_generated_textures(output,root/'.cache/native_textures')
        (output/'texture_storage.json').write_text(json.dumps(texture_storage,indent=2)+'\n')
        result.update(json.loads((output/'export_result.json').read_text()))
        result.update(root_usd=str(output/'world.usda'),
                      source_interface_construction=construction,
                      structural_support_cells=int(np.count_nonzero(structure)) if support_mask_path else 0,
                      support_manifest_sha256=_sha(Path(support_mask_path).resolve().parent/'support_manifest.json') if support_mask_path else None,
                      source_exclusions={'nonterrain_voxels_not_retained_as_structural_support':int(np.count_nonzero(occupancy==2))-int(np.count_nonzero(structure)) if support_mask_path else int(np.count_nonzero(occupancy==2)),
                                         'source_water_voxels_retained_in_IR':int(np.count_nonzero(occupancy==4)),
                                         'exterior_water_geometry_cells':int(result.get('source_exterior_water_cells',0)),
                                         'lava_voxels':int(np.count_nonzero(occupancy==5))},
                      qualification='not_qualified')
    (output/'assembly_result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
