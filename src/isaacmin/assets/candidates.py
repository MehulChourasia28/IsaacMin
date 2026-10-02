"""Unqualified scene candidates from verified master manifests."""
import json
import math
from pathlib import Path

from .network import ServiceError, digest
from .network import atomic_json


def candidate_lighting(workspace: Path):
    """Verified sky-only emitters; geometric scenery always belongs to the USD."""
    workspace=Path(workspace).resolve()
    catalogue=json.loads((workspace/'state/asset_catalogue.json').read_text())['assets']
    decode_path=workspace/'evidence/services/lighting/native_hdri_decode.json'
    if not decode_path.is_file():
        raise ServiceError('lighting_decode','Run prepare_lighting before assigning the actual HDRI master')
    decoded=json.loads(decode_path.read_text())['images']
    conditions={}
    for condition,asset_id,nominal_illuminance in [('directional','kloofendal_48d_partly_cloudy_puresky',30000),
                                                ('diffuse','kloofendal_overcast_puresky',10000)]:
        asset=next((a for a in catalogue if a['asset_id']==asset_id),None)
        if asset is None or 'pure skies' not in asset['metadata'].get('tags',[]):
            raise ServiceError('lighting_provenance','Required verified Pure Sky HDRI is missing')
        files=[f for f in asset['files'] if f.get('role')=='hdri' and f.get('format')=='exr' and f.get('resolution')=='4k']
        if len(files)!=1 or digest(Path(files[0]['path']))!=files[0]['sha256']:
            raise ServiceError('lighting_corruption','Lossless lighting master is missing or changed')
        measurement=next((d for d in decoded if d['asset_id']==asset_id and d['sha256']==files[0]['sha256']),None)
        if measurement is None or not measurement['finite_samples'] or measurement['upper_hemisphere_luminance_integral']<=0:
            raise ServiceError('lighting_decode','Lighting master lacks matching finite native radiance evidence')
        gain=nominal_illuminance/measurement['upper_hemisphere_luminance_integral']
        ev100=math.log2(nominal_illuminance*.18/math.pi*100/12.5)
        conditions[condition]={'asset_id':asset_id,'hdri':files[0]['path'],'hdri_sha256':files[0]['sha256'],
                              'source_page':asset['source_page'],'licence':'CC0-1.0',
                              'dome_texture_format':'latlong','dome_intensity':gain,'dome_exposure':0,
                              'dome_rotation_z_degrees':0,'sun_intensity':0,
                              'sun_policy':'HDRI contains its own illumination; no additional distant sun',
                              'nominal_upper_hemisphere_illuminance_lux':nominal_illuminance,
                              'intensity_method':'nominal illuminance / measured upper-hemisphere relative luminance integral',
                              'photometric_limit':'Absolute lux assumes renderer interprets unit luminance as cd/m2; target meter/gray-card calibration not_run',
                              'camera_default':{'auto_exposure':False,'iso':100,'f_number':8,'shutter_seconds':64/2**ev100,
                                                'ev100':ev100,'white_balance_kelvin':5500,'motion_blur':False},
                              'radiance_evidence':measurement,'target_qualification':'not_run'}
    result={'schema_version':1,'default_condition':'directional','conditions':conditions,
            'background_scope':'Infinite provider-edited sky only. No photographic terrain, trees, mountains or ground can substitute for source USD geometry.',
            'lower_hemisphere_limit':'Provider extends/edits the sky below horizon; source ground must occlude it inside valid coverage. Out-of-coverage ground is never inferred from HDRI.',
            'inspection':'Licensed original tonemapped panoramas visually inspected: sky/clouds and provider lower-hemisphere extension only',
            'qualification':'not_run','no_fog_or_depth_of_field_concealment':True}
    atomic_json(workspace/'state/lighting_candidates.json',result)
    return result


def candidate_materials(workspace: Path):
    path = Path(workspace) / "state/terrain_materials.json"
    if not path.exists():
        raise ServiceError("missing_material_manifest", "Run source-specific asset acquisition first")
    result = json.loads(path.read_text())["materials"]
    catalogue = json.loads((Path(workspace) / "state/asset_catalogue.json").read_text())["assets"]
    hashes = {f["path"]: f["sha256"] for a in catalogue for f in a["files"]}
    for material in result:
        if not 0.1 <= material["repeat_m"] <= 20:
            raise ServiceError("material_scale", "Physical material scale is out of supported bounds")
        for role in ("base_color", "roughness", "normal", *(['height'] if 'height' in material else [])):
            if role not in material or material[role] not in hashes or digest(Path(material[role])) != hashes[material[role]]:
                raise ServiceError("material_provenance", "Required material channel lacks verified master bytes")
    return result


def candidate_placements(workspace: Path, ir_directory: Path, scene_blend: Path, output_dir: Path,
                         runoff_path=None, route_path=None):
    from isaacmin.process import run_worker
    from isaacmin.adapters.workers import native_environment
    workspace, output_dir = Path(workspace).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = [workspace / '.tools/blender/bin/blender',workspace / '.tools/blender/blender',workspace / '.tools/build/blender/bin/blender',
                  workspace / '.tools/native/usr/bin/blender']
    blender = next((p for p in candidates if p.is_file()), None)
    if blender is None:
        raise ServiceError('blocked_dependency', 'Native Blender unavailable for actual final-ground ray queries')
    env = native_environment(workspace)
    if blender == candidates[-1]:
        env.update(BLENDER_SYSTEM_SCRIPTS=str(workspace/'.tools/native/usr/share/blender/scripts'),
                   BLENDER_SYSTEM_DATAFILES=str(workspace/'.tools/native/usr/share/blender/datafiles'))
    report_path = output_dir / 'ecology_manifest.json'
    scene_blend=Path(scene_blend).resolve()
    inventory_path=scene_blend.parent/'export_result.json'
    if not inventory_path.is_file():
        raise ServiceError('missing_ground_inventory','Native export counts are required for placement resource admission')
    exported=json.loads(inventory_path.read_text())
    vertices=exported.get('terrain_vertices');triangles=exported.get('native_usd_terrain_triangles')
    if type(vertices) is not int or type(triangles) is not int or min(vertices,triangles)<=0:
        raise ServiceError('invalid_ground_inventory','Exported native ground counts must be positive integers')
    # Conservative envelope: loaded .blend/dependency data, native mesh/evaluation
    # buffers, direct C++ BVH and temporary numeric arrays. The process guard still
    # measures shared system availability; this estimate is not a measured peak.
    estimate=8*2**30+2*scene_blend.stat().st_size+vertices*96+triangles*192
    timeout=max(3600,min(24*3600,3600+triangles/1000))
    inventory={'vertices':vertices,'triangles':triangles,'export_result':str(inventory_path),
               'export_result_sha256':digest(inventory_path),'scene_blend_bytes':scene_blend.stat().st_size,
               'estimated_memory_bytes':estimate,'estimated_disk_bytes':2**30,'timeout_seconds':timeout,
               'estimate_method':'8GiB process reserve +2*blend bytes +96 bytes/vertex +192 bytes/triangle; direct native BVH, no Python face lists',
               'estimate_status':'conservative admission envelope; actual shared-memory pressure remains monitored'}
    request = {'workspace': str(workspace), 'ir_directory': str(Path(ir_directory).resolve()),
               'scene_blend': str(Path(scene_blend).resolve()), 'output_report': str(report_path),
               'runoff_path': str(runoff_path) if runoff_path else None,
               'route_path': str(route_path) if route_path else None, 'seed': 1729,'resource_inventory':inventory}
    atomic_json(output_dir/'ecology_request.json', request)
    previous = report_path.stat().st_mtime_ns if report_path.exists() else None
    result = run_worker([str(blender), '-b', '--factory-startup', '--disable-autoexec', '--python-use-system-env','--python-exit-code', '1',
                         '--python', str(workspace/'blender_scripts/place_ecology.py'), '--', str(output_dir/'ecology_request.json')],
                        cwd=workspace, log_path=output_dir/'ecology.log', timeout=timeout, environment=env,
                        estimated_memory_bytes=estimate,estimated_disk_bytes=2**30)
    if result['exit_code'] != 0 or not report_path.is_file() or report_path.stat().st_mtime_ns == previous:
        raise ServiceError('placement_failed', 'Final-ground candidate placement failed; process evidence retained')
    report = json.loads(report_path.read_text())
    if report.get('qualification') != 'not_run':
        raise ServiceError('invalid_qualification', 'Placement worker may not self-issue qualification')
    return report
