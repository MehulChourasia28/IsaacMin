"""Small, evidence-backed answers for the local assistant; no arbitrary file reads."""
from pathlib import Path

from isaacmin.io import read_json, sha256_file
from .catalog import load, presets, planner_status
from .views import resolve_target, completed_views


TARGET_SCHEMA = {'type': 'string', 'pattern': r'^(preset|latest|job_[a-f0-9]{32})$'}
TOPICS = ('summary', 'source', 'terrain', 'assets', 'captures', 'qualification', 'delivery')


def capabilities(root):
    return dict(
        planner=planner_status(root),
        agent_harness=dict(name='OpenClaw',version='2026.9.7',runtime='embedded CLI with a local typed-tool plugin',
            credentials='NVIDIA credentials stay in the Python adapter; neither OpenClaw nor native asset workers receive the key.',
            build_control='Create Isaac world calls OpenClaw/Nemotron before each of eleven ordered stages. Only the exact next stage and checkpoint are permitted; native receipts establish completion.',
            tool_policy='Only IsaacMin registered tools; no shell, file, browser, messaging, plugin installation or subagent tools.',
            historical_sessions='Existing custom-harness sessions retain their original runtime on resume.'),
        questions={
            'worlds': 'Available presets, centres, built dimensions, source identity and biome observations.',
            'geography': 'Recorded elevation range, mountain/water/ice policy, terrain resolution and source coverage limits.',
            'assets': 'Recorded materials, source tree species, placed vegetation/litter/rock counts and asset qualification.',
            'views': 'Existing ground/aerial/source captures, image links, counts, camera poses and new-view constraints.',
            'delivery': 'Portable world readiness, size, SHA256, download link and opening instructions.',
            'jobs': 'Persistent progress, phase, errors, attempts, completed results, recovery and queued-job cancellation.',
            'evidence': 'Actual native captures, collision construction versus tested physics, package replay failures and missing qualification.',
            'operation': 'Local workflow, deterministic stage reuse, selected NVIDIA model, privacy, timing evidence and limits.'},
        actions={
            'verify': 'Hash-check registered scene dependencies and published capture evidence; does not certify realism.',
            'prepare': 'Prepare or verify the selected preset portable ZIP with local dependencies.',
            'convert': 'Create a separate 256 m world from the selected registered save, native ground/aerial captures and portable ZIP.',
            'views': 'Capture 1–6 additional actual ground or aerial views of the preset or a completed conversion; preserve geometry/materials.',
            'resume': 'Resume a failed, interrupted or cancelled job for the selected world, up to three total attempts.',
            'cancel': 'Cancel a queued job for the selected world; running native work is preserved.'},
        view_controls=dict(ground_height_m=0.6, aerial_height_m=[20, 600], count=[1, 6],
            coordinates='Minecraft X,Z; a ground location is the camera point; an aerial location is the look-at point.',
            heading='0 degrees north (-Minecraft Z), 90 east (+Minecraft X). Multiple views spread directions evenly.',
            defaults='Blank coordinates select supported ground positions or the built-region centre for aerial views.',
            constraints='Location at least 2m inside built bounds. Ground rejects exposed water, slopes over 25 degrees and tree clearance intersections.',
            output='Native RGB, depth and label evidence locally; RGB shown in the gallery. New captures do not rewrite an existing world ZIP.'),
        operation=dict(
            selection='Choose a world in the UI before acting on it. Preset comparison is read-only.',
            context='Each assistant request is independent. Saved chat is a transcript, not cross-request model memory. The construction/new-result screen binds evidence and view requests to its displayed conversion; Saved worlds defaults to the preset.',
            opening='The root page starts with four Minecraft source choices and a Create Isaac world button. Saved worlds retains previous results. A job URL restores the same conversion after refresh.',
            construction_preview='A lightweight local WebGL diagram uses completed actual terrain and native USD asset geometry/placement anchors. Its reveal animation and symbolic colors are not final Isaac renders or per-instance placement timing. Final native galleries and ZIP appear when conversion completes.',
            pipeline='Immutable source snapshot → source surface/objects → native HighMap refinement → Blender USD → natural asset assembly → final-ground collision → Isaac captures → portable ZIP.',
            native_concurrency=1, hosted_agent_lane='Independent of native work; status requests remain available.',
            recovery='Restart the local studio; interrupted jobs become resumable. Completed native stages and job identities are retained.',
            start='env -u NVIDIA_API_KEY .venv/bin/python -m isaacmin demo --port 8765',
            open_download='Extract the entire ZIP; open scene/world.usda in the pinned local Isaac Sim. Keep all local dependencies together; see package README.',
            source_previews='Actual Chunky renders of the matching source-save region, not live Minecraft screenshots.',
            evidence='Recorded facts only. No image understanding or automatic visual assessment in this assistant.'),
        limits=[
            'Automatic visual inspection, iterative repair and world appearance edits are explicitly excluded.',
            'No arbitrary source paths, uploads, new preset registration, extent changes, arbitrary shell, code, purchases or provider changes through the runtime assistant.',
            'No live browser Isaac rendering, custom video route rendering, robot driving or navigation-stack execution.',
            'No running-job pause/kill, quality threshold changes, autonomous model substitution, source-save changes or credential access tools.',
            'Four 256m development presets; 1km delivery and full realism/motion/contact qualification remain incomplete.',
            'Caves, ravines, underground structures and buildings are excluded; mountains, valleys and source surface water are retained.'])


def list_presets(root):
    return {'worlds': [{k: p.get(k) for k in ('id', 'title', 'subtitle', 'center', 'extent_m', 'qualification')}
                       for p in load(root)['presets']], 'actions_scoped_to_selected_world': True}


def _read(path):
    return read_json(path) if path.is_file() else {}


def named_bounds(values):
    return dict(zip(('min_x','min_z','max_x','max_z'),values))


def inspect_world(root, jobs, preset, target='preset', topic='summary'):
    root = Path(root)
    build, key = resolve_target(root, jobs, preset, target)
    record = read_json(build/'outdoor_build.json')
    terrain = read_json(build/'terrain/terrain.json')
    is_preset = key.startswith('preset_')
    item = next(p for p in load(root)['presets'] if p['id'] == preset)
    bounds=terrain['bounds_source_xz']
    extent=dict(x_metres=bounds[2]-bounds[0],z_metres=bounds[3]-bounds[1])
    summary = dict(preset=preset, title=item['title'], target=key, status=record['status'],
        built_bounds_minecraft_xz=named_bounds(bounds), built_extent=extent, origin_xyz=terrain['origin_xyz'],
        coordinate_transform='USD X = Minecraft X - origin X; USD Y = -Minecraft Z + origin Z; USD Z = Minecraft Y - origin Y; units metres; Z up.',
        view_url='/library?preset='+preset if is_preset else '/results/'+key,
        source_save_sha256=record['identity']['source'], scene_sha256=sha256_file(Path(record['scene'])),
        qualification=record.get('full_end_to_end_qualification', record.get('qualification', 'not_run')))
    if topic == 'summary':
        source=_read(build/'source/macro_surface.json') or _read(Path(terrain['source_surface']['path']).with_suffix('.json'))
        context=source.get('scope',{})
        context_bounds=context.get('bounds_blocks_xz')
        summary.update(extent_m=[terrain['bounds_source_xz'][2]-terrain['bounds_source_xz'][0],
                                terrain['bounds_source_xz'][3]-terrain['bounds_source_xz'][1]],
            completed_stages=list(record.get('stages', {})), source_save_modified=False,
            source_extraction_context=dict(bounds_minecraft_xz=named_bounds(context_bounds),
                extent_m=[context_bounds[2]-context_bounds[0],context_bounds[3]-context_bounds[1]],
                scope='Recorded context around this crop; not the entire save') if context_bounds else None)
        return summary
    result = dict(preset=preset, target=key, topic=topic,built_bounds_minecraft_xz=named_bounds(bounds),built_extent=extent,
        evidence_scope='Recorded development artifacts, not inferred qualification')
    if topic == 'source':
        source = _read(build/'source/macro_surface.json') or _read(Path(terrain['source_surface']['path']).with_suffix('.json'))
        if source.get('source_snapshot_sha256')!=record['identity']['source']:
            raise ValueError('Source metadata does not match the selected world')
        registered = next(w for w in read_json(root/'state/source_worlds.json')['worlds'] if w['id'] == preset)
        context=dict(source.get('scope',{}))
        if 'bounds_blocks_xz' in context:context['bounds_minecraft_xz']=named_bounds(context.pop('bounds_blocks_xz'))
        result.update(source_save=registered['source'], preferred_center=registered['center_minecraft_xz'],
            level_dat_sha256=source.get('source_level_dat_sha256'), source_save_sha256=record['identity']['source'],
            source_context=context, context_biome_samples=source.get('biomes'),
            context_surface=source.get('surface'), unknown_semantics=source.get('unknown_semantics'),
            missing_region_paths=source.get('missing_region_paths'),
            scope='Biomes and coverage are the recorded extraction context, which may exceed the built crop; not the entire save.',
            level_dat_role='Save metadata and identity; actual terrain/biomes come from Anvil chunk data. Input is read-only.')
    elif topic == 'terrain':
        assembly = _read(Path(record['scene']).parent.parent/'assembly.json')
        result.update(origin_xyz=terrain['origin_xyz'],
            grid_shape=terrain['grid_shape'], vertices=terrain['vertices'], triangles=terrain['triangles'],
            recipe=terrain['recipe'], reconstruction=terrain['reconstruction'],
            water=_read(Path(record['scene']).parent.parent/'water.json'),
            source_water_columns=assembly.get('source_water_columns'),
            collision_policy=terrain.get('collision_policy'), underground='excluded_by_user', structures='excluded_by_user')
        if result['water']:
            water=result['water']; levels=water.pop('levels', [])
            water['level_count']=len(levels)
            water['source_level_range_m']=[min(v['source_level_m'] for v in levels),max(v['source_level_m'] for v in levels)] if levels else None
    elif topic == 'assets':
        from .views import object_directory
        objects = object_directory(root, build)
        observations = _read(objects/'objects.json') if objects else {}
        assembly = _read(Path(record['scene']).parent.parent/'assembly.json')
        coverage = _read(build/'biome_coverage.json')
        result.update(ground_materials=terrain.get('material_order'),
            observed_tree_species_context=observations.get('species_counts'),
            placed={k: assembly.get(k) for k in ('trees', 'ground_cover_instances', 'original_litter_instances', 'original_shrub_instances')},
            recorded_biome_families={k: {'samples': v.get('samples'), 'source_biomes': v.get('source_biomes'),
                'material_available': v.get('material_available'), 'qualification': v.get('qualification')}
                for k,v in coverage.get('families', {}).items()},
            unknown_biomes=coverage.get('unknown_biomes'),
            assets_scope='Observed species are source context; placed counts are this assembly if available. Null means no matching receipt, not zero.',
            provenance='Licensed references and CC0 asset provenance/credits are recorded in the portable package; material availability is not ecological qualification.')
    elif topic == 'captures':
        if is_preset:
            result.update(images=item['isaac'], source_images=item['minecraft'])
        else:
            groups=[(name, build/'assembled'/name) for name in ('preview','overview')]
            groups.extend(('extra_'+p.parent.name.removeprefix('job_')+'_'+p.name.removeprefix('attempt_'),p/'capture') for p in completed_views(root,key))
            result['images']=[]
            for name, directory in groups:
                report=read_json(directory/'preview_result.json')
                if report['scene_sha256']!=summary['scene_sha256']:raise ValueError('Capture scene identity changed')
                result['images'].extend(dict(url=f'/results/{key}/{name}/{index}',
                    caption=frame['pose'].get('name'),kind='aerial' if frame['pose']['kind']=='aerial_overview' else 'ground',
                    pose=frame['pose']) for index,frame in enumerate(report['frames']))
        result.update(total_images=len(result['images']), gallery=summary['view_url'])
        # Keep tool responses bounded after many requests. The gallery retains all captures.
        result['images']=result['images'][-24:]
        result['image_list_scope']='Latest 24 images; all remain in the gallery.'
    elif topic == 'qualification':
        replay=_read(root/'evidence/demo/portable_replays.json')
        replay=next((r for r in replay.get('results',[]) if r['preset']==preset), None) if is_preset else None
        result.update(qualification=summary['qualification'],
            exact_ground_collision=_read(Path(record['scene']).parent.parent/'exact_collision.json'),
            route=record.get('stages',{}).get('route',{'status':'not_recorded'}),
            portable_replay=replay or {'status':'not_run_for_this_conversion'},
            full_motion='not_qualified_for_this_world', full_realism='not_qualified',
            actual_navigation_stack='not_run_not_supplied',
            automatic_visual_iteration='excluded_by_explicit_user_instruction')
    elif topic == 'delivery':
        delivery=_read(root/'artifacts/demo/downloads'/((preset if is_preset else key)+'.json'))
        matches=delivery.get('scene_sha256')==summary['scene_sha256']
        archive=Path(delivery['archive']) if delivery else None
        ready=bool(matches and archive and archive.is_file() and archive.stat().st_size==delivery['bytes'])
        result.update(download_ready=ready,
            bytes=delivery.get('bytes') if matches else None, sha256=delivery.get('sha256') if matches else None,
            url='/download/'+(preset if is_preset else key) if ready else None,
            format='ZIP: portable USD + local assets/materials/HDRI + provenance + frozen native replay helpers and original sensor references',
            exclusions='Source saves, Minecraft client reference, credentials and later requested captures are not in the world ZIP.',
            instructions=capabilities(root)['operation']['open_download'])
    else:
        raise ValueError('Unknown evidence topic')
    return result


def timing_evidence(root):
    path=Path(root)/'evidence/demo/actual_conversion_e2e.json'
    record=_read(path)
    construction=_read(Path(root)/'evidence/demo/construction_conversion.json')
    return dict(status=record.get('status','not_run'), elapsed_seconds=record.get('elapsed_seconds'),
        region='256 m Frozen coast', captures=record.get('capture_evidence'),
        additional_runs=[dict(region='256 m Jungle Rivers',status=construction['status'],
            elapsed_seconds=construction['elapsed_seconds_including_queue_and_packaging'],
            scope='Queue submission through native construction, all five planned ground and three aerial captures, and portable ZIP packaging.',
            download_bytes=construction['download']['bytes'])] if construction.get('status')=='pass' else [],
        scope='One measured source-to-native-captures run. ZIP delivery was added later and is excluded from that timing. No guaranteed ETA for another biome or a larger world.')
