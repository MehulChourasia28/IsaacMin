from pathlib import Path
from isaacmin.io import read_json,atomic_json,sha256_file,utc_now


def presets(root):
    return {p['id']:p for p in read_json(Path(root)/'state/demo_presets.json')['presets']}


def refresh(root):
    root=Path(root).resolve();items=[];media={}
    replay_path=root/'evidence/demo/portable_replays.json'
    replays={r['preset']:r for r in read_json(replay_path)['results']} if replay_path.is_file() else {}
    sources={p['id']:p for p in read_json(root/'state/source_worlds.json')['worlds']}
    for ident,preset in presets(root).items():
        build=Path(preset['build']);record=read_json(build/'outdoor_build.json')
        terrain=read_json(build/'terrain/terrain.json');bounds=terrain['bounds_source_xz']
        item=dict(id=ident,title=preset['title'],subtitle=preset['subtitle'],
            center=sources[ident]['center_minecraft_xz'],extent_m=[bounds[2]-bounds[0],bounds[3]-bounds[1]],
            bounds_source_xz=bounds,
            isaac=[],minecraft=[],download=None,qualification='Development world; full realism, motion and contact qualification pending')
        if ident in replays:
            replay=replays[ident].get('comparison') or {}
            item['portable_replay']=dict(status=replay.get('status','failed_without_comparison'),scope='One aerial frame from an extracted download; remaining coverage not run')
            item['qualification']+=('; one aerial package replay passed' if replay.get('status')=='static_component_pass'
                else '; strict package pixel replay failed — see progress report')
        def add_image(path,digest,caption,kind,**extra):
            path=Path(path).resolve()
            if not path.is_relative_to(root/'artifacts') or path.is_symlink() or sha256_file(path)!=digest:
                raise ValueError('Demo image is outside its evidence root or changed')
            key=digest[:24];media[key]={'path':str(path),'sha256':digest,'bytes':path.stat().st_size}
            return dict(id=key,url='/media/'+key,caption=caption,kind=kind,**extra)
        captures=[Path(preset['capture'])]
        overview=root/preset.get('overview_capture','artifacts/demo/captures/'+ident+'/overview_1')
        if (overview/'render.json').is_file() and read_json(overview/'render.json')['status']=='actual_native_capture_complete':
            captures.append(overview/'capture')
        from .views import completed_views
        captures.extend(path/'capture' for path in completed_views(root,'preset_'+ident))
        for capture in captures:
            report=read_json(capture/'preview_result.json')
            if report['status']!='actual_visual_preview_complete' or report['scene_sha256']!=sha256_file(Path(record['scene'])):
                raise ValueError('Demo capture does not match selected native world')
            for frame in report['frames']:
                pose=frame['pose'];aerial=pose.get('kind')=='aerial_overview'
                caption=pose.get('name','Ground view').replace('_',' ').capitalize()
                item['isaac'].append(add_image(capture/frame['rgb'],frame['rgb_sha256'],caption,
                    'aerial' if aerial else 'ground',samples=frame.get('path_tracing_sample_budget'),
                    camera_height_m=pose.get('scout_height_m'),frame=frame['frame']))
        source=root/'artifacts/demo/minecraft'/ident/'render.json'
        if source.is_file():
            report=read_json(source)
            if report['status']=='actual_source_capture_complete' and report['source_save_sha256']==record['identity']['source']:
                for image in report['images']:
                    item['minecraft'].append(add_image(source.parent/image['file'],image['sha256'],image['caption'],'source'))
        delivery=root/'artifacts/demo/downloads'/(ident+'.json')
        if delivery.is_file():
            d=read_json(delivery);path=Path(d['archive'])
            if d['scene_sha256']==sha256_file(Path(record['scene'])) and path.is_file() and path.stat().st_size==d['bytes']:
                item['download']=dict(url='/download/'+ident,bytes=d['bytes'],sha256=d['sha256'],format='ZIP · USD + local assets')
        items.append(item)
    result=dict(updated_at_utc=utc_now(),presets=items,media=media)
    atomic_json(root/'artifacts/demo/catalog.json',result)
    return result


def load(root):
    path=Path(root)/'artifacts/demo/catalog.json'
    data=read_json(path) if path.is_file() else refresh(root)
    from .library import visibility
    hidden=visibility(root)['hidden_presets']
    for item in data['presets']:
        if item['id'] in hidden:
            item.update(isaac=[],download=None,qualification='No saved world. Create an Isaac world from this Minecraft preset.')
            item.pop('portable_replay',None)
    allowed={image['id'] for item in data['presets'] for kind in ('minecraft','isaac') for image in item.get(kind,[])}
    if hidden:data['media']={key:value for key,value in data['media'].items() if key in allowed}
    return data


def planner_status(root):
    from isaacmin.adapters.nim import planner_model
    path=Path(root)/'evidence/demo/planner_capability.json'
    data=read_json(path) if path.is_file() else {}
    return dict(model=planner_model(root),harness='OpenClaw',harness_version='2026.9.7',status='available' if data.get('status')=='pass' and data.get('model')==planner_model(root) else 'unavailable',
        reason=data.get('reason','Hosted tool qualification passed' if data.get('status')=='pass' else 'Hosted tool qualification pending'),scope='Bounded demo tools; no realism verdict')
