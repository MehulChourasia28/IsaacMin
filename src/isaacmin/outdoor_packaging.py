"""Self-contained development deliveries for the above-ground constructor."""
from pathlib import Path
import shutil
import tempfile

from isaacmin.io import atomic_json, hash_object, read_json, sha256_file, utc_now
from isaacmin.packaging import ascii_dependency_closure
from isaacmin.security import safe_path
from isaacmin.validation.provenance import scan_delivery_file


def package_outdoor(workspace, build, *, output=None, reference_capture=None):
    workspace, build = Path(workspace).resolve(), Path(build).resolve()
    record = read_json(build / 'outdoor_build.json')
    if record.get('status') != 'geometry_world':
        raise ValueError('Finish native outdoor construction before packaging')
    scene = Path(record['scene'])
    closure = ascii_dependency_closure(scene)
    if closure['status'] != 'pass':
        raise ValueError('Outdoor dependencies are incomplete or changed')
    if not (scene.parent / 'render_configuration.json').is_file():
        raise ValueError('Outdoor scene has no saved illumination/camera configuration')
    identity = hash_object(dict(build=record['identity'], scene_closure=closure))
    destination = Path(output).resolve() if output else workspace / 'packages' / ('outdoor_' + identity[:16] + '_unqualified')
    if destination.exists() or destination.is_relative_to(build):
        raise ValueError('Preserve existing evidence; choose a fresh package outside the build')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.outdoor-package-', dir=destination.parent))
    for entry in closure['files']:
        source = safe_path(scene.parent, entry['path'], must_exist=True)
        target = safe_path(staging, entry['path'])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    atomic_json(staging / 'dependency_closure.json', closure)
    atomic_json(staging / 'construction.json', record)
    atomic_json(staging / 'qualification.json', dict(status='unqualified_development_artifact',
        evidence='actual native construction and any included development views only',
        appearance='not_qualified', motion='not_run', continuous_long_range='not_run',
        full_contact_qualification='not_run', portable_reproduction='not_run',
        navigation_stack='not_run_not_supplied', original_underground_scope='superseded_by_user'))
    for source, name in ((workspace / 'configs/strict.quality_profile.json', 'quality_profile.json'),
                         (workspace / 'state/surface_navigation_scope.json', 'scope.json'),
                         (workspace / 'ASSET_CREDITS.md', 'ASSET_CREDITS.md'),
                         (workspace / 'isaac_scripts/load_world.py', 'load_world.py'),
                         (workspace / 'artifacts/bootstrap/dependency-lock.json', 'dependency-lock.json')):
        shutil.copyfile(source, staging / name)
    preview = scene.parent.parent / 'preview'
    request_path = scene.parent.parent / 'preview_request.json'
    capture_set='full_development'
    capture_producer=workspace/'isaac_scripts'
    frozen=build/'producer_sources/isaac_scripts'
    if reference_capture is None and frozen.is_dir():
        for name in ('capture_outdoor_visual.py','material_log.py','visible_instance_labels.py'):
            if sha256_file(frozen/name)!=record['identity']['producer_sha256']['isaac_scripts/'+name]:
                raise ValueError('Frozen conversion capture producer changed')
        capture_producer=frozen
    if reference_capture is not None:
        capture_root=Path(reference_capture).resolve()
        receipt=read_json(capture_root/'render.json')
        if (receipt.get('status')!='actual_native_capture_complete'
                or receipt['build_manifest_sha256']!=sha256_file(build/'outdoor_build.json')
                or Path(receipt['scene']).resolve()!=scene.resolve()):
            raise ValueError('External capture must have completed against this exact build')
        preview=capture_root/'capture'
        request_path=capture_root/'request.json'
        capture_producer=capture_root/'producer_sources'
        for name,digest in receipt['producer_sha256'].items():
            if sha256_file(safe_path(capture_producer,name,must_exist=True))!=digest:
                raise ValueError('Captured producer changed')
        capture_set='separate_native_'+receipt['views']
    elif not (preview/'preview_result.json').is_file():
        preview=scene.parent.parent/'focus'
        request_path=scene.parent.parent/'focus_request.json'
        capture_set='focused_development_reduced_view_coverage'
    reproduction = dict(status='not_run', reason='No matching actual capture available')
    if (preview / 'preview_result.json').is_file():
        report = read_json(preview / 'preview_result.json')
        request = read_json(request_path)
        if (report['scene_sha256'] != sha256_file(scene) or not request.get('preserve_authored_scene')
                or report['status'] != 'actual_visual_preview_complete'
                or report['source_closure_sha256']!=sha256_file(scene.parent/'native_dependency_closure.json')):
            raise ValueError('Capture does not match the saved authored scene')
        probe = staging / 'reproduction'
        reference = probe / 'reference'
        reference.mkdir(parents=True)
        for frame in report['frames']:
            for role in ('rgb', 'depth', 'instance_segmentation'):
                source = safe_path(preview, frame[role], must_exist=True)
                if sha256_file(source) != frame[role + '_sha256']:
                    raise ValueError('Actual reference capture changed')
                shutil.copyfile(source, reference / source.name)
        report.update(scope='static development reproduction only; no full portable qualification',
            reference_scene=str(scene.name))
        report.pop('scene', None)
        atomic_json(probe / 'reference.json', report)
        config = read_json(scene.parent / 'render_configuration.json')
        request.update(scene=scene.name, output='set_by_portable_runner', hdri=config['hdri'],
            lighting_parameters=config['lighting_parameters'], preserve_authored_scene=True)
        atomic_json(probe / 'request.json', request)
        for name in ('capture_outdoor_visual.py','material_log.py','visible_instance_labels.py'):
            shutil.copyfile(capture_producer/name,probe/name)
        for name in ('reproduce_outdoor.py','compare_outdoor_reproduction.py',
                     'compare_reproduction.py','reproduce_package.py'):
            shutil.copyfile(workspace / 'isaac_scripts' / name, probe / name)
        reproduction = dict(status='ready_not_run', frames=len(report['frames']),
            capture_set=capture_set,
            command='python3 reproduction/reproduce_outdoor.py --package PACKAGE --isaac-python ISAAC_PYTHON --output NEW_DIRECTORY',
            scope='fresh-process RGB/depth/labels comparison; motion and physical qualification separate')
    (staging / 'README.txt').write_text(
        'IsaacMin outdoor development world — NOT REALISM-QUALIFIED\n\n'
        'Open using the pinned Isaac Python runtime:\n'
        '  env -u NVIDIA_API_KEY /path/to/Isaac/python.sh load_world.py --package .\n\n'
        'All USD, HDRI and texture references are local. The save and NVIDIA key are not needed.\n'
        'The loader restores the recorded camera response and selects the derived ground-height camera.\n'
        'See qualification.json, scope.json, quality_profile.json and ASSET_CREDITS.md.\n'
        'Static replay, when present:\n'
        '  python3 reproduction/reproduce_outdoor.py --package . --isaac-python /path/to/Isaac/python.sh --output /new/output\n'
        'A successful open or static replay does not qualify realism, motion, contacts or navigation.\n')
    inventory=[]
    for path in sorted(staging.rglob('*')):
        if not path.is_file():continue
        scan=scan_delivery_file(path)
        if scan['credential_pattern_found'] or scan['forbidden_filename']:
            raise ValueError('Delivery contains a forbidden credential or secret filename')
        inventory.append(dict(path=path.relative_to(staging).as_posix(),
            sha256=scan['sha256'],bytes=path.stat().st_size))
    manifest=dict(schema_version=1,status='unqualified_development_artifact',
        entrypoint=scene.name,files=inventory,runtime_modules=closure.get('runtime_modules',[]),
        source_save_included=False,credentials_included=False,
        credential_scan_scope='NVIDIA token syntax in all delivered bytes and forbidden secret filenames',
        reproduction=reproduction,package_sha256=hash_object(inventory),
        build_identity_sha256=identity,created_at_utc=utc_now())
    atomic_json(staging/'package.json',manifest)
    staging.rename(destination)
    return dict(status='unqualified_development_artifact',package=str(destination),
        files=len(inventory),bytes=sum(p['bytes'] for p in inventory),
        reproduction=reproduction,package_sha256=manifest['package_sha256'])
