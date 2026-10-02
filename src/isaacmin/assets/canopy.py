"""Explicit native canopy candidates with their original execution provenance."""
from pathlib import Path

from isaacmin.io import atomic_json, read_json, sha256_file, utc_now
from isaacmin.security import safe_path


def _owned(root, path):
    path = Path(path)
    if path.is_absolute():
        path = path.relative_to(root)
    return safe_path(root, path.as_posix(), must_exist=True)


def select_canopy_candidates(workspace, generations):
    """Select actual native outputs for assembly; this does not qualify views.

    Each path is a generation.json from an immutable execution directory.
    Multiple independent seeds/forms for a species may live in separate blends.
    Old state is retained, and old generated assets are never overwritten.
    """
    root = Path(workspace).resolve()
    assets = []
    for generation in generations:
        source = _owned(root, generation)
        record = read_json(source)
        species = record['recipe_species']
        if species not in ('oak', 'white_birch') or record.get('status') != 'external_tool_verified':
            raise ValueError('Expected an actually generated, supported canopy candidate')
        paths = {source}
        def bind(path, expected):
            actual = _owned(root, path)
            if sha256_file(actual) != expected:
                raise ValueError('Frozen native canopy input changed: ' + str(actual))
            paths.add(actual)
            return actual
        blend = bind(record['output_blend'], record['output_sha256'])
        if blend.parent != source.parent:
            raise ValueError('Canopy blend and its original generation record must share a directory')
        def execution(manifest_path, manifest_sha, script_sha):
            manifest_path = bind(manifest_path, manifest_sha)
            manifest = read_json(manifest_path)
            if manifest['execution_script_sha256'] != script_sha:
                raise ValueError('Canopy execution identity mismatch')
            bind(manifest['executed_script'], script_sha)
            for item in manifest['files']:
                bind(item['path'], item['sha256'])
            return manifest_path
        original = execution(record['input_manifest'], record['input_manifest_sha256'],
                             record['generator_script_sha256'])
        bind(original.parent / 'request.json', record['request_sha256'])
        for step in record.get('postprocesses', []):
            manifest = execution(step['input_manifest'], step['input_manifest_sha256'],
                                 step['execution_script_sha256'])
            bind(manifest.parent / 'request.json', step['request_sha256'])
            bind(step['parent_generation'], step['parent_generation_sha256'])
            bind(step['parent_blend'], step['parent_blend_sha256'])
            if step.get('operation') == 'metric_bark_uv' and step.get('geometry_bytes_unchanged') is not True:
                raise ValueError('Bark correction lacks measured native geometry preservation')
        maps = _owned(root, record['maps_manifest'])
        paths.add(maps)
        for path in maps.parent.rglob('*'):
            if path.is_file():
                paths.add(_owned(root, path))
        assets.append({'asset_id': record['asset_id'], 'species': species, 'blend': str(blend),
            'generation': str(source), 'generation_sha256': sha256_file(source),
            'variant_count': len(record['variants']), 'objects': [v['objects'] for v in record['variants']],
            'files': [{'path': str(p), 'bytes': p.stat().st_size, 'sha256': sha256_file(p)} for p in sorted(paths)],
            'generator_execution_script_sha256': record['generator_script_sha256'],
            'code_binding': 'original frozen execution scripts and unchanged original inputs',
            'isaac_qualification': 'not_run', 'appearance_status': 'candidate_pending_full_world_qualification'})
    if not assets:
        raise ValueError('Select at least one actual native canopy')
    state = root / 'state/procedural_assets.json'
    if state.exists():
        previous = root / 'state/procedural_asset_history' / (sha256_file(state) + '.json')
        if not previous.exists():
            atomic_json(previous, read_json(state))
    result = {'schema_version': 2, 'created_at_utc': utc_now(), 'assets': assets,
        'qualification': 'not_run', 'selection_producer_sha256': sha256_file(Path(__file__)),
        'geometry_sharing': 'native object prototypes per independent variant; verify target USD instancing'}
    atomic_json(state, result)
    return result


def load_canopy_candidates(workspace):
    root = Path(workspace).resolve()
    result = {}
    for asset in read_json(root / 'state/procedural_assets.json')['assets']:
        if not asset.get('generator_execution_script_sha256'):
            raise ValueError('Selected canopy lacks original execution provenance')
        for item in asset['files']:
            if sha256_file(_owned(root, item['path'])) != item['sha256']:
                raise ValueError('Selected canopy dependency changed')
        source = _owned(root, asset.get('generation', str(Path(asset['blend']).parent / 'generation.json')))
        if asset.get('generation_sha256') and sha256_file(source) != asset['generation_sha256']:
            raise ValueError('Selected canopy generation record changed')
        generated = read_json(source)
        if generated['generator_script_sha256'] != asset['generator_execution_script_sha256']:
            raise ValueError('Selected canopy generator identity differs')
        if generated['recipe_species'] != asset['species'] or generated['asset_id'] != asset['asset_id']:
            raise ValueError('Selected canopy botanical identity differs')
        if sha256_file(_owned(root, generated['output_blend'])) != generated['output_sha256']:
            raise ValueError('Selected native canopy blend changed')
        for variant in generated['variants']:
            result.setdefault(asset['species'], []).append((generated, variant))
    return result
