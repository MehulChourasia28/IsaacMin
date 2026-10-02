"""Acquire original PBR resolution candidates without changing active masters."""
from pathlib import Path
import json

from .providers import PolyHaven, poly_variant, decode_image, _filename, _poly_record
from .network import ServiceError, atomic_json, digest, utcnow

ROLES = {'base_color': 'Diffuse', 'normal': 'nor_gl', 'roughness': 'Rough', 'height': 'Displacement'}


def acquire_material_candidates(workspace, output, *, resolution='4k', refresh=True):
    workspace, output = Path(workspace).resolve(), Path(output).resolve()
    if output.exists():
        raise ServiceError('immutable_candidate', 'Use a new attributable output for changed material candidates')
    if resolution not in ('4k', '8k'):
        raise ServiceError('material_resolution', 'Shared terrain candidates require original 4K or 8K channels')
    active_paths = [workspace/'state/asset_catalogue.json', workspace/'state/terrain_materials.json']
    active_hashes = {str(p): digest(p) for p in active_paths}
    active = json.loads(active_paths[1].read_text())['materials']
    provider = PolyHaven(workspace)
    catalogue, discovery = provider.catalogue('textures', refresh=refresh)
    selected = ('brown_mud_dry', 'rock_face_03', 'leafy_grass', 'forest_ground_04')
    records, materials = [], []
    for asset_id in selected:
        metadata = catalogue.get(asset_id)
        if metadata is None:
            raise ServiceError('material_unavailable', 'Required source material is absent from provider catalogue')
        files, snapshot = provider.client.metadata('https://api.polyhaven.com/files/'+asset_id, refresh=refresh)
        record = _poly_record(asset_id, metadata, discovery, snapshot, 'textures')
        spec = dict(next(m for m in active if m['asset_id'] == asset_id))
        proof = {}
        for role, provider_role in ROLES.items():
            item = poly_variant(files, provider_role, resolution, 'png')
            target = workspace/'assets/masters/poly_haven'/asset_id/metadata['files_hash']/resolution/_filename(item['url'])
            downloaded = provider.client.download(item, target)
            decoded = decode_image(target)
            dimension = int(resolution[:-1])*1024
            if decoded.get('status') != 'pass' or (decoded['width'], decoded['height']) != (dimension, dimension):
                raise ServiceError('material_decode', 'Provider resolution and fully decoded original dimensions disagree')
            downloaded.update(role=provider_role, resolution=resolution, format='png', image_validation=decoded)
            record['files'].append(downloaded)
            spec[role] = str(target.resolve())
            proof[role] = {'sha256': downloaded['sha256'], 'width': dimension, 'height': dimension,
                           'source_texels_per_m': dimension/spec['repeat_m'], 'provider_role': provider_role}
        spec.update(qualification='not_run', original_channel_proof=proof, resolution=resolution,
                    source_page=record['source_page'], licence=record['licence'], upstream_revision=record['upstream_revision'])
        records.append(record);materials.append(spec)
    if any(digest(Path(p)) != h for p,h in active_hashes.items()):
        raise ServiceError('changed_active_materials', 'Active selections changed during candidate acquisition')
    result = {'schema_version': 1, 'created_at_utc': utcnow(), 'status': 'original_channels_acquired_and_decoded',
              'qualification': 'not_run', 'materials': materials, 'provider_records': records,
              'active_inputs_preserved': active_hashes, 'producer_sha256': digest(Path(__file__)),
              'resolution_policy': 'Exact original provider PNG bytes; no resampling or synthetic channel generation',
              'promotion_policy': 'Separate candidate only; explicit pipeline selection and native qualification required'}
    atomic_json(output, result)
    return result
