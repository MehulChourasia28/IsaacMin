"""Closed native canopy references and measured root support for outdoor scenes."""
from pathlib import Path
import os
import shutil
import numpy as np
from isaacmin.io import read_json, sha256_file


def select_canopy_prototype(tree, biome, prototypes, rules):
    """Resolve a source species through reusable ecology and asset recipes."""
    from isaacmin.assembly.surface_population import random_cells
    rule=next((r for r in rules if tree['species'] in r['source_species']
        and (not r.get('source_biomes') or biome in r['source_biomes'])),None)
    if rule is None:return None
    weights=np.array([row['weight'] for row in rule['assets']],float)
    if not len(weights) or not np.isfinite(weights).all() or np.any(weights<=0):
        raise ValueError('Canopy recipe needs positive finite asset weights')
    x,z=int(np.floor(tree['source_x'])),int(np.floor(tree['source_z']))
    draw=float(random_cells(x,z,83))
    choice=rule['assets'][np.searchsorted(np.cumsum(weights)/weights.sum(),draw)]
    pool=sorted((p for p in prototypes if p['asset_id']==choice['asset_id']),key=lambda p:p['object_name'])
    if not pool:raise ValueError('Required original canopy asset unavailable: '+choice['asset_id'])
    variant=min(int(float(random_cells(x,z,84))*len(pool)),len(pool)-1)
    return pool[variant],rule['name']


def copy_canopy_libraries(workspace, scene, libraries, *, namespace='canopy_library'):
    """Copy only verified immutable native dependencies; never load provider code."""
    result = []
    for relative in libraries:
        manifest = Path(workspace) / relative
        for asset in read_json(manifest)['assets']:
            source = Path(asset['usd'])
            if sha256_file(source) != asset['usd_sha256']:
                raise ValueError('Native canopy library changed')
            if namespace not in ('canopy_library','scenery_library','shrub_library','litter_library'):
                raise ValueError('Unsupported native asset namespace')
            destination = Path(scene) / namespace / asset['asset_id']
            destination.mkdir(parents=True, exist_ok=False)
            closure = read_json(source.parent / 'native_dependency_closure.json')
            for entry in closure['files']:
                p, q = source.parent / entry['path'], destination / entry['path']
                if p.is_symlink() or not p.resolve().is_relative_to(source.parent) or sha256_file(p) != entry['sha256']:
                    raise ValueError('Canopy dependency changed or escaped its library')
                q.parent.mkdir(parents=True, exist_ok=True)
                if p.suffix in ('.mdl', '.usda', '.json'):
                    shutil.copyfile(p, q)
                else:
                    os.link(p, q)
            for proto in asset['prototypes']:
                if not proto['geometry_verified_exact'] or not proto['source_uv_verified_exact']:
                    raise ValueError('Canopy lacks native geometry/UV verification')
                result.append(dict(proto, asset_id=asset['asset_id'],
                    scene_asset='./' + (destination / source.name).relative_to(scene).as_posix(),
                    library_manifest_sha256=sha256_file(manifest)))
    return result


def supported_root_height(query, x, y, yaw_degrees, anchors, *, tolerance=.02):
    """Require the measured original base band to meet actual final ground."""
    a = np.asarray(anchors, float)
    angle = np.deg2rad(yaw_degrees)
    xx = x + a[:, 0]*np.cos(angle) - a[:, 1]*np.sin(angle)
    yy = y + a[:, 0]*np.sin(angle) + a[:, 1]*np.cos(angle)
    heights = query.heights(xx, yy) - a[:, 2]
    if not np.isfinite(heights).all() or np.ptp(heights) > tolerance:
        return None
    return float(np.mean(heights))


def embedded_woody_root(query, x, y, yaw_degrees, anchors, height_m, *,
                        maximum_burial_m=.6, maximum_burial_fraction=.03):
    """Insert a woody stem into soil; never leave its original base above it.

    Tree bases are volumetric, unlike grass contact points. A horizontal cut
    base should lie below a hillside, with the visible trunk crossing the actual
    ground. This is a construction candidate, not an ecology acceptance result.
    """
    a = np.asarray(anchors, float)
    angle = np.deg2rad(yaw_degrees)
    xx = x + a[:, 0]*np.cos(angle) - a[:, 1]*np.sin(angle)
    yy = y + a[:, 0]*np.sin(angle) + a[:, 1]*np.cos(angle)
    supports = query.heights(xx, yy) - a[:, 2]
    if not np.isfinite(supports).all():
        return None
    burial = float(np.ptp(supports))
    limit = min(maximum_burial_m, float(height_m)*maximum_burial_fraction)
    if burial > limit:
        return None
    z = float(np.min(supports))
    return z, dict(method='original woody base embedded beneath final terrain',
        maximum_sampled_base_above_ground_m=0., maximum_sampled_burial_m=burial,
        maximum_constructor_burial_m=limit, root_band_samples=len(a),
        original_mesh_preserved=True, visible_ground_intersection_qualification='not_run')
