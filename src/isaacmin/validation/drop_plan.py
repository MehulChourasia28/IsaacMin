"""Place stability probes on measured footprints near the original scout sites.

This plans a drop; it never declares a successful physical contact. Missing
locations remain explicit, and the camera plan is independent of these choices.
"""
from __future__ import annotations

import numpy as np

from .capture_plan import ground_support


def plan_exterior_drops(mesh, requested, source, *, origin, search_radius_m=4.):
    """Use the same 40 cm / 20 mm landing criteria as cave stability probes.

    Retain the original one-metre release height. Search locally at half-metre
    spacing and within one metre of the original floor elevation. Every sampled
    landing column must have known dry source coverage and the same top surface.
    """
    if not 0 <= search_radius_m <= 4:
        raise ValueError('Exterior drop search must remain within four metres')
    requested = list(requested)
    if not requested:
        return [], []
    spacing = float(source['sample_spacing_m'])
    height = source['height']
    wet = source['water_validity'] & (source['water_height'] >= height)
    valid = source['validity'] & ~wet
    lower = np.asarray(source['min_xz'], float)
    ox, _, oz = origin

    def dry(points):
        points = np.asarray(points, float)
        sxz = np.column_stack((points[:, 0] + ox, -points[:, 1] + oz))
        cells = np.floor((sxz-lower)/spacing).astype(int)
        okay = ((cells[:, 0] >= 0) & (cells[:, 0] < height.shape[1])
                & (cells[:, 1] >= 0) & (cells[:, 1] < height.shape[0]))
        ids = np.flatnonzero(okay)
        okay[ids] &= valid[cells[ids, 1], cells[ids, 0]]
        return okay

    axis = np.arange(-search_radius_m, search_radius_m+.001, .5)
    offsets = sorted(((float(x), float(y)) for x in axis for y in axis
                      if x*x+y*y <= search_radius_m**2+1e-9),
                     key=lambda p: (p[0]**2+p[1]**2, p[0], p[1]))
    candidate_xy = np.concatenate([np.asarray(spec['position'])[:2]+offsets for spec in requested])
    floors, normals = ground_support(mesh, candidate_xy)
    original_z = np.repeat([spec['ground_z'] for spec in requested], len(offsets))
    eligible = (dry(candidate_xy) & np.isfinite(floors) & (normals[:, 2] > .9)
                & (np.abs(floors-original_z) <= 1.))
    eligible_ids = np.flatnonzero(eligible)
    footprint_offsets = np.asarray([(x, y) for x in np.linspace(-.2, .2, 9)
                                   for y in np.linspace(-.2, .2, 9)])
    measurements = {}
    if len(eligible_ids):
        xy = (candidate_xy[eligible_ids, None, :]+footprint_offsets[None, :, :]).reshape(-1, 2)
        # Topmost hits check the full sampled descent, including ledges beside
        # a valid centre. Rays from the centre height could miss an obstruction.
        sampled_z, sampled_normals = ground_support(mesh, xy)
        source_dry = dry(xy).reshape(-1, 81)
        for index, z, nz, known_dry in zip(eligible_ids, sampled_z.reshape(-1, 81),
                                          sampled_normals[:, 2].reshape(-1, 81), source_dry):
            complete = bool(np.isfinite(z).all() and np.isfinite(nz).all())
            spread = float(np.ptp(z)) if complete else None
            same_floor = complete and bool(np.max(np.abs(z-floors[index])) <= .02)
            measurements[int(index)] = {
                'sample_count': 81, 'width_m': .4, 'all_samples_hit': complete,
                'height_spread_m': spread, 'same_source_floor': same_floor,
                'all_samples_known_dry_source': bool(known_dry.all()),
                'suitable': bool(same_floor and spread <= .02 and np.all(nz >= .9) and known_dry.all()),
                'status': 'planned_not_simulated'}

    contacts, missing = [], []
    for i, specification in enumerate(requested):
        chosen = None
        examined = 0
        for index in range(i*len(offsets), (i+1)*len(offsets)):
            examined += 1
            footprint = measurements.get(index)
            if not footprint or not footprint['suitable']:
                continue
            point = np.r_[candidate_xy[index], floors[index]]
            if any(np.linalg.norm(point-np.asarray(p['support_world_xyz'])) < .5 for p in contacts):
                continue
            chosen = dict(specification, position=(point+[0, 0, 1.]).tolist(),
                ground_z=float(point[2]), support_world_xyz=point.tolist(),
                support_normal=normals[index].tolist(), footprint_measurement=footprint,
                original_probe_position=list(specification['position']),
                original_ground_z=float(specification['ground_z']),
                relocation_horizontal_m=float(np.linalg.norm(point[:2]-np.asarray(specification['position'])[:2])),
                search_radius_m=search_radius_m, candidates_examined=examined,
                status='planned_not_simulated')
            break
        if chosen is None:
            missing.append({'original_probe_index': i, 'original_probe_position': list(specification['position']),
                'original_ground_z': float(specification['ground_z']), 'surface_scope': 'exterior',
                'search_radius_m': search_radius_m, 'candidates_examined': examined,
                'reason': 'No distinct stable dry 40 cm final-mesh footprint near the requested site; not simulated'})
        else:
            contacts.append(chosen)
    return contacts, missing
