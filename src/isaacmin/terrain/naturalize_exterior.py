"""Reusable construction form of measured candidate3; not production-selected.

All paths, coordinates and frozen limits are inputs. Original HighMap fields
remain available for material classification. The geometry applies their
refinement here, so a later exporter must explicitly prevent double application.
"""
from pathlib import Path
import time

import numpy as np
import trimesh
from scipy.interpolate import RectBivariateSpline
from scipy.ndimage import binary_dilation, distance_transform_edt, maximum_filter, minimum_filter, map_coordinates
from scipy.sparse import coo_matrix

from isaacmin.contracts.coordinates import CoordinateFrame
from isaacmin.io import atomic_json, read_json, sha256_file, hash_object
from isaacmin.source.semantics import classify
from isaacmin.validation.streaming_columns import streamed_column_intersections


def naturalize(mesh_path, source_surface_path, exterior_delta_path, frozen_budget_path,
               origin, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    paths = [Path(p).resolve() for p in (mesh_path, source_surface_path, exterior_delta_path, frozen_budget_path)]
    paths.append(Path(__file__).resolve())
    inputs = [{'path': str(p), 'sha256': sha256_file(p)} for p in paths]
    mesh_path, source_surface_path, exterior_delta_path, frozen_budget_path = paths[:4]
    budget_record = read_json(frozen_budget_path)
    budget = budget_record['budget']
    if hash_object(budget) != budget_record['sha256']:
        raise ValueError('Frozen source-fidelity budget changed')
    frame = CoordinateFrame(tuple(origin)).record()
    if budget['coordinates']['world_to_source'] != frame['world_to_source']:
        raise ValueError('Naturalization origin differs from the frozen source frame')
    source = np.load(source_surface_path, allow_pickle=False)
    refinement = np.load(exterior_delta_path, allow_pickle=False)
    height = refinement['source_height'].astype(float)
    refined = height + refinement['delta']
    known = source['validity'].astype(bool)
    if (height.ndim != 2 or min(height.shape) < 4 or not known.all()
            or not np.array_equal(height, source['height']) or not np.isfinite(refined).all()):
        raise ValueError('Naturalization requires the same complete finite source height field')
    soil_names = np.array([classify(str(n)) in {'soil', 'sediment'} for n in source['block_names']])
    soil = soil_names[source['substrate_id']]
    exposed_water = source['water_validity'] & (source['water_height'] >= height)
    cliff = maximum_filter(height, size=3) - minimum_filter(height, size=3) > 3
    excluded = (~known | ~soil | exposed_water | refinement['protection'] | cliff)
    excluded[[0, -1], :] = True
    excluded[:, [0, -1]] = True
    safe = ~binary_dilation(excluded, iterations=2)
    collar = np.clip(distance_transform_edt(safe) / 3., 0., 1.)
    collar = collar * collar * (3. - 2. * collar)
    nz, nx = height.shape
    spline = RectBivariateSpline(np.arange(nz), np.arange(nx), refined, kx=3, ky=3, s=0)
    source_spline = RectBivariateSpline(np.arange(nz), np.arange(nx), height, kx=3, ky=3, s=0)
    original_mesh = trimesh.load(mesh_path, force='mesh', process=False)
    original = np.asarray(original_mesh.vertices, float).copy()
    faces = np.asarray(original_mesh.faces, np.int32).copy()
    ox, oy, oz = origin
    mx, mz = source['min_xz']

    def grid(points):
        return points[:, 0] + ox - mx - .5, -points[:, 1] + oz - mz - .5

    gx, gz = grid(original)
    ix = np.clip(np.floor(gx + .5).astype(int), 0, nx - 1)
    iz = np.clip(np.floor(gz + .5).astype(int), 0, nz - 1)
    interior = (gx > 1) & (gx < nx-2) & (gz > 1) & (gz < nz-2)
    depth = source_spline.ev(gz, gx) - (original[:, 2] + oy)
    vertical = np.clip((1.5-depth) / .5, 0., 1.) * (depth >= -1.5)
    mobility = map_coordinates(collar, [gz, gx], order=1, mode='constant', cval=0.) * vertical * interior
    mobility[excluded[iz, ix]] = 0.
    selected = np.flatnonzero(mobility > 0)
    projected = original[selected].copy()
    iterations = []
    for iteration in range(32):
        x, z = grid(projected)
        residual = spline.ev(z, x) - oy - projected[:, 2]
        gradient_x, gradient_y = spline.ev(z, x, dy=1), -spline.ev(z, x, dx=1)
        step = .7 * residual / (1 + gradient_x**2 + gradient_y**2)
        projected += np.column_stack((-gradient_x*step, -gradient_y*step, step))
        iterations.append({'iteration': iteration, 'maximum_surface_residual_m': float(abs(residual).max(initial=0))})
        if abs(residual).max(initial=0) < 1e-7:
            break
    vertices = original.copy()
    vertices[selected] += mobility[selected, None] * (projected - original[selected])
    vertices = vertices.astype(np.float32).astype(float)
    fixed = mobility == 0
    native_original = original.astype(np.float32)
    free = np.flatnonzero(~fixed)
    free_index = np.full(len(vertices), -1, np.int32)
    free_index[free] = np.arange(len(free))
    # Construction constraints use independent original-triangle column hits.
    # They are not promoted as the independent acceptance of this result.
    hits = streamed_column_intersections(vertices.astype(np.float32), faces,
        frame['world_to_source'], source['min_xz'], height.shape)
    metrics = hits.pop('metrics')
    order = np.lexsort((hits['triangle_index'], -hits['source_y'], hits['ray_index']))
    _, starts = np.unique(hits['ray_index'][order], return_index=True)
    top = order[starts]
    ids, selected_faces = hits['ray_index'][top], hits['triangle_index'][top]
    z, x = np.divmod(ids, nx)
    points = np.column_stack((x+mx+.5-ox, -z-mz-.5+oz, hits['source_y'][top]-oy))
    bary = trimesh.triangles.points_to_barycentric(vertices[faces[selected_faces]], points)
    if not np.isfinite(bary).all() or np.any(bary < -1e-8):
        raise ValueError('Construction column hits must lie in original triangles')
    local_ids = free_index[faces[selected_faces]]
    eligible = np.sum(np.where(local_ids >= 0, bary, 0.), axis=1) >= .5
    rows = np.repeat(np.arange(int(eligible.sum())), 3)
    columns, values = local_ids[eligible].ravel(), bary[eligible].ravel()
    active = columns >= 0
    matrix = coo_matrix((values[active], (rows[active], columns[active])),
                        shape=(int(eligible.sum()), len(free))).tocsr()
    target = height[z, x] + refinement['delta'][z, x] * collar[z, x]
    rhs = target[eligible] - hits['source_y'][top][eligible]
    kinds = np.array([classify(str(n)) for n in source['block_names']])[source['substrate_id']][z, x]
    recipe = budget['recipe']
    envelope = np.where(np.isin(kinds, ['soil', 'sediment']), recipe['soil_and_sediment_baseline_m'],
                         np.where(kinds == 'rock', recipe['rock_baseline_m'], 0.))
    protected = refinement['protection'][z, x]
    numeric = budget['protected_interface_numerical_m']
    margin = min(.0001, numeric / 10)
    upper = height[z, x] + np.where(protected, 0., envelope) + numeric - margin
    lower = height[z, x] - np.where(protected, 0., envelope + recipe['highmap_max_lowering_m']) - numeric + margin
    current = hits['source_y'][top][eligible]
    low, high = lower[eligible]-current, upper[eligible]-current
    correction = np.zeros(len(free))
    projection = []
    for iteration in range(128):
        measured = matrix @ correction
        failing = np.flatnonzero((measured < low) | (measured > high))
        projection.append({'iteration': iteration, 'violating_source_columns': len(failing)})
        if not len(failing):
            break
        for row in failing:
            begin, end = matrix.indptr[row:row+2]
            idx, weights = matrix.indices[begin:end], matrix.data[begin:end]
            value = float(weights @ correction[idx])
            desired = float(np.clip(rhs[row], low[row], high[row]))
            correction[idx] += weights * ((desired-value) / float(weights @ weights))
    else:
        raise ValueError('Source-height constraint projection did not converge')
    vertices[free, 2] += correction
    candidate = vertices.astype(np.float32)
    before = np.cross(original[faces[:, 1]]-original[faces[:, 0]], original[faces[:, 2]]-original[faces[:, 0]])
    after = np.cross(candidate[faces[:, 1]].astype(float)-candidate[faces[:, 0]], candidate[faces[:, 2]].astype(float)-candidate[faces[:, 0]])
    valid = bool(np.all(np.linalg.norm(after, axis=1) > 1e-12) and np.all(np.einsum('ij,ij->i', before, after) > 0))
    np.save(output / 'vertices.npy', candidate, allow_pickle=False)
    np.save(output / 'triangles.npy', faces, allow_pickle=False)
    np.savez_compressed(output / 'movement.npz', fixed_vertex_mask=fixed, eligible_source_cells=safe,
                        vertex_indices=selected, mobility=mobility[selected], source_protection=refinement['protection'])
    # The precision subdivider consumes the exact native f32 coordinates and
    # original faces; decimal17 preserves every binary coordinate on reload.
    with (output / 'terrain.obj').open('w') as stream:
        np.savetxt(stream, candidate.astype(float), fmt='v %.17g %.17g %.17g')
        np.savetxt(stream, faces.astype(np.int64)+1, fmt='f %d %d %d')
    for item in inputs:
        if sha256_file(Path(item['path'])) != item['sha256']:
            raise ValueError('Construction input changed during execution')
    result = {'status': 'candidate_requires_independent_validation' if valid else 'failed_local_geometry',
        'inputs': inputs, 'origin': list(origin), 'refinement_applied_to_geometry': True,
        'material_reference_delta': str(exterior_delta_path), 'quality_thresholds_modified': False,
        'native_export_and_renderer': 'not_run', 'production_selected': False,
        'vertices': len(candidate), 'triangles': len(faces), 'eligible_vertices': len(selected),
        'fixed_native_coordinates_unchanged': bool(np.array_equal(candidate[fixed], native_original[fixed])),
        'projection_iterations': projection, 'field_iterations': iterations,
        'construction_column_metrics': metrics, 'elapsed_seconds': time.monotonic()-started,
        'files': [{'path': p.name, 'sha256': sha256_file(p)} for p in sorted(output.iterdir()) if p.is_file()]}
    atomic_json(output / 'naturalization.json', result)
    if not valid or not result['fixed_native_coordinates_unchanged']:
        raise ValueError('Naturalization failed coarse geometry or fixed-coordinate constraints')
    return result


def construct_cached(mesh_path, source_surface_path, exterior_delta_path, frozen_budget_path,
                     origin, output_parent):
    """Reuse construction bytes only after exact current-input verification."""
    paths = [Path(p).resolve() for p in (mesh_path, source_surface_path, exterior_delta_path, frozen_budget_path)]
    paths.append(Path(__file__).resolve())
    inputs = [{'path': str(p), 'sha256': sha256_file(p)} for p in paths]
    output = Path(output_parent) / hash_object({'inputs': inputs, 'origin': list(origin)})
    record = output / 'naturalization.json'
    if record.is_file():
        result = read_json(record)
        if (result.get('status') != 'candidate_requires_independent_validation'
                or result.get('inputs') != inputs or result.get('origin') != list(origin)
                or not result.get('fixed_native_coordinates_unchanged')):
            raise ValueError('Naturalization cache changed or failed; preserve for inspection')
        for item in result['files']:
            file = output / item['path']
            if file.is_symlink() or not file.resolve().is_relative_to(output.resolve()) or sha256_file(file) != item['sha256']:
                raise ValueError('Naturalization cache bytes changed')
    else:
        result = naturalize(*paths[:4], origin, output)
    return output / 'terrain.obj', record
