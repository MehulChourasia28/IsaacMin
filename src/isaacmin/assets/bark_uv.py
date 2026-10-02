"""Metric bark coordinates from actual disconnected bevel tubes and native UVs."""
import numpy as np


def metric_bark_uv(vertices, edges, loop_vertices, native_uv, repeat_m=(.6, .6)):
    """Preserve each tube seam; measure circumference and centreline arc length.

    Blender curve conversion puts normalized spline parameter in native U and
    circumference in native V. A whole-branch0..1 mapping stretches a photograph
    across branches of very different lengths. Returned U wraps the measured
    tube circumference, and V follows measured centreline distance in metres.
    The repeat is an explicit inference when a provider publishes no scale.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    edges = np.asarray(edges, dtype=np.int64)
    loops = np.asarray(loop_vertices, dtype=np.int64)
    uv = np.asarray(native_uv, dtype=np.float32)
    repeat = np.asarray(repeat_m, dtype=float)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or edges.ndim != 2 or edges.shape[1] != 2
            or loops.ndim != 1 or uv.shape != (len(loops), 2) or repeat.shape != (2,)
            or not np.isfinite(vertices).all() or not np.isfinite(uv).all()
            or not np.isfinite(repeat).all() or np.any(repeat <= 0)):
        raise ValueError('Finite native tube geometry, UVs and positive metric repeat are required')
    if (not len(vertices) or not len(edges) or not len(loops)
            or min(edges.min(), loops.min()) < 0 or max(edges.max(), loops.max()) >= len(vertices)):
        raise ValueError('Invalid native tube incidence')
    if np.any(uv < -1e-6) or np.any(uv > 1 + 1e-6):
        raise ValueError('Expected unmodified normalized Blender bevel UVs')
    labels = np.arange(len(vertices), dtype=np.int64)
    for _ in range(128):
        a, b = labels[edges[:, 0]], labels[edges[:, 1]]
        if np.array_equal(a, b):
            break
        np.minimum.at(labels, np.maximum(a, b), np.minimum(a, b))
        while True:
            compressed = labels[labels]
            if np.array_equal(labels, compressed):
                break
            labels = compressed
    else:
        raise ValueError('Native bevel components did not converge')
    parameter = np.full(len(vertices), np.inf, dtype=np.float32)
    np.minimum.at(parameter, loops, uv[:, 0])
    if not np.isfinite(parameter).all() or np.max(abs(parameter[loops] - uv[:, 0])) > 1e-6:
        raise ValueError('Native tube U is not one consistent longitudinal parameter per vertex')
    key = (labels.astype(np.uint64) << np.uint64(32)) | parameter.view(np.uint32).astype(np.uint64)
    ring_keys, ring_index = np.unique(key, return_inverse=True)
    counts = np.bincount(ring_index)
    if np.any(counts < 3):
        raise ValueError('Bark projection requires actual bevel rings of at least three vertices')
    centres = np.column_stack([np.bincount(ring_index, weights=vertices[:, axis]) / counts for axis in range(3)])
    ring_radius = np.bincount(ring_index, weights=np.linalg.norm(vertices - centres[ring_index], axis=1)) / counts
    components = ring_keys >> np.uint64(32)
    starts = np.r_[True, components[1:] != components[:-1]]
    distances = np.r_[0., np.linalg.norm(centres[1:] - centres[:-1], axis=1)]
    distances[starts] = 0.
    prefix = np.cumsum(distances)
    arc = prefix - np.maximum.accumulate(np.where(starts, prefix, 0.))
    ring = ring_index[loops]
    output = np.column_stack((uv[:, 1] * 2 * np.pi * ring_radius[ring] / repeat[0],
                              arc[ring] / repeat[1])).astype(np.float32)
    if not np.isfinite(output).all() or not np.any(arc > 0):
        raise ValueError('Degenerate native tube cannot carry measured bark coordinates')
    evidence = {'method': 'native_bevel_components_measured_circumference_and_centreline_arc',
        'repeat_m': repeat.tolist(), 'scale_provenance': 'explicit botanical engineering candidate; provider scale unmeasured',
        'components': int(starts.sum()), 'rings': len(ring_keys),
        'maximum_branch_arc_m': float(arc.max()), 'maximum_tube_radius_m': float(ring_radius.max()),
        'native_longitudinal_axis': 'U', 'metric_longitudinal_axis': 'V',
        'seam_policy': 'original face-varying circumferential seam retained',
        'geometry_positions_changed': 0, 'target_renderer_qualification': 'not_run'}
    return output, evidence
