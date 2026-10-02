import numpy as np
import pytest
from isaacmin.assets.bark_uv import metric_bark_uv


def tubes():
    positions, edges, loops, uvs = [], [], [], []
    for index, radius in enumerate([.1, .03]):
        offset = len(positions)
        for height in [0., .5, 2.]:
            for k in range(8):
                angle = k * np.pi / 4
                positions.append([index * 2 + radius * np.cos(angle), radius * np.sin(angle), height])
        for row in range(2):
            for k in range(8):
                ids = [offset + row*8 + k, offset + row*8 + (k+1)%8,
                       offset + (row+1)*8 + (k+1)%8, offset + (row+1)*8 + k]
                loops.extend(ids)
                edges.extend(zip(ids, ids[1:] + ids[:1]))
                uvs.extend([[row*.5, k/8], [row*.5, (k+1)/8],
                            [(row+1)*.5, (k+1)/8], [(row+1)*.5, k/8]])
    return np.asarray(positions), np.asarray(edges), np.asarray(loops), np.asarray(uvs)


def test_branch_distance_and_circumference_are_metric_and_seam_survives():
    points, edges, loops, uv = tubes()
    actual, proof = metric_bark_uv(points, edges[::-1], loops, uv, (.5, .5))
    radius = np.where(loops < 24, .1, .03)
    assert np.allclose(actual[:, 1], points[loops, 2] / .5)
    assert np.allclose(actual[:, 0], uv[:, 1] * 2 * np.pi * radius / .5)
    assert proof['components'] == 2 and proof['rings'] == 6
    assert proof['maximum_branch_arc_m'] == 2.
    # The same vertex has both seam endpoints, rather than interpolating across
    # a texture discontinuity or assigning the whole tree one normalized UV.
    assert len(np.unique(actual[loops == 0, 0])) == 2


def test_already_remapped_or_inconsistent_native_parameter_is_rejected():
    points, edges, loops, uv = tubes()
    uv[0, 0] = .25
    with pytest.raises(ValueError, match='consistent longitudinal'):
        metric_bark_uv(points, edges, loops, uv)
