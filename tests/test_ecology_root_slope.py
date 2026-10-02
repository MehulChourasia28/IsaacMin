"""Analytic surface regressions; these do not claim native world execution."""
import numpy as np
import pytest
from isaacmin.assembly import ecology as candidate


def curved_support(x, y, hint):
    # Centre and roots at +-0.1m all have height zero, while each root sits
    # on a 40-degree part of a continuous quartic surface.
    a = .1
    factor = np.tan(np.radians(40)) / (2*a**3)
    z = factor*x*x*(x*x-a*a)
    gradient = factor*(4*x**3-2*a*a*x)
    normal = np.array([-gradient, 0., 1.]); normal /= np.linalg.norm(normal)
    return {'z': z, 'normal': normal.tolist(), 'surface_id': 'analytic', 'mesh_sha256': 'unit_fixture'}


def test_centre_eligibility_and_contact_fit_cannot_admit_steep_root_support():
    prototype = {'contact_anchors_local_m': [[-.1, 0., 0.], [.1, 0., 0.]]}
    old_compatibility, error = candidate.ground_prototype(prototype, 0., 0., 1., 0., curved_support)
    assert error is None and old_compatibility['contact_offset_max_m'] < 1e-12
    grounded, error = candidate.ground_prototype(prototype, 0., 0., 1., 0., curved_support,
                                               max_support_slope_degrees=30.)
    assert grounded is None and error == 'root_footprint_slope_limit'
    grounded, error = candidate.ground_prototype(prototype, 0., 0., 1., 0., curved_support,
                                               max_support_slope_degrees=45.)
    assert error is None and abs(grounded['maximum_root_support_slope_degrees']-40.) < 1e-10


@pytest.mark.parametrize('limit', [-1., 91., float('nan')])
def test_invalid_limits_are_rejected(limit):
    with pytest.raises(ValueError, match='support-slope bound'):
        candidate.ground_prototype({'contact_anchors_local_m': [[0., 0., 0.]]}, 0., 0., 1., 0., curved_support,
                                   max_support_slope_degrees=limit)


def test_missing_anchor_normals_fail_closed():
    def support(x, y, hint):
        result = curved_support(x, y, hint)
        if hint is not None:
            result.pop('normal')
        return result
    grounded, error = candidate.ground_prototype({'contact_anchors_local_m': [[0., 0., 0.]]}, 0., 0., 1., 0., support,
                                               max_support_slope_degrees=30.)
    assert grounded is None and error == 'invalid_anchor_support_normal'
