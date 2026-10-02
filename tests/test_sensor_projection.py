import numpy as np
import pytest

from isaacmin.validation.sensors import camera_rays


def test_asymmetric_camera_transform_and_axial_depth_are_consistent():
    k = {"fx": 640, "fy": 640, "cx": 639.5, "cy": 359.5}
    transform = np.eye(4)
    transform[3, :3] = [13, -21, 7]
    pixels = np.array([[639.5, 359.5], [1279.5, 359.5], [639.5, 999.5]])
    origins, dirs, cosines = camera_rays(pixels, k, transform)
    assert np.allclose(origins, [[13, -21, 7]]*3)
    assert np.allclose(dirs[0], [0, 0, -1])
    assert dirs[1, 0] > 0 and dirs[2, 1] < 0
    plane_z = -3
    distance = (plane_z-origins[:, 2])/dirs[:, 2]
    assert np.allclose(distance*cosines, 10)


def test_reject_scaled_camera_transform():
    m = np.eye(4)*2
    with pytest.raises(ValueError):
        camera_rays([[0, 0]], {"fx": 1, "fy": 1, "cx": 0, "cy": 0}, m)
