import numpy as np
import trimesh

from isaacmin.validation.capture_plan import continuous_motion


def test_ground_motion_has_bounded_speed_stop_turn_reverse_and_revisit():
    mesh = trimesh.creation.box(extents=[20, 20, 1])
    mesh.apply_translation([0, 0, -.5])
    route = np.array([[-4, 0, 0], [0, 0, 0], [4, 0, 0]], float)
    poses, record = continuous_motion(route, mesh, length_limit_m=8)
    positions = np.array([p["position"] for p in poses])
    phases = {p["motion_phase"] for p in poses}
    assert phases == {"forward", "stop", "turn", "reverse"}
    assert np.allclose(positions[:, 2], .6)
    assert np.allclose(positions[0], positions[-1])
    assert np.max(np.linalg.norm(np.diff(positions, axis=0), axis=1))*30 <= .75+1e-8
    assert record["unique_horizontal_distance_target_m"] == 8
    assert record["design_duration_s"] > 2*8/.75
    for phase in ("stop", "turn"):
        for a, b in zip(poses, poses[1:]):
            if a["motion_phase"] == b["motion_phase"] == phase:
                assert np.allclose(a["position"], b["position"])


def test_motion_spans_integral_and_nonintegral_frame_lengths_without_speedup():
    mesh = trimesh.creation.box(extents=[100, 10, 1])
    mesh.apply_translation([0, 0, -.5])
    for length in (8., 10., 11.313708498984761, 40.):
        route = np.array([[-length/2, 0, 0], [length/2, 0, 0]])
        poses, _ = continuous_motion(route, mesh, length_limit_m=length)
        points = np.array([p["position"] for p in poses])
        assert np.isclose(points[:, 0].max()-points[:, 0].min(), length)
        assert np.linalg.norm(np.diff(points, axis=0), axis=1).max()*30 <= .75+1e-10
