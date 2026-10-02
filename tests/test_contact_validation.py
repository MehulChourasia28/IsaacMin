"""Synthetic raw-array tests; these do not claim a native PhysX run."""
import numpy as np
import pytest
import trimesh

from isaacmin.io import atomic_json, sha256_file
from isaacmin.validation.contact import validate_contact_rays


def fixture(tmp_path, *, missing=False, wrong_collider=False):
    mesh = trimesh.creation.box(extents=(4, 4, 1))
    mesh.apply_translation((0, 0, -.5))
    ground = tmp_path / "ground.obj"
    mesh.export(ground)
    source = tmp_path / "surface.npz"
    np.savez(source, height=np.zeros((4, 4)), validity=np.ones((4, 4), bool),
             min_xz=(-2, -2), sample_spacing_m=1.)
    capture = tmp_path / "capture"
    capture.mkdir()
    origins = np.array([[-1., 0, .2], [0., 0, .2], [1., 0, .2]])
    points = origins.copy()
    points[:, 2] = 0
    hits = np.array([True, True, not missing])
    if missing:
        points[-1] = np.nan
    rays = capture / "physx_contact_rays.npz"
    np.savez(rays, origins=origins, directions=np.tile((0, 0, -1.), (3, 1)),
             hit=hits, position=points, collision_index=np.zeros(3, np.int32), max_distance_m=2.)
    atomic_json(capture / "physx_contact_rays.json", {"sha256": sha256_file(rays),
                "final_ground_sha256": sha256_file(ground), "terrain_paths": ["/Terrain"],
                "collider_paths": ["/Probe" if wrong_collider else "/Terrain"]})
    return ground, capture, source


def test_independent_queries_preserve_missing_native_hits(tmp_path):
    ground, capture, source = fixture(tmp_path, missing=True)
    report = validate_contact_rays(ground, capture, source, tmp_path / "validation", origin=(0, 0, 0))
    assert report["status"] == "fail"
    assert report["failed_ray_indices"] == [2]
    data = np.load(tmp_path / "validation/ground_samples.npz")
    assert len(data["render_points"]) == 3
    assert np.isnan(data["collision_points"][-1]).all()


def test_independent_queries_reject_probe_as_ground(tmp_path):
    ground, capture, source = fixture(tmp_path, wrong_collider=True)
    with pytest.raises(ValueError, match="other than the final supporting terrain"):
        validate_contact_rays(ground, capture, source, tmp_path / "validation", origin=(0, 0, 0))


def test_independent_queries_compare_exported_triangles(tmp_path):
    ground, capture, source = fixture(tmp_path)
    report = validate_contact_rays(ground, capture, source, tmp_path / "validation", origin=(0, 0, 0))
    assert report["status"] == "pass"
    assert report["source_strata"] == {"exterior": 3}
    assert report["metrics"]["max_m"] < 1e-12


@pytest.mark.parametrize("condition,expected", [("tilted_support", "pass"), ("floating", "fail"), ("penetrating", "fail")])
def test_full_pose_support_handles_tilt_and_rejects_false_contacts(tmp_path, condition, expected):
    from scipy.spatial.transform import Rotation
    from isaacmin.validation.contact import validate_dropped_probes
    mesh = trimesh.creation.box(extents=(4, 4, 1))
    mesh.apply_translation((0, 0, -.5))
    ground = tmp_path / "ground.obj"
    mesh.export(ground)
    capture = tmp_path / "capture"
    capture.mkdir()
    angle = .3
    q = Rotation.from_rotvec([angle, 0, 0]).as_quat()
    height = .1*(np.cos(angle)+np.sin(angle))
    if condition == "floating":
        height += .1
    elif condition == "penetrating":
        height -= .08
    atomic_json(capture / "capture_request.json", {"contact_probes": [{"position": [0, 0, 1.], "ground_z": 0.}]})
    atomic_json(capture / "capture_result.json", {"physx_contact_rays": {"final_ground_sha256": sha256_file(ground)},
        "contacts": [{"id": 0, "probe_shape": "cube", "edge_length_m": .2,
                      "settled_position": [0, 0, height], "settled_orientation_wxyz": [q[3], *q[:3]],
                      "linear_velocity_mps": [0, 0, 0], "angular_velocity_radps": [0, 0, 0], "status": "fail"}]})
    report = validate_dropped_probes(ground, capture, tmp_path / "validation")
    assert report["status"] == expected
    assert report["probes"][0]["native_upright_estimate_status"] == "fail"
