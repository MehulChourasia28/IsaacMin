"""Raw synthetic trajectories exercise validation; these are not native evidence."""
import json

import numpy as np
import pytest
import trimesh

from isaacmin.io import atomic_json, hash_object, read_json, sha256_file
from isaacmin.validation.traversal import validate_traversal


def make_fixture(tmp_path, *, height=.1, missing=False, wrong_collider=False):
    mesh = trimesh.creation.box(extents=(4, 4, 1))
    mesh.apply_translation([0, 0, -.5])
    ground = tmp_path / "final_ground.obj"
    mesh.export(ground)
    scene = tmp_path / "world.usda"
    scene.write_text("#usda 1.0\n")
    files = [{"path": scene.name, "sha256": sha256_file(scene), "bytes": scene.stat().st_size}]
    capture = tmp_path / "capture"
    capture.mkdir()
    request = {"scene": str(scene), "scene_dependency_files": files, "scene_content_sha256": hash_object(files),
               "surface_scope": "exterior", "final_ground_sha256": sha256_file(ground),
               "footprint_xyz_m": [.35, .25, .2], "route_ground_xyz": [[0, 0, 0], [.05, 0, 0]]}
    atomic_json(capture / "traversal_request.json", request)
    samples = []
    for i in range(3):
        position = [i*.025, 0, height]
        points = np.array([[x+position[0], y, height-.1] for x in [-.175, 0, .175] for y in [-.125, 0, .125]])
        hits = [{"hit": True, "position": [float(x), float(y), 0.], "collision": "/Probe" if wrong_collider else "/Terrain"}
                for x, y, z in points]
        if missing and i == 1:
            hits[-1] = {"hit": False, "position": None}
        samples.append({"simulation_time_s": i/30, "position": position, "orientation_wxyz": [1, 0, 0, 0],
                        "footprint_bottom_samples_world": points.tolist(), "terrain_support_rays": hits,
                        "applied_force_n": [1, 0, 0], "measured_net_contact_force_n": [0, 0, 9.81],
                        "linear_velocity_mps": [.75, 0, 0]})
    raw = capture / "traversal_samples.jsonl"
    raw.write_text("".join(json.dumps(s)+"\n" for s in samples))
    result = {"status": "pass", "request_sha256": sha256_file(capture / "traversal_request.json"),
              "samples_sha256": sha256_file(raw), "final_ground_sha256": sha256_file(ground),
              "sample_count": len(samples), "terrain_paths": ["/Terrain"], "footprint_xyz_m": [.35, .25, .2],
              "completed_waypoint_indices": [0, 1], "height_teleports_after_initialization": 0}
    atomic_json(capture / "traversal_result.json", result)
    return ground, capture


@pytest.mark.parametrize("height,missing,expected", [(.1, False, "pass"), (.15, False, "fail"),
                                                       (.07, False, "fail"), (.1, True, "fail")])
def test_every_dynamic_footprint_is_supported(tmp_path, height, missing, expected):
    ground, capture = make_fixture(tmp_path, height=height, missing=missing)
    result = validate_traversal(ground, capture, tmp_path / "out")
    assert result["status"] == expected
    assert result["frame_count"] == 3
    raw = np.load(tmp_path / "out/traversal_comparison.npz")
    assert len(raw["render_points"]) == 27


def test_traversal_rejects_contact_with_another_body(tmp_path):
    ground, capture = make_fixture(tmp_path, wrong_collider=True)
    with pytest.raises(ValueError, match="unidentified collider"):
        validate_traversal(ground, capture, tmp_path / "out")


def test_traversal_rejects_changed_raw_data(tmp_path):
    ground, capture = make_fixture(tmp_path)
    with (capture / "traversal_samples.jsonl").open("a") as stream:
        stream.write("{}\n")
    with pytest.raises(ValueError, match="measurement bytes changed"):
        validate_traversal(ground, capture, tmp_path / "out")
