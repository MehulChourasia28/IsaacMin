import json

import numpy as np

from isaacmin.contracts.coordinates import CoordinateFrame
from isaacmin.validation.cave_views import create_cave_view_plan
from isaacmin.volumes.topology import source_topology
from test_source_connectivity import cave_fixture, voxel_boundary_mesh


def test_cave_views_have_real_mesh_support_clearance_and_portal_sight(tmp_path):
    ground = cave_fixture(tmp_path)
    frame = CoordinateFrame((12, 0, 12))
    mesh = voxel_boundary_mesh(ground)
    mesh.apply_transform(frame.record()["source_to_world"])
    mesh.export(tmp_path / "ground.obj")
    result = create_cave_view_plan(tmp_path / "ground.obj", tmp_path, frame, candidate_budget=64)
    assert result["status"] == "planned"
    assert result["available_features"]["cave_ids"]
    assert result["available_features"]["portal_ids"]
    assert result["contact_probes"]
    assert result["support_routes"]
    for route in result["support_routes"]:
        assert route["route_length_m"] >= .5
        assert route["status"] == "planned_not_simulated"
        assert route["footprint_m"] == [.35, .25]
        assert len(route["support_world_xyz"]) == 7
        assert np.all(np.asarray(route["footprint_ground_world_xyz"])[:, :, 2] == 6)
    for pose in result["poses_static"]:
        assert pose["camera_nearest_triangle_m"] >= .12
        assert not mesh.contains(np.asarray([pose["position"]]))[0]
        assert pose["support_world_xyz"][2] == 6
        assert pose["support_normal"][2] == 1
    for probe in result["contact_probes"]:
        assert probe["ground_z"] == 6
        assert probe["status"] == "planned_not_simulated"


def test_unknown_source_or_filled_cave_never_gets_camera(tmp_path):
    ground = cave_fixture(tmp_path)
    frame = CoordinateFrame()
    # Actual final mesh has filled the entire original cavity.
    filled = ground.copy()
    filled[:10] = True
    mesh = voxel_boundary_mesh(filled)
    mesh.apply_transform(frame.record()["source_to_world"])
    mesh.export(tmp_path / "filled.obj")
    result = create_cave_view_plan(tmp_path / "filled.obj", tmp_path, frame)
    assert result["status"] == "no_clear_supported_views"
    assert not result["poses_static"]
    assert not result["contact_probes"]
    assert not result["support_routes"]
    assert result["missing_representative_features"]
    # The original mesh cannot make unknown source cells into valid air.
    source = dict(np.load(tmp_path / "natural_occupancy.npz", allow_pickle=False))
    source["validity"][source["occupancy"] == 0] = False
    np.savez_compressed(tmp_path / "natural_occupancy.npz", **source)
    source_topology(tmp_path)
    mesh = voxel_boundary_mesh(ground)
    mesh.apply_transform(frame.record()["source_to_world"])
    mesh.export(tmp_path / "original.obj")
    result = create_cave_view_plan(tmp_path / "original.obj", tmp_path, frame)
    assert result["candidate_source_floors"] == 0
    assert not result["poses_static"]
