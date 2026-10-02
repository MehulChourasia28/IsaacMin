import json

import numpy as np
import trimesh

from isaacmin.validation.local_features import diagnose_local_features
from test_source_fidelity import real_reader_fixture


def test_local_stencils_separate_source_edges_and_expose_caps_and_overlap(tmp_path):
    ir, source_mesh, frame, _ = real_reader_fixture(tmp_path)
    mesh = source_mesh.copy()
    mesh.apply_transform(frame.record()["source_to_world"])
    mesh_path = tmp_path / "exact.obj"
    mesh.export(mesh_path)
    exact = diagnose_local_features(ir, mesh_path, tmp_path / "local_exact", frame.record()["world_to_source"],
                                   stencil_offsets=(-.35, 0., .35))
    assert exact["qualification"].startswith("diagnostic_only")
    assert exact["roofs"]["source_interior"]["samples"] > 0
    assert exact["roofs"]["source_transition_or_boundary"]["samples"] > 0
    assert exact["roofs"]["all"]["minimum_measured_thickness_m"] == 1
    assert exact["roofs"]["all"]["missing_support_at_source_midheight"] == 0
    assert exact["portals"]["samples_with_target_segment_crossings"] == 0
    assert exact["potential_self_intersection_or_tangency_flags"] == []
    with np.load(tmp_path / "local_exact/local_feature_samples.npz") as data:
        rows = data["roof_source_intervals"]
        corner = np.flatnonzero((rows[:, 0] == -6.5) & (rows[:, 1] == -1.5))[0]
        context = data["roof_source_side_context"][corner]
        assert not context[:4].any()  # Cardinal neighbors alone miss this edge.
        assert context[4] == 1  # The negative-X/negative-Z diagonal is open.
    graph = json.loads((ir / "topology/source_topology_graph.json").read_text())
    portal = graph["portals"][0]["faces"][0]
    cap = trimesh.creation.box(extents=(.3, .3, .3))
    cap.apply_translation(portal["center_xyz"])
    with np.load(tmp_path / "local_exact/local_feature_samples.npz") as data:
        roof = data["roof_source_intervals"][0]
    # A second closed cube inside the existing roof creates overlapping solids:
    # watertightness/positive signed volume alone cannot detect that defect.
    overlap = trimesh.creation.box(extents=(.3, .3, .3))
    overlap.apply_translation((roof[0], (roof[2]+roof[3])/2, roof[1]))
    broken = trimesh.util.concatenate((source_mesh, cap, overlap))
    broken.apply_transform(frame.record()["source_to_world"])
    broken_path = tmp_path / "broken.obj"
    broken.export(broken_path)
    result = diagnose_local_features(ir, broken_path, tmp_path / "local_broken", frame.record()["world_to_source"],
                                    stencil_offsets=(-.35, 0., .35))
    assert result["mesh_positive_closed_volume"]
    assert result["portals"]["samples_with_target_segment_crossings"] > 0
    assert result["portals"]["samples_with_inside_endpoint_or_centre"] > 0
    assert result["potential_self_intersection_or_tangency_flags"]
    assert result["self_intersection_scope"].endswith("not_run")
