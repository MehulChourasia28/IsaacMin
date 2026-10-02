import numpy as np
import pytest
import trimesh

from isaacmin.validation.axis_parity import contains_axis_consensus


@pytest.mark.parametrize("case", ["box", "cavity", "separate", "inverted", "open", "rotated"])
def test_axis_matches_independent_diagonal_for_adversarial_geometry(case):
    mesh = trimesh.creation.box(extents=[4, 6, 8])
    if case == "cavity":
        inner = trimesh.creation.box(extents=[2, 2, 2]); inner.invert()
        mesh = trimesh.util.concatenate([mesh, inner])
    elif case == "separate":
        other = mesh.copy(); other.apply_translation([7, 1, 0])
        mesh = trimesh.util.concatenate([mesh, other])
    elif case == "inverted":
        mesh.invert()
    elif case == "open":
        mesh.update_faces(np.arange(len(mesh.faces)-2))
    elif case == "rotated":
        mesh.apply_transform(trimesh.transformations.rotation_matrix(.377, [1, 2, 3]))
    points = np.random.default_rng(203).uniform([-5, -7, -6], [10, 8, 6], (1024, 3))
    actual, evidence = contains_axis_consensus(mesh, points)
    if case != "open":
        assert np.array_equal(actual, mesh.contains(points))
    else:
        # Open meshes still fail the caller's required topology gate. Ambiguous
        # opposing rays retain fallback evidence rather than silently agreeing.
        assert evidence["axis_parity_disagreement"].any()
        ids = evidence["axis_fallback_indices"]
        assert np.array_equal(actual[ids], mesh.contains(points[ids]))
    assert evidence["axis_crossing_counts"].shape == (len(points), 6)


def test_shared_edges_tangencies_and_surface_queries_preserved():
    mesh = trimesh.creation.box()
    points = np.asarray([[0, 0, 0], [0, .5, 0], [.5, .5, .5], [-.5, 0, 0],
                         [.6, .5, 0], [0, 0, .5-1e-5], [0, 0, .5+1e-5]])
    actual, evidence = contains_axis_consensus(mesh, points, batch_size=1)
    assert np.array_equal(actual, mesh.contains(points))
    assert len(evidence["axis_fallback_indices"]) >= 3
    assert actual[0] and actual[-2] and not actual[-1]


def test_invalid_query_is_not_filtered():
    with pytest.raises(ValueError, match="finite"):
        contains_axis_consensus(trimesh.creation.box(), [[np.nan, 0, 0]])
