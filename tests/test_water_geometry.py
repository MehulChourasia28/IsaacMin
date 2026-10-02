"""Saved-fluid geometry properties, not renderer or fluid-simulation qualification."""
import numpy as np
import trimesh

from isaacmin.assembly.water import fluid_surface_mesh


def test_stacked_still_water_keeps_aquifer_and_exterior_levels():
    kinds = np.ones((7, 5, 5), np.uint8)
    kinds[[2, 5], 1:4, 1:4] = 0
    kinds[[1, 4], 1:4, 1:4] = 2
    amount = np.where(kinds == 2, 8, 0)
    result = fluid_surface_mesh(kinds, amount, [0, 0, 0], np.full((5, 5), 4.), origin=(0, 0, 0))
    assert result['source_body_count'] == 2
    assert result['source_water_cells'] == 18
    assert result['scope_triangle_counts'] == {'aquifer': 18, 'exterior': 18}
    assert np.allclose(np.unique(np.asarray(result['vertices'])[:, 2]), [1+8/9, 4+8/9])
    mesh = trimesh.Trimesh(vertices=result['vertices'], faces=result['triangles'], process=False)
    assert (mesh.face_normals[:, 2] == 1).all()


def test_unknown_above_is_never_replaced_by_a_visible_air_interface():
    kinds = np.ones((3, 3, 3), np.uint8)
    kinds[1, 1, 1], kinds[2, 1, 1] = 2, 3
    result = fluid_surface_mesh(kinds, np.where(kinds == 2, 8, 0), [0, 0, 0], np.zeros((3, 3)), origin=(0, 0, 0))
    assert not result['triangles']
    assert result['status'] == 'incomplete'
    assert result['unknown_adjacent_faces'] == [{'cell_xyz': [1, 1, 1], 'direction_xyz': [0, 1, 0]}]


def test_falling_column_has_shared_side_boundaries_and_outward_faces():
    kinds = np.zeros((5, 5, 5), np.uint8)
    kinds[1:3, 2, 2] = 2
    amount = np.where(kinds == 2, 8, 0)
    result = fluid_surface_mesh(kinds, amount, [0, 0, 0], np.zeros((5, 5)), origin=(0, 0, 0))
    mesh = trimesh.Trimesh(vertices=result['vertices'], faces=result['triangles'], process=False)
    assert mesh.is_volume
    assert mesh.volume > 0
    assert result['surface_triangle_counts'] == {'bottom': 2, 'side': 16, 'top': 2}


def test_differing_saved_flow_levels_share_corners():
    kinds = np.ones((3, 4, 5), np.uint8)
    kinds[1, 1:3, 1:4] = 2
    kinds[2, 1:3, 1:4] = 0
    amount = np.where(kinds == 2, 8, 0)
    amount[1, 1:3, 2] = 4
    result = fluid_surface_mesh(kinds, amount, [0, 0, 0], np.zeros((4, 5)), origin=(0, 0, 0))
    vertices = np.asarray(result['vertices'])
    assert len(np.unique(vertices[:, :2], axis=0)) == len(vertices)
    assert vertices[:, 2].min() < 1+8/9
    assert vertices[:, 2].min() > 1+4/9


def test_real_halo_preserves_level_without_exporting_neighbor_coverage():
    kinds = np.zeros((4, 5, 5), np.uint8)
    kinds[0] = 1
    kinds[1] = 2
    amount = np.where(kinds == 2, 8, 0)
    coverage = np.zeros_like(kinds, bool)
    coverage[:, 1:4, 1:4] = True
    result = fluid_surface_mesh(kinds, amount, [0, 0, 0], np.ones((5, 5)),
                                origin=(0, 0, 0), coverage_mask=coverage)
    assert result['source_water_cells'] == 9
    assert result['surface_triangle_counts'] == {'top': 18}
    assert not result['uncertain_corner_neighbors']
    assert np.allclose(np.asarray(result['vertices'])[:, 2], 1+8/9)
    assert min(p[0] for p in result['vertices']) == 1
    assert max(p[0] for p in result['vertices']) == 4


def test_verified_dry_lichen_is_distinct_but_nonoccluding():
    kinds = np.ones((3, 3, 3), np.uint8)
    kinds[1, 1, 1], kinds[2, 1, 1] = 2, 5
    result = fluid_surface_mesh(kinds, np.where(kinds == 2, 8, 0), [0, 0, 0], np.zeros((3, 3)), origin=(0, 0, 0))
    assert result['surface_triangle_counts'] == {'top': 2}
    assert not result['unknown_adjacent_faces']
