"""Actual triangle-query fixtures; no PhysX or final-world qualification claim."""
import numpy as np
import trimesh

from isaacmin.validation.drop_plan import plan_exterior_drops


def scene():
    mesh = trimesh.creation.box(extents=[12, 12, 1])
    mesh.apply_translation([6, -6, -.5])
    z = np.zeros((12, 12))
    source = dict(height=z, water_height=z, water_validity=np.zeros_like(z, bool),
                  validity=np.ones_like(z, bool), min_xz=[0, 0], sample_spacing_m=1.)
    requested = [{'position': [6., -6., 1.], 'ground_z': 0., 'surface_scope': 'exterior'}]
    return mesh, source, requested


def test_flat_support_preserves_site_and_one_metre_release_height():
    mesh, source, requested = scene()
    contacts, missing = plan_exterior_drops(mesh, requested, source, origin=(0, 0, 0))
    assert not missing and len(contacts) == 1
    assert contacts[0]['position'] == requested[0]['position']
    assert contacts[0]['relocation_horizontal_m'] == 0
    assert contacts[0]['footprint_measurement']['sample_count'] == 81
    assert contacts[0]['status'] == 'planned_not_simulated'


def test_flat_centre_cannot_admit_a_ledge_under_the_cube_edge():
    mesh, source, requested = scene()
    ledge = trimesh.creation.box(extents=[.08, .6, .3])
    ledge.apply_translation([6.18, -6., .15])
    mesh = trimesh.util.concatenate([mesh, ledge])
    contacts, missing = plan_exterior_drops(mesh, requested, source, origin=(0, 0, 0))
    assert not missing and len(contacts) == 1
    probe = contacts[0]
    assert 0 < probe['relocation_horizontal_m'] <= 4
    assert probe['footprint_measurement']['height_spread_m'] <= .02
    assert probe['original_probe_position'] == requested[0]['position']
    assert requested[0]['position'] == [6., -6., 1.]


def test_no_stable_floor_is_reported_missing_without_relaxing_the_limit():
    mesh, source, requested = scene()
    mesh.vertices[:, 2] += .2*(mesh.vertices[:, 0]-6)
    contacts, missing = plan_exterior_drops(mesh, requested, source, origin=(0, 0, 0))
    assert contacts == [] and len(missing) == 1
    assert missing[0]['original_probe_position'] == requested[0]['position']


def test_source_water_or_unknown_footprint_is_not_a_drop_site():
    mesh, source, requested = scene()
    for field in ('water_validity', 'validity'):
        source[field][:] = field == 'water_validity'
        contacts, missing = plan_exterior_drops(mesh, requested, source, origin=(0, 0, 0))
        assert not contacts and missing
        source[field][:] = field == 'validity'
