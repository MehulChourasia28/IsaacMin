import numpy as np

from isaacmin.terrain.baseline import priority_drainage, reconstruct_baseline
from isaacmin.terrain.routes import route_candidate


def test_depression_outlets_conserve_area_and_receivers_acyclic():
    h = np.ones((15, 19)) * 10
    h[1:-1, 1:-1] = 3
    h[0, 4] = 7
    valid = np.ones_like(h, bool)
    valid[5:8, 5:8] = False
    result = priority_drainage(h, valid, spacing_m=4)
    receiver = result["receiver_flat_index"].ravel()
    assert result["contributing_area_m2"][result["receiver_flat_index"] == -1].sum() == valid.sum()*16
    assert np.all(result["filled_height_m"][valid] >= h[valid])
    assert np.all(receiver[~valid.ravel()] == -2)
    for first in np.flatnonzero(valid):
        seen = set()
        node = first
        while node >= 0:
            assert node not in seen
            seen.add(node)
            node = receiver[node]


def test_cliff_and_thin_roof_protection_at_every_baseline_step():
    h = np.tile(np.floor(np.arange(30) / 3), (30, 1)).astype(float)
    h[:, 15:] += 20
    material = np.ones_like(h, int)
    material[:, 15:] = 3
    protection = np.zeros_like(h, bool)
    protection[10:15, 5:20] = True
    baseline, record = reconstruct_baseline(h, np.ones_like(h, bool), material, protection)
    assert np.array_equal(baseline[protection], h[protection])
    assert np.min(baseline[:, 15]-baseline[:, 14]) > 18
    assert record["rms_delta_m"] > 0
    assert record["role"] == "baseline_only_not_erosion"


def test_route_does_not_cross_water_or_missing_ground():
    h = np.zeros((30, 40))
    water = np.zeros_like(h, bool)
    water[:, 20] = True
    valid = np.ones_like(h, bool)
    valid[0, :] = False
    result = route_candidate(h, valid, water, spacing_m=4, centre_index=(15, 5), origin_xz=(0, 0))
    indices = result["surface_indices_zx"]
    assert all(x < 20 and z > 0 for z, x in indices)
    assert len(set(map(tuple, indices))) == len(indices)
    assert result["provenance"] == "authored_scenario_route_not_extracted_trail"
