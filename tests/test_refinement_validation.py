"""Numerical evidence tests; fabricated arrays are never native-tool evidence."""
import numpy as np

from isaacmin.io import atomic_json, sha256_file
from isaacmin.validation.refinement import validate_refinement


def fixture(tmp_path, *, altered_import=False, protected_change=False):
    r = tmp_path / "refinement"
    r.mkdir()
    shape = (4, 4)
    source = tmp_path / "source.npz"
    np.savez(source, height=np.full(shape, 10.), min_xz=[0., 0.])
    macro = tmp_path / "macro.npz"
    np.savez(macro, validity=np.ones(shape, bool), min_xz=[0., 0.], sample_offset_m=.5, sample_spacing_m=1.)
    drainage = tmp_path / "flow.npz"
    np.savez(drainage, contributing_area_m2=np.ones(shape), valid=np.ones(shape, bool),
             receiver_flat_index=np.full(shape, -1))
    protection = np.zeros(shape, bool)
    protection[0, 0] = True
    output = np.full(shape, 9.9, np.float32)
    if not protected_change:
        output[0, 0] = 10
    np.full(shape, 10., "<f4").tofile(r / "input.f32")
    np.full(shape, 9.8, "<f4").tofile(r / "bedrock.f32")
    np.full(shape, .2 if altered_import else .1, "<f4").tofile(r / "global_runoff.f32")
    np.savez(r / "refinement.npz", height=output, delta=output-10, protection=protection,
             runoff=np.ones(shape), erosion=10-output)
    np.savez(r / "exterior_delta.npz", source_height=np.full(shape, 10.), delta=output-10, protection=protection)
    atomic_json(r / "worker_result.json", {"status": "success", "backend": "HighMap.imported_flow_stream_power.cpu",
                "array_sha256": sha256_file(r / "refinement.npz")})
    atomic_json(r / "regional_flow_normalization.json", {"global_drainage_sha256": sha256_file(drainage),
                "local_tile_renormalization": False, "regional_clip_area_m2": 10.})
    return source, r, macro, drainage, tmp_path / "validation"


def test_refinement_checks_actual_arrays_and_imported_regional_field(tmp_path):
    assert validate_refinement(*fixture(tmp_path))["status"] == "pass"


def test_refinement_rejects_local_runoff_changes(tmp_path):
    report = validate_refinement(*fixture(tmp_path, altered_import=True))
    assert report["status"] == "fail"
    assert any("local flow" in message for message in report["errors"])


def test_refinement_rejects_protected_interface_motion(tmp_path):
    report = validate_refinement(*fixture(tmp_path, protected_change=True))
    assert report["status"] == "fail"
    assert any("protected" in message for message in report["errors"])
