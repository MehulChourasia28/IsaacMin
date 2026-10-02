import json

from isaacmin.assembly.water import source_water_from_ir
from isaacmin.validation.source_water import audit_source_water
from isaacmin.validation.water_interfaces import validate_water_interfaces, ensure_source_water_audit
from test_source_water import make_water_fixture


def test_independent_raw_water_interface_validation_detects_corner_and_face_loss(tmp_path):
    world, ir = make_water_fixture(tmp_path)
    audit_source_water(world, ir, tmp_path/"water_inventory_verified")
    water = source_water_from_ir(ir, origin=(0,0,0))
    path = tmp_path/"water.json"; path.write_text(json.dumps(water))
    report = validate_water_interfaces(world, ir, path, tmp_path/"pass", origin=(0,0,0))
    assert report["status"] == "pass"
    assert report["expected_face_counts"]["top"] == 5
    assert report["expected_face_counts"]["bottom"] == 2
    assert report["expected_face_counts"]["side"] == 8
    assert report["maximum_corner_error_m"] < 1e-12
    water["vertices"][0][2] += .05
    path.write_text(json.dumps(water))
    changed = validate_water_interfaces(world, ir, path, tmp_path/"changed", origin=(0,0,0))
    assert changed["status"] == "fail"
    assert any(error["kind"] == "corner_or_winding_mismatch" for error in changed["errors"])
    water = source_water_from_ir(ir, origin=(0,0,0))
    for key in ("triangles", "triangle_source_cell_xyz", "triangle_surface_kind", "triangle_scope", "triangle_body_id"):
        water[key] = water[key][2:]
    path.write_text(json.dumps(water))
    removed = validate_water_interfaces(world, ir, path, tmp_path/"removed", origin=(0,0,0))
    assert removed["status"] == "fail"
    assert len(removed["missing_faces"]) == 1


def test_source_water_audit_cache_keeps_corrupt_history_and_refreshes(tmp_path):
    world, ir = make_water_fixture(tmp_path)
    directory = ensure_source_water_audit(world, ir)
    original = (directory/"source_water_samples.npz").read_bytes()
    assert ensure_source_water_audit(world, ir) == directory
    (directory/"source_water_samples.npz").write_bytes(original+b"damaged")
    assert ensure_source_water_audit(world, ir) == directory
    assert (directory/"source_water_samples.npz").read_bytes() == original
    histories = list((directory.parent/"history").glob("*/source_water_samples.npz"))
    assert len(histories) == 1
    assert histories[0].read_bytes() == original+b"damaged"
