import json

from isaacmin.source.independent import ensure_independent_comparisons
from isaacmin.source.macro import extract_macro_surface
from test_source_fidelity import real_reader_fixture


def test_independent_comparison_cache_reuses_and_preserves_changed_evidence(tmp_path):
    ir, _, _, _ = real_reader_fixture(tmp_path)
    snapshot = tmp_path / "fixture_save"
    macro = tmp_path / "macro"
    extract_macro_surface(snapshot, macro, center=(0, 0), extent=32, spacing=1)
    first = ensure_independent_comparisons(snapshot, ir, macro)
    assert first["status"] == "pass"
    assert first["comparisons"]["region"]["action"] == "reused"
    assert first["comparisons"]["macro"]["action"] == "refreshed"
    report = ir / "independent_decode_comparison.json"
    original_report = report.read_bytes()
    original_mtime = report.stat().st_mtime_ns
    second = ensure_independent_comparisons(snapshot, ir, macro)
    assert all(item["action"] == "reused" for item in second["comparisons"].values())
    assert report.stat().st_mtime_ns == original_mtime
    raw = ir / "independent_decode_comparison_samples.json"
    changed_bytes = raw.read_bytes() + b"\n"
    raw.write_bytes(changed_bytes)
    third = ensure_independent_comparisons(snapshot, ir, macro)
    assert third["status"] == "pass"
    assert third["comparisons"]["region"]["action"] == "refreshed"
    assert third["comparisons"]["macro"]["action"] == "reused"
    from pathlib import Path
    history = Path(third["comparisons"]["region"]["history"])
    assert (history / raw.name).read_bytes() == changed_bytes
    assert (history / report.name).read_bytes() == original_report
    assert json.loads((history / "history.json").read_text())["reason"] == "sample bytes changed"
