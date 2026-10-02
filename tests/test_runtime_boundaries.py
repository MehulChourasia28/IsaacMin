from pathlib import Path

import pytest
import jsonschema

from isaacmin.agent.registry import Registry, Tool, BUILD_ID, EMPTY
from isaacmin.packaging import ascii_dependency_closure


def test_runtime_model_cannot_execute_unregistered_code_or_paths():
    registry = Registry()
    called = []
    registry.register(Tool("inspect_build", "inspect", BUILD_ID, lambda args: called.append(args) or {}))
    with pytest.raises(ValueError):
        registry.call("shell", {"command": "whoami"})
    for args in ({"build_id": "../../source"}, {"build_id": "build_0123456789abcdef", "quality": "fast"},
                 {"build_id": "build_0123456789abcdef", "python": "print(1)"}, {}):
        with pytest.raises(jsonschema.ValidationError):
            registry.call("inspect_build", args)
    assert not called
    registry.call("inspect_build", {"build_id": "build_0123456789abcdef"})
    assert len(called) == 1


def test_portability_rejects_missing_absolute_and_escape_dependencies(tmp_path):
    root = tmp_path / "scene"
    root.mkdir()
    scene = root / "world.usda"
    for path in ("missing.png", "/tmp/unrelated.png", "../outside.png", "https://example.invalid/material.png"):
        scene.write_text('#usda 1.0\ndef Shader "Tex" {\n asset inputs:file = @'+path+'@\n}\n')
        assert ascii_dependency_closure(scene)["status"] == "fail"
    (root / "image.png").write_bytes(b"fixture")
    scene.write_text('#usda 1.0\ndef Shader "Tex" {\n asset inputs:file = @image.png@\n}\n')
    result = ascii_dependency_closure(scene)
    assert result["status"] == "pass"
    assert len(result["files"]) == 2


def test_binary_usd_cannot_receive_ascii_closure_pass(tmp_path):
    scene = tmp_path / "world.usdc"
    scene.write_bytes(b"PXR-USDC" + b"\x00"*100)
    assert ascii_dependency_closure(scene)["status"] == "fail"


def test_empty_stage_with_old_export_counts_is_not_geometry_world(tmp_path):
    from isaacmin.validation.profile import freeze_profile
    from isaacmin.validation.runner import evaluate
    directory = tmp_path / "worlds/build_0123456789abcdef"
    (directory / "scene").mkdir(parents=True)
    (directory / "scene/world.usda").write_text('#usda 1.0\n')
    profile = freeze_profile(directory / "quality_profile.json", {})
    manifest = {"build_id": "build_0123456789abcdef", "run_id": "fixture",
                "stages": {"scene": {"status": "success", "terrain_vertices": 1000, "terrain_polygons": 2000}}}
    assert evaluate(tmp_path, directory, manifest, profile)["stage"] is None


def test_package_reuse_verifies_bytes_and_every_payload_record(tmp_path):
    from isaacmin.io import atomic_json, sha256_file
    from isaacmin.packaging import verify_package_files
    world = tmp_path / "world.usda"
    world.write_text("#usda 1.0\n")
    records = [{"path": world.name, "sha256": sha256_file(world), "bytes": world.stat().st_size}]
    atomic_json(tmp_path / "package.json", {"files": records})
    assert verify_package_files(tmp_path, records)
    world.write_text("damaged")
    assert not verify_package_files(tmp_path, records)
    assert not verify_package_files(tmp_path, [])
