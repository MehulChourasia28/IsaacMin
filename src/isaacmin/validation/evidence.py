"""Bind independent raw measurements to the exact candidate and frozen profile."""
from pathlib import Path

from ..io import atomic_json, hash_object, read_json, sha256_file
from ..packaging import ascii_dependency_closure
from ..security import safe_path


def bind_measurement(directory: Path, name: str, measurement: dict, paths: list[Path]) -> dict:
    scene = directory / "scene/world_physics.usda"
    if not scene.exists():
        scene = directory / "scene/world.usda"
    closure = ascii_dependency_closure(scene)
    if closure["status"] != "pass":
        raise ValueError("Cannot bind qualification measurements to unresolved native scene dependencies")
    evidence = directory / "evidence"
    files = []
    for path in paths:
        relative = path.resolve().relative_to(evidence.resolve()).as_posix()
        actual = safe_path(evidence, relative, must_exist=True)
        files.append({"path": relative, "sha256": sha256_file(actual), "bytes": actual.stat().st_size})
    record = {**measurement, "scene_content_sha256": hash_object(closure["files"]),
              "profile_sha256": read_json(directory / "quality_profile.json")["sha256"],
              "files": files}
    atomic_json(evidence / (name+".json"), record)
    return record
