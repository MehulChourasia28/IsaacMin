from __future__ import annotations

from pathlib import Path

from ..io import atomic_json, hash_object, read_json, sha256_file


DEFAULT_PROFILE = {
    "schema_version": "1.0", "name": "strict", "priority": "realism",
    "required_gates": [f"Q{i:02d}" for i in range(14)],
    "camera": {"height_m": 0.60, "scout_heights_m": [0.25, 0.60, 1.50],
               "width": 1280, "height": 720, "horizontal_fov_degrees": 90,
               "intrinsics": {"fx": 640.0, "fy": 640.0, "cx": 640.0, "cy": 360.0},
               "pixel_coordinates": "upper-left image edge is (0,0); pixel centres are (u+0.5,v+0.5)",
               "near_m": 0.02, "far_m": 5000.0, "design_hz": 30,
               "exposure_policy": "fixed_per_lighting_condition"},
    "thresholds": {"seam_gap_max_m": 0.002, "collision_p999_m": 0.010,
                   "collision_max_m": 0.020, "planted_base_p995_m": 0.020,
                   "source_samples_min": 1000, "depth_samples_min": 10000,
                   "static_views_min": 24, "held_out_fraction_min": 0.25,
                   "unique_route_target_m": 1000, "severe_visual_defects_max": 0,
                   "missing_ground_frames_max": 0, "lod_silhouette_trigger_px": 1.0},
    "scout_seed": 1729, "held_out_seed": 93497,
    "lighting_conditions": ["diffuse", "directional"],
    "collision_scope": "terrain_and_cave_support",
    "navigation_stack": "not_run_not_supplied", "human_review": "not_requested",
    "allow_silent_quality_reduction": False, "max_repairs_per_defect_class": 3,
}


def freeze_profile(path: Path, context: dict) -> dict:
    """Changed inputs create a new profile; an existing frozen profile is immutable."""
    profile = {**DEFAULT_PROFILE, "context": context}
    digest = hash_object(profile)
    record = {"profile": profile, "sha256": digest}
    if path.exists():
        if read_json(path) != record:
            raise ValueError("Frozen profile differs: create a new build/profile instead of overwriting")
    else:
        atomic_json(path, record)
    return record


def load_profile(path: Path) -> dict:
    record = read_json(path)
    if hash_object(record["profile"]) != record["sha256"]:
        raise ValueError("Frozen quality profile hash mismatch")
    if record["profile"]["thresholds"] != DEFAULT_PROFILE["thresholds"]:
        raise ValueError("Unrecognized thresholds: this validator only supports the strict version 1 profile")
    return record


def validator_hash() -> str:
    root = Path(__file__).parent
    files = {"validation/"+p.name: sha256_file(p) for p in sorted(root.glob("*.py"))}
    for name in ("mesh_validation.py", "source_evidence.py"):
        path = root.parent / "volumes" / name
        files["volumes/"+name] = sha256_file(path)
    return hash_object(files)
