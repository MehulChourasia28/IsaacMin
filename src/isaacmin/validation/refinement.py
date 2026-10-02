"""Independent numerical checks on actual HighMap output and regional runoff."""
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from ..io import atomic_json, read_json, sha256_file


def validate_refinement(source_surface: Path, refinement: Path, macro_surface: Path,
                        drainage: Path, output: Path) -> dict:
    source = np.load(source_surface, allow_pickle=False)
    native = np.load(refinement / "refinement.npz", allow_pickle=False)
    delta = np.load(refinement / "exterior_delta.npz", allow_pickle=False)
    macro = np.load(macro_surface, allow_pickle=False)
    regional = np.load(drainage, allow_pickle=False)
    execution = read_json(refinement / "worker_result.json")
    normalization = read_json(refinement / "regional_flow_normalization.json")
    errors = []
    shape = source["height"].shape
    original = np.fromfile(refinement / "input.f32", dtype="<f4").reshape(shape)
    bedrock = np.fromfile(refinement / "bedrock.f32", dtype="<f4").reshape(shape)
    imported_flow = np.fromfile(refinement / "global_runoff.f32", dtype="<f4").reshape(shape)
    if execution.get("status") != "success" or execution.get("backend") != "HighMap.imported_flow_stream_power.cpu":
        errors.append("No successful native HighMap imported-drainage execution")
    if execution.get("array_sha256") != sha256_file(refinement / "refinement.npz"):
        errors.append("Native refinement byte provenance changed")
    for key in ("height", "runoff", "erosion", "delta", "protection"):
        if native[key].shape != shape or not np.isfinite(native[key]).all():
            raise ValueError("Invalid native refinement array: " + key)
    changed = int(np.count_nonzero(native["height"] != original))
    if not changed:
        errors.append("HighMap refinement is a no-op")
    lowering = original.astype(float)-native["height"]
    if np.any(native["height"] < bedrock-1e-5) or np.any(lowering < -1e-5) or np.any(lowering > .2+1e-5):
        errors.append("HighMap exceeded frozen erosion/bedrock bounds")
    protection = delta["protection"]
    if protection.dtype != bool or not np.array_equal(protection, native["protection"]):
        errors.append("Native and exported protection differ")
    if np.any(native["height"][protection] != original[protection]):
        errors.append("HighMap changed protected elevations")
    if not np.array_equal(delta["source_height"], source["height"]):
        errors.append("Exterior warp is anchored to another source surface")
    if not np.allclose(source["height"]+delta["delta"], native["height"], atol=1e-5, rtol=0):
        errors.append("Exported deformation does not equal the actual native output")
    if np.max(np.abs(delta["delta"][protection]), initial=0) > 1e-5:
        errors.append("Baseline changed protected source elevations")
    if normalization["global_drainage_sha256"] != sha256_file(drainage):
        errors.append("Refinement used different regional drainage bytes")
    if normalization.get("local_tile_renormalization") is not False:
        errors.append("Runoff was locally renormalized")
    clip = float(normalization["regional_clip_area_m2"])
    expected_clip = float(10*np.sqrt(np.mean(regional["contributing_area_m2"][macro["validity"]])))
    if not np.isclose(clip, expected_clip, atol=1e-9, rtol=0):
        errors.append("Regional runoff normalization does not match the declared full-source policy")
    zz, xx = np.indices(shape)
    center = macro["min_xz"]+macro["sample_offset_m"]
    gz = (source["min_xz"][1]+zz+.5-center[1])/float(macro["sample_spacing_m"])
    gx = (source["min_xz"][0]+xx+.5-center[0])/float(macro["sample_spacing_m"])
    expected_flow = map_coordinates(np.clip(regional["contributing_area_m2"]/max(clip, 1e-12), 0, 1),
                                    [gz, gx], order=1, mode="nearest")
    flow_error = float(np.max(np.abs(expected_flow-imported_flow)))
    if flow_error > 1e-7:
        errors.append("Imported local flow differs from world-aligned regional samples")
    receiver = regional["receiver_flat_index"]
    region_valid = regional["valid"]
    area = regional["contributing_area_m2"]
    expected_area = float(region_valid.sum())*float(macro["sample_spacing_m"])**2
    area_error = abs(float(area[receiver == -1].sum())-expected_area)
    if area_error > 1e-5 or np.any(receiver[region_valid] < -1):
        errors.append("Regional drainage has unaccounted source area")
    output.mkdir(parents=True, exist_ok=True)
    samples = output / "refinement_measurements.npz"
    np.savez_compressed(samples, native_lowering_m=lowering, exterior_delta_m=delta["delta"],
                        protection=protection, imported_regional_flow=imported_flow,
                        expected_regional_flow=expected_flow)
    inputs = [source_surface, macro_surface, drainage, refinement / "worker_result.json",
              refinement / "refinement.npz", refinement / "exterior_delta.npz",
              refinement / "regional_flow_normalization.json", refinement / "input.f32",
              refinement / "bedrock.f32", refinement / "global_runoff.f32"]
    report = {"status": "pass" if not errors else "fail", "errors": errors,
              "changed_samples": changed, "max_lowering_m": float(lowering.max()),
              "protected_samples": int(protection.sum()), "regional_flow_max_error": flow_error,
              "regional_outlet_area_error_m2": area_error, "samples": samples.name,
              "samples_sha256": sha256_file(samples),
              "inputs": [{"path": str(p), "sha256": sha256_file(p)} for p in inputs],
              "limits": ["Numerical execution and regional consistency only; exported forms and target appearance require separate checks",
                         "Priority-flood drainage is a regional diagnostic, not a calibrated hydrological simulation"]}
    atomic_json(output / "refinement_validation.json", report)
    return report
