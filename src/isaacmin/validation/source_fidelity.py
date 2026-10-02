"""Source-fidelity measurements with budgets frozen before target evaluation.

Protected support has zero authored displacement. Its numerical allowance comes
from the stricter existing seam/contact tolerance, not from the voxel size or a
failing reconstruction. These measurements do not qualify materials or views.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from ..io import atomic_json, hash_object, sha256_file
from ..security import safe_path
from ..source.nbt import SourceError
from ..source.semantics import classify
from ..volumes import mesh_validation
from ..volumes.support_validation import structural_support_for_validation


def freeze_source_fidelity_profile(quality_profile: dict, output: Path) -> dict:
    """Freeze the already-declared terrain recipe and strict support budgets."""
    if hash_object(quality_profile["profile"]) != quality_profile["sha256"]:
        raise ValueError("Quality profile hash mismatch")
    thresholds = quality_profile["profile"]["thresholds"]
    coordinates = quality_profile["profile"].get("context", {}).get("coordinates")
    if not coordinates or coordinates.get("metres_per_block") != 1 or "world_to_source" not in coordinates:
        raise ValueError("Source fidelity requires the frozen one-metre source/world coordinate frame")
    numerical = min(float(thresholds["seam_gap_max_m"]), float(thresholds["collision_p999_m"]))
    if not 0 < numerical <= .002:
        raise ValueError("Protected support cannot use a looser allowance than the strict2mm boundary budget")
    budget = {"schema_version": 1, "quality_profile_sha256": quality_profile["sha256"],
              "recipe": {"soil_and_sediment_baseline_m": .85, "rock_baseline_m": .25,
                         "highmap_max_lowering_m": .20, "soil_displacement_peak_to_peak_m": .02,
                         "protected_authored_displacement_m": 0., "protected_air_cover_m": 6.},
              "protected_interface_numerical_m": numerical,
              "protected_roof_max_thickness_loss_m": 2*numerical,
              "minimum_source_comparisons": max(1000, int(thresholds["source_samples_min"])),
              "source_scale_m_per_block": 1., "sampling": "every known source-column centre and every observed source cave support interval",
              "coordinates": coordinates,
              "rationale": ["Baseline envelopes and one-sided HighMap lowering are existing generation limits",
                            "Peak-to-peak0.02m soil displacement permits at most0.01m signed vertex offset",
                            "Protected cave ceilings/floors and structural masonry have zero authored displacement",
                            "Each protected interface may deviate only by the stricter existing seam/contact numerical budget",
                            "Two protected interfaces can lose at most the sum of their individual numerical allowances",
                            "A thick roof meeting an unprotected exterior inherits only that column's declared exterior lowering budget"],
              "validator_sha256": sha256_file(Path(__file__))}
    record = {"budget": budget, "sha256": hash_object(budget)}
    output = Path(output)
    if output.exists():
        if json.loads(output.read_text()) != record:
            raise ValueError("Source-fidelity budget is frozen; create a new build/profile for a justified change")
    else:
        atomic_json(output, record)
    return record


def _read_hashed_files(directory, report):
    records = {}
    for entry in report.get("files", []):
        path = safe_path(directory, entry["path"], must_exist=True)
        if sha256_file(path) != entry["sha256"]:
            raise ValueError("Independent evidence file changed: " + entry["path"])
        records[entry["path"]] = path
    return records


def evaluate_source_fidelity(ir_dir: Path, mesh_path: Path, output: Path, *,
                             frozen_budget_path: Path, exterior_delta_path: Path,
                             topology_samples_path: Path, topology_graph_path: Path,
                             support_manifest_path: Path | None = None) -> dict:
    """Measure final source-space support against bound independent ray evidence.

    The two topology reports must have been produced for this exact mesh and IR
    by the current independent validator. Its raw column intersections are
    reused as measurements, never replaced by generator SDF or success flags.
    """
    ir_dir, mesh_path, output = Path(ir_dir), Path(mesh_path), Path(output)
    budget_record = json.loads(Path(frozen_budget_path).read_text())
    budget = budget_record["budget"]
    if hash_object(budget) != budget_record["sha256"] or budget["validator_sha256"] != sha256_file(Path(__file__)):
        raise ValueError("Source-fidelity budget or validator changed after freezing")
    ir_path = ir_dir / "world_ir.json"
    ir = json.loads(ir_path.read_text())
    ir_hash, mesh_hash = sha256_file(ir_path), sha256_file(mesh_path)
    _read_hashed_files(ir_dir, ir)
    with np.load(ir_dir / "natural_occupancy.npz", allow_pickle=False) as data:
        occupancy, validity, minimum = data["occupancy"], data["validity"], data["min_xyz"].astype(int)
    with np.load(ir_dir / "terrain_surface.npz", allow_pickle=False) as data:
        source_height = data["height"].astype(float)
        surface_valid = data["validity"]
        names = data["block_names"][data["substrate_id"]]
        water_mask = data["water_validity"]
    with np.load(ir_dir / "topology/source_topology_labels.npz", allow_pickle=False) as data:
        covered, portals = data["covered_air"], data["portal_labels"]
    structural, support_context = structural_support_for_validation(ir_dir, occupancy, support_manifest_path)
    supporting = ((occupancy == 1) & validity) | structural
    with np.load(exterior_delta_path, allow_pickle=False) as data:
        if not np.array_equal(data["source_height"], source_height):
            raise ValueError("Declared exterior refinement belongs to another source surface")
        delta, protection = data["delta"].astype(float), data["protection"]
    if delta.shape != source_height.shape or protection.shape != delta.shape or protection.dtype != bool or not np.isfinite(delta[surface_valid]).all():
        raise ValueError("Invalid exterior refinement field")
    sample_report = json.loads(Path(topology_samples_path).read_text())
    graph_report = json.loads(Path(topology_graph_path).read_text())
    for report in (sample_report, graph_report):
        mesh_validation.verify_mesh_backend_evidence(report)
        if report["mesh_sha256"] != mesh_hash or report["source_ir_sha256"] != ir_hash:
            raise ValueError("Independent topology evidence is stale for the mesh/source")
        if report.get("source_topology_sha256") != sha256_file(ir_dir / "topology/source_topology_graph.json"):
            raise ValueError("Independent topology evidence belongs to another source graph")
        if support_manifest_path is not None and report.get("structural_support", {}).get("manifest_sha256") != sha256_file(Path(support_manifest_path)):
            raise ValueError("Independent topology did not validate the declared structural support")
        if report.get("validator_sha256") != sha256_file(Path(mesh_validation.__file__)):
            raise ValueError("Independent topology was produced by another validator revision")
        if report.get("mesh_to_source") != budget["coordinates"]["world_to_source"]:
            raise ValueError("Independent mesh transform differs from the frozen source/world frame")
    sample_files = _read_hashed_files(Path(topology_samples_path).parent, sample_report)
    graph_files = _read_hashed_files(Path(topology_graph_path).parent, graph_report)
    with np.load(sample_files["source_mesh_samples.npz"], allow_pickle=False) as samples:
        parity_errors = int(np.count_nonzero(samples["expected_inside"] != samples["mesh_inside"]))
        unique_inside_samples = len(np.unique(samples["source_xyz"], axis=0))
    with np.load(graph_files["column_ray_reconstruction.npz"], allow_pickle=False) as data:
        hit_rays, hit_y = data["hit_ray_index"], data["hit_source_y"]
        roof_intervals = data["roof_vertical_intervals"]
        protected_voxel_errors = int(data["occupancy_mismatch"].sum())
    nz, nx = source_height.shape
    if graph_report["ray_count"] != nz*nx or not np.isfinite(hit_y).all() or np.any(hit_rays < 0) or np.any(hit_rays >= nz*nx):
        raise ValueError("Independent ray coverage does not match this source crop")
    order = np.lexsort((hit_y, hit_rays))
    hit_rays, hit_y = hit_rays[order], hit_y[order]
    start = np.searchsorted(hit_rays, np.arange(nz*nx), side="left")
    stop = np.searchsorted(hit_rays, np.arange(nz*nx), side="right")
    height = np.full(nz*nx, np.nan)
    present = stop > start
    height[present] = hit_y[stop[present]-1]
    height = height.reshape((nz, nx))
    numeric = budget["protected_interface_numerical_m"]
    recipe = budget["recipe"]
    semantics = np.asarray([classify(str(name)) for name in names.ravel()]).reshape(names.shape)
    soil = np.isin(semantics, ["soil", "sediment"])
    baseline = np.where(soil, recipe["soil_and_sediment_baseline_m"], np.where(semantics == "rock", recipe["rock_baseline_m"], 0.))
    detail = soil.astype(float) * recipe["soil_displacement_peak_to_peak_m"] / 2
    authored_up = np.where(protection, 0., baseline)
    authored_down = np.where(protection, 0., baseline+recipe["highmap_max_lowering_m"])
    detail[protection] = 0
    expected_exterior = source_height.copy()
    if structural.any():
        ys = np.arange(len(occupancy))[:, None, None] + minimum[1] + 1
        structural_top = np.max(np.where(structural, ys, -np.inf), axis=0)
        structural_exterior = structural_top > expected_exterior
        expected_exterior[structural_exterior] = structural_top[structural_exterior]
        authored_up[structural_exterior] = authored_down[structural_exterior] = detail[structural_exterior] = 0
    else:
        structural_exterior = np.zeros_like(surface_valid)
    up_limit, down_limit = authored_up+detail+numeric, authored_down+detail+numeric
    delta_errors = surface_valid & ((delta > authored_up+numeric) | (delta < -authored_down-numeric))
    protected_delta_errors = protection & surface_valid & (np.abs(delta) > numeric)
    difference = height-expected_exterior
    exterior_errors = surface_valid & (~np.isfinite(height) | (difference > up_limit) | (difference < -down_limit))
    # Source-cavity ceilings are fixed. A roof top is fixed unless it coincides
    # with a legally refined, unprotected exterior. Interior stacked floors
    # never inherit the exterior terrain's larger smoothing budget.
    roof_rows, roof_failures = [], []
    for row in roof_intervals:
        x, z, bottom, top, target_bottom, target_top, thickness, target_thickness = map(float, row)
        rx, rz = int(np.floor(x))-minimum[0], int(np.floor(z))-minimum[2]
        exterior_top = abs(top-expected_exterior[rz, rx]) <= 1e-6
        top_up = up_limit[rz, rx] if exterior_top else numeric
        top_down = down_limit[rz, rx] if exterior_top else numeric
        loss_limit = top_down+numeric
        failed = (target_thickness <= 0 or abs(target_bottom-bottom) > numeric
                  or target_top-top > top_up or top-target_top > top_down or thickness-target_thickness > loss_limit)
        roof_rows.append([x, z, bottom, top, target_bottom, target_top, numeric, top_up, top_down, loss_limit, float(failed)])
        if failed:
            roof_failures.append(len(roof_rows)-1)
    # Check every known-air-adjacent supporting floor below covered source air,
    # including exact masonry cells. Reuse the same independent triangle rays.
    floor_mask = np.zeros_like(supporting)
    floor_mask[:-1] = supporting[:-1] & covered[1:] & validity[1:] & (occupancy[1:] == 0)
    floor_rows, floor_failures = [], []
    for y, z, x in np.argwhere(floor_mask):
        ray = z*nx+x
        values = hit_y[start[ray]:stop[ray]]
        centre = minimum[1]+y+.5
        interval = int(np.searchsorted(values, centre, side="right"))
        observed = float(values[interval]) if interval % 2 and interval < len(values) else float("nan")
        expected = float(minimum[1]+y+1)
        failure = not np.isfinite(observed) or abs(observed-expected) > numeric
        floor_rows.append([minimum[0]+x+.5, minimum[2]+z+.5, expected, observed, numeric, float(structural[y, z, x]), float(failure)])
        if failure:
            floor_failures.append(len(floor_rows)-1)
    # Re-evaluate independent decoder rows, including uniqueness and hash-bound
    # source/retained values, rather than trusting a producer's pass string.
    decode_path = ir_dir / "independent_decode_comparison.json"
    decode = json.loads(decode_path.read_text())
    decode_samples_path = safe_path(ir_dir, decode["samples_file"], must_exist=True)
    if decode["source_ir_sha256"] != ir_hash or sha256_file(decode_samples_path) != decode["samples_sha256"]:
        raise ValueError("Independent source decoder evidence is stale")
    rows = json.loads(decode_samples_path.read_text())
    decoder_unique = len({tuple(row["xyz"]) for row in rows})
    decoder_errors = [index for index, row in enumerate(rows) if not row["reference"] == row["production"] == row["retained"]]
    strata = dict(Counter(row["stratum"] for row in rows))
    material_counts = dict(Counter(row["reference"]["Name"] for row in rows))
    # Directly check both observed sides of every source portal face with the
    # independent Java-save reader, complementing original decoder strata.
    from ..source.anvil import dimensions
    from ..source.independent import reference_chunk, reference_block
    source_graph = json.loads((ir_dir / "topology/source_topology_graph.json").read_text())
    portal_points = set()
    for portal in source_graph["portals"]:
        for face in portal["faces"]:
            centre = np.asarray(face["center_xyz"], dtype=float)
            normal = np.zeros(3)
            normal[{"x": 0, "y": 1, "z": 2}[face["normal_axis"]]] = face["normal_sign"]
            for sign in (-1, 1):
                portal_points.add(tuple(np.floor(centre+sign*.25*normal).astype(int).tolist()))
    portal_source_rows, portal_source_errors, reference_cache = [], [], {}
    if portal_points:
        source_region = dimensions(Path(ir["source_path"]))[ir["dimension"]] / "region"
        for xyz in sorted(portal_points):
            x, y, z = xyz
            key = x//16, z//16
            if key not in reference_cache:
                reference_cache[key] = reference_chunk(source_region, *key)
            state = reference_block(reference_cache[key], x, y, z)
            valid_air = state["Name"] in {"minecraft:air", "minecraft:cave_air", "minecraft:void_air"}
            portal_source_rows.append({"source_xyz": list(xyz), "reference_state": state, "is_source_air": valid_air})
            if not valid_air:
                portal_source_errors.append(len(portal_source_rows)-1)
    required = budget["minimum_source_comparisons"]
    missing = {"invalid_surface_columns": int((~surface_valid).sum()), "invalid_volume_cells": int((~validity).sum()),
               "unknown_volume_cells": int((occupancy == 3).sum()), "source_exclusions": ir["scope"]["exclusions"]}
    complete = not (missing["invalid_surface_columns"] or missing["invalid_volume_cells"] or missing["unknown_volume_cells"])
    geometry_fail = bool(exterior_errors.any() or delta_errors.any() or protected_delta_errors.any()
                         or decoder_errors or portal_source_errors or decoder_unique < required or not complete)
    topology_fail = bool(sample_report["status"] != "pass" or graph_report["status"] != "pass"
                         or parity_errors or protected_voxel_errors or unique_inside_samples < 10000
                         or roof_failures or floor_failures)
    output.mkdir(parents=True, exist_ok=True)
    portal_samples_path = output / "independent_source_portal_samples.json"
    atomic_json(portal_samples_path, portal_source_rows)
    raw_path = output / "source_fidelity_samples.npz"
    np.savez_compressed(raw_path, source_exterior_height=expected_exterior, target_exterior_height=height,
        declared_delta=delta, protection=protection, allowed_up=up_limit, allowed_down=down_limit,
        source_validity=surface_valid, exterior_error=exterior_errors, declaration_error=delta_errors,
        source_min_xyz=minimum, source_structural_exterior=structural_exterior,
        roof_intervals=np.asarray(roof_rows).reshape((-1, 11)), cave_floor_samples=np.asarray(floor_rows).reshape((-1, 7)),
        roof_columns=np.asarray(["source_x", "source_z", "source_bottom", "source_top", "target_bottom", "target_top", "bottom_allowance", "top_up_allowance", "top_down_allowance", "thickness_loss_allowance", "failure"]),
        floor_columns=np.asarray(["source_x", "source_z", "source_height", "target_height", "allowance", "structural", "failure"]))
    result = {"schema_version": 1, "kind": "IndependentSourceFidelityMeasurements", "status": "fail" if geometry_fail or topology_fail else "measurements_pass",
        "source_geometry_status": "fail" if geometry_fail else "pass",
        "protected_cave_geometry_status": "fail" if topology_fail else "pass" if roof_rows or floor_rows else "not_applicable",
        "full_Q01_status": "not_run", "full_Q03_status": "fail" if topology_fail else "not_run",
        "quality_profile_sha256": budget["quality_profile_sha256"], "source_fidelity_budget_sha256": budget_record["sha256"],
        "source_ir_sha256": ir_hash, "mesh_sha256": mesh_hash, "validator_sha256": sha256_file(Path(__file__)),
        "decoder": {"unique_positions": decoder_unique, "mismatch_indices": decoder_errors, "strata": strata, "materials": material_counts,
                    "independent_portal_source_positions": len(portal_source_rows), "portal_source_error_indices": portal_source_errors},
        "exterior": {"column_samples": int(surface_valid.sum()), "outside_budget": int(exterior_errors.sum()),
            "declared_refinement_outside_budget": int(delta_errors.sum()), "protected_declared_changes": int(protected_delta_errors.sum()),
            "maximum_measured_raising_m": float(np.maximum(difference[surface_valid], 0).max()) if np.isfinite(height[surface_valid]).all() else None,
            "maximum_measured_lowering_m": float(np.maximum(-difference[surface_valid], 0).max()) if np.isfinite(height[surface_valid]).all() else None,
            "explanation": "Measured differences must fit the frozen substrate baseline, one-sided erosion, eligible physical displacement and numerical budgets"},
        "caves": {"roof_interval_samples": len(roof_rows), "roof_budget_failure_indices": roof_failures,
            "floor_samples": len(floor_rows), "floor_budget_failure_indices": floor_failures,
            "protected_interface_allowance_m": numeric, "inside_outside_errors": parity_errors,
            "protected_voxel_errors": protected_voxel_errors, "unique_inside_outside_samples": unique_inside_samples,
            "portal_segment_count": sample_report["portal_segment_count"], "source_portal_patches": int(np.max(portals))},
        "structural_support": support_context, "missing_coverage": missing,
        "source_water_columns": int(water_mask.sum()),
        "evidence_sha256": {"topology_samples_report": sha256_file(Path(topology_samples_path)),
            "topology_graph_report": sha256_file(Path(topology_graph_path)), "column_ray_reconstruction": sha256_file(graph_files["column_ray_reconstruction.npz"]),
            "point_classifications": sha256_file(sample_files["source_mesh_samples.npz"]), "source_decode_report": sha256_file(decode_path),
            "source_decode_samples": sha256_file(decode_samples_path), "exterior_delta": sha256_file(Path(exterior_delta_path)),
            "frozen_source_budget": sha256_file(Path(frozen_budget_path))},
        "files": [{"path": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size} for path in (raw_path, portal_samples_path)],
        "limitations": ["Measurements do not establish target material/biome/water representation; those source discrepancies require separate bound scene evidence",
            "Actual Isaac portal/interior/overhang views and contact remain required before Q03 is complete",
            "Column-centre and portal-face sampling does not prove arbitrary sub-voxel lateral clearance or complete cave identity outside this crop",
            "No player structures become natural geology; retained masonry and excluded objects keep their recorded scope",
            "No user navigation stack or robot-clearance specification was supplied or tested"]}
    atomic_json(output / "source_fidelity.json", result)
    return result
