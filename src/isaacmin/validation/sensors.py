"""Reproject actual Isaac axial depth onto independently ray-tested final ground."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..io import atomic_json, sha256_file
from .capture_plan import load_ground
from .metrics import compare_depth


def camera_rays(pixels_uv, intrinsics, camera_to_world, *, convention="usd_row_vector"):
    pixels = np.asarray(pixels_uv, float)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("Expected Nx2 pixel positions")
    fx, fy, cx, cy = [float(intrinsics[k]) for k in ("fx", "fy", "cx", "cy")]
    if fx <= 0 or fy <= 0:
        raise ValueError("Focal lengths must be positive")
    matrix = np.asarray(camera_to_world, float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Invalid camera transform")
    # USD camera looks down local-Z; localY is up, image row grows down.
    local = np.column_stack(((pixels[:, 0]-cx)/fx, -(pixels[:, 1]-cy)/fy, -np.ones(len(pixels))))
    cosines = 1./np.linalg.norm(local, axis=1)
    unit = local*cosines[:, None]
    if convention == "usd_row_vector":
        direction = unit @ matrix[:3, :3]
        origin = np.tile(matrix[3, :3], (len(unit), 1))
    elif convention == "column_vector":
        direction = unit @ matrix[:3, :3].T
        origin = np.tile(matrix[:3, 3], (len(unit), 1))
    else:
        raise ValueError("Unsupported camera matrix convention")
    if not np.allclose(np.linalg.norm(direction, axis=1), 1, atol=1e-6):
        raise ValueError("Camera transform contains non-rigid scale")
    return origin, direction, cosines


def validate_capture_depth(mesh_path: Path, capture_dir: Path, output_dir: Path, *, samples_per_frame=1500,
                           minimum_total=10000) -> dict:
    """Requires native segmentation, so foliage/water exclusions are not chosen by error."""
    output_dir.mkdir(parents=True, exist_ok=True)
    record = json.loads((capture_dir / "capture_result.json").read_text())
    near, far = float(record["near_m"]), float(record["far_m"])
    if not 0 < near < far:
        raise ValueError("Invalid recorded native camera clipping planes")
    mesh = load_ground(mesh_path, output_dir/"native_ground_queries")
    rng = np.random.default_rng(37013)
    all_measured, all_geometry, frame_indices, pixel_coordinates = [], [], [], []
    failed_frames = []
    jobs = []
    for frame in record["frames"]:
        if (sha256_file(capture_dir / frame["depth"]) != frame["depth_sha256"]
                or sha256_file(capture_dir / frame["rgb"]) != frame["rgb_sha256"]):
            raise ValueError("Capture bytes differ from the native frame record")
        # Field names are normalized by the capture adapter, never inferred from image values.
        depth = np.load(capture_dir / frame["depth"], allow_pickle=False)
        segmentation_file = frame.get("instance_segmentation")
        if not segmentation_file:
            failed_frames.append({"frame": frame["frame"], "reason": "No native ground segmentation"})
            continue
        if sha256_file(capture_dir / segmentation_file) != frame.get("instance_segmentation_sha256"):
            raise ValueError("Native instance mask is missing its bound content hash or has changed")
        segmentation = np.load(capture_dir / segmentation_file, allow_pickle=False)
        if segmentation.ndim == 3 and segmentation.shape[-1] == 1:
            segmentation = segmentation[:, :, 0]
        labels = frame.get("instance_id_to_labels", frame.get("instance_id_to_prim_path", {}))
        ground_ids = [int(key) for key, value in labels.items() if "Terrain_FinalGround" in str(value)]
        ground_mask = np.isin(segmentation, ground_ids)
        if np.any(ground_mask & (~np.isfinite(depth) | (depth <= 0))):
            failed_frames.append({"frame": frame["frame"], "reason": "Opaque rendered terrain has invalid sensor depth"})
        mask = ground_mask & np.isfinite(depth) & (depth > 0)
        pixels_vu = np.argwhere(mask)
        if not len(pixels_vu):
            failed_frames.append({"frame": frame["frame"], "reason": "No opaque terrain pixels identified"})
            continue
        chosen = pixels_vu[rng.choice(len(pixels_vu), min(samples_per_frame, len(pixels_vu)), replace=False)]
        pixels = chosen[:, ::-1]
        intrinsics = frame.get("intrinsics") or {key: frame[key+"_pixels"] for key in ("fx", "fy", "cx", "cy")}
        matrix = frame.get("camera_to_world", frame.get("camera_world_matrix_row_vectors"))
        # Annotator pixels address cells; actual USD aperture principal point is W/2,H/2.
        origins, dirs, cosines = camera_rays(pixels+.5, intrinsics, matrix)
        # Near clipping is an axial camera plane, not a Euclidean distance. A
        # surface before this plane cannot be the native depth hit. Keep every
        # sampled pixel and query the geometry after the same plane instead of
        # discarding pixels according to their depth error.
        query_origins = origins + dirs*(near/cosines)[:, None]
        jobs.append((frame,chosen,pixels,depth[chosen[:,0],chosen[:,1]],origins,dirs,cosines,query_origins))
    native_queries=getattr(mesh,'queries',None)
    offsets=np.r_[0,np.cumsum([len(job[1]) for job in jobs])]
    if native_queries is not None and jobs:
        all_origins=np.concatenate([job[7] for job in jobs]);all_directions=np.concatenate([job[5] for job in jobs])
        all_locations,_,all_triangles=native_queries.first_hits(all_origins,all_directions)
    for index,(frame,chosen,pixels,measured,origins,dirs,cosines,query_origins) in enumerate(jobs):
        if native_queries is None:
            locations,rays,_=mesh.ray.intersects_location(query_origins,dirs,multiple_hits=False)
        else:
            selection=slice(offsets[index],offsets[index+1]);selected=all_locations[selection];rays=np.flatnonzero(all_triangles[selection]>=0);locations=selected[rays]
        if len(rays) != len(chosen):
            failed_frames.append({"frame": frame["frame"], "reason": "Rendered opaque terrain has missing independent ground rays",
                                  "missing": len(chosen)-len(rays)})
        geometry = np.full(len(chosen), np.nan)
        geometry[rays] = np.linalg.norm(locations-origins[rays], axis=1)*cosines[rays]
        if np.any(geometry > far+1e-5):
            failed_frames.append({"frame": frame["frame"], "reason": "Native opaque depth has no independent hit within far clip"})
        all_measured.extend(measured.tolist())
        all_geometry.extend(geometry.tolist())
        frame_indices.extend([frame["frame"]]*len(chosen))
        pixel_coordinates.extend(pixels.tolist())
    measured, geometry = np.asarray(all_measured), np.asarray(all_geometry)
    samples = output_dir / "depth_samples.npz"
    np.savez_compressed(samples, render_depth=measured, geometry_axial_z=geometry,
                        opaque_valid=np.ones(len(measured), bool), frame=np.asarray(frame_indices),
                        pixels_uv=np.asarray(pixel_coordinates))
    try:
        metrics = compare_depth(measured, geometry, np.ones(len(measured), bool), semantics="axial_z", min_samples=minimum_total)
        status = "pass" if metrics["pass"] and not failed_frames else "fail"
    except ValueError as exc:
        metrics, status = {"reason": str(exc), "samples": len(measured)}, "fail"
    result = {"status": status, "metrics": metrics, "failed_frames": failed_frames,
              "validator_sha256": sha256_file(Path(__file__)),
              "samples": samples.name, "sample_sha256": sha256_file(samples),
              "final_ground_sha256": sha256_file(mesh_path), "capture_sha256": sha256_file(capture_dir / "capture_result.json"),
              "depth_semantics": "axial_z", "independent_query": "trimesh_triangle_rays",
              "camera_clipping_planes_m": [near, far],
              "clipping_policy": "All sampled pixels retained; independent rays start at the declared axial near plane",
              "exclusions": "Native segmentation excludes non-terrain foliage, water, sky and props; no error-based masking"}
    if native_queries is not None:
        result["native_query_evidence"]=mesh.receipt();result["independent_query"]="bounded_CGAL_original_triangle_segments"
    atomic_json(output_dir / "depth_validation.json", result)
    return result
