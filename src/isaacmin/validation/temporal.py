"""Continuous native-capture measurements, without inferring navigation success."""
from pathlib import Path

import numpy as np
from PIL import Image

from ..io import atomic_json, read_json, sha256_file
from .metrics import unique_route_distance
from .sensors import camera_rays


def _intrinsics(frame):
    return frame.get("intrinsics") or {key: frame[key+"_pixels"] for key in ("fx", "fy", "cx", "cy")}


def _read(capture, frame):
    for role in ("rgb", "depth"):
        if sha256_file(capture / frame[role]) != frame[role+"_sha256"]:
            raise ValueError("Temporal input differs from recorded native frame bytes")
    depth = np.load(capture / frame["depth"], allow_pickle=False)
    if sha256_file(capture / frame["instance_segmentation"]) != frame.get("instance_segmentation_sha256"):
        raise ValueError("Temporal segmentation differs from the native frame manifest")
    labels = np.load(capture / frame["instance_segmentation"], allow_pickle=False).squeeze()
    ids = [int(k) for k, v in frame["instance_id_to_prim_path"].items() if "Terrain_FinalGround" in str(v)]
    rgb = np.asarray(Image.open(capture / frame["rgb"]).convert("RGB"), dtype=np.float32)/255.
    return rgb, depth, np.isin(labels, ids)


def measure_motion(capture_dir: Path, output: Path, *, samples_per_pair=4000) -> dict:
    """Known camera/depth geometric flow aligns static ground between raw frames.

    Occlusion and native instance labels define the correspondence mask. RGB
    differences never choose which pixels are evaluated. Residuals are triggers
    to investigate, not an automatic claim of temporal appearance qualification.
    """
    record = read_json(capture_dir / "capture_result.json")
    frames = record["frames"]
    if len(frames) < 2:
        raise ValueError("Temporal evaluation requires a continuous native sequence")
    if any(frame.get("pose", {}).get("kind") != "motion" for frame in frames):
        raise ValueError("Temporal evaluation requires a dedicated motion capture; static/scout jumps cannot count as route distance")
    poses = np.asarray([frame["camera_world_matrix_row_vectors"][3][:3] for frame in frames])
    simulation_time = np.asarray([frame["simulation_time_s"] for frame in frames])
    wall_time = np.asarray([frame["wall_elapsed_s"] for frame in frames])
    continuous = [frame["frame"] for frame in frames] == list(range(len(frames)))
    cadence = np.diff(simulation_time)
    if not np.isfinite(cadence).all() or (cadence <= 0).any():
        raise ValueError("Captured simulation timestamps are invalid or not increasing")
    rng = np.random.default_rng(91283)
    previous = _read(capture_dir, frames[0])
    missing_ground = [] if previous[2].any() else [0]
    pairs, flags = [], []
    for index in range(1, len(frames)):
        a, b = frames[index-1], frames[index]
        current = _read(capture_dir, b)
        ar, ad, am = previous
        br, bd, bm = current
        if not bm.any():
            missing_ground.append(index)
        candidates = np.argwhere(am & np.isfinite(ad) & (ad > 0) & (ad < 50))
        candidates = candidates[rng.choice(len(candidates), min(samples_per_pair, len(candidates)), replace=False)]
        if len(candidates):
            uv = candidates[:, ::-1]+.5
            origin, direction, cosine = camera_rays(uv, _intrinsics(a), a["camera_world_matrix_row_vectors"])
            world = origin+direction*(ad[tuple(candidates.T)]/cosine)[:, None]
            inverse = np.linalg.inv(np.asarray(b["camera_world_matrix_row_vectors"], float))
            local = np.column_stack((world, np.ones(len(world)))) @ inverse
            z = -local[:, 2]
            intr = _intrinsics(b)
            with np.errstate(divide="ignore", invalid="ignore"):
                projected = np.column_stack((local[:, 0]/z*intr["fx"]+intr["cx"],
                                              -local[:, 1]/z*intr["fy"]+intr["cy"]))
            inside = ((z > .02) & np.isfinite(projected).all(axis=1)
                      & (projected[:, 0] >= 0) & (projected[:, 0] < bd.shape[1])
                      & (projected[:, 1] >= 0) & (projected[:, 1] < bd.shape[0]))
            source_ids = np.flatnonzero(inside)
            target_uv = np.floor(projected[inside]).astype(int)
            target_vu = target_uv[:, ::-1]
            target_depth = bd[tuple(target_vu.T)]
            visible = (bm[tuple(target_vu.T)] & np.isfinite(target_depth)
                       & (np.abs(target_depth-z[inside]) <= np.maximum(.02, .01*z[inside])))
            source_ids, target_vu = source_ids[visible], target_vu[visible]
            source_vu = candidates[source_ids]
            residual = np.mean(np.abs(ar[tuple(source_vu.T)]-br[tuple(target_vu.T)]), axis=1)
        else:
            residual = np.empty(0)
        pair = {"from_frame": index-1, "to_frame": index, "visible_correspondences": len(residual),
                "mean_absolute_rgb_residual": float(residual.mean()) if len(residual) else None,
                "p95_absolute_rgb_residual": float(np.quantile(residual, .95)) if len(residual) else None,
                "fraction_residual_above_0_15": float((residual > .15).mean()) if len(residual) else None}
        if len(residual) < 100 or pair["fraction_residual_above_0_15"] > .05:
            flags.append({"frame": index, "reason": "insufficient_correspondences" if len(residual) < 100 else "temporal_rgb_residual",
                          "rgb": b["rgb"], "pose": b["pose"]})
        pairs.append(pair)
        previous = current
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "trajectory.npy", poses)
    result = {"status": "measurements_complete", "appearance_qualification": "not_run",
              "validator_sha256": sha256_file(Path(__file__)),
              "inputs": [{"path": frame[role], "sha256": frame[role+"_sha256"]}
                         for frame in frames for role in ("rgb", "depth", "instance_segmentation")],
              "capture_sha256": sha256_file(capture_dir / "capture_result.json"),
              "frames": len(frames), "continuous_frame_numbering": continuous,
              "simulation_duration_s": float(simulation_time[-1]-simulation_time[0]),
              "simulation_cadence_min_max_s": [float(cadence.min()), float(cadence.max())],
              "measured_wall_fps": (len(frames)-1)/float(wall_time[-1]-wall_time[0]),
              "missing_ground_frames": missing_ground, "route": unique_route_distance(poses),
              "trajectory": "trajectory.npy", "trajectory_sha256": sha256_file(output / "trajectory.npy"),
              "pairs": pairs, "flagged_frames": flags,
              "flow_method": "calibrated camera and axial-depth geometric optical flow for static opaque ground",
              "limits": ["Ground-only correspondences; foliage/shadow/LOD need separate matched detail inspection",
                         "No control stack or vehicle dynamics was run", "Residual threshold triggers investigation, not a perceptual pass"]}
    atomic_json(output / "temporal_measurements.json", result)
    return result
