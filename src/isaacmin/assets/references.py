"""Perspective crops of licensed actual panoramas; never generated scene evidence."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .network import ServiceError, atomic_json, digest, utcnow


def panorama_view(path: Path, yaw_degrees: float, pitch_degrees: float, width=640, height=360, hfov_degrees=90):
    if not 1 <= hfov_degrees < 170 or width * height > 8_000_000:
        raise ServiceError("projection_bounds", "Reference projection parameters exceed supported bounds")
    with Image.open(path) as im:
        source = np.asarray(im.convert("RGB"))
    if not 1.9 < source.shape[1] / source.shape[0] < 2.1:
        raise ServiceError("projection_shape", "Reference panorama is not approximately 2:1 equirectangular")
    x = (2 * (np.arange(width) + 0.5) / width - 1) * math.tan(math.radians(hfov_degrees) / 2)
    y = (1 - 2 * (np.arange(height) + 0.5) / height) * math.tan(math.radians(hfov_degrees) / 2) * height / width
    xx, yy = np.meshgrid(x, y)
    ray = np.stack((xx, yy, np.ones_like(xx)), axis=-1)
    pitch, yaw = math.radians(pitch_degrees), math.radians(yaw_degrees)
    # Camera axes: x right, y up, z forward; panorama u wraps, v increases downward.
    rx = np.array([[1, 0, 0], [0, math.cos(pitch), math.sin(pitch)], [0, -math.sin(pitch), math.cos(pitch)]])
    ry = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
    ray = ray @ (ry @ rx).T
    ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
    u = (np.arctan2(ray[..., 0], ray[..., 2]) / (2 * math.pi) + 0.5) * source.shape[1] - 0.5
    v = (0.5 - np.arcsin(ray[..., 1]) / math.pi) * source.shape[0] - 0.5
    u0 = np.floor(u).astype(int)
    v0 = np.floor(v).astype(int)
    a, b = (u - u0)[..., None], (v - v0)[..., None]
    u1, v1 = (u0 + 1) % source.shape[1], np.clip(v0 + 1, 0, source.shape[0] - 1)
    u0 %= source.shape[1]
    v0 = np.clip(v0, 0, source.shape[0] - 1)
    values = source[v0, u0] * (1 - a) * (1 - b) + source[v0, u1] * a * (1 - b) + source[v1, u0] * (1 - a) * b + source[v1, u1] * a * b
    return Image.fromarray(np.clip(values, 0, 255).astype(np.uint8))


def build_reference_board(workspace: Path, asset_record: dict):
    files = [f for f in asset_record["files"] if f.get("role") == "tonemapped"]
    if asset_record["kind"] != "hdris" or len(files) != 1:
        raise ServiceError("reference_type", "A licensed photographic panorama is required")
    metadata = asset_record["metadata"]
    ecology = {
        "forest_slope": "Coniferous woodland understory; not evidence for deciduous canopy composition",
        "birchwood": "Birch/deciduous woodland appearance; different location and season cannot establish source species density",
        "alps_field": "Alpine meadow and mountain appearance; not a species or geology prescription for this temperate plains source",
    }.get(asset_record["asset_id"], "Ecological scope requires provider metadata and image inspection; no biome equivalence assumed")
    if not any("photograph" in str(role).lower() for role in metadata.get("authors", {}).values()) and not metadata.get("date_taken"):
        raise ServiceError("reference_provenance", "HDRI capture metadata does not establish photographic origin")
    original = Path(files[0]["path"])
    if digest(original) != files[0]["sha256"]:
        raise ServiceError("corrupt_reference", "Original reference hash changed")
    folder = Path(workspace) / "references" / asset_record["asset_id"]
    folder.mkdir(parents=True, exist_ok=True)
    board = Image.new("RGB", (1280, 800), (235, 235, 235))
    draw = ImageDraw.Draw(board)
    record = {"schema_version": 1, "provider": asset_record["provider"], "asset_id": asset_record["asset_id"],
              "authors": asset_record["authors"], "source_page": asset_record["source_page"],
              "licence": asset_record["licence"], "licence_url": asset_record["licence_url"],
              "original_sha256": files[0]["sha256"], "original_path": str(original), "created_at_utc": utcnow(),
              "classification": "photographic_hdr_panorama_provider_tonemapped", "not_scene_evidence": True,
              "capture_metadata": {k: metadata.get(k) for k in ("date_taken", "coords", "whitebalance", "evs_cap", "category", "tags", "attributes")},
              "limitations": ["Capture height and lens calibration not supplied", "Panorama is a different real location from Minecraft",
                              "Provider tonemapping is inherited; no absolute radiometric/color calibration", ecology],
              "views": []}
    for i, yaw in enumerate((0, 90, 180, 270)):
        view = panorama_view(original, yaw, -15)
        target = folder / f"view_{yaw:03d}.png"
        view.save(target)
        x, y = (i % 2) * 640, (i // 2) * 400
        board.paste(view, (x, y))
        draw.text((x + 10, y + 372), f"REAL PHOTO: yaw {yaw}, pitch -15, HFOV 90 deg | Poly Haven", fill=(0, 0, 0))
        record["views"].append({"path": str(target), "sha256": digest(target), "projection": "equirectangular_to_rectilinear_bilinear",
                                "yaw_degrees": yaw, "pitch_degrees": -15, "horizontal_fov_degrees": 90, "resolution_px": [640, 360]})
    target = folder / "reference_board.jpg"
    board.save(target, quality=95)
    record["board"] = {"path": str(target), "sha256": digest(target)}
    atomic_json(folder / "reference_manifest.json", record)
    return record
