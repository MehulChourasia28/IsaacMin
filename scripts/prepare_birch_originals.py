"""Freeze original birch candidate maps and opacity-derived geometry profiles.

This CPU-only preparation copies original bytes. It creates no synthetic bitmap
channels, performs no native rendering, and does not qualify botanical fidelity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def photograph_profile(mask: Path, index: int, material_index: int) -> dict:
    """Sample original horizontal leaf silhouettes in the generator row schema."""
    with Image.open(mask) as image:
        image.load()
        opacity = np.asarray(image.convert("L")) >= 128
    height, width = opacity.shape
    yy, xx = np.where(opacity)
    if not len(xx) or opacity[:40].any():
        raise ValueError("Missing silhouette or opaque original watermark area")
    # These orientations are recorded from inspection of the original photos:
    # 00001 petiole at right, 00002 petiole at left. No image bytes are changed.
    base_x, tip_x = (int(xx.max()), int(xx.min())) if index == 1 else (int(xx.min()), int(xx.max()))
    length = abs(tip_x - base_x)
    base_y = float(np.flatnonzero(opacity[:, base_x]).mean())
    if length < 48:
        raise ValueError("Original silhouette cannot support 49 distinct samples")
    rows = []
    column_records = []
    for j in range(49):
        t = j / 48
        x = round(base_x + (tip_x - base_x) * t)
        occupied = np.flatnonzero(opacity[:, x])
        if not len(occupied):
            raise ValueError("Disconnected original silhouette needs explicit review")
        bottom, top = float(occupied.max()), float(occupied.min())
        # Lateral coordinates increase across each row. Image Y grows downward;
        # Blender V grows upward. UVs point to the original, unchanged texels.
        rows.append({"t": t, "vertices": [
            [(base_y - y) / length, (x + .5) / width, 1 - (y + .5) / height]
            for y in (bottom, (top + bottom) / 2, top)
        ]})
        column_records.append({"x": x, "y_min": int(top), "y_max": int(bottom),
                               "opaque_pixels": int(len(occupied)),
                               "interior_clear_pixels": int(bottom - top + 1 - len(occupied))})
    return {"source_leaf_id": f"3dmd_leaf_{index:05d}", "material_index": material_index,
            "material_prefix": f"birch_leaf_{index:05d}", "rows": rows,
            "leaf_aspect_ratio": float((yy.max() - yy.min() + 1) / length),
            "source_mask_sha256": sha(mask), "source_image_size_px": [width, height],
            "longitudinal_axis": "original_image_x", "base_x_px": base_x, "tip_x_px": tip_x,
            "base_y_px": base_y, "longitudinal_length_px": length,
            "uv_convention": "original image texel centers; u=(x+.5)/width; v=1-(y+.5)/height",
            "mask_threshold": "original JPEG luminance >=128; native alpha candidate threshold0.5",
            "sampled_columns": column_records,
            "watermark_exclusion": {"region_xyxy_exclusive": [0, 0, width, 40],
                                    "opaque_pixels_at_threshold": int(opacity[:40].sum()),
                                    "method": "original provider mask; no edited image or removed watermark"},
            "limitations": ["49 longitudinal samples approximate the photographed silhouette; fine edge serrations still depend on original opacity.",
                            "Physical leaf length is inferred by the generator, not measured from this photograph.",
                            "Photograph00001 petiole reaches image boundary." if index == 1 else "Photograph00002 has both visible tip and petiole endpoint."]}


def prepare(workspace: Path, output: Path) -> dict:
    workspace, output = workspace.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError("Immutable candidate directory exists; choose a separate candidate")
    script = Path(__file__).resolve()
    script_hash = sha(script)
    leaf_manifest = workspace / "assets/masters/3dmd/birch_tree_leaves/original_20060914/asset_manifest.json"
    bark_manifest = workspace / "assets/masters/texturecan/wood_0027/4k/asset_manifest.json"
    twig_manifest = workspace / "assets/masters/ambientcg/Bark012/744c3424f7f42a53/asset_manifest.json"
    leaf, bark, twig = [json.loads(p.read_text()) for p in (leaf_manifest, bark_manifest, twig_manifest)]
    if bark["asset_id"] != "wood_0027" or twig["asset_id"] != "Bark012":
        raise ValueError("Unexpected original source identity")
    maps = []
    materials = {}
    profiles = []

    def select(source, expected, target, role, provider, asset_id, material):
        source = Path(source)
        if not source.is_absolute():
            source = workspace / source
        if sha(source) != expected:
            raise ValueError("Acquired original channel changed: " + str(source))
        maps.append({"source_path": str(source.relative_to(workspace)), "path": target,
                     "sha256": expected, "role": role, "provider": provider,
                     "asset_id": asset_id, "material_prefix": material,
                     "operation": "unchanged original byte copy"})

    for material_index, index in enumerate((1, 2)):
        prefix = f"birch_leaf_{index:05d}"
        spec = {"roughness_value": .55, "ior_value": 1.42}
        for role, original_name in (("base_color", f"leaf_texture_{index:05d}.jpg"),
                                    ("opacity", f"leaf_mask_texture_{index:05d}.jpg")):
            matches = [r["original"] for r in leaf["files"] if Path(r["original"]["path"]).name == original_name]
            if len(matches) != 1:
                raise ValueError("Ambiguous acquired original leaf channel")
            item = matches[0]
            name = f"{prefix}_{role}.jpg"
            select(item["path"], item["sha256"], name, role, "3dmd.net", "birch_tree_leaves", prefix)
            spec[role] = name
            if role == "opacity":
                profiles.append(photograph_profile(workspace / maps[-1]["source_path"], index, material_index))
        materials[prefix] = spec

    for prefix, manifest, folder, roles, provider, asset_id in (
        ("birch_bark", bark, Path(bark["extracted_directory"]), bark["map_roles"], "TextureCan", "wood_0027"),
        ("birch_twig", twig, Path(twig["extracted_directory"]),
         {"base_color": "Bark012_2K-PNG_Color.png", "roughness": "Bark012_2K-PNG_Roughness.png",
          "normal": "Bark012_2K-PNG_NormalGL.png"}, "ambientCG", "Bark012"),
    ):
        spec = {}
        files = manifest.get("extracted_files", manifest.get("files"))
        for role in ("base_color", "roughness", "normal"):
            name = roles[role]
            matches = [f for f in files if f["path"] == name]
            if len(matches) != 1:
                raise ValueError("Ambiguous acquired bark channel")
            target = prefix + "_" + role + Path(name).suffix
            select(folder / name, matches[0]["sha256"], target, role, provider, asset_id, prefix)
            spec[role] = target
        materials[prefix] = spec

    licences = [
        (workspace / leaf["licence_evidence"]["path"], leaf["licence_evidence"]["sha256"], "3dmd_licence.html"),
        (bark_manifest.parent / "licence.html", bark["licence_page_sha256"], "texturecan_licence.html"),
        (workspace / "evidence/services/licences/ambientcg.txt", None, "ambientcg_licence.txt"),
    ]
    proof_paths = [script, leaf_manifest, bark_manifest, twig_manifest]
    for source, expected, _ in licences:
        if expected is not None and sha(source) != expected:
            raise ValueError("Stored original licence changed")
        proof_paths.append(source)
    proof_paths.extend(workspace / m["source_path"] for m in maps)
    frozen_inputs = [{"path": str(p.relative_to(workspace)), "sha256": sha(p)} for p in proof_paths]
    output.mkdir(parents=True)
    write_json(output / "frozen_inputs.json", {"schema_version": 1, "producer_script_sha256": script_hash,
               "frozen_before_copy": True, "files": frozen_inputs})
    for item in maps:
        source, target = workspace / item["source_path"], output / item["path"]
        shutil.copyfile(source, target)
        if sha(target) != item["sha256"]:
            raise ValueError("Candidate channel copy changed original bytes")
        with Image.open(target) as image:
            image.load()
            item["image_validation"] = {"status": "full_decode_pass", "size": list(image.size),
                                        "mode": image.mode, "format": image.format}
        item["size_bytes"] = target.stat().st_size
    licence_records = []
    for source, _, name in licences:
        target = output / name
        shutil.copyfile(source, target)
        licence_records.append({"path": name, "sha256": sha(target), "original_path": str(source.relative_to(workspace))})
    (output / "CREDITS.txt").write_text(
        "Birch leaf photographs and original opacity masks: Andyba / 3dmd.net\n"
        "https://www.3dmd.net/gallery/thumbnails-68.html\n"
        "Licence: https://www.3dmd.net/license.htm (preserved in 3dmd_licence.html).\n"
        "Redistribution within this simulation requires 3dmd.net credit and a link where technically possible.\n"
        "Do not distribute these originals or derivatives as a competing standalone texture pack.\n\n"
        "White birch bark: TextureCan wood_0027, CC0-1.0.\n"
        "https://www.texturecan.com/details/221/ ; https://www.texturecan.com/terms/\n"
        "Artist-created Substance material, not a photographic scan.\n\n"
        "Generic brown twig proxy: ambientCG / Lennart Demes, Bark012, CC0-1.0.\n"
        "https://ambientcg.com/a/Bark012 ; https://docs.ambientcg.com/license/\n"
        "Provider tags this bark as oak; its use on young birch twigs is an unqualified proxy.\n"
    )
    write_json(output / "materials.json", {"schema_version": 1, "materials": materials})
    write_json(output / "leaf_profiles.json", profiles)
    report = {"schema_version": 1, "candidate_id": "birch_originals_v1", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "classification": "original_photographic_leaf_color_and_masks_plus_artist_bark_and_generic_scanned_twig_proxy",
              "producer_script": str(script.relative_to(workspace)), "producer_script_sha256": script_hash,
              "source_manifests": [{"path": str(p.relative_to(workspace)), "sha256": sha(p)} for p in (leaf_manifest, bark_manifest, twig_manifest)],
              "files": maps, "licences": licence_records, "credits": {"path": "CREDITS.txt", "sha256": sha(output / "CREDITS.txt")},
              "profile_sha256": sha(output / "leaf_profiles.json"), "materials_sha256": sha(output / "materials.json"),
              "frozen_inputs_sha256": sha(output / "frozen_inputs.json"), "leaf_profile_count": len(profiles),
              "profile_rows": 49, "normal_convention": "OpenGL for acquired bark maps; leaf normal bitmap absent",
              "estimated_leaf_values": {"roughness": .55, "ior": 1.42, "classification": "inferred unmeasured shader candidates"},
              "watermark_exclusion": [p["watermark_exclusion"] for p in profiles],
              "qualification": "not_run", "isaac_qualification": "not_run", "approved_uses": [],
              "limitations": [
                  "Provider identifies leaf collection as birch; exact Betula species and spring season are unverified.",
                  "Leaf photographs contain original lighting/specular shading and are not calibrated albedo scans.",
                  "No leaf normal, height, roughness, transmission or translucency maps are supplied or invented.",
                  "Leaf roughness0.55 and IOR1.42 are inferred constants requiring target-renderer qualification.",
                  "Original JPEG opacity edges require native cutout, grazing-angle and motion tests.",
                  "TextureCan white bark is an artist-created Substance material, not a scan.",
                  "ambientCG Bark012 is provider-tagged oak; using it for young birch twigs is an explicit generic brown-bark proxy.",
                  "Physical bark and leaf scale are not measured; generator choices must be recorded and tested.",
                  "49 sampled profile rows approximate photographed leaf boundaries; no anatomy or appearance qualification is claimed.",
              ]}
    for item in frozen_inputs:
        if sha(workspace / item["path"]) != item["sha256"]:
            raise ValueError("Frozen producer/input changed during preparation")
    write_json(output / "maps_manifest.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("assets/procedural_candidates/birch_originals_v1/maps"))
    args = parser.parse_args()
    target = args.output if args.output.is_absolute() else args.workspace / args.output
    result = prepare(args.workspace, target)
    print(json.dumps({"candidate": str(target), "profiles": result["leaf_profile_count"],
                      "map_files": len(result["files"]), "qualification": result["qualification"]}))
