"""Derived solid ownership for natural terrain and retained structural stone.

Minecraft dungeon masonry is a supporting structure, not natural substrate.
Keep that identity while including its actual cells in one collision mesh.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from ..io import atomic_json, hash_object, sha256_file
from ..security import safe_path
from ..source.nbt import SourceError
from ..source.semantics import GROUND_CLASSES, classify


STRUCTURAL_SUPPORT_BLOCKS = frozenset({"minecraft:cobblestone", "minecraft:mossy_cobblestone"})


def derive_source_support(ir_dir: Path, output: Path | None = None) -> tuple[np.ndarray, dict]:
    """Return bool [y,z,x] solids plus explicit source-structural provenance.

    Exact sparse block palettes identify the added stone cells. Other nonterrain
    entities, vegetation and fluids remain explicit exclusions. Missing/unknown
    input is rejected, so False never silently manufactures air from uncertainty.
    """
    ir_dir = Path(ir_dir).resolve()
    manifest_path = ir_dir / "world_ir.json"
    source = json.loads(manifest_path.read_text())
    source_hash = sha256_file(manifest_path)
    producer_hash = sha256_file(Path(__file__))
    semantics_path = Path(__file__).parent.parent / "source/semantics.py"
    semantic_hash = sha256_file(semantics_path)
    records = {entry["path"]: entry for entry in source["files"]}
    input_hashes = {}

    def verified_file(relative):
        if relative not in records:
            raise SourceError("Source support input is absent from WorldIR file manifest")
        path = safe_path(ir_dir, relative, must_exist=True)
        digest = sha256_file(path)
        if digest != records[relative]["sha256"]:
            raise SourceError("Source support input differs from retained WorldIR bytes")
        input_hashes[relative] = digest
        return path

    with np.load(verified_file("natural_occupancy.npz"), allow_pickle=False) as data:
        occupancy = data["occupancy"]
        validity = data["validity"]
        minimum = data["min_xyz"].astype(np.int64)
        voxel_size = float(data["voxel_size_m"])
    if (occupancy.ndim != 3 or minimum.shape != (3,) or validity.shape != occupancy.shape or validity.dtype != bool
            or not validity.all() or np.any(occupancy == 3) or voxel_size != 1):
        raise SourceError("Authoritative source support requires complete known 1m source occupancy")
    natural = occupancy == 1
    structural = np.zeros(occupancy.shape, dtype=bool)
    structural_blocks = []
    excluded_structural_blocks = []
    counts = Counter()
    excluded_names = Counter()
    encountered_chunks = set()
    for chunk in source["terrain_volume"]["chunk_records"]:
        cx, cz = map(int, chunk["chunk_xz"])
        if (cx, cz) in encountered_chunks:
            raise SourceError("Duplicate retained source chunk in support ownership")
        encountered_chunks.add((cx, cz))
        with np.load(verified_file(chunk["volume_file"]), allow_pickle=False) as data:
            ids = data["block_id"]
            sections = data["section_y"].astype(int)
            palette = json.loads(str(data["palette_json"]))
            chunk_minimum = data["min_xyz"].astype(int)
        if (ids.shape != (len(sections), 16, 16, 16) or len(np.unique(sections)) != len(sections)
                or chunk_minimum[0] != cx*16 or chunk_minimum[2] != cz*16
                or sorted(sections.tolist()) != sorted(chunk["section_y"]) or ids.max() >= len(palette)):
            raise SourceError("Retained chunk axes or palette do not match source support contract")
        palette_counts = np.bincount(ids.ravel(), minlength=len(palette))
        for palette_id, state in enumerate(palette):
            name = state["Name"]
            semantic = classify(name)
            count = int(palette_counts[palette_id])
            if not count:
                continue
            if name not in STRUCTURAL_SUPPORT_BLOCKS:
                if semantic not in GROUND_CLASSES | {"air"}:
                    excluded_names[name] += count
                if semantic != "structure":
                    continue
            positions = np.argwhere(ids == palette_id)
            for section_index, local_y, local_z, local_x in positions:
                xyz = np.asarray([cx*16+local_x, sections[section_index]*16+local_y, cz*16+local_z], dtype=np.int64)
                index = (xyz-minimum)[[1, 2, 0]]
                if np.any(index < 0) or np.any(index >= np.asarray(occupancy.shape)):
                    raise SourceError("Retained structural position lies outside dense source volume")
                y, z, x = map(int, index)
                if occupancy[y, z, x] != 2:
                    raise SourceError("Structural state disagrees with unchanged nonterrain classification")
                entry = {"source_xyz_min_corner": xyz.tolist(), "dense_yzx": index.tolist(),
                         "source_block_state": state, "source_volume_file": chunk["volume_file"]}
                if name in STRUCTURAL_SUPPORT_BLOCKS:
                    if semantic != "structure" or structural[y, z, x]:
                        raise SourceError("Structural support classification changed or overlaps another retained cell")
                    structural[y, z, x] = True
                    structural_blocks.append(entry)
                    counts[name] += 1
                else:
                    entry["exclusion_reason"] = "Source non-ground object; requires separate scene-object reconstruction"
                    excluded_structural_blocks.append(entry)
    support = natural | structural
    # Ensure no source file or policy changed during the derivation.
    if (sha256_file(manifest_path) != source_hash or any(sha256_file(ir_dir/name) != digest for name, digest in input_hashes.items())
            or sha256_file(Path(__file__)) != producer_hash or sha256_file(semantics_path) != semantic_hash):
        raise SourceError("Source support inputs changed while deriving ownership")
    mask_hash = hashlib.sha256(np.ascontiguousarray(support).tobytes()).hexdigest()
    report = {"schema_version": 1, "kind": "SourceSupportingSolids", "status": "derived_source_support",
              "axis_order": "y,z,x", "shape": list(support.shape), "min_xyz": minimum.tolist(), "voxel_size_m": 1,
              "source_ir_sha256": source_hash, "source_snapshot_sha256": source.get("source_snapshot_sha256"),
              "source_file_sha256": input_hashes, "producer_sha256": producer_hash,
              "semantic_policy_sha256": semantic_hash,
              "allowed_structural_support_blocks": sorted(STRUCTURAL_SUPPORT_BLOCKS),
              "natural_solid_cells": int(natural.sum()), "structural_stone_cells": int(structural.sum()),
              "support_solid_cells": int(support.sum()), "occupancy_sha256": mask_hash,
              "structural_counts": dict(sorted(counts.items())), "structural_blocks": structural_blocks,
              "excluded_non_ground_counts": dict(sorted(excluded_names.items())),
              "excluded_structure_objects": excluded_structural_blocks,
              "material_ownership": "Added cells remain source structural masonry; never reclassified as natural substrate",
              "geometry_ownership": "One union supporting volume; no duplicate floor or independent overlapping cap",
              "source_world_ir_modified": False,
              "qualification": "Source ownership derived only; reconstructed geometry, structural material fidelity and Isaac collision require validation",
              "files": []}
    if output is not None:
        output = Path(output).resolve()
        if output == ir_dir or ir_dir in output.parents:
            raise SourceError("Derived support output must be outside preserved WorldIR")
        output.mkdir(parents=True, exist_ok=True)
        path = output / "support_mask.npz"
        if path.exists() or (output / "support_manifest.json").exists():
            raise SourceError("Preserve prior source support candidate; use a new output directory")
        np.savez_compressed(path, occupancy=support, structural_occupancy=structural,
                            validity=validity, min_xyz=minimum, voxel_size_m=np.asarray(1.))
        report["files"] = [{"path": path.name, "sha256": sha256_file(path), "bytes": path.stat().st_size}]
        report["content_sha256"] = hash_object({"occupancy_sha256": mask_hash,
            "structural_blocks": structural_blocks, "source_ir_sha256": source_hash,
            "producer_sha256": report["producer_sha256"], "files": report["files"]})
        atomic_json(output / "support_manifest.json", report)
    return support, report
