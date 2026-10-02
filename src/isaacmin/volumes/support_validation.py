"""Verify derived structural ownership before consuming it in independent checks."""
from pathlib import Path
import json

import numpy as np

from ..io import sha256_file
from ..security import safe_path
from ..source.nbt import SourceError
from .source_support import derive_source_support


def structural_support_for_validation(ir_dir: Path, occupancy: np.ndarray,
                                      support_manifest_path: Path | None):
    if support_manifest_path is None:
        return np.zeros_like(occupancy, dtype=bool), {"status": "not_requested", "structural_cells": 0}
    path = Path(support_manifest_path)
    report = json.loads(path.read_text())
    support, expected = derive_source_support(ir_dir)
    fields = ("source_ir_sha256", "source_snapshot_sha256", "source_file_sha256", "producer_sha256",
              "semantic_policy_sha256", "occupancy_sha256", "structural_blocks", "structural_counts",
              "shape", "min_xyz", "excluded_structure_objects", "excluded_non_ground_counts")
    if any(report.get(field) != expected[field] for field in fields):
        raise SourceError("Structural support manifest differs from exact retained source derivation")
    structural = support & (occupancy != 1)
    files = {entry["path"]: entry for entry in report.get("files", [])}
    if "support_mask.npz" not in files:
        raise SourceError("Structural support manifest lacks its hashed mask")
    mask_path = safe_path(path.parent, "support_mask.npz", must_exist=True)
    if sha256_file(mask_path) != files["support_mask.npz"]["sha256"]:
        raise SourceError("Structural support mask hash mismatch")
    with np.load(mask_path, allow_pickle=False) as data:
        if (not np.array_equal(data["occupancy"], support)
                or not np.array_equal(data["structural_occupancy"], structural)
                or not np.array_equal(data["min_xyz"], expected["min_xyz"])):
            raise SourceError("Stored support mask differs from exact source cells")
    return structural, {"status": "verified_source_derivation", "manifest_sha256": sha256_file(path),
                        "structural_cells": int(structural.sum()), "structural_counts": expected["structural_counts"],
                        "semantic_classification": "source structural masonry; source natural mask unchanged",
                        "verifier_sha256": sha256_file(Path(__file__))}
