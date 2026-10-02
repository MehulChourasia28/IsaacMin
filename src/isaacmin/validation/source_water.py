"""Independent raw-save water and known-air inventory for Java26.1.

This audits every stored section through nbtlib and independently unpacked bit
strings. It does not use the production maximum-column water surface to select
interfaces. Fluid semantics are separate from ground/vegetation ownership.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy import ndimage

from ..io import atomic_json, sha256_file
from ..source.anvil import dimensions
from ..source.independent import reference_chunk
from ..source.snapshot import canonical_hash, file_inventory


AIR_NAMES = {"minecraft:air", "minecraft:cave_air", "minecraft:void_air"}


def _fluid(state):
    name, properties = state["Name"], state.get("Properties", {})
    if name == "minecraft:water":
        level = int(properties["level"])
        if not 0 <= level <= 15:
            raise ValueError("Water block level is outside the verified runtime state range")
        return 1, level, 8 if level == 0 or level >= 8 else 8-level
    if name == "minecraft:bubble_column":
        return 2, 0, 8
    if properties.get("waterlogged") == "true":
        return 3, 0, 8
    return 0, -1, 0


def _section_indices(section):
    states = section["block_states"]
    palette = [state.unpack() for state in states["palette"]]
    if "data" not in states:
        if len(palette) != 1:
            raise ValueError("Independent fluid reader: missing palette indices")
        return palette, np.zeros((16, 16, 16), dtype=np.uint16)
    width = max(4, (len(palette)-1).bit_length())
    values = []
    for word in states["data"]:
        bits = format(int(word) % (2**64), "064b")[::-1]
        values.extend(int(bits[index*width:(index+1)*width][::-1], 2) for index in range(64//width))
    array = np.asarray(values[:4096], dtype=np.uint16).reshape((16, 16, 16))
    if int(array.max()) >= len(palette):
        raise ValueError("Independent fluid reader: palette index outside source palette")
    return palette, array


def audit_source_water(snapshot: Path, ir_dir: Path, output: Path, *, reference_manifest_path: Path | None = None) -> dict:
    snapshot, ir_dir, output = Path(snapshot), Path(ir_dir), Path(output)
    ir_path = ir_dir / "world_ir.json"
    ir = json.loads(ir_path.read_text())
    from ..security import safe_path
    for entry in ir["files"]:
        path = safe_path(ir_dir, entry["path"], must_exist=True)
        if sha256_file(path) != entry["sha256"]:
            raise ValueError("Retained source artifact changed before independent water inventory")
    snapshot_hash = canonical_hash(file_inventory(snapshot))
    if ir["source_snapshot_sha256"] != snapshot_hash:
        raise ValueError("Water audit source snapshot differs from retained WorldIR")
    with np.load(ir_dir / "natural_occupancy.npz", allow_pickle=False) as data:
        occupancy, validity, minimum = data["occupancy"], data["validity"], data["min_xyz"].astype(int)
    category = np.zeros_like(occupancy, dtype=np.uint8)
    level = np.full_like(occupancy, -1, dtype=np.int8)
    amount = np.zeros_like(occupancy, dtype=np.uint8)
    reference_air = np.zeros_like(validity)
    reference_valid = np.zeros_like(validity)
    fluid_id = np.zeros_like(occupancy, dtype=np.uint16)
    state_palette, state_ids = [], {}
    state_counts, levels, waterlogged_counts = Counter(), Counter(), Counter()
    comparisons, errors = 0, []
    region_dir = dimensions(snapshot)[ir["dimension"]] / "region"
    for record in ir["terrain_volume"]["chunk_records"]:
        cx, cz = record["chunk_xz"]
        version, sections = reference_chunk(region_dir, cx, cz)
        if version != 4786:
            raise ValueError("Fluid audit is explicitly verified for DataVersion4786/Java26.1")
        with np.load(ir_dir / f"terrain_volume/c.{cx}.{cz}.npz", allow_pickle=False) as exact:
            retained_palette = json.loads(str(exact["palette_json"]))
            retained_ys = exact["section_y"].tolist()
            retained_ids = exact["block_id"]
        for sy, section in sections.items():
            palette, ids = _section_indices(section)
            y0, z0, x0 = sy*16-minimum[1], cz*16-minimum[2], cx*16-minimum[0]
            dst = (slice(y0, y0+16), slice(z0, z0+16), slice(x0, x0+16))
            facts = [_fluid(state) for state in palette]
            cats = np.asarray([fact[0] for fact in facts], dtype=np.uint8)[ids]
            category[dst] = cats
            level[dst] = np.asarray([fact[1] for fact in facts], dtype=np.int8)[ids]
            amount[dst] = np.asarray([fact[2] for fact in facts], dtype=np.uint8)[ids]
            reference_air[dst] = np.asarray([state["Name"] in AIR_NAMES for state in palette])[ids]
            reference_valid[dst] = True
            palette_fluid_ids = np.zeros(len(palette), dtype=np.uint16)
            for index, state in enumerate(palette):
                if not facts[index][0]:
                    continue
                identity = json.dumps(state, sort_keys=True, separators=(",", ":"))
                if identity not in state_ids:
                    state_ids[identity] = len(state_palette)
                    state_palette.append(state)
                palette_fluid_ids[index] = state_ids[identity]
                count = int(np.count_nonzero(ids == index))
                if not count:
                    continue
                state_counts[identity] += count
                if state["Name"] == "minecraft:water":
                    levels[str(facts[index][1])] += count
                if facts[index][0] == 3:
                    waterlogged_counts[state["Name"]] += count
            fluid_id[dst] = palette_fluid_ids[ids]
            exact = retained_ids[retained_ys.index(sy)]
            retained_fluid = np.asarray([_fluid(state)[0] > 0 for state in retained_palette])[exact]
            for y, z, x in np.argwhere((cats > 0) | retained_fluid):
                comparisons += 1
                source, retained = palette[int(ids[y, z, x])], retained_palette[int(exact[y, z, x])]
                if source != retained:
                    errors.append({"source_xyz": [int(cx*16+x), int(sy*16+y), int(cz*16+z)], "reference": source, "retained": retained})
    fluid = category > 0
    free = np.zeros_like(fluid)
    free[:-1] = fluid[:-1] & reference_air[1:] & reference_valid[1:]
    source_ground = (occupancy == 1) & validity
    above_ground = np.zeros_like(source_ground)
    above_ground[:-1] = np.maximum.accumulate(source_ground[:0:-1], axis=0)[::-1]
    covered_free = free & above_ground
    upper_unknown = np.zeros_like(fluid)
    upper_unknown[-1] = fluid[-1]
    upper_unknown[:-1] = fluid[:-1] & ~reference_valid[1:]
    same_water_above = np.zeros_like(fluid)
    same_water_above[:-1] = fluid[:-1] & fluid[1:]
    physical_height = np.where(same_water_above, 1., amount.astype(float)/9.)
    positions = np.argwhere(fluid)
    free_positions = np.argwhere(free)
    coordinates = positions[:, [2, 0, 1]]+minimum
    free_coordinates = free_positions[:, [2, 0, 1]]+minimum
    hist = Counter((free_coordinates[:, 1]+1).tolist())
    covered_hist = Counter((np.argwhere(covered_free)[:, 0]+minimum[1]+1).tolist())
    surfaces_per_column = free.sum(axis=0)
    labels, component_count = ndimage.label(fluid, ndimage.generate_binary_structure(3, 1))
    components = []
    for component, slices in enumerate(ndimage.find_objects(labels), start=1):
        if slices is None:
            continue
        member = labels[slices] == component
        exposed = member & free[slices]
        under_cover = member & covered_free[slices]
        coordinates_y = np.argwhere(exposed)[:, 0]+slices[0].start+minimum[1]
        components.append({"id": component, "fluid_cells": int(member.sum()), "upper_air_interfaces": int(exposed.sum()),
            "upper_air_interfaces_under_natural_cover": int(under_cover.sum()),
            "upper_interface_cell_y_levels": sorted(set(coordinates_y.tolist())),
            "crop_boundary": any(part.start == 0 or part.stop == occupancy.shape[axis] for axis, part in enumerate(slices))})
    air_errors = int(np.count_nonzero(reference_valid & (reference_air != (occupancy == 0))))
    water_errors = int(np.count_nonzero(reference_valid & (np.isin(category, [1, 2]) != (occupancy == 4))))
    occupancy_unknown = int((~reference_valid & validity).sum())
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "source_water_samples.npz"
    np.savez_compressed(raw, water_source_xyz=coordinates, source_state_id=fluid_id[fluid], source_state_palette_json=np.asarray(json.dumps(state_palette, sort_keys=True)),
        water_category=category[fluid], block_level=level[fluid], fluid_amount=amount[fluid], same_water_above=same_water_above[fluid],
        fluid_own_or_full_height=physical_height[fluid], component_id=labels[fluid],
        free_upper_source_xyz=free_coordinates, free_upper_state_id=fluid_id[free], free_upper_under_cover=covered_free[free],
        free_upper_own_height=physical_height[free], free_upper_component_id=labels[free], source_min_xyz=minimum,
        known_air=reference_air, known_valid=reference_valid, fluid_category_grid=category, fluid_level_grid=level)
    unverified_waterlogged = {name: count for name, count in waterlogged_counts.items() if name != "minecraft:glow_lichen"}
    if canonical_hash(file_inventory(snapshot)) != snapshot_hash:
        raise ValueError("Source snapshot changed during independent fluid decoding")
    result = {"schema_version": 1, "kind": "IndependentSourceWaterInventory", "status": "fail" if errors or air_errors or water_errors or occupancy_unknown or unverified_waterlogged else "pass",
        "data_version": 4786, "minecraft_version": "26.1", "source_snapshot_sha256": snapshot_hash,
        "source_ir_sha256": sha256_file(ir_path), "validator_sha256": sha256_file(Path(__file__)),
        "independent_nbt_reader_sha256": sha256_file(Path(__file__).parents[1] / "source/independent.py"),
        "method": "nbtlib rawAnvil sections; independent reversed binary-string palette indexing; all rawfluid states compared with exact retained state palettes",
        "source_state_comparisons": comparisons, "state_discrepancies": errors, "water_occupancy_discrepancies": water_errors,
        "known_air_occupancy_discrepancies": air_errors, "validity_discrepancies": occupancy_unknown,
        "water_state_counts": dict(state_counts), "water_block_level_counts": dict(sorted(levels.items(), key=lambda item: int(item[0]))),
        "explicit_water_cells": int((category == 1).sum()), "bubble_column_cells": int((category == 2).sum()),
        "waterlogged_cells": int((category == 3).sum()), "waterlogged_state_counts": dict(waterlogged_counts),
        "unverified_waterlogged_fluid_semantics": unverified_waterlogged,
        "source_water_min_max_cell_y": [int(coordinates[:, 1].min()), int(coordinates[:, 1].max())] if len(coordinates) else None,
        "free_upper_air_interfaces": int(free.sum()), "free_upper_air_interfaces_under_cover": int(covered_free.sum()),
        "free_upper_boundary_y_counts": dict(sorted(hist.items())), "covered_upper_boundary_y_counts": dict(sorted(covered_hist.items())),
        "minimum_free_upper_boundary_y": min(hist) if hist else None,
        "maximum_free_interfaces_in_one_xz_column": int(surfaces_per_column.max()),
        "columns_with_multiple_free_water_layers": int((surfaces_per_column > 1).sum()),
        "water_air_interfaces_lost_by_max_column_only": int(np.maximum(surfaces_per_column.astype(int)-1, 0).sum()),
        "interfaces_with_unknown_or_crop_above": int(upper_unknown.sum()),
        "connected_fluid_components": component_count, "components": components,
        "semantic_reference": {"path": str(reference_manifest_path), "sha256": sha256_file(reference_manifest_path)} if reference_manifest_path else None,
        "files": [{"path": raw.name, "bytes": raw.stat().st_size, "sha256": sha256_file(raw)}],
        "limits": ["Integer upper-boundaryY is the source water-cell/air-cell boundary, not the actual weighted renderer corner height",
            "Recorded own/full fluid height follows verified26.1 amount/9 or samewaterabove logic; rendered shared corner averaging and1mm render inset are separate",
            "Known-air mask includes only independently decoded air/cave_air/void_air; vegetation, structures, lava and unknown cells are never silently air",
            "Waterlogged glow_lichen adds fluid without converting its vegetation or support ownership",
            "Covered freewater includes subsurface pools consistent with aquifers; exact terrain-generator origin is not inferred from block states",
            "Source inventory does not establish exported water material, bank continuity, transparency/depth fidelity or dynamic fluid simulation"]}
    atomic_json(output / "source_water_inventory.json", result)
    return result
