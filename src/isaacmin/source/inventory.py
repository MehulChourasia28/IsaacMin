"""Inventory actual stored chunks; generation state is distinct from coverage."""
from __future__ import annotations

import json
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

from .anvil import Region, dimensions
from .nbt import SourceError, load
from .provenance import begin_source_read, finish_source_read
from .semantics import biome_family, classify
from .snapshot import canonical_hash, sha256, write_json


def components(points: set[tuple[int, int]]) -> list[dict]:
    remaining = set(points)
    result = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        queue = deque([start])
        count, minx, maxx, minz, maxz = 0, start[0], start[0], start[1], start[1]
        while queue:
            x, z = queue.popleft()
            count += 1
            minx, maxx, minz, maxz = min(minx, x), max(maxx, x), min(minz, z), max(maxz, z)
            for point in ((x - 1, z), (x + 1, z), (x, z - 1), (x, z + 1)):
                if point in remaining:
                    remaining.remove(point)
                    queue.append(point)
        result.append({"chunk_count": count, "bounds_blocks_xz": [minx * 16, minz * 16, (maxx + 1) * 16, (maxz + 1) * 16], "seed_chunk": list(start)})
    return sorted(result, key=lambda item: -item["chunk_count"])


def inspect_source(world: Path, output: Path, *, snapshot_sha256: str | None = None) -> dict:
    world, output = Path(world).resolve(), Path(output).resolve()
    if output == world or world in output.parents:
        raise SourceError("Source report destination must be outside input save")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    identity, source_files = begin_source_read(world, "inventory", snapshot_sha256)
    source_index = {entry["path"]: entry for entry in source_files}
    dependencies = {"level.dat"}
    level = load(world / "level.dat")
    data = level.get("Data", level)
    spawn = data.get("spawn")
    if spawn is None:
        spawn = {"pos": [data.get("SpawnX"), data.get("SpawnY"), data.get("SpawnZ")], "dimension": "minecraft:overworld"}
    report = {
        "schema_version": 2, "kind": "SourceInventory", "status": "validated", "source_path": str(world),
        **identity,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_version": int(data.get("DataVersion", 0)), "version": data.get("Version", {}),
        "level_name": str(data.get("LevelName", "")), "spawn": spawn,
        "level_dat_sha256": sha256(world / "level.dat"), "parser": "isaacmin-java-nbt-anvil/1",
        "datapacks": data.get("DataPacks", {}), "datapacks_executed": False,
        "dimensions": [], "unknown_blocks": [], "errors": [],
        "coverage_policy": {"missing_chunk": "unknown/excluded; never air", "incomplete_generation": "inventoried; excluded from production region by default", "unstored_section": "unknown outside stored vertical coverage"},
    }
    all_unknown = set()
    for dimension_id, folder in dimensions(world).items():
        dimension = {"id": dimension_id, "path": folder.relative_to(world).as_posix(), "regions": [], "status_counts": {}, "data_versions": [], "biomes": {}, "biome_families": {}, "blocks": {}, "compression": {}, "external_chunk_count": 0}
        statuses, biomes, blocks, codecs = Counter(), Counter(), Counter(), Counter()
        versions, section_ys, stored, full = set(), set(), set(), set()
        chunks = []
        for region_path in sorted((folder / "region").glob("*.mca")):
            region = Region(region_path)
            dependencies.add(region_path.relative_to(world).as_posix())
            dimension["regions"].append({"path": region_path.relative_to(world).as_posix(), "sha256": sha256(region_path), "stored_chunks": len(region.locations), "region_xz": [region.x, region.z]})
            for cx, cz in sorted(region.locations):
                stored.add((cx, cz))
                try:
                    chunk = region.chunk(cx, cz)
                    if chunk.metadata["external"]:
                        dependencies.add((region_path.parent / f"c.{cx}.{cz}.mcc").relative_to(world).as_posix())
                    statuses[chunk.status] += 1
                    versions.add(chunk.data_version)
                    section_ys.update(chunk.sections)
                    if chunk.full:
                        full.add((cx, cz))
                    codecs[str(chunk.metadata["compression"])] += 1
                    dimension["external_chunk_count"] += int(chunk.metadata["external"])
                    chunk_biomes, chunk_blocks = set(), set()
                    for section in chunk.sections.values():
                        states = section.get("block_states", {})
                        palette = states.get("palette", section.get("Palette", []))
                        if "Blocks" in section:
                            raise SourceError("Legacy numeric block IDs require explicit version registry")
                        for block in palette:
                            name = block["Name"]
                            chunk_blocks.add(name)
                            if classify(name) == "unknown":
                                all_unknown.add(name)
                        chunk_biomes.update(section.get("biomes", {}).get("palette", []))
                    blocks.update(chunk_blocks)
                    biomes.update(chunk_biomes)
                    chunks.append({"x": cx, "z": cz, "status": chunk.status, "data_version": chunk.data_version,
                                   "section_y": sorted(chunk.sections), "biomes": sorted(chunk_biomes), "region": region_path.name})
                except (SourceError, OSError, KeyError, TypeError) as exc:
                    report["errors"].append({"dimension": dimension_id, "chunk_xz": [cx, cz], "reason": str(exc)})
        dimension["status_counts"], dimension["data_versions"] = dict(statuses), sorted(versions)
        dimension["biomes"], dimension["blocks"], dimension["compression"] = dict(biomes), dict(blocks), dict(codecs)
        dimension["biome_families"] = dict(Counter({key: sum(value for biome, value in biomes.items() if biome_family(biome) == key) for key in sorted(set(map(biome_family, biomes)))}))
        dimension["biome_count_semantics"] = "chunks whose section palettes contain each biome; not surface area"
        dimension["stored_chunk_count"], dimension["full_chunk_count"] = len(stored), len(full)
        dimension["connected_full_components"] = components(full)
        dimension["vertical_stored_range_blocks"] = [min(section_ys) * 16, (max(section_ys) + 1) * 16] if section_ys else None
        dimension["bounds_blocks_xz"] = [min(x for x, _ in stored) * 16, min(z for _, z in stored) * 16, (max(x for x, _ in stored) + 1) * 16, (max(z for _, z in stored) + 1) * 16] if stored else None
        dimension["stored_area_m2"] = len(stored) * 256
        dimension["full_generated_area_m2"] = len(full) * 256
        chunk_file = output / (dimension_id.replace(":", "_").replace("/", "_") + "_chunks.json")
        write_json(chunk_file, chunks)
        dimension["coverage_file"] = chunk_file.name
        dimension["coverage_sha256"] = sha256(chunk_file)
        report["dimensions"].append(dimension)
    report["unknown_blocks"] = sorted(all_unknown)
    finish_source_read(world, "inventory", identity, source_files)
    report["source_dependency_closure"] = [source_index[name] for name in sorted(dependencies)]
    report["files"] = [{"path": dimension["coverage_file"], "sha256": dimension["coverage_sha256"],
                        "bytes": (output / dimension["coverage_file"]).stat().st_size} for dimension in report["dimensions"]]
    report["unknown_semantic_policy"] = "retain exact block names/states and mark unknown occupancy; block affected-region qualification"
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    report["status"] = "fail" if report["errors"] else "validated"
    report["content_identity"] = canonical_hash({**identity, "source_dependency_closure": report["source_dependency_closure"], "files": report["files"]})
    write_json(output / "source_inventory.json", report)
    return report
