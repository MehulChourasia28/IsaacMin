"""Exact source volume retention and conservative natural-surface extraction.

This is a source IR, not photographic terrain, mesh refinement, or a visual pass.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .anvil import Region, dimensions
from .nbt import SourceError, load
from .provenance import begin_source_read, finish_source_read
from .semantics import GROUND_CLASSES, biome_family, classify
from .snapshot import canonical_hash, sha256, write_json


OCCUPANCY = {"air": 0, "natural_terrain": 1, "nonterrain": 2, "unknown": 3, "water": 4, "lava": 5}


def _occupancy(semantic: str) -> int:
    if semantic in GROUND_CLASSES:
        return 1
    return {"air": 0, "unknown": 3, "water": 4, "lava": 5}.get(semantic, 2)


def extract_world_ir(snapshot: Path, output: Path, bounds: tuple[int, int, int, int] | None = None,
                     center: tuple[float, float] | None = None, extent: int = 128,
                     dimension: str = "minecraft:overworld", *, snapshot_sha256: str | None = None) -> dict:
    snapshot, output = Path(snapshot).resolve(), Path(output).resolve()
    if output == snapshot or snapshot in output.parents:
        raise SourceError("IR must be outside immutable source snapshot")
    if extent < 32 or extent > 2048 or extent % 16:
        raise SourceError("Region extent must be a multiple of 16 in [32,2048]")
    source_identity, source_files = begin_source_read(snapshot, "world_ir", snapshot_sha256)
    source_index = {entry["path"]: entry for entry in source_files}
    dependencies = {"level.dat"}
    data = load(snapshot / "level.dat")
    data = data.get("Data", data)
    spawn = data.get("spawn", {}).get("pos", [data.get("SpawnX", 0), data.get("SpawnY", 0), data.get("SpawnZ", 0)])
    center_origin = "caller_requested" if center is not None else "world_spawn"
    center = center or (float(spawn[0]), float(spawn[2]))
    if bounds is None:
        xmin = (math.floor(center[0] / 16) - extent // 32) * 16
        zmin = (math.floor(center[1] / 16) - extent // 32) * 16
        bounds = (xmin, zmin, xmin + extent, zmin + extent)
    xmin, zmin, xmax, zmax = map(int, bounds)
    if any(v % 16 for v in bounds) or xmax <= xmin or zmax <= zmin or max(xmax-xmin,zmax-zmin) > 2048:
        raise SourceError("Extraction bounds must be chunk aligned, positive, and at most 2048m per axis")
    folder = dimensions(snapshot).get(dimension)
    if folder is None:
        raise SourceError(f"Requested dimension unavailable: {dimension}")
    if dimension != "minecraft:overworld":
        raise SourceError(f"Natural-world interpretation not qualified for dimension {dimension}")
    region_cache, chunk_records, all_sections, exclusions = {}, [], set(), []
    for cz in range(zmin // 16, zmax // 16):
        for cx in range(xmin // 16, xmax // 16):
            key = (cx // 32, cz // 32)
            if key not in region_cache:
                path = folder / "region" / f"r.{key[0]}.{key[1]}.mca"
                region_cache[key] = Region(path) if path.exists() else None
                if path.exists():
                    dependencies.add(path.relative_to(snapshot).as_posix())
            region = region_cache[key]
            if region is None or (cx, cz) not in region.locations:
                exclusions.append({"chunk_xz": [cx, cz], "reason": "missing_source_chunk"})
                continue
            chunk = region.chunk(cx, cz)
            if chunk.metadata.get("external"):
                dependencies.add((region.path.parent / f"c.{cx}.{cz}.mcc").relative_to(snapshot).as_posix())
            if not chunk.full:
                exclusions.append({"chunk_xz": [cx, cz], "reason": "incomplete_generation", "status": chunk.status})
                continue
            if not chunk.sections:
                raise SourceError("Full chunk without stored sections")
            all_sections.update(chunk.sections)
            chunk_records.append((cx, cz, region))
    if not chunk_records:
        raise SourceError("Requested region has no fully generated source chunks")
    ymin, ymax = min(all_sections) * 16, (max(all_sections) + 1) * 16
    shape = (ymax - ymin, zmax - zmin, xmax - xmin)
    # Refuse an unsafe allocation rather than lowering precision or dropping caves.
    required_bytes = math.prod(shape) * 3
    try:
        available = int(next(line.split()[1] for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))) * 1024
    except (OSError, StopIteration):
        available = 8 * 1024**3
    if required_bytes > min(64 * 1024**3, max(0, available - 2 * 1024**3)):
        raise SourceError("Dense IR exceeds safe memory estimate; extract contextual production tiles with the same 1m source fidelity")
    occupancy = np.full(shape, 3, dtype=np.uint8)
    validity = np.zeros(shape, dtype=np.bool_)
    biome_surface = np.full(shape[1:], "unknown", dtype="U96")
    ground_id = np.zeros(shape, dtype=np.uint16)
    block_names = ["__unknown__"]
    block_ids = {"__unknown__": 0}
    semantics, unknowns, state_counts, biome_counts = Counter(), Counter(), Counter(), Counter()
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".world-ir-", dir=output.parent))
    volume_dir = staging / "terrain_volume"
    volume_dir.mkdir()
    files, chunks_manifest = [], []
    try:
        for cx, cz, region in chunk_records:
            chunk = region.chunk(cx, cz)
            sections = sorted(chunk.sections)
            exact_palette, palette_indices = [], {}
            section_arrays, section_biomes = [], []
            for sy in sections:
                palette, packed = chunk.section(sy)
                array = np.asarray(packed, dtype=np.uint16).reshape((16, 16, 16))
                classes = [classify(block["Name"]) for block in palette]
                global_ids, local_ids = [], []
                for block in palette:
                    name = block["Name"]
                    if name not in block_ids:
                        block_ids[name] = len(block_names)
                        block_names.append(name)
                    global_ids.append(block_ids[name])
                    identity = json.dumps(block, sort_keys=True, separators=(",", ":"))
                    if identity not in palette_indices:
                        palette_indices[identity] = len(exact_palette)
                        exact_palette.append(block)
                    local_ids.append(palette_indices[identity])
                counts = np.bincount(array.ravel(), minlength=len(palette))
                for i, count in enumerate(counts):
                    semantics[classes[i]] += int(count)
                    state_counts[palette[i]["Name"]] += int(count)
                    if classes[i] == "unknown" and count:
                        unknowns[palette[i]["Name"]] += int(count)
                yslice = slice(sy*16-ymin, sy*16-ymin+16)
                zslice, xslice = slice(cz*16-zmin, cz*16-zmin+16), slice(cx*16-xmin, cx*16-xmin+16)
                occupancy[yslice, zslice, xslice] = np.asarray([_occupancy(c) for c in classes], dtype=np.uint8)[array]
                validity[yslice, zslice, xslice] = True
                ground_id[yslice, zslice, xslice] = np.asarray(global_ids, dtype=np.uint16)[array]
                section_arrays.append(np.asarray(local_ids, dtype=np.uint16)[array])
                biome = chunk.biomes(sy)
                if biome:
                    bp, bi = biome
                    biome_counts.update({name: bi.count(i) for i, name in enumerate(bp)})
                    section_biomes.append({"section_y": sy, "palette": bp, "indices": bi})
            name = f"terrain_volume/c.{cx}.{cz}.npz"
            np.savez_compressed(staging / name, block_id=np.stack(section_arrays), section_y=np.asarray(sections, dtype=np.int16),
                                palette_json=np.asarray(json.dumps(exact_palette, sort_keys=True)),
                                biomes_json=np.asarray(json.dumps(section_biomes, sort_keys=True)),
                                min_xyz=np.asarray([cx*16, min(sections)*16, cz*16], dtype=np.int32))
            chunks_manifest.append({"chunk_xz": [cx, cz], "status": chunk.status, "data_version": chunk.data_version,
                                    "volume_file": name, "source_region": region.path.relative_to(snapshot).as_posix(),
                                    "section_y": sections})
        terrain = occupancy == 1
        any_ground = terrain.any(axis=0)
        top_index = shape[0] - 1 - terrain[::-1].argmax(axis=0)
        height = np.where(any_ground, top_index + ymin + 1, 0).astype(np.float32)
        zz, xx = np.indices(shape[1:])
        substrate = ground_id[top_index, zz, xx]
        unknown_above = ((occupancy == 3) & (np.arange(shape[0])[:, None, None] >= top_index[None])).any(axis=0)
        surface_validity = any_ground & ~unknown_above & validity.all(axis=0)
        water = occupancy == 4
        has_water = water.any(axis=0)
        water_index = shape[0] - 1 - water[::-1].argmax(axis=0)
        water_height = np.where(has_water, water_index + ymin + 1, 0).astype(np.float32)
        # Store every natural block with clear air directly above. These preserve
        # stacked floors; connectivity and portal identity require a later 3D stage.
        supporting = np.zeros(shape, dtype=bool)
        supporting[:-1] = terrain[:-1] & (occupancy[1:] == 0) & validity[1:]
        floor_y, floor_z, floor_x = np.nonzero(supporting)
        covered_air = (occupancy == 0) & (np.arange(shape[0])[:, None, None] < top_index[None]) & validity
        protection = covered_air.any(axis=0)
        for cx, cz, region in chunk_records:
            chunk = region.chunk(cx, cz)
            for lz in range(16):
                for lx in range(16):
                    rz, rx = cz*16-zmin+lz, cx*16-xmin+lx
                    sy = int(height[rz, rx]-1) // 16
                    if sy in chunk.sections:
                        biome = chunk.biomes(sy)
                        if biome:
                            bp, bi = biome
                            y = int(height[rz, rx]-1)
                            biome_surface[rz, rx] = bp[bi[((y % 16)//4)*16+(lz//4)*4+lx//4]]
        np.savez_compressed(staging / "terrain_surface.npz", height=height, validity=surface_validity, substrate_id=substrate,
                            block_names=np.asarray(block_names), biome=biome_surface, water_height=water_height, water_validity=has_water,
                            cave_column_protection=protection, min_xz=np.asarray([xmin,zmin]), sample_spacing_m=np.asarray(1.0))
        np.savez_compressed(staging / "natural_occupancy.npz", occupancy=occupancy, validity=validity,
                            min_xyz=np.asarray([xmin, ymin, zmin], dtype=np.int32), voxel_size_m=np.asarray(1.0))
        np.savez_compressed(staging / "supporting_surfaces.npz", xyz=np.stack([floor_x+xmin+0.5,floor_y+ymin+1,floor_z+zmin+0.5],axis=1).astype(np.float32),
                            interpretation=np.asarray("source block top with immediately adjacent air; no robot-clearance inference"))
        for path in sorted(staging.rglob("*")):
            if path.is_file():
                files.append({"path":path.relative_to(staging).as_posix(),"bytes":path.stat().st_size,"sha256":sha256(path)})
        manifest = {
            "schema_version":2,"kind":"WorldIR","status":"validated_source_ir", "created_at_utc":datetime.now(timezone.utc).isoformat(),
            **source_identity, "source_dependency_closure":[source_index[name] for name in sorted(dependencies)],
            "source_path":str(snapshot),"dimension":dimension,"source_level_dat_sha256":sha256(snapshot/"level.dat"),
            "scope":{"bounds_blocks_xz":list(bounds),"vertical_bounds_blocks":[ymin,ymax],"requested_center_xz":list(center),
                     "center_provenance":center_origin,"full_chunks":len(chunk_records),"exclusions":exclusions,
                     "milestone":"first source integration region; not full provided-world delivery", "unique_route_coverage_m":None,
                     "long_range_route_status":"not_run; geographic extent does not establish traversability"},
            "coordinate_frame":{"source":"Minecraft block cells", "source_origin_xyz":[center[0],0,center[1]], "metres_per_block":1,
                                "target":"right-handed Z-up", "transform":[[1,0,0,-center[0]],[0,0,-1,center[1]],[0,1,0,0],[0,0,0,1]],
                                "sample_convention":"source x,z block centre; height is top face y+1"},
            "terrain_surface":{"file":"terrain_surface.npz","axis_order":"z,x","shape":list(shape[1:]),"dtype":"float32","units":"metres Minecraft y",
                               "validity":"false for missing/incomplete/uninterpreted columns; height zero there is not terrain",
                               "multiple_support_surfaces":"supporting_surfaces.npz","support_surface_count":len(floor_x)},
            "terrain_volume":{"dense_file":"natural_occupancy.npz","axis_order":"y,z,x","shape":list(shape),"dtype":"uint8",
                              "occupancy_codes":OCCUPANCY,"validity":"stored full chunk section coverage; unknown is distinct from air",
                              "sparse_exact_blocks":"terrain_volume/*.npz", "exact_axis_order":"section,y,z,x", "chunk_records":chunks_manifest,
                              "source_block_states_preserved":True},
            "semantics":{"block_counts":dict(state_counts),"class_counts":dict(semantics),"unknown_block_counts":dict(unknowns),
                         "biome_quart_samples":dict(biome_counts),"surface_biomes":dict(Counter(biome_surface[surface_validity].tolist())),
                         "structures_policy":"retained in exact source volumes; excluded from natural ground; scene reconstruction pending"},
            "features":{"covered_air_voxels":int(covered_air.sum()),"columns_with_covered_air":int(protection.sum()),
                        "cave_status":"potential subsurface voids detected" if protection.any() else "no covered air in selected source crop",
                        "portal_connectivity_status":"not_run; covered air alone does not prove a connected cave or an exterior portal",
                        "protection":"all columns containing covered source air protected until full 3D ownership analysis"},
            "routes":{"status":"not_authored", "source_dirt_path_blocks":int(state_counts.get("minecraft:dirt_path",0)),
                      "policy":"individual path blocks do not establish a connected trail; any authored route must be labelled synthesized"},
            "inference":{"biome_families":sorted({biome_family(b) for b in biome_counts}),"season":"unspecified", "geology":"source lithology categories only; detailed geological recipe pending"},
            "quality":{"implemented":True,"external_tool_verified":False,"isaac_verified":False,"automated_visual_qualified":False,
                       "affected_semantics_blocked":bool(unknowns)},"files":files,
        }
        finish_source_read(snapshot, "world_ir", source_identity, source_files)
        manifest["content_sha256"] = canonical_hash(files)
        write_json(staging / "world_ir.json", manifest)
        if output.exists():
            existing = json.loads((output / "world_ir.json").read_text()) if (output / "world_ir.json").exists() else None
            if (existing and existing.get("content_sha256") == manifest["content_sha256"]
                    and existing.get("producer") == manifest["producer"]
                    and existing.get("source_snapshot_sha256") == manifest["source_snapshot_sha256"]
                    and existing.get("scope",{}).get("requested_center_xz") == list(center)
                    and existing.get("scope",{}).get("bounds_blocks_xz") == list(bounds)
                    and existing.get("dimension") == dimension):
                for entry in existing["files"]:
                    if sha256(output/entry["path"]) != entry["sha256"]:
                        raise SourceError("Existing IR content failed verification")
                shutil.rmtree(staging)
                return existing
            raise SourceError("IR destination contains another candidate; choose a new output path")
        os.replace(staging, output)
        return manifest
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
