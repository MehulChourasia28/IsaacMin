"""Shared regional coarse surface extraction without dense world volumes."""
from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np

from .anvil import Region, dimensions
from .nbt import SourceError
from .semantics import GROUND_CLASSES, classify
from .snapshot import canonical_hash, file_inventory, sha256, write_json


def macro_producer_identity() -> dict:
    """Every parser/semantic input that can change the sampled source surface."""
    folder = Path(__file__).parent
    names = ("macro.py", "anvil.py", "nbt.py", "semantics.py", "snapshot.py")
    return {"name": "isaacmin.source.macro", "algorithm_version": 3,
            "code_sha256": {name: sha256(folder / name) for name in names}}


def _sample_section(chunk, sy: int, coordinates: np.ndarray) -> tuple[list, np.ndarray]:
    section = chunk.sections[sy]
    states = section.get("block_states")
    if states is None:
        if "Palette" not in section:
            raise SourceError("Macro surface needs named block palettes")
        palette, data = section["Palette"], section.get("BlockStates")
    else:
        palette, data = states["palette"], states.get("data")
    if data is None:
        if len(palette) != 1:
            raise SourceError("Multiple palette states without indices")
        return palette, np.zeros(len(coordinates), dtype=np.uint16)
    bits = max(4, (len(palette)-1).bit_length())
    words = np.asarray(data, dtype=np.int64).view(np.uint64)
    coordinates = coordinates.astype(np.uint64)
    if chunk.data_version >= 2529:
        per_long = 64//bits
        if len(words) != math.ceil(4096/per_long):
            raise SourceError("Macro surface block-state storage length invalid")
        ids = (words[coordinates//per_long] >> ((coordinates%per_long)*bits)) & ((1<<bits)-1)
    else:
        if len(words) != math.ceil(4096*bits/64):
            raise SourceError("Macro surface legacy block-state storage length invalid")
        bit = coordinates*bits
        wi, shift = bit//64, bit%64
        ids = words[wi] >> shift
        crossing = shift+bits > 64
        ids[crossing] |= words[wi[crossing]+1] << (64-shift[crossing])
        ids &= (1<<bits)-1
    if (ids >= len(palette)).any():
        raise SourceError("Macro surface palette index out of range")
    return palette, ids.astype(np.uint16)


def extract_macro_surface(snapshot: Path, output: Path, center: tuple[float,float], extent: int = 2048, spacing: int = 4,
                          *, snapshot_sha256: str | None = None) -> dict:
    if spacing not in (1,2,4,8,16) or extent%16 or extent < 32 or extent > 4096:
        raise SourceError("Macro extent must be chunk-aligned [32,4096] and spacing one of 1,2,4,8,16")
    snapshot,output = Path(snapshot).resolve(),Path(output).resolve()
    if output == snapshot or snapshot in output.parents:
        raise SourceError("Macro output must be outside source snapshot")
    source_files = file_inventory(snapshot)
    verified_snapshot_hash = canonical_hash(source_files)
    snapshot_manifest = snapshot.parent / "snapshot.json"
    if snapshot_manifest.is_file():
        record = json.loads(snapshot_manifest.read_text())
        if record.get("save_sha256") != verified_snapshot_hash or record.get("files") != source_files:
            raise SourceError("Macro snapshot manifest does not match actual source bytes")
    if snapshot_sha256 is not None and snapshot_sha256 != verified_snapshot_hash:
        raise SourceError("Supplied macro snapshot hash differs from actual source bytes")
    source_index = {entry["path"]: entry for entry in source_files}
    dependencies = {"level.dat"}
    producer = macro_producer_identity()
    folder = dimensions(snapshot).get("minecraft:overworld")
    if folder is None:
        raise SourceError("Macro natural surface requires Overworld")
    # Keep the requested center within half a chunk of the selected footprint.
    # Always flooring moved a nearly aligned negative center a full chunk west
    # and could read the unfinished border of an otherwise complete Chunky area.
    xmin,zmin = (math.floor((center[0]+8)/16)-extent//32)*16,(math.floor((center[1]+8)/16)-extent//32)*16
    xmax,zmax = xmin+extent,zmin+extent
    shape=(extent//spacing,extent//spacing)
    height,water_height = np.zeros(shape,dtype=np.float32),np.zeros(shape,dtype=np.float32)
    validity,water_validity = np.zeros(shape,dtype=bool),np.zeros(shape,dtype=bool)
    biome=np.full(shape,"unknown",dtype="U96")
    substrate=np.full(shape,"unknown",dtype="U64")
    status_counts,unknowns=Counter(),Counter()
    exclusions,source_regions=[],{}
    missing_region_paths = []
    regions={}
    local_z,local_x=np.meshgrid(np.arange(spacing//2,16,spacing),np.arange(spacing//2,16,spacing),indexing="ij")
    # A coarse sample observes an actual block centre, not a fabricated average.
    columns=(local_z.ravel()*16+local_x.ravel()).astype(np.uint64)
    source_y=np.repeat(np.arange(15,-1,-1),len(columns))
    indices=(source_y*256+np.tile(columns,16)).astype(np.uint64)
    started=time.monotonic()
    decoded_chunks=0
    for cz in range(zmin//16,zmax//16):
        for cx in range(xmin//16,xmax//16):
            key=(cx//32,cz//32)
            if key not in regions:
                path=folder/"region"/f"r.{key[0]}.{key[1]}.mca"
                regions[key]=Region(path) if path.exists() else None
                if path.exists():
                    relative = path.relative_to(snapshot).as_posix()
                    source_regions[relative]=sha256(path)
                    dependencies.add(relative)
                else:
                    missing_region_paths.append(path.relative_to(snapshot).as_posix())
            region=regions[key]
            if region is None or (cx,cz) not in region.locations:
                status_counts["missing"]+=1
                exclusions.append({"chunk_xz":[cx,cz],"reason":"missing"})
                continue
            chunk=region.chunk(cx,cz)
            if chunk.metadata.get("external"):
                dependencies.add((region.path.parent / f"c.{cx}.{cz}.mcc").relative_to(snapshot).as_posix())
            status_counts[chunk.status]+=1
            if not chunk.full:
                exclusions.append({"chunk_xz":[cx,cz],"reason":chunk.status})
                continue
            decoded_chunks+=1
            n=len(columns)
            heights=np.zeros(n,dtype=np.float32)
            waters=np.zeros(n,dtype=np.float32)
            found=np.zeros(n,dtype=bool)
            water_found=np.zeros(n,dtype=bool)
            unknown=np.zeros(n,dtype=bool)
            names=np.full(n,"unknown",dtype="U64")
            for sy in sorted(chunk.sections,reverse=True):
                section=chunk.sections[sy]
                palette=section.get("block_states",{}).get("palette",section.get("Palette",[]))
                if not palette:
                    raise SourceError("Full macro source section has no named palette")
                classes=[classify(block["Name"]) for block in palette]
                relevant=set(classes)&(GROUND_CLASSES|{"water","unknown"})
                if not relevant:
                    continue
                palette,ids=_sample_section(chunk,sy,indices)
                ids=ids.reshape(16,n)
                for layer,y in enumerate(range(15,-1,-1)):
                    this=np.asarray(classes)[ids[layer]]
                    new_unknown=(this=="unknown")&~found
                    for name in np.asarray([p["Name"] for p in palette])[ids[layer,new_unknown]]:
                        unknowns[str(name)]+=1
                    unknown|=new_unknown
                    natural=np.isin(this,list(GROUND_CLASSES))&~found
                    heights[natural]=sy*16+y+1
                    names[natural]=np.asarray([p["Name"] for p in palette])[ids[layer,natural]]
                    found|=natural
                    new_water=(this=="water")&~water_found
                    waters[new_water]=sy*16+y+1
                    water_found|=new_water
                if found.all():
                    break
            zz=(cz*16-zmin+local_z.ravel())//spacing
            xx=(cx*16-xmin+local_x.ravel())//spacing
            height[zz,xx],validity[zz,xx]=heights,found&~unknown
            water_height[zz,xx],water_validity[zz,xx]=waters,water_found
            substrate[zz,xx]=names
            for i in range(n):
                y=int(heights[i]-1)
                if y//16 in chunk.sections:
                    record=chunk.biomes(y//16)
                    if record:
                        p,b=record
                        biome[zz[i],xx[i]]=p[b[((y%16)//4)*16+(int(local_z.ravel()[i])//4)*4+int(local_x.ravel()[i])//4]]
    output.parent.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix=".macro-",dir=output.parent))
    try:
        # Bind the complete save identity, including previously missing regions
        # and external payloads whose .mca stub did not change. Detect concurrent
        # source mutation before promoting a coherent surface artifact.
        if file_inventory(snapshot) != source_files:
            raise SourceError("Source changed during macro extraction")
        if macro_producer_identity() != producer:
            raise SourceError("Macro reader or semantics changed during extraction")
        np.savez_compressed(staging/"macro_surface.npz",height=height,validity=validity,water_height=water_height,water_validity=water_validity,
                            biome=biome,substrate=substrate,min_xz=np.asarray([xmin,zmin]),sample_spacing_m=np.asarray(float(spacing)),
                            sample_offset_m=np.asarray(float(spacing//2)+0.5))
        report={"schema_version":2,"kind":"RegionalMacroSurface","status":"validated_source_surface",
                "scope":{"bounds_blocks_xz":[xmin,zmin,xmax,zmax],"requested_center_xz":list(center),
                         "alignment":"nearest chunk-aligned extent; at most8m center shift per axis",
                         "actual_center_xz":[(xmin+xmax)/2,(zmin+zmax)/2],
                         "full_chunks":decoded_chunks,"status_counts":dict(status_counts),"exclusions":exclusions},
                "surface":{"file":"macro_surface.npz","shape":list(shape),"axis_order":"z,x","sample_spacing_m":spacing,"sample_offset_m":spacing//2+0.5,
                           "sample_position":"actual block centres at min_xz + index*spacing + sample_offset_m","height":"exact natural block top in Minecraft y metres",
                           "valid_samples":int(validity.sum()),"invalid_samples":int((~validity).sum()),"height_range_m":[float(height[validity].min()),float(height[validity].max())] if validity.any() else None,
                           "coarse_sampling_limit":"macro context only; does not replace 1m exact source volumes or near-field terrain"},
                "biomes":dict(Counter(biome[validity].tolist())),"unknown_semantics":dict(unknowns),
                "source_regions":source_regions,"source_level_dat_sha256":sha256(snapshot/"level.dat"),
                "source_snapshot_sha256":verified_snapshot_hash,
                "source_dependency_closure":[source_index[name] for name in sorted(dependencies)],
                "missing_region_paths":sorted(missing_region_paths), "producer":producer,
                "global_drainage_status":"not_run; this shared contextual surface is its input",
                "route_status":"not_run; 2km extent and connected chunks do not establish a traversable route",
                "elapsed_seconds":round(time.monotonic()-started,3),"files":[{"path":"macro_surface.npz","sha256":sha256(staging/"macro_surface.npz"),"bytes":(staging/"macro_surface.npz").stat().st_size}]}
        report["content_sha256"]=canonical_hash(report["files"])
        write_json(staging/"macro_surface.json",report)
        if output.exists():
            raise SourceError("Macro destination exists; preserve prior artifacts and choose a candidate path")
        os.replace(staging,output)
        return report
    except BaseException:
        shutil.rmtree(staging,ignore_errors=True)
        raise
