"""Actual-save reader parity through an independent NBT parser and bit strings.

The reference path intentionally does not call IsaacMin's NBT, decompressor,
packed-index decoder, or region-location methods. This establishes decoder
parity; it does not establish transformed terrain fidelity or portal topology.
"""
from __future__ import annotations

import gzip
import io
import json
import random
import struct
import zlib
from collections import Counter
from pathlib import Path

import nbtlib
import numpy as np

from .anvil import Region, dimensions
from .nbt import SourceError
from .snapshot import canonical_hash, sha256, write_json
from .semantics import GROUND_CLASSES, classify


def reference_chunk(region_dir: Path, cx: int, cz: int):
    region_path = region_dir / f"r.{cx//32}.{cz//32}.mca"
    with region_path.open("rb") as stream:
        stream.seek(4 * ((cz % 32) * 32 + cx % 32))
        location = int.from_bytes(stream.read(4), "big")
        sector = location >> 8
        if sector < 2:
            raise SourceError("Independent reader: missing chunk")
        stream.seek(sector * 4096)
        count = int.from_bytes(stream.read(4), "big")
        codec = stream.read(1)[0]
        if codec & 128:
            compressed = (region_dir / f"c.{cx}.{cz}.mcc").read_bytes()
        else:
            compressed = stream.read(count - 1)
    if codec & 127 == 1:
        raw = gzip.decompress(compressed)
    elif codec & 127 == 2:
        raw = zlib.decompress(compressed)
    elif codec & 127 == 3:
        raw = compressed
    else:
        raise SourceError("Independent reader: unsupported compression")
    root = nbtlib.File.parse(io.BytesIO(raw))
    chunk = root.get("Level", root)
    return int(root.get("DataVersion", 0)), {int(s["Y"]):s for s in chunk.get("sections",chunk.get("Sections",[]))}


def reference_block(chunk, x: int, y: int, z: int) -> dict:
    version, sections = chunk
    section = sections[y//16]
    states = section.get("block_states")
    palette = states["palette"] if states is not None else section["Palette"]
    longs = states.get("data") if states is not None else section.get("BlockStates")
    if longs is None:
        if len(palette) != 1:
            raise SourceError("Independent reader: missing palette storage")
        return palette[0].unpack()
    width = max(4, (len(palette)-1).bit_length())
    index = ((y % 16) << 8) + ((z % 16) << 4) + (x % 16)
    # String slicing independently traverses the bit representation. It does
    # not call or replicate production's shift/mask packed-index loop.
    if version >= 2529:
        long_index, element = divmod(index, 64 // width)
        bits = format(int(longs[long_index]) % (2**64), "064b")[::-1]
        value = int(bits[element*width:(element+1)*width][::-1], 2)
    else:
        bits = "".join(format(int(word) % (2**64), "064b")[::-1] for word in longs)
        value = int(bits[index*width:(index+1)*width][::-1], 2)
    return palette[value].unpack()


def compare_region(snapshot: Path, ir_dir: Path, output: Path | None = None, samples: int = 2048, seed: int = 4786) -> dict:
    if samples < 1000:
        raise SourceError("Independent integration-region comparison requires at least 1000 samples")
    snapshot, ir_dir = Path(snapshot), Path(ir_dir)
    ir = json.loads((ir_dir / "world_ir.json").read_text())
    xmin,zmin,xmax,zmax = ir["scope"]["bounds_blocks_xz"]
    ymin,ymax = ir["scope"]["vertical_bounds_blocks"]
    records = ir["terrain_volume"]["chunk_records"]
    surface = np.load(ir_dir/"terrain_surface.npz",allow_pickle=False)
    volume = np.load(ir_dir/"natural_occupancy.npz",allow_pickle=False)
    occupancy = volume["occupancy"]
    points = {}
    rng = random.Random(seed)
    # Include every chunk, vertical extrema, actual natural surface, just above
    # and below ground, region boundaries, and randomly stratified interiors.
    for record in records:
        cx,cz = record["chunk_xz"]
        for lx,lz in ((0,0),(15,15),(0,15),(15,0),(7,11)):
            x,z = cx*16+lx,cz*16+lz
            for y in (ymin,ymax-1,int(surface["height"][z-zmin,x-xmin])-1,int(surface["height"][z-zmin,x-xmin])):
                if ymin <= y < ymax:
                    points[(x,y,z)] = "extrema_and_ground"
    for code,label in ((4,"water"),(5,"lava"),(2,"vegetation_or_structure")):
        positions = np.argwhere(occupancy == code)
        for i in np.linspace(0,max(0,len(positions)-1),min(128,len(positions)),dtype=int):
            y,z,x = positions[i]
            points[(int(x)+xmin,int(y)+ymin,int(z)+zmin)] = label
    roof = (occupancy[:-1] == 0) & (occupancy[1:] == 1)
    positions = np.argwhere(roof)
    for i in np.linspace(0,max(0,len(positions)-1),min(192,len(positions)),dtype=int):
        y,z,x = positions[i]
        points[(int(x)+xmin,int(y)+ymin,int(z)+zmin)] = "covered_air_and_roof"
        points[(int(x)+xmin,int(y)+ymin+1,int(z)+zmin)] = "covered_air_and_roof"
    while len(points) < samples:
        cx,cz = records[rng.randrange(len(records))]["chunk_xz"]
        points[(cx*16+rng.randrange(16),rng.randrange(ymin,ymax),cz*16+rng.randrange(16))] = "stratified_random"
    folder = dimensions(snapshot)[ir["dimension"]]/"region"
    reference, production, exact = {}, {}, {}
    mismatches, comparisons, strata, materials = [], [], Counter(), Counter()
    for (x,y,z), stratum in sorted(points.items()):
        cx,cz = x//16,z//16
        if (cx,cz) not in reference:
            reference[cx,cz] = reference_chunk(folder,cx,cz)
            region = Region(folder/f"r.{cx//32}.{cz//32}.mca")
            production[cx,cz] = region.chunk(cx,cz)
            with np.load(ir_dir/f"terrain_volume/c.{cx}.{cz}.npz",allow_pickle=False) as chunkfile:
                exact[cx,cz] = (chunkfile["section_y"].tolist(),chunkfile["block_id"],json.loads(str(chunkfile["palette_json"])))
        expected = reference_block(reference[cx,cz],x,y,z)
        actual = production[cx,cz].block(x,y,z)
        section_ys, array, palette = exact[cx,cz]
        retained = palette[int(array[section_ys.index(y//16),y%16,z%16,x%16])]
        passed = expected == actual == retained
        item = {"xyz":[x,y,z],"stratum":stratum,"reference":expected,"production":actual,"retained":retained,"match":passed}
        comparisons.append(item)
        if not passed:
            mismatches.append(item)
        strata[stratum] += 1
        materials[expected["Name"]] += 1
    report = {"schema_version":1,"kind":"IndependentSourceDecoderComparison","status":"pass" if not mismatches else "fail",
              "scope":ir["scope"],"sample_count":len(comparisons),"minimum_sample_count":1000,"unique_positions":len(points),
              "seed":seed,"chunk_count":len(reference),"strata":dict(strata),"material_sample_counts":dict(materials),
              "mismatch_count":len(mismatches),"mismatches":mismatches,
              "reference_parser":{"name":"nbtlib","version":nbtlib.__version__,"license":"MIT","source":"https://github.com/vberlier/nbtlib","packed_palette_method":"independent reversed binary-string slicing"},
              "production_parser":{"name":"isaacmin-java-nbt-anvil/1","packed_palette_method":"unsigned shift and mask with version-dependent padded storage"},
              "scope_limitations":["Decoder and retained-volume parity only; transformed scene fidelity is a later Q01 check", "Cave roofs included; explicit portal graph comparison remains not_run", "Absent sample strata are reported, not synthesized"],
              "source_ir_sha256":sha256(ir_dir/"world_ir.json"),"source_code_sha256":{name:sha256(Path(__file__).parent/name) for name in ("nbt.py","anvil.py","independent.py","world_ir.py","semantics.py")},
              "comparison_identity":canonical_hash(comparisons)}
    output = output or ir_dir/"independent_decode_comparison.json"
    write_json(output,report)
    details = output.with_name(output.stem+"_samples.json")
    write_json(details,comparisons)
    report["samples_file"] = details.name
    report["samples_sha256"] = sha256(details)
    write_json(output,report)
    return report


def compare_macro_surface(snapshot: Path, macro_dir: Path, samples: int = 1024, seed: int = 1729) -> dict:
    """Check exact coarse source ground against independent raw-save decoding."""
    if samples < 1000:
        raise SourceError("Macro source comparison requires at least1000 positions")
    snapshot,macro_dir=Path(snapshot),Path(macro_dir)
    with np.load(macro_dir/"macro_surface.npz",allow_pickle=False) as arrays:
        height,validity=arrays["height"],arrays["validity"]
        xmin,zmin=map(int,arrays["min_xz"])
        spacing=int(arrays["sample_spacing_m"])
        offset=int(float(arrays["sample_offset_m"])-0.5)
    cells=set()
    for z in np.linspace(0,height.shape[0]-1,32,dtype=int):
        for x in np.linspace(0,height.shape[1]-1,32,dtype=int):
            if validity[z,x]:
                cells.add((int(z),int(x)))
    rng=random.Random(seed)
    available=np.argwhere(validity)
    if len(available)<samples:
        raise SourceError("Insufficient valid unique macro samples for parity gate")
    while len(cells)<samples:
        z,x=available[rng.randrange(len(available))]
        cells.add((int(z),int(x)))
    root=dimensions(snapshot)["minecraft:overworld"]/"region"
    chunks={}
    comparisons=[]
    mismatches=[]
    for rz,rx in sorted(cells):
        x,z=xmin+rx*spacing+offset,zmin+rz*spacing+offset
        key=(x//16,z//16)
        if key not in chunks:
            chunks[key]=reference_chunk(root,*key)
        reference=chunks[key]
        expected=None
        unknown_above=[]
        for sy,section in sorted(reference[1].items(),reverse=True):
            states=section.get("block_states")
            palette=states["palette"] if states is not None else section["Palette"]
            classes={classify(str(p["Name"])) for p in palette}
            if not classes&(GROUND_CLASSES|{"unknown"}):
                continue
            for y in range(sy*16+15,sy*16-1,-1):
                block=reference_block(reference,x,y,z)
                semantic=classify(block["Name"])
                if semantic=="unknown":
                    unknown_above.append(block["Name"])
                if semantic in GROUND_CLASSES:
                    expected=y+1
                    break
            if expected is not None:
                break
        actual=float(height[rz,rx])
        item={"source_xz":[x,z],"reference_height":expected,"macro_height":actual,
              "unknown_above_ground":unknown_above,"match":actual==expected and not unknown_above}
        comparisons.append(item)
        if not item["match"]:
            mismatches.append(item)
    report={"schema_version":1,"kind":"IndependentMacroSourceComparison","status":"pass" if not mismatches else "fail",
            "samples":len(comparisons),"chunks":len(chunks),"mismatch_count":len(mismatches),"mismatches":mismatches,
            "seed":seed,"method":"nbtlib raw-save decode plus independent binary-string indexing and top-down natural-ground search",
            "scope_limitations":["validates coarse original-source samples, not refined terrain or drainage","semantic classification policy is shared and conservative"],
            "macro_surface_sha256":sha256(macro_dir/"macro_surface.npz"),"source_code_sha256":sha256(Path(__file__))}
    write_json(macro_dir/"independent_macro_samples.json",comparisons)
    report["samples_file"]="independent_macro_samples.json"
    report["samples_sha256"]=sha256(macro_dir/report["samples_file"])
    write_json(macro_dir/"independent_macro_comparison.json",report)
    return report


# Cache management below does not alter the independent reading algorithms.
def ensure_independent_comparisons(snapshot: Path, ir_dir: Path, macro_dir: Path) -> dict:
    """Reuse byte/current-reader-bound comparisons or archive and regenerate.

    The unchanged reading-code prefix is an accepted identity for reports made
    before this cache manager was appended. Changing any reading algorithm
    changes that identity; no historical digest is trusted unconditionally.
    Failed but current comparisons are retained as failures, never retried just
    to obtain a passing result. Source artifacts must themselves be current.
    """
    import hashlib
    import shutil
    from .snapshot import file_inventory
    from .provenance import reader_producer_identity
    from .macro import macro_producer_identity

    snapshot, ir_dir, macro_dir = map(Path, (snapshot, ir_dir, macro_dir))
    snapshot_hash = canonical_hash(file_inventory(snapshot))
    module_bytes = Path(__file__).read_bytes()
    marker = b"# Cache management below does not alter the independent reading algorithms."
    reading_prefix = module_bytes.split(marker, 1)[0].rstrip(b"\n") + b"\n"
    accepted_reader_hashes = {sha256(Path(__file__)), hashlib.sha256(reading_prefix).hexdigest()}

    def local_file(folder, name):
        path = folder / name
        if not isinstance(name, str) or Path(name).is_absolute() or ".." in Path(name).parts or path.is_symlink():
            raise SourceError("Unsafe comparison dependency path")
        if not path.resolve().is_relative_to(folder.resolve()):
            raise SourceError("Comparison dependency escapes artifact folder")
        return path

    def verify_artifact(folder, name, producer):
        manifest = json.loads((folder / name).read_text())
        if manifest.get("source_snapshot_sha256") != snapshot_hash:
            raise SourceError(f"{name}: snapshot identity mismatch")
        if manifest.get("producer") != producer:
            raise SourceError(f"{name}: source producer is stale; regenerate artifact first")
        for item in manifest["files"]:
            path = local_file(folder, item["path"])
            if not path.is_file() or path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
                raise SourceError(f"{name}: artifact dependency changed: {item['path']}")
        return manifest

    ir = verify_artifact(ir_dir, "world_ir.json", reader_producer_identity("world_ir"))
    verify_artifact(macro_dir, "macro_surface.json", macro_producer_identity())

    def current_report(folder, report_name, sample_name, kind):
        try:
            report = json.loads((folder / report_name).read_text())
            if report.get("samples_file") != sample_name:
                return False, "sample filename mismatch"
            sample_path = local_file(folder, sample_name)
            if sha256(sample_path) != report.get("samples_sha256"):
                return False, "sample bytes changed"
            rows = json.loads(sample_path.read_text())
            if not isinstance(rows, list) or len(rows) < 1000:
                return False, "fewer than 1000 recorded samples"
            if kind == "region":
                if report.get("source_ir_sha256") != sha256(ir_dir / "world_ir.json") or report.get("scope") != ir["scope"]:
                    return False, "WorldIR identity changed"
                code = report.get("source_code_sha256", {})
                for name in ("nbt.py", "anvil.py", "independent.py", "world_ir.py", "semantics.py"):
                    expected = accepted_reader_hashes if name == "independent.py" else {sha256(Path(__file__).parent / name)}
                    if code.get(name) not in expected:
                        return False, f"reader code changed: {name}"
                unique = len({tuple(row["xyz"]) for row in rows})
                actual_match = [row["reference"] == row["production"] == row["retained"] for row in rows]
                if report.get("comparison_identity") != canonical_hash(rows) or report.get("sample_count") != len(rows) or report.get("unique_positions") != unique:
                    return False, "sample identity or counts changed"
            else:
                if report.get("macro_surface_sha256") != sha256(macro_dir / "macro_surface.npz"):
                    return False, "macro bytes changed"
                if report.get("source_code_sha256") not in accepted_reader_hashes:
                    return False, "independent macro reader code changed"
                unique = len({tuple(row["source_xz"]) for row in rows})
                actual_match = [row["reference_height"] == row["macro_height"] and not row["unknown_above_ground"] for row in rows]
                if report.get("samples") != len(rows):
                    return False, "macro sample count changed"
            mismatch = [row for row, passed in zip(rows, actual_match) if not passed]
            if unique != len(rows) or any(row.get("match") != passed for row, passed in zip(rows, actual_match)):
                return False, "duplicate or contradictory raw samples"
            if report.get("mismatch_count") != len(mismatch) or report.get("mismatches") != mismatch or report.get("status") != ("fail" if mismatch else "pass"):
                return False, "comparison status contradicts samples"
            return True, report
        except (OSError, ValueError, KeyError, TypeError, AttributeError, SourceError) as exc:
            return False, f"unreadable or incomplete comparison: {type(exc).__name__}"

    def archive(folder, report_name, sample_name, reason):
        existing = [folder / name for name in (report_name, sample_name) if (folder / name).is_file()]
        if not existing:
            return None
        identity = canonical_hash({path.name: sha256(path) for path in existing})
        target = folder / "history" / "independent_comparisons" / Path(report_name).stem / identity
        target.mkdir(parents=True, exist_ok=True)
        for path in existing:
            archived = target / path.name
            if archived.exists() and sha256(archived) != sha256(path):
                raise SourceError("Comparison history identity collision")
            if not archived.exists():
                shutil.copy2(path, archived)
        if not (target / "history.json").exists():
            write_json(target / "history.json", {"reason": reason, "identity": identity,
                       "files": [{"path": path.name, "sha256": sha256(path)} for path in existing]})
        return str(target)

    result = {"kind": "IndependentComparisonCache", "source_snapshot_sha256": snapshot_hash, "comparisons": {}}
    for kind, folder, report_name, sample_name in (
        ("region", ir_dir, "independent_decode_comparison.json", "independent_decode_comparison_samples.json"),
        ("macro", macro_dir, "independent_macro_comparison.json", "independent_macro_samples.json"),
    ):
        valid, detail = current_report(folder, report_name, sample_name, kind)
        history = None
        if not valid:
            history = archive(folder, report_name, sample_name, detail)
            if kind == "region":
                compare_region(snapshot, ir_dir)
            else:
                compare_macro_surface(snapshot, macro_dir)
            valid, report = current_report(folder, report_name, sample_name, kind)
            if not valid:
                raise SourceError(f"Fresh independent {kind} comparison failed integrity validation: {report}")
        else:
            report = detail
        result["comparisons"][kind] = {"action": "reused" if history is None and valid and isinstance(detail, dict) else "refreshed",
            "status": report["status"], "report": str(folder / report_name), "report_sha256": sha256(folder / report_name),
            "history": history}
    result["status"] = "pass" if all(item["status"] == "pass" for item in result["comparisons"].values()) else "fail"
    return result
