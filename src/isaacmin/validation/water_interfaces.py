"""Compare fluid triangles with independently decoded Java source interfaces."""
from __future__ import annotations

import json
import os
import tempfile
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np

from ..io import atomic_json, sha256_file
from ..source.anvil import dimensions
from ..source.independent import reference_chunk
from ..source.snapshot import canonical_hash, file_inventory
from .source_water import AIR_NAMES, _fluid, _section_indices
from . import source_water


# Full-cube context occurring in the supplied source. Unsupported shapes fail
# explicitly; no universal block-render-shape interpretation is asserted.
FULL_CUBE = {"stone", "granite", "diorite", "andesite", "deepslate", "tuff", "dirt",
             "grass_block", "coarse_dirt", "rooted_dirt", "gravel", "sand", "clay",
             "bedrock", "cobblestone", "mossy_cobblestone", "calcite", "dripstone_block",
             "coal_ore", "iron_ore", "copper_ore", "gold_ore", "redstone_ore", "lapis_ore",
             "diamond_ore", "emerald_ore", "deepslate_coal_ore", "deepslate_iron_ore",
             "deepslate_copper_ore", "deepslate_gold_ore", "deepslate_redstone_ore",
             "deepslate_lapis_ore", "deepslate_diamond_ore", "deepslate_emerald_ore"}


def ensure_source_water_audit(snapshot, ir_dir):
    """Return current bound inventory directory; preserve stale evidence bytes.

    A successful report for another input/revision is never overwritten. Fresh
    results use content-specific directories and an atomic staging rename.
    """
    from ..security import safe_path
    snapshot, ir_dir = Path(snapshot), Path(ir_dir)
    meta = json.loads((ir_dir/"world_ir.json").read_text())
    snapshot_hash = canonical_hash(file_inventory(snapshot))
    if snapshot_hash != meta["source_snapshot_sha256"]:
        raise ValueError("Source water cache snapshot differs from WorldIR")
    for entry in meta["files"]:
        if sha256_file(safe_path(ir_dir,entry["path"],must_exist=True)) != entry["sha256"]:
            raise ValueError("Source water cache WorldIR payload changed")
    expected = {"source_snapshot_sha256": snapshot_hash, "source_ir_sha256": sha256_file(ir_dir/"world_ir.json"),
        "validator_sha256": sha256_file(Path(source_water.__file__)),
        "independent_nbt_reader_sha256": sha256_file(Path(__file__).parents[1]/"source/independent.py")}

    def valid(directory):
        try:
            report=json.loads((directory/"source_water_inventory.json").read_text())
            if any(report.get(k)!=v for k,v in expected.items()):return False
            files={entry["path"]:entry for entry in report["files"]}
            if "source_water_samples.npz" not in files:return False
            for name,entry in files.items():
                path=safe_path(directory,name,must_exist=True)
                if sha256_file(path)!=entry["sha256"] or path.stat().st_size!=entry["bytes"]:return False
            reference=report.get("semantic_reference")
            if reference and sha256_file(Path(reference["path"]))!=reference["sha256"]:return False
            return True
        except (OSError,ValueError,KeyError,TypeError):
            return False

    legacy=ir_dir.parent/"water_inventory_verified"
    if valid(legacy):return legacy
    root=ir_dir.parent/"water_audits";root.mkdir(parents=True,exist_ok=True)
    destination=root/canonical_hash(expected)
    if valid(destination):return destination
    if destination.exists():
        history=root/"history";history.mkdir(exist_ok=True)
        archive=history/(destination.name+"-"+canonical_hash(file_inventory(destination)))
        if archive.exists():
            archive=Path(tempfile.mkdtemp(prefix=archive.name+"-",dir=history))/"audit"
        os.replace(destination,archive)
    staging=Path(tempfile.mkdtemp(prefix=".audit-",dir=root))
    source_water.audit_source_water(snapshot,ir_dir,staging)
    if not valid(staging):raise ValueError("Generated water inventory failed provenance verification")
    os.replace(staging,destination)
    return destination


def validate_water_interfaces(snapshot, ir_dir, water_mesh_path, output, *, origin=(-1065.38, 0., 688.07)):
    """Inspect every output face and exact raw-source neighbor/corner state.

    Does not call the water generator, its retained-state reader, or its corner
    routine. Rational arithmetic follows the inspected Java26.1 renderer.
    """
    snapshot, ir_dir, water_mesh_path, output = map(Path, (snapshot, ir_dir, water_mesh_path, output))
    meta = json.loads((ir_dir/"world_ir.json").read_text())
    water = json.loads(water_mesh_path.read_text())
    before = canonical_hash(file_inventory(snapshot))
    if before != meta["source_snapshot_sha256"] or before != water["source_snapshot_sha256"]:
        raise ValueError("Independent water validation snapshot mismatch")
    if water["source_ir_sha256"] != sha256_file(ir_dir/"world_ir.json"):
        raise ValueError("Fluid mesh references a different source IR")
    folder = dimensions(snapshot)[meta["dimension"]]/"region"
    chunks, sections, raw_states = {}, {}, {}

    def state(position):
        position = tuple(map(int, position))
        if position in raw_states:
            return raw_states[position]
        x, y, z = position; chunk_key = (x//16, z//16)
        if chunk_key not in chunks:
            version, chunks[chunk_key] = reference_chunk(folder, *chunk_key)
            if version != 4786:
                raise ValueError("Independent fluid shapes are verified for DataVersion4786 only")
        key = (*chunk_key, y//16)
        if key not in sections:
            section = chunks[chunk_key].get(y//16)
            if section is None:
                raise ValueError(f"Missing raw section at required fluid context {position}")
            sections[key] = _section_indices(section)
        palette, indices = sections[key]
        value = palette[int(indices[y%16, z%16, x%16])]
        raw_states[position] = value
        return value

    def category(position):
        value = state(position)
        if _fluid(value)[0]:
            if value.get("Properties", {}).get("waterlogged") == "true" and value["Name"] != "minecraft:glow_lichen":
                raise ValueError("Unverified waterlogged block shape")
            return "water"
        if value["Name"] in AIR_NAMES or value["Name"] in {"minecraft:glow_lichen", "minecraft:lava"}:
            return "nonoccluding"
        if value["Name"].removeprefix("minecraft:") in FULL_CUBE:
            return "solid"
        raise ValueError("Unsupported independent source fluid shape: "+value["Name"])

    def height(p):
        cat = category(p)
        if cat == "solid":
            return Fraction(-1)
        if cat != "water":
            return Fraction(0)
        above = (p[0], p[1]+1, p[2])
        return Fraction(1) if category(above) == "water" else Fraction(_fluid(state(p))[2], 9)

    def corner(p, dx, dz):
        x, y, z = p
        values = [height(p), height((x+dx,y,z)), height((x,y,z+dz))]
        if max(values) >= 1:
            return 1.
        if values[1] > 0 or values[2] > 0:
            values.append(height((x+dx,y,z+dz)))
        if max(values) >= 1:
            return 1.
        weighted = [(v, 10 if v >= Fraction(4,5) else 1) for v in values if v >= 0]
        return float(sum((v*w for v,w in weighted),Fraction(0))/sum(w for _,w in weighted))

    # Fluid cell ownership is checked against the prior complete raw-save audit,
    # then every neighboring state is freshly read through independent NBT bits.
    audit_dir = ensure_source_water_audit(snapshot, ir_dir)
    audit = json.loads((audit_dir/"source_water_inventory.json").read_text())
    sample_file = audit_dir/"source_water_samples.npz"
    entries = {entry["path"]:entry for entry in audit["files"]}
    if audit["status"] != "pass" or sha256_file(sample_file) != entries[sample_file.name]["sha256"]:
        raise ValueError("Complete independent fluid inventory must pass and retain exact sample bytes")
    with np.load(sample_file, allow_pickle=False) as data:
        cells = {tuple(p) for p in data["water_source_xyz"].tolist()}
    observed_cells = {tuple(p) for p in water["source_water_cells_xyz"]}
    if observed_cells != cells:
        raise ValueError("Fluid mesh cell ownership differs from independent complete source inventory")
    expected = {}
    offsets = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]
    for p in sorted(cells):
        x,y,z=p
        visible=[]
        for direction in offsets:
            neighbor=tuple(a+b for a,b in zip(p,direction))
            # Water-vs-lava surfaces are a separate ownership policy; do not
            # mistake a different fluid for an exposed air/lichen interface.
            value=state(neighbor)
            if value["Name"] in AIR_NAMES or (value["Name"]=="minecraft:glow_lichen" and not _fluid(value)[0]):
                visible.append(direction)
        if not visible:continue
        corners={(dx,dz):y+corner(p,dx,dz) for dx in (-1,1) for dz in (-1,1)}
        for direction in visible:
            dx,dy,dz=direction
            if dy==1:
                points=[[x+(sx+1)//2,corners[sx,sz],z+(sz+1)//2] for sx,sz in [(-1,-1),(-1,1),(1,1),(1,-1)]]
            elif dy==-1:
                points=[[x,y,z],[x+1,y,z],[x+1,y,z+1],[x,y,z+1]]
            elif dx:
                points=[[x+(dx+1)//2,corners[dx,sz],z+(sz+1)//2] for sz in (-1,1)]
                points += [[point[0],y,point[2]] for point in points[::-1]]
            else:
                points=[[x+(sx+1)//2,corners[sx,dz],z+(dz+1)//2] for sx in (-1,1)]
                points += [[point[0],y,point[2]] for point in points[::-1]]
            expected[(p,direction)]=np.asarray(points,float)
    vertices=np.asarray(water["vertices"],float)
    # Explicit caller origin, recorded below; no pose or producer metadata is
    # used to infer which source point an output vertex was intended to match.
    vertices=vertices[:,[0,2,1]]*np.asarray([1,1,-1])+np.asarray(origin)
    triangles=np.asarray(water["triangles"],int)
    observed, errors, deviations = {}, [], []
    for i in range(0,len(triangles),2):
        p=tuple(water["triangle_source_cell_xyz"][i]);kind=water["triangle_surface_kind"][i]
        if i+1>=len(triangles) or tuple(water["triangle_source_cell_xyz"][i+1])!=p or water["triangle_surface_kind"][i+1]!=kind:
            raise ValueError("Fluid face provenance is not a complete paired quad")
        pair=vertices[triangles[i:i+2]]
        normals=np.cross(pair[:,1]-pair[:,0],pair[:,2]-pair[:,0])
        if kind in ("top","bottom"):
            direction=(0,1 if kind=="top" else -1,0)
        else:
            horizontal=normals.sum(axis=0);horizontal[1]=0
            axis=int(np.argmax(np.abs(horizontal)))
            direction=tuple(int(np.sign(horizontal[axis])) if j==axis else 0 for j in range(3))
        key=(p,direction)
        if key in observed:errors.append({"kind":"duplicate_face","cell":p,"direction":direction})
        observed[key]=i
        if key not in expected:
            errors.append({"kind":"unexpected_face","cell":p,"direction":direction});continue
        points=np.unique(pair.reshape(-1,3),axis=0);target=expected[key]
        deviation=float(np.linalg.norm(points[:,None]-target[None],axis=2).min(axis=1).max()) if len(points) else float("inf")
        missing_corner=float(np.linalg.norm(target[:,None]-points[None],axis=2).min(axis=1).max())
        deviation=max(deviation,missing_corner);deviations.append(deviation)
        if len(points)!=4 or deviation>1e-9 or np.any(normals@np.asarray(direction)<=0):
            errors.append({"kind":"corner_or_winding_mismatch","cell":p,"direction":direction,"maximum_corner_error_m":deviation})
    missing=sorted(set(expected)-set(observed))
    after=canonical_hash(file_inventory(snapshot))
    if after!=before:raise ValueError("Independent water snapshot changed during validation")
    output.mkdir(parents=True,exist_ok=True)
    palette=sorted({json.dumps(v,sort_keys=True,separators=(",",":")) for v in raw_states.values()})
    palette_ids={v:i for i,v in enumerate(palette)}
    positions=sorted(raw_states)
    np.savez_compressed(output/"raw_neighbor_states.npz",source_xyz=np.asarray(positions),
                        state_id=np.asarray([palette_ids[json.dumps(raw_states[p],sort_keys=True,separators=(",",":"))] for p in positions]),
                        palette_json=np.asarray(json.dumps([json.loads(v) for v in palette])))
    report={"kind":"IndependentFluidInterfaceComparison","status":"pass" if not errors and not missing else "fail",
            "source_snapshot_sha256":before,"source_ir_sha256":sha256_file(ir_dir/"world_ir.json"),
            "water_mesh_sha256":sha256_file(water_mesh_path),"validator_sha256":sha256_file(Path(__file__)),
            "source_reader_sha256":sha256_file(Path(__file__).with_name("source_water.py")),
            "inventory_sha256":sha256_file(audit_dir/"source_water_inventory.json"),"origin_source_xyz":list(origin),
            "source_fluid_cells":len(cells),"raw_neighbor_state_positions":len(raw_states),
            "expected_face_counts":dict(Counter("top" if d[1]==1 else "bottom" if d[1]==-1 else "side" for _,d in expected)),
            "observed_quads":len(observed),"maximum_corner_error_m":max(deviations,default=0.),
            "missing_faces":[{"source_xyz":p,"direction":d} for p,d in missing],"errors":errors,
            "shape_scope":"Java26.1 verified full-cube bank blocks, air, glow_lichen, water/bubble/waterlogged_lichen; unsupported shape raises",
            "qualification_scope":"Source surface ownership, side exposure, exact corners and outward triangle winding; target bank intersections/material rendering pending",
            "files":[{"path":"raw_neighbor_states.npz","sha256":sha256_file(output/"raw_neighbor_states.npz")} ]}
    atomic_json(output/"water_interface_comparison.json",report)
    return report
