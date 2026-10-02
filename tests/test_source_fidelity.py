"""Actual parser/ray execution on declared synthetic source fixtures."""
import gzip
import json

import nbtlib
import numpy as np
import pytest
from nbtlib import Byte, Compound, Int, List, LongArray, String

from isaacmin.contracts.coordinates import CoordinateFrame
from isaacmin.io import hash_object
from isaacmin.source.world_ir import extract_world_ir
from isaacmin.source.independent import compare_region
from isaacmin.volumes.topology import source_topology
from isaacmin.volumes.mesh_validation import validate_source_mesh, validate_source_connectivity
from isaacmin.validation.profile import freeze_profile
from isaacmin.validation.source_fidelity import freeze_source_fidelity_profile, evaluate_source_fidelity
from test_source import make_region, nbt_bytes
from test_source_connectivity import voxel_boundary_mesh


def real_reader_fixture(tmp_path):
    world = tmp_path / "fixture_save"
    world.mkdir()
    (world / "level.dat").write_bytes(gzip.compress(nbt_bytes({"Data": Compound({"DataVersion": Int(4786), "spawn": Compound({"pos": nbtlib.IntArray([0, 16, 0])})})})))
    solid = np.zeros((32, 32, 32), dtype=bool)
    solid[:16] = True
    solid[10:15, 10:22, 5:27] = False
    solid[10:16, 10:14, 5:9] = False
    for cx in (-1, 0):
        for cz in (-1, 0):
            sections = []
            for sy in (0, 1):
                values = solid[sy*16:sy*16+16, (cz+1)*16:(cz+2)*16, (cx+1)*16:(cx+2)*16]
                packed = [sum(int(v) << (i*4) for i, v in enumerate(group)) for group in values.ravel().reshape(-1, 16)]
                sections.append(Compound({"Y": Byte(sy), "block_states": Compound({"palette": List[Compound]([
                    Compound({"Name": String("minecraft:air")}), Compound({"Name": String("minecraft:stone")})]), "data": LongArray(packed)}),
                    "biomes": Compound({"palette": List[String]([String("minecraft:plains")])})}))
            raw = nbt_bytes({"DataVersion": Int(4786), "xPos": Int(cx), "zPos": Int(cz), "Status": String("minecraft:full"), "sections": List[Compound](sections)})
            make_region(world / f"region/r.{cx//32}.{cz//32}.mca", raw, cx, cz)
    ir = tmp_path / "ir"
    extract_world_ir(world, ir, center=(0, 0), extent=32)
    compare_region(world, ir, samples=1024)
    source_topology(ir)
    with np.load(ir / "terrain_surface.npz", allow_pickle=False) as data:
        np.savez_compressed(tmp_path / "exterior_delta.npz", source_height=data["height"],
                            delta=np.zeros((32, 32)), protection=data["cave_column_protection"])
    mesh = voxel_boundary_mesh(solid)
    mesh.apply_translation((-16, 0, -16))
    frame = CoordinateFrame()
    quality = freeze_profile(tmp_path / "quality_profile.json", {"coordinates": frame.record()})
    freeze_source_fidelity_profile(quality, tmp_path / "source_fidelity_profile.json")
    return ir, mesh, frame, quality


def test_frozen_geometry_budgets_reject_thinning_and_excess_exterior_change(tmp_path):
    ir, original, frame, quality = real_reader_fixture(tmp_path)
    results = {}
    for name in ("exact", "roof_thinned", "exterior_raised"):
        mesh = original.copy()
        if name == "roof_thinned":
            # Raise only the cave underside by10mm. All source voxel centres
            # retain their signs; this must still fail the protected roof budget.
            selected = np.isclose(mesh.vertices[:, 1], 15)
            mesh.vertices[selected, 1] += .01
        elif name == "exterior_raised":
            selected = np.isclose(mesh.vertices[:, 1], 16) & (mesh.vertices[:, 0] < -13)
            mesh.vertices[selected, 1] += .6
        mesh.apply_transform(frame.record()["source_to_world"])
        path = tmp_path / (name + ".obj")
        mesh.export(path)
        target = tmp_path / name
        sampled = validate_source_mesh(ir, path, target / "sampled", frame.record()["world_to_source"])
        graph = validate_source_connectivity(ir, path, target / "graph", frame.record()["world_to_source"])
        assert sampled["status"] == graph["status"] == "pass"
        from isaacmin.jobs.stage_checkpoint import run_stage
        from isaacmin.validation.worker import measurement_completed
        results[name] = run_stage(target,'source_fidelity',{'variant':name},['fidelity'],
            lambda:evaluate_source_fidelity(ir, path, target / "fidelity",
                frozen_budget_path=tmp_path / "source_fidelity_profile.json", exterior_delta_path=tmp_path / "exterior_delta.npz",
                topology_samples_path=target / "sampled/source_mesh_validation.json",
                topology_graph_path=target / "graph/mesh_connectivity_comparison.json"),
            accepted=measurement_completed)
        def must_not_rerun():
            raise AssertionError('An unchanged completed measurement was recomputed')
        reused=run_stage(target,'source_fidelity',{'variant':name},['fidelity'],must_not_rerun,
                         accepted=measurement_completed)
        assert reused==results[name]  # Includes the exact nonpassing roof/exterior verdicts.
    assert results["exact"]["status"] == "measurements_pass"
    assert results["exact"]["full_Q03_status"] == "not_run"
    assert results["exact"]["decoder"]["independent_portal_source_positions"] > 0
    assert results["roof_thinned"]["caves"]["roof_budget_failure_indices"]
    assert results["roof_thinned"]["protected_cave_geometry_status"] == "fail"
    assert results["exterior_raised"]["source_geometry_status"] == "fail"
    changed = json.loads(json.dumps(quality))
    changed["profile"]["thresholds"]["seam_gap_max_m"] = .005
    changed["sha256"] = hash_object(changed["profile"])
    with pytest.raises(ValueError, match="looser allowance"):
        freeze_source_fidelity_profile(changed, tmp_path / "looser_budget.json")
