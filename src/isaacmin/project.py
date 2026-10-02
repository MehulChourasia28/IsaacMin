from __future__ import annotations

from pathlib import Path

from .contracts import CoordinateFrame
from .io import atomic_json, read_json


def resolve_project(workspace: Path, world: Path | None = None) -> dict:
    path = workspace / "state/resolved_project.json"
    existing = read_json(path) if path.exists() else {}
    world = world or Path(existing.get("source_world", workspace / "IsaacMin-MinecraftWorld"))
    if existing.get('source_world') and Path(existing['source_world']).resolve()==world.resolve():
        center=tuple(existing.get('centre_minecraft_xz',(-1065.38,688.07)))
        center_provenance=existing.get('centre_provenance','saved user preference for this save')
    elif (world/'level.dat').is_file():
        from .source.nbt import load
        data=load(world/'level.dat');data=data.get('Data',data)
        spawn=data.get('spawn',{}).get('pos',[data.get('SpawnX',0),data.get('SpawnY',0),data.get('SpawnZ',0)])
        center=(float(spawn[0]),float(spawn[2]));center_provenance='level.dat spawn for this save'
    else:
        center=(-1065.38,688.07);center_provenance='configured initial user save centre'
    project = {"schema_version": "1.0", "project": "IsaacMin", "source_world": str(world.resolve()),
               "source_dimension": "minecraft:overworld", "quality": "strict",
               "centre_minecraft_xz": list(center),
               "centre_provenance": center_provenance,
               "coordinate_frame": CoordinateFrame((center[0], 0., center[1])).record(),
               "initial_integration_extent_m": 128, "target_extent_m": [2048, 2048],
               "source_coverage_policy": "complete generated chunks only; unknown remains excluded",
               "camera": {"height_above_ground_m": .6, "resolution_px": [1280, 720],
                          "horizontal_fov_degrees": 90, "design_hz": 30},
               "navigation_stack": "not_run_not_supplied", "optional_human_review": "not_requested",
               "heavy_workers": 1, "shared_memory_reserve_gib": 24,
               "planner_model": "nvidia/nemotron-3-ultra-550b-a55b",
               "vision_model": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
               "model_substitution": False}
    for name in ('terrain_mode','scope_override'):
        if name in existing:project[name]=existing[name]
    atomic_json(path, project)
    return project
