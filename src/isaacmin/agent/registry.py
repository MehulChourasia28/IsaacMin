from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import jsonschema

from ..io import read_json
from ..security import safe_path


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    execute: Callable[[dict], dict]


class Registry:
    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        if tool.name in self.tools:
            raise ValueError("Tool name already registered")
        if tool.parameters.get("additionalProperties") is not False:
            raise ValueError("Runtime tool schemas must reject unregistered arguments")
        self.tools[tool.name] = tool

    def call(self, name: str, arguments: dict) -> dict:
        if name not in self.tools:
            raise ValueError("Unregistered tool")
        tool = self.tools[name]
        jsonschema.Draft202012Validator(tool.parameters).validate(arguments)
        return tool.execute(arguments)

    def schemas(self) -> list[dict]:
        return [{"type": "function", "function": {"name": tool.name,
                "description": tool.description, "parameters": tool.parameters}} for tool in self.tools.values()]


EMPTY = {"type": "object", "properties": {}, "additionalProperties": False}
BUILD_ID = {"type": "object", "properties": {"build_id": {"type": "string", "pattern": "^build_[a-f0-9]{16}$"}},
            "required": ["build_id"], "additionalProperties": False}


def production_registry(workspace: Path, world: Path, *, inventory=None, source_hash=None) -> Registry:
    from ..pipeline import build
    from ..validation.runner import validate_build
    registry = Registry()

    def source_context():
        nonlocal inventory, source_hash
        if inventory is None or source_hash is None:
            from ..pipeline import _source
            from ..project import resolve_project
            snapshot, inventory = _source(workspace, resolve_project(workspace, world))
            source_hash = snapshot["save_sha256"]
        return inventory, source_hash

    def check_build(build_id):
        _, expected = source_context()
        manifest = read_json(safe_path(workspace / "worlds", build_id) / "build.json")
        if manifest.get("dependencies", {}).get("source") != expected:
            raise ValueError("Build belongs to a different source snapshot than this agent run")

    def inspect(_):
        current, expected = source_context()
        return {"data_version": current["data_version"], "source_snapshot_sha256": expected, "dimensions": [
                {"id": d["id"], "full_chunk_count": d["full_chunk_count"], "status_counts": d["status_counts"]}
                for d in current["dimensions"]],
                "source_read_only": True, "unknown_blocks": current.get("unknown_blocks", [])}

    def inspect_build(args):
        check_build(args["build_id"])
        return read_json(safe_path(workspace / "worlds", args["build_id"]) / "qualification.json")

    def validate(args):
        check_build(args["build_id"])
        return validate_build(workspace, args["build_id"])

    registry.register(Tool("inspect_world", "Read the immutable source coverage and decoder status.", EMPTY, inspect))
    def inspect_defects(_):
        from ..jobs.repair_budget import exhausted_repairs
        _, source = source_context()
        return {"source_snapshot_sha256": source, "exhausted_repairs": exhausted_repairs(workspace, source),
                "budget_reset_authority": False}
    registry.register(Tool("inspect_known_defects", "Read measured exhausted repair budgets for this immutable save; cannot reset them.",
                           EMPTY, inspect_defects))
    registry.register(Tool("build_region", "Resume the deterministic strict source-to-Isaac region build. Never lowers quality.", EMPTY,
                           lambda _: build(workspace, world)))
    registry.register(Tool("inspect_build", "Read frozen gate results and unresolved defects.", BUILD_ID, inspect_build))
    registry.register(Tool("validate_build", "Run the independent fixed validators on an exported build.", BUILD_ID,
                           validate))
    return registry
