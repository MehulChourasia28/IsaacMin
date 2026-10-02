"""Crash/replay properties of the controller; no provider execution is claimed."""
from pathlib import Path

import pytest

from isaacmin.agent.registry import Registry, Tool, EMPTY
from isaacmin.io import atomic_json, read_json


def test_interrupted_bounded_call_resumes_without_replanning_or_resetting_budget(tmp_path, monkeypatch):
    from isaacmin.agent import runtime
    from isaacmin import pipeline
    from isaacmin.adapters import nim
    source = tmp_path / "source"; source.mkdir()
    monkeypatch.setattr(runtime, "resolve_project", lambda *args: {"source_world": str(source)})
    monkeypatch.setattr(pipeline, "_source", lambda *args: ({"save_sha256": "fixture-source"}, {}))
    atomic_json(tmp_path / "state/nim_capabilities.json", {"probes": {"native_tools": {"status": "pass"}}})
    invocations, calls = [], []
    completed = tmp_path / "durable-result.json"

    def action(args):
        invocations.append(args)
        if not completed.exists():
            atomic_json(completed, {"status": "synthetic_internal_controller_test"})
            raise KeyboardInterrupt()
        return read_json(completed)

    registry = Registry()
    registry.register(Tool("fixture_operation", "Synthetic control-flow boundary", EMPTY, action))
    monkeypatch.setattr(runtime, "production_registry", lambda *args, **kwargs: registry)

    class Client:
        def __init__(self, *args): pass

        def chat(self, model, messages, **kwargs):
            calls.append(len(messages))
            if len(calls) == 1:
                return {"message": {"content": None, "tool_calls": [{"id": "fixture-call", "type": "function",
                    "function": {"name": "fixture_operation", "arguments": "{}"}}]}}
            assert messages[-1]["role"] == "tool"
            assert messages[-1]["tool_call_id"] == "fixture-call"
            return {"message": {"content": "Fixture stopped; no external or world qualification."}}

    monkeypatch.setattr(nim, "NIMClient", Client)
    with pytest.raises(KeyboardInterrupt):
        runtime.run(tmp_path, source)
    checkpoint = next((tmp_path / "state/agent_runs").glob("*.json"))
    saved = read_json(checkpoint)
    assert saved["active_call_id"] == "fixture-call"
    assert len(saved["pending_calls"]) == 1
    assert saved["planning_calls"] == 1
    result = runtime.run(tmp_path, source, resume_run_id=saved["run_id"])
    assert result["status"] == "incomplete"
    assert len(invocations) == 2
    assert result["planning_calls"] == 2
    assert list(result["attempts"].values()) == [1]
    assert not result["pending_calls"]


def test_registry_refuses_a_success_record_from_another_source(tmp_path):
    from isaacmin.agent.registry import production_registry
    build_id = "build_0123456789abcdef"
    atomic_json(tmp_path / "worlds" / build_id / "build.json", {"dependencies": {"source": "older-source"}})
    registry = production_registry(tmp_path, tmp_path / "source", inventory={}, source_hash="current-source")
    with pytest.raises(ValueError, match="different source snapshot"):
        registry.call("inspect_build", {"build_id": build_id})
