"""Opt-in stable-source watching. Does not mutate the currently evaluated world."""
from __future__ import annotations

import time
from pathlib import Path

from ..io import atomic_json, hash_object, utc_now
from ..jobs import Journal
from ..source.snapshot import snapshot_world


def signature(world: Path) -> str:
    return hash_object([(str(p.relative_to(world)), p.stat().st_size, p.stat().st_mtime_ns)
                        for p in sorted(world.rglob("*")) if p.is_file() and p.name != "session.lock"])


def watch_source(workspace: Path, world: Path, *, interval=10, debounce=30, stop=None):
    previous = signature(world)
    changed_at = None
    while stop is None or not stop.is_set():
        if stop is None:
            time.sleep(interval)
        elif stop.wait(interval):
            break
        current = signature(world)
        if current != previous:
            previous, changed_at = current, time.monotonic()
        elif changed_at is not None and time.monotonic()-changed_at >= debounce:
            snapshot = snapshot_world(world, workspace / "work/snapshots")
            journal = Journal(workspace / "state/journal.sqlite")
            invalidated = journal.invalidate("source", snapshot["save_sha256"])
            request = {"kind": "source_changed", "snapshot": snapshot["snapshot_path"],
                       "source_hash": snapshot["save_sha256"], "invalidated_jobs": invalidated,
                       "state": "queued_candidate", "created_at_utc": utc_now(),
                       "frozen_evaluation_world_modified": False}
            atomic_json(workspace / "state/requests" / (snapshot["save_sha256"] + ".json"), request)
            journal.close()
            from ..pipeline import build
            # A candidate build never replaces a qualified/evaluation artifact in place.
            request["state"] = "running_candidate"
            request_path = workspace / "state/requests" / (snapshot["save_sha256"] + ".json")
            atomic_json(request_path, request)
            try:
                request["result"] = build(workspace, world)
                request["state"] = request["result"]["status"]
            except Exception as exc:
                from ..security import redact
                request["state"] = "failed"
                request["reason"] = redact(str(exc))
            atomic_json(request_path, request)
            changed_at = None
