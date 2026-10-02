"""SQLite work journal: identity, leases, retry, provenance and invalidation.

A completed operation is reusable only with the same fingerprint and verified
output files. An expired worker cannot publish after another worker takes over.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from ..io import canonical_json, hash_object
from ..security import redact


class JobConflict(RuntimeError):
    pass


class LeaseLost(RuntimeError):
    pass


STATES = {"queued", "running", "waiting_provider", "waiting_resource", "blocked",
          "failed", "cancelled", "succeeded"}


class Journal:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=30, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY, run_id TEXT NOT NULL, operation_id TEXT UNIQUE NOT NULL,
                tool TEXT NOT NULL, fingerprint TEXT NOT NULL, input_json TEXT NOT NULL,
                state TEXT NOT NULL, owner TEXT, lease_until REAL, attempts INTEGER NOT NULL DEFAULT 0,
                checkpoint_json TEXT, result_json TEXT, error_json TEXT,
                created REAL NOT NULL, updated REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS dependencies (
                job_id TEXT NOT NULL REFERENCES jobs(id), dependency_key TEXT NOT NULL,
                dependency_hash TEXT NOT NULL, PRIMARY KEY(job_id,dependency_key));
            CREATE TABLE IF NOT EXISTS events (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
                created REAL NOT NULL, kind TEXT NOT NULL, payload_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS requests (
                id TEXT PRIMARY KEY, request_json TEXT NOT NULL, state TEXT NOT NULL, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS memory (
                key TEXT PRIMARY KEY, context_hash TEXT NOT NULL, value_json TEXT NOT NULL, updated REAL NOT NULL);
        """)

    def close(self):
        self.db.close()

    @contextmanager
    def transaction(self):
        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            self.db.execute("COMMIT")
        except BaseException:
            self.db.execute("ROLLBACK")
            raise

    def event(self, job_id: str, kind: str, payload: dict):
        self.db.execute("INSERT INTO events(job_id,created,kind,payload_json) VALUES (?,?,?,?)",
                        (job_id, time.time(), kind, canonical_json(redact(payload)).decode()))

    def submit(self, run_id: str, operation_id: str, tool: str, inputs: dict,
               dependencies: dict[str, str]) -> dict:
        clean = redact(inputs)
        if clean != inputs:
            raise JobConflict("Credentials must not be journal inputs")
        fingerprint = hash_object({"tool": tool, "inputs": inputs, "dependencies": dependencies})
        now = time.time()
        with self.transaction():
            old = self.db.execute("SELECT * FROM jobs WHERE operation_id=?", (operation_id,)).fetchone()
            if old:
                if old["fingerprint"] != fingerprint:
                    raise JobConflict("Operation ID reused with different inputs or dependencies")
                return self._decode(old)
            job_id = uuid.uuid4().hex
            self.db.execute("""INSERT INTO jobs(id,run_id,operation_id,tool,fingerprint,input_json,
                               state,created,updated) VALUES (?,?,?,?,?,?,?,?,?)""",
                            (job_id, run_id, operation_id, tool, fingerprint,
                             canonical_json(inputs).decode(), "queued", now, now))
            self.db.executemany("INSERT INTO dependencies VALUES (?,?,?)",
                                [(job_id, key, value) for key, value in dependencies.items()])
            self.event(job_id, "submitted", {"fingerprint": fingerprint})
        return self.get(job_id)

    def claim(self, job_id: str, *, lease_seconds: float = 120) -> str | None:
        if not 1 <= lease_seconds <= 86400:
            raise ValueError("Invalid lease duration")
        token = uuid.uuid4().hex
        now = time.time()
        with self.transaction():
            job = self.get(job_id)
            if job["state"] == "succeeded" or job["state"] == "cancelled":
                return None
            if job["state"] == "running" and job["lease_until"] > now:
                return None
            self.db.execute("""UPDATE jobs SET state='running',owner=?,lease_until=?,
                               attempts=attempts+1,updated=? WHERE id=?""",
                            (token, now + lease_seconds, now, job_id))
            self.event(job_id, "claimed", {"lease_seconds": lease_seconds})
        return token

    def heartbeat(self, job_id: str, owner: str, checkpoint: dict | None = None,
                  *, lease_seconds: float = 120):
        now = time.time()
        with self.transaction():
            self._require_owner(job_id, owner)
            self.db.execute("UPDATE jobs SET lease_until=?,updated=?,checkpoint_json=? WHERE id=?",
                            (now + lease_seconds, now,
                             canonical_json(redact(checkpoint or {})).decode(), job_id))

    def _require_owner(self, job_id: str, owner: str):
        job = self.get(job_id)
        if job["state"] != "running" or job["owner"] != owner or job["lease_until"] <= time.time():
            raise LeaseLost("Worker has no current lease")

    def finish(self, job_id: str, owner: str, *, state: str, result: dict | None = None,
               error: dict | None = None):
        if state not in STATES - {"running", "queued"}:
            raise ValueError("Invalid final job state")
        if state == "succeeded" and result is None:
            raise ValueError("Success requires a concrete result")
        with self.transaction():
            self._require_owner(job_id, owner)
            self.db.execute("""UPDATE jobs SET state=?,owner=NULL,lease_until=NULL,result_json=?,
                               error_json=?,updated=? WHERE id=?""",
                            (state, canonical_json(redact(result)).decode(),
                             canonical_json(redact(error)).decode(), time.time(), job_id))
            self.event(job_id, state, {"error": error})

    def publish(self, job_id: str, owner: str, result: dict, publish):
        """Hold the lease-check transaction through atomic filesystem publication.

        A crash after rename but before commit leaves a complete validated artifact
        that a new owner may explicitly revalidate and adopt. It is never inferred
        successful merely because files exist.
        """
        with self.transaction():
            self._require_owner(job_id, owner)
            publish()
            self.db.execute("""UPDATE jobs SET state='succeeded',owner=NULL,lease_until=NULL,
                               result_json=?,error_json=NULL,updated=? WHERE id=?""",
                            (canonical_json(redact(result)).decode(), time.time(), job_id))
            self.event(job_id, "succeeded", {"artifact": result.get("artifact")})

    def recover(self) -> list[str]:
        """Expired leases requeue; partial output is never inferred to be success."""
        now = time.time()
        with self.transaction():
            rows = self.db.execute("SELECT id FROM jobs WHERE state='running' AND lease_until<=?", (now,)).fetchall()
            for row in rows:
                self.db.execute("UPDATE jobs SET state='queued',owner=NULL,lease_until=NULL,updated=? WHERE id=?",
                                (now, row["id"]))
                self.event(row["id"], "recovered", {"reason": "expired_lease"})
        return [r["id"] for r in rows]

    def invalidate(self, key: str, new_hash: str) -> list[str]:
        """Dependencies include nonlocal catchments, cave components and renderer."""
        with self.transaction():
            rows = self.db.execute("""SELECT j.id FROM jobs j JOIN dependencies d ON j.id=d.job_id
                                      WHERE d.dependency_key=? AND d.dependency_hash!=?""", (key, new_hash)).fetchall()
            for row in rows:
                # Old artifact remains immutable; the job no longer advertises cache success.
                self.db.execute("UPDATE jobs SET state='blocked',owner=NULL,lease_until=NULL,error_json=?,updated=? WHERE id=?",
                                (canonical_json({"reason": "stale_dependency", "key": key}).decode(), time.time(), row["id"]))
                self.event(row["id"], "invalidated", {"dependency": key, "new_hash": new_hash})
        return [r["id"] for r in rows]

    def remember(self, key: str, context_hash: str, value: dict):
        self.db.execute("INSERT OR REPLACE INTO memory VALUES (?,?,?,?)",
                        (key, context_hash, canonical_json(redact(value)).decode(), time.time()))

    def recall(self, key: str, context_hash: str) -> dict | None:
        row = self.db.execute("SELECT value_json FROM memory WHERE key=? AND context_hash=?", (key, context_hash)).fetchone()
        return json.loads(row[0]) if row else None

    def get(self, job_id: str) -> dict:
        row = self.db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        if not row:
            raise KeyError(job_id)
        return self._decode(row)

    def jobs(self, run_id: str | None = None) -> list[dict]:
        rows = self.db.execute("SELECT * FROM jobs" + (" WHERE run_id=?" if run_id else "") + " ORDER BY created",
                               (run_id,) if run_id else ()).fetchall()
        return [self._decode(row) for row in rows]

    @staticmethod
    def _decode(row) -> dict:
        value = dict(row)
        for key in list(value):
            if key.endswith("_json"):
                value[key[:-5]] = json.loads(value.pop(key)) if value[key] is not None else None
        return value
