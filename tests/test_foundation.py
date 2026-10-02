import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from isaacmin.contracts import CoordinateFrame, seal_directory, verify_artifact, ArtifactError
from isaacmin.contracts.artifacts import promote, validate_array
from isaacmin.io import atomic_json, hash_object
from isaacmin.jobs import Journal, JobConflict, LeaseLost
from isaacmin.security import SecurityError, redact, safe_path, worker_environment
from isaacmin.validation.metrics import compare_depth, compare_ground, compare_seam, critic_calibration, unique_route_distance
from isaacmin.validation.profile import freeze_profile, load_profile


def test_coordinates_negative_asymmetric_and_right_handed():
    frame = CoordinateFrame((-1065.38, 0, 688.07))
    point = (-1010.2, -35.25, 700.125)
    assert np.allclose(frame.to_source(frame.to_world(point)), point)
    rotation = np.array(frame.record()["source_to_world"])[:3, :3]
    assert np.linalg.det(rotation) == pytest.approx(1)
    assert frame.to_world((-1065.38, 7, 689.07)) == pytest.approx((0, -1, 7))


def test_artifact_partial_tamper_and_idempotent_promotion(tmp_path):
    staging = tmp_path / "stage"
    staging.mkdir()
    (staging / "data.bin").write_bytes(b"unfinalized")
    with pytest.raises(FileNotFoundError):
        promote(staging, tmp_path / "published")
    seal_directory(staging, kind="fixture", build_id="test", scope={}, inputs={}, parameters={},
                   toolchain_hash="a" * 64, profile_hash="b" * 64, coordinate_frame=CoordinateFrame().record())
    promote(staging, tmp_path / "published")
    assert verify_artifact(tmp_path / "published")["status"] == "promoted"
    (tmp_path / "published/data.bin").write_bytes(b"tampered")
    with pytest.raises(ArtifactError):
        verify_artifact(tmp_path / "published")


def test_unknown_array_is_not_finite_ground():
    a = np.array([[1., np.nan], [5., 6.]])
    valid = np.array([[True, False], [True, True]])
    assert validate_array(a, shape=(2, 2), validity=valid)["valid_count"] == 3
    with pytest.raises(ArtifactError):
        validate_array(a, shape=(2, 2))
    with pytest.raises(ArtifactError):
        validate_array(np.ones((3, 2)), shape=(2, 3))


def test_journal_idempotency_leases_and_nonlocal_invalidation(tmp_path):
    j = Journal(tmp_path / "journal.sqlite")
    first = j.submit("run", "op", "erode", {"tile": 2}, {"catchment": "old"})
    assert j.submit("run", "op", "erode", {"tile": 2}, {"catchment": "old"})["id"] == first["id"]
    with pytest.raises(JobConflict):
        j.submit("run", "op", "erode", {"tile": 3}, {"catchment": "old"})
    owner = j.claim(first["id"])
    assert j.claim(first["id"]) is None
    j.db.execute("UPDATE jobs SET lease_until=0 WHERE id=?", (first["id"],))
    assert j.recover() == [first["id"]]
    next_owner = j.claim(first["id"])
    with pytest.raises(LeaseLost):
        j.finish(first["id"], owner, state="succeeded", result={})
    j.finish(first["id"], next_owner, state="succeeded", result={"artifact": "verified"})
    assert j.invalidate("catchment", "new") == [first["id"]]
    assert j.get(first["id"])["state"] == "blocked"
    j.close()


def test_process_death_does_not_commit_job_success(tmp_path):
    db = tmp_path / "journal.sqlite"
    script = ("from pathlib import Path; from isaacmin.jobs import Journal; import os; "
              "j=Journal(Path(os.environ['TEST_JOURNAL'])); "
              "x=j.submit('r','o','test',{},{}); j.claim(x['id']); os._exit(9)")
    env = dict(os.environ, TEST_JOURNAL=str(db), PYTHONPATH=str(Path(__file__).parents[1] / "src"))
    proc = subprocess.run([sys.executable, "-c", script], env=env, timeout=10)
    assert proc.returncode == 9
    j = Journal(db)
    assert j.jobs()[0]["state"] == "running"
    assert j.jobs()[0]["result"] is None
    j.db.execute("UPDATE jobs SET lease_until=0")
    j.recover()
    assert j.jobs()[0]["state"] == "queued"
    j.close()


def test_redaction_and_sandbox_paths(tmp_path, monkeypatch):
    fake = "nvapi-fixture-not-a-real-key"
    assert fake not in json.dumps(redact({"error": fake, "authorization": "Bearer " + fake}))
    monkeypatch.setenv("NVIDIA_API_KEY", fake)
    assert "NVIDIA_API_KEY" not in worker_environment()
    for path in ("../escape", "/tmp/escape", "a/../../escape", "a\\evil"):
        with pytest.raises(SecurityError):
            safe_path(tmp_path, path)
    (tmp_path / "link").symlink_to(tmp_path.parent)
    with pytest.raises(SecurityError):
        safe_path(tmp_path, "link/outside")


def test_numerical_gates_reject_deliberate_defects():
    base = np.zeros((10000, 3))
    moved = base.copy()
    moved[-1, 2] = .021
    assert not compare_ground(base, moved)["pass"]
    moved[-1, 2] = .003
    assert not compare_seam(base, moved)["pass"]
    opaque = np.ones((100, 100), bool)
    z = np.ones((100, 100)) * 5
    with pytest.raises(ValueError):
        compare_depth(z, z, opaque, semantics="unknown")
    with pytest.raises(ValueError):
        compare_depth(z, z, opaque, semantics="euclidean_range")
    result = compare_depth(z * 2, z, opaque, semantics="euclidean_range", ray_cosines=np.ones_like(z) * .5)
    assert result["pass"]
    assert not compare_depth(z + .1, z, opaque, semantics="axial_z")["pass"]
    with pytest.raises(ValueError):
        compare_depth(-z, -z, opaque, semantics="axial_z")
    with pytest.raises(ValueError):
        compare_depth(z, z, opaque, semantics="euclidean_range", ray_cosines=np.ones_like(z)*2)


def test_expired_worker_cannot_publish(tmp_path):
    j = Journal(tmp_path / "journal.sqlite")
    job = j.submit("r", "o", "export", {}, {})
    owner = j.claim(job["id"])
    j.db.execute("UPDATE jobs SET lease_until=0 WHERE id=?", (job["id"],))
    published = []
    with pytest.raises(LeaseLost):
        j.publish(job["id"], owner, {"artifact": "x"}, lambda: published.append(True))
    assert not published
    j.recover()
    new_owner = j.claim(job["id"])
    j.publish(job["id"], new_owner, {"artifact": "x"}, lambda: published.append(True))
    assert published == [True]
    assert j.get(job["id"])["state"] == "succeeded"
    j.close()


def test_repeated_laps_do_not_become_unique_coverage():
    route = [[0, 0, 0], [20, 0, 0], [0, 0, 0], [20, 0, 0]]
    result = unique_route_distance(route)
    assert result["raw_distance_m"] == 60
    assert result["unique_distance_m"] == 20


def test_always_clear_critic_and_profile_changes_rejected(tmp_path):
    cases = [{"id": c, "category": c, "severe": True, "held_out": True, "detected": False}
             for c in ("seam", "floating_asset", "missing_material", "broken_opacity", "voxel_steps")]
    cases.append({"id": "clean", "category": "clean", "severe": False, "detected": False})
    assert not critic_calibration(cases)["pass"]
    for case in cases:
        case["detected"] = True
    assert not critic_calibration(cases)["pass"]  # Always-alarming critic is equally unusable.
    cases[-1]["detected"] = False
    assert critic_calibration(cases)["pass"]
    path = tmp_path / "quality_profile.json"
    frozen = freeze_profile(path, {"source": "test"})
    assert load_profile(path) == frozen
    with pytest.raises(ValueError):
        freeze_profile(path, {"source": "changed"})
    record = json.loads(path.read_text())
    record["profile"]["thresholds"]["seam_gap_max_m"] = .5
    record["sha256"] = hash_object(record["profile"])
    atomic_json(path, record)
    with pytest.raises(ValueError):
        load_profile(path)
