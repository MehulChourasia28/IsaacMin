import os
import signal
import sys
import time

import pytest

from isaacmin.io import read_json
from isaacmin.process import ResourcePressureError, run_worker


def test_worker_stops_on_shared_memory_pressure_and_preserves_evidence(tmp_path, monkeypatch):
    # Simulate only the accounting signal; the child is an actual process which
    # must be terminated before its successful output can be published.
    readings = iter([128*2**30, 8*2**30])
    monkeypatch.setattr("isaacmin.process._available_memory", lambda: next(readings))
    with pytest.raises(ResourcePressureError, match="candidate retained"):
        run_worker([sys.executable, "-c", "import time; time.sleep(30)"], cwd=tmp_path,
                   log_path=tmp_path / "worker.log", timeout=60)
    record = read_json(tmp_path / "worker.process.json")
    assert record["resource_limited"] and record["status"] == "failed"
    assert record["elapsed_seconds"] < 5
    assert (tmp_path / "worker.log").exists()


def test_actual_worker_heartbeat_interrupt_reaps_child_and_retains_measured_receipt(tmp_path):
    # Actual /proc telemetry and child termination; only the heartbeat interruption
    # is controlled. No native tool, source map, or renderer result is represented.
    child = tmp_path / 'child.py'
    child.write_text("import os,time\nfrom pathlib import Path\n"
                     "print('child ready; nvapi-controlled-test-only',flush=True)\n"
                     "Path('child.pid').write_text(str(os.getpid()))\n"
                     "time.sleep(60)\n")
    started = time.monotonic()

    def heartbeat():
        if (tmp_path / 'child.pid').exists() and time.monotonic() - started >= 2.1:
            # These files must exist before any exit handling. A lost host
            # cannot execute finally or fabricate an exit code on recovery.
            live = read_json(tmp_path / 'interrupted.heartbeat.json')
            assert live['status'] == 'running' and live['samples'] >= 1
            assert live['boot_id'] and live['artifact_validation'] == 'not_run'
            assert not (tmp_path / 'interrupted.process.json').exists()
            lines = (tmp_path / 'interrupted.resources.jsonl').read_text().splitlines()
            assert len(lines) == live['samples']
            raise KeyboardInterrupt('controlled heartbeat interruption')

    with pytest.raises(KeyboardInterrupt, match='controlled heartbeat interruption'):
        run_worker([sys.executable, str(child)], cwd=tmp_path, log_path=tmp_path / 'interrupted.log',
                   timeout=10, heartbeat=heartbeat, shared_memory_reserve_bytes=0)
    pid = int((tmp_path / 'child.pid').read_text())
    assert not os.path.exists(f'/proc/{pid}')
    with pytest.raises(ChildProcessError):
        os.waitpid(pid, os.WNOHANG)
    record = read_json(tmp_path / 'interrupted.process.json')
    assert record['pid'] == pid and record['child_reaped']
    assert record['status'] == 'interrupted' and record['interrupted']
    assert record['interruption_exception_type'] == 'KeyboardInterrupt'
    assert record['exit_code'] == -signal.SIGTERM
    assert not record['timed_out'] and not record['resource_limited']
    assert record['artifact_validation'] == 'not_run'
    samples = record['resources']['samples']
    assert samples and all(s['system_available_bytes'] > 0 for s in samples)
    assert all(0 <= s['elapsed_s'] <= record['elapsed_seconds'] for s in samples)
    assert record['resources']['sampled_peak_parent_rss_bytes'] == max(s['parent_rss_bytes'] for s in samples)
    assert record['resources']['minimum_system_available_bytes'] == min(s['system_available_bytes'] for s in samples)
    assert record['resources']['estimated_memory_bytes'] is None
    final_heartbeat = read_json(tmp_path / 'interrupted.heartbeat.json')
    assert final_heartbeat['status'] == 'interrupted'
    assert final_heartbeat['exit_code'] == -signal.SIGTERM
    assert final_heartbeat['boot_id'] == record['boot_id']
    log = (tmp_path / 'interrupted.log').read_text()
    assert 'child ready' in log and '[REDACTED]' in log and 'nvapi-' not in log
