"""Bounded subprocess execution; no shell and no implicit success by exit code."""
from __future__ import annotations

import os
import json
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable

from .io import atomic_json, utc_now
from .security import redact, worker_environment


class ResourcePressureError(RuntimeError):
    """The candidate stopped before exhausting the shared memory reserve."""


def _available_memory() -> int:
    return next(int(line.split()[1])*1024 for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:"))


def _thermal_samples() -> list[dict]:
    """Read exposed sensors only; never alter cooling, clocks or power limits."""
    samples = []
    for zone in sorted(Path('/sys/class/thermal').glob('thermal_zone*')):
        try:
            critical = []
            for kind in zone.glob('trip_point_*_type'):
                if kind.read_text().strip() == 'critical':
                    critical.append(int(kind.with_name(kind.name.replace('_type', '_temp')).read_text()))
            samples.append({'zone': zone.name, 'type': (zone / 'type').read_text().strip(),
                            'temperature_millicelsius': int((zone / 'temp').read_text()),
                            'critical_trip_millicelsius': min(critical) if critical else None})
        except (OSError, ValueError):
            continue
    return samples


def _stop_worker(proc: subprocess.Popen, grace_seconds: float) -> None:
    """Terminate the process group and reap the child, including exit races."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(grace_seconds)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()


def run_worker(argv: list[str], *, cwd: Path, log_path: Path, timeout: float,
               environment: dict[str, str] | None = None,
               heartbeat: Callable[[], None] | None = None,
               estimated_memory_bytes: int | None = None,
               estimated_disk_bytes: int = 0,
               shared_memory_reserve_bytes: int = 24*2**30) -> dict:
    if not argv or not Path(argv[0]).is_absolute() or timeout <= 0:
        raise ValueError("Worker needs absolute executable, argument array and positive timeout")
    if redact(argv) != argv:
        raise ValueError("Secrets forbidden in worker command arguments")
    if estimated_memory_bytes is not None:
        from .jobs.resources import admit
        admission = admit(cwd, estimated_memory_bytes, estimated_disk_bytes)
        if admission["status"] != "admitted":
            raise ResourcePressureError("Worker admission deferred: " + admission["reason"])
    if _available_memory() < shared_memory_reserve_bytes:
        raise ResourcePressureError("Worker admission deferred: shared memory reserve")
    started = time.monotonic()
    peak_parent_rss = None
    minimum_system_available = None
    resource_samples = []
    last_sample = 0.
    interruption = None
    interruption_traceback = None
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = utc_now()
    boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    progress_path = log_path.with_suffix('.heartbeat.json')
    samples_path = log_path.with_suffix('.resources.jsonl')
    result_path = log_path.with_suffix('.process.json')
    for previous in (progress_path, samples_path, result_path):
        if previous.exists():
            previous.rename(previous.with_name(previous.name + '.previous.' + str(time.time_ns())))
    # Raw output stays in an anonymous temporary file until redaction.
    with tempfile.TemporaryFile() as output:
        proc = subprocess.Popen(argv, cwd=cwd, env=worker_environment(environment), shell=False,
                                stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        timed_out = False
        resource_limited = False
        try:
            atomic_json(progress_path, {'schema_version': 1, 'status': 'running', 'pid': proc.pid,
                'boot_id': boot_id, 'started_at_utc': started_at, 'samples': 0,
                'artifact_validation': 'not_run'})
            while proc.poll() is None:
                now = time.monotonic()
                if now-last_sample >= 2:
                    last_sample = now
                    try:
                        status = Path(f"/proc/{proc.pid}/status").read_text()
                        rss = next(int(line.split()[1])*1024 for line in status.splitlines() if line.startswith("VmRSS:"))
                        peak_parent_rss = rss if peak_parent_rss is None else max(peak_parent_rss, rss)
                        available = _available_memory()
                        minimum_system_available = available if minimum_system_available is None else min(minimum_system_available, available)
                        sample = {"elapsed_s": now-started, "parent_rss_bytes": rss,
                                  "system_available_bytes": available, "at_utc": utc_now(),
                                  "thermal_zones": _thermal_samples()}
                        resource_samples.append(sample)
                        # Save evidence while the child runs. In-memory samples
                        # and anonymous stdout cannot survive a host power loss.
                        with samples_path.open('a') as sampled:
                            sampled.write(json.dumps(sample, sort_keys=True) + '\n')
                            sampled.flush()
                            os.fsync(sampled.fileno())
                        atomic_json(progress_path, {'schema_version': 1, 'status': 'running',
                            'pid': proc.pid, 'boot_id': boot_id, 'started_at_utc': started_at,
                            'samples': len(resource_samples), 'latest_sample': sample,
                            'sampled_peak_parent_rss_bytes': peak_parent_rss,
                            'minimum_system_available_bytes': minimum_system_available,
                            'shared_memory_reserve_bytes': shared_memory_reserve_bytes,
                            'artifact_validation': 'not_run'})
                        if available < shared_memory_reserve_bytes:
                            resource_limited = True
                            _stop_worker(proc, 2)
                            break
                    except (FileNotFoundError, StopIteration, ProcessLookupError):
                        pass
                if heartbeat:
                    heartbeat()
                if time.monotonic() - started > timeout:
                    timed_out = True
                    _stop_worker(proc, 10)
                    break
                time.sleep(0.25)
            proc.wait()
        except BaseException as exc:
            interruption = exc
            interruption_traceback = exc.__traceback__
            _stop_worker(proc, 10)
        finally:
            output.seek(0)
            with log_path.open("w") as handle:
                for line in output:
                    handle.write(redact(line.decode("utf8", errors="replace")))
    result = {"argv": argv, "cwd": str(cwd), "exit_code": proc.returncode,
              "pid": proc.pid, "child_reaped": proc.returncode is not None,
              "boot_id": boot_id, "started_at_utc": started_at,
              "timed_out": timed_out, "elapsed_seconds": time.monotonic() - started,
              "resource_limited": resource_limited,
              "interrupted": interruption is not None,
              "interruption_exception_type": type(interruption).__name__ if interruption is not None else None,
              "completed_at_utc": utc_now(), "status": "interrupted" if interruption is not None else
                  ("process_exited" if proc.returncode == 0 and not timed_out and not resource_limited else "failed"),
              "artifact_validation": "not_run", "log": str(log_path),
              "resources": {"sampled_peak_parent_rss_bytes": peak_parent_rss,
                            "minimum_system_available_bytes": minimum_system_available,
                            "shared_memory_reserve_bytes": shared_memory_reserve_bytes,
                            "estimated_memory_bytes": estimated_memory_bytes,
                            "estimated_disk_bytes": estimated_disk_bytes,
                            "samples": resource_samples,
                            "accounting": "Parent RSS only; system availability observes all shared CPU/GPU pressure. Never add overlapping GPU/RAM totals."}}
    atomic_json(result_path, result)
    atomic_json(progress_path, {'schema_version': 1, 'status': result['status'],
        'pid': proc.pid, 'boot_id': boot_id, 'started_at_utc': started_at,
        'completed_at_utc': result['completed_at_utc'], 'samples': len(resource_samples),
        'exit_code': proc.returncode, 'child_reaped': result['child_reaped'],
        'process_result': result_path.name, 'artifact_validation': 'not_run'})
    if interruption is not None:
        raise interruption.with_traceback(interruption_traceback)
    if resource_limited:
        raise ResourcePressureError("Worker stopped at shared memory reserve; candidate retained without publication: " + str(log_path))
    return result
