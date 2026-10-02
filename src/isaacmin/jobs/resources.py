from __future__ import annotations

from pathlib import Path
import shutil


def memory_status() -> dict:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, _, value = line.partition(":")
        values[name] = int(value.strip().split()[0]) * 1024
    available = values["MemAvailable"]
    reserve = 24 * 2**30
    return {"shared_system_total_bytes": values["MemTotal"], "available_bytes": available,
            "reserve_bytes": reserve, "heavy_workers": 1,
            "job_ceiling_bytes": max(0, min(80 * 2**30, available - reserve)),
            "accounting": "CPU and GPU share the same physical pool; GPU allocations are not added to RAM"}


def admit(workspace: Path, estimated_memory_bytes: int, estimated_disk_bytes: int) -> dict:
    memory = memory_status()
    disk = shutil.disk_usage(workspace)
    if estimated_memory_bytes > memory["job_ceiling_bytes"]:
        return {"status": "waiting_resource", "reason": "shared_memory_reserve", "memory": memory}
    # Keep ample absolute headroom on the Spark's multi-terabyte drive without
    # treating hundreds of usable GiB as exhausted. Memory/quality gates stay
    # unchanged; estimated output bytes are reserved in addition to this floor.
    disk_reserve = min(int(0.2 * disk.total), 256 * 2**30)
    if disk.free - estimated_disk_bytes < disk_reserve:
        return {"status": "waiting_resource", "reason": "disk_reserve", "memory": memory}
    return {"status": "admitted", "memory": memory, "free_disk_bytes": disk.free,
            "disk_reserve_bytes": disk_reserve}
