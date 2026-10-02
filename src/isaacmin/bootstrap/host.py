from __future__ import annotations

import importlib.metadata
import platform
import shutil
import subprocess
from pathlib import Path

from ..io import atomic_json, utc_now
from ..jobs.resources import memory_status
from ..security import redact, worker_environment


def _probe(argv: list[str], timeout=20) -> dict:
    binary = shutil.which(argv[0])
    if not binary:
        return {"status": "not_found"}
    try:
        result = subprocess.run([binary, *argv[1:]], capture_output=True, text=True,
                                timeout=timeout, env=worker_environment())
        return {"binary": binary, "exit_code": result.returncode,
                "status": "pass" if result.returncode == 0 else "fail",
                "output": redact((result.stdout + result.stderr)[-16000:])}
    except subprocess.TimeoutExpired:
        return {"binary": binary, "status": "fail", "reason": "timeout"}


def inspect_host(workspace: Path, *, persist: bool = False) -> dict:
    os_release = {}
    path = Path("/etc/os-release")
    if path.exists():
        for line in path.read_text().splitlines():
            key, _, value = line.partition("=")
            os_release[key] = value.strip('"')
    disk = shutil.disk_usage(workspace)
    versions = {}
    for name in ("numpy", "httpx", "PyYAML", "Pillow", "jsonschema", "pytest", "nbtlib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not_installed"
    report = {"schema_version": "1.0", "created_at_utc": utc_now(),
              "architecture": platform.machine(), "host": platform.node(),
              "kernel": platform.release(), "os_release": os_release,
              "memory": memory_status(), "disk": {"total_bytes": disk.total, "available_bytes": disk.free},
              "gpu": _probe(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
              "cuda_compiler": _probe(["nvcc", "--version"]),
              "opencl": _probe(["clinfo", "--list"]), "vulkan": _probe(["vulkaninfo", "--summary"]),
              "python": platform.python_version(), "python_packages": versions,
              "existing_ros": [str(p) for p in Path("/opt/ros").glob("*")] if Path("/opt/ros").exists() else [],
              "containers": {name: shutil.which(name) for name in ("docker", "podman", "apptainer")},
              "system_configuration_modified": False,
              "note": "GPU failure inside a sandbox may reflect isolated device nodes; inspect host access before driver conclusions"}
    report["status"] = "pass" if report["architecture"] == "aarch64" and report["gpu"]["status"] == "pass" else "blocked"
    if persist:
        atomic_json(workspace / "artifacts/bootstrap/host.json", report)
    return report

