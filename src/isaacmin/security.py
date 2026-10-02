"""Secret containment, workspace paths and non-executable local configuration."""
from __future__ import annotations

import getpass
import os
import re
import stat
from pathlib import Path
from typing import Any


class SecurityError(ValueError):
    pass


def secret(workspace: Path, prompt: bool = False) -> str | None:
    value = os.environ.get("NVIDIA_API_KEY")
    if value:
        return value.strip()
    path = workspace / ".env"
    if path.exists():
        if path.is_symlink() or stat.S_IMODE(path.stat().st_mode) & 0o077:
            raise SecurityError("Local .env must be a regular file with permissions 0600")
        for line in path.read_text().splitlines():
            if line.startswith("NVIDIA_API_KEY="):
                return line.partition("=")[2].strip().strip("\"'") or None
    if prompt:
        value = getpass.getpass("NVIDIA API key (hidden): ").strip()
        if value:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as handle:
                handle.write("NVIDIA_API_KEY=" + value + "\n")
            return value
    return None


def redact(value: Any) -> Any:
    """Do not retain HTTP auth headers, keys, or arbitrary exception payloads."""
    if isinstance(value, str):
        text = re.sub(r"nvapi-[A-Za-z0-9_-]+", "[REDACTED]", value)
        text = re.sub(r"(?i)(bearer\s+)[^\s,\"']+", r"\1[REDACTED]", text)
        return re.sub(r"(?i)(NVIDIA_API_KEY\s*[=:]\s*)[^\s,\"']+", r"\1[REDACTED]", text)
    if isinstance(value, dict):
        return {str(k): "[REDACTED]" if str(k).lower() in {
            "authorization", "api_key", "nvidia_api_key", "access_token"
        } else redact(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    return value


def safe_path(root: Path, relative: str, *, must_exist: bool = False) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or not relative or "\\" in relative:
        raise SecurityError("Expected a relative workspace path without traversal")
    target = (root / path).resolve()
    if not target.is_relative_to(root.resolve()):
        raise SecurityError("Path escapes the workspace")
    if must_exist and not target.is_file():
        raise SecurityError("Required file is absent")
    return target


def worker_environment(extra: dict[str, str] | None = None) -> dict[str, str]:
    # Explicit allowlist. Asset imports do not inherit API/cloud credentials.
    keys = ("PATH", "HOME", "USER", "LANG", "LC_ALL", "DISPLAY", "XDG_RUNTIME_DIR",
            "LD_LIBRARY_PATH", "CUDA_VISIBLE_DEVICES", "VK_ICD_FILENAMES")
    env = {k: os.environ[k] for k in keys if k in os.environ}
    env.update({"PYTHONNOUSERSITE": "1", "QT_QPA_PLATFORM": "offscreen"})
    if extra:
        if any("KEY" in k or "TOKEN" in k or "SECRET" in k for k in extra):
            raise SecurityError("Secrets are not permitted in worker overrides")
        env.update(extra)
    return env

