#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
if [[ "$(uname -m)" != aarch64 ]]; then
  printf '%s\n' 'IsaacMin native delivery requires Linux aarch64. Source tests may run on other hosts.' >&2
  exit 2
fi
if [[ ! -x .venv/bin/python ]]; then
  python3 -m venv .venv
fi
.venv/bin/python -m pip install --cache-dir .cache/pip -r requirements.lock
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
if [[ -f scripts/bootstrap_native.py ]]; then
  .venv/bin/python scripts/bootstrap_native.py
elif [[ -f scripts/bootstrap_native.sh ]]; then
  bash scripts/bootstrap_native.sh
fi
exec .venv/bin/isaacmin bootstrap "$@"
