#!/usr/bin/env bash
set -euo pipefail
ISAACMIN_PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ISAACMIN_PROJECT"
exec .venv/bin/python scripts/bootstrap_native.py "$@"
