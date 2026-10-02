#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
if [[ ! -x .venv/bin/isaacmin ]]; then
  printf '%s\n' 'Run ./scripts/bootstrap_spark.sh to install the isolated project environment.' >&2
  exit 2
fi
exec .venv/bin/isaacmin "$@"
