"""Continue a frozen, verified cached population without rebuilding terrain."""
import argparse
from pathlib import Path

from isaacmin.assembly.cached_population import populate_cached_request

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--request", type=Path, required=True)
parser.add_argument("--evidence", type=Path, required=True)
args = parser.parse_args()
result = populate_cached_request(Path(__file__).resolve().parents[1], args.request, args.evidence)
print({"status": result["status"], "placed_native_objects": result["placed_native_objects"],
       "appearance_qualification": result["appearance_qualification"]})
