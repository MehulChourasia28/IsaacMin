"""Run using the pinned CPU OpenUSD environment, isolated from Isaac/Blender."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from isaacmin.assembly.usd_instances import share_scene

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--destination', required=True)
    parser.add_argument('--evidence', required=True)
    args = parser.parse_args()
    result = share_scene(args.source, args.destination, args.evidence)
    print(json.dumps({k: v for k, v in result.items() if k != 'prototypes'}, indent=2), flush=True)
