"""Acquire separate original terrain-material candidates; never print credentials."""
from pathlib import Path
import argparse,json,sys
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'))
from isaacmin.assets.material_masters import acquire_material_candidates
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--resolution',choices=('4k','8k'),default='4k')
args=p.parse_args();result=acquire_material_candidates(root,Path(args.output),resolution=args.resolution)
print(json.dumps({'status':result['status'],'materials':len(result['materials']),'qualification':result['qualification'],'output':args.output}))
