"""Attach exact final outdoor triangle colliders in an isolated native USD worker."""
from pathlib import Path
import sys,argparse
from pxr import Usd
from isaacmin.io import atomic_json
from isaacmin.assembly.usd_instances import _closure

parser=argparse.ArgumentParser();parser.add_argument('--scene',type=Path,required=True)
parser.add_argument('--evidence',type=Path,required=True);args=parser.parse_args()
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'isaac_scripts'))
from ground_collision import configure_ground_collision
stage=Usd.Stage.Open(str(args.scene));parent=stage.GetPrimAtPath('/IsaacMinCollision')
if parent:
    parent.SetActive(True)
    for p in parent.GetChildren():p.SetActive(False)
result=configure_ground_collision(stage);stage.GetRootLayer().Save()
atomic_json(args.evidence,result)
atomic_json(args.scene.parent/'native_dependency_closure.json',_closure(args.scene))
print({'exact_collision_parts':len(result['collision_meshes']),'PhysX_contact':'not_run'})
