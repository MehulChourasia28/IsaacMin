"""Run bounded source-interface projection in the root numerical environment."""
import argparse,json
from pathlib import Path
import numpy as np
from isaacmin.volumes.interface_projection import constrain_interfaces

parser=argparse.ArgumentParser();parser.add_argument('--request',required=True,type=Path)
args=parser.parse_args();request=json.loads(args.request.read_text())
ir=Path(request['ir']);source=np.load(ir/'natural_occupancy.npz',allow_pickle=False)
surface=np.load(ir/'terrain_surface.npz',allow_pickle=False)
solid=source['occupancy']==1
if request.get('support_mask'):solid=np.load(request['support_mask'],allow_pickle=False)['occupancy']
covered=np.load(ir/'topology/source_topology_labels.npz',allow_pickle=False)['covered_air']
print(json.dumps(constrain_interfaces(Path(request['mesh']),Path(request['output']),
    supporting_solid=solid,covered_air=covered,min_xyz=source['min_xyz'],
    exterior_height=surface['height'],exterior_validity=surface['validity'],
    known_source_air=(source['occupancy']==0)&source['validity'])),flush=True)
