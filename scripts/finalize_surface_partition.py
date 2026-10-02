"""Verify and compact one fresh native Blender halo export; preserve input arrays."""
from pathlib import Path
import argparse,os
from pxr import Usd
from isaacmin.io import read_json,atomic_json,sha256_file
from isaacmin.terrain.surface_partitions import native_partition
p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);a=p.parse_args()
r=read_json(a.request);out=Path(r['output']);raw=out/'outdoor_terrain.usdc'
source_hash=sha256_file(raw);stage=Usd.Stage.Open(str(raw))
result=native_partition(stage,Path(r['terrain']),r['partition'])
target=out/'verified_terrain.usdc';stage.GetRootLayer().Export(str(target))
# Fresh reopen verifies stored positions and topology, not just in-memory arrays.
import numpy as np
from pxr import UsdGeom
from isaacmin.terrain.surface_partitions import window_array
from isaacmin.terrain.surface_navigation import grid_triangles
check=Usd.Stage.Open(str(target));mesh=UsdGeom.Mesh(check.GetPrimAtPath(result['mesh_path']))
z0,z1,x0,x1=r['partition']['core']
assert np.array_equal(np.asarray(mesh.GetPointsAttr().Get()),window_array(Path(r['terrain']),'vertices.npy',r['partition']['core']))
assert np.array_equal(np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1,3),grid_triangles(z1-z0+1,x1-x0+1))
result.update(native_export_sha256=source_hash,usd_sha256=sha256_file(target),fresh_native_reopen='pass',
    raw_export_storage='transient current-worker halo export; compacted losslessly after complete equality checks')
atomic_json(out/'partition.json',result)
# This transient file was created by this partition job, never an earlier build.
raw.unlink()
print({k:result[k] for k in ('name','vertices','triangles','fresh_native_reopen')})
