"""Read actual generated meshes for base-anchor, UV and geometry diagnostics."""
import bpy
import json
import math
import sys
from pathlib import Path

path=Path(sys.argv[sys.argv.index('--')+1]).resolve()
report=json.loads(path.read_text())
bpy.ops.wm.open_mainfile(filepath=report['output_blend'],use_scripts=False,load_ui=False)
for variant in report['variants']:
    objects=[bpy.data.objects[name] for name in variant['objects']]
    trunk=next(o for o in objects if o.name.endswith('_trunk'))
    minimum=min(v.co.z for v in trunk.data.vertices)
    band=[list(v.co) for v in trunk.data.vertices if v.co.z<=minimum+.005]
    stride=max(1,len(band)//32)
    variant['contact_anchors_local_m']=band[::stride][:32]
    variant['root_plane_z_m']=minimum
    variant['numeric_geometry']={'finite':all(math.isfinite(c) for o in objects for v in o.data.vertices for c in v.co),
                                 'all_meshes_have_uv':all(bool(o.data.uv_layers) for o in objects),
                                 'native_anchor_count':len(variant['contact_anchors_local_m'])}
report['native_root_plane_inspection']='external_tool_verified; final supporting surface contact remains untested'
path.write_text(json.dumps(report,indent=2))
