"""Actual native geometry ray measurement; no RGB or gravity pass is inferred."""
import argparse
import hashlib
import json
from pathlib import Path
import traceback

p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--output',required=True)
a,_=p.parse_known_args();scene=Path(a.scene).resolve();output=Path(a.output).resolve()
output.mkdir(parents=True,exist_ok=True)
from isaacsim import SimulationApp
app=SimulationApp({'headless':True,'disable_viewport_updates':True,
                   'extra_args':[f'--/log/file={output}/kit.log']})
try:
    import omni.usd
    import omni.physx
    from ground_collision import configure_ground_collision
    from contact_rays import sample_contact_rays
    if not omni.usd.get_context().open_stage(str(scene)):raise RuntimeError('Native scene open failed')
    stage=omni.usd.get_context().get_stage()
    collision=configure_ground_collision(stage,preserve_authored=True)
    for _ in range(30):app.update()
    omni.physx.get_physx_interface().force_load_physics_from_usd()
    result=sample_contact_rays(stage,omni.physx.get_physx_scene_query_interface(),
        collision['render_ground_meshes'],output,collision_paths=collision['collision_meshes'])
    result.update(scope='native_same_full_geometry_two_sided_ray_measurement',
                  actual_gravity_test='not_run',RGB='not_run',qualification='not_qualified')
    (output/'measurement_result.json').write_text(json.dumps(result,indent=2)+'\n')
except BaseException:
    (output/'failure.txt').write_text(traceback.format_exc())
    raise
finally:app.close()
