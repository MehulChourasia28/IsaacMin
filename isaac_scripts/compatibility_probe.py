"""Execute only with Isaac's own Python. Produces fresh render/contact evidence."""
import argparse
import json
import pathlib
import time

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
args, kit_args = parser.parse_known_args()
output = pathlib.Path(args.output).resolve()
output.mkdir(parents=True, exist_ok=True)
from isaacsim import SimulationApp
app = SimulationApp({'headless': True, 'width': 640, 'height': 480,
                     'renderer': 'RayTracedLighting',
                     'extra_args': ['--/app/window/enabled=false',
                                    '--/app/renderer/skipWhileMinimized=false',
                                    f'--/log/file={output}/kit.log']})
try:
    import numpy as np
    import omni.replicator.core as rep
    import omni.usd
    from PIL import Image
    from pxr import UsdGeom, UsdLux, UsdPhysics, Gf
    from isaacsim.core.api import World
    from isaacsim.core.api.objects import DynamicCuboid
    world = World(stage_units_in_meters=1.0)
    world.scene.add_default_ground_plane()
    box = world.scene.add(DynamicCuboid(prim_path='/World/ContactProbe',
                         name='contact_probe', position=np.array([0, 0, 1.0]),
                         scale=np.array([0.2, 0.2, 0.2]), mass=1.0))
    stage = omni.usd.get_context().get_stage()
    light = UsdLux.DomeLight.Define(stage, '/World/Light')
    light.CreateIntensityAttr(1500)
    camera = rep.create.camera(position=(2, 2, 1.2), look_at=(0, 0, 0.1))
    product = rep.create.render_product(camera, (640, 480))
    rgb = rep.AnnotatorRegistry.get_annotator('rgb')
    depth = rep.AnnotatorRegistry.get_annotator('distance_to_image_plane')
    rgb.attach(product); depth.attach(product)
    start = time.monotonic()
    world.reset()
    for _ in range(180):
        world.step(render=True)
    for _ in range(10):
        rep.orchestrator.step(rt_subframes=4)
    pixels = rgb.get_data()
    distances = depth.get_data()
    if pixels.shape[:2] != (480, 640) or pixels.std() < 2:
        raise RuntimeError('RTX RGB is missing or degenerate')
    valid = distances[np.isfinite(distances) & (distances > 0)]
    if valid.size < 1000:
        raise RuntimeError('Metric axial depth contains insufficient finite samples')
    position, _ = box.get_world_pose()
    error = abs(float(position[2])-0.1)
    if error > 0.02:
        raise RuntimeError(f'Dropped body failed support: z={position[2]}')
    Image.fromarray(pixels).save(output/'rgb.png')
    np.save(output/'depth_axial_m.npy', distances)
    stage.GetRootLayer().Export(str(output/'probe.usda'))
    (output/'result.json').write_text(json.dumps({
        'status':'pass', 'scope':'synthetic_native_compatibility_only',
        'rgb_shape':list(pixels.shape), 'depth_semantics':'axial_m',
        'depth_finite_samples':int(valid.size), 'contact_height_m':float(position[2]),
        'contact_error_m':error, 'wall_seconds':time.monotonic()-start,
        'navigation_stack':'not_run'}, indent=2)+'\n')
finally:
    app.close()
