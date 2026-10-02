"""Persist the actual outdoor illumination alongside a portable USD world."""
from pathlib import Path
import json
import shutil

from isaacmin.io import atomic_json, read_json, sha256_file


def daylight_camera_response(pose, capture):
    """A modest daylight lift, fading to zero in already-exposed deep shade."""
    response=pose.get('camera_response')
    profile=capture['lighting_parameters'].get('daylight_response')
    if not response or not profile or response.get('daylight_response_applied'):return pose
    original=float(response['ev100'])
    lift=float(profile['maximum_lift_ev'])*max(0.,min(1.,
        (original-float(profile['shade_ev100']))/(float(profile['open_ev100'])-float(profile['shade_ev100']))))
    return dict(pose,camera_response=dict(response,ev100=original-lift,
        daylight_response_applied=True,daylight_lift_ev=lift,
        before_daylight_ev100=original))


def canopy_camera_response(pose, trees):
    """Account for a reconstructed broad crown larger than its voxel canopy.

    This bounded camera-only exposure estimate uses prepared geometry dimensions
    and tree placements. It is not a light change, image filter or calibrated
    radiometer. Original source shade and the measured crown extent are recorded.
    """
    import numpy as np
    candidates=[t for t in trees if 'bounds_world_min' in t or
        (t.get('species')=='jungle' and 'dimensions_m' in t)]
    if not candidates:return pose
    p=np.asarray(pose['position']);grid=np.linspace(-4,4,9)
    xx,yy=np.meshgrid(grid,grid);x=p[0]+xx.ravel();y=p[1]+yy.ravel()
    covered=np.zeros(len(x),bool)
    for tree in candidates:
        if 'bounds_world_min' in tree:
            low=np.asarray(tree['bounds_world_min']);high=np.asarray(tree['bounds_world_max'])
            center=(low+high)*.5;size=high-low;angle=0.
            if center[2]<=p[2]:continue
        else:
            center=np.asarray(tree['position_world_xyz']);size=np.asarray(tree['dimensions_m'])
            if center[2]+.5*size[2]<=p[2]:continue
            angle=np.deg2rad(tree['yaw_degrees'])
        if np.min(size[:2])<=0:continue
        dx=x-center[0];dy=y-center[1]
        u=dx*np.cos(angle)+dy*np.sin(angle);v=-dx*np.sin(angle)+dy*np.cos(angle)
        covered|=(u/(.45*size[0]))**2+(v/(.45*size[1]))**2<=1.
    cover=float(covered.mean())
    response=dict(pose.get('camera_response',{}));original=float(response.get('ev100',13.747247562465875))
    # Broadleaf shade in these daylight defaults spans up to four exposure stops.
    # Existing darker source-derived forest exposure remains unchanged.
    ev=min(original,13.747247562465875-4.*cover)
    response.update(ev100=ev,f_number=2.8 if ev<11 else 8.,iso=100,
        original_source_ev100=original,prepared_crown_footprint_cover=cover,
        policy='source shade plus bounded prepared-crown footprint inference; camera response only; photometric qualification not_run')
    return dict(pose,camera_response=response)


def capture_configuration(workspace, path):
    """Resolve reusable render settings; camera positions come from each save."""
    workspace = Path(workspace).resolve()
    record = read_json(workspace / path)
    if record.get('renderer_recipe') != 'pathtracing_1024':
        raise ValueError('Outdoor capture requires the selected full-quality renderer')
    hdri = Path(record['hdri'])
    hdri = hdri if hdri.is_absolute() else workspace / hdri
    expected = record['lighting_parameters']['hdri_sha256']
    if hdri.is_symlink() or sha256_file(hdri) != expected:
        raise ValueError('Original outdoor HDRI differs from the recorded asset')
    record['hdri'] = str(hdri.resolve())
    return record


def author_outdoor_lighting(stage, directory, capture, poses):
    """Author native lights, the first derived camera, and portable settings.

    Exposure is a pinned Isaac runtime control. The companion JSON lets the
    bundled loader restore it; standard USD lights remain usable independently.
    """
    from pxr import Gf, Sdf, UsdGeom, UsdLux
    directory = Path(directory)
    if not poses:
        raise ValueError('A source-derived initial outdoor view is required')
    source = Path(capture['hdri'])
    digest = sha256_file(source)
    if digest != capture['lighting_parameters']['hdri_sha256']:
        raise ValueError('Outdoor illumination changed before native assembly')
    relative = 'textures/' + digest[:16] + source.suffix
    target = directory / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        shutil.copyfile(source, target)
    if sha256_file(target) != digest:
        raise ValueError('Portable illumination bytes differ')
    params = capture['lighting_parameters']
    light = UsdLux.DomeLight.Define(stage, '/IsaacMinLighting/Dome')
    light.CreateTextureFileAttr(Sdf.AssetPath('./' + relative))
    light.CreateTextureFormatAttr('latlong')
    light.CreateIntensityAttr(float(params['dome_intensity']))
    light.CreateExposureAttr(float(params.get('dome_exposure', 0)))
    UsdGeom.Xformable(light).AddRotateZOp().Set(float(params.get('dome_rotation_z_degrees', 0)))
    sun = UsdLux.DistantLight.Define(stage, '/IsaacMinLighting/Sun')
    sun.CreateIntensityAttr(float(params['sun_intensity']))
    sun.CreateAngleAttr(.53)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(-45, 20, 25))
    pose = poses[0]
    camera_path = '/World/Cameras/NavigationPreview'
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.CreateFocalLengthAttr(18.)
    camera.CreateHorizontalApertureAttr(36.)
    width, height = capture['resolution']
    camera.CreateVerticalApertureAttr(36. * height / width)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.02, 5000.))
    matrix = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*pose['position']),
        Gf.Vec3d(*pose['look_at']), Gf.Vec3d(0, 0, 1)).GetInverse()
    UsdGeom.Xformable(camera).AddTransformOp().Set(matrix)
    portable = json.loads(json.dumps({key: capture[key] for key in (
        'renderer_recipe', 'resolution', 'lighting_parameters', 'native_denoising') if key in capture}))
    portable.update(schema_version=1, hdri=relative, initial_camera=camera_path,
        initial_camera_response=pose.get('camera_response'),
        lighting_saved_in_usd=True, qualification='not_run')
    portable['lighting_parameters']['hdri'] = relative
    original = portable['lighting_parameters']['radiance_evidence']
    # The package needs only its copied original, never a previous workspace.
    original.pop('path', None)
    original['portable_path'] = relative
    atomic_json(directory / 'render_configuration.json', portable)
    stage.GetPrimAtPath('/World').CreateAttribute('isaacmin:renderConfiguration',
        Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('./render_configuration.json'))
    return dict(hdri=relative, hdri_sha256=digest, initial_camera=camera_path,
        render_configuration_sha256=sha256_file(directory / 'render_configuration.json'),
        target_reopen='not_run')
