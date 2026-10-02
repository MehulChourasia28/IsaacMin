"""Explicit thin-leaf target material candidate; no automatic appearance claim."""
from pathlib import Path
import hashlib
import json
import math
import os
import re
import shutil


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _native_texture(shader, socket, role, scene):
    """Reject graph semantics this narrow translator cannot preserve exactly."""
    from pxr import UsdShade
    value = shader.GetInput(socket)
    connection = value.GetConnectedSource() if value else None
    if not connection:
        return None
    texture = UsdShade.Shader(connection[0])
    if texture.GetIdAttr().Get() != 'UsdUVTexture':
        raise ValueError('Unsupported leaf texture graph: ' + role)
    output = str(connection[1])
    if output not in ({'rgb'} if role in ('base_color', 'normal') else {'r', 'a'} if role == 'opacity' else {'r'}):
        raise ValueError('Unsupported leaf texture output: ' + role)
    expected_space = 'sRGB' if role == 'base_color' else 'raw'
    if texture.GetInput('sourceColorSpace').Get() != expected_space:
        raise ValueError('Leaf texture color space differs from explicit contract: ' + role)
    for axis in ('wrapS', 'wrapT'):
        if texture.GetInput(axis).Get() != 'repeat':
            raise ValueError('Unsupported leaf texture wrap mode: ' + role)
    for field, expected in (('scale', (2.,)*4 if role == 'normal' else (1.,)*4),
                            ('bias', (-1.,)*4 if role == 'normal' else (0.,)*4)):
        attribute = texture.GetInput(field)
        actual = attribute.Get() if attribute else None
        if actual is None:
            actual = (1.,)*4 if field == 'scale' else (0.,)*4
        if tuple(actual) != expected:
            raise ValueError('Unsupported leaf texture scale/bias: ' + role)
    coordinate = texture.GetInput('st').GetConnectedSource()
    reader = UsdShade.Shader(coordinate[0]) if coordinate else None
    if not reader or reader.GetIdAttr().Get() != 'UsdPrimvarReader_float2' or reader.GetInput('varname').Get() != 'st':
        raise ValueError('Leaf texture requires unchanged native st UV coordinates')
    asset = texture.GetInput('file').Get()
    if not asset or Path(asset.path).is_absolute():
        raise ValueError('Leaf texture requires an explicit relative source asset')
    file = Path(asset.resolvedPath).resolve() if asset.resolvedPath else (scene.parent / asset.path).resolve()
    if not file.is_relative_to(scene.parent) or not file.is_file():
        raise ValueError('Leaf texture is outside the closed native scene')
    return {'source': file, 'sha256': _sha(file), 'output': output,
            'source_color_space': expected_space, 'wrap': 'repeat', 'uv_primvar': 'st'}


def bind_thin_leaves(workspace, scene_path, material_names, *, transmission_fraction=0.35):
    """Retain original PreviewSurface and add a closed native MDL leaf surface.

    Names must explicitly identify leaf materials, never bark or terrain. The
    caller must rebuild the native dependency closure before rendering. The
    reflect/transmit split is an inferred botanical candidate, not measured
    optical data. Nothing is selected in production by this diagnostic helper.
    """
    from pxr import Usd, UsdShade, Sdf
    workspace = Path(workspace).resolve()
    scene = Path(scene_path).resolve()
    names = set(material_names)
    # Blender suffixes materials imported from a second independent .blend;
    # its native USD writer converts '.001' to '_001'. Preserve those actual
    # material identities instead of silently omitting the younger prototype.
    if not names or any(not re.fullmatch(r'[A-Za-z0-9_]+_Leaf(?:_[0-9]{3})?', name) for name in names):
        raise ValueError('Explicit generated leaf material names are required')
    if not math.isfinite(transmission_fraction) or not 0 <= transmission_fraction <= 1:
        raise ValueError('Leaf tissue fraction must be finite and bounded')
    directory = scene.parent / 'foliage_material'
    if directory.exists():
        raise ValueError('Leaf material candidate output must be immutable')
    stage = Usd.Stage.Open(str(scene))
    if not stage:
        raise ValueError('Cannot open native canopy scene')
    materials = [UsdShade.Material(p) for p in stage.Traverse()
                 if p.IsA(UsdShade.Material) and p.GetName() in names]
    if {m.GetPrim().GetName() for m in materials} != names or len(materials) != len(names):
        raise ValueError('Actual USD leaf materials differ from requested names')
    template_path = workspace / 'recipes/materials/thin_leaf.mdl.in'
    template = template_path.read_text()
    inspected = []
    for material in materials:
        surface = material.GetSurfaceOutput().GetConnectedSource()
        shader = UsdShade.Shader(surface[0]) if surface else None
        if not shader or shader.GetIdAttr().Get() != 'UsdPreviewSurface':
            raise ValueError('Expected actual Blender PreviewSurface leaf material')
        channels = {}
        constants = {}
        for role, socket in [('base_color', 'diffuseColor'), ('roughness', 'roughness'),
                             ('normal', 'normal'), ('opacity', 'opacity')]:
            channel = _native_texture(shader, socket, role, scene)
            if channel:
                channels[role] = channel
            elif role == 'roughness':
                value = shader.GetInput(socket).Get()
                if value is None or not math.isfinite(float(value)) or not 0 <= value <= 1:
                    raise ValueError('Leaf needs an explicit finite roughness value or original texture')
                constants['roughness_value'] = float(value)
            elif role == 'normal':
                # No generated bitmap: an absent normal channel retains the
                # actual interpolated mesh normal. Reject an authored tilt.
                value = shader.GetInput(socket).Get() if shader.GetInput(socket) else None
                if value is not None and tuple(value) != (0., 0., 1.):
                    raise ValueError('Unsupported authored constant leaf normal')
                constants['normal'] = 'actual_native_geometry_normal; no bitmap'
            else:
                raise ValueError('Leaf requires its original color and opacity texture: ' + role)
        ior = shader.GetInput('ior').Get()
        if ior is None or not math.isfinite(float(ior)) or not 1 < ior <= 3:
            raise ValueError('Leaf requires an explicit finite dielectric IOR')
        constants['ior_value'] = float(ior)
        threshold = shader.GetInput('opacityThreshold').Get()
        if threshold != 0.5:
            raise ValueError('Leaf export must retain its declared 0.5 cutout threshold')
        inspected.append((material, channels, constants))
    before = _sha(scene)
    directory.mkdir()
    records = []
    for material, channels, constants in inspected:
        name = material.GetPrim().GetName()
        library = directory / name
        (library / 'textures').mkdir(parents=True)
        module = template
        files = []
        for role, item in channels.items():
            target = library / 'textures' / (item['sha256'] + item['source'].suffix)
            if not target.exists():
                shutil.copy2(item['source'], target)
            if _sha(target) != item['sha256']:
                raise RuntimeError('Original botanical texture bytes changed')
            module = module.replace('@@' + role + '@@', './textures/' + target.name)
            files.append({'role': role, 'path': str(target.relative_to(scene.parent)),
                          'source': str(item['source'].relative_to(scene.parent)),
                          'sha256': item['sha256'], 'output': item['output'],
                          'source_color_space': item['source_color_space'], 'uv_primvar': 'st', 'wrap': 'repeat'})
        roughness = ('tex::lookup_float(texture_2d("@@roughness@@",tex::gamma_linear),uv)'
                     if 'roughness' in channels else repr(constants['roughness_value']))
        alpha = ('tex::lookup_float4(texture_2d("@@opacity@@",tex::gamma_linear),uv).w'
                 if channels['opacity']['output'] == 'a' else
                 'tex::lookup_float(texture_2d("@@opacity@@",tex::gamma_linear),uv)')
        normal = ('float3 n = math::normalize(state::normal());\n'
                  '    float3 tu = state::texture_tangent_u(0);\n'
                  '    float3 tv = state::texture_tangent_v(0);\n'
                  '    float3 tangent = math::normalize(tu-n*math::dot(n,tu));\n'
                  '    float3 bitangent = math::normalize(tv-n*math::dot(n,tv)-tangent*math::dot(tangent,tv));\n'
                  '    float3 nt = tex::lookup_float3(texture_2d("@@normal@@",tex::gamma_linear),uv)*2.0-1.0;\n'
                  '    float3 normal = math::normalize(nt.x*tangent+nt.y*bitangent+nt.z*n);'
                  if 'normal' in channels else 'float3 normal = math::normalize(state::normal());')
        module = module.replace('@@roughness_expression@@', roughness).replace('@@opacity_expression@@', alpha)
        module = module.replace('@@normal_expression@@', normal).replace('@@ior@@', repr(constants['ior_value']))
        module = module.replace('@@transmission_fraction@@', repr(float(transmission_fraction)))
        module = module.replace('@@reflection_fraction@@', repr(float(1-transmission_fraction)))
        # Expressions are inserted after copying: substitute their map tokens
        # using the same already verified original bytes, without making maps.
        for item in files:
            module = module.replace('@@' + item['role'] + '@@', './textures/' + Path(item['path']).name)
        if '@@' in module:
            raise RuntimeError('Leaf material template is incomplete')
        mdl = library / 'IsaacMinThinLeaf.mdl'
        mdl.write_text(module)
        shader = UsdShade.Shader.Define(stage, material.GetPath().AppendChild('IsaacMinThinLeaf'))
        shader.CreateImplementationSourceAttr().Set(UsdShade.Tokens.sourceAsset)
        relative = lambda path: Sdf.AssetPath('./' + Path(os.path.relpath(path, scene.parent)).as_posix())
        shader.SetSourceAsset(relative(mdl), 'mdl')
        shader.SetSourceAssetSubIdentifier('IsaacMinThinLeaf', 'mdl')
        shader.CreateOutput('out', Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput('mdl').ConnectToSource(shader.ConnectableAPI(), 'out')
        material.GetPrim().CreateAttribute('isaacmin:leafTextures', Sdf.ValueTypeNames.AssetArray,
            custom=True).Set([relative(scene.parent / item['path']) for item in files])
        records.append({'material': str(material.GetPath()), 'module': str(mdl.relative_to(scene.parent)),
                        'module_sha256': _sha(mdl), 'channels': files, 'constants': constants,
                        'scalar_provenance': 'Read from actual exported PreviewSurface; inferred shader candidates, not measurements',
                        'normal_model': 'original_OpenGL_texture_in_orthonormalized_native_UV_frame' if 'normal' in channels else 'native_geometry_only'})
    report = {'schema_version': 1, 'status': 'candidate_authored', 'qualification': 'not_run',
              'recipe': 'original_textures_thin_leaf_v1', 'scene_before_sha256': before,
              'template_sha256': _sha(template_path), 'producer_sha256': _sha(__file__),
              'materials': records, 'alpha_threshold': 0.5,
              'tissue_reflection_fraction': 1-transmission_fraction, 'tissue_transmission_fraction': transmission_fraction,
              'control_role': 'candidate65_35' if transmission_fraction == .35 else 'explicit_optical_control_not_production_recipe',
              'normal_convention': 'Original OpenGL texture in exported UV frame when available; otherwise actual mesh normal',
              'optical_parameters': 'inferred candidate; not a measured leaf spectrum',
              'required_target_checks': ['cutout', 'front_and_back', 'backlight', 'grazing_normal',
                                         'continuous_motion', 'held_out_canopy_views']}
    manifest = directory / 'material_manifest.json'
    manifest.write_text(json.dumps(report, indent=2) + '\n')
    for material, _, _ in inspected:
        material.GetPrim().CreateAttribute('isaacmin:leafManifest', Sdf.ValueTypeNames.Asset,
            custom=True).Set(Sdf.AssetPath('./foliage_material/material_manifest.json'))
    stage.GetRootLayer().Save()
    return report
