#!/usr/bin/env python3
"""Original leaf photographs on explicit optical measurement cards, not scenery."""
import argparse
import shutil
from pathlib import Path

from pxr import Usd, UsdGeom, UsdShade, Sdf
from isaacmin.io import atomic_json, read_json, sha256_file
from isaacmin.packaging import ascii_dependency_closure
from isaacmin.assembly.foliage_mdl import bind_thin_leaves


def material(stage, path, channels, scalar_roughness=.55):
    result = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, path + '/Preview')
    shader.CreateIdAttr('UsdPreviewSurface')
    shader.CreateInput('ior', Sdf.ValueTypeNames.Float).Set(1.42)
    shader.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(scalar_roughness)
    shader.CreateInput('opacityThreshold', Sdf.ValueTypeNames.Float).Set(.5)
    shader.CreateOutput('surface', Sdf.ValueTypeNames.Token)
    uv = UsdShade.Shader.Define(stage, path + '/UV')
    uv.CreateIdAttr('UsdPrimvarReader_float2')
    uv.CreateInput('varname', Sdf.ValueTypeNames.Token).Set('st')
    uv.CreateOutput('result', Sdf.ValueTypeNames.Float2)
    for role, file in channels.items():
        socket, output, value_type = {
            'base_color': ('diffuseColor', 'rgb', Sdf.ValueTypeNames.Color3f),
            'roughness': ('roughness', 'r', Sdf.ValueTypeNames.Float),
            'opacity': ('opacity', 'r', Sdf.ValueTypeNames.Float),
            'normal': ('normal', 'rgb', Sdf.ValueTypeNames.Normal3f),
        }[role]
        tex = UsdShade.Shader.Define(stage, path + '/' + role)
        tex.CreateIdAttr('UsdUVTexture')
        tex.CreateInput('file', Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(file))
        tex.CreateInput('sourceColorSpace', Sdf.ValueTypeNames.Token).Set('sRGB' if role == 'base_color' else 'raw')
        for axis in ('wrapS', 'wrapT'):
            tex.CreateInput(axis, Sdf.ValueTypeNames.Token).Set('repeat')
        tex.CreateInput('scale', Sdf.ValueTypeNames.Float4).Set((2.,) * 4 if role == 'normal' else (1.,) * 4)
        tex.CreateInput('bias', Sdf.ValueTypeNames.Float4).Set((-1.,) * 4 if role == 'normal' else (0.,) * 4)
        tex.CreateInput('st', Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(), 'result')
        tex.CreateOutput(output, Sdf.ValueTypeNames.Float3 if output == 'rgb' else Sdf.ValueTypeNames.Float)
        shader.CreateInput(socket, value_type).ConnectToSource(tex.ConnectableAPI(), output)
    result.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
    return result


def prepare(root, output):
    if output.exists():
        raise ValueError('Leaf optical tests use immutable candidate directories')
    output.mkdir(parents=True)
    oak = root / 'assets/procedural/oak/maps'
    birch = root / 'assets/procedural_candidates/birch_originals_v1/maps'
    maps = {
        'oak': {role: oak / ('oak_leaf_' + role + '.png') for role in ('base_color', 'roughness', 'normal', 'opacity')},
        'birch': {role: birch / ('birch_leaf_00002_' + role + '.jpg') for role in ('base_color', 'opacity')},
    }
    scenes = []
    for recipe in ('preview', 'opaque_control', 'thin_candidate'):
        directory = output / recipe
        textures = directory / 'textures'
        textures.mkdir(parents=True)
        scene = directory / 'world.usda'
        stage = Usd.Stage.CreateNew(str(scene))
        stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, '/World').GetPrim())
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1)
        ground = UsdGeom.Mesh.Define(stage, '/World/Terrain_FinalGround')
        ground.CreatePointsAttr([(-10, -10, 0), (10, -10, 0), (10, 10, 0), (-10, 10, 0)])
        ground.CreateFaceVertexCountsAttr([3, 3])
        ground.CreateFaceVertexIndicesAttr([0, 1, 2, 0, 2, 3])
        ground.CreateSubdivisionSchemeAttr('none')
        ground_material = UsdShade.Material.Define(stage, '/World/Materials/DiagnosticGround')
        bsdf = UsdShade.Shader.Define(stage, '/World/Materials/DiagnosticGround/Preview')
        bsdf.CreateIdAttr('UsdPreviewSurface')
        bsdf.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set((.18, .18, .18))
        bsdf.CreateInput('roughness', Sdf.ValueTypeNames.Float).Set(.8)
        bsdf.CreateOutput('surface', Sdf.ValueTypeNames.Token)
        ground_material.CreateSurfaceOutput().ConnectToSource(bsdf.ConnectableAPI(), 'surface')
        UsdShade.MaterialBindingAPI.Apply(ground.GetPrim()).Bind(ground_material)
        (directory / 'final_ground.obj').write_text('o Terrain_FinalGround\nv -10 -10 0\nv 10 -10 0\nv 10 10 0\nv -10 10 0\nf 1 2 3\nf 1 3 4\n')
        poses, inputs = [], []
        for index, (species, channels) in enumerate(maps.items()):
            copied = {}
            for role, path in channels.items():
                digest = sha256_file(path)
                target = textures / (digest + path.suffix)
                shutil.copy2(path, target)
                if sha256_file(target) != digest:
                    raise ValueError('Optical probe changed original photograph bytes')
                copied[role] = './textures/' + target.name
                inputs.append({'species': species, 'role': role, 'original_path': str(path), 'sha256': digest})
            x = float(index)
            # This square is an explicitly labelled material measurement card.
            # It is never included in a generated forest or real-map evidence.
            card = UsdGeom.Mesh.Define(stage, '/World/' + species + '_MeasurementCard')
            card.CreatePointsAttr([(x-.05, 0, .55), (x+.05, 0, .55), (x+.05, 0, .65), (x-.05, 0, .65)])
            card.CreateFaceVertexCountsAttr([4])
            card.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
            card.CreateSubdivisionSchemeAttr('none')
            card.CreateDoubleSidedAttr(True)
            st = UsdGeom.PrimvarsAPI(card).CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, 'faceVarying')
            st.Set([(0, 0), (1, 0), (1, 1), (0, 1)])
            leaf = material(stage, '/World/Materials/' + species + '_Leaf', copied)
            UsdShade.MaterialBindingAPI.Apply(card.GetPrim()).Bind(leaf)
            for name, offset in [('front', (0, -.12, 0)), ('back', (0, .12, 0)), ('grazing', (.10, -.055, 0))]:
                poses.append({'kind': species + '_' + name, 'position': [x + offset[0], offset[1], .6],
                              'look_at': [x, 0, .6], 'ground_clearance_m': .6})
        stage.GetRootLayer().Save()
        if recipe != 'preview':
            bind_thin_leaves(root, scene, ['oak_Leaf', 'birch_Leaf'],
                transmission_fraction=0 if recipe == 'opaque_control' else .35)
        closure = ascii_dependency_closure(scene)
        if closure['status'] != 'pass':
            raise ValueError('Leaf diagnostic dependency closure failed: ' + str(closure))
        closure['root_sha256'] = sha256_file(scene)
        atomic_json(directory / 'native_dependency_closure.json', closure)
        # Keep the required texture credit with each closed diagnostic scene.
        shutil.copy2(birch / 'CREDITS.txt', directory / 'CREDITS.txt')
        scenes.append({'recipe': recipe, 'scene': str(scene), 'poses': poses, 'inputs': inputs})
    record = {'scope': 'actual original photographs on diagnostic material cards; not Minecraft world or botanical geometry',
        'appearance_qualification': 'not_run', 'cards_are_scene_assets': False,
        'lighting_parameters': {'dome_intensity': 30., 'sun_intensity': 3000.},
        'contact_probes': [{'position': [2, 2, 1.], 'ground_z': 0}],
        'renderer_recipe': 'pathtracing_1024', 'scenes': scenes,
        'producer_sha256': sha256_file(Path(__file__))}
    atomic_json(output / 'protocol.json', record)
    print({'status': 'prepared_unrendered', 'output': str(output), 'scenes': len(scenes)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(Path(__file__).resolve().parents[1], args.output.resolve())
