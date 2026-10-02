"""A recorded growing-season tint on original temperate grass leaf materials."""
from pathlib import Path
import re
from isaacmin.io import atomic_json, sha256_file


def author_living_grass(stage, scene):
    from pxr import Sdf
    scene = Path(scene)
    changed = []
    for module in sorted((scene / 'understory_material').glob('grass_medium_*/IsaacMinThinLeaf.mdl')):
        text = module.read_text()
        expression = re.search(r'    color col = (tex::lookup_color\([^\n]+);', text)
        if not expression:
            raise ValueError('Unknown original grass material graph')
        before = sha256_file(module)
        text = text[:expression.start()] + (
            '    // Growing-season pigment response; original scan, cutout and geometry retained.\n'
            '    color original_col = '+expression[1]+';\n'
            '    color col = math::min(original_col*color(0.78,1.16,0.66),color(0.85));'
        ) + text[expression.end():]
        # Donor MDLs are copied, never shared writable links with earlier worlds.
        temporary = module.with_suffix('.new')
        temporary.write_text(text); temporary.replace(module)
        changed.append(dict(module=str(module.relative_to(scene)), before_sha256=before,
            sha256=sha256_file(module)))
    if not changed:
        raise ValueError('Original meadow grass materials are missing')
    report = dict(recipe='growing_season_grass_v1', modules=changed,
        linear_rgb_multiplier=[0.78, 1.16, 0.66], maximum_diffuse_reflectance=0.85,
        scope='Original medium grass assets placed only in allowed temperate meadow/woodland biomes by surface_population; dry shrubs, snow, rock and water unchanged',
        provenance='Inferred seasonal pigment adjustment, not a measured leaf spectrum; all original texture bytes and geometry retained',
        qualification='not_run')
    atomic_json(scene / 'living_grass.json', report)
    stage.GetPrimAtPath('/World').CreateAttribute('isaacmin:livingGrassManifest',
        Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath('./living_grass.json'))
    return report
