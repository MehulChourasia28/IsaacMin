"""Explicit original-texture MDL candidates for the inspected Poly Haven pines."""
from pathlib import Path
import os
import re
import math
from isaacmin.io import sha256_file


_HEADER = '''mdl 1.7;
import ::df::*; import ::state::*; import ::math::*; import ::tex::*; import ::scene::*;
struct BoxSample { color c; float r; float3 delta; };
BoxSample box_sample(uniform texture_2d c, uniform texture_2d r, uniform texture_2d n,
    float3 p, float3 gn) = let {
    float3 sg = float3(gn.x < 0.0 ? -1.0 : 1.0, gn.y < 0.0 ? -1.0 : 1.0, gn.z < 0.0 ? -1.0 : 1.0);
    float3 aw = math::pow(math::abs(gn), float3(4.0));
    float3 w = aw / math::max(aw.x + aw.y + aw.z, 0.00001);
    float2 ux = float2(p.y*sg.x,p.z); float2 uy = float2(-p.x*sg.y,p.z); float2 uz = float2(p.x*sg.z,p.y);
    float3 nx = tex::lookup_float3(n,ux)*2.0-1.0;
    float3 ny = tex::lookup_float3(n,uy)*2.0-1.0;
    float3 nz = tex::lookup_float3(n,uz)*2.0-1.0;
    float3 dx = float3(sg.x*(nx.z-1.0),sg.x*nx.x,nx.y);
    float3 dy = float3(-sg.y*ny.x,sg.y*(ny.z-1.0),ny.y);
    float3 dz = float3(sg.z*nz.x,nz.y,sg.z*(nz.z-1.0));
} in BoxSample(tex::lookup_color(c,ux)*w.x + tex::lookup_color(c,uy)*w.y + tex::lookup_color(c,uz)*w.z,
    tex::lookup_float(r,ux)*w.x + tex::lookup_float(r,uy)*w.y + tex::lookup_float(r,uz)*w.z,
    dx*w.x + dy*w.y + dz*w.z);
'''


def principled_specular_ior(ior, specular_level):
    """Blender's normal-incidence adjustment, from Cycles svm/closure.h."""
    if not (1 <= ior <= 4 and 0 <= specular_level <= 1):
        raise ValueError('Unsupported provider dielectric parameters')
    r = math.sqrt(((ior - 1.) / (ior + 1.)) ** 2 * 2. * specular_level)
    return (1. + r) / (1. - r)


def provider_material(workspace, stage, object_path, directory, card, name, *,
                      uv_scale=(1., 1.), box_scale=(1.4, 1.4, .7),
                      ior=1.5, specular_level=.5, texture_prefix=None,
                      cutout=None, thin_leaf=None,leaf_shader=None,
                      transmission_factor=.25,albedo_value_scale=1.,
                      secondary_uv_prefix=None,secondary_uv_scale=(1.,1.),secondary_uv_rotation=0.):
    from pxr import UsdShade, Sdf
    if not re.fullmatch(r'[a-z0-9_]+', name):
        raise ValueError('Invalid provider material name')
    aid = card['asset_id']
    leaf = name.endswith('_twig')
    alpha_used=leaf if cutout is None else bool(cutout)
    thin=leaf if thin_leaf is None else bool(thin_leaf)
    if leaf_shader not in (None,'original_alpha_mix','energy_normalized_add') or not 0<=transmission_factor<=1 or not 0<albedo_value_scale<=2:
        raise ValueError('Unsupported original leaf shader parameters')
    if leaf_shader=='original_alpha_mix' and (not alpha_used or not thin):
        raise ValueError('Original leaf mix requires its actual cutout and thin-leaf surface')
    trunk = '_trunk_' in name
    stem = texture_prefix or (name if leaf or trunk else aid + '_bark')
    texture_dir = directory / 'textures'
    texture_dir.mkdir(exist_ok=True)
    copied = {}

    def texture(prefix, role, gamma):
        prefix_role = prefix + '_' + role + '_4k'
        rows = [f for f in card['files'] if Path(f['path']).stem == prefix_role
                and Path(f['path']).suffix in ('.png', '.jpg', '.exr')]
        if len(rows) != 1:
            raise ValueError('Original provider texture missing or ambiguous: ' + prefix_role)
        row = rows[0]
        source = Path(row['path'])
        filename = source.name
        if source.is_symlink() or sha256_file(source) != row['sha256']:
            raise ValueError('Original provider texture changed')
        target = texture_dir / filename
        if not target.exists():
            os.link(source, target)
        if sha256_file(target) != row['sha256']:
            raise ValueError('Copied provider texture mismatch')
        copied[filename] = dict(path='textures/' + filename, sha256=row['sha256'], role=role,
                                source_url=row['source_url'], color_space=gamma)
        return 'texture_2d("./textures/' + filename + '",tex::gamma_' + gamma + ')'

    col = texture(stem, 'diff', 'srgb')
    rough = texture(stem, 'rough', 'linear')
    normal = texture(stem, 'nor_gl', 'linear')
    alpha = 'tex::lookup_float(' + texture(stem, 'alpha', 'linear') + ',uv)' if alpha_used else '1.0'
    body = _HEADER + '''export material IsaacMinProviderPBR() = let {
    float3 uvw = state::texture_coordinate(0);
    float2 uv = scene::data_lookup_float2("st",float2(uvw.x,uvw.y))*float2(@@sx@@,@@sy@@);
    color original = tex::lookup_color(@@col@@,uv)*@@value_scale@@;
    float original_rough = tex::lookup_float(@@rough@@,uv);
    float3 n = math::normalize(state::normal());
    float3 tu = state::texture_tangent_u(0); float3 tv = state::texture_tangent_v(0);
    float3 t = math::normalize(tu-n*math::dot(n,tu));
    float3 b = math::normalize(tv-n*math::dot(n,tv)-t*math::dot(t,tv));
    float3 nt = tex::lookup_float3(@@normal@@,uv)*2.0-1.0;
    float3 original_normal = math::normalize(nt.x*t+nt.y*b+nt.z*n);
    @@blend@@
    float raw_alpha = math::clamp(@@alpha@@,0.0,1.0);
    float alpha = @@coverage@@;
    bsdf base = @@base@@;
    bsdf opaque = df::fresnel_layer(
        ior:@@specular_ior@@,weight:1.0,layer:df::microfacet_ggx_smith_bsdf(
            roughness_u:rough*rough,roughness_v:rough*rough,tint:color(1.0),mode:df::scatter_reflect),
        base:base,normal:normal);
    material_surface surface = material_surface(scattering:@@scattering@@);
} in material(thin_walled:@@thin@@,ior:color(@@ior@@),surface:surface,backface:surface,
    geometry:material_geometry(normal:normal,cutout_opacity:math::clamp(alpha,0.0,1.0)));
'''
    if trunk:
        bc = texture(aid + '_bark', 'diff', 'srgb')
        br = texture(aid + '_bark', 'rough', 'linear')
        bn = texture(aid + '_bark', 'nor_gl', 'linear')
        blend = '''float3 p = scene::data_lookup_float3("IsaacMinProviderPosition",float3(0.0))*float3(%s,%s,%s);
    float3 object_n = math::normalize(state::transform_normal(state::coordinate_internal,state::coordinate_object,n));
    BoxSample bark = box_sample(%s,%s,%s,p,object_n);
    float blend = math::clamp(scene::data_lookup_float("IsaacMinProviderBarkBlend",0.0),0.0,1.0);
    color col = original*(1.0-blend)+bark.c*blend;
    float rough = original_rough*(1.0-blend)+bark.r*blend;
    float3 bark_n = math::normalize(n+state::transform_normal(state::coordinate_object,state::coordinate_internal,bark.delta));
    float3 normal = math::normalize(original_normal*(1.0-blend)+bark_n*blend);''' % (*map(repr, box_scale), bc, br, bn)
    else:
        blend = 'color col = original; float rough = original_rough; float3 normal = original_normal;'
    if secondary_uv_prefix:
        if trunk:raise ValueError('UV blend and legacy BOX blend are mutually exclusive')
        bc=texture(secondary_uv_prefix,'diff','srgb');br=texture(secondary_uv_prefix,'rough','linear');bn=texture(secondary_uv_prefix,'nor_gl','linear')
        if not all(math.isfinite(float(v)) for v in (*secondary_uv_scale,secondary_uv_rotation)):
            raise ValueError('Secondary UV transform must be finite')
        sx,sy=map(float,secondary_uv_scale);c=math.cos(secondary_uv_rotation);s=math.sin(secondary_uv_rotation)
        blend=f'''float2 original_uv = scene::data_lookup_float2("st",float2(uvw.x,uvw.y));
    float2 uv2 = float2({c*sx}*original_uv.x-{s*sy}*original_uv.y,{s*sx}*original_uv.x+{c*sy}*original_uv.y);
    float blend = math::clamp(scene::data_lookup_float("IsaacMinProviderBarkBlend",0.0),0.0,1.0);
    color col = math::lerp(original,tex::lookup_color({bc},uv2),blend);
    float rough = math::lerp(original_rough,tex::lookup_float({br},uv2),blend);
    float3 nt2 = tex::lookup_float3({bn},uv2)*2.0-1.0;
    float3 normal2 = math::normalize(nt2.x*t+nt2.y*b+nt2.z*n);
    float3 normal = math::normalize(math::lerp(original_normal,normal2,blend));'''
    base = ('df::normalized_mix(df::bsdf_component[](df::bsdf_component(0.75,df::diffuse_reflection_bsdf(tint:col)),df::bsdf_component(0.25,df::diffuse_transmission_bsdf(tint:col))))'
            if thin else 'df::diffuse_reflection_bsdf(tint:col)')
    coverage='raw_alpha';scattering='opaque'
    if leaf_shader=='original_alpha_mix':
        # Original group: mix(Principled(alpha=a), Translucent, a*t).
        # Factoring transparency gives coverage a*(1+t-a*t), with the same
        # effective reflected/transmitted weights inside the covered portion.
        t=repr(float(transmission_factor));den='(1.0+'+t+'-raw_alpha*'+t+')'
        coverage='raw_alpha*'+den
        base='df::diffuse_reflection_bsdf(tint:col)'
        scattering='df::normalized_mix(df::bsdf_component[](df::bsdf_component(1.0-'+t+'/'+den+',opaque),df::bsdf_component('+t+'/'+den+',df::diffuse_transmission_bsdf(tint:col))))'
    elif leaf_shader=='energy_normalized_add':
        if not alpha_used or not thin:raise ValueError('Thin leaf candidate needs original alpha')
        fraction=transmission_factor/(1+transmission_factor)
        base='df::diffuse_reflection_bsdf(tint:col)'
        scattering=f'df::normalized_mix(df::bsdf_component[](df::bsdf_component({1-fraction},opaque),df::bsdf_component({fraction},df::diffuse_transmission_bsdf(tint:col))))'
    for token, value in dict(sx=repr(float(uv_scale[0])), sy=repr(float(uv_scale[1])),
        col=col, rough=rough, normal=normal, alpha=alpha, blend=blend, base=base,
        value_scale=repr(float(albedo_value_scale)),coverage=coverage,scattering=scattering,
        thin='true' if thin else 'false', ior=repr(float(ior)),
        specular_ior=repr(principled_specular_ior(ior,specular_level))).items():
        body = body.replace('@@' + token + '@@', value)
    if '@@' in body:
        raise RuntimeError('Incomplete native provider material')
    mdl = directory / (name + '.mdl')
    if mdl.exists() and mdl.read_text() != body:
        raise RuntimeError('Conflicting provider material candidates')
    mdl.write_text(body)
    material = UsdShade.Material.Define(stage, object_path.AppendPath('Looks/' + name))
    shader = UsdShade.Shader.Define(stage, material.GetPath().AppendChild('Shader'))
    shader.CreateImplementationSourceAttr().Set(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset(Sdf.AssetPath('./' + mdl.name), 'mdl')
    shader.SetSourceAssetSubIdentifier('IsaacMinProviderPBR', 'mdl')
    shader.CreateOutput('out', Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput('mdl').ConnectToSource(shader.ConnectableAPI(), 'out')
    material.GetPrim().CreateAttribute('isaacmin:originalTextures', Sdf.ValueTypeNames.AssetArray).Set(
        [Sdf.AssetPath('./' + row['path']) for row in copied.values()])
    return material, dict(name=name, module=mdl.name, module_sha256=sha256_file(mdl),
        original_textures=list(copied.values()), uv_scale=list(uv_scale),
        original_ior=ior, original_specular_level=specular_level,
        translated_specular_ior=principled_specular_ior(ior,specular_level),
        original_trunk_vertex_mask=bool(trunk or secondary_uv_prefix), box_scale=list(box_scale) if trunk else None,
        secondary_uv=dict(prefix=secondary_uv_prefix,scale=list(secondary_uv_scale),rotation=secondary_uv_rotation) if secondary_uv_prefix else None,
        optical_recipe=('original alpha/translucency mix and artist value scale; native dielectric Principled approximation'
            if leaf_shader=='original_alpha_mix' else 'energy-normalized original leaf maps with view-dependent colour removed'
            if leaf_shader=='energy_normalized_add' else 'explicit dielectric PBR; needle reflection/transmission75/25 inferred, not measured'),
        cutout_from_original_alpha=alpha_used,thin_leaf=thin,
        original_leaf_group='original_alpha_mix' if leaf_shader=='original_alpha_mix' else None,
        native_leaf_recipe=leaf_shader,transmission_factor=transmission_factor,
        original_albedo_value_scale=albedo_value_scale,
        opacity='original filtered alpha coverage; no extra threshold on subpixel needles',
        translation_limits=['native MDL projection blend differs from Blender BOX blend' if trunk else
            'native dielectric BRDF needs final-renderer comparison with original Principled',
            'provider shader displacement not executed; full LOD0 geometry and original normals retained'],
        qualification='not_run')
