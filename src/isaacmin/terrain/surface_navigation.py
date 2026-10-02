"""Source-derived outdoor terrain under the user's explicit surface-only scope.

This path has one supporting elevation at each covered XY. It deliberately does
not preserve underground topology. All operations use metric source coordinates;
the source save and previous volumetric builds are never modified.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib

import numpy as np
from scipy import ndimage

from isaacmin.io import atomic_json, read_json, sha256_file, utc_now
from isaacmin.adapters.workers import refine_heightfield
from isaacmin.terrain.baseline import priority_drainage


@dataclass(frozen=True)
class SurfaceRecipe:
    spacing_m: float = 0.125
    context_m: int = 64
    narrow_depression_radius_m: int = 8
    dequantization_sigma_m: float = 3.0
    macro_correction_sigma_m: float = 12.0
    erosion_m: float = 0.18
    maximum_erosion_m: float = 0.24
    scan_displacement_amplitude_m: float = 0.02
    sedimentary_cliff_sigma_m: float = 1.6
    badlands_depression_radius_m: int = 8
    badlands_erosion_m: float = 1.25
    badlands_maximum_erosion_m: float = 1.8
    contour_relaxation_seconds: float = 6.0
    ice_dequantization_sigma_m: float = 0.8
    ice_tip_opening_limit_m: float = 0.5


def relax_steep_contours(height, duration):
    """Remove lateral voxel scallops without diffusing across a cliff face.

    Curvature flow acts along elevation contours. The bounded explicit stencil
    smooths small plan-view corners; unlike an isotropic height blur it does not
    spread a straight cliff into its valley. Native drainage erosion still
    follows this source reconstruction step.
    """
    value=np.array(height,dtype=np.float64,copy=True)
    if duration<0 or not np.isfinite(duration):raise ValueError('Invalid contour duration')
    steps=max(1,int(np.ceil(duration/.15)));dt=duration/steps
    for _ in range(steps):
        p=np.pad(value,1,mode='edge')
        dx=(p[1:-1,2:]-p[1:-1,:-2])*.5
        dz=(p[2:,1:-1]-p[:-2,1:-1])*.5
        dxx=p[1:-1,2:]+p[1:-1,:-2]-2*value
        dzz=p[2:,1:-1]+p[:-2,1:-1]-2*value
        dxz=(p[2:,2:]-p[2:,:-2]-p[:-2,2:]+p[:-2,:-2])*.25
        norm=dx*dx+dz*dz
        tangential=(dz*dz*dxx-2*dx*dz*dxz+dx*dx*dzz)/(norm+1e-12)
        steep=np.clip((np.sqrt(norm)-.7)/.7,0.,1.)
        # Bound each update by the local range, including near saddle points.
        updated=value+dt*steep*tangential
        value=np.clip(updated,ndimage.minimum_filter(value,3,mode='nearest'),
                     ndimage.maximum_filter(value,3,mode='nearest'))
    return value


def source_fields(path):
    """Accept exact 1m macro extraction or the original exact region surface IR."""
    with np.load(path, allow_pickle=False) as data:
        fields = {k: data[k] for k in data.files}
    if float(fields.get('sample_spacing_m', 1.0)) != 1:
        raise ValueError('Outdoor construction requires every source column at 1m')
    if float(fields.get('sample_offset_m', .5)) != .5:
        raise ValueError('Expected exact block-centre samples')
    if 'substrate' not in fields:
        fields['substrate'] = fields['block_names'][fields['substrate_id']]
    if 'block_names' not in fields:
        fields['block_names'], inverse = np.unique(fields['substrate'], return_inverse=True)
        fields['substrate_id'] = inverse.reshape(fields['height'].shape)
    return fields


def source_bank_floor(fields, wet):
    """Keep the immediate dry rim above its source water level.

    Removing an overhang can expose a dry depression beside a lake. A bounded
    bank constraint closes that rim without filling the broad valley beyond it.
    """
    floor=np.full(wet.shape,-np.inf,np.float64)
    if wet.any():
        distance,nearest=ndimage.distance_transform_edt(~wet,return_indices=True)
        level=fields['water_height'][tuple(nearest)]
        # A waterfall can border lower dry ground. It is not a bank to raise
        # into a dam; only preserve rims that were already above source water.
        bank=(~wet)&(distance<=1.5)&(fields['height']>=level)
        floor[bank]=level[bank]+.04
    return floor


def source_frozen_water(fields):
    """Observed thin ice over water, separate from packed-ice spires and soil."""
    names=fields.get('substrate')
    if names is None:names=fields['block_names'][fields['substrate_id']]
    thin=np.isin(names,['minecraft:ice','minecraft:frosted_ice'])
    h=fields.get('source_surface_height',fields['height'])
    gap=h-fields['water_height']
    water=fields.get('source_water_validity',fields['water_validity'])
    sheet=thin&water&(gap>=0)&(gap<=1.01)
    if sheet.any():
        distance,nearest=ndimage.distance_transform_edt(~sheet,return_indices=True)
        # A decoder may finish before reaching the next section below an ice
        # column. Adjacent observed thin ice at the exact same level is coherent.
        sheet|=thin&~water&(distance<=2)&(np.abs(h-h[tuple(nearest)])<.01)
    return sheet


def reconstruct_surface(fields, recipe=SurfaceRecipe()):
    """Fill narrow cuts; dequantize locally without removing regional relief.

    The closing's finite support cannot bridge a broad valley. Low-frequency
    correction restores relief removed by the local dequantization kernel. Water
    columns retain their original beds, with a continuous transition at banks.
    Actual drainage-driven native erosion follows this baseline separately.
    """
    h = np.asarray(fields['height'], np.float64)
    valid = np.asarray(fields['validity'], bool)
    if h.ndim != 2 or h.shape != valid.shape or not valid.all() or not np.isfinite(h).all():
        raise ValueError('This candidate needs complete known context; missing chunks are never filled')
    if (recipe.spacing_m <= 0 or abs(round(1/recipe.spacing_m)*recipe.spacing_m-1) > 1e-12
            or recipe.context_m < 2*recipe.narrow_depression_radius_m+8):
        raise ValueError('Metric sampling or context is insufficient for reconstruction')
    source_wet = np.asarray(fields['water_validity'], bool) & (fields['water_height'] >= h)
    closed = ndimage.grey_closing(h, size=2*recipe.narrow_depression_radius_m+1, mode='nearest')
    badland=np.isin(fields['biome'],['minecraft:badlands','minecraft:eroded_badlands','minecraft:wooded_badlands'])
    badland_weight=ndimage.gaussian_filter(badland.astype(np.float64),2.,mode='nearest')
    if badland.any():
        # Keep above-ground buttes and their erosion valleys distinct. A broad
        # generic closing otherwise joins adjacent mesas into a rounded mound.
        narrow=ndimage.grey_closing(h,size=2*recipe.badlands_depression_radius_m+1,mode='nearest')
        closed=closed*(1-badland_weight)+narrow*badland_weight
    # Visible source water is authoritative even inside a narrow valley. The
    # user's exclusion of underground/ravines does not authorize deleting an
    # exposed river or lake. Only dry narrow cuts may be reconstructed.
    wet=source_wet
    # The collar around preserved surface water avoids a hard one-column bank.
    dry_distance = ndimage.distance_transform_edt(~wet) if wet.any() else np.full(h.shape, 10.)
    water_fade = np.clip(dry_distance/4., 0, 1)
    water_fade = water_fade*water_fade*(3-2*water_fade)
    filled = h + np.maximum(closed-h, 0)*water_fade
    names=np.asarray(fields.get('substrate',np.full(h.shape,'')))
    frozen=np.isin(names,['minecraft:ice','minecraft:packed_ice','minecraft:blue_ice','minecraft:frosted_ice'])
    # Dry-cut filling must not join adjacent above-ground ice spires. Retain
    # their observed relief and the snow basins between them before dequantizing.
    ice_relief=frozen|(fields['biome']=='minecraft:ice_spikes')
    filled[ice_relief]=h[ice_relief]
    # Retain the narrow tips of observed ice spires. Opening only removes a
    # bounded sub-block step, and the ice kernel is separate from soil smoothing.
    ice_source=(np.maximum(ndimage.grey_opening(filled,size=3,mode='nearest'),filled-recipe.ice_tip_opening_limit_m)
                if frozen.any() else filled)
    ice_tip_lowering=filled-ice_source
    filled[frozen]=ice_source[frozen]
    smooth = ndimage.gaussian_filter(filled, recipe.dequantization_sigma_m, mode='nearest')
    smooth += ndimage.gaussian_filter(filled-smooth, recipe.macro_correction_sigma_m, mode='nearest')
    if badland.any():
        # Dequantize gentle slopes fully, while preserving steep cliff breaks.
        # Native erosion below supplies the drainage-aligned weathering.
        gz,gx=np.gradient(ndimage.gaussian_filter(filled,2.,mode='nearest'))
        steep=np.clip((np.hypot(gx,gz)-.55)/.65,0,1)
        steep=steep*steep*(3-2*steep)*badland_weight
        cliffs=ndimage.gaussian_filter(filled,recipe.sedimentary_cliff_sigma_m,mode='nearest')
        cliffs+=ndimage.gaussian_filter(filled-cliffs,recipe.macro_correction_sigma_m,mode='nearest')
        smooth=smooth*(1-steep)+cliffs*steep
    smooth=relax_steep_contours(smooth,recipe.contour_relaxation_seconds)
    if frozen.any():
        ice=ndimage.gaussian_filter(filled,recipe.ice_dequantization_sigma_m,mode='nearest')
        ice+=ndimage.gaussian_filter(filled-ice,recipe.macro_correction_sigma_m,mode='nearest')
        # A one-column observed tip must receive the ice reconstruction fully;
        # blurring the mask alone leaves it mostly governed by soil smoothing.
        boundary=np.maximum(frozen,ndimage.gaussian_filter(frozen.astype(float),1.,mode='nearest'))
        tips=frozen&(h==ndimage.maximum_filter(h,3,mode='nearest'))
        tips&=(h-ndimage.grey_opening(h,size=3,mode='nearest'))>2.
        ice[tips]=np.maximum(ice[tips],h[tips]-recipe.ice_tip_opening_limit_m)
        smooth=smooth*(1-boundary)+ice*boundary
    smooth = h + (smooth-h)*water_fade
    smooth[wet] = h[wet]
    bank_floor=source_bank_floor(fields,wet)
    bank_raise=np.maximum(bank_floor-smooth,0)
    smooth=np.maximum(smooth,bank_floor)
    sheet=source_frozen_water(fields) if 'substrate' in fields or 'substrate_id' in fields else np.zeros(h.shape,bool)
    smooth[sheet]=h[sheet]+.04
    return smooth.astype(np.float32), wet, {
        'method': 'metric_narrow_cut_closing_and_macro_restored_dequantization',
        'role': 'baseline_before_actual_HighMap',
        'intentional_fill_columns': int(np.count_nonzero(filled-h > .01)),
        'maximum_intentional_fill_m': float(np.max(filled-h)),
        'source_height_range_m': [float(h.min()), float(h.max())],
        'baseline_height_range_m': [float(smooth.min()), float(smooth.max())],
        'water_bed_columns_preserved': int(wet.sum()),
        'frozen_water_columns_preserved':int(sheet.sum()),
        'frozen_water_freeboard_m':.04,
        'intentional_narrow_cut_water_columns_removed':0,
        'surface_water_policy':'all exposed source water and beds retained; no narrow-cut deletion',
        'water_bank_constraint':dict(method='immediate dry rim above nearest visible source water',
            support_radius_m=1.5,minimum_dry_rim_above_water_m=.04,
            raised_columns=int(np.count_nonzero(bank_raise)),maximum_raise_m=float(bank_raise.max())),
        'underground_fidelity': 'not_applicable_user_authorized_surface_only',
        'mountain_policy': 'no global height normalization or elevation flattening',
        'badlands_columns':int(badland.sum()),
        'badlands_method':'narrow-cut reconstruction, steep cliff retention, native drainage erosion with material resistance',
        'contour_reconstruction':dict(method='bounded_tangential_curvature_flow',
            duration=recipe.contour_relaxation_seconds,straight_cliff_cross_slope_diffusion=False),
        'ice_reconstruction':dict(columns=int(frozen.sum()),sigma_m=recipe.ice_dequantization_sigma_m,
            exterior_ice_uses_water_constrained_narrow_cut_reconstruction=False,
            source_ice_and_inter_spire_snow_relief_retained_before_dequantization=True,
            observed_local_tips_restored_with_opening_limit=True,
            isolated_voxel_needle_opening_radius_m=1,
            tip_opening_limit_m=recipe.ice_tip_opening_limit_m,
            maximum_ice_tip_reconstruction_m=float(np.max(ice_tip_lowering[frozen])) if frozen.any() else 0.),
    }


def sample_raster(field, source_x, source_z, minimum_xz, *, order=1):
    x = np.asarray(source_x)-minimum_xz[0]-.5
    z = np.asarray(source_z)-minimum_xz[1]-.5
    return ndimage.map_coordinates(field, [z, x], order=order, mode='nearest', prefilter=order > 1)


def sample_surface_grid(height, source_x, source_z, minimum_xz):
    """Continuously interpolate relief without ringing above ice or bank tops.

    Ordinary cubic B-splines overshoot a narrow lake-ice plateau over a deep
    seabed, producing false spires above the original ice. Separable monotone
    cubic Hermite interpolation preserves sample values and local flat limits.
    Context surrounds every output point; extrapolated terrain is forbidden.
    """
    from scipy.interpolate import PchipInterpolator
    value=np.asarray(height,dtype=np.float64)
    x=np.arange(value.shape[1])+minimum_xz[0]+.5
    z=np.arange(value.shape[0])+minimum_xz[1]+.5
    along_x=PchipInterpolator(x,value,axis=1,extrapolate=False)(source_x)
    result=PchipInterpolator(z,along_x,axis=0,extrapolate=False)(source_z)
    if not np.isfinite(result).all():raise ValueError('Fine terrain exceeds known source context')
    return result.astype(np.float32)


def grid_triangles(nz, nx):
    """Source Z increases opposite world Y; wind both triangles upward."""
    a = (np.arange(nz-1, dtype=np.int32)[:, None]*nx + np.arange(nx-1, dtype=np.int32)).ravel()
    faces = np.empty((2*len(a), 3), np.int32)
    faces[0::2] = np.column_stack((a, a+nx, a+1))
    faces[1::2] = np.column_stack((a+1, a+nx, a+nx+1))
    return faces


def grid_normals(height, spacing):
    dz, dx = np.gradient(np.asarray(height, np.float64), spacing)
    normals = np.stack((-dx, dz, np.ones_like(dx)), axis=-1)
    return (normals/np.linalg.norm(normals, axis=-1, keepdims=True)).astype(np.float32)


def build_surface_terrain(workspace, source_path, output, origin, *, recipe=SurfaceRecipe(),
                          material_manifest=None, material_origin=None):
    """Real source -> HighMap -> fine surface arrays, ready for native Blender.

    Uses one contextual field before slicing. Adjacent future payloads must slice
    this same field and share boundary samples; they must not refine in isolation.
    """
    workspace, source_path, output = map(Path, (workspace, source_path, output))
    if output.exists():
        raise ValueError('Preserve previous terrain artifacts; choose a new destination')
    output.mkdir(parents=True)
    def phase(name, **values):
        atomic_json(output/'progress.json', dict(phase=name, at_utc=utc_now(), **values))
    phase('source_reconstruction')
    fields = source_fields(source_path)
    # Keep the same spatial detail at every extent. Large jobs must be tiled,
    # never silently coarsened or allowed to consume the shared-memory reserve.
    from isaacmin.jobs.resources import admit
    core_shape=np.asarray(fields['height'].shape)-2*recipe.context_m
    count=int(np.prod(np.ceil(core_shape/recipe.spacing_m)+1))
    admission=admit(workspace,max(2*2**30,count*320),count*220)
    if admission['status']!='admitted':
        raise RuntimeError('Outdoor extent needs shared-field tiled construction: '+admission['reason'])
    baseline, wet, reconstruction = reconstruct_surface(fields, recipe)
    source_hash = sha256_file(source_path)
    phase('shared_context_drainage')
    drainage = priority_drainage(baseline, fields['validity'], spacing_m=1.)
    area = drainage['contributing_area_m2']
    flow_clip = 10*np.sqrt(np.mean(area))
    runoff = np.clip(area/flow_clip, 0, 1).astype(np.float32)
    np.savez_compressed(output/'drainage.npz', **{k:v for k,v in drainage.items() if isinstance(v, np.ndarray)})
    phase('native_HighMap_erosion')
    badland=np.isin(fields['biome'],['minecraft:badlands','minecraft:eroded_badlands','minecraft:wooded_badlands'])
    native_arguments={}
    erosion,maximum=recipe.erosion_m,recipe.maximum_erosion_m
    if badland.any():
        zone=ndimage.gaussian_filter(badland.astype(np.float32),2.,mode='nearest')
        erosion=max(erosion,recipe.badlands_erosion_m);maximum=max(maximum,recipe.badlands_maximum_erosion_m)
        resistant=np.isin(fields['substrate'],['minecraft:white_terracotta','minecraft:yellow_terracotta',
            'minecraft:light_gray_terracotta','minecraft:sandstone','minecraft:stone'])
        resistance=ndimage.gaussian_filter(np.where(resistant,.4,1.).astype(np.float32),1.,mode='nearest')
        erodibility=(1-zone)*recipe.erosion_m/erosion+zone*resistance
        lowering=(1-zone)*recipe.maximum_erosion_m+zone*recipe.badlands_maximum_erosion_m*resistance
        native_arguments=dict(erodibility_field=erodibility,lowering_limit_field=lowering)
        np.savez_compressed(output/'sedimentary_weathering.npz',zone=zone,resistance=resistance,
                            erodibility=erodibility,lowering_limit_m=lowering)
    frozen=np.isin(fields['substrate'],['minecraft:ice','minecraft:packed_ice','minecraft:blue_ice','minecraft:frosted_ice'])
    reconstruction['ice_columns_excluded_from_soil_erosion']=int(frozen.sum())
    native = refine_heightfield(baseline, output/'highmap', protection=wet|frozen|np.isfinite(source_bank_floor(fields,wet)),
        erosion_m=erosion, max_lowering_m=maximum,
        global_runoff=runoff, global_runoff_normalized=True,**native_arguments)
    if native['status'] != 'success':
        raise RuntimeError('Actual HighMap failed; no placeholder terrain is substituted')
    with np.load(output/'highmap/refinement.npz') as refined:
        height = refined['height'].copy()
    delta = height-fields['height']
    np.savez_compressed(output/'surface_fields.npz', **fields, reconstructed_height=height,
                        reconstructed_delta=delta, runoff=runoff)
    phase('continuous_fine_geometry')
    margin = recipe.context_m
    nz, nx = height.shape
    if min(nz, nx) <= margin*2:
        raise ValueError('Requested context consumes all visible terrain')
    mx, mz = fields['min_xz']
    xmin, zmin, xmax, zmax = mx+margin, mz+margin, mx+nx-margin, mz+nz-margin
    x = xmin + np.arange(round((xmax-xmin)/recipe.spacing_m)+1)*recipe.spacing_m
    z = zmin + np.arange(round((zmax-zmin)/recipe.spacing_m)+1)*recipe.spacing_m
    xx, zz = np.meshgrid(x, z)
    fine = sample_surface_grid(height,x,z,fields['min_xz'])
    sheet=source_frozen_water(fields)
    if sheet.any():
        # Retain frozen water as actual supporting geometry over its observed
        # footprint. A narrow edge transition rounds voxel outlines only.
        signed=ndimage.distance_transform_edt(sheet)-ndimage.distance_transform_edt(~sheet)
        signed=ndimage.gaussian_filter(signed,.35,mode='nearest')
        support=sample_raster(signed,xx,zz,fields['min_xz'])
        _,nearest=ndimage.distance_transform_edt(~sheet,return_indices=True)
        level=sample_raster(fields['height'][tuple(nearest)],xx,zz,fields['min_xz'],order=0)+.04
        blend=np.clip(support/.15,0,1)
        fine=(fine*(1-blend)+level*blend).astype(np.float32)
        reconstruction['frozen_water_edge_transition_m']=.15
        reconstruction['frozen_water_fine_surface']='observed thin ice over water; source levels plus4cm freeboard, final supporting ground mesh'
    ox, oy, oz = origin
    points = np.stack((xx-ox, -zz+oz, fine-oy), axis=-1).astype(np.float32)
    rest_normals = grid_normals(fine, recipe.spacing_m)
    # Reconstructed soil ownership is separately recorded; source samples stay exact.
    material_fields = dict(fields)
    material_fields['source_surface_height']=fields['height']
    material_fields['source_water_validity']=fields['water_validity']
    material_fields['height'] = height
    material_fields['water_validity']=wet
    subsurface_biomes=np.isin(fields['biome'],['minecraft:dripstone_caves','minecraft:lush_caves','minecraft:deep_dark'])
    if subsurface_biomes.any():
        # These labels can occur at the bottom of an exposed cut. The user has
        # removed that cut from the intended world. Infer its outdoor biome from
        # the nearest explicitly labelled exterior column, and record the change.
        if subsurface_biomes.all():raise ValueError('No exterior biome context is available')
        distances,indices=ndimage.distance_transform_edt(subsurface_biomes,return_indices=True)
        material_fields['biome']=fields['biome'].copy()
        material_fields['biome'][subsurface_biomes]=fields['biome'][tuple(indices[:,subsurface_biomes])]
        atomic_json(output/'outdoor_biome_inference.json',dict(
            authority='user_excludes_underground_and_allows_cut_fill',columns=int(subsurface_biomes.sum()),
            maximum_neighbour_distance_m=float(distances[subsurface_biomes].max()),
            method='nearest declared outdoor source biome for intentionally removed subsurface labels',
            source_biome_field='unchanged in source artifact; derived material field only'))
    if np.any(delta > 2.):
        organic = np.zeros(height.shape, bool)
        for token in ('grass_block','dirt','podzol','moss_block','mud'):
            organic |= np.char.find(fields['substrate'].astype(str), token) >= 0
        if organic.any():
            distance, nearest = ndimage.distance_transform_edt(~organic, return_indices=True)
            inferred = (delta > 2.) & (distance <= 2*recipe.narrow_depression_radius_m) & ~wet & ~frozen
            material_fields['substrate_id'] = fields['substrate_id'].copy()
            material_fields['substrate_id'][inferred] = fields['substrate_id'][tuple(nearest[:, inferred])]
        else:
            inferred = np.zeros(height.shape, bool)
    else:
        inferred = np.zeros(height.shape, bool)
    material_fields['inferred_fill_soil'] = inferred
    np.savez_compressed(output/'material_fields.npz', **material_fields)
    from isaacmin.assembly.material_assignment import material_blend_weights,REQUIRED
    material_record=read_json(material_manifest) if material_manifest else {}
    biome_materials=material_record.get('recipe')=='outdoor_biome_materials_v1'
    material_order=([m['asset_id'] for m in material_record['materials']] if biome_materials else list(REQUIRED))
    if biome_materials:
        from isaacmin.assembly.surface_materials import material_fields as make_ownership,sample_weights,source_strata_profile
        ownership,ownership_record=make_ownership(material_fields,material_record['materials'])
        atomic_json(output/'biome_material_ownership.json',ownership_record)
        strata=source_strata_profile(material_fields)
        if strata is not None:
            atomic_json(output/'source_exposed_strata.json',dict(
                **{k:v.tolist() for k,v in strata.items()},
                interpretation='Original scans selected from exposed source layers by elevation; no subterranean material fidelity claim',
                qualification='not_run'))
    weights = np.empty((points.size//3, len(material_order)), np.float32)
    p, n = points.reshape(-1,3), rest_normals.reshape(-1,3)
    phase('physical_material_sampling', vertices=len(p))
    for start in range(0,len(p),250000):
        end = min(start+250000,len(p))
        weights[start:end] = (sample_weights(p[start:end],n[start:end],material_fields,ownership,material_order,origin)
            if biome_materials else material_blend_weights(p[start:end],n[start:end],material_fields,origin))
    # Material charts share the source coordinate frame across regions. The
    # mesh itself stays near its local origin for native float32 precision.
    ax,ay,az=material_origin if material_origin is not None else origin
    chart64=p.astype(np.float64)+np.array([ox-ax,-oz+az,oy-ay])
    chart_positions=chart64.astype(np.float32)
    chart_error=float(np.max(np.abs(chart64-chart_positions)))
    if chart_error>.001:
        raise ValueError('Material chart float32 precision exceeds1mm; choose a world-level origin near its working coverage')
    np.save(output/'rest_positions.npy', chart_positions, allow_pickle=False)
    np.save(output/'rest_normals.npy', n, allow_pickle=False)
    if material_manifest is not None:
        from isaacmin.terrain.surface_sampling import displaced_height
        displacement, mapping = displaced_height(chart_positions, n, weights, material_record['materials'],
            amplitude=recipe.scan_displacement_amplitude_m,material_order=material_order)
        fine += displacement.reshape(fine.shape)
        points[:,:,2] = fine-oy
        atomic_json(output/'displacement.json', mapping)
    else:
        raise ValueError('Real original height channels are required; no noise or flat fallback')
    np.save(output/'vertices.npy', points.reshape(-1,3), allow_pickle=False)
    np.save(output/'normals.npy', grid_normals(fine,recipe.spacing_m).reshape(-1,3), allow_pickle=False)
    np.save(output/'weights.npy', weights, allow_pickle=False)
    faces = grid_triangles(*fine.shape)
    np.save(output/'triangles.npy', faces, allow_pickle=False)
    np.save(output/'height.npy', fine-oy, allow_pickle=False)
    # These raw source/rest arrays describe intentional changes, never cave fidelity.
    report = dict(schema_version=1, status='native_HighMap_surface_geometry_constructed', at_utc=utc_now(),
        recipe=asdict(recipe), source_surface={'path':str(source_path.resolve()),'sha256':source_hash},
        material_order=material_order,material_chart_frame='Minecraft X,-Z,Y relative to persistent world material origin',
        fine_interpolation='separable monotone cubic Hermite; no cubic ringing above flat ice or bank limits',
        material_origin_source_xyz=[ax,ay,az],material_chart_maximum_quantization_m=chart_error,
        material_manifest={'path':str(Path(material_manifest).resolve()),'sha256':sha256_file(Path(material_manifest))},
        source_min_xz=fields['min_xz'].tolist(), origin_xyz=list(origin),
        bounds_source_xz=list(map(float,(xmin,zmin,xmax,zmax))), grid_shape=list(fine.shape),
        vertices=len(p), triangles=len(faces), reconstruction=reconstruction,
        inferred_fill_soil_columns=int(inferred.sum()),
        native_erosion_changed_columns=int(np.count_nonzero(height != baseline)),
        maximum_native_erosion_m=float(np.max(baseline-height)),
        context_drainage_flow_clip_m2=float(flow_clip),
        scope='above_ground_navigation; caves/ravines intentionally reconstructed',
        source_save_modified=False, qualification='not_run',
        collision_policy='exact final triangle positions, no heightfield approximation',
        geometry_sha256=hashlib.sha256(memoryview(points)).hexdigest(),
        producer_sha256=sha256_file(Path(__file__)))
    if sha256_file(source_path) != source_hash:
        raise RuntimeError('Source surface changed during construction')
    atomic_json(output/'terrain.json',report)
    phase('ready_for_native_Blender_export', vertices=len(p), triangles=len(faces))
    return report


class SurfaceQuery:
    """Vectorized exact barycentric query of the final, fixed-diagonal mesh."""
    def __init__(self, directory):
        directory = Path(directory)
        self.record = read_json(directory/'terrain.json')
        self.height = np.load(directory/'height.npy', mmap_mode='r')
        self.spacing = self.record['recipe']['spacing_m']
        self.xmin,self.zmin,self.xmax,self.zmax = self.record['bounds_source_xz']
        self.origin = self.record['origin_xyz']

    def heights(self, x, y):
        x,y = np.broadcast_arrays(np.asarray(x,float),np.asarray(y,float))
        u=(x+self.origin[0]-self.xmin)/self.spacing
        v=(-y+self.origin[2]-self.zmin)/self.spacing
        nz,nx=self.height.shape
        valid=(u>=0)&(v>=0)&(u<=nx-1)&(v<=nz-1)
        ix=np.clip(np.floor(u).astype(np.int64),0,nx-2)
        iz=np.clip(np.floor(v).astype(np.int64),0,nz-2)
        a,b=u-ix,v-iz
        h00,h10,h01,h11=self.height[iz,ix],self.height[iz,ix+1],self.height[iz+1,ix],self.height[iz+1,ix+1]
        value=np.where(a+b<=1,h00+(h10-h00)*a+(h01-h00)*b,
                       h11+(h01-h11)*(1-a)+(h10-h11)*(1-b))
        return np.where(valid,value,np.nan)
