"""Biome-specific continuous scan ownership for the outdoor terrain path."""
from pathlib import Path
import shutil
import numpy as np
from scipy import ndimage
from isaacmin.terrain.biomes import BY_BIOME
from isaacmin.terrain.surface_navigation import sample_raster
from isaacmin.io import atomic_json,sha256_file


def source_strata_profile(fields):
    """Infer exposed sedimentary bands only at elevations observed in this save.

    The source column's cap should not paint an entire reconstructed cliff.
    Observed terracotta colours choose original geological scans, with bounded
    interpolation over unobserved heights. This is a recorded interpretation of
    exposed source layers, not fabricated subsurface stratigraphy.
    """
    names=fields['block_names'][fields['substrate_id']].astype(str)
    arid=np.isin(fields['biome'],['minecraft:badlands','minecraft:eroded_badlands','minecraft:wooded_badlands'])
    exposed=arid&(np.char.find(names,'terracotta')>=0)
    source_height=fields.get('source_surface_height',fields['height'])
    if not exposed.any():return None
    lo=int(np.floor(source_height[exposed].min()))-1
    hi=int(np.ceil(source_height[exposed].max()))+1
    bins=np.rint(source_height[exposed]).astype(int)-lo
    total=np.bincount(bins,minlength=hi-lo+1)
    light=np.isin(names[exposed],['minecraft:white_terracotta','minecraft:light_gray_terracotta','minecraft:yellow_terracotta'])
    pale=np.bincount(bins,weights=light,minlength=hi-lo+1)
    observed=total>0;z=np.arange(lo,hi+1,dtype=float)
    fraction=np.interp(z,z[observed],pale[observed]/total[observed])
    return dict(height=z,light_fraction=ndimage.gaussian_filter1d(fraction,.45,mode='nearest'),
        distance_to_observed=ndimage.distance_transform_edt(~observed),counts=total)


def material_fields(fields, materials):
    """No material fallback: missing required recipes remain explicit blockers."""
    ids=[m['asset_id'] for m in materials];height=fields['height'];biome=fields['biome']
    ownership=np.zeros((*height.shape,len(ids)),np.float32);unknown={}
    names=fields['block_names'][fields['substrate_id']].astype(str)
    wet=fields['water_validity']&(fields['water_height']>=height)
    for name in np.unique(biome):
        mask=biome==name;recipe=BY_BIOME.get(str(name).removeprefix('minecraft:'))
        if recipe is None:
            unknown[str(name)]=int(mask.sum());continue
        material=recipe.ground_material
        if material not in ids:
            unknown[str(name)+':missing:'+material]=int(mask.sum());continue
        ownership[:,:,ids.index(material)][mask]=1
    # Bare source rock remains exposed independent of the overlying biome label.
    rock=np.zeros(height.shape,bool)
    for token in ('stone','granite','diorite','andesite','deepslate','tuff','calcite','gravel'):
        rock|=np.char.find(names,token)>=0
    snow=np.char.find(names,'snow')>=0
    sand=(np.char.find(names,'sand')>=0)&~rock
    for mask,material in [(rock,'rock_face_03'),(sand,'sand_03'),(snow,'snow_02')]:
        if mask.any() and material in ids:
            ownership[mask]=0;ownership[:,:,ids.index(material)][mask]=1
    # Sedimentary substrates retain their geology. Substring matching alone
    # classified sandstone as generic grey stone and red sand as beach sand.
    # These are original scans, not recoloured generic forest/rock textures.
    arid=np.isin(biome,['minecraft:badlands','minecraft:eroded_badlands','minecraft:wooded_badlands'])
    sedimentary=[(names=='minecraft:sandstone','rock_06'),
        (names=='minecraft:red_sandstone','cliff_side'),
        (arid&(np.char.find(names,'terracotta')>=0),'cliff_side'),
        (arid&np.isin(names,['minecraft:white_terracotta','minecraft:light_gray_terracotta',
                           'minecraft:yellow_terracotta']),'rock_06'),
        (arid&(names=='minecraft:red_sand'),'sandy_gravel_02')]
    for mask,material in sedimentary:
        if mask.any():
            if material not in ids:
                raise ValueError('Required original sedimentary scan unavailable: '+material)
            ownership[mask]=0;ownership[:,:,ids.index(material)][mask]=1
    # Frozen surface is identified from actual source blocks, not merely a
    # cold biome name. Open water and exposed seabed must remain distinct.
    ice=np.isin(names,['minecraft:ice','minecraft:packed_ice','minecraft:blue_ice','minecraft:frosted_ice'])
    if ice.any():
        ice_material=next((m for m in ('Ice004','Ice003') if m in ids),None)
        if ice_material is None:raise ValueError('Required original ice material unavailable')
        ownership[ice]=0;ownership[:,:,ids.index(ice_material)][ice]=1
    if wet.any() and 'sand_03' in ids:
        # Source sediment under water keeps sediment; rocky beds stay rock.
        mask=wet&~rock;ownership[mask]=0;ownership[:,:,ids.index('sand_03')][mask]=1
    unresolved=ownership.sum(axis=2)==0
    if unresolved.any():
        raise ValueError('No qualified interpretation fallback is allowed for required surface materials: '+str(unknown))
    # Blur ownership in metric source space, without colourizing any source maps.
    for i in range(len(ids)):
        ownership[:,:,i]=ndimage.gaussian_filter(ownership[:,:,i],1.5,mode='nearest')
    ownership/=ownership.sum(axis=2,keepdims=True)
    return ownership,dict(material_order=ids,unmapped=unknown,method='biome_and_substrate_ownership_with_1.5m_continuous_transition',
        sedimentary_interpretation='Original warm cliff, light stratified rock and sandy gravel scans; exposed source terracotta layer elevations guide reconstructed steep faces, not literal Minecraft pigment recovery',
        sedimentary_layer_inference='Regional exposed-source elevation profile; at most2m interpolation beyond observed bands; no underground material claim',
        ice_interpretation='Original ambientCG opaque ice PBR channels on source ice; provider procedural, not a photographed scan; scale and optical fidelity require qualification',
        packed_ice_frost='Original clean frost material when explicitly included, otherwise snow02; supported deposition plus thin exposed-face rime. Climate inference, not source snow observation',
        exposed_ice_ownership='Height-aware exterior ice support within3m prevents low seabed material bleeding up reconstructed ice; not a hidden-volume claim',
        alpine_snow_retention=dict(slope_loss_start_degrees=28,slope_loss_complete_degrees=55,
            default_wind_world_xy=[1.,0.],crest_scale_m=8.,
            method='slope shedding and windward convex-crest scour; source biome limited'),
        biome_complete_visual_qualification='not_run; ground scans alone do not qualify a biome')


def sample_weights(points,normals,fields,ownership,material_order,origin):
    x=points[:,0]+origin[0];z=-points[:,1]+origin[2]
    weights=np.column_stack([sample_raster(ownership[:,:,i],x,z,fields['min_xz']) for i in range(len(material_order))])
    ice_material=next((m for m in ('Ice004','Ice003') if m in material_order),None)
    if ice_material is not None:
        from isaacmin.assembly.frozen_surface import exposed_ice_ownership
        ice_owner=exposed_ice_ownership(points,fields,origin)
        weights*=1-ice_owner[:,None]
        weights[:,material_order.index(ice_material)]+=ice_owner
    up=normals[:,2]/np.linalg.norm(normals,axis=1)
    # Rock exposure grows continuously with slope. Existing rock/snow/sand have
    # separate semantics; dunes and snow banks must not become bare bedrock.
    rock=material_order.index('rock_face_03')
    organic=[i for i,m in enumerate(material_order) if m in ('brown_mud_dry','leafy_grass',
        'forest_ground_04','forest_ground_05','withered_grass','mud_forest')]
    exposure=np.clip((np.cos(np.deg2rad(28))-up)/(np.cos(np.deg2rad(28))-np.cos(np.deg2rad(52))),0,1)
    exposure=exposure*exposure*(3-2*exposure)
    transfer=weights[:,organic]*exposure[:,None];weights[:,organic]-=transfer;weights[:,rock]+=transfer.sum(axis=1)
    if all(m in material_order for m in ('cliff_side','rock_06','sandy_gravel_02')):
        # Loose sediment accumulates at gentle, concave feet of exposed strata.
        # The relief field is derived from terrain; it has no map/camera anchors.
        if '_badland_deposition_context' not in fields:
            zone=np.isin(fields['biome'],['minecraft:badlands','minecraft:eroded_badlands','minecraft:wooded_badlands'])
            concavity=np.maximum(ndimage.gaussian_filter(fields['height'],8.,mode='nearest')-fields['height'],0)
            fields['_badland_deposition_context']=(ndimage.gaussian_filter(zone.astype(np.float32),1.5),concavity)
        zone,concavity=fields['_badland_deposition_context']
        if '_source_strata_profile' not in fields:
            fields['_source_strata_profile']=source_strata_profile(fields)
        strata=fields['_source_strata_profile']
        if strata is not None:
            source_y=points[:,2]+origin[1]
            known=np.interp(source_y,strata['height'],strata['distance_to_observed'],left=1e6,right=1e6)<=2.
            fraction=np.interp(source_y,strata['height'],strata['light_fraction'])
            blend=sample_raster(zone,x,z,fields['min_xz'])*exposure*known
            warm,light=(material_order.index(m) for m in ('cliff_side','rock_06'))
            total=weights[:,warm]+weights[:,light]
            weights[:,warm]=weights[:,warm]*(1-blend)+total*(1-fraction)*blend
            weights[:,light]=weights[:,light]*(1-blend)+total*fraction*blend
        bank=np.clip(sample_raster(concavity,x,z,fields['min_xz'])/2.,0,1)
        gentle=np.clip((up-np.cos(np.deg2rad(34)))/(1-np.cos(np.deg2rad(34))),0,1)
        deposit=.65*sample_raster(zone,x,z,fields['min_xz'])*bank*gentle
        layers=[material_order.index(m) for m in ('cliff_side','rock_06')]
        transfer=weights[:,layers]*deposit[:,None]
        weights[:,layers]-=transfer;weights[:,material_order.index('sandy_gravel_02')]+=transfer.sum(axis=1)
    if 'snow_02' in material_order:
        # Snow remains on gentle terrain and in sheltered hollows, but steep
        # alpine rock faces and windward crests should not wear a uniform coat.
        # The rule is tied to source biome, slope and relief, never map positions.
        if '_alpine_snow_context' not in fields:
            alpine=np.isin(fields['biome'],['minecraft:jagged_peaks','minecraft:frozen_peaks',
                'minecraft:stony_peaks','minecraft:snowy_slopes'])
            crest=fields['height']-ndimage.gaussian_filter(fields['height'],8.,mode='nearest')
            fields['_alpine_snow_context']=(ndimage.gaussian_filter(alpine.astype(np.float32),1.5),crest)
        alpine,crest=fields['_alpine_snow_context']
        zone=sample_raster(alpine,x,z,fields['min_xz'])
        convex=sample_raster(crest,x,z,fields['min_xz'])
        shedding=np.clip((np.cos(np.deg2rad(28))-up)/(np.cos(np.deg2rad(28))-np.cos(np.deg2rad(55))),0,1)
        shedding=shedding*shedding*(3-2*shedding)
        ridge=np.clip((convex-.2)/1.3,0,1);ridge=ridge*ridge*(3-2*ridge)
        windward=np.maximum(-normals[:,0]/np.linalg.norm(normals,axis=1),0)
        loss=zone*(1-(1-shedding)*(1-.6*ridge*windward))
        snow=material_order.index('snow_02');shed=weights[:,snow]*loss
        weights[:,snow]-=shed;weights[:,rock]+=shed
        if ice_material is not None:
            if '_packed_ice_context' not in fields:
                names=fields['block_names'][fields['substrate_id']]
                packed=np.isin(names,['minecraft:packed_ice','minecraft:frosted_ice'])
                # Exterior faces extend between source column centres. A blurred
                # one-column mask otherwise leaves a narrow spire nearly bare.
                # This bounded2m support applies only to already-owned ice.
                support_mask=ndimage.maximum_filter(packed,size=5,mode='nearest')
                fields['_packed_ice_context']=ndimage.gaussian_filter(support_mask.astype(np.float32),.5,mode='nearest')
            support=np.clip((up-np.cos(np.deg2rad(65)))/(1-np.cos(np.deg2rad(65))),0,1)
            deposit=sample_raster(fields['_packed_ice_context'],x,z,fields['min_xz'])*(.55+.30*support)
            if '_frozen_sheet_context' not in fields:
                from isaacmin.terrain.surface_navigation import source_frozen_water
                sheet=source_frozen_water(fields)
                fields['_frozen_sheet_context']=ndimage.gaussian_filter(sheet.astype(np.float32),.45,mode='nearest')
            # Exposed lake ice develops a fine frost layer too. Only actual
            # source ice sheets receive this treatment; open water is unchanged.
            sheet=sample_raster(fields['_frozen_sheet_context'],x,z,fields['min_xz'])
            deposit=np.maximum(deposit,.72*np.clip(2.*sheet,0,1))
            ice=material_order.index(ice_material);frost=weights[:,ice]*deposit
            frost_material=material_order.index('Snow001') if 'Snow001' in material_order else snow
            weights[:,ice]-=frost;weights[:,frost_material]+=frost
    weights=np.maximum(weights,0);weights/=weights.sum(axis=1,keepdims=True)
    return weights.astype(np.float32)


def prepare_surface_library(workspace,materials,output,*,active_indices=None):
    """Author native MDL using exact original maps and per-material primvars."""
    workspace,output=Path(workspace),Path(output)
    output.mkdir(parents=True,exist_ok=False);(output/'textures').mkdir()
    template=(workspace/'recipes/materials/surface_stochastic.mdl.in').read_text()
    prefix=template.split('export material IsaacMinScans()')[0]
    active=set(range(len(materials)) if active_indices is None else active_indices)
    if not active or not active<=set(range(len(materials))):raise ValueError('Invalid active surface material indices')
    lines=['export material IsaacMinScans() = let {',
        '    float3 p = scene::data_lookup_float3("IsaacMinSurfaceRestPosition", state::transform_point(state::coordinate_internal,state::coordinate_world,state::position()));',
        '    float3 gn = math::normalize(state::transform_normal(state::coordinate_internal,state::coordinate_world,state::normal()));',
        '    float3 chart = math::normalize(scene::data_lookup_float3("IsaacMinSurfaceRestNormal",gn));']
    proofs=[]
    for i,material in enumerate(materials):
        if i not in active:
            # The caller measured exact zero at every vertex. Linear primvar
            # interpolation is therefore zero everywhere on every triangle.
            lines.extend([f'    float w{i} = 0.0;',f'    Sample s{i} = Sample(color(0.0),0.0,float3(0.0));'])
            continue
        refs={}
        if material.get('licence')!='CC0-1.0':raise ValueError('Original eligible licence required')
        for role in ('base_color','roughness','normal','height'):
            p=Path(material[role]);proof=material['original_channel_proof'][role]
            if sha256_file(p)!=proof['sha256'] or min(proof['width'],proof['height'])/material['repeat_m']<1024:
                raise ValueError('Original scan channel integrity or metric density failed')
            q=output/'textures'/(proof['sha256']+p.suffix)
            if not q.exists():
                try:q.hardlink_to(p)
                except OSError:shutil.copyfile(p,q)
            refs[role]='./'+q.relative_to(output).as_posix()
            proofs.append(dict(asset_id=material['asset_id'],role=role,path=q.relative_to(output).as_posix(),
                sha256=proof['sha256'],width=proof['width'],height=proof['height'],repeat_m=material['repeat_m'],
                licence=material['licence'],source_page=material['source_page']))
        lines.append(f'    float w{i} = scene::data_lookup_float("IsaacMinSurfaceWeight{i}",0.0);')
        lines.append(f'    Sample s{i} = w{i} == 0.0 ? Sample(color(0.0),0.0,float3(0.0)) : scan(texture_2d("{refs["base_color"]}",tex::gamma_srgb),texture_2d("{refs["roughness"]}",tex::gamma_linear),texture_2d("{refs["normal"]}",tex::gamma_linear),{float(material["repeat_m"])},p,chart);')
    lines += ['    color col = '+'+'.join(f's{i}.c*w{i}' for i in range(len(materials)))+';',
        '    float rough = '+'+'.join(f's{i}.r*w{i}' for i in range(len(materials)))+';',
        '    float surface_ior = '+'+'.join(f'{float(m.get("ior",1.5))}*w{i}' for i,m in enumerate(materials))+';',
        '    float3 normal_world = math::normalize(gn+'+'+'.join(f's{i}.delta*w{i}' for i in range(len(materials)))+');',
        '    float3 normal = math::normalize(state::transform_normal(state::coordinate_world,state::coordinate_internal,normal_world));']
    tail=template.split('} in material(',1)[1]
    tail=tail.replace('ior:color(1.5)','ior:color(surface_ior)').replace('ior:1.5,','ior:surface_ior,')
    module=output/'IsaacMinScans.mdl';module.write_text(prefix+'\n'.join(lines)+'\n} in material('+tail)
    record=dict(schema_version=1,status='biome_scan_library_authored',material_order=[m['asset_id'] for m in materials],
        active_material_indices=sorted(active),inactive_material_policy='Only exactly zero vertex-weight channels may be omitted; no threshold or texture simplification',
        module='IsaacMinScans.mdl',module_sha256=sha256_file(module),textures=proofs,
        projection='same rest position/normal charts for native height displacement and original PBR channels',
        qualification='not_run',producer_sha256=sha256_file(Path(__file__)))
    atomic_json(output/'material_manifest.json',record)
    return record
