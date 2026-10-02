"""Keep seabed scans from bleeding up reconstructed source ice faces."""
import numpy as np
from scipy import ndimage
from isaacmin.terrain.surface_navigation import sample_raster


def exposed_ice_ownership(points,fields,origin):
    """A bounded height-aware interpretation of observed exterior ice columns.

    Spatial material smoothing alone mixes a narrow high ice column with its
    much lower seabed. Only reconstructed relief supported by a nearby observed
    ice top may inherit ice. No hidden ice volume is asserted or reconstructed.
    """
    if '_exposed_ice_support' not in fields:
        names=fields['block_names'][fields['substrate_id']]
        ice=np.isin(names,['minecraft:ice','minecraft:packed_ice','minecraft:blue_ice','minecraft:frosted_ice'])
        if not ice.any():fields['_exposed_ice_support']=None
        else:
            distance,nearest=ndimage.distance_transform_edt(~ice,return_indices=True)
            original=fields.get('source_surface_height',fields['height'])
            fields['_exposed_ice_support']=(ice.astype(float),distance,original[tuple(nearest)],original)
    support=fields['_exposed_ice_support']
    if support is None:return np.zeros(len(points),np.float32)
    ice,distance,ice_top,original=support
    x=points[:,0]+origin[0];z=-points[:,1]+origin[2];height=points[:,2]+origin[1]
    def sample(a):return sample_raster(a,x,z,fields['min_xz'])
    near=np.clip((3.-sample(distance))/1.75,0,1)
    below_source_top=np.clip((sample(ice_top)+1.-height)/.75,0,1)
    raised_side=np.clip((height-sample(original)-.25)/1.5,0,1)
    original_ice=np.clip(2.*sample(ice),0,1)
    return (near*below_source_top*np.maximum(original_ice,raised_side)).astype(np.float32)
