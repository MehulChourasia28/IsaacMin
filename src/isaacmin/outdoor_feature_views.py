"""Additional development views chosen from source features and final support."""
from pathlib import Path
import numpy as np
from scipy import ndimage
from isaacmin.terrain.surface_navigation import source_fields, SurfaceQuery


def ice_spire_ground_pose(terrain):
    """Inspect observed packed ice relief from a dry, supported ground position.

    This is a development camera, not a held-out qualification view. Selection
    uses geometry only; no image score, source name or map coordinate is used.
    """
    terrain=Path(terrain);query=SurfaceQuery(terrain)
    fields=source_fields(terrain/'material_fields.npz')
    h=fields.get('source_surface_height',fields['height'])
    names=fields['block_names'][fields['substrate_id']]
    packed=np.isin(names,['minecraft:packed_ice','minecraft:blue_ice'])
    zz,xx=np.indices(h.shape)
    x=xx+fields['min_xz'][0]+.5-query.origin[0]
    y=-zz-fields['min_xz'][1]-.5+query.origin[2]
    inside=(x>query.xmin-query.origin[0]+12)&(x<query.xmax-query.origin[0]-12)
    inside&=(y>-query.zmax+query.origin[2]+12)&(y<-query.zmin+query.origin[2]-12)
    prominence=h-ndimage.grey_opening(h,size=11,mode='nearest')
    peaks=packed&inside&(h==ndimage.maximum_filter(h,3,mode='nearest'))&(prominence>=6)
    labels,count=ndimage.label(peaks)
    if not count:return None
    peaks=ndimage.maximum_position(prominence,labels,range(1,count+1))
    peaks=sorted(peaks,key=lambda p:(fields['biome'][p]!='minecraft:ice_spikes',
        -float(prominence[p]),-float(h[p]),p))
    wet=fields['water_validity']&(fields['water_height']>=fields['height'])
    dry=inside&~wet
    for rz,rx in peaks:
        tx,ty=float(x[rz,rx]),float(y[rz,rx]);top=float(query.heights(tx,ty))
        distance=np.hypot(x-tx,y-ty)
        iz,ix=np.nonzero(dry&(distance>=12)&(distance<=40))
        if not len(iz):continue
        cx,cy=x[iz,ix],y[iz,ix];ground=query.heights(cx,cy)
        grade=np.hypot(query.heights(cx+.5,cy)-query.heights(cx-.5,cy),
                       query.heights(cx,cy+.5)-query.heights(cx,cy-.5))
        angle=np.arctan2(top-ground,distance[iz,ix])
        valid=(grade<=.35)&(angle>=.2)&(angle<=.8)&np.isfinite(ground)
        indices=np.flatnonzero(valid)
        if not len(indices):continue
        # Prefer a moderate upward view showing the full tip and its base.
        indices=indices[np.argsort(np.abs(angle[indices]-.45),kind='stable')]
        for i in indices[:64]:
            camera=np.array([cx[i],cy[i],ground[i]+.6])
            target=np.array([tx,ty,ground[i]+.55*(top-ground[i])])
            t=np.linspace(.05,.95,24)
            tip=np.array([tx,ty,top+.1])
            sight=camera[None,:]+t[:,None]*(tip-camera)[None,:]
            if np.any(query.heights(sight[:,0],sight[:,1])>sight[:,2]-.1):continue
            return dict(kind='static',name='source_ice_spire_ground',position=camera.tolist(),
                look_at=target.tolist(),scout_height_m=.6,held_out=False,
                surface_scope='outdoor_ice_detail',source_biome=str(fields['biome'][rz,rx]),
                selection='observed packed-ice local relief; dry final-mesh support and terrain sightline; no image-score selection',
                source_feature=dict(world_xy=[tx,ty],source_tip_height_m=float(h[rz,rx]),
                    actual_tip_height_m=top,local_prominence_m=float(prominence[rz,rx])),
                camera_response=dict(ev100=13.747247562465875,f_number=8.,iso=100,
                    policy='same outdoor daylight camera as existing frozen views'))
    return None
