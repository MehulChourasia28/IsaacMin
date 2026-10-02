"""Finite staged footprint queries for source-backed local cave diagnostics."""
import numpy as np
import trimesh


def plan_cave_support_routes(poses,mesh,native,contains,transform,world_to_source,minimum,shape,validity,occupancy):
    options=[]
    for pose_index,pose in enumerate(poses):
        centre=np.asarray(pose['support_world_xyz']);heading=np.asarray(pose['look_at'])[:2]-centre[:2];angle=np.arctan2(heading[1],heading[0])
        for turn in (0.,np.pi/2,-np.pi/2,np.pi):
            forward=np.asarray([np.cos(angle+turn),np.sin(angle+turn)]);lateral=np.asarray([-forward[1],forward[0]])
            centres=np.tile(centre,(7,1));centres[:,:2]+=np.linspace(-.3,.3,7)[:,None]*forward
            footprint=np.asarray([a*forward+b*lateral for a in (-.175,0,.175) for b in (-.125,0,.125)])
            rays=np.repeat(centres,9,axis=0);rays[:,:2]+=np.tile(footprint,(7,1));rays[:,2]+=.45
            options.append({'pose_index':pose_index,'centre':centre,'centres':centres,'forward':forward,'footprint':footprint,'rays':rays,'valid':False})
    if not options:return [],[]
    rays=np.concatenate([option['rays'] for option in options]);hits,ids,faces=mesh.ray.intersects_location(rays,np.tile([0.,0.,-1.],(len(rays),1)),multiple_hits=False)
    all_grounds=np.full_like(rays,np.nan);all_normals=np.full_like(rays,np.nan);all_grounds[ids]=hits;all_normals[ids]=mesh.face_normals[faces]
    bodies=[];eligible=[]
    for index,option in enumerate(options):
        grounds=all_grounds[index*63:(index+1)*63];normals=all_normals[index*63:(index+1)*63]
        if not np.isfinite(grounds).all():continue
        heights=grounds[:,2].reshape(7,9);centre=option['centre'];centres=option['centres'];footprint=option['footprint']
        if np.any(normals[:,2]<.9) or np.any(np.abs(heights-centre[2])>.1) or np.any(np.ptp(heights,axis=1)>.025):continue
        base_heights=heights.max(axis=1)+.015;centres[:,2]=grounds[np.arange(7)*9+4,2]
        body=[];valid=True
        for sample,base_z in zip(centres,base_heights):
            corners=np.asarray([[*(sample[:2]+xy),z] for xy in footprint for z in (base_z,base_z+.1,base_z+.2)])
            source_box=transform(corners,world_to_source);low=np.floor(source_box.min(axis=0)-minimum).astype(int)[[1,2,0]];high=np.floor(source_box.max(axis=0)-minimum).astype(int)[[1,2,0]]
            if np.any(low<0) or np.any(high>=shape):valid=False;break
            selection=tuple(slice(a,b+1) for a,b in zip(low,high))
            if not validity[selection].all() or np.any(occupancy[selection]!=0):valid=False;break
            body.extend(corners)
        if not valid:continue
        body=np.asarray(body);option.update(grounds=grounds,normals=normals,base_heights=base_heights,body=body);eligible.append(index);bodies.append(body)
    if eligible:
        body=np.concatenate(bodies);inside=contains(body)
        if native is not None:_,distances,_=native.closest(body)
        else:_,distances,_=trimesh.proximity.closest_point(mesh,body)
        starts=np.concatenate([option['body'].reshape(7,27,3)[:-1].reshape(-1,3) for option in (options[i] for i in eligible)])
        ends=np.concatenate([option['body'].reshape(7,27,3)[1:].reshape(-1,3) for option in (options[i] for i in eligible)])
        if native is not None:clear=native.unobstructed(starts,ends,epsilon=1e-7)
        else:
            delta=ends-starts;lengths=np.linalg.norm(delta,axis=1);locations,rays,_=mesh.ray.intersects_location(starts,delta/lengths[:,None],multiple_hits=True);clear=np.ones(len(starts),bool)
            if len(locations):clear[rays[np.linalg.norm(locations-starts[rays],axis=1)<lengths[rays]-1e-7]]=False
        for position,index in enumerate(eligible):
            selected=slice(position*189,(position+1)*189);option=options[index]
            option['valid']=bool(not inside[selected].any() and np.all(distances[selected]>=.01) and clear[position*162:(position+1)*162].all());option['minimum_clearance']=float(distances[selected].min())
    routes=[];missing=[];represented=set()
    for pose_index,pose in enumerate(poses):
        key=(pose['surface_scope'],pose.get('source_portal_id',pose['source_feature_id']))
        if key in represented:continue
        successful=next((item for item in options[pose_index*4:(pose_index+1)*4] if item['valid']),None)
        if successful is None:
            missing.append({'surface_scope':pose['surface_scope'],'source_feature_id':pose['source_feature_id'],'source_portal_id':pose.get('source_portal_id'),'reason':'No0.6m local route passes the source-air and sampled final-mesh footprint/clearance checks in four tested headings'});continue
        data=successful;centres=data['centres'];distance=float(np.linalg.norm(np.diff(centres,axis=0),axis=1).sum())
        routes.append({'id':'cave-support-route-%03d'%len(routes),'status':'planned_not_simulated','surface_scope':'portal' if pose['surface_scope']=='cave_portal' else 'cave_floor','original_surface_scope':pose['surface_scope'],'source_feature_id':pose['source_feature_id'],'source_portal_id':pose.get('source_portal_id'),'support_source_class':pose['support_source_class'],'portal_support_distance_m':pose.get('portal_support_distance_m'),'points_world_xyz':centres.tolist(),'planned_distance_m':distance,'support_world_xyz':centres.tolist(),'footprint_ground_world_xyz':data['grounds'].reshape(7,9,3).tolist(),'support_normals':data['normals'].reshape(7,9,3).tolist(),'body_centre_world_xyz':np.column_stack((centres[:,:2],data['base_heights']+.1)).tolist(),'heading_xy':data['forward'].tolist(),'route_length_m':distance,'sample_spacing_m':.1,'footprint_m':[.35,.25],'body_height_m':.2,'body_minimum_sampled_mesh_clearance_m':data['minimum_clearance'],'purpose':'local dynamic ground-contact diagnostic; no user navigation or exterior cave-access claim','clearance_scope':'entire source-cell bounding box known air; final mesh9footprint rays,27body probes per pose and162connecting segments; finite geometric sampling'})
        represented.add(key)
    return routes,missing
