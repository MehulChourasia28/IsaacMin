import numpy as np
from isaacmin.assembly.material_assignment import classify_surface_materials


def source():
    shape=(6,6)
    s={'height':np.full(shape,10.),'validity':np.ones(shape,bool),'biome':np.full(shape,'minecraft:plains',dtype='<U50'),
       'substrate_id':np.zeros(shape,int),'block_names':np.array(['minecraft:grass_block','minecraft:dirt','minecraft:stone']),
       'water_height':np.zeros(shape),'water_validity':np.zeros(shape,bool),'min_xz':np.array([-12,-9])}
    s['biome'][1,1]='minecraft:forest';s['substrate_id'][2,2]=1
    s['water_validity'][5,5]=True;s['water_height'][5,5]=11
    s['water_validity'][0,0]=True;s['water_height'][0,0]=9
    return s


def materials():
    return [{'asset_id':x,'repeat_m':1.3+i} for i,x in enumerate(['brown_mud_dry','rock_face_03','leafy_grass','forest_ground_04'])]


def point(ix,iz,z=10):
    # Source origin [-10,2,-6], target [MinecraftX+10,-MinecraftZ-6,MinecraftY-2].
    return [ix-12+.5+10,-(iz-9+.5)-6,z-2]


def test_real_source_semantics_separate_grass_litter_soil_banks_and_caves():
    points=[point(0,0),point(1,1),point(2,2),point(5,5),point(5,4),point(1,1,6),point(1,1),point(1,1)]
    normals=[[0,0,1]]*6+[[0,0,-1],[1,0,.1]]
    assigned,report=classify_surface_materials(points,normals,source(),[-10,2,-6],materials())
    assert assigned.tolist()==[2,3,0,0,0,1,1,1]
    assert report['aquifer_only_columns_excluded']==1
    assert report['exposed_water_source_columns']==1
    assert report['physical_repeat_m']['leafy_grass']==3.3
    assert report['isaac_qualification']=='not_run'


def test_shifted_exterior_uses_recorded_delta_without_greening_deep_surfaces():
    points=[point(0,0,11.2),point(0,0,7)]
    assigned,report=classify_surface_materials(points,[[0,0,1]]*2,source(),[-10,2,-6],materials(),exterior_delta=np.full((6,6),1.2))
    assert assigned.tolist()==[2,1]


def test_unknown_source_is_explicit_boundary_candidate():
    assigned,report=classify_surface_materials([point(30,30)],[[0,0,1]],source(),[-10,2,-6],materials())
    assert assigned.tolist()==[1] and report['outside_valid_source_faces']==1


def test_retained_masonry_is_labelled_structural_not_silently_natural_rock():
    assigned,report=classify_surface_materials([point(0,0),point(1,1)],[[0,0,1]]*2,source(),[-10,2,-6],materials(),
                                               structural_source_coordinates=[[-12,9,-9]])
    assert assigned.tolist()==[1,3]
    assert report['assignment_reason_counts']['retained_structural_unqualified']==1
    assert report['structural_ownership']['face_indices']==[0]
    assert report['structural_ownership']['isaac_appearance_qualification']=='not_run'


def test_continuous_weights_remove_cell_jumps_and_preserve_caves_and_structure():
    from isaacmin.assembly.material_assignment import material_blend_weights
    s=source();s['biome'][:,:3]='minecraft:plains';s['biome'][:,3:]='minecraft:forest'
    s['substrate_id'][:]=0;s['water_validity'][:]=False
    points=np.asarray([point(2.4999,2),point(2.5001,2),point(2,2,5),point(0,0)])
    w=material_blend_weights(points,[[0,0,1]]*4,s,[-10,2,-6],structural_source_coordinates=[[-12,9,-9]])
    assert np.max(np.abs(w[0]-w[1]))<.001
    assert np.allclose(w.sum(axis=1),1) and (w>=0).all()
    assert 0<w[0,2]<1 and 0<w[0,3]<1
    assert w[2].tolist()==[0,1,0,0] and w[3].tolist()==[0,1,0,0]


def test_bake_color_transfer_and_lossless_png16(tmp_path):
    from isaacmin.assembly.material_bake import _srgb,_linear,_png16
    from PIL import Image
    a=np.linspace(0,1,1024,dtype=np.float32)
    assert np.allclose(_srgb(_linear(a)),a,atol=1e-6)
    p=tmp_path/'data.png';_png16(p,a.reshape(1,-1))
    values=np.asarray(Image.open(p))
    assert np.max(np.abs(values.astype(float)/65535-a))<=1/65535
