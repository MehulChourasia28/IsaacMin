"""Geometry properties only; these checks do not qualify native appearance."""
import itertools
import numpy as np
import pytest
from isaacmin.io import atomic_json
from isaacmin.assembly.surface_water import _positive_triangles, build_surface_water
from isaacmin.terrain.surface_navigation import reconstruct_surface, SurfaceQuery, sample_raster, sample_surface_grid, source_bank_floor, source_frozen_water, relax_steep_contours


@pytest.mark.parametrize('signs', list(itertools.product((-1., 1.), repeat=3)))
def test_shore_clipping_preserves_winding_and_linear_area(signs):
    points=np.array([[0.,0.,3.],[1.,0.,3.],[0.,1.,3.]])
    triangles,area=_positive_triangles(points,np.array([[0,1,2]]),np.array(signs))
    assert area == pytest.approx([0.,.125,.375,.5][sum(v>0 for v in signs)])
    if len(triangles):
        assert (np.cross(triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0])[:,2]>0).all()
        x,y=triangles[:,:,0],triangles[:,:,1]
        assert np.min(signs[0]*(1-x-y)+signs[1]*x+signs[2]*y)>=-1e-12


def test_water_uses_source_level_and_final_ground_with_shifted_origin(tmp_path):
    wet=np.zeros((8,8),bool);wet[2:6,2:6]=True;h=np.where(wet,22.,24.)
    np.savez(tmp_path/'material_fields.npz',height=h,validity=np.ones_like(wet),
        water_validity=wet,water_height=np.full_like(h,23.5),min_xz=[1000,-2000],
        substrate=np.full(h.shape,'minecraft:sand'))
    sx,sz=np.meshgrid(np.linspace(1000,1008,65),np.linspace(-2000,-1992,65))
    np.save(tmp_path/'height.npy',(sample_raster(h,sx,sz,[1000,-2000])-20).astype(np.float32))
    atomic_json(tmp_path/'terrain.json',dict(recipe={'spacing_m':.125},
        bounds_source_xz=[1000,-2000,1008,-1992],origin_xyz=[1000,20,-2000]))
    points,faces,record=build_surface_water(tmp_path)
    assert record['source_water_columns']==record['source_water_centers_above_final_ground']==16
    assert record['levels'][0]['source_components']==1
    assert 16 < record['levels'][0]['surface_area_m2'] < 25
    assert np.all(points[:,2]==3.5)
    query=SurfaceQuery(tmp_path)
    assert np.min(points[:,2]-query.heights(points[:,0],points[:,1]))>=-3e-6
    assert (np.cross(points[faces[:,1]]-points[faces[:,0]],points[faces[:,2]]-points[faces[:,0]])[:,2]>0).all()
    vertices,inverse=np.unique(np.round(points,5),axis=0,return_inverse=True)
    f=inverse[faces];edges=np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]))
    edges,counts=np.unique(np.sort(edges,axis=1),axis=0,return_counts=True)
    boundary=vertices[np.unique(edges[counts==1])]
    assert len(boundary)>0
    assert np.max(np.abs(boundary[:,2]-query.heights(boundary[:,0],boundary[:,1])))<3e-5
    # A filled source pond must fail explicitly, not disappear from the mesh.
    np.save(tmp_path/'height.npy',np.full((65,65),4.,np.float32))
    with pytest.raises(ValueError,match='covers visible source water'):
        build_surface_water(tmp_path)


def test_narrow_cut_cleanup_preserves_broad_valley_and_exposed_water():
    h=np.full((160,160),100.);h[:,48:112]=50.;h[:,16:20]=20.
    wet=np.zeros(h.shape,bool);wet[70:75,18:20]=True
    fields=dict(height=h,validity=np.ones(h.shape,bool),water_validity=wet,
        water_height=np.where(wet,30.,0.),biome=np.full(h.shape,'minecraft:plains'))
    before=h.copy();surface,retained,report=reconstruct_surface(fields)
    assert np.array_equal(h,before)
    assert np.array_equal(retained,wet)
    assert np.array_equal(surface[wet],h[wet])
    assert np.max(np.abs(surface[:,76:84]-50))<.2
    assert surface[30,18]>90
    assert surface.max()>99
    assert report['intentional_narrow_cut_water_columns_removed']==0


def test_waterfall_neighbours_do_not_become_invented_dams():
    h=np.full((12,12),20.);h[:,8:]=40.
    wet=np.zeros(h.shape,bool);wet[6,7]=True
    fields=dict(height=h,water_height=np.full(h.shape,30.))
    floor=source_bank_floor(fields,wet)
    assert np.isneginf(floor[6,6])
    assert floor[6,8]==pytest.approx(30.04)


def test_contour_relaxation_preserves_straight_cliff_and_elevation_offset():
    h=np.tile(np.where(np.arange(80)<40,100.,50.),(80,1))
    assert np.array_equal(relax_steep_contours(h,6),h)
    z,x=np.mgrid[-32:32,-32:32];hill=np.floor(100-np.hypot(x,z))
    relaxed=relax_steep_contours(hill,6)
    assert np.max(np.abs(relax_steep_contours(hill+1000,6)-1000-relaxed))<1e-9
    assert hill.min()<=relaxed.min()<=relaxed.max()<=hill.max()


def test_fine_interpolation_does_not_invent_spires_above_flat_lake_ice():
    h=np.full((12,12),48.);h[4:8,5:7]=63.
    axis=np.arange(.5,11.5001,.125)
    fine=sample_surface_grid(h,axis+1000,axis-2000,[1000,-2000])
    assert 48.<=fine.min()<=fine.max()<=63.
    assert np.array_equal(fine[::8,::8],h)
    assert np.array_equal(sample_surface_grid(h+100,axis,axis,[0,0])-100,fine)
    with pytest.raises(ValueError):
        sample_surface_grid(h,np.array([-.5]),np.array([1.5]),[0,0])


def test_observed_frozen_water_stays_supported_and_liquid_gaps_stay_open():
    h=np.full((80,80),48.);h[20:60,20:60]=63.
    names=np.full(h.shape,'minecraft:gravel',dtype='U30');names[20:60,20:60]='minecraft:ice'
    h[39:41,:]=48.;names[39:41,:]='minecraft:gravel'
    level=np.where(names=='minecraft:ice',62.,63.)
    fields=dict(height=h,validity=np.ones(h.shape,bool),substrate=names,
        water_height=level,water_validity=np.ones(h.shape,bool),
        biome=np.full(h.shape,'minecraft:frozen_ocean'))
    sheet=source_frozen_water(fields);smooth,wet,_=reconstruct_surface(fields)
    assert np.allclose(smooth[sheet],63.04,atol=1e-5)
    assert np.array_equal(smooth[wet],h[wet])
    assert not np.any(sheet&wet)
    # The material stage's visible-liquid mask deliberately excludes ice.
    # Original under-ice water must still identify exactly the same ice sheet.
    material=dict(fields,height=smooth,source_surface_height=h,
                  source_water_validity=fields['water_validity'],water_validity=wet)
    assert np.array_equal(source_frozen_water(material),sheet)
