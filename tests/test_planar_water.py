"""Geometry equivalence of flat water storage; not renderer appearance proof."""
import numpy as np
import pytest
from isaacmin.assembly.planar_water import compact_planar_water
from isaacmin.assembly.surface_water import _positive_triangles
from isaacmin.terrain.surface_navigation import grid_triangles


@pytest.mark.parametrize('kind',['full','empty','shore','islands','checker','random'])
def test_planar_union_area_and_every_boundary_triangle_are_retained(kind):
    z,x=np.mgrid[:33,:48];p=np.stack((x*.125-13.38,-z*.125+3.07,np.full(x.shape,63.)),axis=-1)
    f={'full':np.ones(x.shape),'empty':-np.ones(x.shape),'shore':x-20.31,
       'islands':((x-22)**2+(z-16)**2)-100,'checker':np.where((x+z)%2,1.,-1.),
       'random':np.random.default_rng(721).normal(size=x.shape)}[kind]
    original,area=_positive_triangles(p.reshape(-1,3),grid_triangles(*x.shape),f.ravel())
    actual,other,proof=compact_planar_water(p,f,_positive_triangles)
    assert abs(area-other)<1e-10
    assert np.all(actual[:,:,2]==63.)
    # Every clipped/non-grid shoreline point must remain byte-identical.
    def shore_vertices(t):
        v=t.reshape(-1,3);grid=((v[:,0]+13.38)/.125,(3.07-v[:,1])/.125)
        non_grid=(abs(grid[0]-np.round(grid[0]))>1e-8)|(abs(grid[1]-np.round(grid[1]))>1e-8)
        return {tuple(v) for v in v[non_grid]}
    assert shore_vertices(original)<=shore_vertices(actual)
    if kind=='full':assert len(actual)<len(original)/8
