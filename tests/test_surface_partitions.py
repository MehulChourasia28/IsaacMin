"""Partition ownership properties only; native proof is recorded separately."""
import numpy as np
import pytest
from isaacmin.terrain.surface_partitions import source_cell_grid,source_windows,partition_windows,grid_triangles


@pytest.mark.parametrize('cell',[.12,.6,.22,1.,4.])
@pytest.mark.parametrize('negative_z',[False,True])
def test_negative_source_coordinate_cells_have_exactly_one_partition(cell,negative_z):
    bounds=[-137.3,-205.7,121.1,93.3]
    full=source_cell_grid(bounds,bounds,cell,negative_z=negative_z)
    expected=np.column_stack([v.ravel() for v in full])
    rows=[]
    for part in source_windows(bounds,91):
        grid=source_cell_grid(part,bounds,cell,negative_z=negative_z)
        rows.append(np.column_stack([v.ravel() for v in grid]))
    actual=np.concatenate(rows)
    assert len(actual)==len(expected)
    # Sorting compares cell identities, independent of storage partition order.
    assert np.array_equal(actual[np.lexsort(actual.T)],expected[np.lexsort(expected.T)])


def test_partition_triangles_cover_global_grid_exactly_once():
    nz,nx=17,24;parts=[]
    for part in partition_windows((nz,nx),7):
        z0,z1,x0,x1=part['core']
        ids=(np.arange(z0,z1+1)[:,None]*nx+np.arange(x0,x1+1)).ravel()
        parts.append(ids[grid_triangles(z1-z0+1,x1-x0+1)])
    actual=np.concatenate(parts);expected=grid_triangles(nz,nx)
    assert np.array_equal(actual[np.lexsort(actual.T)],expected[np.lexsort(expected.T)])
