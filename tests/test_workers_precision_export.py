"""Actual isolated precision-worker regressions; no world qualification implied."""
from pathlib import Path
import subprocess

import numpy as np
import pytest

from isaacmin.security import worker_environment

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT/'.tools/native_precision/write_native_obj').is_file(),reason='Private precision writer unavailable')
def test_actual_obj_writer_preserves_sub_decimal_float32_geometry(tmp_path):
    # Six-decimal output loses these distinct native points and their tiny edge.
    points=np.array([[0.00000017,-32.93000030517578,72.0],
                     [0.00000018,-32.93000030517578,72.0],
                     [0.00000017,-32.929996490478516,72.0]],dtype=np.float32)
    faces=np.array([[0,1,2]],dtype=np.int32)
    vp=tmp_path/'points.npy';fp=tmp_path/'faces.npy';obj=tmp_path/'ground.obj'
    np.save(vp,points);np.save(fp,faces)
    v=np.load(vp,mmap_mode='r');f=np.load(fp,mmap_mode='r')
    result=subprocess.run([str(ROOT/'.tools/native_precision/write_native_obj'),str(vp),str(v.offset),str(len(v)),
                           str(fp),str(f.offset),str(len(f)),str(obj)],env=worker_environment(),capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    lines=obj.read_text().splitlines()
    observed=np.array([[float(x) for x in line.split()[1:]] for line in lines if line.startswith('v ')])
    triangles=np.array([[int(x)-1 for x in line.split()[1:]] for line in lines if line.startswith('f ')])
    np.testing.assert_array_equal(observed,points.astype(np.float64))
    np.testing.assert_array_equal(triangles,faces)
    assert np.linalg.norm(np.cross(observed[1]-observed[0],observed[2]-observed[0]))>0


@pytest.mark.skipif(not (ROOT/'.tools/native_precision/subdivide_double').is_file(),reason='Private subdivision worker unavailable')
def test_actual_bilinear_three_levels_preserve_source_plane_and_closed_topology(tmp_path):
    off=tmp_path/'tetra.off'
    off.write_text('OFF\n4 4 0\n0 0 72\n1 0 72\n0 1 72\n0 0 71\n3 0 1 2\n3 0 3 1\n3 1 3 2\n3 2 3 0\n')
    vp=tmp_path/'v.npy';fp=tmp_path/'q.npy'
    result=subprocess.run([str(ROOT/'.tools/native_precision/subdivide_double'),str(off),'3',str(vp),str(fp)],
                          env=worker_environment(),capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    points=np.load(vp);faces=np.load(fp)
    assert points.shape==(194,3) and faces.shape==(192,4)
    assert points.dtype==np.float32 and faces.dtype==np.int32
    assert np.isfinite(points).all()
    assert (points[:,2]<=72).all() and (points[:,2]>=71).all()
    assert np.count_nonzero(points[:,2]==72)>40
    from collections import Counter
    edges=Counter(tuple(sorted((int(a),int(b)))) for face in faces for a,b in zip(face,np.roll(face,-1)))
    assert set(edges.values())=={2}
    assert len(points)-len(edges)+len(faces)==2
