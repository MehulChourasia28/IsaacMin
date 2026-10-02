"""Preview provenance and progress contracts; fixtures are not native evidence."""
import numpy as np
import pytest
from isaacmin.demo.construction import surface_preview,progress
from isaacmin.demo.jobs import Jobs
from isaacmin.io import atomic_json,read_json,sha256_file


def test_preview_retains_measured_heights_coordinates_water_and_read_only_inputs(tmp_path):
    build=tmp_path/'build';source=build/'source';source.mkdir(parents=True)
    heights=np.arange(64,dtype=np.float32).reshape(8,8)+20
    fields=dict(height=heights,min_xz=[100,200],biome=np.full((8,8),'minecraft:plains'),
        substrate=np.full((8,8),'minecraft:grass_block'),validity=np.ones((8,8),bool),
        water_validity=np.zeros((8,8),bool),water_height=heights+2)
    fields['water_validity'][3,4]=True
    np.savez(source/'macro_surface.npz',**fields)
    atomic_json(source/'macro_surface.json',{'scope':{'actual_center_xz':[104,204]}})
    atomic_json(build/'outdoor_build.json',{'identity':{'extent':8,'center':[103.2,203.7],'source':'fixture-source'}})
    terrain=build/'terrain';terrain.mkdir();final_height=np.arange(81,dtype=np.float32).reshape(9,9)+100
    np.save(terrain/'height.npy',final_height)
    atomic_json(terrain/'terrain.json',dict(bounds_source_xz=[100,200,108,208],origin_xyz=[103.2,0,203.7]))
    inputs=list(build.rglob('*'));before={str(p):sha256_file(p) for p in inputs if p.is_file()}
    surface_preview(build,tmp_path/'preview');surface_preview(build,tmp_path/'preview',final=True)
    preview=read_json(tmp_path/'preview/terrain.json');positions=np.array(preview['positions']).reshape(9,9,3)
    assert np.array_equal(positions[:,:,2],final_height)
    assert np.allclose(positions[0,0],[-3.2,3.7,100])
    assert np.allclose(positions[-1,-1],[4.8,-4.3,180])
    assert preview['water']==[float(fields['water_height'][3,4])]
    assert preview['input_manifest_sha256']==before[str(terrain/'terrain.json')]
    assert before=={str(p):sha256_file(p) for p in inputs if p.is_file()}
    fields['validity'][0,0]=False;np.savez(source/'macro_surface.npz',**fields)
    with pytest.raises(ValueError,match='unknown'):surface_preview(build,tmp_path/'unknown')


def test_only_completed_stages_advance_progress_and_failed_preview_does_not_pass(tmp_path):
    atomic_json(tmp_path/'state/demo_presets.json',{'presets':[{'id':'fixture'}]})
    jobs=Jobs(tmp_path);job=jobs.submit('fixture','convert','construction-fixture')
    directory=tmp_path/'artifacts/demo/jobs'/job['id']
    atomic_json(directory/'build/outdoor_build.json',dict(stages={'source':{},'objects':{}}))
    atomic_json(directory/'construction/terrain_error.json',dict(status='preview_unavailable',world_build_affected=False))
    atomic_json(directory/'build/assembled/preview/preview_checkpoint.json',dict(completed_frames=2,planned_frames=5))
    jobs.finish(job['id'],'running')
    data=progress(tmp_path,jobs,job['id'])
    assert data['completed_stages']==2 and data['stage_count']==11
    assert [s['id'] for s in data['stages'] if s['status']=='running']==['terrain']
    assert data['previews']=={} and data['preview_errors'][0]['status']=='preview_unavailable'
    assert data['captures']==[dict(kind='preview',frames=2,planned=5)]
    jobs.finish(job['id'],'failed',error='fixture interruption')
    data=progress(tmp_path,jobs,job['id'])
    assert data['completed_stages']==2 and not any(s['status']=='running' for s in data['stages'])
    other=jobs.submit('fixture','verify','not-construction')
    with pytest.raises(ValueError,match='conversion'):progress(tmp_path,jobs,other['id'])
