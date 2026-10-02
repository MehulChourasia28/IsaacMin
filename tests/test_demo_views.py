"""Camera/control fixture properties; these do not establish native-render quality."""
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from isaacmin.demo.jobs import Jobs
from isaacmin.demo.planner import tool_registry
from isaacmin.demo.views import plan_views, validate_view, resolve_target
from isaacmin.io import atomic_json, read_json, sha256_file
from isaacmin.outdoor_render import validate_capture_result


@pytest.fixture
def world(tmp_path):
    build=tmp_path/'build';terrain=build/'terrain';terrain.mkdir(parents=True)
    atomic_json(tmp_path/'state/demo_presets.json',dict(presets=[dict(id='fixture',build=str(build)),dict(id='other',build=str(build))]))
    scene=build/'fixture.usda';scene.write_text('Fixture bytes, not a native-rendered scene')
    atomic_json(build/'outdoor_build.json',dict(scene=str(scene),identity={'source':'fixture-source'},status='geometry_world'))
    atomic_json(terrain/'terrain.json',dict(bounds_source_xz=[100.,-60.,116.,-44.],origin_xyz=[108.,0.,-52.],recipe={'spacing_m':1}))
    np.save(terrain/'height.npy',np.full((17,17),10.,np.float32))
    np.savez(terrain/'material_fields.npz',height=np.full((16,16),10.),min_xz=[100,-60],
        water_height=np.full((16,16),10.),water_validity=np.zeros((16,16),bool),
        biome=np.full((16,16),'minecraft:plains'),substrate=np.full((16,16),'minecraft:grass_block'))
    (build/'objects').mkdir();atomic_json(build/'objects/objects.json',{'trees':[]})
    np.savez(build/'objects/canopy.npz',cover_fraction=np.zeros((16,16)))
    return tmp_path,build


def test_camera_frame_height_direction_and_deterministic_orbit(world):
    root,build=world
    poses=plan_views(root,build,dict(kind='ground',count=3,source_xz=[108,-52],heading_degrees=0),42)
    assert np.allclose(poses[0]['position'],[0,0,10.6])
    assert np.allclose(poses[0]['look_at'],[0,8,10.4])
    assert np.allclose(poses[1]['look_at'][:2],[8*np.sin(2*np.pi/3),8*np.cos(2*np.pi/3)])
    options=dict(kind='aerial',count=3,source_xz=[108,-52],heading_degrees=90,height_m=30)
    a=plan_views(root,build,options,42);b=plan_views(root,build,options,42)
    assert a==b and len(a)==3
    assert a[0]['position'][0]<0 and abs(a[0]['position'][1])<1e-6
    assert np.allclose(a[0]['look_at'],[0,0,10]) and a[0]['position'][2]==40
    assert validate_view(dict(kind='aerial',count=1.0))['count']==1


@pytest.mark.parametrize('options',[
    dict(kind='ground',count=0),dict(kind='aerial',count=7),dict(kind='ground',height_m=20),
    dict(kind='aerial',source_xz=[float('nan'),0]),dict(kind='aerial',heading_degrees=float('inf')),
    dict(kind='aerial',height_m=601),dict(kind='ground',script='unregistered'),dict(kind='aerial',count=True)])
def test_invalid_camera_requests_are_rejected(options):
    with pytest.raises(ValueError):validate_view(options)


def test_ground_point_requires_region_water_slope_and_tree_clearance(world):
    root,build=world;options=dict(kind='ground',source_xz=[108,-52])
    with pytest.raises(ValueError,match='outside'):
        plan_views(root,build,dict(kind='ground',source_xz=[100,-52]),1)
    path=build/'terrain/material_fields.npz'
    with np.load(path) as data:fields={k:data[k] for k in data.files}
    fields['water_validity'][8,8]=True;np.savez(path,**fields)
    with pytest.raises(ValueError,match='water'):plan_views(root,build,options,1)
    fields['water_validity'][8,8]=False;np.savez(path,**fields)
    np.save(build/'terrain/height.npy',np.tile(np.arange(17,dtype=float),(17,1))+10)
    with pytest.raises(ValueError,match='steeper'):plan_views(root,build,options,1)
    np.save(build/'terrain/height.npy',np.full((17,17),10.,np.float32))
    atomic_json(build/'objects/objects.json',dict(trees=[dict(source_x=108,source_z=-52)]))
    with pytest.raises(ValueError,match='tree clearance'):plan_views(root,build,options,1)


def test_latest_view_request_is_frozen_on_first_submission(world):
    root,build=world;jobs=Jobs(root)
    def completed(key):
        job=jobs.submit('fixture','convert',key)
        destination=root/'artifacts/demo/jobs'/job['id']/'build'
        shutil.copytree(build,destination)
        jobs.finish(job['id'],'complete',result={'status':'fixture_only'})
        return job['id']
    first=completed('first-conversion')
    requested=jobs.submit('fixture','views','frozen-view-operation',view={'kind':'aerial'},target='latest')
    second=completed('second-conversion')
    assert first!=second and jobs.latest_conversion('fixture')==second
    replayed=jobs.submit('fixture','views','frozen-view-operation',view={'kind':'aerial'},target='latest')
    assert replayed['id']==requested['id'] and replayed['target']==first
    with pytest.raises(ValueError,match='another request'):
        jobs.submit('fixture','views','frozen-view-operation',view={'kind':'ground'},target='latest')
    with pytest.raises(ValueError,match='same world'):resolve_target(root,jobs,'other',first)
    with pytest.raises(ValueError,match='Invalid view target'):resolve_target(root,jobs,'fixture',{})
    for index in range(62):jobs.submit('fixture','verify','recent-fixture-'+str(index))
    assert not any(j['id']==second for j in jobs.list())
    assert resolve_target(root,jobs,'fixture','latest')[1]==second


def test_controls_are_transactional_idempotent_and_bound_to_selected_world(world):
    root,_=world;jobs=Jobs(root);a=jobs.submit('fixture','verify','control-job-a')
    b=jobs.submit('other','verify','control-job-b')
    registry=tool_registry(root,jobs,'fixture','job_'+'f'*32)
    args=dict(id=a['id'],action='cancel');cancelled=registry.call('control_job',args)
    assert cancelled['status']=='cancelled'
    assert registry.call('control_job',args)==cancelled
    assert registry.call('control_job',dict(id=a['id'],action='resume'))['status']=='queued'
    # A replayed cancellation must not cancel the subsequently resumed job.
    assert registry.call('control_job',args)==cancelled
    assert jobs.get(a['id'])['status']=='queued'
    with pytest.raises(ValueError,match='selected world'):
        registry.call('control_job',dict(id=b['id'],action='cancel'))
    jobs.finish(a['id'],'running')
    with pytest.raises(ValueError,match='Only queued'):jobs.control(a['id'],'cancel','different-operation')
    assert jobs.get(a['id'])['status']=='running'


def test_sensor_bytes_and_poses_are_validated_before_completed_capture_reuse(tmp_path):
    pose=dict(kind='static',position=[0,0,.6]);frame={'pose':pose}
    for sensor in ('rgb','depth','instance_segmentation'):
        path=tmp_path/(sensor+'.fixture');path.write_bytes(sensor.encode())
        frame[sensor]=path.name;frame[sensor+'_sha256']=sha256_file(path)
    report=tmp_path/'preview_result.json'
    atomic_json(report,dict(status='actual_visual_preview_complete',scene_sha256='fixture-scene',frames=[frame]))
    validate_capture_result(report,'fixture-scene',[pose])
    with pytest.raises(ValueError,match='camera poses'):validate_capture_result(report,'fixture-scene',[])
    (tmp_path/'depth.fixture').write_bytes(b'corruption')
    with pytest.raises(ValueError,match='sensor bytes'):validate_capture_result(report,'fixture-scene',[pose])


def test_assistant_context_is_fixed_to_its_selected_conversion(world):
    root,build=world;jobs=Jobs(root)
    conversion=jobs.submit('fixture','convert','context-conversion')
    other=jobs.submit('other','convert','other-conversion')
    agent=jobs.submit('fixture','agent','context-agent',prompt='More views',context_job=conversion['id'])
    assert agent['context_job']==conversion['id']
    with pytest.raises(ValueError,match='selected world'):
        jobs.submit('fixture','agent','wrong-context',prompt='More views',context_job=other['id'])
    with pytest.raises(ValueError,match='selected world'):
        jobs.submit('fixture','verify','not-agent-context',context_job=conversion['id'])
    registry=tool_registry(root,jobs,'fixture',agent['id'],conversion['id'])
    with pytest.raises(ValueError,match='completed conversion'):
        registry.call('request_views',dict(preset='fixture',view={'kind':'aerial'}))
    with pytest.raises(ValueError,match='saved preset'):
        registry.call('start_job',dict(preset='fixture',action='verify'))
    destination=root/'artifacts/demo/jobs'/conversion['id']/'build';shutil.copytree(build,destination)
    jobs.finish(conversion['id'],'complete',result={'status':'fixture_only'})
    # The same saved assistant still refers to the original conversion after it finishes.
    assert jobs.submit('fixture','agent','context-agent',prompt='More views',context_job=conversion['id'])['id']==agent['id']
    requested=registry.call('request_views',dict(preset='fixture',view={'kind':'aerial'}))
    assert requested['target']==conversion['id']
    assert registry.call('request_views',dict(preset='fixture',view={'kind':'aerial'}))['id']==requested['id']
