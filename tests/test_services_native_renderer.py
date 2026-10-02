"""Capture schema validation; these fixtures are not native renderer evidence."""
from copy import deepcopy
import pytest
from isaacmin.assets.usd_provenance import validated_native_renderer
from isaacmin.assets.network import ServiceError


def path_trace_record():
    return {'renderer':'PathTracing','renderer_recipe':'pathtracing_1024',
            'path_tracing_sample_budget':1024,
            'frames':[{'path_tracing_sample_budget':1024},{'path_tracing_sample_budget':1024}],
            'native_renderer_settings':{
                '/rtx/rendermode':'PathTracing','/rtx/post/aa/op':0,
                '/rtx/pathtracing/clampSpp':64,'/rtx/pathtracing/spp':64,
                '/rtx/pathtracing/totalSpp':1024,'/rtx/pathtracing/maxBounces':12,
                '/rtx/pathtracing/maxSpecularAndTransmissionBounces':12,
                '/rtx/pathtracing/maxVolumeBounces':4,
                '/rtx/pathtracing/adaptiveSampling/enabled':False,
                '/rtx/pathtracing/optixDenoiser/enabled':False,
                '/rtx/post/motionblur/enabled':False}}


def test_retained_legacy_and_observed_path_trace_protocols_are_distinct():
    assert validated_native_renderer({'renderer':'RayTracedLighting'})=='RayTracedLighting'
    assert validated_native_renderer(path_trace_record())=='PathTracing'
    with pytest.raises(ServiceError):validated_native_renderer({'renderer':'PathTracing'})


@pytest.mark.parametrize('change',('missing_frame_budget','undersampled','adaptive','wrong_type','empty_frames'))
def test_new_renderer_label_cannot_hide_unmeasured_or_changed_sampling(change):
    record=deepcopy(path_trace_record())
    if change=='missing_frame_budget':record['frames'][1].pop('path_tracing_sample_budget')
    elif change=='undersampled':record['path_tracing_sample_budget']=64
    elif change=='adaptive':record['native_renderer_settings']['/rtx/pathtracing/adaptiveSampling/enabled']=True
    elif change=='wrong_type':record['native_renderer_settings']['/rtx/post/aa/op']=False
    elif change=='empty_frames':record['frames']=[]
    with pytest.raises(ServiceError):validated_native_renderer(record)
