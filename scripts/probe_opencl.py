#!/usr/bin/env python3
"""Read-only OpenCL enumeration and a real bounded float kernel, no CUDA assumptions."""
import ctypes as c
import json
from pathlib import Path

def main():
    cl=c.CDLL('libOpenCL.so.1')
    ptr=c.c_void_p;uint=c.c_uint;size=c.c_size_t;integer=c.c_int
    signatures={
      'clGetPlatformIDs':(integer,[uint,c.POINTER(ptr),c.POINTER(uint)]),
      'clGetDeviceIDs':(integer,[ptr,c.c_ulong,uint,c.POINTER(ptr),c.POINTER(uint)]),
      'clGetDeviceInfo':(integer,[ptr,uint,size,ptr,c.POINTER(size)]),
      'clCreateContext':(ptr,[ptr,uint,c.POINTER(ptr),ptr,ptr,c.POINTER(integer)]),
      'clCreateCommandQueue':(ptr,[ptr,ptr,c.c_ulong,c.POINTER(integer)]),
      'clCreateBuffer':(ptr,[ptr,c.c_ulong,size,ptr,c.POINTER(integer)]),
      'clCreateProgramWithSource':(ptr,[ptr,uint,c.POINTER(c.c_char_p),c.POINTER(size),c.POINTER(integer)]),
      'clBuildProgram':(integer,[ptr,uint,c.POINTER(ptr),c.c_char_p,ptr,ptr]),
      'clCreateKernel':(ptr,[ptr,c.c_char_p,c.POINTER(integer)]),
      'clSetKernelArg':(integer,[ptr,uint,size,ptr]),
      'clEnqueueNDRangeKernel':(integer,[ptr,ptr,uint,ptr,c.POINTER(size),ptr,uint,ptr,ptr]),
      'clEnqueueReadBuffer':(integer,[ptr,ptr,uint,size,size,ptr,uint,ptr,ptr])}
    for name,(restype,args) in signatures.items():
        fn=getattr(cl,name);fn.restype=restype;fn.argtypes=args
    count=uint();status=cl.clGetPlatformIDs(0,None,c.byref(count))
    if status or not count.value:return {'status':'blocked_dependency','opencl_status':status,'platforms':count.value}
    platforms=(ptr*count.value)();cl.clGetPlatformIDs(count,platforms,None)
    devices=[]
    def check(code):
        if code:raise RuntimeError(f'OpenCL status {code}')
    for platform in platforms:
        n=uint();status=cl.clGetDeviceIDs(platform,0xffffffff,0,None,c.byref(n))
        if status:continue
        ids=(ptr*n.value)();check(cl.clGetDeviceIDs(platform,0xffffffff,n,ids,None))
        for device in ids:
            name=c.create_string_buffer(1024);check(cl.clGetDeviceInfo(device,0x102b,1024,name,None))
            d=(ptr*1)(device);error=integer()
            ctx=cl.clCreateContext(None,1,d,None,None,c.byref(error));check(error.value)
            queue=cl.clCreateCommandQueue(ctx,device,0,c.byref(error));check(error.value)
            data=(c.c_float*64)(*range(64))
            memory=ptr(cl.clCreateBuffer(ctx,1|32,c.sizeof(data),data,c.byref(error)));check(error.value)
            source=c.c_char_p(b'__kernel void probe(__global float* x){size_t i=get_global_id(0);x[i]=2.0f*x[i]+1.0f;}')
            program=cl.clCreateProgramWithSource(ctx,1,c.byref(source),None,c.byref(error));check(error.value)
            check(cl.clBuildProgram(program,1,d,None,None,None))
            kernel=cl.clCreateKernel(program,b'probe',c.byref(error));check(error.value)
            check(cl.clSetKernelArg(kernel,0,c.sizeof(memory),c.byref(memory)))
            extent=(size*1)(64)
            check(cl.clEnqueueNDRangeKernel(queue,kernel,1,None,extent,None,0,None,None))
            check(cl.clEnqueueReadBuffer(queue,memory,1,0,c.sizeof(data),data,0,None,None))
            error=max(abs(float(v)-(2*i+1)) for i,v in enumerate(data))
            devices.append({'name':name.value.decode(),'kernel_samples':64,'max_absolute_error':error,
                            'status':'pass' if error==0 else 'fail'})
    return {'status':'pass' if devices and all(d['status']=='pass' for d in devices) else 'fail',
            'devices':devices,'terrain_backend':'qualified_native_HighMap_CPU','gpu_erosion':'not_run'}

if __name__=='__main__':
    try:result=main()
    except Exception as exc:result={'status':'failed','error':str(exc)}
    target=Path(__file__).resolve().parents[1]/'artifacts/bootstrap/opencl.json'
    target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
