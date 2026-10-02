#!/usr/bin/env python3
"""Private numerical worker; never imports asset files or receives credentials."""
import argparse,json
from pathlib import Path
from isaacmin.volumes.precision_algorithms import condition_coarse,finalize_native_topology,subdivide_double

parser=argparse.ArgumentParser();parser.add_argument('--request',type=Path,required=True)
args=parser.parse_args();request=json.loads(args.request.read_text())
if request['mode']=='coarse':result=condition_coarse(request,request['output'])
elif request['mode']=='subdivision':result=subdivide_double(request,request['output'])
elif request['mode']=='final':result=finalize_native_topology(request,request['output'])
else:raise ValueError('Unknown native precision operation')
print(json.dumps({'status':result['status'],'output':request['output']}))
