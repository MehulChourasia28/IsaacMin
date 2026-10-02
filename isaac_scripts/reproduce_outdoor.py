"""Replay a portable outdoor delivery without source data or generator imports."""
import argparse
import json
import sys
from pathlib import Path
sys.dont_write_bytecode=True
from reproduce_package import digest, local, verify, run


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--isaac-python',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--views',type=int,nargs='+',help='Optional subset, zero-based; subset never establishes complete coverage')
    args=parser.parse_args()
    root,output=args.package.resolve(),args.output.resolve()
    if output.is_relative_to(root):raise ValueError('Replay output must be outside the package')
    output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((root/'package.json').read_text());manifest_hash=digest(root/'package.json')
    verify(root,manifest['files'])
    request=json.loads((root/'reproduction/request.json').read_text())
    selected=args.views if args.views is not None else list(range(len(request['poses'])))
    if not selected or len(set(selected))!=len(selected) or any(i<0 or i>=len(request['poses']) for i in selected):
        raise ValueError('Replay view indices are invalid')
    request.update(scene=str(local(root,manifest['entrypoint'])),output=str(output/'capture'),
        poses=[request['poses'][i] for i in selected],preserve_authored_scene=True)
    request['hdri']=str(local(root,request['hdri']))
    request_path=output/'capture_request.json';request_path.write_text(json.dumps(request,indent=2)+'\n')
    (output/'selected_views.json').write_text(json.dumps(selected)+'\n')
    runtime=str(args.isaac_python.resolve());probe=root/'reproduction'
    run([runtime,str(probe/'capture_outdoor_visual.py'),'--request',str(request_path)],output,'capture')
    verify(root,manifest['files'])
    if digest(root/'package.json')!=manifest_hash:raise ValueError('Package changed during replay')
    run([runtime,str(probe/'compare_outdoor_reproduction.py'),'--package',str(root),'--output',str(output)],output,'comparison')


if __name__=='__main__':main()
