"""Prepare declared native asset candidates without inheriting local credentials."""
from pathlib import Path
import argparse
import json
from isaacmin.assets.prepare import prepare_materials, prepare_native_catalogue, prepare_lighting
from isaacmin.assets.candidates import candidate_lighting
from isaacmin.assets.provenance import finalize_catalogue


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--workspace',type=Path,default=Path.cwd())
    args=parser.parse_args()
    workspace=args.workspace.resolve()
    # Preserve pinned provider normalization independently of the USD exporter.
    blender=workspace/'.tools/native/usr/bin/blender'
    environment={'LD_LIBRARY_PATH':str(workspace/'.tools/native/usr/lib/aarch64-linux-gnu')+':'+str(workspace/'.tools/native/usr/lib'),
                 'BLENDER_SYSTEM_SCRIPTS':str(workspace/'.tools/native/usr/share/blender/scripts'),
                 'BLENDER_SYSTEM_DATAFILES':str(workspace/'.tools/native/usr/share/blender/datafiles')}
    report=prepare_native_catalogue(workspace,blender,environment)
    prepare_materials(workspace)
    prepare_lighting(workspace)
    candidate_lighting(workspace)
    finalize_catalogue(workspace,fetch_licences=False)
    print(json.dumps({'status':report['status'],'normalized_assets':len(report['assets']),
                      'failures':report['failures'],'isaac_qualification':'not_run'}))
    return 0 if not report['failures'] else 2


if __name__=='__main__':
    raise SystemExit(main())
