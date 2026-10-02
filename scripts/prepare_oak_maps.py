"""Copy verified CC0 masters and derive leaf silhouettes from their opacity map."""
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image


def prepare(workspace: Path):
    catalogue=json.loads((workspace/'state/asset_catalogue.json').read_text())['assets']
    output=workspace/'assets/procedural/oak/maps';output.mkdir(parents=True,exist_ok=True)
    records=[]
    for asset_id,prefix in [('LeafSet016','oak_leaf'),('Bark012','oak_bark')]:
        asset=next(a for a in catalogue if a['asset_id']==asset_id)
        folder=Path(asset['extracted_directory'])
        for role,suffix in [('base_color','Color'),('roughness','Roughness'),('normal','NormalGL'),('opacity','Opacity')]:
            matching=[m for m in asset['extracted_files'] if m['path'].endswith('_'+suffix+'.png')]
            if not matching:continue
            if len(matching)!=1:raise ValueError('Ambiguous verified map role')
            source=folder/matching[0]['path']
            if hashlib.sha256(source.read_bytes()).hexdigest()!=matching[0]['sha256']:raise ValueError('Master checksum changed')
            target=output/(prefix+'_'+role+'.png');shutil.copy2(source,target)
            records.append({'path':str(target),'source_path':str(source),'sha256':matching[0]['sha256'],'role':role,
                            'provider':'ambientcg','asset_id':asset_id,'licence':'CC0-1.0','operation':'unchanged copy'})
    opacity=np.asarray(Image.open(output/'oak_leaf_opacity.png').convert('L'))>127
    h,w=opacity.shape
    profiles=[]
    for row in range(2):
        for col in range(3):
            x0,x1=int(col*w/3),int((col+1)*w/3)
            y0,y1=int(row*h/2),int((row+1)*h/2)
            crop=opacity[y0:y1,x0:x1]
            ys=np.flatnonzero(crop.sum(axis=1)>12)
            top,bottom=int(ys.min()),int(ys.max())
            length=bottom-top
            basex=np.flatnonzero(crop[bottom]).mean()+x0
            rows=[]
            for j in range(49):
                py=round(bottom-j*length/48)
                xs=np.flatnonzero(crop[py])
                if not len(xs):raise ValueError('Disconnected photo silhouette needs explicit processing')
                left,right=float(xs.min()+x0),float(xs.max()+x0)
                center=(left+right)/2
                rows.append({'t':j/48,'vertices':[[((px-basex)/length),px/w,1-(py+y0)/h] for px in (left,center,right)]})
            profiles.append({'atlas_cell':[col,row],'rows':rows,'leaf_aspect_ratio':float(crop.sum(axis=0).astype(bool).sum()/length)})
    (output/'leaf_profiles.json').write_text(json.dumps(profiles))
    report={'classification':'CC0_photogrammetric_oak_maps_and_opacity_derived_geometry_profiles','source_assets':['LeafSet016','Bark012'],
            'files':records,'leaf_profile_count':len(profiles),'normal_convention':'OpenGL','qualification':'not_run',
            'limitations':['Atlas holes and fine edge hairs need target opacity qualification','Physical leaf size is inferred8-14cm; provider scale metadata insufficient'],
            'profile_sha256':hashlib.sha256((output/'leaf_profiles.json').read_bytes()).hexdigest()}
    (output/'maps_manifest.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    import sys
    prepare(Path(sys.argv[1]).resolve())
