"""Acquire original CC0 channels for distinct outdoor biome candidates."""
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor
from isaacmin.assets.providers import PolyHaven
from isaacmin.io import atomic_json,sha256_file,utc_now

parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();root=Path.cwd();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
ids=['sand_03','snow_02','rocky_gravel','forest_ground_05','withered_grass','mud_forest']
def acquire(aid):
    provider=PolyHaven(root)
    record=provider.acquire(aid,'textures',[(role,'4k','png') for role in ('Diffuse','Rough','nor_gl','Displacement')])
    roles={'Diffuse':'base_color','Rough':'roughness','nor_gl':'normal','Displacement':'height'}
    dimensions=record['dimensions_m']
    if not dimensions or min(dimensions)<=0 or max(dimensions)>4:
        raise ValueError('Asset needs a different native resolution for its actual physical scale')
    material={'asset_id':aid,'name':aid,'licence':'CC0-1.0','source_page':record['source_page'],
              'authors':record['authors'],'repeat_m':dimensions[0],'physical_dimensions_m':dimensions,
              'upstream_revision':record['upstream_revision'],'original_channel_proof':{}}
    for channel in record['files']:
        role=roles[channel['role']];material[role]=channel['path']
        image=channel['image_validation']
        if image['status']!='pass' or min(image['width'],image['height'])<4096:raise ValueError('Full native original decode failed')
        material['original_channel_proof'][role]={'sha256':channel['sha256'],'width':image['width'],'height':image['height']}
    atomic_json(out/(aid+'.json'),{'provider_record':record,'material':material,'isaac_qualification':'not_run'})
    print({'acquired_original_channels':aid,'resolution':'4k'},flush=True)
    return material
with ThreadPoolExecutor(max_workers=2) as pool:materials=list(pool.map(acquire,ids))
atomic_json(out/'materials.json',{'at_utc':utc_now(),'materials':materials,'status':'original_materials_acquired',
    'qualification':'not_run','scope':'biome-specific candidates; does not qualify the corresponding biome',
    'producer_sha256':sha256_file(Path(__file__))})
