"""Acquire explicitly selected original CC0 scans through the verified API."""
from pathlib import Path
import argparse
from concurrent.futures import ThreadPoolExecutor
from isaacmin.assets.providers import PolyHaven
from isaacmin.io import atomic_json,sha256_file,utc_now


def acquire(root,output,asset_id):
    provider=PolyHaven(root)
    catalogue,_=provider.catalogue('textures')
    if asset_id not in catalogue:raise ValueError('Asset missing from provider catalogue: '+asset_id)
    dimensions=[v/1000 for v in catalogue[asset_id].get('dimensions',[])]
    if len(dimensions)!=2 or min(dimensions)<=0:raise ValueError('Unknown original physical scan size')
    resolution=next((r for r in (4,8,16) if r*1024/max(dimensions)>=1024),None)
    if resolution is None:raise ValueError('Scan cannot satisfy frozen native texel density')
    roles={'Diffuse':'base_color','Rough':'roughness','nor_gl':'normal','Displacement':'height'}
    record=provider.acquire(asset_id,'textures',[(role,str(resolution)+'k','png') for role in roles])
    material=dict(asset_id=asset_id,name=asset_id,licence='CC0-1.0',source_page=record['source_page'],
        authors=record['authors'],repeat_m=dimensions[0],physical_dimensions_m=dimensions,
        upstream_revision=record['upstream_revision'],original_channel_proof={})
    for channel in record['files']:
        role=roles[channel['role']];image=channel['image_validation']
        if image['status']!='pass' or min(image['width'],image['height'])/max(dimensions)<1024:
            raise ValueError('Original scan decode or metric density failed')
        material[role]=channel['path']
        material['original_channel_proof'][role]=dict(sha256=channel['sha256'],width=image['width'],height=image['height'])
    atomic_json(output/(asset_id+'.json'),dict(provider_record=record,material=material,isaac_qualification='not_run'))
    print(asset_id,'original maps acquired',flush=True)
    return material


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--ids',nargs='+',required=True)
    a=p.parse_args();root=Path.cwd();output=a.output.resolve();output.mkdir(parents=True,exist_ok=False)
    # Two bounded independent public transfers; no native asset code executes.
    with ThreadPoolExecutor(max_workers=2) as pool:
        materials=list(pool.map(lambda aid:acquire(root,output,aid),a.ids))
    atomic_json(output/'materials.json',dict(at_utc=utc_now(),materials=materials,status='original_scans_acquired',
        qualification='not_run',producer_sha256=sha256_file(Path(__file__))))
