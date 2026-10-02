"""Retain original ambientCG channels and explicit, unqualified scale choices."""
from pathlib import Path
import argparse
from isaacmin.assets.providers import AmbientCG
from isaacmin.io import atomic_json, sha256_file, utc_now


def acquire(root, asset_id, repeat_m, output):
    if not 0 < repeat_m <= 4:
        raise ValueError('4K material must retain at least 1024 original texels/m')
    provider=AmbientCG(root);asset,snapshot=provider.asset_by_id(asset_id)
    record=provider.acquire(asset,snapshot,'4K-PNG')
    material=dict(asset_id=asset_id,name=asset['title'],licence=record['licence'],
        source_page=record['source_page'],authors=record['authors'],repeat_m=repeat_m,
        upstream_revision=record['upstream_revision'],original_channel_proof={},
        technique=asset['technique'],physical_dimensions_m=None,
        scale_provenance='explicit construction assumption; provider supplies no measured scale',
        scale_qualification='not_run',provider_manifest=record)
    for role,suffix in [('base_color','Color'),('roughness','Roughness'),('normal','NormalGL'),('height','Displacement')]:
        entries=[f for f in record['extracted_files'] if f['path'].endswith('_'+suffix+'.png')]
        if len(entries)!=1:raise ValueError('Missing or ambiguous original channel: '+role)
        entry=entries[0];proof=entry['image_validation']
        if proof['status']!='pass' or min(proof['width'],proof['height'])/repeat_m<1024:
            raise ValueError('Original decode or native texel density failed')
        path=Path(record['extracted_directory'])/entry['path']
        if sha256_file(path)!=entry['sha256']:raise ValueError('Original channel changed')
        material[role]=str(path)
        material['original_channel_proof'][role]=dict(sha256=entry['sha256'],width=proof['width'],height=proof['height'])
    output.mkdir(parents=True,exist_ok=False)
    atomic_json(output/'material.json',dict(at_utc=utc_now(),material=material,
        technique=asset['technique'],qualification='not_run',producer_sha256=sha256_file(Path(__file__))))
    return material


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--id',required=True)
    p.add_argument('--repeat-metres',type=float,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();acquire(Path.cwd(),a.id,a.repeat_metres,a.output)
    print('Original material retained; physical scale and native appearance are not qualified.')
