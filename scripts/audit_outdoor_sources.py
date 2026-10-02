"""Inspect registered real saves without modifying them or claiming render proof."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from isaacmin.io import atomic_json,read_json,sha256_file,utc_now
from isaacmin.outdoor_pipeline import source_center
from isaacmin.source.snapshot import snapshot_world,verify_source_unchanged
from isaacmin.source.inventory import inspect_source
from isaacmin.source.macro import extract_macro_surface
from isaacmin.source.provenance import begin_source_read
from isaacmin.terrain.biomes import biome_inventory
import numpy as np


def audit(workspace,output):
    workspace,output=Path(workspace).resolve(),Path(output).resolve()
    registry=read_json(workspace/'state/source_worlds.json')
    library=read_json(workspace/'state/surface_asset_library.json')
    output.mkdir(parents=True,exist_ok=True)
    producer=sha256_file(Path(__file__))
    result=dict(schema_version=1,created_at_utc=utc_now(),worlds=[],
        registry_sha256=sha256_file(workspace/'state/source_worlds.json'),
        producer_sha256=producer,scope='actual source inspection; not native build or visual qualification')
    for entry in registry['worlds']:
        destination=output/entry['id'];destination.mkdir(exist_ok=True)
        def phase(name):
            atomic_json(output/'progress.json',dict(world=entry['id'],phase=name,at_utc=utc_now()))
            print(entry['id'],name,flush=True)
        phase('snapshot')
        snapshot=snapshot_world(workspace/entry['source'],workspace/'work/snapshots')
        atomic_json(destination/'snapshot.json',snapshot)
        source=Path(snapshot['snapshot_path'])
        center,provenance=source_center(workspace,workspace/entry['source'],metadata_world=source)
        phase('inventory')
        inventory_path=destination/'inventory/source_inventory.json'
        if inventory_path.is_file():
            inventory=read_json(inventory_path)
            if inventory['source_snapshot_sha256']!=snapshot['save_sha256']:
                raise ValueError('Source changed; use a new audit destination')
        else:
            inventory=inspect_source(source,destination/'inventory',snapshot_sha256=snapshot['save_sha256'])
        row=dict(id=entry['id'],label=entry['label'],source=entry['source'],center_minecraft_xz=list(center),
            center_provenance=provenance,save_sha256=snapshot['save_sha256'],
            level_dat_sha256=sha256_file(source/'level.dat'),data_version=inventory['data_version'],
            inventory=str(inventory_path.relative_to(workspace)),inventory_status=inventory['status'],
            source_surveys={},native_build='not_run',appearance='not_qualified')
        for name,extent,spacing in [('center_context',384,1),('regional_context',2048,8)]:
            phase(name)
            folder=destination/name
            if (folder/'macro_surface.json').is_file():
                surface=read_json(folder/'macro_surface.json')
                if surface['source_snapshot_sha256']!=snapshot['save_sha256'] or surface['scope']['requested_center_xz']!=list(center):
                    raise ValueError('Audit dependencies changed; choose a new output')
            else:
                surface=extract_macro_surface(source,folder,center,extent=extent,spacing=spacing,snapshot_sha256=snapshot['save_sha256'])
            with np.load(folder/'macro_surface.npz',allow_pickle=False) as data:
                biomes=biome_inventory(data['biome'],data['validity'],available_materials=library['ground_materials'])
            atomic_json(folder/'biome_coverage.json',biomes)
            row['source_surveys'][name]=dict(extent_m=extent,spacing_m=spacing,
                bounds_source_xz=surface['scope']['bounds_blocks_xz'],
                full_chunks=surface['scope']['full_chunks'],status_counts=surface['scope']['status_counts'],
                valid_samples=surface['surface']['valid_samples'],invalid_samples=surface['surface']['invalid_samples'],
                height_range_m=surface['surface']['height_range_m'],biomes=surface['biomes'],
                recipe_families=sorted(biomes['families']),unknown_biomes=biomes['unknown_biomes'],
                missing_ground_materials=sorted({f['recipe']['ground_material'] for f in biomes['families'].values() if not f['material_available']}),
                evidence=str((folder/'macro_surface.json').relative_to(workspace)))
        phase('source_unchanged')
        row['input_unchanged']=verify_source_unchanged(snapshot)
        atomic_json(destination/'summary.json',row)
        result['worlds'].append(row)
        atomic_json(output/'summary.json',result)
    result.update(completed_at_utc=utc_now(),status='source_inspection_complete',
        native_generalisation='not_run; source decoding is not end-to-end conversion',
        source_unchanged=all(w['input_unchanged']['status']=='pass' for w in result['worlds']))
    if sha256_file(Path(__file__))!=producer:raise ValueError('Audit code changed while running')
    atomic_json(output/'summary.json',result)
    phase('complete')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--workspace',type=Path,default=Path.cwd())
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();audit(a.workspace,a.output)
