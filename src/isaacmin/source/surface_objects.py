"""Read exterior canopy and rooted tree species without building underground IR."""
from pathlib import Path
from collections import Counter
import numpy as np
from scipy import ndimage
from isaacmin.source.anvil import Region, dimensions
from isaacmin.source.provenance import begin_source_read, finish_source_read
from isaacmin.terrain.surface_navigation import source_fields
from isaacmin.io import atomic_json,sha256_file,utc_now


def extract_surface_objects(snapshot, surface_path, output):
    snapshot,surface_path,output=map(Path,(snapshot,surface_path,output))
    output.mkdir(parents=True,exist_ok=False)
    surface_hash=sha256_file(surface_path)
    fields=source_fields(surface_path);h=fields['height'];xmin,zmin=map(int,fields['min_xz'])
    identity,files=begin_source_read(snapshot,'surface_objects')
    folder=dimensions(snapshot)['minecraft:overworld'];regions={};roots={};canopy=np.zeros(h.shape,bool)
    dependencies={'level.dat'}
    for cz in range(zmin//16,(zmin+h.shape[0])//16):
        for cx in range(xmin//16,(xmin+h.shape[1])//16):
            key=(cx//32,cz//32);path=folder/'region'/f'r.{key[0]}.{key[1]}.mca'
            if key not in regions:regions[key]=Region(path) if path.is_file() else None
            region=regions[key]
            if region is None or (cx,cz) not in region.locations:continue
            chunk=region.chunk(cx,cz)
            if not chunk.full:continue
            dependencies.add(path.relative_to(snapshot).as_posix())
            if chunk.metadata.get('external'):dependencies.add((path.parent/f'c.{cx}.{cz}.mcc').relative_to(snapshot).as_posix())
            x0,z0=cx*16-xmin,cz*16-zmin;localh=h[z0:z0+16,x0:x0+16]
            for sy in sorted(chunk.sections,reverse=True):
                if sy*16+16 < localh.min():continue
                section=chunk.sections[sy];palette=section.get('block_states',{}).get('palette',section.get('Palette',[]))
                names=[p['Name'] for p in palette]
                if not any(n.endswith(('_leaves','_log')) for n in names):continue
                palette,ids=chunk.section(sy);a=np.asarray(ids).reshape(16,16,16)
                for i,state in enumerate(palette):
                    name=state['Name']
                    if not name.endswith(('_leaves','_log')) or 'stripped_' in name:continue
                    yy,zz,xx=np.nonzero(a==i);y=yy+sy*16
                    above=y>=localh[zz,xx]-.1
                    yy,zz,xx,y=[v[above] for v in (yy,zz,xx,y)]
                    if name.endswith('_leaves'):
                        canopy[z0+zz,x0+xx]=True
                    else:
                        if state.get('Properties',{}).get('axis','y') != 'y':continue
                        for bx,by,bz in zip(xx,y,zz):
                            if by>localh[bz,bx]+1:continue
                            position=(int(cx*16+bx),int(cz*16+bz));prior=roots.get(position)
                            if prior is None or by<prior['source_y']:
                                roots[position]={'source_x':position[0]+.5,'source_z':position[1]+.5,
                                    'source_y':int(by),'block':name,'species':name.removeprefix('minecraft:').removesuffix('_log')}
    # Multi-column trunks (large spruce and dark oak) are one observed tree,
    # not four overlapping mature assets. Preserve all supporting source cells.
    grouped=[];pending=set(roots)
    for first in sorted(roots):
        if first not in pending:continue
        pending.remove(first);component=[first];todo=[first]
        while todo:
            x,z=todo.pop()
            for near in ((x-1,z),(x+1,z),(x,z-1),(x,z+1)):
                if near in pending and roots[near]['species']==roots[first]['species'] and abs(roots[near]['source_y']-roots[first]['source_y'])<=1:
                    pending.remove(near);component.append(near);todo.append(near)
        row=dict(roots[first]);row.update(
            source_x=float(np.mean([p[0]+.5 for p in component])),
            source_z=float(np.mean([p[1]+.5 for p in component])),
            observed_trunk_columns=[dict(roots[p]) for p in sorted(component)])
        grouped.append(row)
    # Exact source columns; canopy availability is a 9m neighbourhood inference.
    np.savez_compressed(output/'canopy.npz',leaf_columns=canopy,
                        cover_fraction=ndimage.uniform_filter(canopy.astype(float),size=9,mode='constant'))
    if sha256_file(surface_path) != surface_hash:raise RuntimeError('Surface changed')
    record={'at_utc':utc_now(),'status':'source_exterior_objects_extracted','source_identity':identity,
        'source_surface_sha256':sha256_file(surface_path),'trees':grouped,
        'species_counts':dict(Counter(p['species'] for p in grouped)),
        'structures':'excluded_by_user','underground':'not_read_for_object_placement',
        'root_detection':'unstripped vertical log within 1m above upper natural terrain; contiguous same-species columns form one tree',
        'qualification':'source_observations_not_asset_or_ecology_qualification',
        'producer_sha256':sha256_file(Path(__file__))}
    # Same immutable snapshot inventory check as the other source readers.
    finish_source_read(snapshot,'surface_objects',identity,files)
    atomic_json(output/'objects.json',record)
    return record
