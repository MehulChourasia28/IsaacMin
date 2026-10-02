"""Real Chunky source previews, derived from a build's immutable source and bounds."""
from pathlib import Path
import math
import shutil

import numpy as np
from PIL import Image

from isaacmin.io import atomic_json, read_json, sha256_file, hash_object, utc_now
from isaacmin.process import run_worker
from isaacmin.source.anvil import dimensions, Region


def render_source(workspace, build, output):
    root,build,output=map(lambda p:Path(p).resolve(),(workspace,build,output))
    if output.exists():raise ValueError('Choose a new source capture directory to preserve evidence')
    record=read_json(build/'outdoor_build.json');terrain=read_json(build/'terrain/terrain.json')
    source_hash=record['identity']['source'];snapshot=root/'work/snapshots'/source_hash
    saved=read_json(snapshot/'snapshot.json')
    if saved['save_sha256']!=source_hash or saved['status']!='validated':raise ValueError('Source snapshot is not validated')
    world=Path(saved['snapshot_path']);dimension=dimensions(world)['minecraft:overworld']
    bounds=terrain['bounds_source_xz'];xmin,zmin,xmax,zmax=bounds
    chunks=[[x,z] for x in range(math.floor(xmin/16),math.ceil(xmax/16))
                   for z in range(math.floor(zmin/16),math.ceil(zmax/16))]
    regions={};sections=[]
    for x,z in chunks:
        key=(x//32,z//32)
        if key not in regions:regions[key]=Region(dimension/'region'/f'r.{key[0]}.{key[1]}.mca')
        chunk=regions[key].chunk(x,z)
        if not chunk.full:raise ValueError('Source preview requires complete generated chunks')
        sections.extend(chunk.sections)
    if not sections:raise ValueError('Source preview has no stored block sections')
    home=root/'.tools/chunky/home';textures=root/'.tools/reference/minecraft-26.1-client.jar'
    jars=sorted((home/'lib').glob('*.jar'))
    if not jars or not textures.is_file():raise ValueError('Pinned local Chunky and Minecraft reference textures are missing')
    height=np.load(build/'terrain/height.npy',mmap_mode='r')
    span=max(xmax-xmin,zmax-zmin);distance=span*1.25
    cx,cz=(xmin+xmax)/2,(zmin+zmax)/2
    cy=float(np.max(height))+span*260/256;ground=float(np.median(height))
    output.mkdir(parents=True);bridge=output/'source';bridge.mkdir()
    (bridge/'level.dat').symlink_to(world/'level.dat');(bridge/'region').symlink_to(dimension/'region',target_is_directory=True)
    name='source_'+hash_object(dict(source=source_hash,bounds=bounds,output=str(output)))[:16]
    request=output/(name+'.json')
    parameters=dict(name=name,sdfVersion=9,width=1280,height=720,spp=0,sppTarget=128,
        rayDepth=8,pathTrace=True,saveSnapshots=True,dumpFrequency=128,biomeColorsEnabled=True,
        renderActors=False,waterWorldEnabled=False,fogDensity=0,skyFogDensity=0,exposure=1,postprocess='TONEMAP1',
        yClipMin=min(sections)*16,yClipMax=(max(sections)+1)*16,chunkList=chunks,
        world=dict(path=str(bridge),dimension=0),camera=dict(projectionMode='PINHOLE',fov=49,dof='Infinity',
        position=dict(x=cx,y=cy,z=cz-distance),orientation=dict(yaw=math.pi/2,
            pitch=-math.pi/2+math.atan2(cy-ground,distance),roll=0)))
    atomic_json(request,parameters)
    identity=dict(source_save_sha256=source_hash,level_dat_sha256=sha256_file(world/'level.dat'),
        bounds_source_xz=bounds,renderer='Chunky 2.4.6 CPU',source_modified=False,
        textures_sha256=sha256_file(textures),renderer_jars={p.name:sha256_file(p) for p in jars},
        request_sha256=sha256_file(request))
    atomic_json(output/'input.json',identity)
    java=shutil.which('java')
    if not java:raise ValueError('A compatible local Java runtime is required')
    command=[java,'-Xmx6g','-Djava.awt.headless=true','-Dchunky.home='+str(home),'-cp',str(home/'lib/*'),
        'se.llbit.chunky.main.Chunky','-texture',str(textures),'-scene-dir',str(output),
        '-threads','6','-render',name,'-reload-chunks','-f']
    process=run_worker(command,cwd=root,log_path=output/'worker.log',timeout=900,estimated_memory_bytes=8*2**30)
    image=output/'snapshots'/(name+'-128.png')
    if not image.is_file():
        alternative=home/'scenes'/name/'snapshots'/image.name
        if alternative.is_file() and alternative.stat().st_mtime>request.stat().st_mtime:
            image.parent.mkdir(exist_ok=True);shutil.copyfile(alternative,image)
    log=(output/'worker.log').read_text(errors='replace').replace('\r','\n')
    if process['exit_code'] or not image.is_file() or f'({len(chunks)} of {len(chunks)})' not in log:
        raise RuntimeError('Native source preview failed; preserve its log and process receipt')
    with Image.open(image) as im:
        size=list(im.size);im.verify()
    report=dict(**identity,status='actual_source_capture_complete',at_utc=utc_now(),
        exit_code=process['exit_code'],elapsed_seconds=process['elapsed_seconds'],chunks_loaded=len(chunks),
        resolution_px=size,samples_per_pixel=128,log_sha256=sha256_file(output/'worker.log'),
        renderer_claim='Chunky source-save rendering, not a Minecraft game-client screenshot',
        block_texture_qualification='not_exhaustive; inspect unsupported-block warnings in the native log',
        images=[dict(file=image.relative_to(output).as_posix(),sha256=sha256_file(image),caption='Source-save aerial')])
    atomic_json(output/'render.json',report)
    return report
