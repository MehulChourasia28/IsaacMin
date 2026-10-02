"""Deterministic procedural candidate maps, explicitly not photographs or scans."""
from pathlib import Path
import json
import hashlib
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def generate(output: Path, seed=1729, size=2048):
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    low = Image.fromarray((rng.random((96, 96))*255).astype(np.uint8)).resize((size, size), Image.Resampling.BICUBIC)
    fine = rng.normal(0, 0.015, (size, size))
    noise = np.asarray(low)/255 - 0.5
    base = np.clip(0.81 + 0.12*noise + fine, 0, 1)
    lenticels = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(lenticels)
    for _ in range(1350):
        x, y = rng.uniform(0, size, 2)
        width = rng.uniform(3, 75)
        height = rng.uniform(1, 4)
        draw.ellipse((x-width/2, y-height/2, x+width/2, y+height/2), fill=int(rng.uniform(65, 245)))
    marks = np.asarray(lenticels.filter(ImageFilter.GaussianBlur(0.6)))/255
    grain = base - marks*0.6
    bark = np.stack((grain*1.00, grain*0.975, grain*0.905), axis=-1)
    Image.fromarray((np.clip(bark, 0, 1)*255).astype(np.uint8)).save(output/'birch_bark_base_color.png')
    rough = np.clip(0.66+noise*0.10+marks*0.10,0,1)
    Image.fromarray((rough*255).astype(np.uint8)).save(output/'birch_bark_roughness.png')
    height_field = noise*0.15-marks*0.4
    dy, dx = np.gradient(height_field)
    normal = np.stack((-dx*3, -dy*3, np.ones_like(dx)), axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    Image.fromarray(((normal*0.5+0.5)*255).astype(np.uint8)).save(output/'birch_bark_normal.png')
    twig=np.stack((.24+.08*noise-marks*.06,.12+.055*noise-marks*.04,.065+.04*noise-marks*.02),axis=-1)
    Image.fromarray((np.clip(twig,0,1)*255).astype(np.uint8)).save(output/'birch_twig_base_color.png')
    Image.fromarray((np.clip(.65+noise*.12,0,1)*255).astype(np.uint8)).save(output/'birch_twig_roughness.png')
    Image.fromarray(((normal*0.5+.5)*255).astype(np.uint8)).save(output/'birch_twig_normal.png')
    # Venation and chlorophyll variation follow leaf coordinates: u transverse,
    # v petiole-to-tip. Serrations and curvature are geometry in the tree worker.
    s=1024
    u,v=np.meshgrid((np.arange(s)+.5)/s,(np.arange(s)+.5)/s)
    mid=np.exp(-((u-.5)/.007)**2)
    side=np.zeros_like(u)
    for y in np.linspace(.12,.86,10):
        distance=np.abs(v-(y+np.abs(u-.5)*.65))
        side=np.maximum(side,np.exp(-(distance/.0035)**2))
    veins=np.maximum(mid,side*.6)
    mottling=rng.normal(0,.013,(s,s))+0.02*np.sin(u*37+v*21)
    green=np.stack((.20+.08*veins+mottling,.36+.12*veins+mottling,.06+.04*veins+mottling),axis=-1)
    Image.fromarray((np.clip(green,0,1)*255).astype(np.uint8)).save(output/'birch_leaf_base_color.png')
    Image.fromarray((np.clip(.52+veins*.09+mottling,0,1)*255).astype(np.uint8)).save(output/'birch_leaf_roughness.png')
    dy,dx=np.gradient(veins*.05)
    n=np.stack((-dx*2,-dy*2,np.ones_like(dx)),axis=-1);n/=np.linalg.norm(n,axis=-1,keepdims=True)
    Image.fromarray(((n*.5+.5)*255).astype(np.uint8)).save(output/'birch_leaf_normal.png')
    report={'classification':'procedural_candidate_maps_not_photographic','seed':seed,'birch_bark_repeat_m':[.8,2.0],
            'leaf_length_m_range':[.04,.09],'normal_convention':'OpenGL','qualification':'not_run',
            'limitations':['Procedural appearance requires actual Isaac closeup and backlight comparison against photographic birchwood reference',
                           'Bark normal detail is not geometric displacement; larger peeling/fractures remain a detail gap'],
            'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob('*.png')}}
    (output/'maps_manifest.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    import sys
    generate(Path(sys.argv[1]))
