"""Decode original HDRI radiance without altering it; native Blender evidence only."""
import bpy
import hashlib
import json
import math
import numpy as np
from pathlib import Path
import sys

request=json.loads(Path(sys.argv[sys.argv.index('--')+1]).read_text())
results=[]
for entry in request['images']:
    path=Path(entry['path']);image=bpy.data.images.load(str(path),check_existing=False)
    width,height=image.size;channels=image.channels
    pixels=np.empty(width*height*channels,dtype=np.float32)
    image.pixels.foreach_get(pixels)
    rgb=pixels.reshape(height,width,channels)[::-1,:,:3]
    if width!=2*height or not np.isfinite(rgb).all() or float(rgb.max())<=0:raise RuntimeError('Invalid latlong HDRI decode')
    luminance=rgb@np.array([.2126,.7152,.0722],np.float32)
    lat=math.pi/2-(np.arange(height)+.5)*math.pi/height
    solid_angle=(2*math.pi/width)*(math.pi/height)*np.cos(lat)
    projected=solid_angle*np.maximum(np.sin(lat),0)
    irradiance=float(np.sum(luminance.sum(axis=1,dtype=np.float64)*projected))
    row,column=np.unravel_index(int(luminance.argmax()),luminance.shape)
    results.append({'asset_id':entry['asset_id'],'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                    'native_decoder':'Blender '+bpy.app.version_string,'width':width,'height':height,'channels':channels,
                    'finite_samples':True,'rgb_max':float(rgb.max()),'luminance_max':float(luminance.max()),
                    'luminance_median':float(np.median(luminance)),'upper_hemisphere_luminance_integral':irradiance,
                    'integral_units':'relative provider luminance times projected steradians; not independently measured lux',
                    'brightest_texture_longitude_degrees':float((column+.5)/width*360-180),
                    'brightest_texture_latitude_degrees':float(lat[row]*180/math.pi),
                    'orientation_limit':'Texture longitude/latitude, not yet verified target-world sun direction',
                    'isaac_qualification':'not_run'})
    bpy.data.images.remove(image)
Path(request['output']).write_text(json.dumps({'images':results},indent=2))
