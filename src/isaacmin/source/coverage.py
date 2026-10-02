"""A vector diagnostic map of real stored generation coverage, not Isaac RGB."""
from __future__ import annotations

import json
from pathlib import Path
from xml.etree.ElementTree import Element, SubElement, ElementTree


def write_coverage_svg(inventory_path: Path, output: Path, center=(-1065.38,688.07)) -> dict:
    inventory_path,output=Path(inventory_path),Path(output)
    inventory=json.loads(inventory_path.read_text())
    dimension=next(d for d in inventory["dimensions"] if d["id"]=="minecraft:overworld")
    chunks=json.loads((inventory_path.parent/dimension["coverage_file"]).read_text())
    xmin,zmin,xmax,zmax=dimension["bounds_blocks_xz"]
    colors={"minecraft:full":"#54845a","minecraft:initialize_light":"#bfab62","minecraft:carvers":"#9c8271","minecraft:biomes":"#78919c","minecraft:structure_starts":"#b8c2c8"}
    scale=min(1000/(xmax-xmin),800/(zmax-zmin))
    width,height=1200,1080
    svg=Element("svg",{"xmlns":"http://www.w3.org/2000/svg","width":str(width),"height":str(height),"viewBox":f"0 0 {width} {height}"})
    SubElement(svg,"rect",{"width":"100%","height":"100%","fill":"#f7f5f1"})
    def text(x,y,value,size=18,color="#263238"):
        element=SubElement(svg,"text",{"x":str(x),"y":str(y),"font-family":"sans-serif","font-size":str(size),"fill":color})
        element.text=value
    text(45,45,"IsaacMin: actual Minecraft source coverage",28)
    text(45,74,"Generation-state diagnostic — not an Isaac render or a realism qualification",17)
    left,top=100,115
    for chunk in chunks:
        x,z=chunk["x"]*16,chunk["z"]*16
        SubElement(svg,"rect",{"x":str(left+(x-xmin)*scale),"y":str(top+(z-zmin)*scale),"width":str(16*scale+.02),"height":str(16*scale+.02),"fill":colors.get(chunk["status"],"#bd4862")})
    scopes=[]
    for relative,color,dash in [("macro_region/macro_surface.json","#22417d","8 4"),("region_corrected/world_ir.json","#c32148","")]:
        manifest_path=inventory_path.parent/relative
        if manifest_path.exists():
            scopes.append((json.loads(manifest_path.read_text())["scope"]["bounds_blocks_xz"],color,dash))
    for bounds,color,dash in scopes:
        x0,z0,x1,z1=bounds
        values={"x":str(left+(x0-xmin)*scale),"y":str(top+(z0-zmin)*scale),"width":str((x1-x0)*scale),"height":str((z1-z0)*scale),"fill":"none","stroke":color,"stroke-width":"2"}
        if dash:values["stroke-dasharray"]=dash
        SubElement(svg,"rect",values)
    cx,cz=left+(center[0]-xmin)*scale,top+(center[1]-zmin)*scale
    SubElement(svg,"circle",{"cx":str(cx),"cy":str(cz),"r":"5","fill":"#fbf6da","stroke":"#93153a","stroke-width":"2"})
    for x in range((xmin//500)*500,xmax+1,500):
        if xmin<=x<=xmax:text(left+(x-xmin)*scale-22,top+(zmax-zmin)*scale+28,str(x),14)
    for z in range((zmin//500)*500,zmax+1,500):
        if zmin<=z<=zmax:text(35,top+(z-zmin)*scale+5,str(z),14)
    text(420,top+(zmax-zmin)*scale+56,"Minecraft x (metres); z increases downward",16)
    y=960
    for index,(status,color) in enumerate(colors.items()):
        x=45+(index%3)*385
        yy=y+(index//3)*31
        SubElement(svg,"rect",{"x":str(x),"y":str(yy-14),"width":"18","height":"18","fill":color})
        text(x+26,yy+1,status.replace("minecraft:",""),16)
    text(45,1046,"Blue dashed: 2,048m shared drainage scope. Red: 128m integration region. Dot: requested centre.",16)
    output.parent.mkdir(parents=True,exist_ok=True)
    ElementTree(svg).write(output,encoding="unicode",xml_declaration=True)
    return {"status":"source_diagnostic","file":str(output),"source_inventory":str(inventory_path),"stored_chunks":len(chunks),"not_isaac_rgb":True}
