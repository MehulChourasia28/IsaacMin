"""Run with --factory-startup --disable-autoexec; read downloaded .blend as data."""
import bpy
import json
import sys
from pathlib import Path

source, output = sys.argv[sys.argv.index("--") + 1:]
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
with bpy.data.libraries.load(source, link=False) as (available, loaded):
    loaded.objects = available.objects
for obj in loaded.objects:
    if obj is not None:
        bpy.context.scene.collection.objects.link(obj)
objects = []
for obj in loaded.objects:
    if obj is None:
        continue
    objects.append({"name": obj.name, "type": obj.type, "dimensions": list(obj.dimensions),
                    "location": list(obj.location), "scale": list(obj.scale),
                    "vertices": len(obj.data.vertices) if obj.type == "MESH" else 0,
                    "polygons": len(obj.data.polygons) if obj.type == "MESH" else 0,
                    "asset_tagged": bool(obj.asset_data), "hide_render": obj.hide_render,
                    "parent": obj.parent.name if obj.parent else None,
                    "modifiers": [{"type": x.type, "name": x.name} for x in obj.modifiers],
                    "materials": [x.name for x in obj.data.materials if x] if obj.type == "MESH" else []})
Path(output).write_text(json.dumps({"blender_version": bpy.app.version_string, "objects": objects}, indent=2))
