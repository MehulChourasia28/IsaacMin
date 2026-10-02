"""Native Blender normalization; invoke with --disable-autoexec and a trusted request."""
import bpy
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from mathutils import Matrix, Vector

request_path = Path(sys.argv[sys.argv.index("--") + 1])
request = json.loads(request_path.read_text())
source = Path(request["source_blend"]).resolve()
destination = Path(request["output_blend"]).resolve()
report_path = Path(request["output_report"]).resolve()
destination.parent.mkdir(parents=True, exist_ok=True)
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
with bpy.data.libraries.load(str(source), link=False) as (available, loaded):
    loaded.objects = available.objects
for obj in loaded.objects:
    if obj is not None:
        bpy.context.scene.collection.objects.link(obj)
bpy.context.view_layer.update()
selected = []
excluded = []
for obj in list(bpy.context.scene.objects):
    if obj.type != "MESH":
        excluded.append({"name": obj.name, "reason": "non-mesh library helper"})
    elif "_geonodes_" in obj.name or "geometry_nodes" in obj.name:
        excluded.append({"name": obj.name, "reason": "provider scatter demonstration/duplicate helper; individual highest-detail prototypes retained"})
    elif re.search(r"_LOD[1-9][0-9]*$", obj.name):
        excluded.append({"name": obj.name, "reason": "lower detail variant; original retained in masters"})
    else:
        selected.append(obj)
for obj in list(bpy.context.scene.objects):
    if obj not in selected:
        bpy.data.objects.remove(obj, do_unlink=True)
if not selected:
    raise RuntimeError("No highest-detail prototypes found; normalize this schema explicitly")
materials = []
images = []
for image in bpy.data.images:
    if image.source != "FILE":
        continue
    path = Path(bpy.path.abspath(image.filepath))
    if not path.exists():
        relative = image.filepath.removeprefix("//")
        path = source.parent / relative
    if not path.exists():
        raise RuntimeError("Required asset image unresolved: " + image.name)
    allowed = request.get('allowed_images', {})
    if str(path.resolve()) not in allowed or hashlib.sha256(path.read_bytes()).hexdigest() != allowed[str(path.resolve())]:
        raise RuntimeError("Asset image is outside the verified declared dependency set: " + image.name)
    image.filepath = str(path.resolve())
    image.reload()
    if image.size[0] <= 0 or image.size[1] <= 0:
        raise RuntimeError("Required asset image failed decode: " + image.name)
    images.append({"name": image.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                   "size_px": list(image.size), "colorspace": image.colorspace_settings.name, "packed": True})
    image.pack()
for mat in bpy.data.materials:
    nodes = list(mat.node_tree.nodes) if mat.use_nodes else []
    material = {"name": mat.name, "node_types": sorted(set(n.bl_idname for n in nodes)),
                "image_nodes": [{"node": n.name, "image": n.image.name if n.image else None,
                                 "color_space": n.image.colorspace_settings.name if n.image else None} for n in nodes if n.type == "TEX_IMAGE"],
                "backface_culling": mat.use_backface_culling,
                "target_translation": "not_run; preserve source nodes for explicit conversion"}
    materials.append(material)
records = []
for obj in selected:
    world_matrix = obj.matrix_world.copy()
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = bpy.data.meshes.new_from_object(evaluated, preserve_all_data_layers=True, depsgraph=bpy.context.evaluated_depsgraph_get())
    basis = world_matrix.to_3x3()
    for vertex in mesh.vertices:
        vertex.co = basis @ vertex.co
    coords = [tuple(v.co) for v in mesh.vertices]
    if not coords or not all(math.isfinite(c) for p in coords for c in p):
        raise RuntimeError("Asset has empty/non-finite geometry")
    minimum = min(v[2] for v in coords)
    base = [v for v in coords if v[2] <= minimum + 0.005]
    anchor = Vector((sum(v[0] for v in base) / len(base), sum(v[1] for v in base) / len(base), minimum))
    for vertex in mesh.vertices:
        vertex.co -= anchor
    obj.modifiers.clear()
    obj.parent = None
    obj.data = mesh
    obj.matrix_world = Matrix.Identity(4)
    obj.hide_render = False
    mesh.update()
    mesh.calc_loop_triangles()
    xyz = [tuple(v.co) for v in mesh.vertices]
    lo = [min(v[i] for v in xyz) for i in range(3)]
    hi = [max(v[i] for v in xyz) for i in range(3)]
    anchors = [[v[i] - anchor[i] for i in range(3)] for v in base]
    stride = max(1, len(anchors) // 32)
    anchors = anchors[::stride][:32]
    if not mesh.uv_layers:
        raise RuntimeError("Asset lacks required UV data")
    records.append({"object_name": obj.name, "vertices": len(mesh.vertices), "triangles": len(mesh.loop_triangles),
                    "bounds_m": [lo, hi], "dimensions_m": [hi[i] - lo[i] for i in range(3)],
                    "original_world_matrix": [list(row) for row in world_matrix],
                    "normalization_translation_m": list(-anchor), "normalized_contact_z": min(v[2] for v in xyz),
                    "contact_anchors_local_m": anchors, "uv_layers": [u.name for u in mesh.uv_layers],
                    "normalization_geometry_policy": "highest provider detail, no simplification; apply original rotation/scale and move base to local Z=0",
                    "contact_limitations": "lowest 5mm vertex band is a candidate support anchor; irregular-surface and target-renderer grounding must be validated"})
bpy.context.scene.unit_settings.system = "METRIC"
bpy.context.scene.unit_settings.scale_length = 1.0
# Remove potentially executable text/driver payloads from the normalized product.
for text in list(bpy.data.texts):
    bpy.data.texts.remove(text)
for obj in bpy.data.objects:
    obj.animation_data_clear()
for collection in (bpy.data.materials,bpy.data.node_groups,bpy.data.worlds,bpy.data.meshes,bpy.data.scenes):
    for datablock in collection:
        datablock.animation_data_clear()
        if getattr(datablock,'node_tree',None):
            datablock.node_tree.animation_data_clear()
bpy.ops.wm.save_as_mainfile(filepath=str(destination), compress=True)
report = {"schema_version": 1, "asset_id": request["asset_id"], "source_sha256": request["source_sha256"],
          "blender_version": bpy.app.version_string, "output_blend": str(destination), "objects": records,
          "images": images, "materials": materials, "excluded_library_objects": excluded,
          "status": "external_tool_verified", "isaac_qualification": "not_run", "root_contact_test": "not_run",
          "units": "metres", "scale_basis": "Blender native units retained; provider millimetre dimensions converted separately"}
report_path.write_text(json.dumps(report, indent=2))
