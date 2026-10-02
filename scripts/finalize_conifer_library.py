"""Author original pine PBR in the isolated USD interpreter after Blender export."""
from pathlib import Path
import argparse
import numpy as np
from pxr import Usd, UsdGeom, UsdShade, Sdf, Vt
from isaacmin.io import read_json, atomic_json, sha256_file, utc_now
from isaacmin.assembly.usd_instances import _closure
from isaacmin.assembly.provider_pbr import provider_material

parser = argparse.ArgumentParser()
parser.add_argument('--request', type=Path, required=True)
args = parser.parse_args()
request = read_json(args.request)
root, out = Path(request['workspace']), Path(request['output'])
library = read_json(out / 'geometry_library.json')
for asset in library['assets']:
    usd = Path(asset['usd'])
    if sha256_file(usd) != asset['raw_usd_sha256']:
        raise RuntimeError('Raw Blender export changed')
    card_path = Path(asset['original_card'])
    if sha256_file(card_path) != asset['original_card_sha256']:
        raise RuntimeError('Original provider manifest changed')
    card = read_json(card_path)
    stage = Usd.Stage.Open(str(usd))
    for prototype in asset['prototypes']:
        name = prototype['object_name']
        arrays = np.load(prototype['array_file'])
        top = stage.GetPrimAtPath('/Library/' + name)
        meshes = [p for p in Usd.PrimRange(top) if p.IsA(UsdGeom.Mesh)]
        if len(meshes) != 1:
            raise RuntimeError('Expected one original whole-tree mesh')
        mesh = UsdGeom.Mesh(meshes[0])
        for source_array, target in [('points', mesh.GetPointsAttr()),
                                     ('indices', mesh.GetFaceVertexIndicesAttr()),
                                     ('counts', mesh.GetFaceVertexCountsAttr())]:
            if not np.array_equal(arrays[source_array], np.asarray(target.Get())):
                raise RuntimeError('Native export changed ' + name + ':' + source_array)
        mesh.CreateSubdivisionSchemeAttr().Set('none')
        mesh.CreateDoubleSidedAttr().Set(True)
        pv = UsdGeom.PrimvarsAPI(mesh)
        st = pv.CreatePrimvar('st', Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.faceVarying)
        st.Set(Vt.Vec2fArray.FromNumpy(arrays['uvs']))
        st.BlockIndices()
        pv.CreatePrimvar('IsaacMinProviderPosition', Sdf.ValueTypeNames.Float3Array,
            UsdGeom.Tokens.vertex).Set(Vt.Vec3fArray.FromNumpy(arrays['points']))
        if 'mask' in arrays:
            interpolation = {'POINT': UsdGeom.Tokens.vertex, 'CORNER': UsdGeom.Tokens.faceVarying}.get(prototype['mask_domain'])
            if interpolation is None:
                raise RuntimeError('Unsupported provider bark blend domain')
            pv.CreatePrimvar('IsaacMinProviderBarkBlend', Sdf.ValueTypeNames.FloatArray,
                interpolation).Set(Vt.FloatArray.FromNumpy(arrays['mask']))
        binding = UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim())
        material_records = []
        ids = arrays['material_ids']
        for slot, material_name in enumerate(prototype['material_names']):
            faces = np.flatnonzero(ids == slot).astype(np.int32)
            if not len(faces):
                continue
            if '_trunk_' in material_name and 'mask' not in arrays:
                raise RuntimeError('Provider trunk blend mask lost')
            if prototype['material_recipes'][material_name].get('secondary_uv_prefix') and 'mask' not in arrays:
                raise RuntimeError('Original trunk UV blend mask lost')
            material, receipt = provider_material(root, stage, top.GetPath(),
                usd.parent, card, material_name, **prototype['material_recipes'][material_name])
            subset = binding.CreateMaterialBindSubset('Material' + str(slot), Vt.IntArray.FromNumpy(faces))
            UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(material)
            material_records.append(receipt)
        binding.SetMaterialBindSubsetsFamilyType(UsdGeom.Tokens.partition)
        prototype.update(usd_prim=str(top.GetPath()), geometry_verified_exact=True,
            source_uv_verified_exact=True, materials=material_records,
            contact='original root band measured; final terrain footprint not qualified')
        arrays.close()
    stage.GetRootLayer().Save()
    atomic_json(usd.parent / 'native_dependency_closure.json', _closure(usd))
    asset.update(usd_sha256=sha256_file(usd), native_material_authoring='candidate_authored')
    atomic_json(usd.parent / 'export.json', asset)
    print({'asset':asset['asset_id'], 'native_prototypes':len(asset['prototypes'])}, flush=True)
library.update(status='native_conifer_candidates', updated_at_utc=utc_now(),
    material_producer_sha256=sha256_file(root / 'src/isaacmin/assembly/provider_pbr.py'),
    finalizer_sha256=sha256_file(Path(__file__)), qualification='not_run')
atomic_json(out / 'library.json', library)
