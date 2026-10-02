"""Lossless native USD sharing with complete composed-property comparisons.

This changes storage/composition only. It never qualifies rendered appearance.
Asset identities nominate candidates; every authored child property, array,
subset and material relationship must match before a prototype is shared.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from isaacmin.io import atomic_json, sha256_file, utc_now


class PropertyFingerprint:
    """Hash array bytes without expanding large native arrays into Python lists."""

    def __init__(self):
        self.arrays = {}
        self.array_bytes_hashed = 0

    def value(self, value, owner):
        from pxr import Sdf
        if value is None or isinstance(value, (bool, int, float, str)):
            return value
        if isinstance(value, Sdf.Path):
            return str(value.ReplacePrefix(owner, Sdf.Path('/INSTANCE'))) if value.HasPrefix(owner) else str(value)
        if isinstance(value, Sdf.AssetPath):
            return {'asset': value.path}
        if isinstance(value, dict):
            return {str(k): self.value(v, owner) for k, v in sorted(value.items())}
        if isinstance(value, (list, tuple)):
            return [self.value(v, owner) for v in value]
        if type(value).__module__=='pxr.Gf' and type(value).__name__.startswith('Quat'):
            return {'native_type':type(value).__name__,'real':float(value.GetReal()),
                    'imaginary':[float(v) for v in value.GetImaginary()]}
        if type(value).__module__.startswith(('pxr.Vt', 'pxr.Gf')):
            array = np.asarray(value)
            if array.dtype.kind in 'biuf' and array.flags.c_contiguous:
                key = (array.__array_interface__['data'][0], array.dtype.str, array.shape, array.strides)
                cached = self.arrays.get(key)
                if cached is None:
                    digest = hashlib.sha256(memoryview(array).cast('B') if array.nbytes else b'').hexdigest()
                    # Keep the original immutable Vt storage alive: addresses may
                    # otherwise be reused for different arrays later in this run.
                    self.arrays[key] = (value, array, digest)
                    self.array_bytes_hashed += array.nbytes
                else:
                    digest = cached[2]
                return {'array': digest, 'dtype': array.dtype.str, 'shape': list(array.shape)}
            return [self.value(v, owner) for v in value]
        return {'native_type': type(value).__name__, 'value': str(value)}

    def prim(self, prim, owner, *, instance_root=False):
        metadata = prim.GetAllAuthoredMetadata()
        if instance_root:
            metadata = {k: v for k, v in metadata.items() if k not in ('references', 'instanceable')}
        record = {'path': self.value(prim.GetPath(), owner),
                  'metadata': self.value(metadata, owner), 'attributes': {}, 'relationships': {}}
        for attr in sorted(prim.GetAuthoredAttributes(), key=lambda a: a.GetName()):
            meta = {k: v for k, v in attr.GetAllAuthoredMetadata().items()
                    if k not in ('default', 'timeSamples', 'connectionPaths')}
            record['attributes'][attr.GetName()] = {
                'type': str(attr.GetTypeName()), 'metadata': self.value(meta, owner),
                'default': self.value(attr.Get(), owner),
                'time_samples': [[t, self.value(attr.Get(t), owner)] for t in attr.GetTimeSamples()],
                'connections': self.value(attr.GetConnections(), owner)}
        for rel in sorted(prim.GetAuthoredRelationships(), key=lambda r: r.GetName()):
            meta = {k: v for k, v in rel.GetAllAuthoredMetadata().items() if k != 'targetPaths'}
            record['relationships'][rel.GetName()] = {
                'metadata': self.value(meta, owner), 'targets': self.value(rel.GetTargets(), owner)}
        return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

    def children(self, prim):
        from pxr import Usd
        records = []
        for child in prim.GetFilteredChildren(Usd.TraverseInstanceProxies()):
            records.extend(self.prim(p, prim.GetPath())
                           for p in Usd.PrimRange(child, Usd.TraverseInstanceProxies()))
        return hashlib.sha256(''.join(sorted(records)).encode()).hexdigest()


def _closure(scene):
    from pxr import Sdf, Usd, UsdUtils
    layers, assets, unresolved = UsdUtils.ComputeAllDependencies(Sdf.AssetPath(str(scene)))
    if unresolved:
        raise ValueError('Unresolved native USD dependencies: ' + repr(list(unresolved)))
    paths = {Path(layer.realPath).resolve() for layer in layers}
    paths.update(Path(p).resolve() for p in assets)
    if any(not p.is_relative_to(scene.parent) or not p.is_file() for p in paths):
        raise ValueError('USD dependency escapes the portable scene')
    if any(Path(p).is_absolute() for layer in layers for p in layer.GetExternalReferences()):
        raise ValueError('USD contains an absolute external asset reference')
    return {'status': 'pass', 'root_asset': scene.name, 'root_sha256': sha256_file(scene),
            'files': [{'path': str(p.relative_to(scene.parent)), 'sha256': sha256_file(p),
                       'bytes': p.stat().st_size} for p in sorted(paths)],
            'unresolved_paths': [], 'absolute_asset_paths': [],
            'method': 'UsdUtils.ComputeAllDependencies', 'toolchain': {'usd': list(Usd.GetVersion())}}


def share_scene(source, destination, evidence):
    """Publish a separate portable candidate, preserving every existing scene byte."""
    from pxr import Sdf, Usd
    from isaacmin.assets.usd_provenance import verify_scene_closure
    source, destination, evidence = map(lambda p: Path(p).resolve(), (source, destination, evidence))
    if destination.exists() or destination.is_relative_to(source.parent):
        raise ValueError('Use a fresh destination outside the original scene')
    if evidence.exists():
        raise ValueError('Preserve previous sharing evidence; use a fresh report path')
    final_destination = destination
    destination = destination.with_name(destination.name + '.staging')
    if destination.exists():
        raise ValueError('Preserve incomplete staging; use a fresh destination')
    started = time.monotonic()
    closure_path = source.parent / 'native_dependency_closure.json'
    closure_hash = sha256_file(closure_path)
    verified = verify_scene_closure(source, closure_hash)
    source_stage = Usd.Stage.Open(str(source), load=Usd.Stage.LoadNone)
    source_content = Sdf.Layer.FindOrOpen(str(source.parent / 'content.usdc'))
    if source_stage is None or source_content is None:
        raise ValueError('Expected native Blender root plus content layer')
    if source_stage.GetPrototypes():
        raise ValueError('This conversion accepts an uninstanced export, never double conversion')
    fingerprints = PropertyFingerprint()
    assets, groups, original = {}, defaultdict(list), {}
    for prim in source_stage.Traverse():
        identity = prim.GetAttribute('isaacmin:isaacmin_source_blend_sha256')
        is_asset = bool(identity and identity.Get())
        path = str(prim.GetPath())
        original[path] = fingerprints.prim(prim, prim.GetPath(), instance_root=is_asset)
        if not is_asset:
            continue
        if prim.GetTypeName() != 'Xform' or not prim.GetChildren():
            raise ValueError('Expected an explicit asset Xform and original mesh children')
        if prim.HasAuthoredReferences() or prim.HasAuthoredPayloads():
            raise ValueError('Asset already uses composition arcs')
        for child in prim.GetChildren():
            for descendant in Usd.PrimRange(child):
                if any(spec.layer != source_content for spec in descendant.GetPrimStack()):
                    raise ValueError('Asset child overrides need an explicit layer-merge adapter')
        payload = fingerprints.children(prim)
        key = (str(identity.Get()), str(prim.GetAttribute('isaacmin:isaacmin_source_object').Get()), payload)
        groups[key].append(path)
        assets[path] = {'payload_sha256': payload, 'children': [c.GetName() for c in prim.GetChildren()]}
    if not assets:
        raise ValueError('No explicit source-bound assets to instance')
    evidence.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(evidence.with_name(evidence.stem + '.progress.json'), {
        'status': 'compared_source_arrays', 'assets': len(assets), 'distinct_payloads': len(groups),
        'unique_array_bytes_hashed': fingerprints.array_bytes_hashed})
    destination.mkdir(parents=True)
    for item in verified['files']:
        path = Path(item['path']); target = destination / path.relative_to(source.parent)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    result_scene = destination / source.name
    content = Sdf.Layer.FindOrOpen(str(destination / 'content.usdc'))
    stage = Usd.Stage.Open(str(result_scene))
    stage.SetEditTarget(content)
    namespace = '/IsaacMinAssetPrototypes'
    if stage.GetPrimAtPath(namespace):
        raise ValueError('Prototype namespace already exists')
    stage.CreateClassPrim(namespace)
    prototypes = []
    for index, (key, paths) in enumerate(sorted(groups.items())):
        proto = stage.DefinePrim(namespace + '/P' + str(index), 'Xform')
        original_parent = source_stage.GetPrimAtPath(paths[0])
        for child in original_parent.GetChildren():
            target = proto.GetPath().AppendChild(child.GetName())
            if not Sdf.CopySpec(source_content, child.GetPath(), content, target):
                raise ValueError('Could not copy exact prototype description')
        with Sdf.ChangeBlock():
            for path in paths:
                spec = content.GetPrimAtPath(path)
                for name in assets[path]['children']:
                    del spec.nameChildren[name]
                spec.referenceList.prependedItems = [Sdf.Reference(primPath=proto.GetPath())]
                spec.instanceable = True
        prototypes.append({'path': str(proto.GetPath()), 'source_blend_sha256': key[0],
                           'source_object': key[1], 'payload_sha256': key[2], 'instances': len(paths)})
    content.Save()
    # Reopen from saved bytes; compare EVERY original composed prim/property.
    content.Reload()
    checked = Usd.Stage.Open(str(result_scene))
    after = PropertyFingerprint()
    remaining = set(original)
    for prim in checked.Traverse(Usd.TraverseInstanceProxies()):
        path = str(prim.GetPath())
        if path not in original:
            raise ValueError('Unexpected visible prim after sharing: ' + path)
        measured = after.prim(prim, prim.GetPath(), instance_root=path in assets)
        if measured != original[path]:
            raise ValueError('Composed property mismatch after sharing: ' + path)
        remaining.remove(path)
    if remaining:
        raise ValueError('Visible prims lost after sharing: ' + repr(sorted(remaining)[:5]))
    count = sum(p.IsInstance() for p in checked.Traverse())
    if count != len(assets) or len(checked.GetPrototypes()) != len(groups):
        raise ValueError('Native USD did not create the expected actual instances')
    output_closure = _closure(result_scene)
    atomic_json(destination / 'native_dependency_closure.json', output_closure)
    verify_scene_closure(source, closure_hash)
    destination.rename(final_destination)
    destination = final_destination
    result_scene = destination / source.name
    result = {'schema_version': 1, 'status': 'lossless_native_instancing_verified',
              'created_at_utc': utc_now(), 'source_scene': str(source),
              'source_scene_sha256': sha256_file(source), 'source_closure_sha256': closure_hash,
              'scene': str(result_scene), 'scene_sha256': sha256_file(result_scene),
              'closure_sha256': sha256_file(destination / 'native_dependency_closure.json'),
              'producer_sha256': sha256_file(Path(__file__)), 'native_instances': count,
              'native_prototypes': len(checked.GetPrototypes()), 'prototypes': prototypes,
              'original_composed_prims_compared': len(original),
              'comparison': 'Every original composed prim: metadata, property values, complete array bytes, connections and bindings; only instanceable/reference metadata on asset roots changes',
              'geometry_material_transform_reduction': False, 'source_unchanged': True,
              'content_bytes_before': (source.parent / 'content.usdc').stat().st_size,
              'content_bytes_after': (destination / 'content.usdc').stat().st_size,
              'elapsed_seconds': time.monotonic() - started,
              'appearance_motion_sensor_qualification': 'not_run'}
    atomic_json(evidence, result)
    return result
