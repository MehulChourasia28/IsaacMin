"""Find an immutable native ground cache without trusting a previous build pass.

Selection saves construction time after unrelated code changes. The ordinary
assembler still constructs the current complete request and calls verify_bare
again before Blender can open the cache. No old measurement is promoted.
"""
from pathlib import Path

from isaacmin.io import atomic_json, read_json, sha256_file, utc_now
from isaacmin.assembly.native_cache import record_bare, verify_bare


def complete_bare_manifest(workspace, output_directory):
    """Finish the export handoff, including a native export from an older cache.

    This records construction evidence only. Existing manifests and failed
    exports are never repaired or promoted by this operation.
    """
    output = Path(output_directory).resolve()
    request = read_json(output / 'assembly_request.json')
    if request.get('assets'):
        raise ValueError('Only a bare terrain/water export can complete this handoff')
    manifest = output / 'bare_scene_cache.json'
    if manifest.is_symlink():
        raise ValueError('Bare cache manifest must be owned by this export')
    if not manifest.exists():
        record_bare(request, workspace, output)
    return verify_bare(request, workspace, output)


def find_bare_scene(workspace, ir_directory, output_directory, *, materials, origin,
                    support_mask_path, exterior_delta, geometry_detail,
                    terrain_material_recipe, evidence):
    root, ir, output = map(lambda p: Path(p).resolve(),
                           (workspace, ir_directory, output_directory))
    detail = dict(geometry_detail or {})
    if int(detail.get('subdivision_levels', 0)):
        detail.setdefault('subdivision_method', 'bilinear_double_no_limit_v1')
    report = {'schema_version': 1, 'created_at_utc': utc_now(), 'status': 'no_matching_cache',
              'qualification': 'not_inferred', 'candidates': [], 'selected': None,
              'producer_sha256': sha256_file(Path(__file__)),
              'final_guard': 'Ordinary assembler verifies its independently constructed current request before cache open'}
    candidates = sorted((root / 'worlds').glob('*/data/bare_scene/bare_scene_cache.json'),
                        key=lambda p: p.stat().st_mtime_ns, reverse=True)
    for manifest in candidates:
        directory = manifest.parent
        if directory.resolve() == output:
            continue
        entry = {'directory': str(directory.relative_to(root))}
        report['candidates'].append(entry)
        try:
            if (directory.is_symlink() or not directory.resolve().is_relative_to(root / 'worlds')
                    or manifest.is_symlink()):
                raise ValueError('Cache directory must be an owned world output')
            request_path = directory / 'assembly_request.json'
            request = read_json(request_path)
            # The exact current terrain mesh and water bytes are checked again
            # by assemble_region. Never rewrite this recorded native request.
            current = dict(request, output=str(output), assets=[], materials=materials,
                minecraft_origin=list(origin), source_surface=str(ir / 'terrain_surface.npz'),
                support_manifest=str(Path(support_mask_path).resolve().parent / 'support_manifest.json')
                                 if support_mask_path else None,
                exterior_delta=exterior_delta, geometry_detail=detail,
                texture_repeat_m=float(materials[0].get('repeat_m', 2)),
                terrain_material_recipe=terrain_material_recipe)
            verified = verify_bare(current, root, directory)
            entry.update(status='verified_cache_candidate',
                manifest_sha256=verified['manifest_sha256'],
                original_request_sha256=sha256_file(request_path),
                ground_sha256=verified['manifest']['ground_sha256'])
            report.update(status='verified_candidate_selected', selected=entry)
            atomic_json(Path(evidence), report)
            return directory
        except (ValueError, KeyError, TypeError, OSError) as exc:
            entry.update(status='rejected', reason=str(exc), error_type=type(exc).__name__)
    atomic_json(Path(evidence), report)
    return None


def assemble_bare(workspace, ir_directory, output_directory, *, evidence, **parameters):
    """Use a verified candidate, with fresh construction on an input mismatch."""
    from isaacmin.assembly.pipeline import assemble_region
    keys=('materials','origin','support_mask_path','exterior_delta','geometry_detail',
          'terrain_material_recipe')
    candidate=find_bare_scene(workspace,ir_directory,output_directory,
        evidence=evidence,**{key:parameters.get(key) for key in keys})
    try:
        result = assemble_region(ir_directory,output_directory,
                                 bare_scene_directory=candidate,**parameters)
    except ValueError as exc:
        if candidate is None or str(exc)!='Bare geometry/material producer or inputs changed':
            raise
        # The complete current mesh/water request rejected the candidate before
        # native launch. Preserve that rejection and build the current inputs.
        output=Path(output_directory)
        if any((output/name).exists() for name in ('scene.blend','world.usda','export_result.json')):
            raise RuntimeError('Unexpected native output before cache verification; preserve for inspection') from exc
        report=read_json(Path(evidence))
        report.update(status='current_request_rejected_cache_fresh_construction',
                      actual_request_rejection=str(exc))
        atomic_json(Path(evidence),report)
        result = assemble_region(ir_directory,output_directory,**parameters)
    if result.get('status') == 'success':
        complete_bare_manifest(workspace, output_directory)
    return result
