import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('native_material_log',
    Path(__file__).resolve().parents[1] / 'isaac_scripts/material_log.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_real_native_compiler_errors_cannot_be_accepted_as_rgb_success(tmp_path):
    log = tmp_path / 'kit.log'
    log.write_text('[Info] [rtx.neuraylib.plugin] [EntityResolver] add MDL search path\n')
    guard = module.MaterialLogGuard(log, tmp_path / 'diagnostics.json')
    assert guard.check()['status'] == 'no_observed_native_material_errors'
    # These are the actual pinned runtime diagnostic forms encountered by the
    # failed leaf probe, not evidence of a native execution in this unit test.
    with log.open('a') as stream:
        stream.write("[Error] [rtx.neuraylib.plugin] [MDLC:COMPILER] C147 no overloaded version of function 'float2(float3)' matches calling parameters\n")
    with pytest.raises(RuntimeError, match='material compilation/import failed'):
        guard.check()

    log.write_text('')
    with pytest.raises(RuntimeError, match='truncated'):
        guard.check()


def test_incremental_log_preserves_split_diagnostics_and_rejects_truncation(tmp_path):
    log = tmp_path / 'kit.log'
    log.write_text('[Error] [omni.rtx.mater')
    guard = module.MaterialLogGuard(log, tmp_path / 'diagnostics.json')
    guard.check()
    with log.open('a') as stream:
        stream.write('ials] Unable to find SdrShaderNode for prim\n')
    with pytest.raises(RuntimeError, match='material compilation/import failed'):
        guard.check()


def test_native_rtx_missing_mdl_texture_warning_is_an_import_failure(tmp_path):
    log=tmp_path/'kit.log'
    log.write_text("[Warning] [omni.rtx] Texture file referenced in the material body wasn't resolved properly: file:/scene/LeafSet012/Opacity.png\n")
    guard=module.MaterialLogGuard(log,tmp_path/'diagnostics.json')
    with pytest.raises(RuntimeError,match='material compilation/import failed'):
        guard.check()
