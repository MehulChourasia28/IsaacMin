"""Read-only enforcement of measured, source-specific exhausted repair budgets.

Changing unrelated source code, textures or a build ID does not reset a known
geometry defect's budget. A completed independent correction can resolve that
specific entry; it never resets a budget or skips the next world's checks.
"""
from pathlib import Path

from ..io import read_json, sha256_file
from ..security import safe_path


def exhausted_repairs(workspace: Path, source_sha256: str, world_ir_sha256: str | None = None) -> list[dict]:
    pointer = workspace / 'state/known_failures.json'
    if not pointer.is_file():
        return []
    records = []
    for entry in read_json(pointer).get('failures', []):
        if entry['source_snapshot_sha256'] != source_sha256:
            continue
        if world_ir_sha256 is not None and entry['source_ir_sha256'] != world_ir_sha256:
            continue
        path = safe_path(workspace, entry['ledger'], must_exist=True)
        if sha256_file(path) != entry['ledger_sha256']:
            raise ValueError('Known repair ledger changed; it must be reconciled before construction')
        ledger = read_json(path)
        attempts = ledger['attempts']
        if (ledger.get('status') != 'exhausted_failed'
                or ledger.get('defect_class') != entry['defect_class']
                or len(attempts) != ledger['maximum_repair_attempts']
                or [a['attempt'] for a in attempts] != list(range(1, len(attempts)+1))):
            raise ValueError('Exhausted repair record does not have a complete bounded attempt ledger')
        from .geometry_resolution import verify_geometry_resolution
        if verify_geometry_resolution(workspace, entry):
            continue
        records.append({**entry, 'attempts': len(attempts), 'status': 'blocked_repair_budget',
                        'reason': 'Measured source-specific defect exhausted its unchanged automatic repair budget'})
    return records
