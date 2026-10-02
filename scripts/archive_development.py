#!/usr/bin/env python3
"""Reversible local housekeeping. No deletion, copying, or source-save changes.

Plan first, inspect reports/cleanup_plan.json, then apply that exact plan while
the studio is stopped. Restore refuses to overwrite files created subsequently.
This intentionally preserves whole referenced run/library directories: their
relative USD, MDL and binary dependencies must stay together.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import stat

from isaacmin.io import atomic_json, read_json, utc_now

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/cleanup_plan.json'
TEXT = {'.json', '.jsonl', '.py', '.md', '.html', '.yaml', '.yml', '.toml', '.js', '.cjs', '.sh', '.usda', '.mdl'}
PATTERN = re.compile(r'(?:artifacts/development|artifacts/demo/jobs|packages)/[^\s"\'<>`\)\],;]+')


def unit(path):
    """Smallest safe run directory within our explicit cleanup scope."""
    parts = path.relative_to(ROOT).parts
    if parts[:2] == ('artifacts', 'development') and len(parts) >= 3:
        return ROOT.joinpath(*parts[:4])
    if parts[:3] == ('artifacts', 'demo', 'jobs') and len(parts) >= 4:
        return ROOT.joinpath(*parts[:4])
    if parts[0] == 'packages' and len(parts) >= 2:
        return ROOT.joinpath(*parts[:2])


def files(path):
    if path.is_file():
        yield path
    elif path.is_dir() and not path.is_symlink():
        for base, dirs, names in os.walk(path, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in {'.git', '.venv', '.tools', '__pycache__'})
            for name in sorted(names):
                p = Path(base)/name
                if not p.is_symlink() and name != '.env' and not name.startswith('.env.'):
                    yield p


def tree_receipt(path):
    """Rename identity: every entry's inode, size, timestamp and symlink target.

    Renaming on one filesystem preserves file bytes/hardlinks. We deliberately
    avoid copying or rehashing hundreds of GB of identical payload bytes.
    """
    digest = hashlib.sha256(); count = size = 0
    def visit(p):
        nonlocal count, size
        s = p.lstat(); name = str(p.relative_to(path))
        if stat.S_ISDIR(s.st_mode):
            digest.update(json.dumps([name, s.st_ino, s.st_mode]).encode())
            for child in sorted(p.iterdir()):visit(child)
        else:
            target = os.readlink(p) if p.is_symlink() else None
            digest.update(json.dumps([name,s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,target]).encode())
            count += 1; size += s.st_size
    visit(path)
    return dict(entries=count, logical_bytes=size, metadata_sha256=digest.hexdigest(), device=path.lstat().st_dev)


def assert_idle():
    database = ROOT/'state/demo.sqlite'
    if database.exists():
        with sqlite3.connect(f'file:{database}?mode=ro', uri=True) as db:
            if db.execute("SELECT COUNT(*) FROM jobs WHERE status IN ('queued','running')").fetchone()[0]:
                raise RuntimeError('Finish queued/running demo jobs before archiving')
    # Require both queue owners and the UI to be stopped during filesystem moves.
    import shlex, subprocess
    for line in subprocess.check_output(['ps','-eo','args'],text=True).splitlines():
        try:argv=shlex.split(line)
        except ValueError:continue
        if len(argv)>3 and argv[1:3]==['-m','isaacmin'] and argv[3] in ('demo','demo-worker'):
            raise RuntimeError('Stop the local studio and its idle workers before archiving')


def plan():
    protected = {}; queue = []; seen = set()
    def pin(path, reason):
        path = Path(path)
        if not path.is_absolute():path=ROOT/path
        path = Path(os.path.abspath(path))
        if not path.is_relative_to(ROOT) or not path.exists():return
        scope = unit(path)
        if scope is not None and scope not in protected:
            protected[scope] = reason; queue.append(scope)

    # Active and historical state remains intact, including failure budgets and
    # resume journals. Only top-level state + material manifests seed the scan;
    # old conversation transcripts/history are not live configuration.
    seeds = list((ROOT/'state').glob('*.json')) + list((ROOT/'state/material_candidates').glob('*.json'))
    seeds.extend(p for p in files(ROOT/'assets') if p.suffix=='.json')
    for folder in ('src','scripts','blender_scripts','isaac_scripts','recipes','tests','docs','configs','reports'):
        seeds.extend(files(ROOT/folder))
    presets=read_json(ROOT/'state/demo_presets.json')['presets']
    for preset in presets:
        pin(preset['build'], 'Registered preset build')
        pin(preset['capture'], 'Registered preset capture')
        delivery=ROOT/'artifacts/demo/downloads'/(preset['id']+'.json')
        if delivery.is_file():seeds.append(delivery)
    from isaacmin.demo.jobs import Jobs
    for job in Jobs(ROOT).list():
        pin(ROOT/'artifacts/demo/jobs'/job['id'], 'Visible saved world/job')
        pin(ROOT/'packages'/('demo_conversion_'+job['id']), 'Visible saved world download')
    # Small ledgers and fixtures must retain their historical paths.
    for path in ('artifacts/development/float32_interop_repair/repair_ledger.json',
                 'artifacts/development/portable_first_map/repair_ledger.json',
                 'artifacts/development/terrain_terrace_diagnosis_20261001/naturalization_ledger.json',
                 'artifacts/development/native_precision_continuation/attempt_4/patch_ring_1.npz'):
        pin(path, 'Preserved repair budget or regression input')
    def scan(path):
        if path in seen or path.suffix not in TEXT or path.stat().st_size > 24*1024*1024:return
        seen.add(path)
        # Never inspect secret files or session prompts/logs.
        if path.name.startswith('.env'):return
        content=path.read_text(errors='replace')
        for match in PATTERN.finditer(content):
            pin(match.group(0).rstrip('.:'), str(path.relative_to(ROOT)))
    for path in seeds:
        # Do not let previous cleanup plans or this planner pin every candidate.
        if path == Path(__file__).resolve() or path.name.startswith(('cleanup_', 'CLEANUP')):continue
        scan(path)
    while queue:
        directory=queue.pop()
        for path in files(directory):scan(path)

    candidates=[]
    def split(path):
        if any(path==p or p in path.parents for p in protected):return
        if any(path in p.parents for p in protected):
            if path.is_dir() and not path.is_symlink():
                for child in sorted(path.iterdir()):split(child)
        else:candidates.append(path)
    split(ROOT/'artifacts/development')
    split(ROOT/'artifacts/demo/jobs')
    split(ROOT/'packages')
    # These are extracted duplicate packages used for past replay checks. Keep
    # replay receipts and native captures beside them as qualification evidence.
    candidates.extend(sorted((ROOT/'artifacts/demo/portable').glob('*/unpacked')))
    visible={j['id'] for j in Jobs(ROOT).list()}
    for receipt in (ROOT/'artifacts/demo/downloads').glob('job_*.json'):
        if receipt.stem in visible:continue
        data=read_json(receipt); archive=Path(data['archive'])
        if not archive.is_relative_to(ROOT/'artifacts/demo/downloads'):continue
        if archive.is_file():candidates.append(archive)
        candidates.append(receipt)
    # Keep all test reports: they are small, linked evidence, not payload debris.
    candidates = sorted(set(p for p in candidates if p.exists()))
    destination='other/cleanup_'+utc_now().replace(':','').replace('-','').split('.')[0].replace('+','_')
    moves=[dict(source=str(p.relative_to(ROOT)), destination=str(Path(destination)/p.relative_to(ROOT)),
                receipt=tree_receipt(p)) for p in candidates]
    result=dict(schema_version=1, created_at_utc=utc_now(), archive=destination,
        policy='Same-filesystem rename only; no deletion; protect referenced run/library directories and all source saves, installed tools, state, assets, evidence and source code.',
        protected=[dict(path=str(p.relative_to(ROOT)),reason=reason) for p,reason in sorted(protected.items())],
        moves=moves, entries=sum(m['receipt']['entries'] for m in moves),
        logical_bytes=sum(m['receipt']['logical_bytes'] for m in moves),
        size_note='Logical file sizes include hardlinks. Archival does not free disk space.')
    atomic_json(REPORT,result)
    print(json.dumps(dict(plan=str(REPORT.relative_to(ROOT)),moves=len(moves),protected=len(protected),
                         entries=result['entries'],logical_gib=result['logical_bytes']/1024**3)))


def relocate(restore=False):
    assert_idle()
    data=read_json(REPORT); base=ROOT/data['archive']
    if not base.is_relative_to(ROOT/'other'):raise ValueError('Invalid archive destination')
    base.mkdir(parents=True, exist_ok=True)
    journal=base/'manifest.json'; atomic_json(journal,data)
    rows=reversed(data['moves']) if restore else data['moves']
    for row in rows:
        original=Path(os.path.abspath(ROOT/row['source']))
        archived=Path(os.path.abspath(ROOT/row['destination']))
        if not original.is_relative_to(ROOT):raise ValueError('Source outside workspace')
        if unit(original) is None and not (original.is_relative_to(ROOT/'artifacts/demo/portable')
                                           or original.is_relative_to(ROOT/'artifacts/demo/downloads')):
            raise ValueError('Path outside explicit cleanup scope')
        if not archived.is_relative_to(base):raise ValueError('Destination outside archive')
        source,destination=(archived,original) if restore else (original,archived)
        if destination.exists():
            if source.exists():raise FileExistsError(f'Refusing to overwrite {destination}')
            if tree_receipt(destination)!=row['receipt']:raise ValueError('Previously moved tree changed')
            continue
        if tree_receipt(source)!=row['receipt']:raise ValueError(f'Tree changed since plan: {source}')
        destination.parent.mkdir(parents=True,exist_ok=True)
        if source.lstat().st_dev!=destination.parent.stat().st_dev:raise ValueError('Cross-filesystem copying is forbidden')
        source.rename(destination)
        if tree_receipt(destination)!=row['receipt']:raise ValueError('Rename integrity check failed')
        row['status']='restored' if restore else 'archived'
        atomic_json(journal,data)
    data.update(status='restored' if restore else 'archived',finished_at_utc=utc_now())
    atomic_json(journal,data);atomic_json(REPORT,data)
    print(json.dumps(dict(status=data['status'],moves=len(data['moves']),manifest=str(journal.relative_to(ROOT)))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['plan','apply','restore'])
    parser.add_argument('--manifest',type=Path,help='Existing plan/archive manifest; useful for restoring an older archive')
    args=parser.parse_args()
    if args.manifest:
        REPORT=args.manifest.resolve()
        if not REPORT.is_relative_to(ROOT):parser.error('Manifest must be inside the workspace')
    if args.action=='plan':plan()
    else:relocate(args.action=='restore')
