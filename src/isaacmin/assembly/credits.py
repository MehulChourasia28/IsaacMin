"""Keep original attribution and licence text inside the portable USD closure."""
import hashlib
from pathlib import Path
import shutil


def attach_credits(workspace, scene_path, assets):
    from pxr import Usd, Sdf
    root = Path(workspace).resolve()
    scene = Path(scene_path).resolve()
    entries = [root / 'ASSET_CREDITS.md']
    for asset in assets:
        for entry in asset.get('attribution_files', []):
            path = Path(entry['path']).resolve()
            if (not path.is_relative_to(root) or path.suffix.lower() not in ('.txt', '.md', '.html')
                    or path.name.startswith('.env')):
                raise ValueError('Invalid explicit native asset attribution file')
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
                raise ValueError('Original asset attribution changed')
            entries.append(path)
    records = []
    for path in sorted(set(entries)):
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        target = scene.parent / 'asset_credits' / (digest + path.suffix.lower())
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(path, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError('Portable attribution bytes changed')
        records.append({'path': target.relative_to(scene.parent).as_posix(), 'sha256': digest,
                        'original_name': path.name})
    stage = Usd.Stage.Open(str(scene))
    stage.GetDefaultPrim().CreateAttribute('isaacmin:creditFiles', Sdf.ValueTypeNames.AssetArray,
        custom=True).Set([Sdf.AssetPath('./' + entry['path']) for entry in records])
    stage.GetRootLayer().Save()
    return records
