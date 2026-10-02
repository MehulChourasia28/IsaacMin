"""Reversible library clearing; source inputs and historical evidence are retained."""
from pathlib import Path
from isaacmin.io import atomic_json, read_json, utc_now


def visibility(root):
    path = Path(root) / 'state/demo_library.json'
    return read_json(path) if path.is_file() else dict(hidden_presets=[], archived_jobs=[])


def require_saved_preset(root, preset):
    if preset in visibility(root)['hidden_presets']:
        raise ValueError('This preset has no saved world. Create an Isaac world first, then select that conversion.')


def clear_saved_worlds(root, jobs):
    from .catalog import presets
    state = visibility(root)
    with jobs.connect() as db:
        rows = db.execute('SELECT id,status FROM jobs').fetchall()
        if any(row['status'] in ('queued', 'running') for row in rows):
            raise ValueError('Finish active jobs before clearing the saved-world library')
        state.update(hidden_presets=list(presets(root)),
            archived_jobs=sorted({row['id'] for row in rows} | set(state['archived_jobs'])),
            cleared_at_utc=utc_now(), policy='Hidden from studio and its result/download APIs; historical artifacts retained for evidence. Minecraft saves unchanged.')
        atomic_json(Path(root) / 'state/demo_library.json', state)
    return state
