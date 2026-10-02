from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .io import atomic_json, read_json
from .security import redact


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="isaacmin", description="Minecraft to Isaac Sim with explicit evidence gates")
    p.add_argument("--workspace", type=Path, default=Path.cwd())
    p.add_argument("--json", action="store_true", help="Machine readable output (also accepted after subcommand)")
    commands = p.add_subparsers(dest="command", required=True)
    for name in ("bootstrap", "inspect-source", "smoke", "build"):
        sub = commands.add_parser(name)
        sub.add_argument("--world", type=Path)
        sub.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
        if name in {"smoke", "build"}:
            sub.add_argument("--quality", choices=["strict"], default="strict")
    doc = commands.add_parser("doctor")
    demo = commands.add_parser('demo',help='Local four-preset studio with saved native images and durable jobs')
    demo.add_argument('--port',type=int,default=8765)
    commands.add_parser('demo-worker',help=argparse.SUPPRESS).add_argument('--lane',choices=['work','agent'],default='work')
    source_render=commands.add_parser('render-source',help='Actual Chunky aerial of a built region in its immutable Minecraft save')
    source_render.add_argument('--build',type=Path,required=True)
    source_render.add_argument('--output',type=Path,required=True)
    outdoor = commands.add_parser("build-outdoor", help="Above-ground conversion; caves and structures excluded")
    outdoor.add_argument("--world", type=Path)
    outdoor.add_argument("--extent", type=int, default=256)
    outdoor.add_argument("--center", nargs=2, type=float, metavar=("X", "Z"))
    outdoor.add_argument("--output", type=Path)
    outdoor.add_argument("--asset-library", type=Path, help="Explicit versioned native asset manifest; defaults to the selected project library")
    outdoor.add_argument("--no-capture", action="store_true", help="Build native geometry; leave Isaac capture explicitly not_run")
    outdoor.add_argument("--capture-set", choices=['full','focus'], default='full',
        help="focus captures two development views at unchanged render quality; full retains ground and aerial coverage")
    outdoor.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    render = commands.add_parser('render-outdoor',help='Capture an existing outdoor world without terrain reconstruction')
    render.add_argument('--build',type=Path,required=True)
    render.add_argument('--output',type=Path,required=True)
    render.add_argument('--views',choices=['full','overview','focus'],default='full')
    render.add_argument('--json',action='store_true',default=argparse.SUPPRESS)
    outdoor_package = commands.add_parser('package-outdoor', help='Portable outdoor development delivery; no realism pass implied')
    outdoor_package.add_argument('--build', type=Path, required=True)
    outdoor_package.add_argument('--output', type=Path)
    outdoor_package.add_argument('--reference-capture', type=Path,
        help='Completed render-outdoor directory to include as native replay evidence')
    outdoor_package.add_argument('--json', action='store_true', default=argparse.SUPPRESS)
    doc.add_argument("scope", choices=["host", "nim", "workers", "assets"], nargs="?", default="host")
    doc.add_argument("--recheck-access", action="store_true", help="Intentionally recheck provider access after a credential/account change")
    doc.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    commands.add_parser("secret", help="Masked local key prompt; never accepts a key argument")
    validate = commands.add_parser("validate")
    validate.add_argument("--build", required=True)
    validate.add_argument("--suite", choices=["all"], default="all")
    validate.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    for name in ("report", "package"):
        sub = commands.add_parser(name)
        sub.add_argument("--build", required=True)
        sub.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
        if name == "package":
            sub.add_argument("--require-qualified", action="store_true")
            sub.add_argument("--reference-capture", type=Path, help="Actual matching Isaac capture for portable replay")
    portable = commands.add_parser("verify-package")
    portable.add_argument("--package", type=Path, required=True)
    portable.add_argument("--output", type=Path, required=True, help="New directory for immutable replay evidence")
    portable.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    resume = commands.add_parser("resume")
    resume.add_argument("--run", required=True)
    resume.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    agent = commands.add_parser("agent")
    action = agent.add_subparsers(dest="action", required=True)
    run = action.add_parser("run")
    run.add_argument("--world", type=Path)
    run.add_argument("--quality", choices=["strict"], default="strict")
    run.add_argument("--watch", action="store_true", help="Opt in to source watching; idle runs do not call NIM")
    run.add_argument("--resume", metavar="AGENT_RUN", help="Resume a saved agent conversation and pending bounded tool call")
    run.add_argument("--json", action="store_true", default=argparse.SUPPRESS)
    return p


def dispatch(args) -> dict:
    root = args.workspace.resolve()
    if args.command=='demo':
        from .demo.server import serve
        serve(root,args.port)
        return {'status':'stopped'}
    if args.command=='demo-worker':
        from .demo.jobs import worker
        worker(root,args.lane)
        return {'status':'stopped'}
    if args.command=='render-source':
        from .demo.source_preview import render_source
        return render_source(root,args.build,args.output)
    if args.command == "build-outdoor":
        from .outdoor_pipeline import build_outdoor
        return build_outdoor(root,args.world,extent=args.extent,center=args.center,output=args.output,capture=not args.no_capture,asset_library=args.asset_library,capture_set=args.capture_set)
    if args.command == 'package-outdoor':
        from .outdoor_packaging import package_outdoor
        return package_outdoor(root,args.build,output=args.output,reference_capture=args.reference_capture)
    if args.command == 'render-outdoor':
        from .outdoor_render import render_outdoor
        return render_outdoor(root,args.build,args.output,views=args.views)
    if args.command == "secret":
        from .security import secret
        return {"status": "saved" if secret(root, prompt=True) else "missing", "key_disclosed": False}
    if args.command == "doctor":
        if args.scope == "host":
            from .bootstrap import inspect_host
            return inspect_host(root)
        if args.scope == "nim":
            from .adapters.nim import probe_nim
            return probe_nim(root, recheck_access=args.recheck_access)
        if args.scope == "assets":
            from .assets.providers import bootstrap_assets
            return bootstrap_assets(root)
        from .adapters.workers import probe_workers
        report = probe_workers(root)
        report["status"] = "pass" if report["cross_runtime_qualified"] else "blocked"
        return report
    if args.command == "bootstrap":
        from .pipeline import bootstrap
        return bootstrap(root, args.world)
    if args.command == "inspect-source":
        from .source import inspect_source, snapshot_world
        from .project import resolve_project
        config = resolve_project(root, args.world)
        snapshot = snapshot_world(Path(config["source_world"]), root / "work/snapshots")
        atomic_json(root / "artifacts/source/source_snapshot.json", snapshot)
        return inspect_source(Path(snapshot["snapshot_path"]), root / "artifacts/source")
    if args.command in {"build", "smoke"}:
        project_path=root/'state/resolved_project.json'
        if project_path.is_file() and read_json(project_path).get('terrain_mode')=='surface_navigation':
            from .outdoor_pipeline import build_outdoor
            return build_outdoor(root,args.world,extent=128 if args.command=='smoke' else 256)
        from .pipeline import build
        return build(root, args.world, integration_only=args.command == "smoke")
    if args.command == "resume":
        from .pipeline import resume
        return resume(root, args.run)
    if args.command == "validate":
        from .validation.runner import validate_build
        return validate_build(root, args.build)
    if args.command == "report":
        from .security import safe_path
        return read_json(safe_path(root / "worlds", args.build) / "qualification.json")
    if args.command == "package":
        from .packaging import package_build
        return package_build(root, args.build, require_qualified=args.require_qualified,
                             reference_capture=args.reference_capture)
    if args.command == "verify-package":
        from .portable import verify_portable
        return verify_portable(root, args.package, args.output.resolve())
    if args.command == "agent":
        from .agent.runtime import run
        return run(root, args.world, watch=args.watch, resume_run_id=args.resume)
    raise ValueError("Unsupported command")


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        result = redact(dispatch(args))
        status = result.get("status", "unknown")
        if args.json:
            print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        else:
            print(f"IsaacMin: {status}")
            for key in ("build_id", "run_id", "stage", "reason", "report", "package"):
                if key in result:
                    print(f"{key}: {result[key]}")
            for item in result.get("blockers", []):
                print("Blocked: " + str(item))
        return 2 if status in {"blocked", "fail", "failed", "incomplete", "blocked_provider", "blocked_repair_budget", "waiting_dependency"} else 0
    except KeyboardInterrupt:
        print("Interrupted; durable jobs remain resumable.", file=sys.stderr)
        return 130
    except Exception as exc:
        result = {"status": "failed", "error_type": type(exc).__name__, "reason": redact(str(exc))}
        print(json.dumps(result) if args.json else f"IsaacMin failed: {result['reason']}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
