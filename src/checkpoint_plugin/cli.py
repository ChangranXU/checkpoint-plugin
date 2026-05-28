"""Command-line interface for checkpoint-plugin."""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from .coordinator import CheckpointCoordinator, TurnRecord
from .env.collector import environment_from_blob
from .fs.snapshot import filesystem_from_blob
from .integrations.hook_installer import install_hooks, uninstall_hooks
from .paths import config_path, load_config, sessions_dir, write_config
from .resume import ResumeOptions, ResumeOrchestrator
from .retention import clean_keep_last
from .store import CheckpointStore
from .types import ResumePlan
from .ui.diff_viewer import show_diff_viewer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="checkpoint")
    sub = parser.add_subparsers(dest="command", required=True)

    save = sub.add_parser("save", help="Manual checkpoint of current state")
    save.add_argument("--session")
    save.add_argument("--note", default="")

    list_cmd = sub.add_parser("list", help="List sessions or turns")
    list_cmd.add_argument("--session")

    show = sub.add_parser("show", help="Show a checkpoint")
    show.add_argument("session")
    show.add_argument("turn", type=int)

    diff = sub.add_parser("diff", help="Diff current state against a checkpoint")
    diff.add_argument("session")
    diff.add_argument("turn", type=int)

    resume = sub.add_parser("resume", help="Restore a checkpoint")
    resume.add_argument("session")
    resume.add_argument("turn", type=int)
    resume.add_argument("--yes", action="store_true")
    resume.add_argument("--target")

    clean = sub.add_parser("clean", help="Apply retention")
    clean.add_argument("--keep-last", type=int, default=100)

    hooks = sub.add_parser("hooks", help="Install or uninstall agent lifecycle hooks")
    hooks.add_argument("action", choices=["install", "uninstall"])
    hooks.add_argument("provider", nargs="?", default="all", help="claude, codex, or all")

    config = sub.add_parser("config", help="Read/write plugin config")
    config.add_argument("action", choices=["get", "set"])
    config.add_argument("key")
    config.add_argument("value", nargs="?")

    args = parser.parse_args(argv)
    return int(_dispatch(args))


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "save":
        coordinator = CheckpointCoordinator(session_id=args.session)
        coordinator.on_session_start()
        manifest = coordinator.on_turn_end(TurnRecord(user_message=args.note, metadata={"source": "cli"}))
        print(f"Saved checkpoint {manifest.session_id} turn {manifest.turn_id}")
        return 0
    if args.command == "list":
        return _cmd_list(args.session)
    if args.command == "show":
        return _cmd_show(args.session, args.turn)
    if args.command == "diff":
        orchestrator = ResumeOrchestrator()
        print(orchestrator.plan(args.session, args.turn).render())
        return 0
    if args.command == "resume":
        cwd = Path(args.target).expanduser() if args.target else None
        orchestrator = ResumeOrchestrator(cwd=cwd)
        plan = orchestrator.plan(args.session, args.turn)
        store = CheckpointStore(sessions_dir() / args.session)
        try:
            confirm = _auto_confirm if args.yes else lambda text: _interactive_resume_confirm(text, plan, store)
            report = orchestrator.execute(plan, confirm)
        except RuntimeError as exc:
            if str(exc) == "Resume cancelled":
                print(str(exc), file=sys.stderr)
                return 1
            raise
        print(f"Restored into new session {report.new_session_id}")
        if report.target_cwd is not None:
            print(f"Workspace: {report.target_cwd}")
        print(f"Backup: {report.backup_dir}")
        print(f"Changed files: {len(report.changed_files)}")
        return 0
    if args.command == "clean":
        removed = clean_keep_last(args.keep_last)
        print(f"Removed {removed} old manifest(s)")
        return 0
    if args.command == "hooks":
        return _cmd_hooks(args.action, args.provider)
    if args.command == "config":
        return _cmd_config(args.action, args.key, args.value)
    raise AssertionError(args.command)


def _cmd_list(session: str | None) -> int:
    root = sessions_dir()
    if session is None:
        if not root.exists():
            return 0
        for child in sorted(root.iterdir()):
            if child.is_dir():
                metadata = _read_session_metadata(child)
                title = _display_metadata_value(metadata.get("session_title"))
                source = _display_metadata_value(metadata.get("source"))
                print(f"{child.name}  {title}  {source}")
        return 0

    store = CheckpointStore(root / session)
    for manifest in store.list_manifests():
        preview = manifest.user_message_preview.replace("\n", " ")
        print(f"{manifest.turn_id:04d}  {manifest.created_ts}  {preview}")
    return 0


def _read_session_metadata(session_dir: Path) -> dict[str, Any]:
    path = session_dir / "metadata.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _display_metadata_value(value: Any) -> str:
    return str(value) if value not in (None, "") else "-"


def _cmd_show(session: str, turn: int) -> int:
    store = CheckpointStore(sessions_dir() / session)
    manifest = store.read_manifest(turn)
    env = environment_from_blob(manifest.env_ref, store)
    fs = filesystem_from_blob(manifest.fs_ref, store)
    print(json.dumps(manifest.to_json(), indent=2, sort_keys=True))
    print()
    print("Environment:")
    print(json.dumps(env.to_json(), indent=2, sort_keys=True))
    print()
    print(f"Filesystem: {len(fs.files)} files at {fs.cwd}")
    return 0


def _cmd_config(action: str, key: str, value: str | None) -> int:
    config = load_config()
    if action == "get":
        if key == ".":
            print(json.dumps(config, indent=2, sort_keys=True))
        else:
            print(json.dumps(_get_nested(config, key), indent=2, sort_keys=True))
        return 0
    if value is None:
        print("checkpoint config set requires VALUE", file=sys.stderr)
        return 2
    _set_nested(config, key, _parse_value(value))
    write_config(config)
    print(f"Updated {config_path()}")
    return 0


def _cmd_hooks(action: str, provider: str) -> int:
    try:
        results = install_hooks(provider) if action == "install" else uninstall_hooks(provider)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    for result in results:
        state = "updated" if result.changed else "already current"
        print(f"{result.provider}: {state} {result.path}")
    return 0


def _interactive_resume_confirm(text: str, plan: ResumePlan, store: CheckpointStore) -> bool | ResumeOptions:
    print(text)
    while True:
        answer = input("Proceed? [y/N/d] ")
        if answer.lower() in {"y", "yes"}:
            return _interactive_resume_options(plan)
        if answer.lower() in {"d", "diff"}:
            show_diff_viewer(plan, store)
            continue
        return False


def _interactive_resume_options(plan: ResumePlan) -> ResumeOptions:
    answer = input("Restore where? [i]n-place/[c]opy (default: in-place) ")
    if answer.lower() not in {"c", "copy"}:
        return ResumeOptions(proceed=True)
    default_path = _default_copy_path(Path(plan.target_fs.cwd), plan.turn_id)
    raw_path = input(f"Copy folder [{default_path}]: ").strip()
    target_cwd = Path(raw_path).expanduser() if raw_path else default_path
    return ResumeOptions(proceed=True, target_cwd=target_cwd)


def _default_copy_path(cwd: Path, turn_id: int) -> Path:
    suffix = f"checkpoint-copy-{turn_id}-{uuid.uuid4().hex[:6]}"
    return cwd.parent / f"{cwd.name}-{suffix}"


def _auto_confirm(text: str) -> bool:
    print(text)
    return True


def _get_nested(data: dict[str, Any], key: str) -> Any:
    current: Any = data
    for part in key.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _set_nested(data: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    current = data
    for part in parts[:-1]:
        next_value = current.setdefault(part, {})
        if not isinstance(next_value, dict):
            next_value = {}
            current[part] = next_value
        current = next_value
    current[parts[-1]] = value


def _parse_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


if __name__ == "__main__":
    raise SystemExit(main())
