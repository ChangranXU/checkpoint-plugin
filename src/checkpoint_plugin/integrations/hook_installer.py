"""Install and remove checkpoint lifecycle hooks for agent CLIs."""

from __future__ import annotations

import json
import os
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HookInstallResult:
    provider: str
    path: Path
    changed: bool


@dataclass(frozen=True)
class HookSpec:
    provider: str
    path: Path
    events: dict[str, list[dict[str, Any]]]
    commands: frozenset[str]


def install_hooks(provider: str) -> list[HookInstallResult]:
    return [_apply(spec, install=True) for spec in _selected_specs(provider)]


def uninstall_hooks(provider: str) -> list[HookInstallResult]:
    return [_apply(spec, install=False) for spec in _selected_specs(provider)]


def _selected_specs(provider: str) -> list[HookSpec]:
    normalized = provider.strip().lower().replace("_", "-")
    specs = {
        "claude": _claude_spec(),
        "claude-code": _claude_spec(),
        "codex": _codex_spec(),
    }
    if normalized == "all":
        return [_claude_spec(), _codex_spec()]
    if normalized not in specs:
        raise ValueError(f"Unknown provider {provider!r}; expected claude, codex, or all")
    return [specs[normalized]]


def _claude_spec() -> HookSpec:
    module = "checkpoint_plugin.integrations.claude_code_hook"
    command_start = _module_command(module, "session_start")
    command_turn = _module_command(module, "turn_end")
    return HookSpec(
        provider="claude",
        path=_base_home() / ".claude" / "settings.json",
        commands=frozenset({command_start, command_turn, *_legacy_commands(module)}),
        events={
            "SessionStart": [_entry("*", command_start)],
            "Stop": [_entry("*", command_turn)],
        },
    )


def _codex_spec() -> HookSpec:
    module = "checkpoint_plugin.integrations.codex_hook"
    command_start = _module_command(module, "session_start")
    command_turn = _module_command(module, "turn_end")
    return HookSpec(
        provider="codex",
        path=Path(os.environ.get("CODEX_HOME", str(_base_home() / ".codex"))).expanduser() / "hooks.json",
        commands=frozenset({command_start, command_turn, *_legacy_commands(module)}),
        events={
            "SessionStart": [
                _entry("startup|resume|clear|compact", command_start, "Creating checkpoint session metadata")
            ],
            "Stop": [_entry(None, command_turn, "Saving checkpoint")],
        },
    )


def _module_command(module: str, event: str) -> str:
    return f"{shlex.quote(sys.executable)} -m {module} {event}"


def _legacy_commands(module: str) -> tuple[str, str]:
    return (
        f"python -m {module} session_start",
        f"python -m {module} turn_end",
    )


def _entry(matcher: str | None, command: str, status_message: str | None = None) -> dict[str, Any]:
    hook: dict[str, Any] = {"type": "command", "command": command}
    if status_message is not None:
        hook["statusMessage"] = status_message
    entry: dict[str, Any] = {"hooks": [hook]}
    if matcher is not None:
        entry["matcher"] = matcher
    return entry


def _apply(spec: HookSpec, install: bool) -> HookInstallResult:
    data = _read_json(spec.path)
    before = json.dumps(data, sort_keys=True, separators=(",", ":"))
    hooks = data.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        hooks = {}
        data["hooks"] = hooks

    if install:
        current_commands = _current_commands(spec)
        _remove_commands(hooks, spec.commands - current_commands)
        _remove_commands_from_unmanaged_events(hooks, frozenset(spec.events), spec.commands)
        for event, entries in spec.events.items():
            event_entries = hooks.setdefault(event, [])
            if not isinstance(event_entries, list):
                event_entries = []
                hooks[event] = event_entries
            for entry in entries:
                if not _has_command(event_entries, _first_command(entry)):
                    event_entries.append(entry)
    else:
        _remove_commands(hooks, spec.commands)

    after = json.dumps(data, sort_keys=True, separators=(",", ":"))
    changed = before != after
    if changed or install:
        _write_json(spec.path, data)
    return HookInstallResult(provider=spec.provider, path=spec.path, changed=changed)


def _current_commands(spec: HookSpec) -> frozenset[str]:
    commands: set[str] = set()
    for entries in spec.events.values():
        for entry in entries:
            command = _first_command(entry)
            if command is not None:
                commands.add(command)
    return frozenset(commands)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _has_command(entries: list[Any], command: str | None) -> bool:
    return command is not None and any(command == existing for existing in _iter_commands(entries))


def _iter_commands(entries: list[Any]) -> list[str]:
    commands: list[str] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        hooks = entry.get("hooks")
        if not isinstance(hooks, list):
            continue
        for hook in hooks:
            if isinstance(hook, dict) and hook.get("type") == "command" and isinstance(hook.get("command"), str):
                commands.append(hook["command"])
    return commands


def _first_command(entry: dict[str, Any]) -> str | None:
    hooks = entry.get("hooks")
    if not isinstance(hooks, list):
        return None
    for hook in hooks:
        if isinstance(hook, dict) and isinstance(hook.get("command"), str):
            return hook["command"]
    return None


def _remove_commands(hooks: dict[str, Any], commands: frozenset[str]) -> None:
    for event in list(hooks):
        entries = hooks[event]
        if not isinstance(entries, list):
            continue
        kept_entries: list[Any] = []
        for entry in entries:
            if not isinstance(entry, dict):
                kept_entries.append(entry)
                continue
            entry_hooks = entry.get("hooks")
            if not isinstance(entry_hooks, list):
                kept_entries.append(entry)
                continue
            kept_hooks = [
                hook
                for hook in entry_hooks
                if not (
                    isinstance(hook, dict)
                    and hook.get("type") == "command"
                    and hook.get("command") in commands
                )
            ]
            if kept_hooks:
                new_entry = dict(entry)
                new_entry["hooks"] = kept_hooks
                kept_entries.append(new_entry)
        if kept_entries:
            hooks[event] = kept_entries
        else:
            del hooks[event]
    if not hooks:
        hooks.clear()


def _remove_commands_from_unmanaged_events(
    hooks: dict[str, Any],
    managed_events: frozenset[str],
    commands: frozenset[str],
) -> None:
    unmanaged = {event: entries for event, entries in hooks.items() if event not in managed_events}
    _remove_commands(unmanaged, commands)
    for event in list(hooks):
        if event not in managed_events:
            if event in unmanaged:
                hooks[event] = unmanaged[event]
            else:
                del hooks[event]


def _base_home() -> Path:
    return Path(os.environ.get("TEST_HOME", str(Path.home()))).expanduser()
