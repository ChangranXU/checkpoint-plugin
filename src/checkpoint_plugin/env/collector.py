"""Collect provider environment state into checkpoint blobs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

from checkpoint_plugin.store import CheckpointStore
from checkpoint_plugin.types import EnvironmentState

from .providers import ProviderLayout


def collect_environment(
    cwd: Path,
    provider: ProviderLayout,
    store: CheckpointStore,
) -> EnvironmentState:
    cwd = cwd.expanduser().resolve()
    return EnvironmentState(
        provider=provider.name,
        model=_first_env("ANTHROPIC_MODEL", "CLAUDE_MODEL", "OPENAI_MODEL", "CODEX_MODEL"),
        permission_mode=_first_env("CLAUDE_PERMISSION_MODE", "CODEX_SANDBOX_MODE"),
        memory_files=_collect_tree(provider.memory_dir, store),
        mcp_config=_store_file(provider.mcp_config, store),
        skills=_collect_tree(provider.skills_dir, store),
        settings=_collect_settings(provider.settings_files, store),
        project_context=_collect_project_context(cwd, provider.project_files, store),
        extra={
            "provider_home": str(provider.home),
            "cwd": str(cwd),
        },
    )


def _first_env(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _store_file(path: Path | None, store: CheckpointStore) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    return store.store_blob(path.read_bytes())


def _collect_tree(root: Path | None, store: CheckpointStore) -> dict[str, str]:
    if root is None or not root.exists() or not root.is_dir():
        return {}
    result: dict[str, str] = {}
    for path in _iter_files(root):
        rel = path.relative_to(root).as_posix()
        result[rel] = store.store_blob(path.read_bytes())
    return result


def _collect_settings(paths: Iterable[Path], store: CheckpointStore) -> dict[str, str]:
    settings: dict[str, str] = {}
    for path in paths:
        if path.exists() and path.is_file():
            settings[path.name] = store.store_blob(path.read_bytes())
    return settings


def _collect_project_context(cwd: Path, project_files: list[str], store: CheckpointStore) -> dict[str, str]:
    context: dict[str, str] = {}
    for root in _ancestor_chain(_nearest_project_root(cwd), cwd):
        for rel_name in project_files:
            path = root / rel_name
            if path.exists() and path.is_file():
                key = path.relative_to(root).as_posix()
                context[str(root / key)] = store.store_blob(path.read_bytes())
    return context


def _iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def _nearest_project_root(cwd: Path) -> Path:
    for path in (cwd, *cwd.parents):
        if (path / ".git").exists():
            return path
    return cwd


def _ancestor_chain(root: Path, cwd: Path) -> list[Path]:
    try:
        relative = cwd.relative_to(root)
    except ValueError:
        return [cwd]

    paths = [root]
    current = root
    for part in relative.parts:
        current = current / part
        paths.append(current)
    return paths


def environment_to_blob(state: EnvironmentState, store: CheckpointStore) -> str:
    return store.store_json_blob(state.to_json())


def environment_from_blob(sha: str, store: CheckpointStore) -> EnvironmentState:
    data = store.load_json_blob(sha)
    if not isinstance(data, dict):
        raise ValueError(f"Environment blob {sha} is not a JSON object")
    return EnvironmentState.from_json(data)
