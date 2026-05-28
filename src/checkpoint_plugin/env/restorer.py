"""Restore provider environment state with backups."""

from __future__ import annotations

import shutil
from pathlib import Path

from checkpoint_plugin.store import CheckpointStore
from checkpoint_plugin.types import EnvironmentState, RestoreReport

from .providers import ProviderLayout


def restore_environment(
    target: EnvironmentState,
    provider: ProviderLayout,
    store: CheckpointStore,
    backup_dir: Path,
) -> RestoreReport:
    changed: list[str] = []
    backed_up: list[str] = []

    changed.extend(_restore_tree(target.memory_files, provider.memory_dir, store, backup_dir / "memory", backed_up))
    changed.extend(_restore_tree(target.skills, provider.skills_dir, store, backup_dir / "skills", backed_up))
    if provider.mcp_config is not None:
        changed.extend(_restore_optional_file(target.mcp_config, provider.mcp_config, store, backup_dir / "mcp", backed_up))
    changed.extend(_restore_settings(target.settings, provider.settings_files, store, backup_dir / "settings", backed_up))
    changed.extend(_restore_project_context(target.project_context, store, backup_dir / "project-context", backed_up))

    return RestoreReport(changed=changed, backed_up=backed_up, backup_dir=str(backup_dir))


def _restore_tree(
    target: dict[str, str],
    root: Path | None,
    store: CheckpointStore,
    backup_dir: Path,
    backed_up: list[str],
) -> list[str]:
    if root is None:
        return []
    changed: list[str] = []
    existing = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if root.exists() and path.is_file()
    }
    for rel, path in existing.items():
        if rel not in target:
            _backup(path, backup_dir / rel, backed_up)
            path.unlink()
            changed.append(str(path))
    for rel, sha in target.items():
        path = root / rel
        current = path.read_bytes() if path.exists() and path.is_file() else None
        wanted = store.load_blob(sha)
        if current != wanted:
            if path.exists():
                _backup(path, backup_dir / rel, backed_up)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(wanted)
            changed.append(str(path))
    return changed


def _restore_settings(
    settings: dict[str, str],
    settings_files: list[Path],
    store: CheckpointStore,
    backup_dir: Path,
    backed_up: list[str],
) -> list[str]:
    by_name = {path.name: path for path in settings_files}
    changed: list[str] = []
    for name, path in by_name.items():
        if name not in settings and path.exists():
            _backup(path, backup_dir / name, backed_up)
            path.unlink()
            changed.append(str(path))
    for name, sha in settings.items():
        path = by_name.get(name)
        if path is None and settings_files:
            path = settings_files[0].parent / name
        if path is not None:
            changed.append(str(_restore_blob_to(sha, path, store, backup_dir, backed_up)))
    return changed


def _restore_optional_file(
    sha: str | None,
    path: Path,
    store: CheckpointStore,
    backup_dir: Path,
    backed_up: list[str],
) -> list[str]:
    if sha is None:
        if path.exists():
            _backup(path, backup_dir / path.name, backed_up)
            path.unlink()
            return [str(path)]
        return []
    return [str(_restore_blob_to(sha, path, store, backup_dir, backed_up))]


def _restore_project_context(
    project_context: dict[str, str],
    store: CheckpointStore,
    backup_dir: Path,
    backed_up: list[str],
) -> list[str]:
    changed: list[str] = []
    for key, sha in project_context.items():
        path = Path(key)
        if not path.is_absolute():
            continue
        changed.append(str(_restore_blob_to(sha, path, store, backup_dir / _mirror_path(path), backed_up)))
    return changed


def _restore_blob_to(
    sha: str,
    path: Path,
    store: CheckpointStore,
    backup_path_or_dir: Path,
    backed_up: list[str],
) -> Path:
    wanted = store.load_blob(sha)
    current = path.read_bytes() if path.exists() and path.is_file() else None
    if current != wanted:
        if path.exists():
            backup_path = backup_path_or_dir if backup_path_or_dir.suffix else backup_path_or_dir / path.name
            _backup(path, backup_path, backed_up)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(wanted)
    return path


def _backup(path: Path, backup_path: Path, backed_up: list[str]) -> None:
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup_path)
    backed_up.append(str(backup_path))


def _mirror_path(path: Path) -> Path:
    return Path(*path.parts[1:]) if path.is_absolute() else path
