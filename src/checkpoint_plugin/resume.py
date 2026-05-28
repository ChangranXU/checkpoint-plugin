"""Diff-first checkpoint resume orchestration."""

from __future__ import annotations

import shutil
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .env.collector import collect_environment, environment_from_blob
from .env.differ import diff_environments, render_diff
from .env.providers import detect_provider
from .env.restorer import restore_environment
from .fs.ignore import IgnoreMatcher
from .fs.restorer import diff_filesystems, render_fs_diff, restore_cwd
from .fs.snapshot import filesystem_from_blob, snapshot_cwd
from .paths import backups_dir, ensure_home, load_config, session_dir
from .store import CheckpointStore
from .types import ResumePlan, ResumeReport


class ResumeOrchestrator:
    def __init__(self, plugin_home: Path | None = None, cwd: Path | None = None) -> None:
        self.home = ensure_home(plugin_home)
        self.cwd = Path(cwd or Path.cwd()).expanduser().resolve()

    def plan(self, session_id: str, turn_id: int) -> ResumePlan:
        store = CheckpointStore(session_dir(session_id, self.home))
        manifest = store.read_manifest(turn_id)
        provider = detect_provider(self.cwd)
        current_env = collect_environment(self.cwd, provider, store)
        target_env = environment_from_blob(manifest.env_ref, store)
        config = load_config(self.home)
        ignore = IgnoreMatcher(self.cwd, config.get("exclude_patterns") or [])
        current_fs = snapshot_cwd(self.cwd, store, ignore)
        target_fs = filesystem_from_blob(manifest.fs_ref, store)
        env_diff = diff_environments(current_env, target_env)
        fs_diff = diff_filesystems(current_fs, target_fs)
        return ResumePlan(
            session_id=session_id,
            turn_id=turn_id,
            target_manifest=manifest,
            current_env=current_env,
            target_env=target_env,
            current_fs=current_fs,
            target_fs=target_fs,
            env_diff_text=render_diff(env_diff, current_env, target_env),
            fs_diff_text=render_fs_diff(fs_diff, target_fs.cwd),
        )

    def execute(self, plan: ResumePlan, confirm: Callable[[str], bool]) -> ResumeReport:
        rendered = plan.render()
        if not confirm(rendered):
            raise RuntimeError("Resume cancelled")
        original_store = CheckpointStore(session_dir(plan.session_id, self.home))
        backup_root = backups_dir(self.home) / f"{_stamp()}-{plan.session_id}"
        provider = detect_provider(self.cwd)
        env_report = restore_environment(
            plan.target_env,
            provider,
            original_store,
            backup_root / "environment",
        )
        config = load_config(self.home)
        ignore = IgnoreMatcher(self.cwd, config.get("exclude_patterns") or [])
        fs_report = restore_cwd(
            plan.target_fs,
            self.cwd,
            original_store,
            backup_root / "filesystem",
            ignore,
        )
        new_session_id = f"{plan.session_id}-resumed-from-{plan.turn_id}"
        self._copy_session_prefix(original_store, plan, new_session_id)
        return ResumeReport(
            new_session_id=new_session_id,
            backup_dir=str(backup_root),
            env=env_report,
            fs=fs_report,
        )

    def _copy_session_prefix(self, store: CheckpointStore, plan: ResumePlan, new_session_id: str) -> None:
        target_dir = session_dir(new_session_id, self.home)
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_store = CheckpointStore(target_dir)
        if (store.session_dir / "metadata.json").exists():
            shutil.copy2(store.session_dir / "metadata.json", target_dir / "metadata.json")
        for manifest in store.list_manifests():
            if manifest.turn_id <= plan.turn_id:
                target_store.write_manifest(replace(manifest, session_id=new_session_id))
        if store.blobs_dir.exists():
            shutil.copytree(store.blobs_dir, target_store.blobs_dir, dirs_exist_ok=True)
        target_store._atomic_write(
            target_store.trajectory_path,
            store.slice_trajectory(plan.target_manifest.trajectory_offset),
        )


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
