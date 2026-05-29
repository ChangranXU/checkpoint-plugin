"""Diff-first checkpoint resume orchestration."""

from __future__ import annotations

import json
import os
import shutil
import sys
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .env.collector import collect_environment, environment_from_blob
from .env.differ import diff_environments, render_diff
from .env.providers import layout_for_provider
from .env.restorer import restore_environment
from .fs.ignore import IgnoreMatcher
from .fs.restorer import diff_filesystems, render_fs_diff, restore_cwd
from .fs.snapshot import filesystem_from_blob, snapshot_cwd
from .integrations._trajectory_slicer import claude_key, codex_key
from .paths import backups_dir, ensure_home, load_config, session_dir
from .store import CheckpointStore
from .types import CheckpointManifest, ResumePlan, ResumeReport, TrajectoryReference


@dataclass(frozen=True)
class TrajectoryPrefix:
    data: bytes
    spans: dict[int, tuple[int, int, int]]


@dataclass(frozen=True)
class ResumeOptions:
    proceed: bool
    target_cwd: Path | None = None


class ResumeOrchestrator:
    def __init__(self, plugin_home: Path | None = None, cwd: Path | None = None) -> None:
        self.home = ensure_home(plugin_home)
        self.cwd = Path(cwd).expanduser().resolve() if cwd is not None else None

    def plan(self, session_id: str, turn_id: int) -> ResumePlan:
        store = CheckpointStore(session_dir(session_id, self.home))
        manifest = store.read_manifest(turn_id)
        target_env = environment_from_blob(manifest.env_ref, store)
        target_fs = filesystem_from_blob(manifest.fs_ref, store)
        cwd = self.cwd or Path(target_fs.cwd).expanduser().resolve()
        self.cwd = cwd
        provider = layout_for_provider(target_env.provider)
        current_env = collect_environment(cwd, provider, store)
        config = load_config(self.home)
        ignore = IgnoreMatcher(cwd, config.get("exclude_patterns") or [])
        current_fs = snapshot_cwd(cwd, store, ignore)
        ignore_plugin_hooks = bool(config.get("ignore_plugin_hook_diffs", True))
        env_diff = diff_environments(
            current_env,
            target_env,
            blob_loader=store.load_blob,
            ignore_plugin_hooks=ignore_plugin_hooks,
        )
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
            ignore_plugin_hooks=ignore_plugin_hooks,
        )

    def execute(self, plan: ResumePlan, confirm: Callable[[str], bool | ResumeOptions]) -> ResumeReport:
        rendered = plan.render()
        options = _coerce_resume_options(confirm(rendered))
        if not options.proceed:
            raise RuntimeError("Resume cancelled")
        original_store = CheckpointStore(session_dir(plan.session_id, self.home))
        backup_root = backups_dir(self.home) / f"{_stamp()}-{plan.session_id}-{uuid.uuid4().hex[:8]}"
        target_cwd = _prepare_resume_cwd(self.cwd, options.target_cwd)
        self.cwd = target_cwd
        provider = layout_for_provider(plan.target_env.provider)
        env_report = restore_environment(
            plan.target_env,
            provider,
            original_store,
            backup_root / "environment",
            ignore_plugin_hooks=plan.ignore_plugin_hooks,
        )
        config = load_config(self.home)
        ignore = IgnoreMatcher(target_cwd, config.get("exclude_patterns") or [])
        fs_report = restore_cwd(
            plan.target_fs,
            target_cwd,
            original_store,
            backup_root / "filesystem",
            ignore,
        )
        new_session_id = _new_resume_session_id()
        trajectory = _trajectory_prefix(original_store, plan)
        source_meta = _codex_source_session_meta(plan) if provider.name == "codex" else None
        provider_session_path = _write_provider_session(
            provider.name,
            provider.home,
            target_cwd,
            new_session_id,
            trajectory.data,
            plan.target_env.model,
            plan.target_env.permission_mode,
            source_meta,
        )
        _carry_provider_session_state(provider.name, provider.home, plan.session_id, new_session_id)
        self._copy_session_prefix(original_store, plan, new_session_id, provider_session_path, trajectory, target_cwd)
        return ResumeReport(
            new_session_id=new_session_id,
            backup_dir=str(backup_root),
            env=env_report,
            fs=fs_report,
            provider_session_path=str(provider_session_path) if provider_session_path is not None else None,
            target_cwd=str(target_cwd),
            resume_command=_resume_command(provider.name, new_session_id),
        )

    def _copy_session_prefix(
        self,
        store: CheckpointStore,
        plan: ResumePlan,
        new_session_id: str,
        provider_session_path: Path | None,
        trajectory: TrajectoryPrefix,
        cwd: Path,
    ) -> None:
        target_dir = session_dir(new_session_id, self.home)
        target_store = CheckpointStore(target_dir)
        _write_resumed_metadata(store, target_store, plan, new_session_id, provider_session_path, cwd)
        if store.blobs_dir.exists():
            shutil.copytree(store.blobs_dir, target_store.blobs_dir, dirs_exist_ok=True)
        for manifest in store.list_manifests():
            if manifest.turn_id <= plan.turn_id:
                target_store.write_manifest(
                    _resumed_manifest(manifest, new_session_id, provider_session_path, trajectory, target_store, cwd)
                )
        if trajectory.data:
            target_store._atomic_write(target_store.trajectory_path, trajectory.data)


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def _new_resume_session_id() -> str:
    return str(uuid.uuid4())


def _coerce_resume_options(value: bool | ResumeOptions) -> ResumeOptions:
    if isinstance(value, ResumeOptions):
        return value
    return ResumeOptions(proceed=bool(value))


def _prepare_resume_cwd(current_cwd: Path | None, target_cwd: Path | None) -> Path:
    if current_cwd is None:
        raise RuntimeError("Resume cwd is not initialized")
    current_cwd = current_cwd.expanduser().resolve()
    if target_cwd is None:
        return current_cwd
    target_cwd = target_cwd.expanduser().resolve()
    if target_cwd == current_cwd:
        return current_cwd
    if target_cwd.exists():
        if any(target_cwd.iterdir()):
            raise RuntimeError(f"Target folder is not empty: {target_cwd}")
    else:
        target_cwd.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(current_cwd, target_cwd, dirs_exist_ok=True)
    return target_cwd


def _trajectory_resume_offset(plan: ResumePlan) -> int:
    if plan.target_manifest.trajectory_end_offset is not None:
        return plan.target_manifest.trajectory_end_offset
    return plan.target_manifest.trajectory_offset


def _codex_source_session_meta(plan: ResumePlan) -> dict[str, object] | None:
    ref = plan.target_manifest.trajectory_ref
    if ref is None or ref.provider != "codex" or not ref.transcript_path:
        return None
    path = Path(ref.transcript_path).expanduser()
    if not path.is_file():
        return None
    try:
        with path.open("rb") as handle:
            first_line = handle.readline()
    except OSError:
        return None
    if not first_line.strip():
        return None
    try:
        record = json.loads(first_line.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(record, dict) or record.get("type") != "session_meta":
        return None
    payload = record.get("payload")
    return payload if isinstance(payload, dict) else None


def _trajectory_prefix(store: CheckpointStore, plan: ResumePlan) -> TrajectoryPrefix:
    chunks: list[bytes] = []
    spans: dict[int, tuple[int, int, int]] = {}
    offset = 0
    manifests = [m for m in store.list_manifests() if m.turn_id <= plan.turn_id]
    for manifest in manifests:
        if manifest.trajectory_ref is None:
            continue
        is_latest = manifest.turn_id == plan.turn_id
        try:
            chunk = _read_trajectory_slice_for_manifest(store, manifest, extend_to_eof=is_latest)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Warning: trajectory unavailable for turn {manifest.turn_id}: {exc}", file=sys.stderr)
            continue
        if not chunk:
            continue
        chunks.append(chunk)
        end_offset = offset + len(chunk)
        spans[manifest.turn_id] = (offset, end_offset, _count_jsonl_records(chunk))
        offset = end_offset
    if chunks:
        return TrajectoryPrefix(b"".join(chunks), spans)
    legacy = store.slice_trajectory(_trajectory_resume_offset(plan))
    if plan.target_manifest.trajectory_offset < len(legacy):
        spans[plan.turn_id] = (
            plan.target_manifest.trajectory_offset,
            len(legacy),
            _count_jsonl_records(legacy[plan.target_manifest.trajectory_offset :]),
        )
    return TrajectoryPrefix(legacy, spans)


def _read_trajectory_slice_for_manifest(
    store: CheckpointStore,
    manifest: CheckpointManifest,
    extend_to_eof: bool,
) -> bytes:
    ref = manifest.trajectory_ref
    if ref is None:
        return b""
    base = store.read_trajectory_slice(ref)
    if not extend_to_eof:
        return base
    tail = _recover_trailing_tail(ref)
    return base + tail


def _recover_trailing_tail(ref: TrajectoryReference) -> bytes:
    """Recover bytes flushed after the hook captured `end_offset`.

    Guarded so we don't pull in records from a new turn or a mid-flush write:
    - the candidate tail must end with a newline (no truncated JSON line);
    - no record in the tail may carry a per-turn key distinct from `ref`'s
      anchor key (no new turn started).
    """
    if not ref.transcript_path:
        return b""
    path = Path(ref.transcript_path).expanduser()
    try:
        size = path.stat().st_size
    except OSError:
        return b""
    if size <= ref.end_offset:
        return b""
    try:
        with path.open("rb") as handle:
            handle.seek(ref.end_offset)
            tail = handle.read(size - ref.end_offset)
    except OSError:
        return b""
    if not tail.endswith(b"\n"):
        return b""
    extractor = claude_key if ref.provider == "claude" else codex_key if ref.provider == "codex" else None
    if extractor is None:
        return tail
    anchor = _anchor_key(ref, extractor)
    for line in tail.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        key = extractor(record)
        if key is not None and key != anchor:
            return b""
    return tail


def _anchor_key(ref: TrajectoryReference, extractor: Callable[[dict[str, object]], object]) -> object:
    path = Path(ref.transcript_path).expanduser()
    try:
        with path.open("rb") as handle:
            handle.seek(ref.start_offset)
            data = handle.read(ref.end_offset - ref.start_offset)
    except OSError:
        return None
    for line in data.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(record, dict):
            continue
        key = extractor(record)
        if key is not None:
            return key
    return None


def _write_resumed_metadata(
    source_store: CheckpointStore,
    target_store: CheckpointStore,
    plan: ResumePlan,
    new_session_id: str,
    provider_session_path: Path | None,
    cwd: Path,
) -> None:
    metadata: dict[str, object] = {}
    metadata_path = source_store.session_dir / "metadata.json"
    if metadata_path.exists():
        try:
            raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            raw_metadata = {}
        if isinstance(raw_metadata, dict):
            metadata = raw_metadata
    metadata["session_id"] = new_session_id
    metadata["resumed_from_session_id"] = plan.session_id
    metadata["resumed_from_turn_id"] = plan.turn_id
    metadata["resumed_ts"] = _now()
    if provider_session_path is not None:
        metadata["provider_session_path"] = str(provider_session_path)
    metadata["cwd"] = str(cwd)
    target_store._atomic_write(
        target_store.session_dir / "metadata.json",
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _resumed_manifest(
    manifest: CheckpointManifest,
    new_session_id: str,
    provider_session_path: Path | None,
    trajectory: TrajectoryPrefix,
    target_store: CheckpointStore,
    cwd: Path,
) -> CheckpointManifest:
    fs_ref = _rewrite_fs_ref_for_cwd(manifest.fs_ref, target_store, cwd)
    trajectory_ref = manifest.trajectory_ref
    if trajectory_ref is not None and provider_session_path is not None:
        start_offset, end_offset, record_count = trajectory.spans.get(
            manifest.turn_id,
            (manifest.trajectory_offset, manifest.trajectory_end_offset or trajectory_ref.end_offset, trajectory_ref.record_count),
        )
        trajectory_ref = TrajectoryReference(
            provider=trajectory_ref.provider,
            transcript_path=str(provider_session_path),
            start_offset=start_offset,
            end_offset=end_offset,
            record_count=record_count,
        )
        return replace(
            manifest,
            session_id=new_session_id,
            fs_ref=fs_ref,
            trajectory_offset=start_offset,
            trajectory_end_offset=end_offset,
            trajectory_ref=trajectory_ref,
        )
    return replace(manifest, session_id=new_session_id, fs_ref=fs_ref, trajectory_ref=trajectory_ref)


def _rewrite_fs_ref_for_cwd(fs_ref: str, store: CheckpointStore, cwd: Path) -> str:
    snapshot = filesystem_from_blob(fs_ref, store)
    rewritten = replace(snapshot, cwd=str(cwd))
    return store.store_json_blob(rewritten.to_json())


def _write_provider_session(
    provider_name: str,
    provider_home: Path,
    cwd: Path,
    new_session_id: str,
    trajectory: bytes,
    model: str | None,
    permission_mode: str | None,
    source_meta: dict[str, object] | None = None,
) -> Path | None:
    if not trajectory:
        return None
    if provider_name == "codex":
        return _write_codex_session(
            provider_home, cwd, new_session_id, trajectory, model, permission_mode, source_meta
        )
    if provider_name == "claude":
        return _write_claude_session(provider_home, cwd, new_session_id, trajectory, model, permission_mode)
    return None


def _write_codex_session(
    codex_home: Path,
    cwd: Path,
    new_session_id: str,
    trajectory: bytes,
    model: str | None,
    permission_mode: str | None,
    source_meta: dict[str, object] | None,
) -> Path:
    now = datetime.now(timezone.utc)
    session_dir_path = codex_home / "sessions" / now.strftime("%Y") / now.strftime("%m") / now.strftime("%d")
    path = session_dir_path / f"rollout-{now.strftime('%Y-%m-%dT%H-%M-%S')}-{new_session_id}.jsonl"
    _write_bytes_atomic(
        path,
        _rewrite_codex_trajectory(trajectory, new_session_id, cwd, model, permission_mode, source_meta),
    )
    return path


def _write_claude_session(
    claude_home: Path,
    cwd: Path,
    new_session_id: str,
    trajectory: bytes,
    model: str | None,
    permission_mode: str | None,
) -> Path:
    path = claude_home / "projects" / _claude_project_dir_name(cwd) / f"{new_session_id}.jsonl"
    _write_bytes_atomic(path, _rewrite_claude_trajectory(trajectory, new_session_id, cwd, model, permission_mode))
    return path


def _rewrite_codex_trajectory(
    trajectory: bytes,
    new_session_id: str,
    cwd: Path,
    model: str | None,
    permission_mode: str | None,
    source_meta: dict[str, object] | None,
) -> bytes:
    lines: list[bytes] = []
    records = _jsonl_records(trajectory)
    if not records or records[0].get("type") != "session_meta":
        lines.append(_json_line(_codex_session_meta(new_session_id, cwd, source_meta)))
    for record in records:
        payload = record.get("payload")
        if isinstance(payload, dict):
            if record.get("type") == "session_meta":
                _apply_preserved_meta_fields(payload, source_meta)
                _mark_codex_session_visible(payload)
            if "id" in payload:
                payload["id"] = new_session_id
            if "thread_id" in payload:
                payload["thread_id"] = new_session_id
            if "cwd" in payload:
                payload["cwd"] = str(cwd)
            if payload.get("type") == "turn_context":
                if model:
                    payload["model"] = model
                if permission_mode:
                    payload["permission_profile"] = permission_mode
        if "id" in record:
            record["id"] = new_session_id
        if "session_id" in record:
            record["session_id"] = new_session_id
        lines.append(_json_line(record))
    return b"".join(lines)


def _codex_session_meta(
    new_session_id: str,
    cwd: Path,
    source_meta: dict[str, object] | None,
) -> dict[str, object]:
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    payload: dict[str, object] = {
        "id": new_session_id,
        "timestamp": now,
        "cwd": str(cwd),
    }
    _apply_preserved_meta_fields(payload, source_meta)
    _mark_codex_session_visible(payload)
    return {
        "timestamp": now,
        "type": "session_meta",
        "payload": payload,
    }


_PRESERVED_CODEX_META_FIELDS = (
    "cli_version",
    "model_provider",
    "base_instructions",
    "dynamic_tools",
    "git",
)


def _apply_preserved_meta_fields(
    payload: dict[str, object], source_meta: dict[str, object] | None
) -> None:
    if not source_meta:
        return
    for key in _PRESERVED_CODEX_META_FIELDS:
        if key in payload:
            continue
        if key in source_meta:
            payload[key] = source_meta[key]


def _mark_codex_session_visible(payload: dict[str, object]) -> None:
    payload["originator"] = "Codex Desktop"
    payload["source"] = "vscode"
    payload["thread_source"] = "user"


def _rewrite_claude_trajectory(
    trajectory: bytes,
    new_session_id: str,
    cwd: Path,
    model: str | None,
    permission_mode: str | None,
) -> bytes:
    lines: list[bytes] = []
    last_uuid: str | None = None
    uuid_map: dict[str, str] = {}
    records = _jsonl_records(trajectory)
    records = _ensure_permission_mode_record(records, permission_mode, new_session_id)
    for record in records:
        record["sessionId"] = new_session_id
        if "cwd" in record:
            record["cwd"] = str(cwd)
        if model and "model" in record:
            record["model"] = model
        if permission_mode and record.get("type") == "permission-mode":
            record["permissionMode"] = permission_mode
        if isinstance(record.get("uuid"), str):
            old_uuid = str(record["uuid"])
            new_uuid = str(uuid.uuid4())
            uuid_map[old_uuid] = new_uuid
            record["uuid"] = new_uuid
        if isinstance(record.get("parentUuid"), str):
            record["parentUuid"] = uuid_map.get(str(record["parentUuid"]), last_uuid)
        elif "parentUuid" in record and record.get("type") not in {"summary", "permission-mode"}:
            record["parentUuid"] = last_uuid
        if isinstance(record.get("uuid"), str) and record.get("type") in {"user", "assistant"}:
            last_uuid = str(record["uuid"])
        lines.append(_json_line(record))
    return b"".join(lines)


def _ensure_permission_mode_record(
    records: list[dict[str, object]],
    permission_mode: str | None,
    new_session_id: str,
) -> list[dict[str, object]]:
    if not permission_mode:
        return records
    if any(record.get("type") == "permission-mode" for record in records):
        return records
    synthetic = {
        "type": "permission-mode",
        "permissionMode": permission_mode,
        "sessionId": new_session_id,
    }
    insert_at = 0
    for idx, record in enumerate(records):
        if record.get("type") == "user":
            insert_at = idx
            break
        insert_at = idx + 1
    return [*records[:insert_at], synthetic, *records[insert_at:]]


def _claude_project_dir_name(cwd: Path) -> str:
    return str(cwd).replace("/", "-")


def _jsonl_records(data: bytes) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line in data.splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            records.append(value)
    return records


def _json_line(record: dict[str, object]) -> bytes:
    return (json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def _write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _count_jsonl_records(data: bytes) -> int:
    return sum(1 for line in data.splitlines() if line.strip())


def _resume_command(provider_name: str, new_session_id: str) -> str | None:
    if provider_name == "claude":
        return f"claude --resume {new_session_id}"
    if provider_name == "codex":
        return f"codex resume {new_session_id}"
    return None


def _carry_provider_session_state(
    provider_name: str,
    provider_home: Path,
    old_session_id: str,
    new_session_id: str,
) -> None:
    """Reuse the original session's append-only state under the new session id.

    Hardlinks (not copies) so the resumed session forks cleanly: shared baseline
    blobs cost zero extra disk, and any new writes by the resumed session land
    on new inodes without touching the original.
    """
    if provider_name != "claude":
        return
    _hardlink_tree(
        provider_home / "file-history" / old_session_id,
        provider_home / "file-history" / new_session_id,
    )
    _hardlink_todos(provider_home / "todos", old_session_id, new_session_id)
    _carry_claude_subagents(provider_home, old_session_id, new_session_id)


def _carry_claude_subagents(provider_home: Path, old_session_id: str, new_session_id: str) -> None:
    """Carry a session's subagent transcripts to the resumed session (B4).

    Claude stores subagents under `projects/<project>/<session>/subagents/`.
    Hardlinking them under the new session id lets a resumed run still see the
    subagent context that the parent turn depended on.
    """
    projects_root = provider_home / "projects"
    if not projects_root.exists() or not projects_root.is_dir():
        return
    for project_dir in projects_root.iterdir():
        if not project_dir.is_dir():
            continue
        src = project_dir / old_session_id / "subagents"
        if src.exists() and src.is_dir():
            _hardlink_tree(src, project_dir / new_session_id / "subagents")


def _hardlink_tree(src: Path, dst: Path) -> None:
    if not src.exists() or not src.is_dir():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for entry in src.rglob("*"):
        rel = entry.relative_to(src)
        target = dst / rel
        if entry.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not entry.is_file():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            continue
        try:
            os.link(entry, target)
        except OSError:
            shutil.copy2(entry, target)


def _hardlink_todos(todos_dir: Path, old_session_id: str, new_session_id: str) -> None:
    if not todos_dir.exists() or not todos_dir.is_dir():
        return
    for entry in todos_dir.glob(f"{old_session_id}-*"):
        if not entry.is_file():
            continue
        target = todos_dir / entry.name.replace(old_session_id, new_session_id, 1)
        if target.exists():
            continue
        try:
            os.link(entry, target)
        except OSError:
            shutil.copy2(entry, target)
