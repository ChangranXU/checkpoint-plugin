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
from .integrations._trajectory_slicer import claude_key, codex_key, jsonl_count_records
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
        _refuse_subagent_resume(store, self.home)
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
        if provider.name == "codex" and provider_session_path is not None:
            _append_codex_session_index(
                provider.home, new_session_id, _source_session_title(original_store)
            )
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
        # P4-3: realign spans to the REWRITTEN provider file so resumed manifests'
        # byte offsets match the file their trajectory_ref points at (otherwise a
        # resume-of-a-resume reads stale raw-concat offsets and drops records).
        realigned = replace(
            trajectory,
            spans=_realign_spans_to_provider_file(provider_session_path, trajectory.spans),
        )
        for manifest in store.list_manifests():
            if manifest.turn_id <= plan.turn_id:
                target_store.write_manifest(
                    _resumed_manifest(manifest, new_session_id, provider_session_path, realigned, target_store, cwd)
                )
        if trajectory.data:
            target_store._atomic_write(target_store.trajectory_path, trajectory.data)


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def _refuse_subagent_resume(store: CheckpointStore, home: Path) -> None:
    """Refuse to resume a subagent checkpoint standalone, redirect to parent (H2).

    A subagent is never a real entry point: it is spawned by a parent `Task`
    tool_use, its runtime context is the Task prompt plus an agent-type system
    prompt that is not in the transcript, and the parent's context is absent.
    Synthesizing a standalone top-level session would fabricate a session the
    provider never produced and diverge immediately. Instead we point the user at
    the parent turn that spawned this subagent, where the subagent context is
    carried (H3) and the Task result already lives in the parent thread.
    """
    metadata = _read_session_metadata(store)
    lineage = metadata.get("lineage")
    if not isinstance(lineage, dict):
        return
    parent_session_id = lineage.get("parent_session_id")
    if not isinstance(parent_session_id, str) or not parent_session_id:
        return
    agent_id = lineage.get("agent_id") if isinstance(lineage.get("agent_id"), str) else None
    turn_id = _parent_turn_for_subagent(home, parent_session_id, agent_id, metadata)
    target = f"{parent_session_id} {turn_id}" if turn_id is not None else parent_session_id
    raise RuntimeError(
        "Cannot resume a subagent standalone; a subagent has no faithful "
        "standalone session. Resume its parent instead: "
        f"checkpoint resume {target}"
    )


def _read_session_metadata(store: CheckpointStore) -> dict[str, object]:
    metadata_path = store.session_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _parent_turn_for_subagent(
    home: Path,
    parent_session_id: str,
    agent_id: str | None,
    subagent_metadata: dict[str, object],
) -> int | None:
    """Best-effort parent turn that spawned this subagent (for the redirect).

    Prefer the parent turn whose trajectory slice references the subagent's
    agent_id (the `Task` tool_use that launched it). Fall back to the latest
    parent turn created at or before the subagent's start, else the latest turn.
    """
    try:
        parent_store = CheckpointStore(session_dir(parent_session_id, home))
        manifests = parent_store.list_manifests()
    except OSError:
        return None
    if not manifests:
        return None
    if agent_id:
        for manifest in manifests:
            if _manifest_references_agent(manifest, agent_id):
                return manifest.turn_id
    start_ts = subagent_metadata.get("start_ts")
    if isinstance(start_ts, str):
        eligible = [m for m in manifests if m.created_ts <= start_ts]
        if eligible:
            return max(eligible, key=lambda m: m.turn_id).turn_id
    return max(manifests, key=lambda m: m.turn_id).turn_id


def _manifest_references_agent(manifest: CheckpointManifest, agent_id: str) -> bool:
    ref = manifest.trajectory_ref
    if ref is None or not ref.transcript_path:
        return False
    path = Path(ref.transcript_path).expanduser()
    try:
        with path.open("rb") as handle:
            handle.seek(ref.start_offset)
            data = handle.read(max(0, ref.end_offset - ref.start_offset))
    except OSError:
        return False
    return agent_id.encode("utf-8") in data


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
    # F3: a forked/resumed session's first captured turn anchors mid-transcript
    # (the new promptId), but the inherited pre-fork history lives inline in the
    # SAME transcript at [0:first_start_offset]. Prepend it so the resumed
    # provider session reproduces the full context rather than starting amnesiac.
    inherited = _inherited_fork_prefix(manifests)
    if inherited:
        chunks.append(inherited)
        offset = len(inherited)
    for manifest in manifests:
        if manifest.trajectory_ref is None:
            continue
        is_latest = manifest.turn_id == plan.turn_id
        try:
            chunk = _read_trajectory_slice_for_manifest(store, manifest, extend_to_eof=is_latest)
        except (OSError, ValueError) as exc:
            print(f"Warning: trajectory unavailable for turn {manifest.turn_id}: {exc}", file=sys.stderr)
            continue
        if not chunk:
            continue
        chunks.append(chunk)
        end_offset = offset + len(chunk)
        spans[manifest.turn_id] = (offset, end_offset, jsonl_count_records(chunk))
        offset = end_offset
    if chunks:
        return TrajectoryPrefix(b"".join(chunks), spans)
    legacy = store.slice_trajectory(_trajectory_resume_offset(plan))
    if plan.target_manifest.trajectory_offset < len(legacy):
        spans[plan.turn_id] = (
            plan.target_manifest.trajectory_offset,
            len(legacy),
            jsonl_count_records(legacy[plan.target_manifest.trajectory_offset :]),
        )
    return TrajectoryPrefix(legacy, spans)


def _inherited_fork_prefix(manifests: list[CheckpointManifest]) -> bytes:
    """Bytes of inherited history preceding the first captured turn (F3).

    When the earliest turn's slice begins past byte 0, the records before it are
    pre-fork context the provider wrote inline into the same transcript. We read
    `[0:start_offset]` from that transcript so resume restores the full thread.
    Returns b"" for normal sessions (first turn anchored at byte 0) or when the
    transcript is gone.
    """
    first = next((m for m in manifests if m.trajectory_ref is not None), None)
    if first is None or first.trajectory_ref is None:
        return b""
    ref = first.trajectory_ref
    if not ref.transcript_path or ref.start_offset <= 0:
        return b""
    path = Path(ref.transcript_path).expanduser()
    try:
        with path.open("rb") as handle:
            prefix = handle.read(ref.start_offset)
    except OSError:
        return b""
    # start_offset is a line boundary; guard against a partial trailing line.
    if prefix and not prefix.endswith(b"\n"):
        cut = prefix.rfind(b"\n")
        prefix = prefix[: cut + 1] if cut >= 0 else b""
    return prefix


def _realign_spans_to_provider_file(
    provider_session_path: Path | None,
    spans: dict[int, tuple[int, int, int]],
) -> dict[int, tuple[int, int, int]]:
    """Recompute turn spans as line-aligned byte ranges over the REWRITTEN file (P4-3).

    `_write_provider_session` re-serializes the trajectory (sort_keys, uuid remap,
    a synthetic leading permission-mode/session_meta record), so the raw-concat
    spans from `_trajectory_prefix` no longer align to the file the resumed
    manifests point at. Reading them later (resume-of-a-resume) raw-seeks mid-line
    and drops records. We retile the rewritten file by record count: records with
    no owning turn (inherited fork prefix + any synthetic record) fold into the
    earliest turn, then each later turn takes its own record_count. Produces
    full, gap-free, boundary-aligned coverage.
    """
    if provider_session_path is None or not spans:
        return spans
    try:
        data = provider_session_path.read_bytes()
    except OSError:
        return spans
    line_ends: list[int] = []  # byte offset just past each non-blank record
    offset = 0
    for line in data.splitlines(keepends=True):
        end = offset + len(line)
        if line.strip():
            line_ends.append(end)
        offset = end
    total = len(line_ends)
    if total == 0:
        return spans
    ordered = sorted(spans.items())
    assigned = sum(count for _, (_, _, count) in ordered)
    # Leading records (inherited prefix + synthetic) fold into the first turn.
    leading = max(0, total - assigned)
    realigned: dict[int, tuple[int, int, int]] = {}
    consumed = 0
    start_byte = 0
    for idx, (turn_id, (_, _, count)) in enumerate(ordered):
        take = count + (leading if idx == 0 else 0)
        consumed = min(consumed + take, total)
        end_byte = line_ends[consumed - 1] if consumed > 0 else start_byte
        realigned[turn_id] = (start_byte, end_byte, take)
        start_byte = end_byte
    # Safety net: the last turn always extends to EOF (covers any rounding).
    last_turn = ordered[-1][0]
    last_start, _, last_count = realigned[last_turn]
    realigned[last_turn] = (last_start, len(data), last_count)
    return realigned


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


def _first_session_meta_id(records: list[dict[str, object]]) -> str | None:
    """Id of the first session_meta record = the original session id (P4-5)."""
    for record in records:
        if record.get("type") == "session_meta":
            payload = record.get("payload")
            if isinstance(payload, dict):
                value = payload.get("id")
                return value if isinstance(value, str) and value else None
            return None
    return None


def _source_session_title(store: CheckpointStore) -> str | None:
    """The source session's recorded title (for the codex resume index, M5)."""
    metadata_path = store.session_dir / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(metadata, dict):
        return None
    title = metadata.get("session_title")
    return title if isinstance(title, str) and title else None


def _append_codex_session_index(
    codex_home: Path, new_session_id: str, title: str | None
) -> None:
    """Register the resumed codex session so the picker can discover it (M5).

    `~/.codex/session_index.jsonl` is a JSONL of `{id, thread_name, updated_at}`
    that drives the Codex resume picker. Resume writes the rollout file but never
    registered it here, so the new session was invisible in the list. Append an
    entry (rewriting the whole file atomically) so it shows up.
    """
    index_path = codex_home / "session_index.jsonl"
    entry = {"id": new_session_id, "thread_name": title, "updated_at": _now()}
    try:
        existing = index_path.read_bytes() if index_path.exists() else b""
    except OSError:
        existing = b""
    if existing and not existing.endswith(b"\n"):
        existing += b"\n"
    _write_bytes_atomic(index_path, existing + _json_line(entry))


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


_TRANSIENT_CODEX_EVENT_MSGS = ("thread_rolled_back",)


def _is_transient_codex_event(record: dict[str, object]) -> bool:
    """Fork-control event_msgs that describe the PARENT's history surgery (M1).

    A forked/subagent codex transcript replays markers like `thread_rolled_back`
    that belong to the original thread, not to the fresh resumed session. Carried
    verbatim they make a reloaded session replay a spurious rollback, so we drop
    them when rewriting. The set is intentionally narrow so meaningful events
    (user_message, task_started, ...) are never stripped.
    """
    if record.get("type") != "event_msg":
        return False
    payload = record.get("payload")
    return isinstance(payload, dict) and payload.get("type") in _TRANSIENT_CODEX_EVENT_MSGS


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
    # P4-5: the new session forked FROM the original session, so its session_meta
    # lineage should point at the original id — not keep the stale `forked_from_id`
    # of whatever the original itself forked from. Capture the original id before
    # rewriting (the first session_meta's id).
    original_session_id = _first_session_meta_id(records)
    if not records or records[0].get("type") != "session_meta":
        lines.append(_json_line(_codex_session_meta(new_session_id, cwd, source_meta)))
    # H1: a forked/subagent codex transcript carries a CHAIN of stacked
    # session_meta records in its inherited prefix (e.g. e400<-ed20<-3a77). The
    # resumed file must contain exactly ONE leading session_meta — rewriting
    # every meta's id to new_session_id produced 2-3 identical-id headers, which
    # is not a valid single-session rollout. Keep only the first; drop ancestors.
    seen_session_meta = False
    for record in records:
        if _is_transient_codex_event(record):
            continue  # M1
        if record.get("type") == "session_meta":
            if seen_session_meta:
                continue  # H1: drop inlined ancestor metas
            seen_session_meta = True
        payload = record.get("payload")
        if isinstance(payload, dict):
            if record.get("type") == "session_meta":
                _apply_preserved_meta_fields(payload, source_meta)
                _mark_codex_session_visible(payload)
                # Re-point lineage at the session we forked from (or drop a stale
                # ancestor when we have nothing meaningful to point at).
                if original_session_id:
                    payload["forked_from_id"] = original_session_id
                elif "forked_from_id" in payload:
                    del payload["forked_from_id"]
            if "id" in payload:
                payload["id"] = new_session_id
            if "thread_id" in payload:
                payload["thread_id"] = new_session_id
            if "cwd" in payload:
                payload["cwd"] = str(cwd)
            # P4-2: real Codex turn_context carries `type` at the RECORD level;
            # `payload` holds model/permission_profile/sandbox_policy but no `type`
            # key. Gate on record["type"] so this actually fires on live data
            # (the old payload["type"] check was dead code).
            if record.get("type") == "turn_context":
                if model:
                    payload["model"] = model
                # F1: turn_context.permission_profile is a STRUCTURED object
                # ({type, file_system, network, ...}) and sits alongside
                # sandbox_policy/approval_policy. The hook-derived permission_mode
                # is a bare string of a different vocabulary; assigning it here
                # corrupts the object and breaks Codex load. The captured turn
                # already holds the exact permission profile, so we preserve it
                # verbatim and only re-pin a string profile if the original was
                # itself a string (legacy/simple form).
                if permission_mode and isinstance(payload.get("permission_profile"), str):
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


_CLAUDE_POINTER_KEYS = ("messageId", "leafUuid")


def _drop_dangling_trailing_pointers(
    records: list[dict[str, object]],
    uuid_map: dict[str, str],
) -> list[dict[str, object]]:
    """Drop trailing keyless records whose pointer targets a uuid absent here (M4).

    The latest turn's EOF tail can end with keyless claude records
    (file-history-snapshot, last-prompt) whose messageId/leafUuid references a
    message uuid in the NEXT turn — a forward reference outside this prefix.
    After the two-pass remap those pointers would dangle. We trim only from the
    END: stop at the first record that carries its own uuid (a real message) or
    whose pointer resolves within this file; an interior pointer is left intact.
    """
    end = len(records)
    while end > 0:
        record = records[end - 1]
        if isinstance(record.get("uuid"), str):
            break  # a real message record; stop trimming
        pointer = next(
            (record[key] for key in _CLAUDE_POINTER_KEYS if isinstance(record.get(key), str)),
            None,
        )
        if pointer is None:
            break  # not a pointer record; leave the tail intact
        if pointer in uuid_map:
            break  # resolvable within this file; keep it
        end -= 1  # dangling forward reference — drop it
    return records[:end]


def _rewrite_claude_trajectory(
    trajectory: bytes,
    new_session_id: str,
    cwd: Path,
    model: str | None,
    permission_mode: str | None,
) -> bytes:
    lines: list[bytes] = []
    last_uuid: str | None = None
    records = _jsonl_records(trajectory)
    records = _ensure_permission_mode_record(records, permission_mode, new_session_id)
    # P4-4: build the FULL old->new uuid map first. messageId (file-history-
    # snapshot) and leafUuid (last-prompt) can reference a message uuid that
    # appears later in the file, so a single forward pass would leave them
    # dangling. Two passes: map every uuid, then remap all references against it.
    uuid_map: dict[str, str] = {}
    for record in records:
        old_uuid = record.get("uuid")
        if isinstance(old_uuid, str) and old_uuid not in uuid_map:
            uuid_map[old_uuid] = str(uuid.uuid4())
    # M4: the latest turn's EOF tail can end with keyless records
    # (file-history-snapshot, last-prompt) whose messageId/leafUuid point FORWARD
    # to a message uuid that belongs to the next turn — outside this slice. After
    # the remap below those pointers would dangle. Trim them from the end.
    records = _drop_dangling_trailing_pointers(records, uuid_map)
    for record in records:
        record["sessionId"] = new_session_id
        if "cwd" in record:
            record["cwd"] = str(cwd)
        # F2: Claude records carry the model at `message.model` on assistant
        # records, not as a top-level field. Re-pin it there. (The legacy
        # top-level branch is kept for any record that does carry it.)
        if model:
            if "model" in record:
                record["model"] = model
            message = record.get("message")
            if (
                record.get("type") == "assistant"
                and isinstance(message, dict)
                and "model" in message
            ):
                message["model"] = model
        if permission_mode and record.get("type") == "permission-mode":
            record["permissionMode"] = permission_mode
        if isinstance(record.get("uuid"), str):
            record["uuid"] = uuid_map[str(record["uuid"])]
        # Remap message-uuid pointers carried by non-message records so they keep
        # referencing the same (renamed) records: file-history-snapshot.messageId
        # and last-prompt.leafUuid.
        for pointer_key in ("messageId", "leafUuid"):
            value = record.get(pointer_key)
            if isinstance(value, str) and value in uuid_map:
                record[pointer_key] = uuid_map[value]
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
    Carrying them under the new session id lets a resumed run still see the
    subagent context that the parent turn depended on. Each carried record's
    `sessionId` is rewritten to the new parent id (H3) — hardlinking verbatim
    left the content pointing at the OLD parent, so Claude couldn't associate
    the sidechain with the resumed session.
    """
    projects_root = provider_home / "projects"
    if not projects_root.exists() or not projects_root.is_dir():
        return
    for project_dir in projects_root.iterdir():
        if not project_dir.is_dir():
            continue
        src = project_dir / old_session_id / "subagents"
        if src.exists() and src.is_dir():
            _carry_subagent_tree(src, project_dir / new_session_id / "subagents", new_session_id)


def _carry_subagent_tree(src: Path, dst: Path, new_session_id: str) -> None:
    """Copy a subagent tree, rewriting each record's sessionId (H3).

    A subagent transcript is a self-contained sidechain: its internal
    uuid/parentUuid are independent and it holds no back-pointer to the parent's
    Task tool_use id (the link is parent->subagent via the agent id / filename).
    So the minimal correct carry rewrites ONLY `sessionId` to the new parent id
    and leaves every other field byte-identical. Non-jsonl entries (rare) keep
    the cheap hardlink/copy path.
    """
    if not src.exists() or not src.is_dir():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for entry in src.rglob("*"):
        target = dst / entry.relative_to(src)
        if entry.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if not entry.is_file() or target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if entry.suffix == ".jsonl":
            try:
                records = _jsonl_records(entry.read_bytes())
            except OSError:
                continue
            for record in records:
                if "sessionId" in record:
                    record["sessionId"] = new_session_id
            _write_bytes_atomic(target, b"".join(_json_line(record) for record in records))
            continue
        try:
            os.link(entry, target)
        except OSError:
            shutil.copy2(entry, target)


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
