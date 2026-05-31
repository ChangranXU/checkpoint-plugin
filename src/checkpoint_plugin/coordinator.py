"""Turn-boundary checkpoint lifecycle."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .env.collector import collect_environment, environment_to_blob
from .env.providers import detect_provider
from .fs.ignore import IgnoreMatcher
from .fs.snapshot import filesystem_to_blob, snapshot_cwd
from .integrations._trajectory_slicer import claude_key, codex_key
from .paths import ensure_home, load_config, session_dir
from .store import CheckpointStore
from .types import CheckpointManifest, TrajectoryReference


@dataclass(frozen=True)
class TurnRecord:
    user_message: str = ""
    assistant_text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class CheckpointCoordinator:
    def __init__(self, session_id: str | None = None, cwd: Path | None = None, plugin_home: Path | None = None) -> None:
        self.home = ensure_home(plugin_home)
        self.session_id = session_id or str(uuid.uuid4())
        self.cwd = Path(cwd or Path.cwd()).expanduser().resolve()
        self.session_dir = session_dir(self.session_id, self.home)
        self.store = CheckpointStore(self.session_dir)

    def on_session_start(
        self,
        source: str | None = None,
        session_env: dict[str, str] | None = None,
        lineage: dict[str, Any] | None = None,
        source_transcript_path: str | None = None,
    ) -> None:
        # F13: the last turn of a prior run never gets its end_offset back-filled
        # (`_close_previous_trajectory_ref` only extends a turn against the NEXT turn's
        # start, and the last turn has no successor), so it stops short of EOF when the
        # provider flushed a trailing record after the Stop hook read the file. There
        # is no finalize/SessionEnd hook to fix this at write time, so we re-anchor it
        # lazily here — the next session_start is the first moment the transcript is
        # guaranteed fully flushed. Resume already recovers this tail at read time;
        # this makes the STORED manifest faithful for non-resume consumers too.
        self._reanchor_last_turn_to_eof()
        provider = detect_provider(self.cwd)
        metadata_path = self.session_dir / "metadata.json"
        existing = _read_metadata(metadata_path)
        if existing.get("resumed_from_session_id"):
            return
        metadata = {
            "session_id": self.session_id,
            "provider": provider.name,
            "cwd": str(self.cwd),
            "start_ts": _now(),
            "session_title": _session_title(provider.name, provider.home, self.session_id, None),
        }
        if source:
            metadata["source"] = source
        # A native fork (resume/compact) starts a fresh plugin session; record the
        # provider transcript it forked from so lineage can be traced later (B5).
        # Capture the byte offset + record count at fork time as the anchor point
        # in the parent's history where this session branched (F5).
        if source in {"resume", "compact"} and source_transcript_path:
            # P6-15: for claude the SessionStart `transcript_path` is the session's
            # OWN file (Claude resumes in place under the same id), so recording it
            # verbatim makes forked_from_transcript a self-reference. The true
            # ancestor is named by `forkedFrom.sessionId` inside that transcript;
            # resolve it, and drop the field entirely when it would point at self.
            # F6/F7 (codex): the codex resume `transcript_path` is ALSO the session's
            # own rollout (its first session_meta carries `forked_from_id` = the true
            # ancestor). Recording the path verbatim self-references and anchoring on
            # the self-file overshoots the branch point (the self-file already holds
            # the resume's own new turns by hook time). Resolve the ancestor by
            # `forked_from_id` and anchor on the PARENT (its EOF is the branch point),
            # keeping source="resume" (do NOT relabel to "fork").
            if provider.name == "codex":
                resumed = _codex_resume_lineage(provider.home, source_transcript_path, self.session_id)
                if resumed is not None:
                    parent_transcript, anchor = resumed
                    metadata["forked_from_transcript"] = parent_transcript
                    if anchor is not None:
                        metadata["forked_at_offset"] = anchor[0]
                        metadata["forked_at_record_count"] = anchor[1]
            else:
                ancestor = _resolve_fork_ancestor_transcript(
                    provider.name, source_transcript_path, self.session_id
                )
                if ancestor is not None:
                    metadata["forked_from_transcript"] = ancestor
                    anchor = _fork_anchor(ancestor)
                    if anchor is not None:
                        metadata["forked_at_offset"] = anchor[0]
                        metadata["forked_at_record_count"] = anchor[1]
        # M2: Codex forks ("fork chat") arrive with source="startup" (or a
        # structured subagent source), so the resume/compact guard never fires.
        # Detect the fork structurally from the new rollout's first session_meta
        # `forked_from_id` and record the lineage + anchor anyway.
        if provider.name == "codex" and "forked_from_transcript" not in metadata:
            fork = _codex_fork_lineage(provider.home, source_transcript_path)
            if fork is not None:
                parent_transcript, anchor = fork
                metadata["forked_from_transcript"] = parent_transcript
                # P6-16: a structurally-detected codex fork arrives as
                # source="startup"; normalize it to "fork" so readers (and the
                # picker) don't mistake a branch for a cold start.
                metadata["source"] = "fork"
                if anchor is not None:
                    metadata["forked_at_offset"] = anchor[0]
                    metadata["forked_at_record_count"] = anchor[1]
        clean_env = {key: value for key, value in (session_env or {}).items() if value}
        if clean_env:
            metadata["session_env"] = clean_env
        clean_lineage = {key: value for key, value in (lineage or {}).items() if value}
        if clean_lineage:
            metadata["lineage"] = clean_lineage
        self.store._atomic_write(
            metadata_path,
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )

    def on_turn_end(
        self,
        turn_record: TurnRecord,
        trajectory_ref: TrajectoryReference | None = None,
    ) -> CheckpointManifest:
        with self.store.session_lock():
            latest = self.store.latest_manifest()
            turn_id = latest.turn_id + 1 if latest else 0
            provider = detect_provider(self.cwd)
            if trajectory_ref is None:
                trajectory_ref = self._write_manual_trajectory_ref(provider.name, turn_id, turn_record)
            self._close_previous_trajectory_ref(latest, trajectory_ref)
            self._refresh_metadata_title(provider.name, provider.home, trajectory_ref)
            env_state = collect_environment(self.cwd, provider, self.store)
            env_ref = environment_to_blob(env_state, self.store)
            config = load_config(self.home)
            ignore = IgnoreMatcher(self.cwd, config.get("exclude_patterns") or [])
            fs_snapshot = snapshot_cwd(self.cwd, self.store, ignore)
            fs_ref = filesystem_to_blob(fs_snapshot, self.store)
            manifest = CheckpointManifest(
                turn_id=turn_id,
                session_id=self.session_id,
                created_ts=_now(),
                env_ref=env_ref,
                fs_ref=fs_ref,
                trajectory_offset=trajectory_ref.start_offset,
                trajectory_end_offset=trajectory_ref.end_offset,
                trajectory_ref=trajectory_ref,
                user_message_preview=_user_message_preview(turn_record, trajectory_ref),
                parent_turn_id=latest.turn_id if latest else None,
            )
            self.store.write_manifest(manifest)
            return manifest

    def list_checkpoints(self) -> list[CheckpointManifest]:
        return self.store.list_manifests()

    def get_checkpoint(self, turn_id: int) -> CheckpointManifest:
        return self.store.read_manifest(turn_id)

    def _write_manual_trajectory_ref(
        self,
        provider: str,
        turn_id: int,
        turn_record: TurnRecord,
    ) -> TrajectoryReference:
        start_offset, end_offset = self.store.append_trajectory(
            {
                "type": "turn",
                "turn_id": turn_id,
                "created_ts": _now(),
                **turn_record.to_json(),
            }
        )
        return TrajectoryReference(
            provider=provider,
            transcript_path=str(self.store.trajectory_path),
            start_offset=start_offset,
            end_offset=end_offset,
            record_count=1,
        )

    def _close_previous_trajectory_ref(
        self,
        latest: CheckpointManifest | None,
        next_ref: TrajectoryReference,
    ) -> None:
        if latest is None or latest.trajectory_ref is None:
            return
        previous_ref = latest.trajectory_ref
        if not previous_ref.transcript_path or previous_ref.transcript_path != next_ref.transcript_path:
            return
        if previous_ref.end_offset >= next_ref.start_offset:
            return
        refreshed_ref = _ref_with_end_offset(previous_ref, next_ref.start_offset)
        self.store.write_manifest(
            replace(
                latest,
                trajectory_end_offset=refreshed_ref.end_offset,
                trajectory_ref=refreshed_ref,
            )
        )

    def _reanchor_last_turn_to_eof(self) -> None:
        """Extend the last stored turn's end_offset to the transcript EOF (F13).

        Called at session_start, when the prior run's transcript is fully flushed.
        Only extends when the bytes between the stored end_offset and EOF are a
        same-turn complete tail (newline-terminated, no record bearing a DIFFERENT
        per-turn key) — the identical guard resume uses in `_recover_trailing_tail`,
        so we never absorb a later turn's records. A no-op when there are no
        manifests, the transcript is gone, or the slice already reaches EOF.
        """
        try:
            with self.store.session_lock():
                latest = self.store.latest_manifest()
                if latest is None or latest.trajectory_ref is None:
                    return
                ref = latest.trajectory_ref
                tail = _trailing_same_turn_tail(ref)
                if not tail:
                    return
                new_end = ref.end_offset + len(tail)
                refreshed = _ref_with_end_offset(ref, new_end)
                self.store.write_manifest(
                    replace(
                        latest,
                        trajectory_end_offset=refreshed.end_offset,
                        trajectory_ref=refreshed,
                    )
                )
        except OSError:
            return

    def _refresh_metadata_title(
        self,
        provider: str,
        provider_home: Path,
        trajectory_ref: TrajectoryReference,
    ) -> None:
        title = _session_title(provider, provider_home, self.session_id, trajectory_ref)
        metadata_path = self.session_dir / "metadata.json"
        metadata: dict[str, Any]
        if metadata_path.exists():
            try:
                raw_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                raw_metadata = {}
            metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
        else:
            metadata = {}
        metadata.setdefault("session_id", self.session_id)
        metadata.setdefault("provider", provider)
        metadata.setdefault("cwd", str(self.cwd))
        metadata.setdefault("start_ts", _now())
        metadata["session_title"] = title
        self.store._atomic_write(
            metadata_path,
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _resolve_fork_ancestor_transcript(
    provider_name: str,
    source_transcript_path: str,
    own_session_id: str,
) -> str | None:
    """The ANCESTOR transcript a fork branched from, never a self-reference (P6-15).

    For codex the SessionStart `transcript_path` is genuinely the parent rollout, so
    we return it as-is. For claude the path is the session's OWN file (Claude resumes
    in place under the same id); the real ancestor is named by `forkedFrom.sessionId`
    inside that transcript. We resolve a sibling `<ancestor>.jsonl` and return it; if
    the path is self-referential and no distinct ancestor resolves, return None so the
    misleading self-pointer is dropped rather than recorded.
    """
    path = Path(source_transcript_path).expanduser()
    if provider_name != "claude":
        return source_transcript_path
    if path.stem != own_session_id:
        return source_transcript_path  # already names a different (ancestor) file
    ancestor_id = _claude_forked_from_session_id(path)
    if not ancestor_id or ancestor_id == own_session_id:
        return None
    sibling = path.with_name(f"{ancestor_id}.jsonl")
    return str(sibling) if sibling.exists() else None


def _claude_forked_from_session_id(transcript_path: Path) -> str | None:
    """The `forkedFrom.sessionId` recorded in a claude fork transcript (P6-15)."""
    try:
        with transcript_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict):
                    forked_from = record.get("forkedFrom")
                    if isinstance(forked_from, dict):
                        sid = forked_from.get("sessionId")
                        if isinstance(sid, str) and sid:
                            return sid
    except OSError:
        return None
    return None


def _fork_anchor(transcript_path: str) -> tuple[int, int] | None:
    """(byte offset, record count) of a forked-from transcript at fork time (F5).

    The byte size is the point in the parent's history this session branched from;
    pairing it with a record count lets lineage be traced to an exact turn later.
    Returns None when the transcript is absent/unreadable.
    """
    path = Path(transcript_path).expanduser()
    try:
        data = path.read_bytes()
    except OSError:
        return None
    record_count = sum(1 for line in data.splitlines() if line.strip())
    return len(data), record_count


def _codex_fork_lineage(
    codex_home: Path,
    own_transcript_path: str | None,
) -> tuple[str, tuple[int, int] | None] | None:
    """Lineage for a Codex fork detected via its own session_meta (M2).

    Codex "fork chat" sessions arrive at SessionStart with source="startup", so
    the resume/compact path misses them. The fork link lives in the NEW rollout's
    first `session_meta.forked_from_id`. We read that, discover the parent rollout
    by the `rollout-<ts>-<id>.jsonl` filename convention, and return
    `(parent_transcript_path, anchor)` where anchor is `_fork_anchor(parent)`.

    Returns None when this is not a fork (no forked_from_id) or the rollout can't
    be read. When the parent file can't be found we still return the bare
    forked_from_id as the transcript reference (no anchor) so lineage isn't lost.
    """
    if not own_transcript_path:
        return None
    path = Path(own_transcript_path).expanduser()
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
    forked_from_id = payload.get("forked_from_id") if isinstance(payload, dict) else None
    if not isinstance(forked_from_id, str) or not forked_from_id:
        return None
    matches = sorted(codex_home.glob(f"sessions/**/rollout-*-{forked_from_id}.jsonl"))
    # P6-10: anchor at the fork's OWN inlined prefix length, not the parent's live
    # EOF. The parent rollout keeps growing after the branch, so its current size
    # over-counts the branch point; the fork inlines the parent history up to the
    # branch, so the inlined prefix is the faithful, drift-free anchor.
    anchor = _codex_inlined_prefix_anchor(path)
    if anchor is None and matches:
        anchor = _fork_anchor(str(matches[0]))  # fallback: parent file (legacy)
    if not matches:
        return forked_from_id, anchor
    return str(matches[0]), anchor


def _codex_inlined_prefix_anchor(fork_path: Path) -> tuple[int, int] | None:
    """(byte offset, record count) of the inlined ancestor prefix in a codex fork (P6-10).

    A codex fork inlines the parent's history at the head of its OWN rollout. The
    branch point is the end of that inlined prefix, measured in the fork's own file,
    which (unlike the parent's live EOF) never drifts as the parent keeps growing.
    At SessionStart the fork file holds only the inlined history (plus its own
    session_meta), so the full current length IS the inlined-prefix anchor.
    """
    try:
        data = fork_path.read_bytes()
    except OSError:
        return None
    count = sum(1 for line in data.splitlines() if line.strip())
    if count == 0:
        return None
    return len(data), count


def _codex_resume_lineage(
    codex_home: Path,
    own_transcript_path: str | None,
    own_session_id: str,
) -> tuple[str, tuple[int, int] | None] | None:
    """Lineage for a Codex RESUME, anchored on the true parent (F6/F7).

    A codex resume's SessionStart `transcript_path` is the resume session's OWN
    rollout, whose first `session_meta.forked_from_id` names the ancestor it
    continues. Recording the path verbatim self-references (F6); anchoring on that
    self-file overshoots the branch point because, by the time the hook fires, the
    rollout already carries the resume's own new turns (F7 — verified on 8c17:
    self-EOF rec47/byte139338 vs true boundary rec33/byte118563).

    We resolve the ancestor rollout via `forked_from_id` and anchor on the PARENT:
    a resume continues from the parent's end, so the parent's EOF (size, record
    count) is the branch point — and it lives in the parent's own coordinate space,
    not the inflated self-file. Falls back to treating a distinct, non-self
    `transcript_path` as the ancestor directly (e.g. a bare prior rollout with no
    inlined meta), so a genuinely-distinct parent path is still recorded.
    """
    if not own_transcript_path:
        return None
    forked_from_id = _codex_forked_from_id(Path(own_transcript_path).expanduser())
    if forked_from_id:
        matches = sorted(codex_home.glob(f"sessions/**/rollout-*-{forked_from_id}.jsonl"))
        if matches:
            parent = str(matches[0])
            return parent, _fork_anchor(parent)
        return forked_from_id, None  # parent file gone; keep the bare id as the ref
    if _path_is_distinct_ancestor(own_transcript_path, own_session_id):
        return own_transcript_path, _fork_anchor(own_transcript_path)
    return None


def _codex_forked_from_id(transcript_path: Path) -> str | None:
    """The `forked_from_id` in a codex rollout's first session_meta, if any."""
    try:
        with transcript_path.open("rb") as handle:
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
    value = payload.get("forked_from_id") if isinstance(payload, dict) else None
    return value if isinstance(value, str) and value else None


def _path_is_distinct_ancestor(transcript_path: str, own_session_id: str) -> bool:
    """True when `transcript_path` names a file other than this session's own rollout."""
    stem = Path(transcript_path).stem
    # codex rollout filenames are `rollout-<ts>-<session_id>`; a self-reference ends
    # with the own id. Anything else is a distinct ancestor path.
    return own_session_id not in stem


def _ref_with_end_offset(ref: TrajectoryReference, end_offset: int) -> TrajectoryReference:
    path = Path(ref.transcript_path).expanduser()
    try:
        data = path.read_bytes()
    except OSError:
        record_count = ref.record_count
    else:
        record_count = _count_jsonl_records(data[ref.start_offset : end_offset])
    return TrajectoryReference(
        provider=ref.provider,
        transcript_path=ref.transcript_path,
        start_offset=ref.start_offset,
        end_offset=end_offset,
        record_count=record_count,
    )


def _count_jsonl_records(data: bytes) -> int:
    return sum(1 for line in data.splitlines() if line.strip())


def _trailing_same_turn_tail(ref: TrajectoryReference) -> bytes:
    """Bytes flushed after `ref.end_offset` that still belong to `ref`'s turn (F13).

    Mirrors resume's `_recover_trailing_tail` guard so the stored manifest and a
    resume agree: the candidate tail must end with a newline (no truncated JSON) and
    contain no record carrying a per-turn key distinct from the turn's anchor key (no
    new turn started). Returns b"" when there is nothing safe to absorb.
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
    anchor = _anchor_key_for_ref(ref, extractor)
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


def _anchor_key_for_ref(ref: TrajectoryReference, extractor: Any) -> Any:
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
        if isinstance(record, dict):
            key = extractor(record)
            if key is not None:
                return key
    return None


def _user_message_preview(turn_record: TurnRecord, trajectory_ref: TrajectoryReference) -> str:
    explicit = turn_record.user_message.strip()
    if explicit:
        return explicit[:200]
    inferred = _user_message_from_trajectory(trajectory_ref)
    return inferred[:200] if inferred else ""


def _user_message_from_trajectory(ref: TrajectoryReference) -> str:
    if not ref.transcript_path or ref.end_offset <= ref.start_offset:
        return ""
    path = Path(ref.transcript_path).expanduser()
    try:
        data = path.read_bytes()[ref.start_offset : ref.end_offset]
    except OSError:
        return ""

    fallback = ""
    for line in data.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        message = _explicit_user_message(record)
        if message:
            return _normalize_preview(message)
        if not fallback:
            fallback = _role_user_message(record)
    return _normalize_preview(fallback)


def _explicit_user_message(record: Any) -> str:
    if not isinstance(record, dict):
        return ""
    payload = record.get("payload")
    if isinstance(payload, dict) and payload.get("type") == "user_message":
        return _string_or_content_text(payload.get("message"))
    if record.get("type") == "user":
        return _string_or_content_text(record.get("message"))
    return ""


def _role_user_message(record: Any) -> str:
    if not isinstance(record, dict):
        return ""
    payload = record.get("payload")
    if isinstance(payload, dict):
        return _role_user_message(payload)
    message = record.get("message")
    if isinstance(message, dict) and message.get("role") == "user":
        return _string_or_content_text(message)
    if record.get("role") == "user":
        return _string_or_content_text(record)
    return ""


def _string_or_content_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return ""
    content = value.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts)
    return ""


def _normalize_preview(message: str) -> str:
    return " ".join(message.split())


def _session_title(
    provider: str,
    provider_home: Path,
    session_id: str,
    trajectory_ref: TrajectoryReference | None,
) -> str | None:
    if provider == "codex":
        return _codex_session_title(provider_home, session_id)
    if provider == "claude" and trajectory_ref is not None:
        return _claude_session_title(trajectory_ref)
    return None


def _codex_session_title(codex_home: Path, session_id: str) -> str | None:
    index_path = codex_home / "session_index.jsonl"
    if not index_path.exists():
        return None
    try:
        lines = index_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        if session_id not in line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("id") == session_id:
            title = record.get("thread_name")
            return title if isinstance(title, str) and title else None
    return None


def _claude_session_title(ref: TrajectoryReference) -> str | None:
    if not ref.transcript_path:
        return None
    path = Path(ref.transcript_path).expanduser()
    try:
        data = path.read_bytes()[ref.start_offset : ref.end_offset]
    except OSError:
        return None
    for line in data.splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        title = record.get("slug") if isinstance(record, dict) else None
        if isinstance(title, str) and title:
            return title
    return None
