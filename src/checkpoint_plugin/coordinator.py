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

    def on_session_start(self) -> None:
        provider = detect_provider(self.cwd)
        metadata = {
            "session_id": self.session_id,
            "provider": provider.name,
            "cwd": str(self.cwd),
            "start_ts": _now(),
            "session_title": _session_title(provider.name, provider.home, self.session_id, None),
        }
        self.store._atomic_write(
            self.session_dir / "metadata.json",
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
