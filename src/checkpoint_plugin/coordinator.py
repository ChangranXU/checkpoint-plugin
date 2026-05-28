"""Turn-boundary checkpoint lifecycle."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .env.collector import collect_environment, environment_to_blob
from .env.providers import detect_provider
from .fs.ignore import IgnoreMatcher
from .fs.snapshot import filesystem_to_blob, snapshot_cwd
from .paths import ensure_home, load_config, session_dir
from .store import CheckpointStore
from .types import CheckpointManifest


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
            "model": None,
        }
        self.store._atomic_write(
            self.session_dir / "metadata.json",
            json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        )

    def on_turn_end(self, turn_record: TurnRecord) -> CheckpointManifest:
        latest = self.store.latest_manifest()
        turn_id = latest.turn_id + 1 if latest else 0
        trajectory_offset = self.store.append_trajectory(
            {
                "type": "turn",
                "turn_id": turn_id,
                "created_ts": _now(),
                **turn_record.to_json(),
            }
        )
        provider = detect_provider(self.cwd)
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
            trajectory_offset=trajectory_offset,
            user_message_preview=turn_record.user_message[:200],
            parent_turn_id=latest.turn_id if latest else None,
        )
        self.store.write_manifest(manifest)
        return manifest

    def list_checkpoints(self) -> list[CheckpointManifest]:
        return self.store.list_manifests()

    def get_checkpoint(self, turn_id: int) -> CheckpointManifest:
        return self.store.read_manifest(turn_id)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
