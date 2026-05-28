"""JSON-serializable checkpoint data contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class EnvironmentState:
    provider: str
    model: str | None = None
    permission_mode: str | None = None
    memory_files: dict[str, str] = field(default_factory=dict)
    mcp_config: str | None = None
    skills: dict[str, str] = field(default_factory=dict)
    settings: dict[str, str] = field(default_factory=dict)
    project_context: dict[str, str] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "EnvironmentState":
        return cls(
            provider=str(data.get("provider", "generic")),
            model=data.get("model"),
            permission_mode=data.get("permission_mode"),
            memory_files=dict(data.get("memory_files") or {}),
            mcp_config=data.get("mcp_config"),
            skills=dict(data.get("skills") or {}),
            settings=dict(data.get("settings") or {}),
            project_context=dict(data.get("project_context") or {}),
            extra=dict(data.get("extra") or {}),
        )


@dataclass(frozen=True)
class FilesystemSnapshot:
    cwd: str
    files: dict[str, str] = field(default_factory=dict)
    git: dict[str, str] | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "FilesystemSnapshot":
        git = data.get("git")
        return cls(
            cwd=str(data["cwd"]),
            files=dict(data.get("files") or {}),
            git=dict(git) if isinstance(git, dict) else None,
        )


@dataclass(frozen=True)
class TrajectoryReference:
    provider: str
    transcript_path: str
    start_offset: int
    end_offset: int
    record_count: int

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "TrajectoryReference":
        return cls(
            provider=str(data.get("provider", "generic")),
            transcript_path=str(data.get("transcript_path", "")),
            start_offset=int(data.get("start_offset", 0)),
            end_offset=int(data.get("end_offset", 0)),
            record_count=int(data.get("record_count", 0)),
        )


@dataclass(frozen=True)
class CheckpointManifest:
    turn_id: int
    session_id: str
    created_ts: str
    env_ref: str
    fs_ref: str
    trajectory_offset: int = 0
    trajectory_end_offset: int | None = None
    trajectory_ref: TrajectoryReference | None = None
    user_message_preview: str = ""
    parent_turn_id: int | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "CheckpointManifest":
        trajectory_ref = data.get("trajectory_ref")
        return cls(
            turn_id=int(data["turn_id"]),
            session_id=str(data["session_id"]),
            created_ts=str(data["created_ts"]),
            env_ref=str(data["env_ref"]),
            fs_ref=str(data["fs_ref"]),
            trajectory_offset=int(data.get("trajectory_offset", 0)),
            trajectory_end_offset=(
                int(data["trajectory_end_offset"]) if data.get("trajectory_end_offset") is not None else None
            ),
            trajectory_ref=(
                TrajectoryReference.from_json(trajectory_ref)
                if isinstance(trajectory_ref, dict)
                else None
            ),
            user_message_preview=str(data.get("user_message_preview", "")),
            parent_turn_id=data.get("parent_turn_id"),
        )


@dataclass(frozen=True)
class RestoreReport:
    changed: list[str] = field(default_factory=list)
    backed_up: list[str] = field(default_factory=list)
    backup_dir: str | None = None


@dataclass(frozen=True)
class ResumePlan:
    session_id: str
    turn_id: int
    target_manifest: CheckpointManifest
    current_env: EnvironmentState
    target_env: EnvironmentState
    current_fs: FilesystemSnapshot
    target_fs: FilesystemSnapshot
    env_diff_text: str
    fs_diff_text: str

    def render(self) -> str:
        parts = [
            f"Resume: session {self.session_id}, turn {self.turn_id}",
            "",
            self.env_diff_text,
            "",
            self.fs_diff_text,
        ]
        return "\n".join(part for part in parts if part.strip())


@dataclass(frozen=True)
class ResumeReport:
    new_session_id: str
    backup_dir: str
    env: RestoreReport
    fs: RestoreReport

    @property
    def changed_files(self) -> list[str]:
        return [*self.env.changed, *self.fs.changed]
