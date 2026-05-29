"""Claude Code hook adapter.

The hook reads the JSON event payload from stdin and writes checkpoints through
the shared coordinator. It intentionally keeps Claude-specific logic at the
edge so storage remains provider-neutral.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

from checkpoint_plugin.coordinator import CheckpointCoordinator, TurnRecord
from checkpoint_plugin.types import TrajectoryReference

from ._hook_common import empty_trajectory_ref as _empty_trajectory_ref
from ._hook_common import first_string as _first_string
from ._hook_common import read_payload as _read_payload
from ._trajectory_slicer import claude_key, jsonl_ref_for_turn


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("event", choices=["session_start", "turn_end", "subagent_end"])
    args = parser.parse_args(argv)
    payload = _read_payload()
    cwd = Path(os.environ.get("CLAUDE_PROJECT_DIR") or payload.get("cwd") or Path.cwd())
    parent_session_id = os.environ.get("CLAUDE_SESSION_ID") or str(payload.get("session_id") or "claude-session")
    _seed_claude_env(parent_session_id, payload)

    if args.event == "subagent_end":
        return _on_subagent_end(payload, cwd, parent_session_id)

    coordinator = CheckpointCoordinator(session_id=parent_session_id, cwd=cwd)

    if args.event == "session_start":
        coordinator.on_session_start(
            source=_first_string(payload, "source"),
            session_env=_session_env(payload),
            source_transcript_path=_first_string(payload, "transcript_path", "transcriptPath"),
        )
        return 0

    if not _is_stop_event(payload):
        return 0

    turn_record = _turn_record(payload)
    coordinator.on_turn_end(turn_record, _trajectory_ref(payload, provider="claude") or _empty_trajectory_ref("claude"))
    return 0


def _on_subagent_end(payload: dict[str, Any], cwd: Path, parent_session_id: str) -> int:
    """Checkpoint a finished subagent as its own session (B4).

    Claude writes each subagent to a separate transcript
    `<parent>/subagents/agent-<agentId>.jsonl` with its own sessionId/promptId,
    so a subagent turn is recorded under a derived plugin session keyed by the
    agent id. The parent timeline is left untouched; lineage is kept in metadata.
    """
    agent_id = _first_string(payload, "agent_id", "agentId")
    transcript_path = _subagent_transcript_path(payload, agent_id)
    if agent_id is None and transcript_path is None:
        return 0  # Not enough to attribute this subagent; skip rather than guess.
    sub_session_id = f"{parent_session_id}--subagent-{agent_id or _stem(transcript_path)}"
    coordinator = CheckpointCoordinator(session_id=sub_session_id, cwd=cwd)
    coordinator.on_session_start(
        source="subagent",
        session_env=_session_env(payload),
        lineage={"parent_session_id": parent_session_id, "agent_id": agent_id, "agent_type": _first_string(payload, "agent_type", "agentType")},
    )
    ref = _subagent_trajectory_ref(payload, transcript_path) or _empty_trajectory_ref("claude")
    coordinator.on_turn_end(_turn_record(payload), ref)
    return 0


def _seed_claude_env(session_id: str, payload: dict[str, Any]) -> None:
    os.environ["CHECKPOINT_PROVIDER"] = "claude"
    os.environ.setdefault("CLAUDE_SESSION_ID", session_id)
    model = _first_string(payload, "model")
    if model:
        os.environ.setdefault("ANTHROPIC_MODEL", model)
    permission_mode = _first_string(payload, "permission_mode", "permissionMode")
    if permission_mode:
        os.environ.setdefault("CLAUDE_PERMISSION_MODE", permission_mode)
    effort = _effort_level(payload)
    if effort:
        os.environ.setdefault("CLAUDE_EFFORT", effort)
    agent_type = _first_string(payload, "agent_type", "agentType")
    if agent_type:
        os.environ.setdefault("CLAUDE_AGENT_TYPE", agent_type)
    agent_id = _first_string(payload, "agent_id", "agentId")
    if agent_id:
        os.environ.setdefault("CLAUDE_AGENT_ID", agent_id)


def _effort_level(payload: dict[str, Any]) -> str | None:
    effort = payload.get("effort")
    if isinstance(effort, dict):
        level = effort.get("level")
        if isinstance(level, str):
            return level
    if isinstance(effort, str):
        return effort
    return None


def _session_env(payload: dict[str, Any]) -> dict[str, str]:
    """Provider fields delivered at SessionStart but not at Stop (e.g. model)."""
    fields = {
        "model": _first_string(payload, "model"),
        "permission_mode": _first_string(payload, "permission_mode", "permissionMode"),
        "effort": _effort_level(payload),
        "agent_type": _first_string(payload, "agent_type", "agentType"),
    }
    return {key: value for key, value in fields.items() if value}


def _turn_record(payload: dict[str, Any]) -> TurnRecord:
    user_message = _first_string(payload, "prompt", "user_message", "userMessage", "input")
    assistant_text = _first_string(payload, "assistant_text", "assistantText", "response", "output")
    tool_calls = payload.get("tool_calls") or payload.get("toolCalls") or []
    if not isinstance(tool_calls, list):
        tool_calls = [tool_calls]
    return TurnRecord(
        user_message=user_message or "",
        assistant_text=assistant_text or "",
        tool_calls=tool_calls,
        metadata={"hook_payload": payload},
    )


def _is_stop_event(payload: dict[str, Any]) -> bool:
    return _first_string(payload, "hook_event_name", "hookEventName") == "Stop"


def _trajectory_ref(payload: dict[str, Any], provider: str) -> TrajectoryReference | None:
    transcript_path = _first_string(payload, "transcript_path", "transcriptPath")
    if transcript_path is None:
        return None
    turn_id = payload.get("turn_id") or payload.get("turnId")
    return jsonl_ref_for_turn(provider, Path(transcript_path), turn_id, claude_key)


def _subagent_transcript_path(payload: dict[str, Any], agent_id: str | None) -> Path | None:
    """Resolve the subagent's own transcript file.

    Prefer an explicit path in the payload. Otherwise derive it from the parent
    transcript: Claude stores subagents at `<parent-dir>/subagents/agent-<id>.jsonl`
    (or a flat sibling). We probe both rather than assume one layout.
    """
    explicit = _first_string(payload, "transcript_path", "transcriptPath")
    parent = _first_string(payload, "parent_transcript_path", "parentTranscriptPath")
    if explicit:
        path = Path(explicit)
        # If the explicit path is the parent transcript and we have an agent id,
        # prefer the nested subagent file when it exists.
        if agent_id:
            nested = path.parent / path.stem / "subagents" / f"agent-{agent_id}.jsonl"
            if nested.exists():
                return nested
        return path
    if parent and agent_id:
        base = Path(parent)
        candidates = [
            base.parent / base.stem / "subagents" / f"agent-{agent_id}.jsonl",
            base.parent / "subagents" / f"agent-{agent_id}.jsonl",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]
    return None


def _subagent_trajectory_ref(payload: dict[str, Any], transcript_path: Path | None) -> TrajectoryReference | None:
    if transcript_path is None:
        return None
    turn_id = payload.get("turn_id") or payload.get("turnId")
    return jsonl_ref_for_turn("claude", transcript_path, turn_id, claude_key)


def _stem(path: Path | None) -> str:
    return path.stem if path is not None else "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
