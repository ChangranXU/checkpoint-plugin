"""Claude Code hook adapter.

The hook reads the JSON event payload from stdin and writes checkpoints through
the shared coordinator. It intentionally keeps Claude-specific logic at the
edge so storage remains provider-neutral.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from checkpoint_plugin.coordinator import CheckpointCoordinator, TurnRecord
from checkpoint_plugin.types import TrajectoryReference

from ._trajectory_slicer import claude_key, jsonl_ref_for_turn


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("event", choices=["session_start", "turn_end"])
    args = parser.parse_args(argv)
    payload = _read_payload()
    cwd = Path(os.environ.get("CLAUDE_PROJECT_DIR") or payload.get("cwd") or Path.cwd())
    session_id = os.environ.get("CLAUDE_SESSION_ID") or str(payload.get("session_id") or "claude-session")
    _seed_claude_env(session_id, payload)
    coordinator = CheckpointCoordinator(session_id=session_id, cwd=cwd)

    if args.event == "session_start":
        coordinator.on_session_start(source=_first_string(payload, "source"))
        return 0

    if not _is_stop_event(payload):
        return 0

    turn_record = _turn_record(payload)
    coordinator.on_turn_end(turn_record, _trajectory_ref(payload, provider="claude") or _empty_trajectory_ref("claude"))
    return 0


def _read_payload() -> dict[str, Any]:
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}
    return value if isinstance(value, dict) else {"payload": value}


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


def _empty_trajectory_ref(provider: str) -> TrajectoryReference:
    return TrajectoryReference(
        provider=provider,
        transcript_path="",
        start_offset=0,
        end_offset=0,
        record_count=0,
    )


def _first_string(payload: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return None


if __name__ == "__main__":
    raise SystemExit(main())
