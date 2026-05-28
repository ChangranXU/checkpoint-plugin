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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("event", choices=["session_start", "turn_end"])
    args = parser.parse_args(argv)
    payload = _read_payload()
    cwd = Path(os.environ.get("CLAUDE_PROJECT_DIR") or payload.get("cwd") or Path.cwd())
    session_id = os.environ.get("CLAUDE_SESSION_ID") or str(payload.get("session_id") or "claude-session")
    coordinator = CheckpointCoordinator(session_id=session_id, cwd=cwd)

    if args.event == "session_start":
        coordinator.on_session_start()
        return 0

    coordinator.on_turn_end(_turn_record(payload))
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


def _first_string(payload: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return None


if __name__ == "__main__":
    raise SystemExit(main())
