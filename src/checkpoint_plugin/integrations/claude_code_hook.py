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
    return _jsonl_ref_for_turn(provider, Path(transcript_path), turn_id)


def _empty_trajectory_ref(provider: str) -> TrajectoryReference:
    return TrajectoryReference(
        provider=provider,
        transcript_path="",
        start_offset=0,
        end_offset=0,
        record_count=0,
    )


def _jsonl_ref_for_turn(provider: str, path: Path, turn_id: Any) -> TrajectoryReference | None:
    try:
        data = path.expanduser().read_bytes()
    except OSError:
        return None

    lines: list[tuple[int, int, Any]] = []
    offset = 0
    for line in data.splitlines(keepends=True):
        end = offset + len(line)
        if line.strip():
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                record = None
            lines.append((offset, end, record))
        offset = end

    matches = [
        (start, end)
        for start, end, record in lines
        if turn_id is None or _turn_ids_match(_record_turn_id(record), turn_id)
    ]
    if not matches and turn_id is not None:
        return _jsonl_ref_for_turn(provider, path, None)
    if not matches:
        return None
    start_offset = matches[0][0]
    next_turn_start = _next_turn_start(lines, start_offset, turn_id)
    end_offset = next_turn_start if next_turn_start is not None else len(data)
    return TrajectoryReference(
        provider=provider,
        transcript_path=str(path.expanduser().resolve()),
        start_offset=start_offset,
        end_offset=end_offset,
        record_count=_count_jsonl_records(data[start_offset:end_offset]),
    )


def _next_turn_start(lines: list[tuple[int, int, Any]], start_offset: int, turn_id: Any) -> int | None:
    if turn_id is None:
        return None
    for line_start, _, record in lines:
        record_turn_id = _record_turn_id(record)
        if line_start > start_offset and record_turn_id is not None and not _turn_ids_match(record_turn_id, turn_id):
            return line_start
    return None


def _count_jsonl_records(data: bytes) -> int:
    return sum(1 for line in data.splitlines() if line.strip())


def _record_turn_id(record: Any) -> Any:
    if not isinstance(record, dict):
        return None
    if "turn_id" in record:
        return record["turn_id"]
    if "turnId" in record:
        return record["turnId"]
    payload = record.get("payload")
    if isinstance(payload, dict):
        return payload.get("turn_id") or payload.get("turnId")
    return None


def _turn_ids_match(left: Any, right: Any) -> bool:
    return left == right or (left is not None and right is not None and str(left) == str(right))


def _first_string(payload: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return None


if __name__ == "__main__":
    raise SystemExit(main())
