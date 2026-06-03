"""OpenCode hook adapter.

The hook reads the JSON event payload from stdin and writes checkpoints through
the shared coordinator. It maps OpenCode hook fields onto the provider-neutral
checkpoint lifecycle.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from checkpoint_plugin.coordinator import CheckpointCoordinator, TurnRecord
from checkpoint_plugin.types import TrajectoryReference

from ._hook_common import empty_trajectory_ref as _empty_trajectory_ref
from ._hook_common import first_string as _first_string
from ._hook_common import parent_session_env as _parent_session_env
from ._hook_common import read_payload as _read_payload
from ._trajectory_slicer import codex_key, jsonl_after_leading_metas, jsonl_ref_for_turn

# OpenCode uses similar architecture to Codex: SubagentStop fires before the turn-closing
# record is flushed. Apply the same settle timeout optimization with read-time tail recovery.
_SETTLE_TIMEOUT_ENV = "CHECKPOINT_SIDECHAIN_SETTLE_TIMEOUT"
_SETTLE_POLL_ENV = "CHECKPOINT_SIDECHAIN_SETTLE_POLL"


def _settle_timeout_s() -> float:
    try:
        return float(os.environ.get(_SETTLE_TIMEOUT_ENV, "1.0"))
    except ValueError:
        return 1.0


def _settle_poll_s() -> float:
    try:
        return float(os.environ.get(_SETTLE_POLL_ENV, "0.1"))
    except ValueError:
        return 0.1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("event", nargs="?", choices=["session_start", "turn_end", "subagent_end"])
    args = parser.parse_args(argv)
    payload = _read_payload()
    event = args.event or _event_from_payload(payload)
    cwd = Path(_first_string(payload, "cwd", "directory") or Path.cwd())
    session_id = (
        os.environ.get("OPENCODE_SESSION_ID")
        or _first_string(payload, "session_id", "sessionID", "sessionId")
        or "opencode-session"
    )
    _seed_opencode_env(session_id, payload)

    if event == "subagent_end" or _is_subagent_stop_event(payload):
        _on_subagent_end(payload, cwd, session_id)
        _write_ok()
        return 0

    coordinator = CheckpointCoordinator(session_id=session_id, cwd=cwd)

    if event == "session_start":
        coordinator.on_session_start(
            source=_first_string(payload, "source"),
            session_env=_session_env(payload),
            source_transcript_path=_first_string(payload, "transcript_path", "transcriptPath"),
        )
        _write_ok()
        return 0

    if not _is_stop_event(payload):
        _write_ok()
        return 0

    turn_record = _turn_record(payload)
    coordinator.on_turn_end(turn_record, _trajectory_ref(payload, provider="opencode") or _empty_trajectory_ref("opencode"))
    _write_ok()
    return 0


def _on_subagent_end(payload: dict[str, Any], cwd: Path, parent_session_id: str) -> None:
    """Checkpoint a finished OpenCode subagent as its own session.

    OpenCode subagent hooks carry the parent session id; we derive a distinct
    plugin session keyed by agent id (falling back to the transcript stem) so the
    subagent gets its own checkpoint timeline without disturbing the parent.
    """
    agent_id = _first_string(payload, "agent_id", "agentId")
    # OpenCode SubagentStop carries the subagent's own rollout in `agent_transcript_path`;
    # the common `transcript_path` is the PARENT rollout. Slice the subagent file.
    transcript_path = _first_string(payload, "agent_transcript_path", "agentTranscriptPath") or _first_string(
        payload, "transcript_path", "transcriptPath"
    )
    if agent_id is None and transcript_path is None:
        return
    suffix = agent_id or (Path(transcript_path).stem if transcript_path else "unknown")
    sub_session_id = f"{parent_session_id}--subagent-{suffix}"
    coordinator = CheckpointCoordinator(session_id=sub_session_id, cwd=cwd)
    # Inherit the parent's pinned session_env (model/effort) for fields the
    # subagent Stop payload may omit.
    sub_env = {**_parent_session_env(parent_session_id), **_session_env(payload)}
    coordinator.on_session_start(
        source="subagent",
        session_env=sub_env,
        lineage={
            "parent_session_id": parent_session_id,
            "agent_id": agent_id,
            "agent_type": _first_string(payload, "agent_type", "agentType"),
        },
    )
    if transcript_path is not None:
        _settle_subagent_rollout(Path(transcript_path))
    ref = _subagent_trajectory_ref(payload, transcript_path) or _empty_trajectory_ref("opencode")
    coordinator.on_turn_end(_turn_record(payload), ref)


def _settle_subagent_rollout(transcript_path: Path) -> None:
    """Block (bounded) for opencode's turn-closing `task_complete` to flush.

    LATENCY OPTIMIZATION ONLY — not a correctness mechanism. The subagent's final
    event lands moments after SubagentStop, and the hook fires before opencode
    even enqueues it, so this poll cannot guarantee the record is present. Read-time
    tail recovery (`recover_trailing_tail` on every read path) is what guarantees
    completeness; this merely front-loads it so a `show`/`list` immediately after
    capture is already at EOF without a recovery write. Poll until the last non-blank
    record is a `task_complete`, or the (short) timeout elapses.
    """
    if _settle_timeout_s() <= 0:
        return
    deadline = time.monotonic() + _settle_timeout_s()
    poll = _settle_poll_s()
    while True:
        if _subagent_tail_is_complete(transcript_path):
            return
        if not transcript_path.exists():
            return
        if time.monotonic() >= deadline:
            return
        time.sleep(poll)


def _subagent_tail_is_complete(transcript_path: Path) -> bool:
    """True when the rollout's last record is a `task_complete` event (turn closed)."""
    try:
        data = transcript_path.read_bytes()
    except OSError:
        return False
    if not data.endswith(b"\n"):
        return False
    for line in reversed(data.splitlines()):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return False
        payload = record.get("payload") if isinstance(record, dict) else None
        return isinstance(payload, dict) and payload.get("type") == "task_complete"
    return False


def _event_from_payload(payload: dict[str, Any]) -> str:
    hook_event_name = _first_string(payload, "hook_event_name", "hookEventName")
    if hook_event_name == "SessionStart":
        return "session_start"
    if hook_event_name == "SubagentStop":
        return "subagent_end"
    # Check event_metadata for hook_event_name
    event_meta = payload.get("event_metadata", {})
    if isinstance(event_meta, dict):
        meta_event = event_meta.get("hook_event_name")
        if meta_event == "SessionStart":
            return "session_start"
        if meta_event == "SubagentStop":
            return "subagent_end"
    return "turn_end"


def _is_stop_event(payload: dict[str, Any]) -> bool:
    hook_event_name = _first_string(payload, "hook_event_name", "hookEventName")
    if hook_event_name == "Stop":
        return True
    # Check event_metadata
    event_meta = payload.get("event_metadata", {})
    if isinstance(event_meta, dict):
        return event_meta.get("hook_event_name") == "Stop"
    return False


def _is_subagent_stop_event(payload: dict[str, Any]) -> bool:
    return _first_string(payload, "hook_event_name", "hookEventName") == "SubagentStop"


def _seed_opencode_env(session_id: str, payload: dict[str, Any]) -> None:
    os.environ["CHECKPOINT_PROVIDER"] = "opencode"
    os.environ.setdefault("OPENCODE_SESSION_ID", session_id)
    model = _first_string(payload, "model")
    if model:
        os.environ.setdefault("OPENCODE_MODEL", model)
    permission_mode = _first_string(payload, "permission_mode", "permissionMode")
    if permission_mode:
        os.environ.setdefault("OPENCODE_PERMISSION_MODE", permission_mode)
    mode = _first_string(payload, "collaboration_mode_kind", "collaborationModeKind", "mode")
    if mode:
        os.environ.setdefault("OPENCODE_MODE", mode)


def _session_env(payload: dict[str, Any]) -> dict[str, str]:
    """Provider fields recorded at SessionStart for fallback at later turns.

    Capture opencode's approval/sandbox policy when the hook payload carries it.
    The authoritative policy lives in the rollout's `turn_context` (replayed
    verbatim on resume), but recording it in `session_env` makes the checkpoint
    metadata self-describing. Best-effort: opencode does not always deliver
    these at the hook, so they're included only when present.
    """
    fields = {
        "model": _first_string(payload, "model"),
        "permission_mode": _first_string(payload, "permission_mode", "permissionMode"),
        "mode": _first_string(payload, "collaboration_mode_kind", "collaborationModeKind", "mode"),
        "approval_policy": _first_string(payload, "approval_policy", "approvalPolicy"),
        "sandbox_mode": _first_string(payload, "sandbox_mode", "sandboxMode", "sandbox_policy", "sandboxPolicy"),
    }
    return {key: value for key, value in fields.items() if value}


def _turn_record(payload: dict[str, Any]) -> TurnRecord:
    # OpenCode plugin sends messages array; extract last user/assistant pair
    messages = payload.get("messages", [])
    user_message = ""
    assistant_text = ""

    if messages:
        # Find the last user message and last assistant message
        for msg in reversed(messages):
            if isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "user" and not user_message:
                    user_message = content if isinstance(content, str) else str(content)
                elif role == "assistant" and not assistant_text:
                    assistant_text = content if isinstance(content, str) else str(content)
            if user_message and assistant_text:
                break

    # Fallback to direct fields if messages array not available
    if not user_message:
        user_message = _first_string(payload, "prompt", "user_message", "userMessage", "input") or ""
    if not assistant_text:
        assistant_text = _first_string(
            payload,
            "last_assistant_message",
            "assistant_text",
            "assistantText",
            "response",
            "output",
        ) or ""

    return TurnRecord(
        user_message=user_message,
        assistant_text=assistant_text,
        tool_calls=_tool_calls(payload),
        metadata={"hook_payload": payload},
    )


def _tool_calls(payload: dict[str, Any]) -> list[dict[str, Any]]:
    tool_name = _first_string(payload, "tool_name", "toolName")
    if tool_name is None:
        return []
    call: dict[str, Any] = {"tool_name": tool_name}
    for source_key, target_key in (
        ("tool_use_id", "tool_use_id"),
        ("toolUseId", "tool_use_id"),
        ("tool_input", "tool_input"),
        ("toolInput", "tool_input"),
        ("tool_response", "tool_response"),
        ("toolResponse", "tool_response"),
    ):
        if source_key in payload:
            call[target_key] = payload[source_key]
    return [call]


def _trajectory_ref(payload: dict[str, Any], provider: str) -> TrajectoryReference | None:
    transcript_path = _first_string(payload, "transcript_path", "transcriptPath")
    if transcript_path is None:
        return None
    turn_id = payload.get("turn_id") or payload.get("turnId")
    return jsonl_ref_for_turn(provider, Path(transcript_path), turn_id, codex_key, claim_leading_keyless=True)


def _subagent_trajectory_ref(payload: dict[str, Any], transcript_path: str | None) -> TrajectoryReference | None:
    if transcript_path is None:
        return None
    # A subagent's dedicated rollout carries inherited ancestor session_meta
    # records at the head, then the subagent's OWN turns. Capture everything
    # after the leading meta block (the full subagent conversation), not just the
    # SubagentStop turn.
    return jsonl_after_leading_metas(
        "opencode",
        Path(transcript_path),
        is_leading_meta=lambda record: record.get("type") == "session_meta",
    )


def _write_ok() -> None:
    print("{}")


if __name__ == "__main__":
    raise SystemExit(main())
