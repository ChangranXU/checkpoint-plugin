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
from ._hook_common import parent_session_env as _parent_session_env
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
    # SubagentStop omits `model` (SessionStart-only); inherit the parent's pinned
    # session_env so the subagent checkpoint records the same model/effort (G2).
    sub_env = {**_parent_session_env(parent_session_id), **_session_env(payload)}
    # P6-6: persist whatever durable spawn link SubagentStop provides — the
    # agent_id (primary match via _manifest_references_agent) plus the sidechain
    # filename stem, which survives even when agent_id is absent from any slice.
    lineage: dict[str, Any] = {
        "parent_session_id": parent_session_id,
        "agent_id": agent_id,
        "agent_type": _first_string(payload, "agent_type", "agentType"),
    }
    if transcript_path is not None:
        lineage["sidechain_stem"] = transcript_path.stem
    ref = _subagent_trajectory_ref(payload, transcript_path)
    if ref is None:
        # P6-9: no sidechain file was found, so the slice is empty. Record WHY at a
        # defined location so a reader knows the empty trajectory is expected, not a
        # capture bug.
        lineage["capture_status"] = "no_sidechain_file"
        ref = _empty_trajectory_ref("claude")
    elif transcript_path is not None:
        # F12: SubagentStop can fire before the subagent's final assistant record is
        # flushed (verified: af44fcc2 slice caught 3/4 records, the deliverable rec3
        # flushed ~14ms later). The slice reads to the current EOF, so a later flush
        # is silently truncated. We can't reliably block for the flush, but we record
        # the sidechain's observed size+mtime so a consumer (or a later re-slice on
        # the next session_start) can DETECT that the stored slice is stale — the file
        # having grown past `sidechain_observed_size` means records were missed.
        observed = _sidechain_observed_state(transcript_path, ref.end_offset)
        if observed is not None:
            lineage["sidechain_observed_size"], lineage["sidechain_observed_mtime"] = observed
    coordinator.on_session_start(
        source="subagent",
        session_env=sub_env,
        lineage=lineage,
    )
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
    """Resolve the subagent's OWN transcript file (never the parent's).

    Claude writes a subagent to a dedicated sidechain file
    `<parent-dir>/<session>/subagents/agent-<id>.jsonl` (or a flat sibling). The
    Stop payload's `transcript_path`, however, is often the PARENT main
    transcript. Slicing that would mislabel the parent's own thread as the
    subagent's work (F4), so we only ever return a path that is genuinely a
    subagent transcript; if we can't find one we return None and the caller
    records lineage without a (misleading) trajectory slice.
    """
    explicit = _first_string(payload, "transcript_path", "transcriptPath")
    parent = _first_string(payload, "parent_transcript_path", "parentTranscriptPath")
    if explicit:
        path = Path(explicit)
        if _is_subagent_transcript(path):
            return path
        # `explicit` is the parent main transcript: locate the real sidechain.
        nested = _existing_nested_subagent(path, agent_id)
        return nested  # None if no dedicated subagent file exists (don't slice parent).
    if parent and agent_id:
        return _existing_nested_subagent(Path(parent), agent_id)
    return None


def _is_subagent_transcript(path: Path) -> bool:
    """A subagent transcript lives in a `subagents/` dir or is named `agent-*`."""
    return path.parent.name == "subagents" or path.stem.startswith("agent-")


def _existing_nested_subagent(parent_transcript: Path, agent_id: str | None) -> Path | None:
    """First existing `subagents/agent-<id>.jsonl` derived from a parent path."""
    if not agent_id:
        return None
    candidates = [
        parent_transcript.parent / parent_transcript.stem / "subagents" / f"agent-{agent_id}.jsonl",
        parent_transcript.parent / "subagents" / f"agent-{agent_id}.jsonl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _subagent_trajectory_ref(payload: dict[str, Any], transcript_path: Path | None) -> TrajectoryReference | None:
    if transcript_path is None:
        return None
    turn_id = payload.get("turn_id") or payload.get("turnId")
    return jsonl_ref_for_turn("claude", transcript_path, turn_id, claude_key)


def _stem(path: Path | None) -> str:
    return path.stem if path is not None else "unknown"


def _sidechain_observed_state(transcript_path: Path, sliced_end_offset: int) -> tuple[int, str] | None:
    """The sidechain file's (size, mtime-iso) at capture time, for staleness checks (F12).

    A later flush grows the file beyond the captured slice's end; recording the size
    we sliced against lets a reader tell whether the stored slice still covers the
    whole file. Returns None if the file can't be stat'd. We record the size we
    actually sliced to (`sliced_end_offset`) so a future re-stat that returns a larger
    size is an unambiguous staleness signal.
    """
    try:
        stat = transcript_path.stat()
    except OSError:
        return None
    from datetime import datetime, timezone

    mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()
    return sliced_end_offset, mtime


if __name__ == "__main__":
    raise SystemExit(main())
