"""Shared JSONL transcript slicer for hook adapters.

Slices a provider transcript into a byte range that covers the current turn.
Each provider supplies a `key_extractor` returning a per-turn key; records
without a key are attributed to the most recent keyed record above them.
Turn 0 always anchors at byte 0 so leading provider-emitted records (mode,
permission-mode, file-history snapshots) are never lost.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from checkpoint_plugin.types import TrajectoryReference

KeyExtractor = Callable[[dict[str, Any]], Any]


def claude_key(record: dict[str, Any]) -> Any:
    """promptId is the only stable per-turn key Claude transcripts carry."""
    return record.get("promptId")


def codex_key(record: dict[str, Any]) -> Any:
    if "turn_id" in record:
        return record["turn_id"]
    if "turnId" in record:
        return record["turnId"]
    payload = record.get("payload")
    if isinstance(payload, dict):
        return payload.get("turn_id") or payload.get("turnId")
    return None


def jsonl_ref_for_turn(
    provider: str,
    path: Path,
    turn_id: Any,
    key_extractor: KeyExtractor,
) -> TrajectoryReference | None:
    try:
        data = path.expanduser().read_bytes()
    except OSError:
        return None

    lines = _parse_jsonl_lines(data)
    if not lines:
        return None

    keyed = [(start, end, key_extractor(record) if isinstance(record, dict) else None) for start, end, record in lines]

    if turn_id is not None:
        match = _slice_for_turn_id(keyed, turn_id, len(data))
        if match is not None:
            start, end = match
            return _build_ref(provider, path, data, start, end)

    latest_key = _latest_distinct_key(keyed)
    if latest_key is None:
        return _build_ref(provider, path, data, 0, len(data))

    start_offset = _first_offset_for_key(keyed, latest_key, len(data))
    if start_offset == 0 or _no_prior_keys(keyed, latest_key):
        return _build_ref(provider, path, data, 0, len(data))
    return _build_ref(provider, path, data, start_offset, len(data))


def jsonl_count_records(data: bytes) -> int:
    return sum(1 for line in data.splitlines() if line.strip())


def _parse_jsonl_lines(data: bytes) -> list[tuple[int, int, Any]]:
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
    return lines


def _build_ref(provider: str, path: Path, data: bytes, start: int, end: int) -> TrajectoryReference:
    return TrajectoryReference(
        provider=provider,
        transcript_path=str(path.expanduser().resolve()),
        start_offset=start,
        end_offset=end,
        record_count=jsonl_count_records(data[start:end]),
    )


def _slice_for_turn_id(
    keyed: list[tuple[int, int, Any]],
    turn_id: Any,
    file_size: int,
) -> tuple[int, int] | None:
    matches = [(start, end) for start, end, key in keyed if _keys_match(key, turn_id)]
    if not matches:
        return None
    start_offset = matches[0][0]
    next_start = _next_distinct_key_offset(keyed, start_offset, turn_id)
    end_offset = next_start if next_start is not None else file_size
    if start_offset == 0 or _no_prior_keys_before(keyed, start_offset):
        return 0, end_offset
    return start_offset, end_offset


def _next_distinct_key_offset(
    keyed: list[tuple[int, int, Any]],
    start_offset: int,
    turn_id: Any,
) -> int | None:
    for line_start, _, key in keyed:
        if line_start > start_offset and key is not None and not _keys_match(key, turn_id):
            return line_start
    return None


def _latest_distinct_key(keyed: list[tuple[int, int, Any]]) -> Any:
    for _, _, key in reversed(keyed):
        if key is not None:
            return key
    return None


def _first_offset_for_key(
    keyed: list[tuple[int, int, Any]],
    target: Any,
    fallback: int,
) -> int:
    for start, _, key in keyed:
        if key is not None and _keys_match(key, target):
            return start
    return fallback


def _no_prior_keys(keyed: list[tuple[int, int, Any]], target: Any) -> bool:
    """True if `target` is the only distinct key in the transcript."""
    for _, _, key in keyed:
        if key is not None and not _keys_match(key, target):
            return False
    return True


def _no_prior_keys_before(keyed: list[tuple[int, int, Any]], offset: int) -> bool:
    for start, _, key in keyed:
        if start >= offset:
            return True
        if key is not None:
            return False
    return True


def _keys_match(left: Any, right: Any) -> bool:
    return left == right or (left is not None and right is not None and str(left) == str(right))
