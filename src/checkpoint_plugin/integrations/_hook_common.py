"""Helpers shared by the Codex and Claude hook adapters.

These are provider-neutral: stdin payload parsing, first-string extraction, and
the empty trajectory reference used when a Stop event carries no usable
transcript. Keeping them in one place avoids drift between the two adapters.
"""

from __future__ import annotations

import json
import sys
from typing import Any

from checkpoint_plugin.types import TrajectoryReference


def read_payload() -> dict[str, Any]:
    raw = sys.stdin.read().strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}
    return value if isinstance(value, dict) else {"payload": value}


def first_string(payload: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str):
            return value
    return None


def empty_trajectory_ref(provider: str) -> TrajectoryReference:
    return TrajectoryReference(
        provider=provider,
        transcript_path="",
        start_offset=0,
        end_offset=0,
        record_count=0,
    )
