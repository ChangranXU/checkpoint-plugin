"""Append-only trajectory helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .store import CheckpointStore


def append_event(store: CheckpointStore, event: dict[str, Any]) -> int:
    return store.append_trajectory(event)


def read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    return events
