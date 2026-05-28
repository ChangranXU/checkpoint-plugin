"""Retention cleanup policies."""

from __future__ import annotations

from pathlib import Path

from .paths import sessions_dir
from .store import CheckpointStore


def clean_keep_last(keep_last: int, plugin_home: Path | None = None) -> int:
    removed = 0
    for session in sessions_dir(plugin_home).glob("*"):
        if not session.is_dir():
            continue
        store = CheckpointStore(session)
        manifests = store.list_manifests()
        for manifest in manifests[:-keep_last] if keep_last >= 0 else manifests:
            path = store.manifest_dir / f"turn_{manifest.turn_id:04d}.json"
            if path.exists():
                path.unlink()
                removed += 1
        remaining = [m for m in manifests[-keep_last:]] if keep_last > 0 else []
        store._atomic_write(
            store.index_path,
            "{\n"
            + ",\n".join(f'  "{m.turn_id}": "turn_{m.turn_id:04d}.json"' for m in remaining)
            + ("\n" if remaining else "")
            + "}\n",
        )
    return removed
