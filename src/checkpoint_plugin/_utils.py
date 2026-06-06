"""Shared utility functions used across checkpoint plugin modules."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def expand_and_resolve(path: Path | str) -> Path:
    """Expand user home and resolve path to absolute form."""
    return Path(path).expanduser().resolve()


def non_empty_str(value: Any) -> str | None:
    """Return value as string if it's a non-empty string, otherwise None."""
    return value if isinstance(value, str) and value else None


def clean_string_dict(d: dict[Any, Any] | None) -> dict[str, str]:
    """Convert dict to clean string-to-string dict, filtering empty values."""
    if not d:
        return {}
    return {str(k): str(v) for k, v in d.items() if v}


def load_json_safe(path: Path, default: Any = None) -> Any:
    """Load JSON from path, returning default on error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def load_json_value(data: dict[str, Any], key: str, default: Any = None) -> Any:
    """Get JSON value from dict, handling missing keys."""
    value = data.get(key, default)
    return value if value is not None else default


def read_metadata_json(path: Path) -> dict[str, Any]:
    """Read metadata.json file, returning empty dict on error."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def backup_file(path: Path, backup_path: Path, backed_up: list[str]) -> None:
    """Back up a file to backup_path and record it in backed_up list."""
    backup_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, backup_path)
    backed_up.append(str(backup_path))
