"""Small path helpers shared by checkpoint restore flows."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

PathRootKind = Literal["file", "directory"]


def mirror_path(path: Path) -> Path:
    return Path(*path.parts[1:]) if path.is_absolute() else path


def path_within(path: Path, root: Path) -> bool:
    resolved = path.resolve(strict=False)
    resolved_root = root.resolve(strict=False)
    return resolved == resolved_root or resolved_root in resolved.parents


def path_matches_root(path: Path, root: Path, *, kind: PathRootKind = "directory") -> bool:
    resolved = path.resolve(strict=False)
    resolved_root = root.resolve(strict=False)
    if kind == "file":
        return resolved == resolved_root
    return resolved == resolved_root or resolved_root in resolved.parents
