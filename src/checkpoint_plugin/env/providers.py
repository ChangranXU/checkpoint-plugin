"""Provider-specific environment layouts."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProviderLayout:
    name: str
    home: Path
    memory_dir: Path | None
    mcp_config: Path | None
    settings_files: list[Path]
    skills_dir: Path | None
    project_files: list[str]


def _home() -> Path:
    return Path(os.environ.get("TEST_HOME", str(Path.home()))).expanduser()


def claude_layout() -> ProviderLayout:
    home = _home()
    claude_home = home / ".claude"
    return ProviderLayout(
        name="claude",
        home=claude_home,
        memory_dir=claude_home / "memories",
        mcp_config=home / ".claude.json",
        settings_files=[
            claude_home / "settings.json",
            claude_home / "settings.local.json",
            claude_home / "config.json",
        ],
        skills_dir=claude_home / "skills",
        project_files=[
            "CLAUDE.md",
            "CLAUDE.local.md",
            ".mcp.json",
            ".claude/CLAUDE.md",
            ".claude/settings.json",
            ".claude/settings.local.json",
        ],
    )


def codex_layout() -> ProviderLayout:
    home = _home()
    codex_home = Path(os.environ.get("CODEX_HOME", str(home / ".codex"))).expanduser()
    return ProviderLayout(
        name="codex",
        home=codex_home,
        memory_dir=codex_home / "memories",
        mcp_config=codex_home / "mcp.json",
        settings_files=[
            codex_home / "config.toml",
            codex_home / "auth.json",
            codex_home / "AGENTS.md",
        ],
        skills_dir=codex_home / "skills",
        project_files=[
            "AGENTS.override.md",
            "AGENTS.md",
            ".codex/config.toml",
            ".codex/hooks.json",
            ".codex/requirements.toml",
        ],
    )


def generic_layout() -> ProviderLayout:
    home = _home()
    return ProviderLayout(
        name="generic",
        home=home,
        memory_dir=None,
        mcp_config=None,
        settings_files=[],
        skills_dir=None,
        project_files=["AGENTS.md", "CLAUDE.md", ".mcp.json"],
    )


def detect_provider(cwd: Path) -> ProviderLayout:
    env_provider = os.environ.get("CHECKPOINT_PROVIDER") or os.environ.get("CLAUDE_PROVIDER")
    if env_provider:
        lowered = env_provider.strip().lower()
        if lowered == "claude":
            return claude_layout()
        if lowered == "codex":
            return codex_layout()

    if os.environ.get("CLAUDE_SESSION_ID") or os.environ.get("CLAUDE_PROJECT_DIR"):
        return claude_layout()
    if os.environ.get("CODEX_HOME") or os.environ.get("CODEX_SESSION_ID"):
        return codex_layout()

    cwd = cwd.resolve()
    if any((path / "CLAUDE.md").exists() or (path / ".claude").exists() for path in (cwd, *cwd.parents)):
        return claude_layout()
    if any((path / "AGENTS.md").exists() or (path / ".codex").exists() for path in (cwd, *cwd.parents)):
        return codex_layout()
    return generic_layout()
