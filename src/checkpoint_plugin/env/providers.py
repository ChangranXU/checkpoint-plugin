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
    mcp_config_files: list[Path]
    settings_files: list[Path]
    skills_dirs: dict[str, Path]
    project_files: list[str]


def _home() -> Path:
    return Path(os.environ.get("TEST_HOME", str(Path.home()))).expanduser()


def claude_layout() -> ProviderLayout:
    home = _home()
    claude_home = home / ".claude"
    managed_root = Path("/Library/Application Support/ClaudeCode") if os.name != "nt" else Path.home()
    return ProviderLayout(
        name="claude",
        home=claude_home,
        memory_dir=claude_home / "memories",
        mcp_config=home / ".claude.json",
        mcp_config_files=[
            home / ".claude.json",
            managed_root / "managed-mcp.json",
        ],
        settings_files=[
            home / ".claude.json",
            managed_root / "managed-settings.json",
            managed_root / "managed-mcp.json",
            claude_home / "settings.json",
            claude_home / "settings.local.json",
            claude_home / "config.json",
            claude_home / "CLAUDE.md",
            claude_home / "rules.json",
        ],
        skills_dirs={
            "user": claude_home / "skills",
        },
        project_files=[
            "CLAUDE.md",
            "CLAUDE.local.md",
            ".mcp.json",
            ".claude/CLAUDE.md",
            ".claude/settings.json",
            ".claude/settings.local.json",
            ".claude/memory",
            ".claude/file-history",
            ".claude/shell-snapshots",
            ".claude/todos",
            ".claude/skills",
            ".claude/agents",
            ".claude/commands",
            ".claude/output-styles",
        ],
    )


def codex_layout() -> ProviderLayout:
    home = _home()
    codex_home = Path(os.environ.get("CODEX_HOME", str(home / ".codex"))).expanduser()
    system_codex = Path("/etc/codex") if os.name != "nt" else codex_home
    return ProviderLayout(
        name="codex",
        home=codex_home,
        memory_dir=codex_home / "memories",
        mcp_config=codex_home / "config.toml",
        mcp_config_files=[
            system_codex / "managed_config.toml",
            system_codex / "requirements.toml",
            codex_home / "config.toml",
            home / ".mcp.json",
        ],
        settings_files=[
            system_codex / "managed_config.toml",
            system_codex / "requirements.toml",
            codex_home / "config.toml",
            codex_home / "auth.json",
            codex_home / "AGENTS.md",
            codex_home / "hooks.json",
            codex_home / "rules.json",
        ],
        skills_dirs={
            "codex-user": codex_home / "skills",
            "agent-user": home / ".agents" / "skills",
            "codex-admin": Path("/etc/codex/skills"),
        },
        project_files=[
            "AGENTS.override.md",
            "AGENTS.md",
            ".mcp.json",
            ".codex/config.toml",
            ".codex/hooks.json",
            ".codex/requirements.toml",
            ".codex/rules",
            ".codex/skills",
            ".agents/skills",
        ],
    )


def generic_layout() -> ProviderLayout:
    home = _home()
    return ProviderLayout(
        name="generic",
        home=home,
        memory_dir=None,
        mcp_config=None,
        mcp_config_files=[],
        settings_files=[],
        skills_dirs={},
        project_files=["AGENTS.md", "CLAUDE.md", ".mcp.json"],
    )


def layout_for_provider(name: str) -> ProviderLayout:
    normalized = name.strip().lower()
    if normalized == "claude":
        return claude_layout()
    if normalized == "codex":
        return codex_layout()
    return generic_layout()


def detect_provider(cwd: Path) -> ProviderLayout:
    env_provider = os.environ.get("CHECKPOINT_PROVIDER") or os.environ.get("CLAUDE_PROVIDER")
    if env_provider:
        return layout_for_provider(env_provider)

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
