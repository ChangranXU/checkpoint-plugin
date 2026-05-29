from checkpoint_plugin.env.collector import collect_environment
from checkpoint_plugin.env.providers import claude_layout, codex_layout
from checkpoint_plugin.store import CheckpointStore
import hashlib


def test_collect_environment_with_mock_claude_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    cwd = tmp_path / "project"
    claude = home / ".claude"
    (claude / "memories").mkdir(parents=True)
    (claude / "skills" / "skill-a").mkdir(parents=True)
    cwd.mkdir()
    monkeypatch.setenv("TEST_HOME", str(home))
    monkeypatch.setenv("CLAUDE_MODEL", "sonnet-test")

    (claude / "memories" / "note.md").write_text("memory", encoding="utf-8")
    (claude / "settings.json").write_text('{"permissions": {}}', encoding="utf-8")
    (claude / "skills" / "skill-a" / "SKILL.md").write_text("skill", encoding="utf-8")
    (home / ".claude.json").write_text('{"mcpServers": {"ctx": {"command": "x"}}}', encoding="utf-8")
    (cwd / "CLAUDE.md").write_text("project", encoding="utf-8")

    store = CheckpointStore(tmp_path / "session")
    env = collect_environment(cwd, claude_layout(), store)

    assert env.provider == "claude"
    assert env.model == "sonnet-test"
    assert "note.md" in env.memory_files
    # ~/.claude.json is captured structurally (R2), never as a raw blob.
    assert env.mcp_config is None
    assert env.mcp_servers == {"ctx": "active"}
    assert "settings.json" in env.settings
    assert "user/skill-a/SKILL.md" in env.skills
    assert str(cwd / "CLAUDE.md") in env.project_context


def test_collect_environment_follows_symlinked_skill_dirs(tmp_path, monkeypatch):
    home = tmp_path / "home"
    cwd = tmp_path / "project"
    claude = home / ".claude"
    shared = home / ".cc-switch" / "skills" / "linked-skill"
    (claude / "skills").mkdir(parents=True)
    shared.mkdir(parents=True)
    cwd.mkdir()
    monkeypatch.setenv("TEST_HOME", str(home))

    (claude / "skills" / ".DS_Store").write_text("metadata", encoding="utf-8")
    (claude / "skills" / "linked-skill").symlink_to(shared, target_is_directory=True)
    (shared / "SKILL.md").write_text("linked skill", encoding="utf-8")
    (shared / "notes.md").write_text("notes", encoding="utf-8")

    store = CheckpointStore(tmp_path / "session")
    env = collect_environment(cwd, claude_layout(), store)

    assert "user/linked-skill/SKILL.md" in env.skills
    assert "user/linked-skill/notes.md" in env.skills
    assert "user/.DS_Store" in env.skills
    assert env.skill_status == {"linked-skill": "present"}
    assert env.extra["skill_symlinks"] == {
        "user/linked-skill": str(shared),
    }


def test_collect_environment_skips_recursive_skill_symlink(tmp_path, monkeypatch):
    home = tmp_path / "home"
    cwd = tmp_path / "project"
    claude = home / ".claude"
    skill = claude / "skills" / "loop-skill"
    skill.mkdir(parents=True)
    cwd.mkdir()
    monkeypatch.setenv("TEST_HOME", str(home))

    (skill / "SKILL.md").write_text("loop skill", encoding="utf-8")
    (skill / "self").symlink_to(skill, target_is_directory=True)

    store = CheckpointStore(tmp_path / "session")
    env = collect_environment(cwd, claude_layout(), store)

    assert "user/loop-skill/SKILL.md" in env.skills
    assert not any(path.startswith("user/loop-skill/self/") for path in env.skills)
    assert env.extra["skill_symlinks"] == {
        "user/loop-skill/self": str(skill),
    }


def test_collect_codex_structured_env_status(tmp_path, monkeypatch):
    home = tmp_path / "home"
    cwd = tmp_path / "project"
    codex = home / ".codex"
    (codex / "skills" / "skill-a").mkdir(parents=True)
    (home / ".agents" / "skills" / "global-agent-skill").mkdir(parents=True)
    cwd.mkdir()
    monkeypatch.setenv("TEST_HOME", str(home))
    monkeypatch.setenv("CODEX_HOME", str(codex))

    (cwd / ".agents" / "skills" / "project-skill").mkdir(parents=True)
    (codex / "skills" / "skill-a" / "SKILL.md").write_text("skill", encoding="utf-8")
    (home / ".agents" / "skills" / "global-agent-skill" / "SKILL.md").write_text(
        "global agent skill",
        encoding="utf-8",
    )
    (cwd / ".agents" / "skills" / "project-skill" / "SKILL.md").write_text("project skill", encoding="utf-8")
    (cwd / ".mcp.json").write_text('{"mcpServers":{"project_mcp":{"command":"local"}}}', encoding="utf-8")
    (cwd / ".codex").mkdir()
    (cwd / ".codex" / "config.toml").write_text(
        """
[mcp_servers.project_config_mcp]
command = "local"

[plugins."project-plugin"]
enabled = true
""",
        encoding="utf-8",
    )
    (codex / "config.toml").write_text(
        """
[mcp_servers.context7]
type = "stdio"
command = "npx"

[mcp_servers.disabled_server]
command = "nope"
enabled = false

[plugins."github@openai-curated"]
enabled = true

[plugins."browser@openai-bundled"]
enabled = false

[[skills.config]]
path = "{skill_path}"
enabled = false
""".format(skill_path=(codex / "skills" / "skill-a" / "SKILL.md").as_posix()),
        encoding="utf-8",
    )

    store = CheckpointStore(tmp_path / "session")
    env = collect_environment(cwd, codex_layout(), store)

    assert env.mcp_config is not None
    assert env.mcp_servers == {
        "context7": "active",
        "disabled_server": "inactive",
        "project_config_mcp": "active",
        "project_mcp": "active",
    }
    assert env.plugin_status == {
        "browser@openai-bundled": "inactive",
        "github@openai-curated": "active",
        "project-plugin": "active",
    }
    assert "codex-user/skill-a/SKILL.md" in env.skills
    assert "agent-user/global-agent-skill/SKILL.md" in env.skills
    assert any(key.endswith(".agents/skills/project-skill/SKILL.md") for key in env.skills)
    assert env.skill_status["skill-a"] == "inactive"
    assert env.skill_status["global-agent-skill"] == "present"
    assert env.skill_status["project-skill"] == "present"
    assert any(key.endswith(".mcp.json") for key in env.mcp_configs)


def test_collect_codex_plugin_cache_skills(tmp_path, monkeypatch):
    home = tmp_path / "home"
    cwd = tmp_path / "project"
    plugin_skill = (
        home
        / ".codex"
        / "plugins"
        / "cache"
        / "openai-bundled"
        / "browser"
        / "26.519.81530"
        / "skills"
        / "browser"
    )
    cwd.mkdir()
    plugin_skill.mkdir(parents=True)
    monkeypatch.setenv("TEST_HOME", str(home))

    (plugin_skill / "SKILL.md").write_text("browser skill", encoding="utf-8")

    store = CheckpointStore(tmp_path / "session")
    env = collect_environment(cwd, codex_layout(), store)

    key = "plugin:openai-bundled:browser:26.519.81530/browser/SKILL.md"
    assert key in env.skills
    assert store.load_blob(env.skills[key]) == b"browser skill"
    assert env.skill_status["browser"] == "present"


def test_collect_claude_structured_env_status(tmp_path, monkeypatch):
    home = tmp_path / "home"
    cwd = tmp_path / "project"
    claude = home / ".claude"
    plugin = claude / "plugins" / "marketplaces" / "official" / "plugins" / "code-review"
    external_plugin = claude / "plugins" / "marketplaces" / "official" / "external_plugins" / "context7"
    (claude / "skills" / "skill-a").mkdir(parents=True)
    plugin.mkdir(parents=True)
    external_plugin.mkdir(parents=True)
    cwd.mkdir()
    monkeypatch.setenv("TEST_HOME", str(home))

    (claude / "skills" / "skill-a" / "SKILL.md").write_text("skill", encoding="utf-8")
    (cwd / ".claude" / "skills" / "project-skill").mkdir(parents=True)
    (cwd / ".claude" / "skills" / "project-skill" / "SKILL.md").write_text("project skill", encoding="utf-8")
    (home / ".claude.json").write_text(
        """
{
  "mcpServers": {
    "context7": {"type": "stdio", "command": "npx"}
  },
  "enabledPlugins": ["code-review"],
  "skillOverrides": {"skill-a": false},
  "projects": {
    "%s": {
      "mcpServers": {"project_server": {"command": "local"}},
      "enabledMcpjsonServers": ["enabled_project"],
      "disabledMcpjsonServers": ["disabled_project"]
    }
  }
}
"""
        % cwd.as_posix(),
        encoding="utf-8",
    )

    store = CheckpointStore(tmp_path / "session")
    env = collect_environment(cwd, claude_layout(), store)

    assert env.mcp_servers == {
        "context7": "active",
        "disabled_project": "inactive",
        "enabled_project": "active",
        "project_server": "active",
    }
    assert env.skill_status == {
        "project-skill": "present",
        "skill-a": "inactive",
    }
    assert env.plugin_status == {
        "code-review": "active",
        "context7": "present",
    }


def test_collect_environment_never_stores_secret_files(tmp_path, monkeypatch):
    home = tmp_path / "home"
    cwd = tmp_path / "project"
    codex = home / ".codex"
    codex.mkdir(parents=True)
    cwd.mkdir()
    monkeypatch.setenv("TEST_HOME", str(home))
    monkeypatch.setenv("CODEX_HOME", str(codex))

    secret = b'{"OPENAI_API_KEY": "sk-must-not-be-stored"}'
    (codex / "auth.json").write_bytes(secret)
    (codex / "config.toml").write_text('model = "gpt-test"\n', encoding="utf-8")
    (cwd / ".env").write_bytes(b"TOKEN=must-not-be-stored\n")
    (cwd / "AGENTS.md").write_text("project rules", encoding="utf-8")

    store = CheckpointStore(tmp_path / "session")
    env = collect_environment(cwd, codex_layout(), store)

    # config.toml is still captured; auth.json/.env are filtered out entirely.
    assert "config.toml" in env.settings
    assert "auth.json" not in env.settings
    assert not any(key.endswith(".env") for key in env.project_context)

    # Defense in depth: the secret bytes never reached the blob store.
    secret_sha = hashlib.sha256(secret).hexdigest()
    assert not store.blob_path(secret_sha).exists()
