from checkpoint_plugin.env.collector import collect_environment
from checkpoint_plugin.env.providers import claude_layout
from checkpoint_plugin.store import CheckpointStore


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
    (home / ".claude.json").write_text('{"mcpServers": {}}', encoding="utf-8")
    (cwd / "CLAUDE.md").write_text("project", encoding="utf-8")

    store = CheckpointStore(tmp_path / "session")
    env = collect_environment(cwd, claude_layout(), store)

    assert env.provider == "claude"
    assert env.model == "sonnet-test"
    assert "note.md" in env.memory_files
    assert env.mcp_config is not None
    assert "settings.json" in env.settings
    assert "skill-a/SKILL.md" in env.skills
    assert str(cwd / "CLAUDE.md") in env.project_context
