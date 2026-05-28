from checkpoint_plugin.env.differ import diff_environments, render_diff
from checkpoint_plugin.types import EnvironmentState


def test_diff_environments_includes_structured_status_changes():
    current = EnvironmentState(
        provider="codex",
        mcp_servers={"context7": "inactive"},
        skill_status={"skill-a": "active"},
        plugin_status={"github": "active"},
    )
    target = EnvironmentState(
        provider="codex",
        mcp_servers={"context7": "active"},
        skill_status={"skill-a": "inactive"},
        plugin_status={"github": "inactive"},
    )

    text = render_diff(diff_environments(current, target), current, target)

    assert "MCP servers" in text
    assert "Skill status" in text
    assert "Plugin status" in text
    assert "~ context7" in text
    assert "~ skill-a" in text
    assert "~ github" in text
