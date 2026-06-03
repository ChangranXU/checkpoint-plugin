"""Test OpenCode integration and hook installation."""

import json
from pathlib import Path

import pytest

from checkpoint_plugin.integrations.hook_installer import install_hooks, uninstall_hooks


def test_opencode_plugin_installs_typescript_file(tmp_path, monkeypatch):
    """Verify that installing OpenCode hooks creates a TypeScript plugin file."""
    config_dir = tmp_path / ".config" / "opencode"
    monkeypatch.setenv("OPENCODE_HOME", str(config_dir))

    results = install_hooks("opencode")

    assert len(results) == 1
    result = results[0]
    assert result.provider == "opencode"
    assert result.path == config_dir / "plugins" / "checkpoint.ts"
    assert result.changed is True
    assert result.path.exists()

    # Verify it's a TypeScript file with correct content
    content = result.path.read_text(encoding="utf-8")
    assert "export const CheckpointPlugin" in content
    assert "event.type === \"session.created\"" in content
    assert "event.type === \"session.idle\"" in content
    assert "checkpoint_plugin.integrations.opencode_hook" in content


def test_opencode_plugin_uninstall_removes_file(tmp_path, monkeypatch):
    """Verify that uninstalling OpenCode hooks removes the plugin file."""
    config_dir = tmp_path / ".config" / "opencode"
    monkeypatch.setenv("OPENCODE_HOME", str(config_dir))

    # Install first
    install_results = install_hooks("opencode")
    plugin_path = install_results[0].path
    assert plugin_path.exists()

    # Uninstall
    uninstall_results = uninstall_hooks("opencode")
    assert len(uninstall_results) == 1
    assert uninstall_results[0].changed is True
    assert not plugin_path.exists()


def test_opencode_plugin_reinstall_is_idempotent(tmp_path, monkeypatch):
    """Verify that reinstalling doesn't change an already-installed plugin."""
    config_dir = tmp_path / ".config" / "opencode"
    monkeypatch.setenv("OPENCODE_HOME", str(config_dir))

    # First install
    results1 = install_hooks("opencode")
    assert results1[0].changed is True

    # Second install - should be idempotent
    results2 = install_hooks("opencode")
    assert results2[0].changed is False
    assert results2[0].path.exists()


def test_opencode_hook_handles_session_created_payload(tmp_path, monkeypatch):
    """Verify the Python hook processes session.created payloads correctly."""
    from io import StringIO
    from checkpoint_plugin.integrations.opencode_hook import main

    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(tmp_path))
    monkeypatch.setenv("OPENCODE_SESSION_ID", "test-session")

    # Simulate TypeScript plugin calling Python hook with session_start event
    payload = json.dumps({
        "sessionID": "test-session",
        "directory": str(tmp_path),
        "agent_type": "primary",
        "event_metadata": {
            "timestamp": "2026-06-03T14:00:00Z",
            "hook_event_name": "SessionStart",
        }
    })

    monkeypatch.setattr("sys.stdin", StringIO(payload))

    exit_code = main(["session_start"])
    assert exit_code == 0

    # Verify checkpoint was created
    session_dir = tmp_path / "sessions" / "test-session"
    assert session_dir.exists()
    metadata = session_dir / "metadata.json"
    assert metadata.exists()


def test_opencode_hook_handles_session_idle_payload(tmp_path, monkeypatch):
    """Verify the Python hook processes session.idle payloads correctly."""
    from io import StringIO
    from checkpoint_plugin.integrations.opencode_hook import main

    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(tmp_path))
    monkeypatch.setenv("OPENCODE_SESSION_ID", "test-session")

    # Create session first
    session_dir = tmp_path / "sessions" / "test-session"
    session_dir.mkdir(parents=True)
    (session_dir / "manifests").mkdir()
    (session_dir / "blobs").mkdir()
    metadata = session_dir / "metadata.json"
    metadata.write_text(json.dumps({
        "session_id": "test-session",
        "created_at": "2026-06-03T14:00:00Z",
    }), encoding="utf-8")

    # Simulate TypeScript plugin calling Python hook with turn_end event
    payload = json.dumps({
        "sessionID": "test-session",
        "directory": str(tmp_path),
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ],
        "event_metadata": {
            "timestamp": "2026-06-03T14:01:00Z",
            "hook_event_name": "Stop",
            "message_count": 2,
        }
    })

    monkeypatch.setattr("sys.stdin", StringIO(payload))

    exit_code = main(["turn_end"])
    assert exit_code == 0

    # Verify turn was recorded - check manifests directory
    manifests_dir = session_dir / "manifests"
    assert manifests_dir.exists()
    # A turn should create a manifest file
    manifest_files = list(manifests_dir.glob("*.json"))
    assert len(manifest_files) > 0
