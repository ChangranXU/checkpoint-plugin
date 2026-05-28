import io
import json

from checkpoint_plugin.env.collector import environment_from_blob
from checkpoint_plugin.integrations import codex_hook
from checkpoint_plugin.store import CheckpointStore


def test_codex_session_start_writes_metadata(tmp_path, monkeypatch):
    plugin_home = tmp_path / "plugin"
    cwd = tmp_path / "work"
    cwd.mkdir()
    payload = {
        "hook_event_name": "SessionStart",
        "session_id": "codex-s1",
        "cwd": str(cwd),
        "model": "gpt-test",
        "permission_mode": "plan",
        "source": "startup",
    }
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(plugin_home))
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))

    assert codex_hook.main([]) == 0

    metadata = json.loads((plugin_home / "sessions" / "codex-s1" / "metadata.json").read_text())
    assert metadata["provider"] == "codex"
    assert metadata["cwd"] == str(cwd)


def test_codex_turn_end_maps_payload_to_checkpoint(tmp_path, monkeypatch):
    plugin_home = tmp_path / "plugin"
    codex_home = tmp_path / "home" / ".codex"
    cwd = tmp_path / "work"
    codex_home.mkdir(parents=True)
    cwd.mkdir()
    (codex_home / "config.toml").write_text('model = "gpt-test"\n', encoding="utf-8")
    (cwd / "AGENTS.md").write_text("agent", encoding="utf-8")
    payload = {
        "hook_event_name": "Stop",
        "session_id": "codex-s1",
        "cwd": str(cwd),
        "turn_id": "turn-1",
        "model": "gpt-test",
        "permission_mode": "plan",
        "last_assistant_message": "done",
    }
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(plugin_home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))

    assert codex_hook.main([]) == 0

    store = CheckpointStore(plugin_home / "sessions" / "codex-s1")
    manifest = store.read_manifest(0)
    env = environment_from_blob(manifest.env_ref, store)
    trajectory = (plugin_home / "sessions" / "codex-s1" / "trajectory.jsonl").read_text()
    assert env.provider == "codex"
    assert env.model == "gpt-test"
    assert env.permission_mode == "plan"
    assert "config.toml" in env.settings
    assert manifest.user_message_preview == ""
    assert '"assistant_text": "done"' in trajectory
