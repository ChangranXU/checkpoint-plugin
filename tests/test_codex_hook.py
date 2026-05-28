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
    assert metadata["source"] == "startup"


def test_codex_turn_end_maps_payload_to_checkpoint(tmp_path, monkeypatch):
    plugin_home = tmp_path / "plugin"
    codex_home = tmp_path / "home" / ".codex"
    cwd = tmp_path / "work"
    transcript = tmp_path / "rollout.jsonl"
    codex_home.mkdir(parents=True)
    cwd.mkdir()
    (codex_home / "config.toml").write_text('model = "gpt-test"\n', encoding="utf-8")
    (cwd / "AGENTS.md").write_text("agent", encoding="utf-8")
    transcript.write_text(
        "\n".join(
            [
                json.dumps({"turn_id": "turn-0", "message": "previous"}),
                json.dumps({"type": "event_msg", "payload": {"turn_id": "turn-1", "type": "user_message", "message": "current"}}),
                "",
            ]
        ),
        encoding="utf-8",
    )
    payload = {
        "hook_event_name": "Stop",
        "session_id": "codex-s1",
        "cwd": str(cwd),
        "turn_id": "turn-1",
        "transcript_path": str(transcript),
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
    assert env.provider == "codex"
    assert env.model == "gpt-test"
    assert env.permission_mode == "plan"
    assert "config.toml" in env.settings
    assert manifest.trajectory_ref is not None
    assert manifest.trajectory_ref.transcript_path == str(transcript)
    assert manifest.trajectory_ref.record_count == 1
    assert manifest.user_message_preview == "current"
    assert b'"message": "current"' in store.read_trajectory_slice(manifest.trajectory_ref)
    assert not (plugin_home / "sessions" / "codex-s1" / "trajectory.jsonl").exists()


def test_codex_reference_includes_intervening_records_until_next_turn(tmp_path):
    transcript = tmp_path / "rollout.jsonl"
    transcript.write_text(
        "\n".join(
            [
                json.dumps({"type": "event_msg", "payload": {"turn_id": "turn-1", "type": "task_started"}}),
                json.dumps({"type": "response_item", "payload": {"message": "assistant"}}),
                json.dumps({"type": "event_msg", "payload": {"turn_id": "turn-1", "type": "task_complete"}}),
                json.dumps({"type": "event_msg", "payload": {"turn_id": "turn-2", "type": "task_started"}}),
                "",
            ]
        ),
        encoding="utf-8",
    )

    ref = codex_hook._trajectory_ref(
        {"transcript_path": str(transcript), "turn_id": "turn-1"},
        provider="codex",
    )

    assert ref is not None
    assert ref.record_count == 3
    assert transcript.read_bytes()[ref.start_offset : ref.end_offset].count(b"\n") == 3


def test_codex_tool_events_do_not_create_checkpoint(tmp_path, monkeypatch):
    plugin_home = tmp_path / "plugin"
    cwd = tmp_path / "work"
    cwd.mkdir()
    (cwd / "AGENTS.md").write_text("agent", encoding="utf-8")
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(plugin_home))

    tool_payload = {
        "hook_event_name": "PostToolUse",
        "session_id": "codex-s1",
        "cwd": str(cwd),
        "turn_id": "provider-turn-1",
        "tool_name": "Bash",
        "tool_input": {"command": "pwd"},
        "tool_response": str(cwd),
    }
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(tool_payload)))
    assert codex_hook.main([]) == 0

    store = CheckpointStore(plugin_home / "sessions" / "codex-s1")
    assert store.list_manifests() == []
    assert not store.trajectory_path.exists()


def test_codex_stop_without_transcript_does_not_copy_trajectory(tmp_path, monkeypatch):
    plugin_home = tmp_path / "plugin"
    cwd = tmp_path / "work"
    cwd.mkdir()
    (cwd / "AGENTS.md").write_text("agent", encoding="utf-8")
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(plugin_home))
    payload = {
        "hook_event_name": "Stop",
        "session_id": "codex-s1",
        "cwd": str(cwd),
        "last_assistant_message": "done",
    }
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))

    assert codex_hook.main([]) == 0

    store = CheckpointStore(plugin_home / "sessions" / "codex-s1")
    manifest = store.read_manifest(0)
    assert manifest.trajectory_ref is not None
    assert manifest.trajectory_ref.record_count == 0
    assert not store.trajectory_path.exists()
