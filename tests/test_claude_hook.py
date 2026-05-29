import io
import json

from checkpoint_plugin.env.collector import environment_from_blob
from checkpoint_plugin.integrations import claude_code_hook
from checkpoint_plugin.store import CheckpointStore


def test_claude_tool_events_do_not_create_checkpoint(tmp_path, monkeypatch):
    plugin_home = tmp_path / "plugin"
    cwd = tmp_path / "work"
    cwd.mkdir()
    (cwd / "AGENTS.md").write_text("agent", encoding="utf-8")
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(plugin_home))
    monkeypatch.setenv("CLAUDE_SESSION_ID", "claude-s1")

    tool_payload = {
        "hook_event_name": "PostToolUse",
        "session_id": "claude-s1",
        "cwd": str(cwd),
        "tool_name": "Read",
        "tool_input": {"file_path": str(cwd / "AGENTS.md")},
        "tool_response": {"file": {"content": "agent"}},
    }
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(tool_payload)))
    assert claude_code_hook.main(["turn_end"]) == 0

    store = CheckpointStore(plugin_home / "sessions" / "claude-s1")
    assert store.list_manifests() == []
    assert not store.trajectory_path.exists()


def test_claude_stop_records_trajectory_reference(tmp_path, monkeypatch):
    plugin_home = tmp_path / "plugin"
    cwd = tmp_path / "work"
    transcript = tmp_path / "claude.jsonl"
    cwd.mkdir()
    (cwd / "AGENTS.md").write_text("agent", encoding="utf-8")
    transcript.write_text(
        "\n".join(
            [
                json.dumps({"turnId": "provider-turn-1", "message": "tool"}),
                json.dumps({"turnId": "provider-turn-1", "message": "done"}),
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(plugin_home))
    monkeypatch.setenv("CLAUDE_SESSION_ID", "claude-s1")
    stop_payload = {
        "hook_event_name": "Stop",
        "session_id": "claude-s1",
        "cwd": str(cwd),
        "turnId": "provider-turn-1",
        "transcript_path": str(transcript),
        "last_assistant_message": "done",
    }
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(stop_payload)))
    assert claude_code_hook.main(["turn_end"]) == 0

    store = CheckpointStore(plugin_home / "sessions" / "claude-s1")
    manifests = store.list_manifests()
    assert len(manifests) == 1
    assert manifests[0].trajectory_ref is not None
    assert manifests[0].trajectory_ref.record_count == 2
    assert store.read_trajectory_slice(manifests[0].trajectory_ref).count(b"\n") == 2


def test_claude_slices_by_prompt_id(tmp_path, monkeypatch):
    plugin_home = tmp_path / "plugin"
    cwd = tmp_path / "work"
    transcript = tmp_path / "claude.jsonl"
    cwd.mkdir()
    transcript.write_text(
        "\n".join(
            [
                json.dumps({"type": "mode", "mode": "default"}),
                json.dumps({"type": "permission-mode", "permissionMode": "default"}),
                json.dumps({"type": "user", "promptId": "p-1", "message": {"role": "user", "content": "first"}}),
                json.dumps({"type": "assistant", "message": {"role": "assistant", "content": "ok"}}),
                json.dumps({"type": "system", "subtype": "stop_hook_summary"}),
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(plugin_home))
    monkeypatch.setenv("CLAUDE_SESSION_ID", "claude-prompt")
    payload = {
        "hook_event_name": "Stop",
        "session_id": "claude-prompt",
        "cwd": str(cwd),
        "transcript_path": str(transcript),
    }
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    assert claude_code_hook.main(["turn_end"]) == 0

    store = CheckpointStore(plugin_home / "sessions" / "claude-prompt")
    ref = store.list_manifests()[0].trajectory_ref
    assert ref is not None
    assert ref.start_offset == 0  # turn 0 anchors at beginning of file
    assert ref.record_count == 5

    # Append turn 2 and confirm slicer picks up the new promptId boundary
    with transcript.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "user", "promptId": "p-2", "message": {"role": "user", "content": "second"}}) + "\n")
        handle.write(json.dumps({"type": "assistant", "message": {"role": "assistant", "content": "ok"}}) + "\n")
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    assert claude_code_hook.main(["turn_end"]) == 0
    manifests = store.list_manifests()
    assert len(manifests) == 2
    second_ref = manifests[1].trajectory_ref
    assert second_ref is not None
    assert second_ref.start_offset > 0
    assert second_ref.record_count == 2


def test_claude_seeds_payload_fields(tmp_path, monkeypatch):
    plugin_home = tmp_path / "plugin"
    cwd = tmp_path / "work"
    cwd.mkdir()
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(plugin_home))
    monkeypatch.setenv("CLAUDE_SESSION_ID", "claude-fields")
    monkeypatch.delenv("CLAUDE_PERMISSION_MODE", raising=False)
    monkeypatch.delenv("CLAUDE_EFFORT", raising=False)
    monkeypatch.delenv("CLAUDE_AGENT_TYPE", raising=False)
    payload = {
        "hook_event_name": "Stop",
        "session_id": "claude-fields",
        "cwd": str(cwd),
        "transcript_path": str(tmp_path / "missing.jsonl"),
        "permission_mode": "plan",
        "effort": {"level": "high"},
        "agent_type": "Explore",
    }
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))
    assert claude_code_hook.main(["turn_end"]) == 0

    store = CheckpointStore(plugin_home / "sessions" / "claude-fields")
    manifest = store.list_manifests()[0]
    env = environment_from_blob(manifest.env_ref, store)
    assert env.permission_mode == "plan"
    assert env.effort == "high"
    assert env.agent_type == "Explore"

