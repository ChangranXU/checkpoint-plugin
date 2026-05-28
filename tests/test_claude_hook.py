import io
import json

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
