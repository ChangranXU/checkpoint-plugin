"""Test OpenCode resume ID ordering and continuation logic."""

import json
import tempfile
from pathlib import Path

import pytest


def test_opencode_resume_generates_time_ordered_message_ids(tmp_path):
    """OpenCode requires time-ordered message IDs for continuation logic.

    OpenCode's session continuation logic (packages/opencode/src/session/prompt.ts)
    uses lexicographic ID comparison to find the "latest" user and assistant messages:

        if (info.role === "user" && (!user || info.id > user.id)) user = info
        if (info.role === "assistant" && (!assistant || info.id > assistant.id)) assistant = info

    If IDs are random (uuid4), the ordering breaks and the loop incorrectly identifies
    an intermediate assistant message as "latest", potentially triggering infinite
    continuation if that message has finish="tool-calls".

    The fix uses uuid7 (time-ordered) instead of uuid4 for message and part IDs.
    """
    from checkpoint_plugin.coordinator import CheckpointCoordinator

    # Create a minimal OpenCode session with 3 turns
    session_dir = tmp_path / "plugin" / "sessions" / "test_oc_session"
    session_dir.mkdir(parents=True)

    coordinator = CheckpointCoordinator(session_id="test_oc_session", cwd=tmp_path)
    coordinator.session_dir = session_dir

    # Simulate 3 turns with different finish reasons
    turns = [
        {"user": "hello", "assistant": "Hi!", "finish": "stop"},
        {"user": "read file.txt", "assistant": "", "finish": "tool-calls"},
        {"user": "", "assistant": "File contains: test", "finish": "stop"},
    ]

    trajectory = []
    for i, turn in enumerate(turns):
        trajectory.append(
            json.dumps({
                "type": "turn",
                "user_message": turn["user"],
                "assistant_text": turn["assistant"],
                "created_ts": f"2026-06-03T10:00:{i:02d}+00:00",
                "metadata": {
                    "hook_payload": {
                        "finish": turn["finish"],
                    }
                }
            })
        )

    (session_dir / "trajectory.jsonl").write_text("\n".join(trajectory))
    (session_dir / "metadata.json").write_text(json.dumps({
        "session_id": "test_oc_session",
        "provider": "opencode",
        "cwd": str(tmp_path),
        "start_ts": "2026-06-03T10:00:00+00:00"
    }))
    (session_dir / "manifests").mkdir()
    (session_dir / "manifests" / "index.json").write_text(json.dumps({"0": "turn_0000.json"}))
    (session_dir / "manifests" / "turn_0000.json").write_text(json.dumps({
        "turn": 0,
        "trajectory_ref": None,
        "cwd": str(tmp_path),
    }))

    # Resume at turn 2 (all 3 turns should be in the import file)
    opencode_home = tmp_path / ".config" / "opencode"
    opencode_home.mkdir(parents=True)

    from checkpoint_plugin.resume import _write_opencode_session

    trajectory_bytes = (session_dir / "trajectory.jsonl").read_bytes()
    import_path = _write_opencode_session(
        opencode_home, tmp_path, "ses_new_resume", trajectory_bytes
    )

    assert import_path is not None
    import_data = json.loads(import_path.read_text())

    # Check that message IDs are in chronological order
    messages = import_data["messages"]
    assert len(messages) > 0

    msg_ids = [m["info"]["id"] for m in messages]
    sorted_ids = sorted(msg_ids)

    # Time-ordered IDs should sort lexicographically in chronological order
    assert msg_ids == sorted_ids, (
        f"Message IDs are not time-ordered!\n"
        f"Original: {msg_ids}\n"
        f"Sorted:   {sorted_ids}\n"
        f"This breaks OpenCode's continuation logic which uses ID comparison."
    )

    # Check that part IDs within each message are also ordered
    for i, msg in enumerate(messages):
        part_ids = [p["id"] for p in msg.get("parts", []) if isinstance(p, dict)]
        sorted_part_ids = sorted(part_ids)
        assert part_ids == sorted_part_ids, (
            f"Part IDs in message {i} are not time-ordered!\n"
            f"Original: {part_ids}\n"
            f"Sorted:   {sorted_part_ids}"
        )


def test_opencode_resume_sets_finish_stop_in_fallback_reconstruction():
    """Reconstructed messages must have finish="stop" to prevent infinite loops.

    When raw_messages are unavailable (pre-fix captures), the fallback
    _reconstruct_opencode_messages() builds minimal message structures.
    These MUST include finish="stop" on assistant messages to signal completion.

    Without finish, or with finish="tool-calls", OpenCode's continuation logic
    will immediately generate a new turn, causing an infinite loop.
    """
    from checkpoint_plugin.resume import _reconstruct_opencode_messages

    records = [
        {
            "type": "turn",
            "user_message": "hello",
            "assistant_text": "Hi! How can I help?",
            "created_ts": "2026-06-03T10:00:00+00:00",
        }
    ]

    messages = _reconstruct_opencode_messages(records, "ses_test")

    assert len(messages) == 2  # user + assistant

    assistant_msg = messages[1]
    assert assistant_msg["info"]["role"] == "assistant"
    assert assistant_msg["info"]["finish"] == "stop", (
        "Reconstructed assistant message must have finish='stop' "
        "to prevent infinite continuation loop"
    )
    assert "completed" in assistant_msg["info"]["time"], (
        "Reconstructed assistant message must have time.completed "
        "to signal the message is complete"
    )


def test_opencode_resume_preserves_parent_id_chain():
    """parentID references must be remapped to maintain message threading."""
    from checkpoint_plugin.resume import _write_opencode_session

    # Create trajectory with raw_messages that have parentID references
    raw_messages = [
        {
            "info": {"id": "msg_orig_001", "sessionID": "ses_orig", "role": "user", "time": {"created": 1000}, "agent": "build"},
            "parts": [{"id": "prt_001", "type": "text", "text": "hello", "messageID": "msg_orig_001", "sessionID": "ses_orig"}]
        },
        {
            "info": {
                "id": "msg_orig_002",
                "sessionID": "ses_orig",
                "role": "assistant",
                "parentID": "msg_orig_001",  # References first message
                "time": {"created": 2000, "completed": 3000},
                "finish": "stop",
                "mode": "build",
                "agent": "build",
                "modelID": "big-pickle",
                "providerID": "opencode",
                "path": {"cwd": "/test", "root": "/"},
                "cost": 0,
                "tokens": {"input": 10, "output": 5, "reasoning": 2, "cache": {"read": 0, "write": 0}}
            },
            "parts": [{"id": "prt_002", "type": "text", "text": "Hi!", "messageID": "msg_orig_002", "sessionID": "ses_orig"}]
        }
    ]

    trajectory = json.dumps({
        "type": "turn",
        "user_message": "hello",
        "assistant_text": "Hi!",
        "metadata": {
            "hook_payload": {
                "raw_messages": raw_messages,
                "session_info": {"id": "ses_orig", "slug": "test", "projectID": "global"}
            }
        }
    }).encode()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmppath = Path(tmpdir)
        opencode_home = tmppath / ".config" / "opencode"
        opencode_home.mkdir(parents=True)

        import_path = _write_opencode_session(
            opencode_home, tmppath, "ses_resumed", trajectory
        )

        import_data = json.loads(import_path.read_text())
        messages = import_data["messages"]

        # IDs should be remapped
        assert messages[0]["info"]["id"] != "msg_orig_001"
        assert messages[1]["info"]["id"] != "msg_orig_002"

        # But parentID reference should still point to the first message's NEW id
        parent_id = messages[1]["info"]["parentID"]
        assert parent_id == messages[0]["info"]["id"], (
            f"parentID chain broken: assistant parentID={parent_id}, "
            f"but user id={messages[0]['info']['id']}"
        )


def test_opencode_resume_preserves_session_info_fields(tmp_path):
    from checkpoint_plugin.resume import _write_opencode_session

    raw_messages = [
        {
            "info": {
                "id": "msg_user",
                "sessionID": "ses_orig",
                "role": "user",
                "time": {"created": 1000},
                "agent": "build",
            },
            "parts": [
                {
                    "id": "prt_user",
                    "type": "text",
                    "text": "hello",
                    "messageID": "msg_user",
                    "sessionID": "ses_orig",
                }
            ],
        },
        {
            "info": {
                "id": "msg_assistant",
                "sessionID": "ses_orig",
                "role": "assistant",
                "parentID": "msg_user",
                "time": {"created": 2000, "completed": 3000},
                "finish": "stop",
                "agent": "build",
                "path": {"cwd": "/old/cwd", "root": "/"},
            },
            "parts": [
                {
                    "id": "prt_assistant",
                    "type": "text",
                    "text": "Hi!",
                    "messageID": "msg_assistant",
                    "sessionID": "ses_orig",
                }
            ],
        },
    ]
    session_info = {
        "id": "ses_orig",
        "parentID": "ses_parent",
        "slug": "original-slug",
        "projectID": "project-old",
        "directory": "/old/cwd",
        "path": "old/path",
        "title": "Original title",
        "agent": "build",
        "model": {"id": "big-pickle", "providerID": "opencode"},
        "permission": [{"permission": "task", "action": "deny", "pattern": "*"}],
        "metadata": {"runtime": "kept"},
        "workspaceID": "workspace-1",
        "share": {"url": "https://example.com/share"},
        "revert": {"messageID": "msg_user"},
        "time": {"created": 1234, "updated": 2345},
    }
    trajectory = (
        json.dumps(
            {
                "type": "turn",
                "user_message": "hello",
                "assistant_text": "Hi!",
                "metadata": {
                    "hook_payload": {
                        "raw_messages": raw_messages,
                        "session_info": session_info,
                    }
                },
            }
        )
        + "\n"
    ).encode()

    opencode_home = tmp_path / ".config" / "opencode"
    opencode_home.mkdir(parents=True)
    import_path = _write_opencode_session(opencode_home, tmp_path, "ses_resumed", trajectory)

    assert import_path is not None
    import_data = json.loads(import_path.read_text())
    info = import_data["info"]
    assert info["id"] == "ses_resumed"
    assert "parentID" not in info
    assert info["directory"] == str(tmp_path)
    assert info["agent"] == "build"
    assert info["model"] == {"id": "big-pickle", "providerID": "opencode"}
    assert info["permission"] == [{"permission": "task", "action": "deny", "pattern": "*"}]
    assert info["metadata"] == {"runtime": "kept"}
    assert info["workspaceID"] == "workspace-1"
    assert info["share"] == {"url": "https://example.com/share"}
    assert info["revert"] == {"messageID": "msg_user"}
    assert info["time"]["created"] == 1234
    assert info["time"]["updated"] >= 1234
    assert all(
        msg["info"].get("path", {}).get("cwd") == str(tmp_path)
        for msg in import_data["messages"]
        if msg["info"].get("path")
    )


def test_opencode_resume_command_carries_runtime_mcp_overlay(tmp_path):
    from checkpoint_plugin.resume import _resume_command
    from checkpoint_plugin.types import EnvironmentState

    target_env = EnvironmentState(
        provider="opencode",
        mcp_servers={"context7": "inactive", "filesystem": "active", "failed_server": "failed"},
        extra={
            "opencode_config_content": json.dumps(
                {
                    "mcp": {"context7": {"enabled": True, "type": "local"}},
                    "model": "opencode/model",
                }
            ),
            "opencode_runtime_env": {
                "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
                "OPENCODE_PERMISSION": json.dumps({"bash": {"*": "ask"}}),
            },
        },
    )
    import_path = tmp_path / "imports" / "ses_resumed.json"

    command = _resume_command("opencode", "ses_resumed", import_path, target_env)

    assert command is not None
    assert "OPENCODE_CONFIG_CONTENT=" in command
    assert '"context7":{"enabled":false' in command
    assert '"filesystem":{"enabled":true}' in command
    assert '"model":"opencode/model"' in command
    assert "failed_server" not in command
    assert "OPENCODE_DISABLE_PROJECT_CONFIG=1" in command
    assert "OPENCODE_PERMISSION=" in command
    assert f"opencode import {import_path}" in command
    assert "opencode --session ses_resumed" in command
