import json
from multiprocessing import Process

from checkpoint_plugin.coordinator import CheckpointCoordinator, TurnRecord
from checkpoint_plugin.store import CheckpointStore
from checkpoint_plugin.types import TrajectoryReference


def test_full_turn_cycle(tmp_path, monkeypatch):
    home = tmp_path / "plugin"
    cwd = tmp_path / "work"
    cwd.mkdir()
    (cwd / "AGENTS.md").write_text("agent", encoding="utf-8")
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(home))

    coordinator = CheckpointCoordinator(session_id="s1", cwd=cwd)
    coordinator.on_session_start()
    manifest = coordinator.on_turn_end(TurnRecord(user_message="hello", assistant_text="hi"))

    assert manifest.turn_id == 0
    assert manifest.trajectory_ref is not None
    assert manifest.user_message_preview == "hello"
    assert coordinator.get_checkpoint(0) == manifest
    assert (home / "sessions" / "s1" / "metadata.json").exists()


def test_metadata_session_title_defaults_to_none(tmp_path):
    home = tmp_path / "plugin"
    cwd = tmp_path / "work"
    cwd.mkdir()
    (cwd / "AGENTS.md").write_text("agent", encoding="utf-8")

    coordinator = CheckpointCoordinator(session_id="s1", cwd=cwd, plugin_home=home)
    coordinator.on_session_start()

    metadata = json.loads((home / "sessions" / "s1" / "metadata.json").read_text())
    assert metadata["session_title"] is None
    assert "model" not in metadata


def test_codex_session_title_is_read_from_session_index(tmp_path, monkeypatch):
    home = tmp_path / "plugin"
    codex_home = tmp_path / ".codex"
    cwd = tmp_path / "work"
    cwd.mkdir()
    codex_home.mkdir()
    (cwd / "AGENTS.md").write_text("agent", encoding="utf-8")
    (codex_home / "session_index.jsonl").write_text(
        '{"id":"other","thread_name":"Other"}\n'
        '{"id":"s1","thread_name":"Respond to greeting"}\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    coordinator = CheckpointCoordinator(session_id="s1", cwd=cwd, plugin_home=home)
    coordinator.on_session_start()

    metadata = json.loads((home / "sessions" / "s1" / "metadata.json").read_text())
    assert metadata["session_title"] == "Respond to greeting"
    assert "model" not in metadata


def test_claude_session_title_is_read_from_transcript_slug(tmp_path, monkeypatch):
    home = tmp_path / "plugin"
    claude_home = tmp_path / ".claude"
    cwd = tmp_path / "work"
    transcript = tmp_path / "claude.jsonl"
    cwd.mkdir()
    claude_home.mkdir()
    (cwd / "CLAUDE.md").write_text("agent", encoding="utf-8")
    transcript.write_text(
        '{"type":"user","message":{"role":"user","content":"hello"},"slug":"complete-environment-configuration-documentation"}\n',
        encoding="utf-8",
    )
    monkeypatch.delenv("CHECKPOINT_PROVIDER", raising=False)
    monkeypatch.delenv("CODEX_SESSION_ID", raising=False)
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setenv("TEST_HOME", str(tmp_path))
    monkeypatch.setenv("CLAUDE_SESSION_ID", "s1")

    coordinator = CheckpointCoordinator(session_id="s1", cwd=cwd, plugin_home=home)
    coordinator.on_session_start()
    coordinator.on_turn_end(
        TurnRecord(),
        TrajectoryReference("claude", str(transcript), 0, transcript.stat().st_size, 1),
    )

    metadata = json.loads((home / "sessions" / "s1" / "metadata.json").read_text())
    assert metadata["session_title"] == "complete-environment-configuration-documentation"


def test_turn_manifest_stores_external_trajectory_reference(tmp_path):
    home = tmp_path / "plugin"
    cwd = tmp_path / "work"
    transcript = tmp_path / "provider.jsonl"
    cwd.mkdir()
    (cwd / "AGENTS.md").write_text("agent", encoding="utf-8")
    transcript.write_text('{"turn_id":"provider-turn-1","message":"hi"}\n', encoding="utf-8")

    coordinator = CheckpointCoordinator(session_id="s1", cwd=cwd, plugin_home=home)
    ref = TrajectoryReference(
        provider="codex",
        transcript_path=str(transcript),
        start_offset=0,
        end_offset=transcript.stat().st_size,
        record_count=1,
    )
    manifest = coordinator.on_turn_end(TurnRecord(assistant_text="done"), ref)

    assert manifest.trajectory_ref == ref
    assert not (home / "sessions" / "s1" / "trajectory.jsonl").exists()


def test_turn_preview_falls_back_to_codex_user_message_event(tmp_path):
    home = tmp_path / "plugin"
    cwd = tmp_path / "work"
    transcript = tmp_path / "codex.jsonl"
    cwd.mkdir()
    (cwd / "AGENTS.md").write_text("agent", encoding="utf-8")
    transcript.write_text(
        '{"type":"event_msg","payload":{"type":"user_message","message":"hello from codex\\n"}}\n',
        encoding="utf-8",
    )

    coordinator = CheckpointCoordinator(session_id="s1", cwd=cwd, plugin_home=home)
    manifest = coordinator.on_turn_end(
        TurnRecord(),
        TrajectoryReference("codex", str(transcript), 0, transcript.stat().st_size, 1),
    )

    assert manifest.user_message_preview == "hello from codex"


def test_turn_preview_falls_back_to_claude_user_record(tmp_path):
    home = tmp_path / "plugin"
    cwd = tmp_path / "work"
    transcript = tmp_path / "claude.jsonl"
    cwd.mkdir()
    (cwd / "CLAUDE.md").write_text("agent", encoding="utf-8")
    transcript.write_text(
        '{"type":"user","message":{"role":"user","content":"hello from claude"}}\n',
        encoding="utf-8",
    )

    coordinator = CheckpointCoordinator(session_id="s1", cwd=cwd, plugin_home=home)
    manifest = coordinator.on_turn_end(
        TurnRecord(),
        TrajectoryReference("claude", str(transcript), 0, transcript.stat().st_size, 1),
    )

    assert manifest.user_message_preview == "hello from claude"


def test_next_turn_closes_previous_reference_range(tmp_path):
    home = tmp_path / "plugin"
    cwd = tmp_path / "work"
    transcript = tmp_path / "provider.jsonl"
    cwd.mkdir()
    (cwd / "AGENTS.md").write_text("agent", encoding="utf-8")
    transcript.write_text(
        '{"turn_id":"one","message":"start"}\n'
        '{"turn_id":"one","message":"late"}\n'
        '{"turn_id":"two","message":"start"}\n',
        encoding="utf-8",
    )
    second_start = transcript.read_bytes().index(b'{"turn_id":"two"')

    coordinator = CheckpointCoordinator(session_id="s1", cwd=cwd, plugin_home=home)
    coordinator.on_turn_end(
        TurnRecord(assistant_text="one"),
        TrajectoryReference("codex", str(transcript), 0, 35, 1),
    )
    coordinator.on_turn_end(
        TurnRecord(assistant_text="two"),
        TrajectoryReference("codex", str(transcript), second_start, transcript.stat().st_size, 1),
    )

    refreshed = CheckpointStore(home / "sessions" / "s1").read_manifest(0)
    assert refreshed.trajectory_ref is not None
    assert refreshed.trajectory_ref.end_offset == second_start
    assert refreshed.trajectory_ref.record_count == 2


def test_concurrent_turn_end_writes_are_serialized(tmp_path):
    home = tmp_path / "plugin"
    cwd = tmp_path / "work"
    cwd.mkdir()
    (cwd / "AGENTS.md").write_text("agent", encoding="utf-8")

    processes = [
        Process(target=_write_turn, args=(home, cwd, "s1", f"message-{idx}"))
        for idx in range(8)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join()

    assert [process.exitcode for process in processes] == [0] * len(processes)

    store = CheckpointStore(home / "sessions" / "s1")
    manifests = store.list_manifests()

    assert [manifest.turn_id for manifest in manifests] == list(range(len(processes)))
    for manifest in manifests:
        assert manifest.trajectory_ref is not None
        assert manifest.trajectory_ref.record_count == 1


def _write_turn(home, cwd, session_id, user_message):
    coordinator = CheckpointCoordinator(session_id=session_id, cwd=cwd, plugin_home=home)
    coordinator.on_turn_end(TurnRecord(user_message=user_message))


def test_resolve_fork_ancestor_transcript_avoids_self_reference(tmp_path):
    """P6-15: a claude resume's transcript_path is the session's OWN file; the
    ancestor must be resolved via forkedFrom.sessionId, never recorded as self."""
    from checkpoint_plugin.coordinator import _resolve_fork_ancestor_transcript

    proj = tmp_path
    own = proj / "SELF.jsonl"
    ancestor = proj / "ANCESTOR.jsonl"
    ancestor.write_text(json.dumps({"type": "user", "sessionId": "ANCESTOR"}) + "\n", encoding="utf-8")
    own.write_text(
        "\n".join(
            [
                json.dumps({"type": "user", "sessionId": "SELF", "forkedFrom": {"sessionId": "ANCESTOR", "messageUuid": "m1"}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # claude self-named file -> resolves to the distinct ancestor sibling.
    resolved = _resolve_fork_ancestor_transcript("claude", str(own), "SELF")
    assert resolved == str(ancestor)

    # No forkedFrom / ancestor missing -> None (drop the self-pointer rather than lie).
    lonely = proj / "LONE.jsonl"
    lonely.write_text(json.dumps({"type": "user", "sessionId": "LONE"}) + "\n", encoding="utf-8")
    assert _resolve_fork_ancestor_transcript("claude", str(lonely), "LONE") is None

    # codex path is the real parent rollout already -> returned verbatim.
    assert _resolve_fork_ancestor_transcript("codex", "/prior/rollout.jsonl", "FORKED") == "/prior/rollout.jsonl"


def test_last_turn_end_offset_reanchored_to_eof_on_next_session_start(tmp_path, monkeypatch):
    """F13: the last turn's end_offset trails EOF when the provider flushes a
    trailing same-turn record after the Stop hook reads the file. There is no
    finalize hook, so the next session_start re-anchors it to the (now fully
    flushed) EOF — provided the tail is same-turn complete."""
    home = tmp_path / "plugin"
    cwd = tmp_path / "work"
    cwd.mkdir()
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(home))
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        json.dumps({"type": "user", "promptId": "p0", "uuid": "u0", "message": {"role": "user", "content": "hi"}}) + "\n"
        + json.dumps({"type": "assistant", "uuid": "a0", "promptId": "p0", "message": {"role": "assistant", "content": "x"}}) + "\n",
        encoding="utf-8",
    )
    c = CheckpointCoordinator(session_id="sx", cwd=cwd)
    c.on_session_start()
    captured = transcript.stat().st_size
    c.on_turn_end(TurnRecord(user_message="hi"), TrajectoryReference("claude", str(transcript), 0, captured, 2))
    assert c.store.read_manifest(0).trajectory_ref.end_offset == captured

    # Provider flushes a trailing same-turn record AFTER the Stop hook.
    with transcript.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "system", "subtype": "stop_hook_summary", "promptId": "p0"}) + "\n")
    grown = transcript.stat().st_size

    CheckpointCoordinator(session_id="sx", cwd=cwd).on_session_start()
    assert c.store.read_manifest(0).trajectory_ref.end_offset == grown


def test_reanchor_does_not_absorb_a_new_turn_tail(tmp_path, monkeypatch):
    """F13 guard: a trailing record bearing a DIFFERENT per-turn key is a new turn
    and must NOT be folded into the last captured turn."""
    home = tmp_path / "plugin"
    cwd = tmp_path / "work"
    cwd.mkdir()
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(home))
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(
        json.dumps({"type": "user", "promptId": "p0", "uuid": "u0", "message": {"role": "user", "content": "hi"}}) + "\n",
        encoding="utf-8",
    )
    c = CheckpointCoordinator(session_id="sy", cwd=cwd)
    c.on_session_start()
    captured = transcript.stat().st_size
    c.on_turn_end(TurnRecord(user_message="hi"), TrajectoryReference("claude", str(transcript), 0, captured, 1))

    # A NEW turn (distinct promptId) lands in the tail.
    with transcript.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "user", "promptId": "p1", "uuid": "u1", "message": {"role": "user", "content": "next"}}) + "\n")

    CheckpointCoordinator(session_id="sy", cwd=cwd).on_session_start()
    assert c.store.read_manifest(0).trajectory_ref.end_offset == captured
