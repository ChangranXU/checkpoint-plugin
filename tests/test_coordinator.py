from checkpoint_plugin.coordinator import CheckpointCoordinator, TurnRecord


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
    assert manifest.user_message_preview == "hello"
    assert coordinator.get_checkpoint(0) == manifest
    assert (home / "sessions" / "s1" / "metadata.json").exists()
