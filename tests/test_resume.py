from checkpoint_plugin.coordinator import CheckpointCoordinator, TurnRecord
from checkpoint_plugin.resume import ResumeOrchestrator
from checkpoint_plugin.store import CheckpointStore


def test_resume_diff_backup_and_restore(tmp_path, monkeypatch):
    plugin_home = tmp_path / "plugin"
    cwd = tmp_path / "work"
    cwd.mkdir()
    target_file = cwd / "file.txt"
    target_file.write_text("v1", encoding="utf-8")
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(plugin_home))

    coordinator = CheckpointCoordinator(session_id="s1", cwd=cwd)
    coordinator.on_session_start()
    coordinator.on_turn_end(TurnRecord(user_message="checkpoint"))

    target_file.write_text("v2", encoding="utf-8")
    (cwd / "new.txt").write_text("new", encoding="utf-8")

    orchestrator = ResumeOrchestrator(cwd=cwd)
    plan = orchestrator.plan("s1", 0)
    assert "modified: 1 files" in plan.fs_diff_text

    report = orchestrator.execute(plan, lambda _text: True)

    assert target_file.read_text(encoding="utf-8") == "v1"
    assert not (cwd / "new.txt").exists()
    assert report.new_session_id == "s1-resumed-from-0"
    resumed_store = CheckpointStore(plugin_home / "sessions" / report.new_session_id)
    assert resumed_store.read_manifest(0).session_id == report.new_session_id
    assert (plugin_home / "backups").exists()
