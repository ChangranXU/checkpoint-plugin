import json

from checkpoint_plugin.coordinator import CheckpointCoordinator, TurnRecord
from checkpoint_plugin.resume import ResumeOrchestrator
from checkpoint_plugin.store import CheckpointStore
from checkpoint_plugin.types import TrajectoryReference


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


def test_resume_copies_trajectory_through_target_turn(tmp_path, monkeypatch):
    plugin_home = tmp_path / "plugin"
    cwd = tmp_path / "work"
    cwd.mkdir()
    (cwd / "file.txt").write_text("v1", encoding="utf-8")
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(plugin_home))

    coordinator = CheckpointCoordinator(session_id="s1", cwd=cwd)
    coordinator.on_session_start()
    coordinator.on_turn_end(TurnRecord(user_message="first", assistant_text="one"))
    coordinator.on_turn_end(TurnRecord(user_message="second", assistant_text="two"))

    orchestrator = ResumeOrchestrator(cwd=cwd)
    report = orchestrator.execute(orchestrator.plan("s1", 1), lambda _text: True)

    resumed_store = CheckpointStore(plugin_home / "sessions" / report.new_session_id)
    events = [
        json.loads(line)
        for line in resumed_store.trajectory_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [event["turn_id"] for event in events] == [0, 1]
    assert [event["user_message"] for event in events] == ["first", "second"]


def test_resume_copies_referenced_transcript_slices(tmp_path, monkeypatch):
    plugin_home = tmp_path / "plugin"
    cwd = tmp_path / "work"
    transcript = tmp_path / "provider.jsonl"
    cwd.mkdir()
    (cwd / "file.txt").write_text("v1", encoding="utf-8")
    transcript.write_bytes(b'{"turn_id":"one"}\n{"turn_id":"two"}\n')
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(plugin_home))

    coordinator = CheckpointCoordinator(session_id="s1", cwd=cwd)
    coordinator.on_session_start()
    coordinator.on_turn_end(
        TurnRecord(user_message="first"),
        TrajectoryReference("codex", str(transcript), 0, 18, 1),
    )
    coordinator.on_turn_end(
        TurnRecord(user_message="second"),
        TrajectoryReference("codex", str(transcript), 18, transcript.stat().st_size, 1),
    )

    orchestrator = ResumeOrchestrator(cwd=cwd)
    report = orchestrator.execute(orchestrator.plan("s1", 1), lambda _text: True)

    resumed_store = CheckpointStore(plugin_home / "sessions" / report.new_session_id)
    assert resumed_store.trajectory_path.read_bytes() == transcript.read_bytes()


def test_resume_skips_missing_referenced_transcript(tmp_path, monkeypatch, capsys):
    plugin_home = tmp_path / "plugin"
    cwd = tmp_path / "work"
    missing = tmp_path / "missing.jsonl"
    cwd.mkdir()
    (cwd / "file.txt").write_text("v1", encoding="utf-8")
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(plugin_home))

    coordinator = CheckpointCoordinator(session_id="s1", cwd=cwd)
    coordinator.on_session_start()
    coordinator.on_turn_end(
        TurnRecord(user_message="first"),
        TrajectoryReference("codex", str(missing), 0, 10, 1),
    )

    orchestrator = ResumeOrchestrator(cwd=cwd)
    report = orchestrator.execute(orchestrator.plan("s1", 0), lambda _text: True)

    resumed_store = CheckpointStore(plugin_home / "sessions" / report.new_session_id)
    assert not resumed_store.trajectory_path.exists()
    assert "trajectory unavailable" in capsys.readouterr().err
