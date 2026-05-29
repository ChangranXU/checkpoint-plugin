import json
import uuid
from pathlib import Path

from checkpoint_plugin.cli import main
from checkpoint_plugin.coordinator import CheckpointCoordinator, TurnRecord
from checkpoint_plugin.paths import load_config, write_config
from checkpoint_plugin.resume import ResumeOptions, ResumeOrchestrator
from checkpoint_plugin.store import CheckpointStore
from checkpoint_plugin.types import TrajectoryReference


def _isolate_provider_env(monkeypatch):
    for name in (
        "CHECKPOINT_PROVIDER",
        "CLAUDE_PROVIDER",
        "CLAUDE_SESSION_ID",
        "CLAUDE_PROJECT_DIR",
        "CODEX_HOME",
        "CODEX_SESSION_ID",
        "ANTHROPIC_MODEL",
        "CLAUDE_MODEL",
        "OPENAI_MODEL",
        "CODEX_MODEL",
        "CLAUDE_PERMISSION_MODE",
        "CODEX_PERMISSION_MODE",
        "CODEX_SANDBOX_MODE",
    ):
        monkeypatch.delenv(name, raising=False)


def test_resume_diff_backup_and_restore(tmp_path, monkeypatch):
    plugin_home = tmp_path / "plugin"
    cwd = tmp_path / "work"
    cwd.mkdir()
    target_file = cwd / "file.txt"
    target_file.write_text("v1", encoding="utf-8")
    _isolate_provider_env(monkeypatch)
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
    uuid.UUID(report.new_session_id)
    resumed_store = CheckpointStore(plugin_home / "sessions" / report.new_session_id)
    assert resumed_store.read_manifest(0).session_id == report.new_session_id
    assert (plugin_home / "backups").exists()


def test_resume_copies_trajectory_through_target_turn(tmp_path, monkeypatch):
    plugin_home = tmp_path / "plugin"
    cwd = tmp_path / "work"
    cwd.mkdir()
    (cwd / "file.txt").write_text("v1", encoding="utf-8")
    _isolate_provider_env(monkeypatch)
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
    _isolate_provider_env(monkeypatch)
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


def test_resume_same_checkpoint_multiple_times_creates_distinct_sessions(tmp_path, monkeypatch):
    plugin_home = tmp_path / "plugin"
    cwd = tmp_path / "work"
    cwd.mkdir()
    (cwd / "file.txt").write_text("v1", encoding="utf-8")
    _isolate_provider_env(monkeypatch)
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(plugin_home))

    coordinator = CheckpointCoordinator(session_id="s1", cwd=cwd)
    coordinator.on_session_start()
    coordinator.on_turn_end(TurnRecord(user_message="checkpoint"))

    orchestrator = ResumeOrchestrator(cwd=cwd)
    first = orchestrator.execute(orchestrator.plan("s1", 0), lambda _text: True)
    second = orchestrator.execute(orchestrator.plan("s1", 0), lambda _text: True)

    assert first.new_session_id != second.new_session_id
    assert (plugin_home / "sessions" / first.new_session_id).exists()
    assert (plugin_home / "sessions" / second.new_session_id).exists()


def test_resume_can_restore_into_new_folder_copy(tmp_path, monkeypatch):
    plugin_home = tmp_path / "plugin"
    cwd = tmp_path / "work"
    copy_cwd = tmp_path / "work-copy"
    cwd.mkdir()
    target_file = cwd / "file.txt"
    target_file.write_text("v1\n", encoding="utf-8")
    _isolate_provider_env(monkeypatch)
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(plugin_home))

    coordinator = CheckpointCoordinator(session_id="s1", cwd=cwd)
    coordinator.on_session_start()
    coordinator.on_turn_end(TurnRecord(user_message="checkpoint"))
    target_file.write_text("v2\n", encoding="utf-8")
    (cwd / "new.txt").write_text("new\n", encoding="utf-8")

    orchestrator = ResumeOrchestrator(cwd=cwd)
    report = orchestrator.execute(
        orchestrator.plan("s1", 0),
        lambda _text: ResumeOptions(proceed=True, target_cwd=copy_cwd),
    )

    assert target_file.read_text(encoding="utf-8") == "v2\n"
    assert (cwd / "new.txt").exists()
    assert (copy_cwd / "file.txt").read_text(encoding="utf-8") == "v1\n"
    assert not (copy_cwd / "new.txt").exists()
    assert report.target_cwd == str(copy_cwd)
    resumed_store = CheckpointStore(plugin_home / "sessions" / report.new_session_id)
    resumed_fs = resumed_store.load_json_blob(resumed_store.read_manifest(0).fs_ref)
    metadata = json.loads((resumed_store.session_dir / "metadata.json").read_text(encoding="utf-8"))
    assert resumed_fs["cwd"] == str(copy_cwd)
    assert metadata["cwd"] == str(copy_cwd)


def test_resume_materializes_codex_native_session(tmp_path, monkeypatch):
    plugin_home = tmp_path / "plugin"
    home = tmp_path / "home"
    codex_home = home / ".codex"
    cwd = tmp_path / "work"
    transcript = tmp_path / "codex.jsonl"
    cwd.mkdir()
    transcript.write_text(
        '\n'.join(
            [
                json.dumps(
                    {
                        "type": "session_meta",
                        "payload": {
                            "id": "old",
                            "cwd": str(cwd),
                            "cli_version": "1.2.3",
                            "model_provider": "test-provider",
                            "base_instructions": {"text": "be helpful"},
                        },
                    }
                ),
                json.dumps({"type": "event_msg", "payload": {"turn_id": "turn-1", "type": "turn_start"}}),
                json.dumps(
                    {
                        "type": "turn_context",
                        "payload": {
                            "type": "turn_context",
                            "turn_id": "turn-1",
                            "model": "old-model",
                            "permission_profile": "old-permission",
                            "sandbox_policy": "workspace-write",
                        },
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (cwd / "file.txt").write_text("v1", encoding="utf-8")
    _isolate_provider_env(monkeypatch)
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(plugin_home))
    monkeypatch.setenv("TEST_HOME", str(home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setenv("CHECKPOINT_PROVIDER", "codex")
    monkeypatch.setenv("CODEX_MODEL", "gpt-target")
    monkeypatch.setenv("CODEX_PERMISSION_MODE", "acceptEdits")

    coordinator = CheckpointCoordinator(session_id="s1", cwd=cwd)
    coordinator.on_session_start()
    coordinator.on_turn_end(
        TurnRecord(user_message="first"),
        TrajectoryReference("codex", str(transcript), 0, transcript.stat().st_size, 2),
    )

    report = ResumeOrchestrator(cwd=cwd).execute(ResumeOrchestrator(cwd=cwd).plan("s1", 0), lambda _text: True)

    assert report.provider_session_path is not None
    provider_path = codex_home / "sessions"
    materialized = list(provider_path.glob("**/*.jsonl"))
    assert [path.as_posix() for path in materialized] == [report.provider_session_path]
    records = [json.loads(line) for line in materialized[0].read_text(encoding="utf-8").splitlines()]
    assert records[0]["payload"]["id"] == report.new_session_id
    assert records[0]["payload"]["originator"] == "Codex Desktop"
    assert records[0]["payload"]["source"] == "vscode"
    assert records[0]["payload"]["thread_source"] == "user"
    assert records[0]["payload"]["cli_version"] == "1.2.3"
    assert records[0]["payload"]["model_provider"] == "test-provider"
    assert records[0]["payload"]["base_instructions"] == {"text": "be helpful"}
    assert records[2]["payload"]["model"] == "gpt-target"
    assert records[2]["payload"]["permission_profile"] == "acceptEdits"
    # Permission mode must not bleed into the sandbox policy (B2): it stays as
    # whatever the original transcript recorded.
    assert records[2]["payload"]["sandbox_policy"] == "workspace-write"
    # M5: the resumed codex session is registered in the picker index.
    index_path = codex_home / "session_index.jsonl"
    assert index_path.exists()
    index_ids = [
        json.loads(line)["id"]
        for line in index_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert report.new_session_id in index_ids

    resumed_store = CheckpointStore(plugin_home / "sessions" / report.new_session_id)
    manifest = resumed_store.read_manifest(0)
    assert manifest.trajectory_ref is not None
    assert manifest.trajectory_ref.transcript_path == report.provider_session_path
    metadata = json.loads((resumed_store.session_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["session_id"] == report.new_session_id
    assert metadata["resumed_from_session_id"] == "s1"


def test_resume_copy_materializes_codex_session_with_copy_cwd(tmp_path, monkeypatch):
    plugin_home = tmp_path / "plugin"
    home = tmp_path / "home"
    codex_home = home / ".codex"
    cwd = tmp_path / "work"
    copy_cwd = tmp_path / "work-copy"
    transcript = tmp_path / "codex.jsonl"
    cwd.mkdir()
    transcript.write_text(
        '\n'.join(
            [
                json.dumps({"type": "session_meta", "payload": {"id": "old", "cwd": str(cwd)}}),
                json.dumps({"type": "event_msg", "payload": {"turn_id": "turn-1", "type": "turn_start"}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (cwd / "file.txt").write_text("v1", encoding="utf-8")
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(plugin_home))
    monkeypatch.setenv("TEST_HOME", str(home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setenv("CHECKPOINT_PROVIDER", "codex")

    coordinator = CheckpointCoordinator(session_id="s1", cwd=cwd)
    coordinator.on_session_start()
    coordinator.on_turn_end(
        TurnRecord(user_message="first"),
        TrajectoryReference("codex", str(transcript), 0, transcript.stat().st_size, 2),
    )

    orchestrator = ResumeOrchestrator(cwd=cwd)
    report = orchestrator.execute(
        orchestrator.plan("s1", 0),
        lambda _text: ResumeOptions(proceed=True, target_cwd=copy_cwd),
    )

    assert report.provider_session_path is not None
    records = [
        json.loads(line)
        for line in Path(report.provider_session_path).read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["payload"]["cwd"] == str(copy_cwd)


def test_resume_materializes_codex_session_meta_for_sliced_trajectory(tmp_path, monkeypatch):
    plugin_home = tmp_path / "plugin"
    home = tmp_path / "home"
    codex_home = home / ".codex"
    cwd = tmp_path / "work"
    copy_cwd = tmp_path / "work-copy"
    transcript = tmp_path / "codex.jsonl"
    cwd.mkdir()
    prefix = json.dumps({"type": "session_meta", "payload": {"id": "old", "cwd": str(cwd)}}) + "\n"
    suffix = json.dumps({"type": "event_msg", "payload": {"turn_id": "turn-1", "type": "turn_start"}}) + "\n"
    transcript.write_text(prefix + suffix, encoding="utf-8")
    (cwd / "file.txt").write_text("v1", encoding="utf-8")
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(plugin_home))
    monkeypatch.setenv("TEST_HOME", str(home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setenv("CHECKPOINT_PROVIDER", "codex")

    coordinator = CheckpointCoordinator(session_id="s1", cwd=cwd)
    coordinator.on_session_start()
    coordinator.on_turn_end(
        TurnRecord(user_message="first"),
        TrajectoryReference("codex", str(transcript), len(prefix.encode("utf-8")), transcript.stat().st_size, 1),
    )

    orchestrator = ResumeOrchestrator(cwd=cwd)
    report = orchestrator.execute(
        orchestrator.plan("s1", 0),
        lambda _text: ResumeOptions(proceed=True, target_cwd=copy_cwd),
    )

    assert report.provider_session_path is not None
    records = [
        json.loads(line)
        for line in Path(report.provider_session_path).read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["type"] == "session_meta"
    assert records[0]["payload"]["id"] == report.new_session_id
    assert records[0]["payload"]["cwd"] == str(copy_cwd)
    assert records[0]["payload"]["originator"] == "Codex Desktop"
    assert records[0]["payload"]["source"] == "vscode"
    assert records[0]["payload"]["thread_source"] == "user"
    assert records[1]["type"] == "event_msg"


def test_resume_materializes_claude_native_session(tmp_path, monkeypatch):
    plugin_home = tmp_path / "plugin"
    home = tmp_path / "home"
    claude_home = home / ".claude"
    cwd = tmp_path / "work"
    transcript = tmp_path / "claude.jsonl"
    cwd.mkdir()
    transcript.write_text(
        '\n'.join(
            [
                json.dumps({"type": "permission-mode", "sessionId": "old", "permissionMode": "default"}),
                json.dumps(
                    {
                        "type": "user",
                        "sessionId": "old",
                        "uuid": "old-user",
                        "parentUuid": None,
                        "cwd": "/old",
                        "message": {"role": "user", "content": "hi"},
                    }
                ),
                json.dumps(
                    {
                        "type": "assistant",
                        "sessionId": "old",
                        "uuid": "old-assistant",
                        "parentUuid": "old-user",
                        "cwd": "/old",
                        "message": {"role": "assistant", "content": []},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (cwd / "file.txt").write_text("v1", encoding="utf-8")
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(plugin_home))
    monkeypatch.setenv("TEST_HOME", str(home))
    monkeypatch.setenv("CHECKPOINT_PROVIDER", "claude")
    monkeypatch.setenv("CLAUDE_MODEL", "sonnet-target")
    monkeypatch.setenv("CLAUDE_PERMISSION_MODE", "acceptEdits")

    coordinator = CheckpointCoordinator(session_id="s1", cwd=cwd)
    coordinator.on_session_start()
    coordinator.on_turn_end(
        TurnRecord(user_message="first"),
        TrajectoryReference("claude", str(transcript), 0, transcript.stat().st_size, 3),
    )

    report = ResumeOrchestrator(cwd=cwd).execute(ResumeOrchestrator(cwd=cwd).plan("s1", 0), lambda _text: True)

    assert report.provider_session_path is not None
    materialized = claude_home / "projects" / str(cwd).replace("/", "-") / f"{report.new_session_id}.jsonl"
    assert str(materialized) == report.provider_session_path
    records = [json.loads(line) for line in materialized.read_text(encoding="utf-8").splitlines()]
    assert {record["sessionId"] for record in records} == {report.new_session_id}
    assert records[0]["permissionMode"] == "acceptEdits"
    assert records[1]["cwd"] == str(cwd)
    assert records[2]["parentUuid"] == records[1]["uuid"]


def test_resume_restores_environment_with_target_provider_layout(tmp_path, monkeypatch):
    plugin_home = tmp_path / "plugin"
    home = tmp_path / "home"
    claude_home = home / ".claude"
    codex_home = home / ".codex"
    cwd = tmp_path / "work"
    cwd.mkdir()
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(plugin_home))
    monkeypatch.setenv("TEST_HOME", str(home))
    monkeypatch.setenv("CHECKPOINT_PROVIDER", "claude")

    (claude_home / "skills" / "skill-a").mkdir(parents=True)
    (claude_home / "skills" / "skill-a" / "SKILL.md").write_text("claude skill", encoding="utf-8")
    (claude_home / "settings.json").write_text('{"target": true}', encoding="utf-8")
    (cwd / "file.txt").write_text("v1", encoding="utf-8")

    coordinator = CheckpointCoordinator(session_id="s1", cwd=cwd)
    coordinator.on_session_start()
    coordinator.on_turn_end(TurnRecord(user_message="checkpoint"))

    (claude_home / "skills" / "skill-a" / "SKILL.md").write_text("changed", encoding="utf-8")
    (codex_home / "skills" / "codex-only").mkdir(parents=True)
    (codex_home / "skills" / "codex-only" / "SKILL.md").write_text("do not delete", encoding="utf-8")
    monkeypatch.setenv("CHECKPOINT_PROVIDER", "codex")
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    ResumeOrchestrator(cwd=cwd).execute(ResumeOrchestrator(cwd=cwd).plan("s1", 0), lambda _text: True)

    assert (claude_home / "skills" / "skill-a" / "SKILL.md").read_text(encoding="utf-8") == "claude skill"
    assert (claude_home / "settings.json").read_text(encoding="utf-8") == '{"target": true}'
    assert (codex_home / "skills" / "codex-only" / "SKILL.md").read_text(encoding="utf-8") == "do not delete"


def test_resume_reports_only_environment_files_that_changed(tmp_path, monkeypatch):
    plugin_home = tmp_path / "plugin"
    home = tmp_path / "home"
    codex_home = home / ".codex"
    cwd = tmp_path / "work"
    cwd.mkdir()
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(plugin_home))
    monkeypatch.setenv("TEST_HOME", str(home))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setenv("CHECKPOINT_PROVIDER", "codex")

    codex_home.mkdir(parents=True)
    (codex_home / "config.toml").write_text("model = 'old'\n", encoding="utf-8")
    (codex_home / "auth.json").write_text('{"token":"same"}\n', encoding="utf-8")
    (cwd / "README.md").write_text("v1\n", encoding="utf-8")
    coordinator = CheckpointCoordinator(session_id="s1", cwd=cwd)
    coordinator.on_session_start()
    coordinator.on_turn_end(TurnRecord(user_message="checkpoint"))

    (codex_home / "config.toml").write_text("model = 'new'\n", encoding="utf-8")
    (cwd / "README.md").write_text("v2\n", encoding="utf-8")

    orchestrator = ResumeOrchestrator(cwd=cwd)
    report = orchestrator.execute(orchestrator.plan("s1", 0), lambda _text: True)

    assert sorted(Path(path).name for path in report.env.changed) == ["config.toml"]
    assert sorted(Path(path).name for path in report.fs.changed) == ["README.md"]
    assert len(report.changed_files) == 2


def test_resume_plan_diffs_environment_with_target_provider_layout(tmp_path, monkeypatch):
    plugin_home = tmp_path / "plugin"
    home = tmp_path / "home"
    codex_home = home / ".codex"
    cwd = tmp_path / "work"
    cwd.mkdir()
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(plugin_home))
    monkeypatch.setenv("TEST_HOME", str(home))
    monkeypatch.setenv("CHECKPOINT_PROVIDER", "codex")
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    (codex_home / "skills" / "codex-skill").mkdir(parents=True)
    (codex_home / "skills" / "codex-skill" / "SKILL.md").write_text("skill", encoding="utf-8")
    (codex_home / "config.toml").write_text(
        """
[plugins."hugging-face@openai-curated"]
enabled = true
""",
        encoding="utf-8",
    )
    (cwd / "file.txt").write_text("v1", encoding="utf-8")

    coordinator = CheckpointCoordinator(session_id="s1", cwd=cwd)
    coordinator.on_session_start()
    coordinator.on_turn_end(TurnRecord(user_message="checkpoint"))

    monkeypatch.setenv("CHECKPOINT_PROVIDER", "claude")

    plan = ResumeOrchestrator(cwd=cwd).plan("s1", 0)

    assert "Provider: claude -> codex" not in plan.env_diff_text
    assert "Skills" not in plan.env_diff_text
    assert "Plugin status" not in plan.env_diff_text


def test_cli_resume_cancel_returns_without_traceback(tmp_path, monkeypatch, capsys):
    plugin_home = tmp_path / "plugin"
    cwd = tmp_path / "work"
    cwd.mkdir()
    _isolate_provider_env(monkeypatch)
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(plugin_home))
    monkeypatch.chdir(cwd)

    (cwd / "file.txt").write_text("v1", encoding="utf-8")
    coordinator = CheckpointCoordinator(session_id="s1", cwd=cwd)
    coordinator.on_session_start()
    coordinator.on_turn_end(TurnRecord(user_message="checkpoint"))
    (cwd / "file.txt").write_text("v2", encoding="utf-8")
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    assert main(["resume", "s1", "0"]) == 1
    captured = capsys.readouterr()
    assert "Resume cancelled" in captured.err
    assert "Traceback" not in captured.err
    assert (cwd / "file.txt").read_text(encoding="utf-8") == "v2"


def test_cli_resume_can_show_file_diff_then_restore(tmp_path, monkeypatch, capsys):
    plugin_home = tmp_path / "plugin"
    cwd = tmp_path / "work"
    cwd.mkdir()
    _isolate_provider_env(monkeypatch)
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(plugin_home))
    monkeypatch.chdir(cwd)

    target_file = cwd / "file.txt"
    target_file.write_text("v1\n", encoding="utf-8")
    coordinator = CheckpointCoordinator(session_id="s1", cwd=cwd)
    coordinator.on_session_start()
    coordinator.on_turn_end(TurnRecord(user_message="checkpoint"))
    target_file.write_text("v2\n", encoding="utf-8")

    answers = iter(["d", "1", "q", "y", "i"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    assert main(["resume", "s1", "0"]) == 0
    captured = capsys.readouterr()
    assert captured.out.count("Resume: session s1, turn 0") == 1
    assert "Detailed resume changes:" in captured.out
    assert "Filesystem:" in captured.out
    assert "--- current/file.txt" in captured.out
    assert "+++ checkpoint/file.txt" in captured.out
    assert "-v2" in captured.out
    assert "+v1" in captured.out
    assert target_file.read_text(encoding="utf-8") == "v1\n"


def test_cli_resume_prints_resume_command_hint(tmp_path, monkeypatch, capsys):
    """P4-6: the CLI must surface the `claude --resume <id>` hint and the
    materialized provider session path (previously built but never printed)."""
    plugin_home = tmp_path / "plugin"
    home = tmp_path / "home"
    cwd = tmp_path / "work"
    transcript = tmp_path / "claude.jsonl"
    cwd.mkdir()
    transcript.write_text(
        json.dumps({"type": "permission-mode", "sessionId": "old", "permissionMode": "default"}) + "\n"
        + json.dumps({"type": "user", "sessionId": "old", "uuid": "u1", "parentUuid": None, "promptId": "p", "cwd": "/old", "message": {"role": "user", "content": "hi"}}) + "\n",
        encoding="utf-8",
    )
    (cwd / "file.txt").write_text("v1", encoding="utf-8")
    _isolate_provider_env(monkeypatch)
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(plugin_home))
    monkeypatch.setenv("TEST_HOME", str(home))
    monkeypatch.setenv("CHECKPOINT_PROVIDER", "claude")
    monkeypatch.chdir(cwd)

    coordinator = CheckpointCoordinator(session_id="s1", cwd=cwd)
    coordinator.on_session_start()
    coordinator.on_turn_end(
        TurnRecord(user_message="first"),
        TrajectoryReference("claude", str(transcript), 0, transcript.stat().st_size, 2),
    )

    assert main(["resume", "s1", "0", "--yes"]) == 0
    out = capsys.readouterr().out
    assert "Resume with: claude --resume " in out
    assert "Provider session:" in out


def test_cli_resume_can_show_file_diff_then_cancel(tmp_path, monkeypatch, capsys):
    plugin_home = tmp_path / "plugin"
    cwd = tmp_path / "work"
    cwd.mkdir()
    _isolate_provider_env(monkeypatch)
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(plugin_home))
    monkeypatch.chdir(cwd)

    target_file = cwd / "file.txt"
    target_file.write_text("v1\n", encoding="utf-8")
    coordinator = CheckpointCoordinator(session_id="s1", cwd=cwd)
    coordinator.on_session_start()
    coordinator.on_turn_end(TurnRecord(user_message="checkpoint"))
    target_file.write_text("v2\n", encoding="utf-8")

    answers = iter(["d", "q", "n"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    assert main(["resume", "s1", "0"]) == 1
    captured = capsys.readouterr()
    assert captured.out.count("Resume: session s1, turn 0") == 1
    assert "Detailed resume changes:" in captured.out
    assert "Filesystem:" in captured.out
    assert "Resume cancelled" in captured.err
    assert target_file.read_text(encoding="utf-8") == "v2\n"


def test_cli_resume_defaults_to_checkpoint_cwd(tmp_path, monkeypatch, capsys):
    plugin_home = tmp_path / "plugin"
    checkpoint_cwd = tmp_path / "checkpoint-work"
    other_cwd = tmp_path / "other-work"
    checkpoint_cwd.mkdir()
    other_cwd.mkdir()
    _isolate_provider_env(monkeypatch)
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(plugin_home))

    target_file = checkpoint_cwd / "file.txt"
    target_file.write_text("v1\n", encoding="utf-8")
    (other_cwd / "unrelated.txt").write_text("unrelated\n", encoding="utf-8")
    coordinator = CheckpointCoordinator(session_id="s1", cwd=checkpoint_cwd)
    coordinator.on_session_start()
    coordinator.on_turn_end(TurnRecord(user_message="checkpoint"))
    target_file.write_text("v2\n", encoding="utf-8")
    monkeypatch.chdir(other_cwd)
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")

    assert main(["resume", "s1", "0"]) == 1
    captured = capsys.readouterr()
    assert f"Filesystem (cwd: {checkpoint_cwd})" in captured.out
    assert "modified: 1 files" in captured.out
    assert "deleted: 0 files" in captured.out
    assert "unrelated.txt" not in captured.out


def test_cli_resume_can_restore_into_chosen_copy(tmp_path, monkeypatch):
    plugin_home = tmp_path / "plugin"
    cwd = tmp_path / "work"
    copy_cwd = tmp_path / "chosen-copy"
    cwd.mkdir()
    _isolate_provider_env(monkeypatch)
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(plugin_home))
    monkeypatch.chdir(cwd)

    target_file = cwd / "file.txt"
    target_file.write_text("v1\n", encoding="utf-8")
    coordinator = CheckpointCoordinator(session_id="s1", cwd=cwd)
    coordinator.on_session_start()
    coordinator.on_turn_end(TurnRecord(user_message="checkpoint"))
    target_file.write_text("v2\n", encoding="utf-8")

    answers = iter(["y", "c", str(copy_cwd)])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    assert main(["resume", "s1", "0"]) == 0
    assert target_file.read_text(encoding="utf-8") == "v2\n"
    assert (copy_cwd / "file.txt").read_text(encoding="utf-8") == "v1\n"


def test_cli_resume_diff_viewer_includes_environment_changes(tmp_path, monkeypatch, capsys):
    plugin_home = tmp_path / "plugin"
    home = tmp_path / "home"
    codex_home = home / ".codex"
    cwd = tmp_path / "work"
    cwd.mkdir()
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(plugin_home))
    monkeypatch.setenv("TEST_HOME", str(home))
    monkeypatch.setenv("CHECKPOINT_PROVIDER", "codex")
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.chdir(cwd)

    (codex_home / "config.toml").parent.mkdir(parents=True)
    (codex_home / "config.toml").write_text("model = 'old'\n", encoding="utf-8")
    (cwd / "file.txt").write_text("v1\n", encoding="utf-8")
    coordinator = CheckpointCoordinator(session_id="s1", cwd=cwd)
    coordinator.on_session_start()
    coordinator.on_turn_end(TurnRecord(user_message="checkpoint"))

    (codex_home / "config.toml").write_text("model = 'new'\n", encoding="utf-8")
    (cwd / "file.txt").write_text("v2\n", encoding="utf-8")
    answers = iter(["d", "3", "q", "n"])
    monkeypatch.setattr("builtins.input", lambda _prompt: next(answers))

    assert main(["resume", "s1", "0"]) == 1
    captured = capsys.readouterr()
    assert "Environment:" in captured.out
    assert "  Settings (1 changes):" in captured.out
    assert "    ~ config.toml" in captured.out
    assert "Filesystem:" in captured.out
    assert "  ~ file.txt" in captured.out
    assert "--- current/environment/Settings/config.toml" in captured.out
    assert "+++ checkpoint/environment/Settings/config.toml" in captured.out
    assert "-model = 'new'" in captured.out
    assert "+model = 'old'" in captured.out


def test_resume_skips_missing_referenced_transcript(tmp_path, monkeypatch, capsys):
    plugin_home = tmp_path / "plugin"
    cwd = tmp_path / "work"
    missing = tmp_path / "missing.jsonl"
    cwd.mkdir()
    (cwd / "file.txt").write_text("v1", encoding="utf-8")
    _isolate_provider_env(monkeypatch)
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


def test_resume_empty_trajectory_ref_does_not_crash(tmp_path, monkeypatch, capsys):
    """P4-1: a checkpoint with an empty trajectory_ref (e.g. a subagent with no
    sidechain file) must resume without crashing. Empty path resolves to '.'
    (a directory), which previously raised IsADirectoryError out of execute()."""
    plugin_home = tmp_path / "plugin"
    cwd = tmp_path / "work"
    cwd.mkdir()
    (cwd / "file.txt").write_text("v1", encoding="utf-8")
    _isolate_provider_env(monkeypatch)
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(plugin_home))

    coordinator = CheckpointCoordinator(session_id="s1", cwd=cwd)
    coordinator.on_session_start()
    coordinator.on_turn_end(
        TurnRecord(user_message="first"),
        TrajectoryReference("claude", "", 0, 0, 0),
    )

    orchestrator = ResumeOrchestrator(cwd=cwd)
    report = orchestrator.execute(orchestrator.plan("s1", 0), lambda _text: True)

    # Resume completes; no provider session is materialized (no trajectory bytes).
    assert report.new_session_id
    assert report.provider_session_path is None
    assert "trajectory unavailable" in capsys.readouterr().err


def _settings_without_plugin_hooks() -> str:
    return json.dumps({"hooks": {}, "model": "sonnet"}, indent=2, sort_keys=True) + "\n"


def _settings_with_plugin_hooks() -> str:
    return (
        json.dumps(
            {
                "hooks": {
                    "Stop": [
                        {
                            "hooks": [
                                {
                                    "type": "command",
                                    "command": "/usr/bin/python3 -m checkpoint_plugin.integrations.claude_code_hook turn_end",
                                }
                            ]
                        }
                    ]
                },
                "model": "sonnet",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def test_resume_keeps_freshly_installed_plugin_hooks(tmp_path, monkeypatch):
    plugin_home = tmp_path / "plugin"
    home = tmp_path / "home"
    claude_home = home / ".claude"
    cwd = tmp_path / "work"
    cwd.mkdir()
    claude_home.mkdir(parents=True)
    _isolate_provider_env(monkeypatch)
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(plugin_home))
    monkeypatch.setenv("TEST_HOME", str(home))
    monkeypatch.setenv("CHECKPOINT_PROVIDER", "claude")

    (claude_home / "settings.json").write_text(_settings_without_plugin_hooks(), encoding="utf-8")

    coordinator = CheckpointCoordinator(session_id="s1", cwd=cwd)
    coordinator.on_session_start()
    coordinator.on_turn_end(TurnRecord(user_message="checkpoint"))

    (claude_home / "settings.json").write_text(_settings_with_plugin_hooks(), encoding="utf-8")

    orchestrator = ResumeOrchestrator(cwd=cwd)
    plan = orchestrator.plan("s1", 0)
    assert "Settings" not in plan.env_diff_text

    orchestrator.execute(plan, lambda _text: True)

    after = (claude_home / "settings.json").read_text(encoding="utf-8")
    parsed = json.loads(after)
    assert parsed["model"] == "sonnet"
    commands = [
        hook["command"]
        for entry in parsed["hooks"].get("Stop", [])
        for hook in entry["hooks"]
    ]
    assert any("checkpoint_plugin.integrations" in c for c in commands)


def test_resume_does_not_reinstall_uninstalled_plugin_hooks(tmp_path, monkeypatch):
    plugin_home = tmp_path / "plugin"
    home = tmp_path / "home"
    claude_home = home / ".claude"
    cwd = tmp_path / "work"
    cwd.mkdir()
    claude_home.mkdir(parents=True)
    _isolate_provider_env(monkeypatch)
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(plugin_home))
    monkeypatch.setenv("TEST_HOME", str(home))
    monkeypatch.setenv("CHECKPOINT_PROVIDER", "claude")

    (claude_home / "settings.json").write_text(_settings_with_plugin_hooks(), encoding="utf-8")

    coordinator = CheckpointCoordinator(session_id="s1", cwd=cwd)
    coordinator.on_session_start()
    coordinator.on_turn_end(TurnRecord(user_message="checkpoint"))

    (claude_home / "settings.json").write_text(_settings_without_plugin_hooks(), encoding="utf-8")

    orchestrator = ResumeOrchestrator(cwd=cwd)
    plan = orchestrator.plan("s1", 0)
    assert "Settings" not in plan.env_diff_text

    orchestrator.execute(plan, lambda _text: True)

    after = json.loads((claude_home / "settings.json").read_text(encoding="utf-8"))
    assert after["model"] == "sonnet"
    assert after["hooks"] == {}


def test_resume_reverts_plugin_hooks_when_flag_disabled(tmp_path, monkeypatch):
    plugin_home = tmp_path / "plugin"
    home = tmp_path / "home"
    claude_home = home / ".claude"
    cwd = tmp_path / "work"
    cwd.mkdir()
    claude_home.mkdir(parents=True)
    _isolate_provider_env(monkeypatch)
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(plugin_home))
    monkeypatch.setenv("TEST_HOME", str(home))
    monkeypatch.setenv("CHECKPOINT_PROVIDER", "claude")

    (claude_home / "settings.json").write_text(_settings_without_plugin_hooks(), encoding="utf-8")

    coordinator = CheckpointCoordinator(session_id="s1", cwd=cwd)
    coordinator.on_session_start()
    coordinator.on_turn_end(TurnRecord(user_message="checkpoint"))

    (claude_home / "settings.json").write_text(_settings_with_plugin_hooks(), encoding="utf-8")

    config = load_config()
    config["ignore_plugin_hook_diffs"] = False
    write_config(config)

    orchestrator = ResumeOrchestrator(cwd=cwd)
    plan = orchestrator.plan("s1", 0)
    assert "Settings" in plan.env_diff_text

    orchestrator.execute(plan, lambda _text: True)

    after = json.loads((claude_home / "settings.json").read_text(encoding="utf-8"))
    assert after["hooks"] == {}


def _seed_claude_session_for_resume(
    plugin_home, home, cwd, transcript, *, transcript_text, file_history=None, todos=None
):
    claude_home = home / ".claude"
    cwd.mkdir(exist_ok=True)
    transcript.write_text(transcript_text, encoding="utf-8")
    (cwd / "file.txt").write_text("v1", encoding="utf-8")
    if file_history:
        history_dir = claude_home / "file-history" / "s1"
        history_dir.mkdir(parents=True, exist_ok=True)
        for name, content in file_history.items():
            (history_dir / name).write_text(content, encoding="utf-8")
    if todos:
        todos_dir = claude_home / "todos"
        todos_dir.mkdir(parents=True, exist_ok=True)
        for suffix, content in todos.items():
            (todos_dir / f"s1-{suffix}").write_text(content, encoding="utf-8")
    coordinator = CheckpointCoordinator(session_id="s1", cwd=cwd)
    coordinator.on_session_start()
    coordinator.on_turn_end(
        TurnRecord(user_message="hi"),
        TrajectoryReference("claude", str(transcript), 0, transcript.stat().st_size, 0),
    )


def test_resume_extends_latest_turn_to_eof_when_tail_is_complete(tmp_path, monkeypatch):
    plugin_home = tmp_path / "plugin"
    home = tmp_path / "home"
    cwd = tmp_path / "work"
    transcript = tmp_path / "claude.jsonl"
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(plugin_home))
    monkeypatch.setenv("TEST_HOME", str(home))
    monkeypatch.setenv("CHECKPOINT_PROVIDER", "claude")

    captured = (
        json.dumps({"type": "user", "promptId": "p-1", "uuid": "u1", "parentUuid": None,
                    "message": {"role": "user", "content": "hi"}}) + "\n"
        + json.dumps({"type": "assistant", "uuid": "a1", "parentUuid": "u1",
                      "message": {"role": "assistant", "content": []}}) + "\n"
    )
    _seed_claude_session_for_resume(plugin_home, home, cwd, transcript, transcript_text=captured)

    # Simulate the trailing flush: same promptId records, complete lines.
    with transcript.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "system", "subtype": "stop_hook_summary"}) + "\n")
        handle.write(json.dumps({"type": "system", "subtype": "turn_duration", "durationMs": 12}) + "\n")

    report = ResumeOrchestrator(cwd=cwd).execute(
        ResumeOrchestrator(cwd=cwd).plan("s1", 0), lambda _text: True
    )
    materialized = Path(report.provider_session_path)
    records = [json.loads(line) for line in materialized.read_text(encoding="utf-8").splitlines()]
    subtypes = [record.get("subtype") for record in records if record.get("type") == "system"]
    assert "stop_hook_summary" in subtypes
    assert "turn_duration" in subtypes


def test_resume_does_not_extend_when_tail_starts_new_turn(tmp_path, monkeypatch):
    plugin_home = tmp_path / "plugin"
    home = tmp_path / "home"
    cwd = tmp_path / "work"
    transcript = tmp_path / "claude.jsonl"
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(plugin_home))
    monkeypatch.setenv("TEST_HOME", str(home))
    monkeypatch.setenv("CHECKPOINT_PROVIDER", "claude")

    captured = (
        json.dumps({"type": "user", "promptId": "p-1", "uuid": "u1", "parentUuid": None,
                    "message": {"role": "user", "content": "hi"}}) + "\n"
        + json.dumps({"type": "assistant", "uuid": "a1", "parentUuid": "u1",
                      "message": {"role": "assistant", "content": []}}) + "\n"
    )
    _seed_claude_session_for_resume(plugin_home, home, cwd, transcript, transcript_text=captured)

    # User raced ahead and started turn 2 before resume fired.
    with transcript.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "user", "promptId": "p-2", "uuid": "u2", "parentUuid": "a1",
                                 "message": {"role": "user", "content": "next"}}) + "\n")

    report = ResumeOrchestrator(cwd=cwd).execute(
        ResumeOrchestrator(cwd=cwd).plan("s1", 0), lambda _text: True
    )
    materialized = Path(report.provider_session_path)
    prompt_ids = [
        record.get("promptId")
        for record in (json.loads(line) for line in materialized.read_text(encoding="utf-8").splitlines())
        if record.get("promptId") is not None
    ]
    assert "p-2" not in prompt_ids


def test_resume_hardlinks_file_history_and_todos(tmp_path, monkeypatch):
    plugin_home = tmp_path / "plugin"
    home = tmp_path / "home"
    cwd = tmp_path / "work"
    transcript = tmp_path / "claude.jsonl"
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(plugin_home))
    monkeypatch.setenv("TEST_HOME", str(home))
    monkeypatch.setenv("CHECKPOINT_PROVIDER", "claude")

    captured = json.dumps({"type": "user", "promptId": "p-1", "uuid": "u1", "parentUuid": None,
                           "message": {"role": "user", "content": "hi"}}) + "\n"
    _seed_claude_session_for_resume(
        plugin_home,
        home,
        cwd,
        transcript,
        transcript_text=captured,
        file_history={"006a1ba@v1": "snapshot-bytes"},
        todos={"agent-x.json": '{"items": []}'},
    )

    report = ResumeOrchestrator(cwd=cwd).execute(
        ResumeOrchestrator(cwd=cwd).plan("s1", 0), lambda _text: True
    )

    src_history = home / ".claude" / "file-history" / "s1" / "006a1ba@v1"
    dst_history = home / ".claude" / "file-history" / report.new_session_id / "006a1ba@v1"
    assert dst_history.exists()
    assert src_history.stat().st_ino == dst_history.stat().st_ino  # hardlink, not copy

    dst_todo = home / ".claude" / "todos" / f"{report.new_session_id}-agent-x.json"
    src_todo = home / ".claude" / "todos" / "s1-agent-x.json"
    assert dst_todo.exists()
    assert src_todo.stat().st_ino == dst_todo.stat().st_ino


def test_resume_command_is_set_in_report(tmp_path, monkeypatch):
    plugin_home = tmp_path / "plugin"
    home = tmp_path / "home"
    cwd = tmp_path / "work"
    transcript = tmp_path / "claude.jsonl"
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(plugin_home))
    monkeypatch.setenv("TEST_HOME", str(home))
    monkeypatch.setenv("CHECKPOINT_PROVIDER", "claude")

    captured = json.dumps({"type": "user", "promptId": "p-1", "uuid": "u1", "parentUuid": None,
                           "message": {"role": "user", "content": "hi"}}) + "\n"
    _seed_claude_session_for_resume(plugin_home, home, cwd, transcript, transcript_text=captured)

    report = ResumeOrchestrator(cwd=cwd).execute(
        ResumeOrchestrator(cwd=cwd).plan("s1", 0), lambda _text: True
    )
    assert report.resume_command == f"claude --resume {report.new_session_id}"


def test_resume_parent_uuid_chain_skips_summary_records(tmp_path, monkeypatch):
    plugin_home = tmp_path / "plugin"
    home = tmp_path / "home"
    cwd = tmp_path / "work"
    transcript = tmp_path / "claude.jsonl"
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(plugin_home))
    monkeypatch.setenv("TEST_HOME", str(home))
    monkeypatch.setenv("CHECKPOINT_PROVIDER", "claude")

    transcript_text = (
        json.dumps({"type": "user", "promptId": "p-1", "uuid": "u1", "parentUuid": None,
                    "message": {"role": "user", "content": "hi"}}) + "\n"
        + json.dumps({"type": "assistant", "uuid": "a1", "parentUuid": "u1",
                      "message": {"role": "assistant", "content": []}}) + "\n"
        + json.dumps({"type": "summary", "uuid": "sum1", "parentUuid": None}) + "\n"
        + json.dumps({"type": "user", "promptId": "p-2", "uuid": "u2", "parentUuid": "a1",
                      "message": {"role": "user", "content": "again"}}) + "\n"
    )
    _seed_claude_session_for_resume(plugin_home, home, cwd, transcript, transcript_text=transcript_text)

    report = ResumeOrchestrator(cwd=cwd).execute(
        ResumeOrchestrator(cwd=cwd).plan("s1", 0), lambda _text: True
    )
    records = [
        json.loads(line)
        for line in Path(report.provider_session_path).read_text(encoding="utf-8").splitlines()
    ]
    by_type = {record["type"]: record for record in records if "uuid" in record}
    # The second user record's parentUuid must point at the assistant uuid,
    # NOT at the summary uuid even though summary was written between them.
    assistant_uuid = by_type["assistant"]["uuid"]
    second_user = [record for record in records if record.get("type") == "user" and record.get("promptId") == "p-2"][0]
    assert second_user["parentUuid"] == assistant_uuid


def test_codex_rewrite_repoints_forked_from_id():
    """P4-5: session_meta.forked_from_id must point at the session we forked FROM
    (the original id), not keep the stale ancestor it had on disk."""
    from checkpoint_plugin.resume import _rewrite_codex_trajectory

    trajectory = (
        json.dumps({"type": "session_meta", "payload": {"id": "ORIGINAL", "forked_from_id": "STALE-ANCESTOR", "cwd": "/old"}}) + "\n"
        + json.dumps({"type": "turn_context", "payload": {"turn_id": "t-1", "model": "m"}}) + "\n"
    ).encode("utf-8")
    out = _rewrite_codex_trajectory(trajectory, "NEW", Path("/new"), None, None, None)
    metas = [json.loads(l)["payload"] for l in out.splitlines() if json.loads(l).get("type") == "session_meta"]
    assert metas, "expected a session_meta record"
    meta = metas[0]
    assert meta["id"] == "NEW"
    # Lineage now points at the original session, not the stale ancestor.
    assert meta["forked_from_id"] == "ORIGINAL"
    assert meta["forked_from_id"] != "STALE-ANCESTOR"


def test_codex_rewrite_preserves_structured_permission_profile():
    """F1: a real turn_context.permission_profile is an object; resume must not
    overwrite it with the bare permission_mode string. Uses the REAL Codex shape
    (type at record level, payload carries no `type`) so it exercises the live
    path (P4-2)."""
    from checkpoint_plugin.resume import _rewrite_codex_trajectory

    profile = {"type": "managed", "file_system": {"type": "restricted"}, "network": "restricted"}
    trajectory = (
        json.dumps({"type": "session_meta", "payload": {"id": "old", "cwd": "/old"}}) + "\n"
        + json.dumps(
            {
                "type": "turn_context",
                "payload": {
                    "turn_id": "t-1",
                    "model": "old-model",
                    "permission_profile": profile,
                    "sandbox_policy": {"type": "workspace-write"},
                    "approval_policy": "on-request",
                },
            }
        )
        + "\n"
    ).encode("utf-8")

    out = _rewrite_codex_trajectory(trajectory, "NEW", Path("/new"), "gpt-target", "acceptEdits", None)
    records = [json.loads(line) for line in out.splitlines()]
    turn_context = next(r["payload"] for r in records if r.get("type") == "turn_context")
    # Structured profile is preserved verbatim; model is re-pinned; sandbox/approval untouched.
    assert turn_context["permission_profile"] == profile
    assert turn_context["model"] == "gpt-target"
    assert turn_context["sandbox_policy"] == {"type": "workspace-write"}
    assert turn_context["approval_policy"] == "on-request"


def test_codex_rewrite_repins_model_on_real_turn_context_shape():
    """P4-2: model must be re-pinned on turn_context even when `type` lives only
    at the record level (the real Codex shape)."""
    from checkpoint_plugin.resume import _rewrite_codex_trajectory

    trajectory = (
        json.dumps({"type": "session_meta", "payload": {"id": "old", "cwd": "/old"}}) + "\n"
        + json.dumps({"type": "turn_context", "payload": {"turn_id": "t-1", "model": "old-model"}}) + "\n"
    ).encode("utf-8")
    out = _rewrite_codex_trajectory(trajectory, "NEW", Path("/new"), "gpt-target", None, None)
    tc = next(json.loads(l)["payload"] for l in out.splitlines() if json.loads(l).get("type") == "turn_context")
    assert tc["model"] == "gpt-target"


def test_codex_rewrite_repins_string_permission_profile():
    """Legacy/simple string permission_profile is still re-pinned for back-compat."""
    from checkpoint_plugin.resume import _rewrite_codex_trajectory

    trajectory = (
        json.dumps({"type": "session_meta", "payload": {"id": "old", "cwd": "/old"}}) + "\n"
        + json.dumps(
            {"type": "turn_context", "payload": {"permission_profile": "old", "sandbox_policy": "workspace-write"}}
        )
        + "\n"
    ).encode("utf-8")
    out = _rewrite_codex_trajectory(trajectory, "NEW", Path("/new"), None, "acceptEdits", None)
    tc = next(json.loads(l)["payload"] for l in out.splitlines() if json.loads(l).get("type") == "turn_context")
    assert tc["permission_profile"] == "acceptEdits"
    assert tc["sandbox_policy"] == "workspace-write"


def test_claude_rewrite_repins_model_on_assistant_message():
    """F2: Claude model lives at message.model on assistant records, not top-level."""
    from checkpoint_plugin.resume import _rewrite_claude_trajectory

    trajectory = (
        json.dumps({"type": "user", "uuid": "u1", "parentUuid": None, "promptId": "p", "message": {"role": "user", "content": "hi"}}) + "\n"
        + json.dumps({"type": "assistant", "uuid": "a1", "parentUuid": "u1", "message": {"role": "assistant", "model": "claude-opus-4-8", "content": []}}) + "\n"
    ).encode("utf-8")
    out = _rewrite_claude_trajectory(trajectory, "NEW", Path("/new"), "claude-sonnet-4-6", None)
    records = [json.loads(line) for line in out.splitlines()]
    assistant = next(r for r in records if r.get("type") == "assistant")
    assert assistant["message"]["model"] == "claude-sonnet-4-6"
    # sessionId rewritten across the board.
    assert {r["sessionId"] for r in records} == {"NEW"}


def test_claude_rewrite_remaps_file_history_message_id():
    """P4-4: file-history-snapshot.messageId and last-prompt.leafUuid must be
    remapped through the uuid map (incl. forward references), not left dangling."""
    from checkpoint_plugin.resume import _rewrite_claude_trajectory

    # messageId points FORWARD to an assistant uuid that appears later.
    trajectory = (
        json.dumps({"type": "file-history-snapshot", "messageId": "a1", "snapshot": {}}) + "\n"
        + json.dumps({"type": "user", "uuid": "u1", "parentUuid": None, "promptId": "p", "message": {"role": "user", "content": "hi"}}) + "\n"
        + json.dumps({"type": "assistant", "uuid": "a1", "parentUuid": "u1", "message": {"role": "assistant", "content": []}}) + "\n"
        + json.dumps({"type": "last-prompt", "leafUuid": "a1"}) + "\n"
    ).encode("utf-8")
    out = _rewrite_claude_trajectory(trajectory, "NEW", Path("/new"), None, None)
    records = [json.loads(line) for line in out.splitlines()]
    new_uuids = {r["uuid"] for r in records if isinstance(r.get("uuid"), str)}
    fhs = next(r for r in records if r.get("type") == "file-history-snapshot")
    last_prompt = next(r for r in records if r.get("type") == "last-prompt")
    assistant = next(r for r in records if r.get("type") == "assistant")
    # Pointers were remapped to the NEW assistant uuid, not left as "a1".
    assert fhs["messageId"] != "a1"
    assert fhs["messageId"] == assistant["uuid"]
    assert fhs["messageId"] in new_uuids
    assert last_prompt["leafUuid"] == assistant["uuid"]


def test_resume_forked_session_includes_inherited_prefix(tmp_path, monkeypatch):
    """F3: resuming a forked session (first turn anchored mid-transcript) must
    materialize the inherited pre-fork records, not start amnesiac."""
    plugin_home = tmp_path / "plugin"
    home = tmp_path / "home"
    claude_home = home / ".claude"
    cwd = tmp_path / "work"
    transcript = tmp_path / "claude.jsonl"
    cwd.mkdir()
    # Inherited pre-fork history, then the forked turn (new promptId).
    inherited = (
        json.dumps({"type": "mode", "mode": "normal", "sessionId": "old"}) + "\n"
        + json.dumps({"type": "user", "sessionId": "old", "uuid": "iu", "parentUuid": None, "promptId": "old-p", "cwd": "/old", "message": {"role": "user", "content": "INHERITED-PROMPT"}}) + "\n"
        + json.dumps({"type": "assistant", "sessionId": "old", "uuid": "ia", "parentUuid": "iu", "cwd": "/old", "message": {"role": "assistant", "content": []}}) + "\n"
    )
    fork_offset = len(inherited.encode("utf-8"))
    forked_turn = json.dumps({"type": "user", "sessionId": "old", "uuid": "fu", "parentUuid": "ia", "promptId": "fork-p", "cwd": "/old", "message": {"role": "user", "content": "FORKED-PROMPT"}}) + "\n"
    transcript.write_text(inherited + forked_turn, encoding="utf-8")
    (cwd / "file.txt").write_text("v1", encoding="utf-8")
    _isolate_provider_env(monkeypatch)
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(plugin_home))
    monkeypatch.setenv("TEST_HOME", str(home))
    monkeypatch.setenv("CHECKPOINT_PROVIDER", "claude")

    coordinator = CheckpointCoordinator(session_id="forked", cwd=cwd)
    coordinator.on_session_start(source="resume", source_transcript_path=str(transcript))
    # The captured turn anchors at the fork offset, mirroring real on-disk forks.
    coordinator.on_turn_end(
        TurnRecord(user_message="forked"),
        TrajectoryReference("claude", str(transcript), fork_offset, transcript.stat().st_size, 1),
    )

    report = ResumeOrchestrator(cwd=cwd).execute(ResumeOrchestrator(cwd=cwd).plan("forked", 0), lambda _text: True)
    materialized = Path(report.provider_session_path).read_text(encoding="utf-8")
    # Both the inherited history AND the forked turn are present in the resume.
    assert "INHERITED-PROMPT" in materialized
    assert "FORKED-PROMPT" in materialized
    records = [json.loads(line) for line in materialized.splitlines()]
    assert {r["sessionId"] for r in records} == {report.new_session_id}


def test_resume_of_a_resume_preserves_all_records(tmp_path, monkeypatch):
    """P4-3: resuming a resumed session must not drop records. The resumed
    manifests' byte offsets must align to the REWRITTEN provider file, else the
    next resume raw-seeks mid-line and loses a record per generation."""
    plugin_home = tmp_path / "plugin"
    home = tmp_path / "home"
    cwd = tmp_path / "work"
    transcript = tmp_path / "claude.jsonl"
    cwd.mkdir()
    # Leading control record + two turns (so the rewriter inserts a synthetic
    # permission-mode record and re-serializes, shifting byte offsets).
    lines = [
        json.dumps({"type": "mode", "mode": "normal", "sessionId": "old"}),
        json.dumps({"type": "user", "sessionId": "old", "uuid": "u1", "parentUuid": None, "promptId": "p1", "cwd": "/old", "message": {"role": "user", "content": "turn one"}}),
        json.dumps({"type": "assistant", "sessionId": "old", "uuid": "a1", "parentUuid": "u1", "cwd": "/old", "message": {"role": "assistant", "model": "m", "content": []}}),
        json.dumps({"type": "user", "sessionId": "old", "uuid": "u2", "parentUuid": "a1", "promptId": "p2", "cwd": "/old", "message": {"role": "user", "content": "turn two"}}),
        json.dumps({"type": "assistant", "sessionId": "old", "uuid": "a2", "parentUuid": "u2", "cwd": "/old", "message": {"role": "assistant", "model": "m", "content": []}}),
    ]
    data = ("\n".join(lines) + "\n").encode("utf-8")
    transcript.write_bytes(data)
    (cwd / "file.txt").write_text("v1", encoding="utf-8")
    _isolate_provider_env(monkeypatch)
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(plugin_home))
    monkeypatch.setenv("TEST_HOME", str(home))
    monkeypatch.setenv("CHECKPOINT_PROVIDER", "claude")
    monkeypatch.setenv("CLAUDE_PERMISSION_MODE", "acceptEdits")

    # Two turns sliced by promptId boundary.
    turn0_end = data.find(b'"p2"')
    turn0_end = data.rfind(b"\n", 0, turn0_end) + 1
    coordinator = CheckpointCoordinator(session_id="s1", cwd=cwd)
    coordinator.on_session_start()
    coordinator.on_turn_end(TurnRecord(user_message="one"), TrajectoryReference("claude", str(transcript), 0, turn0_end, 3))
    coordinator.on_turn_end(TurnRecord(user_message="two"), TrajectoryReference("claude", str(transcript), turn0_end, len(data), 2))

    def _count(path):
        return sum(1 for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip())

    gen1 = ResumeOrchestrator(cwd=cwd).execute(ResumeOrchestrator(cwd=cwd).plan("s1", 1), lambda _t: True)
    gen1_records = _count(gen1.provider_session_path)

    # Resume the resumed session — this is where stale offsets would drop records.
    gen2 = ResumeOrchestrator(cwd=cwd).execute(ResumeOrchestrator(cwd=cwd).plan(gen1.new_session_id, 1), lambda _t: True)
    gen2_records = _count(gen2.provider_session_path)

    assert gen2_records == gen1_records, f"resume-of-resume dropped records: {gen1_records} -> {gen2_records}"
    # Chain still coherent after two generations.
    recs = [json.loads(line) for line in Path(gen2.provider_session_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    assert {r["sessionId"] for r in recs} == {gen2.new_session_id}
    seen: set[str] = set()
    for r in recs:
        pu = r.get("parentUuid")
        if isinstance(pu, str):
            assert pu in seen, "broken parentUuid chain after resume-of-resume"
        if isinstance(r.get("uuid"), str):
            seen.add(r["uuid"])



def test_codex_rewrite_collapses_session_meta_chain():
    """H1: a forked/subagent codex transcript carries a CHAIN of stacked
    session_meta records; the rewrite must emit exactly ONE leading meta with the
    new id, pointing forked_from_id at the original (first) session id."""
    from checkpoint_plugin.resume import _rewrite_codex_trajectory

    trajectory = (
        json.dumps({"type": "session_meta", "payload": {"id": "SUB", "forked_from_id": "PARENT", "cwd": "/old"}}) + "\n"
        + json.dumps({"type": "session_meta", "payload": {"id": "PARENT", "forked_from_id": "GRAND", "cwd": "/old"}}) + "\n"
        + json.dumps({"type": "session_meta", "payload": {"id": "GRAND", "cwd": "/old"}}) + "\n"
        + json.dumps({"type": "turn_context", "payload": {"turn_id": "t-1", "model": "m"}}) + "\n"
    ).encode("utf-8")
    out = _rewrite_codex_trajectory(trajectory, "NEW", Path("/new"), None, None, None)
    records = [json.loads(line) for line in out.splitlines()]
    metas = [r for r in records if r.get("type") == "session_meta"]
    assert len(metas) == 1, "exactly one session_meta must survive"
    assert metas[0]["payload"]["id"] == "NEW"
    assert metas[0]["payload"]["forked_from_id"] == "SUB"
    assert records[0]["type"] == "session_meta"


def test_codex_rewrite_strips_thread_rolled_back():
    """M1: transient fork-control event_msgs (thread_rolled_back) describe the
    parent's history surgery and must not be carried into the resumed session."""
    from checkpoint_plugin.resume import _rewrite_codex_trajectory

    trajectory = (
        json.dumps({"type": "session_meta", "payload": {"id": "old", "cwd": "/old"}}) + "\n"
        + json.dumps({"type": "event_msg", "payload": {"type": "thread_rolled_back", "num_turns": 1}}) + "\n"
        + json.dumps({"type": "event_msg", "payload": {"type": "task_started"}}) + "\n"
    ).encode("utf-8")
    out = _rewrite_codex_trajectory(trajectory, "NEW", Path("/new"), None, None, None)
    assert b"thread_rolled_back" not in out
    assert b"task_started" in out


def test_render_diff_shows_effort_change():
    """M3: a thinking-effort drift must be surfaced in the rendered diff."""
    from checkpoint_plugin.env.differ import diff_environments, render_diff
    from checkpoint_plugin.types import EnvironmentState

    current = EnvironmentState(provider="claude", effort="high")
    target = EnvironmentState(provider="claude", effort="medium")
    diff = diff_environments(current, target)
    assert diff.effort_changed
    rendered = render_diff(diff, current, target)
    assert "Effort: high -> medium" in rendered


def test_claude_rewrite_drops_dangling_trailing_pointer():
    """M4: a trailing keyless file-history-snapshot whose messageId points at a
    uuid outside the slice (a forward reference) is dropped, not left dangling."""
    from checkpoint_plugin.resume import _rewrite_claude_trajectory

    trajectory = (
        json.dumps({"type": "user", "uuid": "u-1", "sessionId": "old", "message": {"role": "user", "content": "hi"}}) + "\n"
        + json.dumps({"type": "assistant", "uuid": "a-1", "parentUuid": "u-1", "sessionId": "old", "message": {"role": "assistant", "content": "yo"}}) + "\n"
        + json.dumps({"type": "file-history-snapshot", "messageId": "FORWARD-MISSING", "snapshot": {}}) + "\n"
    ).encode("utf-8")
    out = _rewrite_claude_trajectory(trajectory, "NEW", Path("/new"), None, None)
    records = [json.loads(line) for line in out.splitlines()]
    assert all(r.get("type") != "file-history-snapshot" for r in records), "dangling snapshot must be dropped"
    # The real message records survive and are remapped.
    assert any(r.get("type") == "assistant" for r in records)


def test_resume_subagent_session_refuses_with_parent_redirect(tmp_path, monkeypatch):
    """H2: resuming a subagent checkpoint standalone is refused; the error names
    the parent session and a redirect command."""
    import pytest

    plugin_home = tmp_path / "plugin"
    cwd = tmp_path / "work"
    cwd.mkdir()
    _isolate_provider_env(monkeypatch)
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(plugin_home))

    # Parent session with one turn.
    parent = CheckpointCoordinator(session_id="parent-1", cwd=cwd)
    parent.on_session_start()
    parent.on_turn_end(TurnRecord(user_message="spawn agent agent-xyz"))

    # Subagent session keyed under the parent, carrying lineage metadata.
    sub = CheckpointCoordinator(session_id="parent-1--subagent-agent-xyz", cwd=cwd)
    sub.on_session_start(
        source="subagent",
        lineage={"parent_session_id": "parent-1", "agent_id": "agent-xyz"},
    )
    sub.on_turn_end(TurnRecord(user_message="sub work"))

    orchestrator = ResumeOrchestrator(cwd=cwd)
    with pytest.raises(RuntimeError) as exc:
        orchestrator.plan("parent-1--subagent-agent-xyz", 0)
    message = str(exc.value)
    assert "parent-1" in message
    assert "checkpoint resume parent-1" in message


def test_parent_resume_rewrites_carried_subagent_sessionid(tmp_path, monkeypatch):
    """H3: when a parent resume carries subagent transcripts, each carried
    record's sessionId is rewritten to the new parent id while uuid/parentUuid
    stay byte-identical."""
    plugin_home = tmp_path / "plugin"
    home = tmp_path / "home"
    claude_home = home / ".claude"
    cwd = tmp_path / "work"
    transcript = tmp_path / "claude.jsonl"
    cwd.mkdir()
    transcript.write_text(
        "\n".join(
            [
                json.dumps({"type": "permission-mode", "sessionId": "old", "permissionMode": "default"}),
                json.dumps({"type": "user", "sessionId": "old", "uuid": "u-1", "parentUuid": None, "cwd": "/old", "message": {"role": "user", "content": "hi"}}),
                json.dumps({"type": "assistant", "sessionId": "old", "uuid": "a-1", "parentUuid": "u-1", "cwd": "/old", "message": {"role": "assistant", "content": []}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    # A subagent transcript stored under the OLD parent session id.
    project = str(cwd).replace("/", "-")
    sub_dir = claude_home / "projects" / project / "s1" / "subagents"
    sub_dir.mkdir(parents=True)
    sub_file = sub_dir / "agent-abc.jsonl"
    sub_records = [
        {"type": "user", "sessionId": "s1", "uuid": "sub-u1", "parentUuid": None, "isSidechain": True, "message": {"role": "user", "content": "go"}},
        {"type": "assistant", "sessionId": "s1", "uuid": "sub-a1", "parentUuid": "sub-u1", "isSidechain": True, "message": {"role": "assistant", "content": []}},
    ]
    sub_file.write_text("\n".join(json.dumps(r) for r in sub_records) + "\n", encoding="utf-8")

    (cwd / "file.txt").write_text("v1", encoding="utf-8")
    _isolate_provider_env(monkeypatch)
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(plugin_home))
    monkeypatch.setenv("TEST_HOME", str(home))
    monkeypatch.setenv("CHECKPOINT_PROVIDER", "claude")

    coordinator = CheckpointCoordinator(session_id="s1", cwd=cwd)
    coordinator.on_session_start()
    coordinator.on_turn_end(
        TurnRecord(user_message="first"),
        TrajectoryReference("claude", str(transcript), 0, transcript.stat().st_size, 3),
    )

    report = ResumeOrchestrator(cwd=cwd).execute(ResumeOrchestrator(cwd=cwd).plan("s1", 0), lambda _text: True)

    carried = claude_home / "projects" / project / report.new_session_id / "subagents" / "agent-abc.jsonl"
    assert carried.exists()
    out = [json.loads(line) for line in carried.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert {r["sessionId"] for r in out} == {report.new_session_id}
    # Internal uuid/parentUuid chain is preserved verbatim (self-contained sidechain).
    assert [r["uuid"] for r in out] == ["sub-u1", "sub-a1"]
    assert [r.get("parentUuid") for r in out] == [None, "sub-u1"]
