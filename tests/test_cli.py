import io
import json

from checkpoint_plugin.cli import (
    main,
    _colorize,
    _edit_send_replaced_turns,
    _rolled_back_count,
    _supports_color,
)


def test_list_sessions_shows_title_and_source(tmp_path, monkeypatch, capsys):
    home = tmp_path / "plugin"
    session = home / "sessions" / "s1"
    session.mkdir(parents=True)
    (session / "metadata.json").write_text(
        json.dumps(
            {
                "session_id": "s1",
                "session_title": "Respond to greeting",
                "source": "startup",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(home))

    assert main(["list"]) == 0

    assert capsys.readouterr().out == "s1  Respond to greeting  startup\n"


def test_list_sessions_handles_missing_metadata(tmp_path, monkeypatch, capsys):
    home = tmp_path / "plugin"
    (home / "sessions" / "s1").mkdir(parents=True)
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(home))

    assert main(["list"]) == 0

    assert capsys.readouterr().out == "s1  -  -\n"


class _Stream(io.StringIO):
    def __init__(self, tty: bool) -> None:
        super().__init__()
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


def test_colorize_wraps_on_tty_and_is_plain_otherwise(monkeypatch):
    """The resume-command hint is colored only when stdout is a real TTY."""
    monkeypatch.delenv("NO_COLOR", raising=False)
    cmd = "codex resume 4adbaa3b-f00a-4882-8dd8-0f6184650a60"

    colored = _colorize(cmd, "bold green", stream=_Stream(tty=True))
    assert colored == f"\033[1m\033[32m{cmd}\033[0m"
    # The raw command is still present (selectable/copyable) inside the escapes.
    assert cmd in colored

    # Non-TTY (piped/redirected) gets no escape codes.
    assert _colorize(cmd, "bold green", stream=_Stream(tty=False)) == cmd


def test_colorize_respects_no_color_env(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    cmd = "codex resume abc"
    assert _colorize(cmd, "bold green", stream=_Stream(tty=True)) == cmd
    assert _supports_color(_Stream(tty=True)) is False


def _seed_turn(coordinator, transcript, user_message, end_offset, *, boundary_mode="per_turn_key"):
    from checkpoint_plugin.coordinator import TurnRecord
    from checkpoint_plugin.types import TrajectoryReference

    record_count = sum(1 for line in transcript.read_bytes()[:end_offset].splitlines() if line.strip())
    coordinator.on_turn_end(
        TurnRecord(user_message=user_message),
        TrajectoryReference("codex", str(transcript), 0, end_offset, record_count, boundary_mode=boundary_mode),
    )


def test_list_session_reanchors_last_turn_to_eof(tmp_path, monkeypatch, capsys):
    """F1: `list --session` recovers a trailing record flushed after the Stop hook,
    matching show/diff/resume. The stored manifest was short; list reads at EOF.

    This is the timeout-free path: by read time the transcript is fully flushed, so
    recovery does not depend on the capture-time settle winning any race.
    """
    from checkpoint_plugin.coordinator import CheckpointCoordinator

    home = tmp_path / "plugin"
    cwd = tmp_path / "work"
    cwd.mkdir()
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(home))
    monkeypatch.setenv("CHECKPOINT_SIDECHAIN_SETTLE_TIMEOUT", "0")

    transcript = tmp_path / "rollout.jsonl"
    transcript.write_text(
        json.dumps({"type": "response_item", "turn_id": "t1", "payload": {"type": "message"}}) + "\n",
        encoding="utf-8",
    )
    c = CheckpointCoordinator(session_id="codexsess", cwd=cwd)
    c.on_session_start()
    captured = transcript.stat().st_size
    _seed_turn(c, transcript, "do work", captured)
    assert c.store.read_manifest(0).trajectory_ref.end_offset == captured

    # Provider flushes the turn-closing record AFTER the hook captured the slice.
    with transcript.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "event_msg", "payload": {"type": "task_complete"}, "turn_id": "t1"}) + "\n")
    grown = transcript.stat().st_size
    assert grown > captured

    assert main(["list", "--session", "codexsess"]) == 0
    capsys.readouterr()  # drain
    # The stored manifest is now reanchored to EOF as a side effect of the read.
    assert c.store.read_manifest(0).trajectory_ref.end_offset == grown


def _rolled_back_transcript(path):
    """A codex rollout with an edit-send: turn t2 rolls back turn t1 (version 1)."""
    path.write_text(
        json.dumps({"type": "event_msg", "payload": {"type": "user_message", "message": "version 1"}, "turn_id": "t1"}) + "\n"
        + json.dumps({"type": "event_msg", "payload": {"type": "thread_rolled_back", "num_turns": 1}}) + "\n"
        + json.dumps({"type": "event_msg", "payload": {"type": "user_message", "message": "version 2"}, "turn_id": "t2"}) + "\n",
        encoding="utf-8",
    )


def test_edit_send_replaced_turns_detects_rollback(tmp_path, monkeypatch):
    """F2: a turn whose slice carries `thread_rolled_back num_turns=K` supersedes the
    K preceding turns. The mapping marks each replaced turn with its replacement."""
    from checkpoint_plugin.coordinator import CheckpointCoordinator
    from checkpoint_plugin.types import TrajectoryReference
    from checkpoint_plugin.coordinator import TurnRecord

    home = tmp_path / "plugin"
    cwd = tmp_path / "work"
    cwd.mkdir()
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(home))
    transcript = tmp_path / "rollout.jsonl"
    _rolled_back_transcript(transcript)
    data = transcript.read_bytes()
    # turn 0 = version 1 (rolled back); turn 1 = the rollback marker + version 2.
    first_nl = data.index(b"\n") + 1
    c = CheckpointCoordinator(session_id="es", cwd=cwd)
    c.on_session_start()
    c.on_turn_end(TurnRecord(user_message="version 1"), TrajectoryReference("codex", str(transcript), 0, first_nl, 1))
    c.on_turn_end(TurnRecord(user_message="version 2"), TrajectoryReference("codex", str(transcript), first_nl, len(data), 2))

    manifests = c.store.list_manifests()
    replaced = _edit_send_replaced_turns(c.store, manifests)
    assert replaced == {0: 1}
    assert _rolled_back_count(manifests[1]) == 1
    assert _rolled_back_count(manifests[0]) == 0


def test_edit_send_no_rollback_is_empty(tmp_path, monkeypatch):
    """No thread_rolled_back marker -> no replaced turns (the common case)."""
    from checkpoint_plugin.coordinator import CheckpointCoordinator, TurnRecord
    from checkpoint_plugin.types import TrajectoryReference

    home = tmp_path / "plugin"
    cwd = tmp_path / "work"
    cwd.mkdir()
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(home))
    transcript = tmp_path / "rollout.jsonl"
    transcript.write_text(
        json.dumps({"type": "event_msg", "payload": {"type": "user_message", "message": "hi"}, "turn_id": "t1"}) + "\n",
        encoding="utf-8",
    )
    c = CheckpointCoordinator(session_id="noroll", cwd=cwd)
    c.on_session_start()
    c.on_turn_end(TurnRecord(user_message="hi"), TrajectoryReference("codex", str(transcript), 0, transcript.stat().st_size, 1))
    assert _edit_send_replaced_turns(c.store, c.store.list_manifests()) == {}


def test_list_marks_replaced_turn(tmp_path, monkeypatch, capsys):
    """F2: `list --session` annotates an edit-send-replaced turn."""
    from checkpoint_plugin.coordinator import CheckpointCoordinator, TurnRecord
    from checkpoint_plugin.types import TrajectoryReference

    home = tmp_path / "plugin"
    cwd = tmp_path / "work"
    cwd.mkdir()
    monkeypatch.setenv("CHECKPOINT_PLUGIN_HOME", str(home))
    monkeypatch.setenv("NO_COLOR", "1")
    transcript = tmp_path / "rollout.jsonl"
    _rolled_back_transcript(transcript)
    data = transcript.read_bytes()
    first_nl = data.index(b"\n") + 1
    c = CheckpointCoordinator(session_id="esl", cwd=cwd)
    c.on_session_start()
    c.on_turn_end(TurnRecord(user_message="version 1"), TrajectoryReference("codex", str(transcript), 0, first_nl, 1))
    c.on_turn_end(TurnRecord(user_message="version 2"), TrajectoryReference("codex", str(transcript), first_nl, len(data), 2))

    assert main(["list", "--session", "esl"]) == 0
    out = capsys.readouterr().out
    assert "[replaced by turn 1]" in out
    # Only the dead turn is marked; the replacement is not.
    replaced_lines = [line for line in out.splitlines() if "[replaced" in line]
    assert len(replaced_lines) == 1
    assert replaced_lines[0].startswith("0000")

