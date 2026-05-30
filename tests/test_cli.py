import io
import json

from checkpoint_plugin.cli import main, _colorize, _supports_color


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
