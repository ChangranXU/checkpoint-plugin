import json

from checkpoint_plugin.cli import main


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
