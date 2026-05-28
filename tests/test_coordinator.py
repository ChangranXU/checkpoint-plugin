import json
from multiprocessing import Process

from checkpoint_plugin.coordinator import CheckpointCoordinator, TurnRecord
from checkpoint_plugin.store import CheckpointStore


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
    trajectory = store.trajectory_path.read_bytes()
    line_boundaries = _jsonl_line_boundaries(trajectory)

    assert [manifest.turn_id for manifest in manifests] == list(range(len(processes)))
    assert len(line_boundaries) == len(processes)
    for manifest in manifests:
        assert (manifest.trajectory_offset, manifest.trajectory_end_offset) in line_boundaries


def _write_turn(home, cwd, session_id, user_message):
    coordinator = CheckpointCoordinator(session_id=session_id, cwd=cwd, plugin_home=home)
    coordinator.on_turn_end(TurnRecord(user_message=user_message))


def _jsonl_line_boundaries(data: bytes) -> set[tuple[int, int]]:
    offset = 0
    boundaries: set[tuple[int, int]] = set()
    for line in data.splitlines(keepends=True):
        json.loads(line)
        end = offset + len(line)
        boundaries.add((offset, end))
        offset = end
    return boundaries
