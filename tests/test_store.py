from checkpoint_plugin.store import CheckpointStore
from checkpoint_plugin.types import CheckpointManifest


def test_blob_dedup_and_manifest_roundtrip(tmp_path):
    store = CheckpointStore(tmp_path / "session")

    first = store.store_blob(b"hello")
    second = store.store_blob(b"hello")
    assert first == second
    assert store.load_blob(first) == b"hello"

    manifest = CheckpointManifest(
        turn_id=3,
        session_id="s1",
        created_ts="2026-05-28T00:00:00Z",
        env_ref=first,
        fs_ref=second,
        trajectory_offset=12,
        trajectory_end_offset=34,
        user_message_preview="hi",
        parent_turn_id=2,
    )
    store.write_manifest(manifest)

    assert store.read_manifest(3) == manifest
    assert store.list_turn_ids() == [3]


def test_append_trajectory_returns_start_and_end_offsets(tmp_path):
    store = CheckpointStore(tmp_path / "session")

    first_start, first_end = store.append_trajectory({"event": 1})
    second_start, second_end = store.append_trajectory({"event": 2})

    assert first_start == 0
    assert first_end == second_start
    assert second_end > second_start
    assert store.slice_trajectory(first_end).count(b"\n") == 1
    assert store.slice_trajectory(second_end).count(b"\n") == 2
