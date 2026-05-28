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
        user_message_preview="hi",
        parent_turn_id=2,
    )
    store.write_manifest(manifest)

    assert store.read_manifest(3) == manifest
    assert store.list_turn_ids() == [3]


def test_append_trajectory_returns_start_offset(tmp_path):
    store = CheckpointStore(tmp_path / "session")

    first = store.append_trajectory({"event": 1})
    second = store.append_trajectory({"event": 2})

    assert first == 0
    assert second > first
    assert store.slice_trajectory(second).count(b"\n") == 1
