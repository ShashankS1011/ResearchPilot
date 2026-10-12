from research.persistence import SQLiteStore


def test_sqlite_store_persists_session_across_instances(tmp_path):
    database = tmp_path / "researchpilot.db"
    first = SQLiteStore(database)

    first.save_session(
        "session-1",
        {"query": "machine learning"},
        "2026-10-12T00:00:00+00:00",
    )

    second = SQLiteStore(database)

    assert second.load_session("session-1") == {
        "query": "machine learning"
    }


def test_sqlite_store_updates_and_deletes_session(tmp_path):
    store = SQLiteStore(tmp_path / "researchpilot.db")

    store.save_session("session-1", {"query": "old"}, "1")
    store.save_session("session-1", {"query": "new"}, "2")

    assert store.load_session("session-1") == {"query": "new"}

    store.delete_session("session-1")

    assert store.load_session("session-1") is None


def test_sqlite_store_lists_research_artifacts_in_order(tmp_path):
    store = SQLiteStore(tmp_path / "researchpilot.db")

    store.save_artifact("research_runs", "session-1", [1], "1")
    store.save_artifact("research_runs", "session-1", [2], "2")

    assert store.list_artifacts("research_runs", "session-1") == [
        [1],
        [2],
    ]
