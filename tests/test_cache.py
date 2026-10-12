import json
from pathlib import Path

from research.cache import CacheStore


def test_cache_set_and_get(tmp_path):
    cache = CacheStore(
        directory=tmp_path,
    )

    cache.set(
        "test-key",
        {
            "answer": "cached",
            "count": 3,
        },
    )

    assert cache.get(
        "test-key"
    ) == {
        "answer": "cached",
        "count": 3,
    }


def test_cache_miss_returns_none(tmp_path):
    cache = CacheStore(
        directory=tmp_path,
    )

    assert cache.get(
        "missing-key"
    ) is None


def test_disabled_cache_does_not_read_or_write(tmp_path):
    cache = CacheStore(
        directory=tmp_path,
        enabled=False,
    )

    cache.set(
        "test-key",
        {"value": 1},
    )

    assert cache.get(
        "test-key"
    ) is None

    assert list(
        tmp_path.iterdir()
    ) == []


def test_cache_key_is_deterministic():
    first = CacheStore.build_key(
        "search",
        {
            "query": "machine learning",
            "max_results": 3,
        },
    )

    second = CacheStore.build_key(
        "search",
        {
            "max_results": 3,
            "query": "machine learning",
        },
    )

    assert first == second


def test_different_namespaces_produce_different_keys():
    search_key = CacheStore.build_key(
        "search",
        {"query": "test"},
    )

    extraction_key = CacheStore.build_key(
        "extract",
        {"query": "test"},
    )

    assert search_key != extraction_key


def test_cache_key_does_not_expose_raw_query():
    query = (
        "machine learning "
        "this should not become a filename"
    )

    key = CacheStore.build_key(
        "search",
        {"query": query},
    )

    assert query not in key
    assert "machine" not in key


def test_expired_entry_is_removed(tmp_path):
    current_time = [100.0]

    cache = CacheStore(
        directory=tmp_path,
        ttl_seconds=10,
        clock=lambda: current_time[0],
    )

    cache.set(
        "test-key",
        {"value": 1},
    )

    assert cache.get(
        "test-key"
    ) == {
        "value": 1,
    }

    current_time[0] = 111.0

    assert cache.get(
        "test-key"
    ) is None

    assert not (
        tmp_path / "test-key.json"
    ).exists()


def test_zero_ttl_means_no_expiration(tmp_path):
    current_time = [100.0]

    cache = CacheStore(
        directory=tmp_path,
        ttl_seconds=0,
        clock=lambda: current_time[0],
    )

    cache.set(
        "test-key",
        {"value": 1},
    )

    current_time[0] = 100000.0

    assert cache.get(
        "test-key"
    ) == {
        "value": 1,
    }


def test_corrupted_json_is_treated_as_cache_miss(
    tmp_path,
):
    cache = CacheStore(
        directory=tmp_path,
    )

    path = tmp_path / "broken.json"

    path.write_text(
        "{this is not valid json",
        encoding="utf-8",
    )

    assert cache.get(
        "broken"
    ) is None

    assert not path.exists()


def test_malformed_cache_entry_is_treated_as_miss(
    tmp_path,
):
    cache = CacheStore(
        directory=tmp_path,
    )

    path = tmp_path / "malformed.json"

    path.write_text(
        json.dumps(
            {
                "created_at": 100.0,
            }
        ),
        encoding="utf-8",
    )

    assert cache.get(
        "malformed"
    ) is None

    assert not path.exists()


def test_invalid_timestamp_is_treated_as_miss(
    tmp_path,
):
    cache = CacheStore(
        directory=tmp_path,
    )

    path = tmp_path / "invalid-time.json"

    path.write_text(
        json.dumps(
            {
                "created_at": "not-a-number",
                "value": 123,
            }
        ),
        encoding="utf-8",
    )

    assert cache.get(
        "invalid-time"
    ) is None

    assert not path.exists()


def test_cache_delete(tmp_path):
    cache = CacheStore(
        directory=tmp_path,
    )

    cache.set(
        "test-key",
        {"value": 1},
    )

    cache.delete(
        "test-key"
    )

    assert cache.get(
        "test-key"
    ) is None


def test_cache_clear(tmp_path):
    cache = CacheStore(
        directory=tmp_path,
    )

    cache.set(
        "first",
        {"value": 1},
    )

    cache.set(
        "second",
        {"value": 2},
    )

    cache.clear()

    assert list(
        tmp_path.glob("*.json")
    ) == []


def test_cache_handles_nested_json_values(tmp_path):
    cache = CacheStore(
        directory=tmp_path,
    )

    value = {
        "results": [
            {
                "title": "Example",
                "scores": [1, 2, 3],
            }
        ],
        "metadata": {
            "source": "test",
        },
    }

    cache.set(
        "nested",
        value,
    )

    assert cache.get(
        "nested"
    ) == value


def test_cache_write_is_valid_json(tmp_path):
    cache = CacheStore(
        directory=tmp_path,
    )

    cache.set(
        "json-test",
        {
            "message": "hello",
        },
    )

    path = tmp_path / "json-test.json"

    assert path.exists()

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    assert data["value"] == {
        "message": "hello",
    }

    assert isinstance(
        data["created_at"],
        float,
    )