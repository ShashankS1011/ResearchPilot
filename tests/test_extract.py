
from dataclasses import replace

import pytest

import research.extract as extract


class FakeTavilyClient:
    """Simple Tavily client test double."""

    instances = []

    def __init__(
        self,
        api_key,
    ):
        self.api_key = api_key
        self.calls = []
        self.responses = []
        self.errors = []

        FakeTavilyClient.instances.append(
            self
        )

    def extract(
        self,
        urls,
    ):
        self.calls.append(
            list(urls)
        )

        if self.errors:
            error = self.errors.pop(
                0
            )

            if error is not None:
                raise error

        if self.responses:
            return self.responses.pop(
                0
            )

        return {
            "results": []
        }


@pytest.fixture(autouse=True)
def reset_fake_client():
    """Reset the fake Tavily client between tests."""

    FakeTavilyClient.instances = []

    yield

    FakeTavilyClient.instances = []


def _replace_settings(
    monkeypatch,
    **changes,
):
    """Replace the frozen Settings instance for an isolated test."""

    monkeypatch.setattr(
        extract,
        "SETTINGS",
        replace(
            extract.SETTINGS,
            **changes,
        ),
    )


def _patch_tavily(
    monkeypatch,
):
    """Patch TavilyClient with the fake implementation."""

    monkeypatch.setattr(
        extract,
        "TavilyClient",
        FakeTavilyClient,
    )


def test_extract_content_returns_empty_result_for_empty_urls(
    monkeypatch,
):
    """Empty URL input should not call Tavily."""

    _patch_tavily(
        monkeypatch
    )

    result = extract.extract_content(
        []
    )

    assert result == {
        "results": []
    }

    assert FakeTavilyClient.instances == []


def test_extract_content_requires_api_key(
    monkeypatch,
):
    """Missing Tavily credentials should fail clearly."""

    monkeypatch.delenv(
        "TAVILY_API_KEY",
        raising=False,
    )

    with pytest.raises(
        ValueError,
        match="TAVILY_API_KEY was not found",
    ):
        extract.extract_content(
            ["https://example.com"]
        )


def test_extract_content_calls_tavily_and_caches_success(
    monkeypatch,
    tmp_path,
):
    """A successful extraction should be written to the cache."""

    monkeypatch.setenv(
        "TAVILY_API_KEY",
        "test-key",
    )

    _replace_settings(
        monkeypatch,
        cache_dir=str(tmp_path),
        cache_enabled=True,
        tavily_extract_cache_enabled=True,
    )

    _patch_tavily(
        monkeypatch
    )

    response = {
        "results": [
            {
                "url": "https://example.com",
                "raw_content": "Example content",
            }
        ]
    }

    def fake_extract(
        self,
        urls,
    ):
        self.calls.append(
            list(urls)
        )

        return response

    monkeypatch.setattr(
        FakeTavilyClient,
        "extract",
        fake_extract,
    )

    result = extract.extract_content(
        ["https://example.com"]
    )

    assert result == response

    assert len(
        FakeTavilyClient.instances
    ) == 1

    assert FakeTavilyClient.instances[0].calls == [
        ["https://example.com"]
    ]

    cache = extract._get_extract_cache()

    key = extract._build_extract_cache_key(
        ["https://example.com"]
    )

    assert cache.get(
        key
    ) == response


def test_extract_content_uses_cache_on_second_call(
    monkeypatch,
    tmp_path,
):
    """Repeated extraction requests should avoid Tavily."""

    monkeypatch.setenv(
        "TAVILY_API_KEY",
        "test-key",
    )

    _replace_settings(
        monkeypatch,
        cache_dir=str(tmp_path),
        cache_enabled=True,
        tavily_extract_cache_enabled=True,
    )

    _patch_tavily(
        monkeypatch
    )

    response = {
        "results": [
            {
                "url": "https://example.com",
                "raw_content": "Cached content",
            }
        ]
    }

    def fake_extract(
        self,
        urls,
    ):
        self.calls.append(
            list(urls)
        )

        return response

    monkeypatch.setattr(
        FakeTavilyClient,
        "extract",
        fake_extract,
    )

    first_result = extract.extract_content(
        ["https://example.com"]
    )

    assert first_result == response

    assert len(
        FakeTavilyClient.instances
    ) == 1

    first_client = FakeTavilyClient.instances[0]

    second_result = extract.extract_content(
        ["https://example.com"]
    )

    assert second_result == response

    assert len(
        FakeTavilyClient.instances
    ) == 1

    assert first_client.calls == [
        ["https://example.com"]
    ]


def test_extract_content_ignores_malformed_cached_response(
    monkeypatch,
):
    monkeypatch.setenv(
        "TAVILY_API_KEY",
        "test-key",
    )

    _replace_settings(
        monkeypatch,
        cache_enabled=True,
        tavily_extract_cache_enabled=True,
    )

    cache = type(
        "Cache",
        (),
        {
            "get": lambda self, key: ["not a response"],
            "set": lambda self, key, value: None,
        },
    )()
    response = {"results": [{"url": "https://example.com"}]}
    client = type(
        "Client",
        (),
        {
            "extract": lambda self, urls: response,
        },
    )()

    monkeypatch.setattr(extract, "_get_extract_cache", lambda: cache)
    monkeypatch.setattr(extract, "TavilyClient", lambda api_key: client)

    assert extract.extract_content(["https://example.com"]) == response


def test_extract_content_skips_invalid_urls(
    monkeypatch,
):
    monkeypatch.setenv(
        "TAVILY_API_KEY",
        "test-key",
    )
    _replace_settings(
        monkeypatch,
        cache_enabled=False,
        tavily_extract_cache_enabled=False,
    )
    _patch_tavily(monkeypatch)

    result = extract.extract_content(
        [
            "example.com/page",
            "ftp://example.com/file",
            "https://example.com/page",
        ]
    )

    assert result == {"results": []}
    assert len(FakeTavilyClient.instances) == 1
    assert FakeTavilyClient.instances[0].calls == [
        ["https://example.com/page"]
    ]


def test_extract_content_rejects_malformed_api_response(
    monkeypatch,
):
    monkeypatch.setenv(
        "TAVILY_API_KEY",
        "test-key",
    )
    _replace_settings(
        monkeypatch,
        tavily_max_retries=0,
        cache_enabled=False,
        tavily_extract_cache_enabled=False,
    )
    _patch_tavily(monkeypatch)

    monkeypatch.setattr(
        FakeTavilyClient,
        "extract",
        lambda self, urls: {"results": "not a list"},
    )

    with pytest.raises(
        RuntimeError,
        match="Tavily extraction failed",
    ):
        extract.extract_content(["https://example.com"])


def test_extract_content_passes_configured_timeout(monkeypatch):
    monkeypatch.setenv(
        "TAVILY_API_KEY",
        "test-key",
    )
    _replace_settings(
        monkeypatch,
        cache_enabled=False,
        tavily_extract_cache_enabled=False,
        request_timeout_seconds=19,
    )

    calls = []

    class FakeClient:
        def __init__(self, api_key):
            self.api_key = api_key

        def extract(self, **kwargs):
            calls.append(kwargs)
            return {"results": []}

    monkeypatch.setattr(extract, "TavilyClient", FakeClient)

    extract.extract_content(["https://example.com"])

    assert calls[0]["timeout"] == 19


def test_extract_content_does_not_cache_failed_request(
    monkeypatch,
    tmp_path,
):
    """A failed extraction must not create a successful cache entry."""

    monkeypatch.setenv(
        "TAVILY_API_KEY",
        "test-key",
    )

    _replace_settings(
        monkeypatch,
        cache_dir=str(tmp_path),
        cache_enabled=True,
        tavily_extract_cache_enabled=True,
        tavily_max_retries=0,
    )

    _patch_tavily(
        monkeypatch
    )

    def failing_extract(
        self,
        urls,
    ):
        self.calls.append(
            list(urls)
        )

        raise RuntimeError(
            "authentication failed"
        )

    monkeypatch.setattr(
        FakeTavilyClient,
        "extract",
        failing_extract,
    )

    with pytest.raises(
        RuntimeError,
        match="Tavily extraction failed",
    ):
        extract.extract_content(
            ["https://example.com"]
        )

    cache = extract._get_extract_cache()

    key = extract._build_extract_cache_key(
        ["https://example.com"]
    )

    assert cache.get(
        key
    ) is None


def test_extract_content_retries_retryable_errors(
    monkeypatch,
    tmp_path,
):
    """Retryable Tavily errors should use the configured retry count."""

    monkeypatch.setenv(
        "TAVILY_API_KEY",
        "test-key",
    )

    _replace_settings(
        monkeypatch,
        cache_dir=str(tmp_path),
        cache_enabled=True,
        tavily_extract_cache_enabled=True,
        tavily_max_retries=2,
        retry_base_seconds=0.01,
        retry_max_seconds=0.02,
    )

    _patch_tavily(
        monkeypatch
    )

    sleep_calls = []

    monkeypatch.setattr(
        extract.time,
        "sleep",
        sleep_calls.append,
    )

    response = {
        "results": [
            {
                "url": "https://example.com",
                "raw_content": "Recovered",
            }
        ]
    }

    def retrying_extract(
        self,
        urls,
    ):
        self.calls.append(
            list(urls)
        )

        if len(self.calls) < 3:
            raise RuntimeError(
                "503 service unavailable"
            )

        return response

    monkeypatch.setattr(
        FakeTavilyClient,
        "extract",
        retrying_extract,
    )

    result = extract.extract_content(
        ["https://example.com"]
    )

    assert result == response

    assert len(
        FakeTavilyClient.instances
    ) == 1

    assert FakeTavilyClient.instances[0].calls == [
        ["https://example.com"],
        ["https://example.com"],
        ["https://example.com"],
    ]

    assert sleep_calls == [
        0.01,
        0.02,
    ]


def test_extract_content_does_not_retry_non_retryable_error(
    monkeypatch,
    tmp_path,
):
    """Non-retryable errors should immediately stop extraction."""

    monkeypatch.setenv(
        "TAVILY_API_KEY",
        "test-key",
    )

    _replace_settings(
        monkeypatch,
        cache_dir=str(tmp_path),
        cache_enabled=True,
        tavily_extract_cache_enabled=True,
        tavily_max_retries=3,
    )

    _patch_tavily(
        monkeypatch
    )

    sleep_calls = []

    monkeypatch.setattr(
        extract.time,
        "sleep",
        sleep_calls.append,
    )

    def failing_extract(
        self,
        urls,
    ):
        self.calls.append(
            list(urls)
        )

        raise RuntimeError(
            "401 unauthorized"
        )

    monkeypatch.setattr(
        FakeTavilyClient,
        "extract",
        failing_extract,
    )

    with pytest.raises(
        RuntimeError,
        match="Tavily extraction failed",
    ):
        extract.extract_content(
            ["https://example.com"]
        )

    assert len(
        FakeTavilyClient.instances
    ) == 1

    assert FakeTavilyClient.instances[0].calls == [
        ["https://example.com"]
    ]

    assert sleep_calls == []


def test_extract_content_bypasses_cache_when_globally_disabled(
    monkeypatch,
    tmp_path,
):
    """Global cache disablement should force Tavily calls."""

    monkeypatch.setenv(
        "TAVILY_API_KEY",
        "test-key",
    )

    _replace_settings(
        monkeypatch,
        cache_dir=str(tmp_path),
        cache_enabled=False,
        tavily_extract_cache_enabled=True,
    )

    _patch_tavily(
        monkeypatch
    )

    response = {
        "results": [
            {
                "url": "https://example.com",
                "raw_content": "Fresh",
            }
        ]
    }

    def fake_extract(
        self,
        urls,
    ):
        self.calls.append(
            list(urls)
        )

        return response

    monkeypatch.setattr(
        FakeTavilyClient,
        "extract",
        fake_extract,
    )

    first = extract.extract_content(
        ["https://example.com"]
    )

    second = extract.extract_content(
        ["https://example.com"]
    )

    assert first == response
    assert second == response

    assert len(
        FakeTavilyClient.instances
    ) == 2

    assert all(
        client.calls == [
            ["https://example.com"]
        ]
        for client in FakeTavilyClient.instances
    )


def test_extract_content_bypasses_cache_when_extract_cache_disabled(
    monkeypatch,
    tmp_path,
):
    """Tavily extraction cache disablement should force Tavily calls."""

    monkeypatch.setenv(
        "TAVILY_API_KEY",
        "test-key",
    )

    _replace_settings(
        monkeypatch,
        cache_dir=str(tmp_path),
        cache_enabled=True,
        tavily_extract_cache_enabled=False,
    )

    _patch_tavily(
        monkeypatch
    )

    response = {
        "results": [
            {
                "url": "https://example.com",
                "raw_content": "Fresh",
            }
        ]
    }

    def fake_extract(
        self,
        urls,
    ):
        self.calls.append(
            list(urls)
        )

        return response

    monkeypatch.setattr(
        FakeTavilyClient,
        "extract",
        fake_extract,
    )

    extract.extract_content(
        ["https://example.com"]
    )

    extract.extract_content(
        ["https://example.com"]
    )

    assert len(
        FakeTavilyClient.instances
    ) == 2


def test_extract_cache_key_changes_for_different_urls():
    """Different URL collections must produce different cache keys."""

    first = extract._build_extract_cache_key(
        [
            "https://example.com/a"
        ]
    )

    second = extract._build_extract_cache_key(
        [
            "https://example.com/b"
        ]
    )

    assert first != second


def test_extract_cache_key_preserves_url_order():
    """Different URL ordering should produce different cache keys."""

    first = extract._build_extract_cache_key(
        [
            "https://example.com/a",
            "https://example.com/b",
        ]
    )

    second = extract._build_extract_cache_key(
        [
            "https://example.com/b",
            "https://example.com/a",
        ]
    )

    assert first != second


def test_extract_content_removes_duplicate_and_empty_urls(
    monkeypatch,
    tmp_path,
):
    """Extraction should avoid redundant and empty URL inputs."""

    monkeypatch.setenv(
        "TAVILY_API_KEY",
        "test-key",
    )

    _replace_settings(
        monkeypatch,
        cache_dir=str(tmp_path),
        cache_enabled=False,
        tavily_extract_cache_enabled=False,
    )

    _patch_tavily(
        monkeypatch
    )

    response = {
        "results": []
    }

    def fake_extract(
        self,
        urls,
    ):
        self.calls.append(
            list(urls)
        )

        return response

    monkeypatch.setattr(
        FakeTavilyClient,
        "extract",
        fake_extract,
    )

    result = extract.extract_content(
        [
            "",
            "https://example.com/a",
            "https://example.com/a",
            " ",
            "https://example.com/b",
        ]
    )

    assert result == response

    assert FakeTavilyClient.instances[0].calls == [
        [
            "https://example.com/a",
            "https://example.com/b",
        ]
    ]
