"""Observability tests for Tavily search, retries, and caching."""

from dataclasses import replace
import logging
from unittest.mock import Mock

import pytest

import research.search as search


class FakeSearchCache:
    """Small in-memory cache fake for deterministic tests."""

    def __init__(self, initial_value=None):
        self.initial_value = initial_value
        self.values = {}
        self.get_calls = []
        self.set_calls = []

    def get(self, key):
        self.get_calls.append(key)
        if key in self.values:
            return self.values[key]
        return self.initial_value

    def set(self, key, value):
        self.set_calls.append((key, value))
        self.values[key] = value


@pytest.fixture
def configure_search(monkeypatch):
    """Configure deterministic Tavily settings and disable real sleeping."""
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-secret")
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-secret")

    # Settings is frozen; install a modified copy instead of mutating it.
    monkeypatch.setattr(
        search,
        "SETTINGS",
        replace(
            search.SETTINGS,
            cache_enabled=True,
            tavily_search_cache_enabled=True,
            tavily_max_retries=1,
            retry_base_seconds=0.01,
            retry_max_seconds=0.01,
        ),
    )

    monkeypatch.setattr(search.time, "sleep", Mock())

    fake_cache = FakeSearchCache()
    monkeypatch.setattr(
        search,
        "_get_search_cache",
        Mock(return_value=fake_cache),
    )

    return fake_cache


@pytest.fixture
def caplog_search(caplog):
    """Capture search observability logs."""
    caplog.set_level(logging.INFO)
    return caplog


def test_cache_hit_is_logged_and_skips_tavily_request(
    monkeypatch,
    configure_search,
    caplog_search,
):
    """A valid cache hit should return results without calling Tavily."""
    cached_results = [
        {
            "title": "Cached source",
            "url": "https://example.com/cached",
            "content": "Cached evidence.",
            "score": 0.9,
        }
    ]
    cache = FakeSearchCache(initial_value=cached_results)

    monkeypatch.setattr(
        search,
        "_get_search_cache",
        Mock(return_value=cache),
    )

    client_factory = Mock()
    monkeypatch.setattr(search, "TavilyClient", client_factory)

    results = search.search_web("machine learning", max_results=1)

    assert results == cached_results
    assert "event=cache_hit provider=tavily operation=search" in (
        caplog_search.text
    )
    client_factory.assert_not_called()


def test_success_logs_cache_miss_request_and_cache_write(
    monkeypatch,
    configure_search,
    caplog_search,
):
    """A successful request should log its lifecycle and cache the result."""
    cache = configure_search

    client = Mock()
    client.search.return_value = {
        "results": [
            {
                "title": "Machine learning overview",
                "url": "https://example.com/ml",
                "content": "An overview of machine learning.",
                "score": 0.95,
            }
        ]
    }
    monkeypatch.setattr(
        search,
        "TavilyClient",
        Mock(return_value=client),
    )

    results = search.search_web("machine learning", max_results=1)

    assert results
    assert "event=cache_miss provider=tavily operation=search" in (
        caplog_search.text
    )
    assert "event=request_started provider=tavily operation=search" in (
        caplog_search.text
    )
    assert "event=request_succeeded provider=tavily operation=search" in (
        caplog_search.text
    )
    assert "event=cache_write provider=tavily operation=search" in (
        caplog_search.text
    )
    assert cache.set_calls


def test_retryable_failure_logs_retry_and_then_succeeds(
    monkeypatch,
    configure_search,
    caplog_search,
):
    """A retryable Tavily error should be logged before a successful retry."""
    client = Mock()
    client.search.side_effect = [
        RuntimeError("temporary service unavailable"),
        {
            "results": [
                {
                    "title": "Recovered source",
                    "url": "https://example.com/recovered",
                    "content": "Recovered evidence.",
                    "score": 0.9,
                }
            ]
        },
    ]
    monkeypatch.setattr(
        search,
        "TavilyClient",
        Mock(return_value=client),
    )

    results = search.search_web("retry test", max_results=1)

    assert results
    assert client.search.call_count == 2
    assert "event=request_failed provider=tavily operation=search" in (
        caplog_search.text
    )
    assert "event=retry_scheduled provider=tavily operation=search" in (
        caplog_search.text
    )
    assert "event=request_succeeded provider=tavily operation=search" in (
        caplog_search.text
    )


def test_non_retryable_failure_logs_stop_and_exhaustion(
    monkeypatch,
    configure_search,
    caplog_search,
):
    """A non-retryable Tavily error should stop without another attempt."""
    monkeypatch.setattr(
        search,
        "SETTINGS",
        replace(search.SETTINGS, tavily_max_retries=3),
    )

    client = Mock()
    client.search.side_effect = PermissionError(
        "Permission denied: invalid credentials"
    )
    monkeypatch.setattr(
        search,
        "TavilyClient",
        Mock(return_value=client),
    )

    with pytest.raises(RuntimeError):
        search.search_web("non-retryable test", max_results=1)

    assert client.search.call_count == 1
    assert "event=retry_stopped provider=tavily operation=search" in (
        caplog_search.text
    )


def test_tavily_api_key_is_redacted_from_logs_and_exception(
    monkeypatch,
    configure_search,
    caplog_search,
):
    """The Tavily API key must not appear in logs or raised errors."""
    monkeypatch.setattr(
        search,
        "SETTINGS",
        replace(search.SETTINGS, tavily_max_retries=0),
    )

    client = Mock()
    client.search.side_effect = RuntimeError(
        "Request rejected: test-tavily-secret"
    )
    monkeypatch.setattr(
        search,
        "TavilyClient",
        Mock(return_value=client),
    )

    with pytest.raises(RuntimeError) as exc_info:
        search.search_web("credential redaction test", max_results=1)

    assert "test-tavily-secret" not in caplog_search.text
    assert "test-tavily-secret" not in str(exc_info.value)


def test_failed_search_is_not_cached(
    monkeypatch,
    configure_search,
):
    """Failed requests must not create cache entries."""
    cache = configure_search

    monkeypatch.setattr(
        search,
        "SETTINGS",
        replace(search.SETTINGS, tavily_max_retries=0),
    )

    client = Mock()
    client.search.side_effect = RuntimeError(
        "temporary service unavailable"
    )
    monkeypatch.setattr(
        search,
        "TavilyClient",
        Mock(return_value=client),
    )

    with pytest.raises(RuntimeError):
        search.search_web("failed search cache test", max_results=1)

    assert cache.set_calls == []


def test_missing_api_key_is_logged_without_request(
    monkeypatch,
    configure_search,
    caplog_search,
):
    """A missing Tavily API key should be logged without a network request."""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    client_factory = Mock()
    monkeypatch.setattr(search, "TavilyClient", client_factory)

    with pytest.raises((RuntimeError, ValueError)):
        search.search_web("missing key test", max_results=1)

    assert "event=request_rejected provider=tavily operation=search" in (
        caplog_search.text
    )
    client_factory.assert_not_called()