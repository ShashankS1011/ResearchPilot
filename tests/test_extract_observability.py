"""Observability tests for Tavily content extraction."""

from __future__ import annotations

import logging
from typing import Any

import pytest

import research.extract as extract


class FakeCache:
    """Small in-memory cache for isolated tests."""

    def __init__(
        self,
        cached_value: Any = None,
    ) -> None:
        self.cached_value = cached_value
        self.writes: list[tuple[str, Any]] = []

    def get(self, key: str) -> Any:
        return self.cached_value

    def set(
        self,
        key: str,
        value: Any,
    ) -> None:
        self.writes.append((key, value))


class FakeTavilyClient:
    """Configurable Tavily client fake."""

    def __init__(
        self,
        *,
        api_key: str,
        outcomes: list[Any],
    ) -> None:
        self.api_key = api_key
        self.outcomes = list(outcomes)
        self.calls = 0

    def extract(
        self,
        *,
        urls: list[str],
    ) -> dict[str, Any]:
        self.calls += 1

        if not self.outcomes:
            raise AssertionError("No fake extraction outcome configured.")

        outcome = self.outcomes.pop(0)

        if isinstance(outcome, Exception):
            raise outcome

        return outcome


def _install_fake_client(
    monkeypatch: pytest.MonkeyPatch,
    outcomes: list[Any],
) -> FakeTavilyClient:
    """Install a fake Tavily client and return its instance."""

    clients: list[FakeTavilyClient] = []

    def factory(*, api_key: str) -> FakeTavilyClient:
        client = FakeTavilyClient(
            api_key=api_key,
            outcomes=outcomes,
        )
        clients.append(client)
        return client

    monkeypatch.setattr(extract, "TavilyClient", factory)
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-secret")

    # The factory is called once per extraction operation.
    return _ClientFactoryProxy(clients)  # type: ignore[return-value]


class _ClientFactoryProxy:
    """Proxy exposing the client created by the fake factory."""

    def __init__(
        self,
        clients: list[FakeTavilyClient],
    ) -> None:
        self.clients = clients

    @property
    def calls(self) -> int:
        return self.clients[0].calls if self.clients else 0


def test_cache_hit_emits_structured_event(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    expected = {"results": [{"url": "https://example.com"}]}
    cache = FakeCache(cached_value=expected)

    monkeypatch.setattr(extract, "_get_extract_cache", lambda: cache)
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-secret")

    with caplog.at_level(logging.INFO, logger="researchpilot.extract"):
        result = extract.extract_content(["https://example.com"])

    assert result == expected
    assert "event=cache_hit" in caplog.text
    assert "provider=tavily" in caplog.text
    assert "operation=extract" in caplog.text
    assert "url_count=1" in caplog.text


def test_retry_emits_failure_and_scheduled_events(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    outcomes = [
        RuntimeError("HTTP 503 temporarily unavailable"),
        {"results": [{"url": "https://example.com"}]},
    ]
    client = _install_fake_client(monkeypatch, outcomes)
    cache = FakeCache()

    monkeypatch.setattr(extract, "_get_extract_cache", lambda: cache)
    monkeypatch.setattr(extract.time, "sleep", lambda delay: None)

    with caplog.at_level(logging.INFO, logger="researchpilot.extract"):
        result = extract.extract_content(["https://example.com"])

    assert result["results"]
    assert client.calls == 2
    assert "event=cache_miss" in caplog.text
    assert "event=request_started" in caplog.text
    assert "event=request_failed" in caplog.text
    assert "retryable=True" in caplog.text
    assert "event=retry_scheduled" in caplog.text
    assert "event=request_succeeded" in caplog.text
    assert "event=cache_write" in caplog.text


def test_non_retryable_error_does_not_schedule_retry(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    client = _install_fake_client(
        monkeypatch,
        [RuntimeError("HTTP 401 unauthorized")],
    )
    cache = FakeCache()

    monkeypatch.setattr(extract, "_get_extract_cache", lambda: cache)
    monkeypatch.setattr(extract.time, "sleep", lambda delay: None)

    with caplog.at_level(logging.INFO, logger="researchpilot.extract"):
        with pytest.raises(RuntimeError, match="Tavily extraction failed"):
            extract.extract_content(["https://example.com"])

    assert client.calls == 1
    assert "retryable=False" in caplog.text
    assert "event=retry_stopped" in caplog.text
    assert "event=request_exhausted" in caplog.text
    assert "event=retry_scheduled" not in caplog.text
    assert cache.writes == []


def test_api_key_is_redacted_from_logs_and_final_exception(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    secret = "very-sensitive-tavily-key"
    monkeypatch.setenv("TAVILY_API_KEY", secret)

    _install_fake_client(
        monkeypatch,
        [RuntimeError(f"request failed with credential {secret}")],
    )
    monkeypatch.setattr(
        extract,
        "_get_extract_cache",
        lambda: FakeCache(),
    )

    with caplog.at_level(logging.INFO, logger="researchpilot.extract"):
        with pytest.raises(RuntimeError) as exc_info:
            extract.extract_content(["https://example.com"])

    assert secret not in caplog.text
    assert secret not in str(exc_info.value)
    assert "[REDACTED]" in caplog.text
    assert "[REDACTED]" in str(exc_info.value)


def test_empty_url_input_emits_skipped_event(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="researchpilot.extract"):
        result = extract.extract_content([])

    assert result == {"results": []}
    assert "event=request_skipped" in caplog.text
    assert "reason=no_urls" in caplog.text