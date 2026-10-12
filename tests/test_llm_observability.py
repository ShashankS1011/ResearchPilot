"""Observability tests for Gemini synthesis."""

import logging
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from research import llm


@pytest.fixture
def gemini_ledger():
    """Return evidence using the schema expected by Gemini context building."""
    return {
        "query": "How does machine learning work?",
        "intent": "mechanism",
        "definition": {
            "text": (
                "Machine learning is a method "
                "for learning patterns from data."
            ),
            "source_id": 1,
        },
        "mechanism": {
            "text": (
                "Machine learning works by training "
                "algorithms on data to identify patterns."
            ),
            "source_id": 2,
        },
        "types": {
            "category": {
                "text": "Supervised learning",
                "source_id": 3,
            },
        },
    }


@pytest.fixture
def caplog_gemini(caplog):
    """Capture logs emitted by the Gemini module."""
    caplog.set_level(
        logging.DEBUG,
        logger="researchpilot.llm",
    )
    return caplog


@pytest.fixture
def configure_gemini(monkeypatch):
    """Configure a mocked Gemini client with one retry."""
    monkeypatch.setenv(
        "GEMINI_API_KEY",
        "test-gemini-secret",
    )

    monkeypatch.setattr(
        llm,
        "SETTINGS",
        replace(
            llm.SETTINGS,
            cache_enabled=False,
            gemini_cache_enabled=False,
            gemini_max_retries=1,
        ),
    )

    client = Mock()
    client.interactions.create.return_value = SimpleNamespace(
        output_text="Machine learning identifies patterns."
    )

    monkeypatch.setattr(
        llm,
        "_get_gemini_client",
        Mock(return_value=client),
    )

    monkeypatch.setattr(
        llm.time,
        "sleep",
        Mock(),
    )

    return client


def test_retryable_failure_logs_failure_and_retry(
    configure_gemini,
    gemini_ledger,
    caplog_gemini,
):
    """A transient Gemini error should be logged and retried."""
    client = configure_gemini

    client.interactions.create.side_effect = [
        RuntimeError("service temporarily unavailable"),
        SimpleNamespace(
            output_text="Machine learning identifies patterns."
        ),
    ]

    result = llm.generate_with_gemini(
        "How does machine learning work?",
        gemini_ledger,
    )

    assert "Machine learning identifies patterns." in result
    assert client.interactions.create.call_count == 2

    messages = [
        record.getMessage()
        for record in caplog_gemini.records
    ]

    assert any(
        "Gemini generation failed on attempt" in message
        for message in messages
    )
    assert any(
        "Retrying Gemini generation" in message
        for message in messages
    )


def test_non_retryable_failure_is_logged_without_retry(
    configure_gemini,
    gemini_ledger,
    caplog_gemini,
):
    """A non-retryable Gemini error should not be retried."""
    client = configure_gemini

    client.interactions.create.side_effect = PermissionError(
        "Permission denied: invalid credentials"
    )

    with pytest.raises(
        RuntimeError,
        match="Gemini synthesis failed",
    ):
        llm.generate_with_gemini(
            "How does machine learning work?",
            gemini_ledger,
        )

    assert client.interactions.create.call_count == 1

    messages = [
        record.getMessage()
        for record in caplog_gemini.records
    ]

    assert any(
        "Gemini error is not retryable" in message
        for message in messages
    )


def test_api_key_is_redacted_from_logs_and_final_exception(
    configure_gemini,
    gemini_ledger,
    caplog_gemini,
):
    """Credentials must not leak into logs or raised exceptions."""
    client = configure_gemini
    secret = "test-gemini-secret"

    client.interactions.create.side_effect = PermissionError(
        f"Request rejected: API key={secret}"
    )

    with pytest.raises(RuntimeError) as exc_info:
        llm.generate_with_gemini(
            "How does machine learning work?",
            gemini_ledger,
        )

    logged_messages = "\n".join(
        record.getMessage()
        for record in caplog_gemini.records
    )

    assert secret not in logged_messages
    assert secret not in str(exc_info.value)


def test_exhausted_retries_are_logged(
    configure_gemini,
    gemini_ledger,
    caplog_gemini,
):
    """Exhausting retries should emit a final log event."""
    client = configure_gemini

    client.interactions.create.side_effect = [
        RuntimeError("service temporarily unavailable"),
        RuntimeError("service temporarily unavailable"),
    ]

    with pytest.raises(
        RuntimeError,
        match="Gemini synthesis failed",
    ):
        llm.generate_with_gemini(
            "How does machine learning work?",
            gemini_ledger,
        )

    assert client.interactions.create.call_count == 2

    messages = [
        record.getMessage()
        for record in caplog_gemini.records
    ]

    assert any(
        "Gemini generation exhausted all retry attempts"
        in message
        for message in messages
    )


def test_cache_hit_is_logged_and_skips_gemini_request(
    monkeypatch,
    gemini_ledger,
    caplog_gemini,
):
    """A cached answer should bypass the Gemini client."""
    monkeypatch.setenv(
        "GEMINI_API_KEY",
        "test-gemini-secret",
    )

    monkeypatch.setattr(
        llm,
        "SETTINGS",
        replace(
            llm.SETTINGS,
            cache_enabled=True,
            gemini_cache_enabled=True,
        ),
    )

    cached_answer = "Cached Gemini answer."

    cache = Mock()
    cache.get.return_value = cached_answer

    monkeypatch.setattr(
        llm,
        "_get_gemini_cache",
        Mock(return_value=cache),
    )

    client_factory = Mock()

    monkeypatch.setattr(
        llm,
        "_get_gemini_client",
        client_factory,
    )

    result = llm.generate_with_gemini(
        "How does machine learning work?",
        gemini_ledger,
    )

    assert cached_answer in result
    cache.get.assert_called_once()
    cache.set.assert_not_called()
    client_factory.assert_not_called()

    messages = [
        record.getMessage()
        for record in caplog_gemini.records
    ]

    assert any(
        "Gemini cache hit" in message
        for message in messages
    )


def test_cache_miss_and_successful_write_are_logged(
    monkeypatch,
    gemini_ledger,
    caplog_gemini,
):
    """A cache miss followed by success should store the answer."""
    monkeypatch.setenv(
        "GEMINI_API_KEY",
        "test-gemini-secret",
    )

    monkeypatch.setattr(
        llm,
        "SETTINGS",
        replace(
            llm.SETTINGS,
            cache_enabled=True,
            gemini_cache_enabled=True,
            gemini_max_retries=0,
        ),
    )

    cache = Mock()
    cache.get.return_value = None

    monkeypatch.setattr(
        llm,
        "_get_gemini_cache",
        Mock(return_value=cache),
    )

    client = Mock()
    client.interactions.create.return_value = SimpleNamespace(
        output_text="Machine learning identifies patterns."
    )

    monkeypatch.setattr(
        llm,
        "_get_gemini_client",
        Mock(return_value=client),
    )

    result = llm.generate_with_gemini(
        "How does machine learning work?",
        gemini_ledger,
    )

    assert "Machine learning identifies patterns." in result
    cache.get.assert_called_once()
    cache.set.assert_called_once()
    client.interactions.create.assert_called_once()

    messages = [
        record.getMessage()
        for record in caplog_gemini.records
    ]

    assert any(
        "Gemini cache miss" in message
        for message in messages
    )
    assert any(
        "Gemini response cached" in message
        for message in messages
    )


def test_failed_gemini_response_is_not_cached(
    monkeypatch,
    gemini_ledger,
):
    """Failed Gemini requests must not write cache entries."""
    monkeypatch.setenv(
        "GEMINI_API_KEY",
        "test-gemini-secret",
    )

    monkeypatch.setattr(
        llm,
        "SETTINGS",
        replace(
            llm.SETTINGS,
            cache_enabled=True,
            gemini_cache_enabled=True,
            gemini_max_retries=0,
        ),
    )

    cache = Mock()
    cache.get.return_value = None

    monkeypatch.setattr(
        llm,
        "_get_gemini_cache",
        Mock(return_value=cache),
    )

    client = Mock()
    client.interactions.create.side_effect = RuntimeError(
        "service temporarily unavailable"
    )

    monkeypatch.setattr(
        llm,
        "_get_gemini_client",
        Mock(return_value=client),
    )

    with pytest.raises(
        RuntimeError,
        match="Gemini synthesis failed",
    ):
        llm.generate_with_gemini(
            "How does machine learning work?",
            gemini_ledger,
        )

    client.interactions.create.assert_called_once()
    cache.set.assert_not_called()


def test_malformed_gemini_cache_entry_is_ignored(
    monkeypatch,
    gemini_ledger,
):
    monkeypatch.setenv(
        "GEMINI_API_KEY",
        "test-gemini-secret",
    )
    monkeypatch.setattr(
        llm,
        "SETTINGS",
        replace(
            llm.SETTINGS,
            cache_enabled=True,
            gemini_cache_enabled=True,
            gemini_max_retries=0,
        ),
    )

    cache = Mock()
    cache.get.return_value = {"answer": "not text"}
    monkeypatch.setattr(
        llm,
        "_get_gemini_cache",
        Mock(return_value=cache),
    )

    client = Mock()
    client.interactions.create.return_value = SimpleNamespace(
        output_text="Fresh answer."
    )
    monkeypatch.setattr(
        llm,
        "_get_gemini_client",
        Mock(return_value=client),
    )

    result = llm.generate_with_gemini(
        "How does machine learning work?",
        gemini_ledger,
    )

    assert "Fresh answer." in result
    client.interactions.create.assert_called_once()