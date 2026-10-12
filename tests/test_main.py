from __future__ import annotations

from typing import Any

import pytest

import main


def _search_results() -> list[dict[str, Any]]:
    return [
        {
            "title": "Source One",
            "url": "https://example.com/one",
            "content": "Evidence from source one.",
            "score": 0.9,
        },
        {
            "title": "Source Two",
            "url": "https://example.com/two",
            "content": "Evidence from source two.",
            "score": 0.8,
        },
        {
            "title": "Source Three",
            "url": "https://example.com/three",
            "content": "Evidence from source three.",
            "score": 0.7,
        },
    ]


def test_search_gracefully_returns_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = _search_results()

    monkeypatch.setattr(
        main,
        "search_web",
        lambda query, max_results: expected,
    )

    result = main._search_gracefully(
        "test query",
        3,
    )

    assert result == expected


def test_search_gracefully_handles_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_search(
        query: str,
        max_results: int,
    ) -> list[dict[str, Any]]:
        raise RuntimeError(
            "Tavily search failed: temporary failure"
        )

    monkeypatch.setattr(
        main,
        "search_web",
        failing_search,
    )

    result = main._search_gracefully(
        "test query",
        3,
    )

    assert result is None


def test_search_gracefully_rejects_invalid_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        main,
        "search_web",
        lambda query, max_results: {
            "results": []
        },
    )

    result = main._search_gracefully(
        "test query",
        3,
    )

    assert result is None


def test_search_gracefully_filters_invalid_items(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        main,
        "search_web",
        lambda query, max_results: [
            {
                "url": "https://example.com/one",
            },
            "invalid",
            None,
            {
                "url": "https://example.com/two",
            },
        ],
    )

    result = main._search_gracefully(
        "test query",
        3,
    )

    assert result == [
        {
            "url": "https://example.com/one",
        },
        {
            "url": "https://example.com/two",
        },
    ]


def test_batch_extraction_succeeds_without_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_results = _search_results()

    calls: list[list[str]] = []

    def fake_extract(
        urls: list[str],
    ) -> dict[str, Any]:
        calls.append(urls)

        return {
            "results": [
                {
                    "url": url,
                    "raw_content": f"Content for {url}",
                }
                for url in urls
            ]
        }

    monkeypatch.setattr(
        main,
        "extract_content",
        fake_extract,
    )

    result = main._extract_sources_gracefully(
        search_results
    )

    assert len(result) == 3
    assert calls == [
        [
            "https://example.com/one",
            "https://example.com/two",
            "https://example.com/three",
        ]
    ]


def test_batch_extraction_failure_recovers_individual_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_results = _search_results()

    calls: list[list[str]] = []

    def fake_extract(
        urls: list[str],
    ) -> dict[str, Any]:
        calls.append(urls)

        if len(urls) > 1:
            raise RuntimeError(
                "batch extraction failed"
            )

        return {
            "results": [
                {
                    "url": urls[0],
                    "raw_content": (
                        f"Recovered content for {urls[0]}"
                    ),
                }
            ]
        }

    monkeypatch.setattr(
        main,
        "extract_content",
        fake_extract,
    )

    result = main._extract_sources_gracefully(
        search_results
    )

    assert len(result) == 3

    assert calls == [
        [
            "https://example.com/one",
            "https://example.com/two",
            "https://example.com/three",
        ],
        ["https://example.com/one"],
        ["https://example.com/two"],
        ["https://example.com/three"],
    ]


def test_individual_extraction_failure_does_not_discard_other_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_results = _search_results()

    def fake_extract(
        urls: list[str],
    ) -> dict[str, Any]:
        if len(urls) > 1:
            raise RuntimeError(
                "batch extraction failed"
            )

        if urls[0].endswith("/two"):
            raise RuntimeError(
                "source two cannot be extracted"
            )

        return {
            "results": [
                {
                    "url": urls[0],
                    "raw_content": (
                        f"Recovered content for {urls[0]}"
                    ),
                }
            ]
        }

    monkeypatch.setattr(
        main,
        "extract_content",
        fake_extract,
    )

    result = main._extract_sources_gracefully(
        search_results
    )

    assert len(result) == 2

    recovered_urls = {
        item["url"]
        for item in result
    }

    assert recovered_urls == {
        "https://example.com/one",
        "https://example.com/three",
    }


def test_extraction_failure_returns_empty_list_when_all_sources_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_results = _search_results()

    def fake_extract(
        urls: list[str],
    ) -> dict[str, Any]:
        raise RuntimeError(
            "extraction unavailable"
        )

    monkeypatch.setattr(
        main,
        "extract_content",
        fake_extract,
    )

    result = main._extract_sources_gracefully(
        search_results
    )

    assert result == []


def test_extraction_with_no_urls_returns_empty_list(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def fake_extract(
        urls: list[str],
    ) -> dict[str, Any]:
        nonlocal called
        called = True
        return {
            "results": []
        }

    monkeypatch.setattr(
        main,
        "extract_content",
        fake_extract,
    )

    result = main._extract_sources_gracefully(
        [
            {
                "title": "No URL",
            }
        ]
    )

    assert result == []
    assert called is False


def test_error_sanitization_removes_api_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tavily_key = "tavily-secret-key"
    gemini_key = "gemini-secret-key"

    monkeypatch.setenv(
        "TAVILY_API_KEY",
        tavily_key,
    )

    monkeypatch.setenv(
        "GEMINI_API_KEY",
        gemini_key,
    )

    error = RuntimeError(
        (
            "request failed with "
            f"{tavily_key} and {gemini_key}"
        )
    )

    message = main._sanitize_error_message(
        error
    )

    assert tavily_key not in message
    assert gemini_key not in message
    assert message.count("[REDACTED]") == 2


def test_error_sanitization_preserves_non_secret_details() -> None:
    error = RuntimeError(
        "Tavily returned HTTP 503"
    )

    message = main._sanitize_error_message(
        error
    )

    assert message == "Tavily returned HTTP 503"


def test_perform_research_stops_cleanly_on_search_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        main,
        "_search_gracefully",
        lambda query, max_results: None,
    )

    monkeypatch.setattr(
        main,
        "prepare_sources",
        lambda *args, **kwargs: pytest.fail(
            "prepare_sources should not run after search failure"
        ),
    )

    result = main.perform_research(
        "test query"
    )

    assert result is None


def test_perform_research_continues_with_search_evidence_when_extraction_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    search_results = _search_results()

    monkeypatch.setattr(
        main,
        "_search_gracefully",
        lambda query, max_results: search_results,
    )

    monkeypatch.setattr(
        main,
        "_extract_sources_gracefully",
        lambda results: [],
    )

    prepared_sources = [
        {
            "id": 1,
            "title": "Search evidence source",
            "url": "https://example.com/one",
            "evidence": [
                {
                    "text": "Search snippet evidence.",
                }
            ],
        }
    ]

    monkeypatch.setattr(
        main,
        "prepare_sources",
        lambda query, search, extracted: prepared_sources,
    )

    monkeypatch.setattr(
        main,
        "print_sources",
        lambda sources: None,
    )

    result = main.perform_research(
        "test query"
    )

    assert result == prepared_sources


def test_generate_and_display_answer_survives_unexpected_llm_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = [
        {
            "id": 1,
            "title": "Source",
            "url": "https://example.com",
            "evidence": [
                {
                    "text": (
                        "This is useful evidence for the research question."
                    )
                }
            ],
        }
    ]

    monkeypatch.setattr(
        main,
        "generate_answer",
        lambda query, sources: (_ for _ in ()).throw(
            RuntimeError("Gemini unavailable")
        ),
    )

    monkeypatch.setattr(
        main,
        "print_stage",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        main,
        "print_status",
        lambda *args, **kwargs: None,
    )

    monkeypatch.setattr(
        main,
        "print_report",
        lambda report: None,
    )

    monkeypatch.setattr(
        main,
        "save_report",
        lambda report: None,
    )

    fallback_report = {
        "executive_answer": (
            "Evidence-based fallback answer."
        )
    }

    monkeypatch.setattr(
        main,
        "build_evidence_fallback_report",
        lambda query, sources, mode: fallback_report,
    )

    result = main.generate_and_display_answer(
        "test query",
        sources,
    )

    assert result == "Evidence-based fallback answer."


def test_run_followup_uses_graceful_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        main,
        "load_session",
        lambda: {
            "query": "what is machine learning?",
            "topic": "machine learning",
        },
    )

    monkeypatch.setattr(
        "builtins.input",
        lambda prompt: "what are its types?",
    )

    monkeypatch.setattr(
        main,
        "_search_gracefully",
        lambda query, max_results: None,
    )

    monkeypatch.setattr(
        main,
        "print_stage",
        lambda *args, **kwargs: None,
    )

    result = main.run_followup()

    assert result is None