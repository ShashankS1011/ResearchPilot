
from dataclasses import replace
from types import SimpleNamespace

import pytest

from research import search


def _replace_settings(monkeypatch, **changes):
    """Replace the frozen Settings instance for an isolated test."""

    monkeypatch.setattr(
        search,
        "SETTINGS",
        replace(
            search.SETTINGS,
            **changes,
        ),
    )


def test_search_web_uses_cache_on_second_call(
    monkeypatch,
    tmp_path,
):
    calls = []

    class FakeTavily:
        def __init__(self, api_key):
            self.api_key = api_key

        def search(
            self,
            query,
            search_depth,
            max_results,
        ):
            calls.append(
                {
                    "query": query,
                    "search_depth": search_depth,
                    "max_results": max_results,
                }
            )

            return {
                "results": [
                    {
                        "title": "Example",
                        "url": "https://example.com/page",
                        "content": "Example content",
                    }
                ]
            }

    monkeypatch.setattr(
        search,
        "TavilyClient",
        FakeTavily,
    )

    monkeypatch.setenv(
        "TAVILY_API_KEY",
        "test-key",
    )

    _replace_settings(
        monkeypatch,
        cache_dir=str(tmp_path),
        cache_enabled=True,
        tavily_search_cache_enabled=True,
    )

    first = search.search_web(
        "machine learning",
        max_results=1,
    )

    second = search.search_web(
        "machine learning",
        max_results=1,
    )

    assert first == second
    assert len(calls) == 1


def test_search_web_does_not_cache_failed_request(
    monkeypatch,
    tmp_path,
):
    calls = []

    class FakeTavily:
        def __init__(self, api_key):
            self.api_key = api_key

        def search(
            self,
            query,
            search_depth,
            max_results,
        ):
            calls.append(query)

            raise RuntimeError(
                "401 unauthorized"
            )

    monkeypatch.setattr(
        search,
        "TavilyClient",
        FakeTavily,
    )

    monkeypatch.setenv(
        "TAVILY_API_KEY",
        "test-key",
    )

    _replace_settings(
        monkeypatch,
        cache_dir=str(tmp_path),
        cache_enabled=True,
        tavily_search_cache_enabled=True,
        tavily_max_retries=0,
    )

    with pytest.raises(
        RuntimeError,
        match="Tavily search failed",
    ):
        search.search_web(
            "failing query"
        )

    assert len(calls) == 1

    assert list(
        tmp_path.glob("*.json")
    ) == []


def test_search_cache_key_changes_with_result_limit():
    first = search._build_search_cache_key(
        "machine learning",
        3,
    )

    second = search._build_search_cache_key(
        "machine learning",
        5,
    )

    assert first != second


def test_search_cache_key_changes_with_search_depth(
    monkeypatch,
):
    _replace_settings(
        monkeypatch,
        tavily_search_depth="advanced",
    )

    first = search._build_search_cache_key(
        "machine learning",
        3,
    )

    _replace_settings(
        monkeypatch,
        tavily_search_depth="basic",
    )

    second = search._build_search_cache_key(
        "machine learning",
        3,
    )

    assert first != second


def test_search_web_bypasses_cache_when_disabled(
    monkeypatch,
    tmp_path,
):
    calls = []

    class FakeTavily:
        def __init__(self, api_key):
            self.api_key = api_key

        def search(
            self,
            query,
            search_depth,
            max_results,
        ):
            calls.append(query)

            return {
                "results": [
                    {
                        "title": "Example",
                        "url": (
                            "https://example.com/page"
                        ),
                    }
                ]
            }

    monkeypatch.setattr(
        search,
        "TavilyClient",
        FakeTavily,
    )

    monkeypatch.setenv(
        "TAVILY_API_KEY",
        "test-key",
    )

    _replace_settings(
        monkeypatch,
        cache_dir=str(tmp_path),
        cache_enabled=False,
        tavily_search_cache_enabled=True,
    )

    first = search.search_web(
        "machine learning"
    )

    second = search.search_web(
        "machine learning"
    )

    assert first == second
    assert len(calls) == 2

    assert list(
        tmp_path.glob("*.json")
    ) == []


def test_search_web_cache_respects_tavily_cache_setting(
    monkeypatch,
    tmp_path,
):
    calls = []

    class FakeTavily:
        def __init__(self, api_key):
            self.api_key = api_key

        def search(
            self,
            query,
            search_depth,
            max_results,
        ):
            calls.append(query)

            return {
                "results": [
                    {
                        "title": "Example",
                        "url": (
                            "https://example.com/page"
                        ),
                    }
                ]
            }

    monkeypatch.setattr(
        search,
        "TavilyClient",
        FakeTavily,
    )

    monkeypatch.setenv(
        "TAVILY_API_KEY",
        "test-key",
    )

    _replace_settings(
        monkeypatch,
        cache_dir=str(tmp_path),
        cache_enabled=True,
        tavily_search_cache_enabled=False,
    )

    search.search_web(
        "machine learning"
    )

    search.search_web(
        "machine learning"
    )

    assert len(calls) == 2

    assert list(
        tmp_path.glob("*.json")
    ) == []


def test_search_web_caches_selected_results_not_raw_results(
    monkeypatch,
    tmp_path,
):
    class FakeTavily:
        def __init__(self, api_key):
            self.api_key = api_key

        def search(
            self,
            query,
            search_depth,
            max_results,
        ):
            return {
                "results": [
                    {
                        "title": "First",
                        "url": "https://one.example/a",
                    },
                    {
                        "title": "Second",
                        "url": "https://two.example/b",
                    },
                    {
                        "title": "Third",
                        "url": "https://one.example/c",
                    },
                ]
            }

    monkeypatch.setattr(
        search,
        "TavilyClient",
        FakeTavily,
    )

    monkeypatch.setenv(
        "TAVILY_API_KEY",
        "test-key",
    )

    _replace_settings(
        monkeypatch,
        cache_dir=str(tmp_path),
        cache_enabled=True,
        tavily_search_cache_enabled=True,
    )

    results = search.search_web(
        "test",
        max_results=2,
    )

    assert len(results) == 2

    assert {
        item["title"]
        for item in results
    } == {
        "First",
        "Second",
    }


def test_search_web_ignores_malformed_cached_results(
    monkeypatch,
):
    cache = type(
        "Cache",
        (),
        {
            "get": lambda self, key: [
                {"title": "Missing URL"},
                "not a result",
            ],
            "set": lambda self, key, value: None,
        },
    )()
    client = type(
        "Client",
        (),
        {
            "search": lambda self, **kwargs: {
                "results": [
                    {
                        "title": "Fresh result",
                        "url": "https://example.com/fresh",
                    }
                ]
            }
        },
    )()

    monkeypatch.setattr(search, "_get_search_cache", lambda: cache)
    monkeypatch.setattr(search, "TavilyClient", lambda api_key: client)
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")

    results = search.search_web("test", max_results=1)

    assert results[0]["title"] == "Fresh result"


def test_search_web_skips_results_without_absolute_urls(
    monkeypatch,
):
    class FakeTavily:
        def __init__(self, api_key):
            self.api_key = api_key

        def search(self, **kwargs):
            return {
                "results": [
                    {"title": "Missing URL"},
                    {"title": "Relative", "url": "example.com/page"},
                    {
                        "title": "Valid",
                        "url": "https://example.com/page",
                    },
                ]
            }

    monkeypatch.setattr(search, "TavilyClient", FakeTavily)
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    _replace_settings(
        monkeypatch,
        cache_enabled=False,
        tavily_search_cache_enabled=False,
    )

    results = search.search_web("test", max_results=3)

    assert [item["title"] for item in results] == ["Valid"]


def test_search_web_passes_configured_timeout(monkeypatch):
    calls = []

    class FakeTavily:
        def __init__(self, api_key):
            self.api_key = api_key

        def search(self, **kwargs):
            calls.append(kwargs)
            return {"results": []}

    monkeypatch.setattr(search, "TavilyClient", FakeTavily)
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    _replace_settings(
        monkeypatch,
        cache_enabled=False,
        tavily_search_cache_enabled=False,
        request_timeout_seconds=17,
    )

    search.search_web("timeout test", max_results=1)

    assert calls[0]["timeout"] == 17


def test_search_research_plan_benefits_from_search_cache(
    monkeypatch,
    tmp_path,
):
    calls = []

    class FakeTavily:
        def __init__(self, api_key):
            self.api_key = api_key

        def search(
            self,
            query,
            search_depth,
            max_results,
        ):
            calls.append(query)

            return {
                "results": [
                    {
                        "title": query,
                        "url": (
                            "https://example.com/"
                            f"{query.replace(' ', '-')}"
                        ),
                    }
                ]
            }

    monkeypatch.setattr(
        search,
        "TavilyClient",
        FakeTavily,
    )

    monkeypatch.setenv(
        "TAVILY_API_KEY",
        "test-key",
    )

    _replace_settings(
        monkeypatch,
        cache_dir=str(tmp_path),
        cache_enabled=True,
        tavily_search_cache_enabled=True,
    )

    plan = SimpleNamespace(
        queries=[
            "query one",
            "query two",
        ]
    )

    first = search.search_research_plan(
        plan,
        max_results=4,
    )

    second = search.search_research_plan(
        plan,
        max_results=4,
    )

    assert first == second

    assert calls == [
        "query one",
        "query two",
    ]


def test_search_research_plan_continues_after_query_failure(
    monkeypatch,
):
    calls = []

    def fake_search_web(
        query,
        max_results,
    ):
        calls.append(query)

        if query == "query two":
            raise RuntimeError(
                "Tavily search failed: temporary outage"
            )

        return [
            {
                "title": query,
                "url": (
                    "https://example.com/"
                    f"{query.replace(' ', '-')}"
                ),
            }
        ]

    monkeypatch.setattr(
        search,
        "search_web",
        fake_search_web,
    )

    plan = SimpleNamespace(
        queries=[
            "query one",
            "query two",
            "query three",
        ]
    )

    result = search.search_research_plan(
        plan,
        max_results=3,
    )

    assert calls == [
        "query one",
        "query two",
        "query three",
    ]

    assert [
        item["research_query"]
        for item in result
    ] == [
        "query one",
        "query three",
    ]


def test_search_research_plan_returns_empty_when_all_queries_fail(
    monkeypatch,
):
    calls = []

    def fake_search_web(
        query,
        max_results,
    ):
        calls.append(query)

        raise RuntimeError(
            "Tavily search failed: service unavailable"
        )

    monkeypatch.setattr(
        search,
        "search_web",
        fake_search_web,
    )

    plan = SimpleNamespace(
        queries=[
            "query one",
            "query two",
        ]
    )

    result = search.search_research_plan(
        plan,
        max_results=4,
    )

    assert result == []

    assert calls == [
        "query one",
        "query two",
    ]
