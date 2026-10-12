"""Provider interfaces used by the research pipeline."""

from __future__ import annotations

from typing import Any, Protocol


class SearchProvider(Protocol):
    """Interface implemented by web search providers."""

    def search(
        self,
        query: str,
        max_results: int,
    ) -> list[dict[str, Any]]:
        """Return normalized search results."""


class ExtractionProvider(Protocol):
    """Interface implemented by source extraction providers."""

    def extract(
        self,
        urls: list[str],
    ) -> dict[str, Any]:
        """Return an extraction response."""


class SynthesisProvider(Protocol):
    """Interface implemented by answer synthesis providers."""

    def generate(
        self,
        query: str,
        evidence: dict[str, Any],
    ) -> str:
        """Generate an answer from structured evidence."""


class FailoverProvider:
    """Try providers in order and retain the last actionable error."""

    def __init__(self, providers: list[Any]) -> None:
        if not providers:
            raise ValueError("At least one provider is required.")
        self.providers = tuple(providers)

    def call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        last_error: Exception | None = None
        for provider in self.providers:
            try:
                return getattr(provider, method)(*args, **kwargs)
            except Exception as error:
                last_error = error
        raise RuntimeError(
            f"All providers failed for {method}: {last_error}"
        ) from last_error


class CancellationToken:
    """Small cooperative cancellation token for workflow callers."""

    def __init__(self) -> None:
        self.is_cancelled = False

    def cancel(self) -> None:
        self.is_cancelled = True


class TavilySearchProvider:
    """Concrete adapter for the existing Tavily search implementation."""

    def search(self, query: str, max_results: int) -> list[dict[str, Any]]:
        from research.search import search_web

        return search_web(query, max_results=max_results)


class TavilyExtractionProvider:
    """Concrete adapter for the existing Tavily extraction implementation."""

    def extract(self, urls: list[str]) -> dict[str, Any]:
        from research.extract import extract_content

        return extract_content(urls)


class GeminiSynthesisProvider:
    """Concrete adapter for the existing Gemini synthesis implementation."""

    def generate(self, query: str, evidence: dict[str, Any]) -> str:
        from research.llm import generate_with_gemini

        return generate_with_gemini(query, evidence)
