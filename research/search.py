
"""Tavily search with retry, caching, deduplication, and source diversity."""

from __future__ import annotations

import os
import re
import time
import inspect
from typing import Any
from urllib.parse import urlparse, urlunparse

from tavily import TavilyClient

from .cache import CacheStore
from .config import SETTINGS
from .logging_config import logger
from .reliability import (
    retry_after_seconds,
    retry_budget_allows,
    retry_delay,
    sanitize_error_message,
)


def _sanitize_error_message(error: Exception | str) -> str:
    """Redact known credentials and common credential-labelled values."""

    return sanitize_error_message(error)


def _is_retryable_error(error: Exception) -> bool:
    """Return True when an exception is likely temporary."""

    message = str(error).lower()

    retry_markers = (
        "429",
        "500",
        "502",
        "503",
        "504",
        "timeout",
        "timed out",
        "temporarily unavailable",
        "connection",
        "rate limit",
        "rate_limit",
        "server error",
        "internal server",
        "service unavailable",
    )

    return any(marker in message for marker in retry_markers)


def _retry_delay(attempt: int) -> float:
    """Calculate a bounded exponential retry delay."""

    return retry_delay(
        attempt,
        SETTINGS.retry_base_seconds,
        SETTINGS.retry_max_seconds,
        SETTINGS.retry_jitter_seconds,
    )


def _canonical_url(url: Any) -> str:
    """Normalize a URL enough to identify duplicate search results."""

    try:
        parsed = urlparse(str(url or "").strip())
    except Exception:
        return ""

    hostname = (parsed.hostname or "").lower()

    if not hostname:
        return ""

    if hostname.startswith("www."):
        hostname = hostname[4:]

    path = parsed.path.rstrip("/") or "/"

    return urlunparse(
        (
            parsed.scheme.lower() or "https",
            hostname,
            path,
            "",
            parsed.query,
            "",
        )
    )


def _domain(url: Any) -> str:
    """Return a normalized hostname."""

    try:
        hostname = (
            urlparse(str(url or "")).hostname or ""
        ).lower()
    except Exception:
        return ""

    if hostname.startswith("www."):
        hostname = hostname[4:]

    return hostname


def _select_diverse_results(
    results: Any,
    max_results: int,
) -> list[dict[str, Any]]:
    """
    Deduplicate URLs and prefer distinct domains without discarding
    useful results.
    """

    if not isinstance(results, list) or max_results <= 0:
        return []

    unique: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    seen_domains: set[str] = set()

    # First pass: select at most one result per domain.
    for result in results:
        if not isinstance(result, dict):
            continue

        url = str(result.get("url") or "").strip()

        if not url:
            continue

        canonical = _canonical_url(url)

        if not canonical or canonical in seen_urls:
            continue

        domain = _domain(url)

        if domain in seen_domains:
            continue

        seen_urls.add(canonical)
        seen_domains.add(domain)
        unique.append(result)

        if len(unique) >= max_results:
            return unique

    # Second pass: fill remaining slots with other unique URLs.
    for result in results:
        if not isinstance(result, dict):
            continue

        url = str(result.get("url") or "").strip()
        canonical = _canonical_url(url)

        if not url or not canonical or canonical in seen_urls:
            continue

        seen_urls.add(canonical)
        unique.append(result)

        if len(unique) >= max_results:
            break

    return unique


def _is_valid_cached_results(value: Any) -> bool:
    """Return whether a cached search result has the expected shape."""

    if not isinstance(value, list):
        return False

    return all(
        isinstance(result, dict)
        and bool(_canonical_url(result.get("url")))
        for result in value
    )


def _search_request_kwargs() -> dict[str, Any]:
    """Return timeout options supported by the installed Tavily client."""
    try:
        parameters = inspect.signature(TavilyClient.search).parameters
    except (AttributeError, TypeError, ValueError):
        return {}
    if (
        "timeout" not in parameters
        and not any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
    ):
        return {}
    return {"timeout": SETTINGS.request_timeout_seconds}


def _build_search_cache_key(
    query: str,
    max_results: int,
) -> str:
    """Build a deterministic cache key for a Tavily search."""

    return CacheStore.build_key(
        "tavily-search",
        {
            "query": str(query or "").strip(),
            "max_results": int(max_results),
            "search_depth": SETTINGS.tavily_search_depth,
        },
    )


def _get_search_cache() -> CacheStore:
    """Return the configured Tavily search cache."""

    return CacheStore(
        directory=SETTINGS.cache_dir,
        ttl_seconds=SETTINGS.cache_ttl_seconds,
        enabled=(
            SETTINGS.cache_enabled
            and SETTINGS.tavily_search_cache_enabled
        ),
    )


def search_web(
    query: str,
    max_results: int = 5,
) -> list[dict[str, Any]]:
    """Search Tavily with caching, retry/backoff, and source diversity."""

    query = str(query or "").strip()

    if not query:
        raise ValueError("Search query cannot be empty.")

    requested = max(int(max_results), 1)
    cache = _get_search_cache()
    cache_key = _build_search_cache_key(query, requested)

    cached_results = cache.get(cache_key)

    if cached_results is not None:
        if _is_valid_cached_results(cached_results):
            logger.info(
                "event=cache_hit provider=tavily operation=search "
                "max_results=%s",
                requested,
            )
            return cached_results

        logger.warning(
            "event=cache_entry_ignored provider=tavily operation=search "
            "reason=malformed_value",
        )

    logger.info(
        "event=cache_miss provider=tavily operation=search "
        "max_results=%s",
        requested,
    )

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        logger.error(
            "event=request_rejected provider=tavily operation=search "
            "reason=missing_api_key",
        )
        raise ValueError("TAVILY_API_KEY was not found in .env")

    tavily = TavilyClient(api_key=api_key)
    last_error: Exception | None = None

    max_retries = max(0, SETTINGS.tavily_max_retries)
    max_attempts = max_retries + 1
    retry_started_at = time.monotonic()

    tavily_limit = min(max(requested * 2, requested), 20)

    for attempt in range(max_attempts):
        attempt_number = attempt + 1

        logger.info(
            "event=request_started provider=tavily operation=search "
            "attempt=%s max_attempts=%s max_results=%s",
            attempt_number,
            max_attempts,
            requested,
        )

        try:
            response = tavily.search(
                query=query,
                search_depth=SETTINGS.tavily_search_depth,
                max_results=tavily_limit,
                **_search_request_kwargs(),
            )

            if not isinstance(response, dict):
                logger.warning(
                    "event=request_invalid_response provider=tavily "
                    "operation=search reason=response_not_dict",
                )
                return []

            results = response.get("results", [])

            if not isinstance(results, list):
                logger.warning(
                    "event=request_invalid_response provider=tavily "
                    "operation=search reason=results_not_list",
                )
                return []

            selected = _select_diverse_results(
                results,
                requested,
            )

            logger.info(
                "event=request_succeeded provider=tavily operation=search "
                "result_count=%s selected_count=%s",
                len(results),
                len(selected),
            )

            # Cache only successful, selected results.
            cache.set(cache_key, selected)

            logger.info(
                "event=cache_write provider=tavily operation=search "
                "status=success result_count=%s",
                len(selected),
            )

            return selected

        except Exception as error:
            last_error = error
            safe_error = _sanitize_error_message(error)
            retryable = _is_retryable_error(error)

            logger.warning(
                "event=request_failed provider=tavily operation=search "
                "attempt=%s max_attempts=%s retryable=%s error=%s",
                attempt_number,
                max_attempts,
                retryable,
                safe_error,
            )

            if not retryable:
                logger.error(
                    "event=retry_stopped provider=tavily operation=search "
                    "reason=non_retryable_error attempt=%s",
                    attempt_number,
                )
                break

            if attempt_number >= max_attempts:
                break

            delay = retry_delay(
                attempt,
                SETTINGS.retry_base_seconds,
                SETTINGS.retry_max_seconds,
                SETTINGS.retry_jitter_seconds,
                retry_after_seconds(error),
            )

            if not retry_budget_allows(
                retry_started_at,
                delay,
                SETTINGS.retry_budget_seconds,
            ):
                logger.warning(
                    "event=retry_stopped provider=tavily operation=search "
                    "reason=retry_budget_exhausted",
                )
                break

            logger.info(
                "event=retry_scheduled provider=tavily operation=search "
                "next_attempt=%s max_attempts=%s delay_seconds=%.2f",
                attempt_number + 1,
                max_attempts,
                delay,
            )

            time.sleep(delay)

    safe_final_error = _sanitize_error_message(
        last_error or "Unknown search error."
    )

    logger.error(
        "event=request_exhausted provider=tavily operation=search "
        "attempts=%s error=%s",
        attempt_number,
        safe_final_error,
    )

    raise RuntimeError(
        f"Tavily search failed: {safe_final_error}"
    ) from last_error


def search_research_plan(
    plan: Any,
    max_results: int = 5,
) -> list[dict[str, Any]]:
    """
    Execute planned searches and globally deduplicate results.

    Individual query failures are isolated from the rest of the plan.
    Failed queries are skipped so successful queries can still contribute
    evidence. If every query fails, an empty result list is returned.
    """

    if plan is None:
        return []

    queries = [
        str(query).strip()
        for query in getattr(plan, "queries", ())
        if str(query).strip()
    ]

    if not queries:
        logger.info(
            "event=research_plan_skipped provider=tavily "
            "reason=no_queries",
        )
        return []

    requested = max(int(max_results), 1)

    per_query = max(
        2,
        (requested + len(queries) - 1) // len(queries),
    )

    combined: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    failed_queries = 0

    logger.info(
        "event=research_plan_started provider=tavily "
        "query_count=%s max_results=%s",
        len(queries),
        requested,
    )

    for query_index, query in enumerate(queries, start=1):
        try:
            results = search_web(
                query,
                max_results=per_query,
            )
        except (RuntimeError, ValueError) as error:
            failed_queries += 1

            logger.warning(
                "event=research_plan_query_failed provider=tavily "
                "query_index=%s query_count=%s error=%s",
                query_index,
                len(queries),
                _sanitize_error_message(error),
            )
            continue

        for result in results:
            if not isinstance(result, dict):
                continue

            canonical = _canonical_url(result.get("url", ""))

            if not canonical or canonical in seen_urls:
                continue

            seen_urls.add(canonical)

            enriched = dict(result)

            # Preserve query provenance for downstream evidence processing.
            enriched["research_query"] = query

            combined.append(enriched)

    selected = _select_diverse_results(
        combined,
        requested,
    )

    logger.info(
        "event=research_plan_completed provider=tavily "
        "query_count=%s failed_query_count=%s combined_count=%s "
        "selected_count=%s",
        len(queries),
        failed_queries,
        len(combined),
        len(selected),
    )

    return selected
