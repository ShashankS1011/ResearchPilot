
"""Tavily content extraction with retry, caching, and safe observability."""

from __future__ import annotations

import logging
import os
import re
import time
import inspect
from typing import Any
from urllib.parse import urlparse

from tavily import TavilyClient

from research.cache import CacheStore
from research.config import SETTINGS
from research.reliability import (
    retry_after_seconds,
    retry_budget_allows,
    retry_delay,
    sanitize_error_message,
)


logger = logging.getLogger("researchpilot.extract")


def _sanitize_error_message(error: Exception | str) -> str:
    """Remove known credentials and common secret patterns from errors."""

    return sanitize_error_message(error)


def _is_retryable_error(error: Exception) -> bool:
    """Return whether an extraction error appears transient."""

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
    )

    return any(marker in message for marker in retry_markers)


def _retry_delay(attempt: int) -> float:
    """Calculate a bounded exponential backoff delay."""

    return retry_delay(
        attempt,
        SETTINGS.retry_base_seconds,
        SETTINGS.retry_max_seconds,
        SETTINGS.retry_jitter_seconds,
    )


def _normalize_urls(urls: Any) -> list[str]:
    """Normalize extraction URLs and remove duplicates while preserving order."""

    if urls is None:
        return []

    if isinstance(urls, str):
        urls = [urls]

    if not isinstance(urls, (list, tuple)):
        raise TypeError("urls must be a string, list, or tuple.")

    normalized: list[str] = []
    seen: set[str] = set()

    for url in urls:
        value = str(url or "").strip()

        parsed = urlparse(value)

        if (
            not value
            or value in seen
            or parsed.scheme not in {"http", "https"}
            or not parsed.netloc
        ):
            continue

        seen.add(value)
        normalized.append(value)

    return normalized


def _is_valid_extraction_response(value: Any) -> bool:
    """Return whether an extraction response has the expected shape."""

    return (
        isinstance(value, dict)
        and isinstance(value.get("results"), list)
    )


def _extract_request_kwargs() -> dict[str, Any]:
    """Return timeout options supported by the installed Tavily client."""
    try:
        parameters = inspect.signature(TavilyClient.extract).parameters
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


def _build_extract_cache_key(urls: list[str]) -> str:
    """Build a deterministic cache key for an extraction request."""

    return CacheStore.build_key(
        "tavily-extract",
        {"urls": urls},
    )


def _get_extract_cache() -> CacheStore:
    """Return the extraction cache configured by application settings."""

    return CacheStore(
        directory=SETTINGS.cache_dir,
        ttl_seconds=SETTINGS.cache_ttl_seconds,
        enabled=(
            SETTINGS.cache_enabled
            and SETTINGS.tavily_extract_cache_enabled
        ),
    )


def extract_content(urls: Any) -> dict[str, Any]:
    """
    Extract page content from Tavily with retry and persistent caching.

    Only successful dictionary responses are cached. Error messages are
    sanitized before being written to logs or exposed in the final exception.
    """

    normalized_urls = _normalize_urls(urls)

    if not SETTINGS.tavily_search_depth:
        logger.warning(
            "event=configuration_warning provider=tavily "
            "operation=extract reason=search_depth_not_configured"
        )

    if not normalized_urls:
        logger.info(
            "event=request_skipped provider=tavily operation=extract "
            "reason=no_urls"
        )
        return {"results": []}

    api_key = os.getenv("TAVILY_API_KEY")

    if not api_key:
        logger.error(
            "event=request_rejected provider=tavily operation=extract "
            "reason=missing_api_key"
        )
        raise ValueError("TAVILY_API_KEY was not found in .env")

    cache = _get_extract_cache()
    cache_key = _build_extract_cache_key(normalized_urls)

    cached_response = cache.get(cache_key)

    if cached_response is not None:
        if _is_valid_extraction_response(cached_response):
            logger.info(
                "event=cache_hit provider=tavily operation=extract "
                "url_count=%s",
                len(normalized_urls),
            )
            return cached_response

        logger.warning(
            "event=cache_entry_ignored provider=tavily operation=extract "
            "reason=malformed_value",
        )

    logger.info(
        "event=cache_miss provider=tavily operation=extract url_count=%s",
        len(normalized_urls),
    )

    tavily = TavilyClient(api_key=api_key)

    max_retries = max(0, SETTINGS.tavily_max_retries)
    max_attempts = max_retries + 1
    retry_started_at = time.monotonic()

    last_error: Exception | None = None
    attempts_made = 0

    for attempt in range(max_attempts):
        attempts_made = attempt + 1

        logger.info(
            "event=request_started provider=tavily operation=extract "
            "attempt=%s max_attempts=%s url_count=%s",
            attempts_made,
            max_attempts,
            len(normalized_urls),
        )

        try:
            response = tavily.extract(
                urls=normalized_urls,
                **_extract_request_kwargs(),
            )

            if not isinstance(response, dict):
                raise TypeError(
                    "Tavily extraction returned an invalid response type."
                )

            if not _is_valid_extraction_response(response):
                raise TypeError(
                    "Tavily extraction returned malformed results."
                )

            results = response["results"]

            logger.info(
                "event=request_succeeded provider=tavily operation=extract "
                "attempt=%s result_count=%s",
                attempts_made,
                len(results) if isinstance(results, list) else 0,
            )

            try:
                cache.set(cache_key, response)
                logger.info(
                    "event=cache_write provider=tavily operation=extract "
                    "status=success"
                )
            except Exception as cache_error:
                logger.warning(
                    "event=cache_write provider=tavily operation=extract "
                    "status=failed error=%s",
                    _sanitize_error_message(cache_error),
                )

            return response

        except Exception as error:
            last_error = error
            safe_error = _sanitize_error_message(error)
            retryable = _is_retryable_error(error)

            logger.warning(
                "event=request_failed provider=tavily operation=extract "
                "attempt=%s max_attempts=%s retryable=%s error=%s",
                attempts_made,
                max_attempts,
                retryable,
                safe_error,
            )

            if not retryable:
                logger.error(
                    "event=retry_stopped provider=tavily operation=extract "
                    "reason=non_retryable_error attempt=%s",
                    attempts_made,
                )
                break

            if attempts_made >= max_attempts:
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
                    "event=retry_stopped provider=tavily operation=extract "
                    "reason=retry_budget_exhausted",
                )
                break

            logger.info(
                "event=retry_scheduled provider=tavily operation=extract "
                "next_attempt=%s max_attempts=%s delay_seconds=%.2f",
                attempts_made + 1,
                max_attempts,
                delay,
            )

            time.sleep(delay)

    safe_final_error = _sanitize_error_message(
        last_error or "Unknown extraction error."
    )

    logger.error(
        "event=request_exhausted provider=tavily operation=extract "
        "attempts=%s error=%s",
        attempts_made,
        safe_final_error,
    )

    raise RuntimeError(
        f"Tavily extraction failed: {safe_final_error}"
    ) from last_error
