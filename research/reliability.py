"""Shared reliability helpers for provider requests."""

from __future__ import annotations

import random
import re
import time
from email.utils import parsedate_to_datetime
from typing import Any


def retry_after_seconds(error: Any) -> float | None:
    """Return a valid Retry-After delay from an exception, if present."""

    headers = getattr(error, "headers", None)
    if headers is None:
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None)

    if headers is None:
        return None

    try:
        value = headers.get("Retry-After")
    except AttributeError:
        return None

    if value is None:
        return None

    try:
        delay = float(str(value).strip())
    except ValueError:
        try:
            target = parsedate_to_datetime(str(value).strip()).timestamp()
        except (TypeError, ValueError, OverflowError):
            return None
        delay = target - time.time()

    return max(delay, 0.0) if delay >= 0 else None


def retry_delay(
    attempt: int,
    base_seconds: float,
    max_seconds: float,
    jitter_seconds: float = 0.0,
    retry_after: float | None = None,
    random_value: float | None = None,
) -> float:
    """Calculate bounded exponential backoff with optional jitter."""

    exponential = min(
        max(float(base_seconds), 0.0) * (2**max(attempt, 0)),
        max(float(max_seconds), 0.0),
    )
    delay = max(exponential, retry_after or 0.0)

    if jitter_seconds > 0:
        value = (
            random.uniform(0.0, jitter_seconds)
            if random_value is None
            else random_value
        )
        delay += max(float(value), 0.0)

    return delay


def retry_budget_allows(
    started_at: float,
    delay: float,
    budget_seconds: float,
    clock: Any = time.monotonic,
) -> bool:
    """Return whether waiting and making another attempt fits the budget."""

    if budget_seconds <= 0:
        return True

    return clock() - started_at + delay <= budget_seconds


def sanitize_error_message(error: Exception | str) -> str:
    """Redact common credential values from an error message."""

    message = str(error)

    for secret_name in ("GEMINI_API_KEY", "TAVILY_API_KEY"):
        secret = __import__("os").getenv(secret_name)
        if secret:
            message = message.replace(secret, "[REDACTED]")

    message = re.sub(
        r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+",
        r"\1[REDACTED]",
        message,
    )
    return re.sub(
        r"(?i)\b(api[_-]?key|access[_-]?token|authorization|"
        r"credential|credentials|secret|password)"
        r"(\s*[:=]\s*|\s+)[^\s,;]+",
        r"\1\2[REDACTED]",
        message,
    )
