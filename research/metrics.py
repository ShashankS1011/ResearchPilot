"""Small in-process metrics collector for provider operations."""

from __future__ import annotations

from collections import defaultdict
from threading import Lock
from time import monotonic
from typing import Any


class MetricsRegistry:
    """Collect request counts and latency without external dependencies."""

    def __init__(self) -> None:
        self._lock = Lock()
        self._counts: dict[str, int] = defaultdict(int)
        self._failures: dict[str, int] = defaultdict(int)
        self._latency: dict[str, float] = defaultdict(float)

    def observe(
        self,
        operation: str,
        started_at: float,
        success: bool,
    ) -> None:
        with self._lock:
            self._counts[operation] += 1
            self._latency[operation] += max(monotonic() - started_at, 0.0)
            if not success:
                self._failures[operation] += 1

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                operation: {
                    "count": self._counts[operation],
                    "failures": self._failures[operation],
                    "total_latency_seconds": self._latency[operation],
                }
                for operation in sorted(self._counts)
            }


METRICS = MetricsRegistry()
