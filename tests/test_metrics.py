from time import monotonic

from research.metrics import MetricsRegistry


def test_metrics_registry_tracks_counts_failures_and_latency():
    registry = MetricsRegistry()
    registry.observe("search", monotonic(), True)
    registry.observe("search", monotonic(), False)

    snapshot = registry.snapshot()

    assert snapshot["search"]["count"] == 2
    assert snapshot["search"]["failures"] == 1
    assert snapshot["search"]["total_latency_seconds"] >= 0
