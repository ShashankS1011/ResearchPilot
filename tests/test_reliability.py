from types import SimpleNamespace

from research.providers import CancellationToken, FailoverProvider
from research.reliability import retry_after_seconds, retry_delay


def test_retry_delay_honors_retry_after_and_jitter():
    assert retry_delay(
        0,
        1,
        10,
        jitter_seconds=0.5,
        retry_after=4,
        random_value=0.25,
    ) == 4.25


def test_retry_after_reads_headers():
    error = SimpleNamespace(headers={"Retry-After": "3"})
    assert retry_after_seconds(error) == 3


def test_failover_provider_uses_next_provider():
    class Failing:
        def search(self, query, max_results):
            raise RuntimeError("offline")

    class Working:
        def search(self, query, max_results):
            return [{"url": "https://example.com"}]

    provider = FailoverProvider([Failing(), Working()])
    assert provider.call("search", "test", 1) == [
        {"url": "https://example.com"}
    ]


def test_cancellation_token_is_cooperative():
    token = CancellationToken()
    assert not token.is_cancelled
    token.cancel()
    assert token.is_cancelled
