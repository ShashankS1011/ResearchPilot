from research.config import SETTINGS


def test_phase5_retry_configuration_is_valid():
    assert SETTINGS.tavily_max_retries >= 0
    assert SETTINGS.gemini_max_retries >= 0
    assert SETTINGS.retry_base_seconds > 0
    assert SETTINGS.retry_max_seconds > 0
    assert SETTINGS.retry_max_seconds >= SETTINGS.retry_base_seconds


def test_phase5_cache_configuration_is_valid():
    assert isinstance(
        SETTINGS.cache_enabled,
        bool,
    )

    assert isinstance(
        SETTINGS.gemini_cache_enabled,
        bool,
    )

    assert isinstance(
        SETTINGS.tavily_search_cache_enabled,
        bool,
    )

    assert isinstance(
        SETTINGS.tavily_extract_cache_enabled,
        bool,
    )

    assert SETTINGS.cache_dir

    assert SETTINGS.cache_ttl_seconds >= 0


def test_existing_evidence_limits_remain_valid():
    assert SETTINGS.max_evidence_per_source >= 1
    assert SETTINGS.max_total_evidence >= 1


def test_deep_mode_has_at_least_as_many_sources_as_quick():
    assert SETTINGS.deep_sources >= SETTINGS.quick_sources