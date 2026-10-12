"""Application configuration for ResearchPilot."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


def _int(
    name: str,
    default: int,
    minimum: int = 1,
) -> int:
    """Read a bounded integer environment variable."""

    try:
        return max(
            int(os.getenv(name, default)),
            minimum,
        )
    except (
        TypeError,
        ValueError,
    ):
        return default


def _float(
    name: str,
    default: float,
    minimum: float = 0.0,
) -> float:
    """Read a bounded floating-point environment variable."""

    try:
        return max(
            float(os.getenv(name, default)),
            minimum,
        )
    except (
        TypeError,
        ValueError,
    ):
        return default


def _bool(
    name: str,
    default: bool,
) -> bool:
    """Read a boolean environment variable."""

    value = os.getenv(name)

    if value is None:
        return default

    normalized = value.strip().lower()

    if normalized in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return True

    if normalized in {
        "0",
        "false",
        "no",
        "off",
    }:
        return False

    return default


@dataclass(frozen=True)
class Settings:
    """Runtime configuration for ResearchPilot."""

    # ------------------------------------------------------------------
    # Gemini
    # ------------------------------------------------------------------

    gemini_model: str = os.getenv(
        "GEMINI_MODEL",
        "gemini-3.8-flash",
    )

    gemini_max_retries: int = _int(
        "GEMINI_MAX_RETRIES",
        2,
        0,
    )

    gemini_cache_enabled: bool = _bool(
        "GEMINI_CACHE_ENABLED",
        True,
    )

    # ------------------------------------------------------------------
    # Research source limits
    # ------------------------------------------------------------------

    quick_sources: int = _int(
        "QUICK_SOURCES",
        3,
    )

    deep_sources: int = _int(
        "DEEP_SOURCES",
        8,
    )

    compare_sources: int = _int(
        "COMPARE_SOURCES",
        6,
    )

    max_evidence_per_source: int = _int(
        "MAX_EVIDENCE_PER_SOURCE",
        8,
    )

    max_total_evidence: int = _int(
        "MAX_TOTAL_EVIDENCE",
        40,
    )

    # ------------------------------------------------------------------
    # Tavily
    # ------------------------------------------------------------------

    tavily_search_depth: str = os.getenv(
        "TAVILY_SEARCH_DEPTH",
        "advanced",
    )

    tavily_max_retries: int = _int(
        "TAVILY_MAX_RETRIES",
        2,
        0,
    )

    tavily_search_cache_enabled: bool = _bool(
        "TAVILY_SEARCH_CACHE_ENABLED",
        True,
    )

    tavily_extract_cache_enabled: bool = _bool(
        "TAVILY_EXTRACT_CACHE_ENABLED",
        True,
    )

    # ------------------------------------------------------------------
    # Retry policy
    # ------------------------------------------------------------------

    retry_base_seconds: float = _float(
        "RETRY_BASE_SECONDS",
        1.5,
        0.1,
    )

    retry_max_seconds: float = _float(
        "RETRY_MAX_SECONDS",
        30.0,
        0.1,
    )

    retry_jitter_seconds: float = _float(
        "RETRY_JITTER_SECONDS",
        0.0,
        0.0,
    )

    retry_budget_seconds: float = _float(
        "RETRY_BUDGET_SECONDS",
        0.0,
        0.0,
    )

    request_timeout_seconds: int = _int(
        "REQUEST_TIMEOUT_SECONDS",
        45,
    )

    # ------------------------------------------------------------------
    # Cache
    # ------------------------------------------------------------------

    cache_enabled: bool = _bool(
        "CACHE_ENABLED",
        True,
    )

    cache_dir: str = os.getenv(
        "CACHE_DIR",
        ".researchpilot_cache",
    )

    cache_ttl_seconds: int = _int(
        "CACHE_TTL_SECONDS",
        86400,
        0,
    )

    # ------------------------------------------------------------------
    # Durable persistence
    # ------------------------------------------------------------------

    persistence_enabled: bool = _bool(
        "PERSISTENCE_ENABLED",
        True,
    )

    persistence_db: str = os.getenv(
        "PERSISTENCE_DB",
        ".researchpilot_data/researchpilot.db",
    )

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    log_level: str = os.getenv(
        "LOG_LEVEL",
        "INFO",
    ).upper()


SETTINGS = Settings()