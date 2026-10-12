
"""Tests for ResearchPilot LLM and evidence synthesis."""

from dataclasses import replace

import pytest

from research import llm


def _replace_settings(monkeypatch, **changes):
    """Replace the frozen Settings instance for an isolated test."""

    monkeypatch.setattr(
        llm,
        "SETTINGS",
        replace(
            llm.SETTINGS,
            **changes,
        ),
    )


def _texts(result, key):
    """Return evidence text values for a result category."""

    return [
        item["text"]
        for item in result[key]
    ]


# ============================================================
# TYPE EXTRACTION
# ============================================================


def test_supervised_learning_is_classified_as_supervised():
    evidence = [{
        "text": (
            "Supervised learning uses labeled "
            "training data to predict known outputs."
        )
    }]

    result = llm.extract_type_evidence(
        evidence
    )

    assert len(
        result["supervised"]
    ) == 1

    assert not result["unsupervised"]
    assert not result["reinforcement"]
    assert not result["generative"]


def test_unsupervised_learning_is_classified_as_unsupervised():
    evidence = [{
        "text": (
            "Unsupervised learning identifies "
            "patterns in unlabeled data."
        )
    }]

    result = llm.extract_type_evidence(
        evidence
    )

    assert len(
        result["unsupervised"]
    ) == 1

    assert not result["supervised"]
    assert not result["reinforcement"]
    assert not result["generative"]


def test_reinforcement_learning_is_classified_as_reinforcement():
    evidence = [{
        "text": (
            "Reinforcement learning uses rewards "
            "and penalties through trial and error."
        )
    }]

    result = llm.extract_type_evidence(
        evidence
    )

    assert len(
        result["reinforcement"]
    ) == 1

    assert not result["supervised"]
    assert not result["unsupervised"]
    assert not result["generative"]


def test_generative_ai_is_classified_as_generative():
    evidence = [{
        "text": (
            "Generative AI creates new content "
            "such as text, images, and audio."
        )
    }]

    result = llm.extract_type_evidence(
        evidence
    )

    assert len(
        result["generative"]
    ) == 1

    assert not result["supervised"]
    assert not result["unsupervised"]
    assert not result["reinforcement"]


def test_association_and_clustering_are_not_top_level_types():
    evidence = [
        {
            "text": (
                "2. Association: Discover rules that "
                "describe large portions of the data."
            )
        },
        {
            "text": (
                "3. Clustering: Group similar "
                "observations together."
            )
        },
    ]

    result = llm.extract_type_evidence(
        evidence
    )

    assert result == {
        "supervised": [],
        "unsupervised": [],
        "reinforcement": [],
        "generative": [],
    }


def test_type_sections_do_not_cross_contaminate():
    evidence = [{
        "text": (
            "1. Supervised learning uses labeled data. "
            "2. Association: Discover rules that describe "
            "large portions of the data. "
            "3. Unsupervised learning identifies patterns "
            "in unlabeled data. "
            "4. Clustering: Group similar observations "
            "together. "
            "5. Reinforcement learning uses rewards "
            "through trial and error."
        )
    }]

    result = llm.extract_type_evidence(
        evidence
    )

    supervised = _texts(
        result,
        "supervised",
    )

    unsupervised = _texts(
        result,
        "unsupervised",
    )

    reinforcement = _texts(
        result,
        "reinforcement",
    )

    assert supervised == [
        "Supervised learning uses labeled data."
    ]

    assert unsupervised == [
        "Unsupervised learning identifies patterns in unlabeled data."
    ]

    assert reinforcement == [
        "Reinforcement learning uses rewards through trial and error."
    ]

    assert all(
        "Association" not in text
        for text in (
            supervised
            + unsupervised
            + reinforcement
        )
    )

    assert all(
        "Clustering" not in text
        for text in (
            supervised
            + unsupervised
            + reinforcement
        )
    )


def test_multiple_type_descriptions_are_preserved():
    evidence = [{
        "text": (
            "1. Supervised learning uses labeled data. "
            "2. Unsupervised learning identifies patterns "
            "in unlabeled data. "
            "3. Reinforcement learning uses rewards "
            "through trial and error. "
            "4. Generative AI creates new content."
        )
    }]

    result = llm.extract_type_evidence(
        evidence
    )

    assert len(
        result["supervised"]
    ) == 1

    assert len(
        result["unsupervised"]
    ) == 1

    assert len(
        result["reinforcement"]
    ) == 1

    assert len(
        result["generative"]
    ) == 1


def test_type_extraction_ignores_non_dictionary_items():
    evidence = [
        None,
        "not a dictionary",
        123,
        {
            "text": (
                "Supervised learning uses labeled "
                "training data."
            )
        },
    ]

    result = llm.extract_type_evidence(
        evidence
    )

    assert len(
        result["supervised"]
    ) == 1


def test_type_extraction_preserves_evidence_metadata():
    evidence = [{
        "text": (
            "Supervised learning uses labeled "
            "training data."
        ),
        "source_id": 3,
        "confidence": 0.91,
        "quality_score": 0.88,
    }]

    result = llm.extract_type_evidence(
        evidence
    )

    item = result["supervised"][0]

    assert item["source_id"] == 3
    assert item["confidence"] == 0.91
    assert item["quality_score"] == 0.88


# ============================================================
# TYPE SELECTION
# ============================================================


def test_choose_best_types_returns_descriptions():
    evidence = [
        {
            "text": (
                "Machine learning systems can be categorized "
                "as supervised and unsupervised learning."
            ),
            "source_id": 1,
            "agreement_count": 2,
            "confidence": 0.85,
            "quality_score": 0.90,
        },
        {
            "text": (
                "Supervised learning uses labeled "
                "training data."
            ),
            "source_id": 1,
            "agreement_count": 2,
            "confidence": 0.85,
            "quality_score": 0.90,
        },
        {
            "text": (
                "Unsupervised learning identifies "
                "patterns in unlabeled data."
            ),
            "source_id": 2,
            "agreement_count": 2,
            "confidence": 0.85,
            "quality_score": 0.90,
        },
    ]

    result = llm.choose_best_types(
        evidence
    )

    assert result is not None
    assert result["category"] is not None
    assert result["descriptions"]["supervised"]
    assert result["descriptions"]["unsupervised"]


def test_choose_best_types_returns_none_without_type_evidence():
    evidence = [{
        "text": (
            "Machine learning models are trained "
            "using data."
        ),
        "source_id": 1,
        "confidence": 0.90,
        "quality_score": 0.90,
    }]

    result = llm.choose_best_types(
        evidence
    )

    assert result is None


# ============================================================
# GEMINI RELIABILITY HELPERS
# ============================================================


def test_gemini_retryable_error_detects_transient_errors():
    assert llm._is_retryable_gemini_error(
        RuntimeError("503 Service Unavailable")
    )

    assert llm._is_retryable_gemini_error(
        RuntimeError("429 Resource exhausted")
    )

    assert llm._is_retryable_gemini_error(
        RuntimeError("connection timed out")
    )


def test_gemini_retryable_error_rejects_auth_errors():
    assert not llm._is_retryable_gemini_error(
        RuntimeError("401 invalid API key")
    )


def test_gemini_retry_delay_is_exponential():
    original_base = (
        llm.SETTINGS.retry_base_seconds
    )

    assert llm._retry_delay(0) == original_base
    assert llm._retry_delay(1) == original_base * 2
    assert llm._retry_delay(2) == original_base * 4


def test_gemini_retry_delay_is_bounded(monkeypatch):
    _replace_settings(
        monkeypatch,
        retry_base_seconds=20.0,
    )

    assert llm._retry_delay(0) == 20.0
    assert llm._retry_delay(1) == 30.0
    assert llm._retry_delay(2) == 30.0


def test_gemini_request_kwargs_passes_configured_timeout(
    monkeypatch,
):
    class Interactions:
        def create(self, model, input, timeout=None):
            return None

    class Client:
        interactions = Interactions()

    _replace_settings(
        monkeypatch,
        request_timeout_seconds=19,
    )
    monkeypatch.setattr(
        llm,
        "_GEMINI_CLIENT",
        Client(),
    )

    assert llm._gemini_request_kwargs() == {
        "timeout": 19,
    }


# ============================================================
# GEMINI TEST DOUBLES
# ============================================================


class FakeGeminiResponse:
    """Minimal Gemini response used by tests."""

    def __init__(self, output_text):
        self.output_text = output_text


class FakeGeminiInteractions:
    """Fake Gemini interactions endpoint."""

    def __init__(self, responses):
        self.responses = list(
            responses
        )
        self.calls = []

    def create(self, model, input):
        self.calls.append(
            {
                "model": model,
                "input": input,
            }
        )

        if not self.responses:
            raise RuntimeError(
                "Fake Gemini response queue is empty."
            )

        response = self.responses.pop(0)

        if isinstance(
            response,
            Exception,
        ):
            raise response

        return response


class FakeGeminiClient:
    """Fake Gemini client."""

    instances = []

    def __init__(self, responses):
        self.interactions = (
            FakeGeminiInteractions(
                responses
            )
        )

        FakeGeminiClient.instances.append(
            self
        )


def _patch_gemini(
    monkeypatch,
    responses,
):
    """Patch Gemini client construction with a shared response queue."""

    FakeGeminiClient.instances = []

    shared_responses = list(
        responses
    )

    def factory(**kwargs):
        client = FakeGeminiClient(
            []
        )

        client.interactions.responses = (
            shared_responses
        )

        return client

    monkeypatch.setattr(
        llm.genai,
        "Client",
        factory,
    )


def _sample_ledger():
    return {
        "query": "What is machine learning?",
        "intent": "definition",
        "definition": {
            "source_id": 1,
            "text": (
                "Machine learning is a branch of artificial "
                "intelligence that learns from data."
            ),
        },
        "mechanism": None,
        "types": None,
        "applications": [],
        "supporting": [],
    }


# ============================================================
# GEMINI SYNTHESIS
# ============================================================


def test_generate_with_gemini_returns_cleaned_response(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv(
        "GEMINI_API_KEY",
        "test-key",
    )

    _replace_settings(
        monkeypatch,
        cache_dir=str(tmp_path),
        cache_enabled=True,
        gemini_cache_enabled=True,
        gemini_max_retries=0,
    )

    _patch_gemini(
        monkeypatch,
        [
            FakeGeminiResponse(
                "## Answer\n\nMachine learning is useful. [1]"
            )
        ],
    )

    result = llm.generate_with_gemini(
        "What is machine learning?",
        _sample_ledger(),
    )

    assert "Answer" not in result
    assert "Machine learning is useful. [1]" in result

    assert len(
        FakeGeminiClient.instances
    ) == 1

    assert len(
        FakeGeminiClient.instances[0]
        .interactions
        .calls
    ) == 1


def test_generate_with_gemini_uses_cache_on_second_call(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv(
        "GEMINI_API_KEY",
        "test-key",
    )

    _replace_settings(
        monkeypatch,
        cache_dir=str(tmp_path),
        cache_enabled=True,
        gemini_cache_enabled=True,
        gemini_max_retries=0,
    )

    _patch_gemini(
        monkeypatch,
        [
            FakeGeminiResponse(
                "Machine learning is a field of AI. [1]"
            )
        ],
    )

    ledger = _sample_ledger()

    first = llm.generate_with_gemini(
        "What is machine learning?",
        ledger,
    )

    second = llm.generate_with_gemini(
        "What is machine learning?",
        ledger,
    )

    assert first == second

    assert len(
        FakeGeminiClient.instances
    ) == 1

    assert len(
        FakeGeminiClient.instances[0]
        .interactions
        .calls
    ) == 1


def test_generate_with_gemini_bypasses_cache_when_disabled(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv(
        "GEMINI_API_KEY",
        "test-key",
    )

    _replace_settings(
        monkeypatch,
        cache_dir=str(tmp_path),
        cache_enabled=True,
        gemini_cache_enabled=False,
        gemini_max_retries=0,
    )

    _patch_gemini(
        monkeypatch,
        [
            FakeGeminiResponse(
                "First response. [1]"
            ),
            FakeGeminiResponse(
                "Second response. [1]"
            ),
        ],
    )

    ledger = _sample_ledger()

    first = llm.generate_with_gemini(
        "What is machine learning?",
        ledger,
    )

    second = llm.generate_with_gemini(
        "What is machine learning?",
        ledger,
    )

    assert first != second

    assert len(
        FakeGeminiClient.instances
    ) == 2


def test_generate_with_gemini_retries_transient_failure(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv(
        "GEMINI_API_KEY",
        "test-key",
    )

    _replace_settings(
        monkeypatch,
        cache_dir=str(tmp_path),
        cache_enabled=True,
        gemini_cache_enabled=False,
        gemini_max_retries=2,
        retry_base_seconds=0.01,
    )

    _patch_gemini(
        monkeypatch,
        [
            RuntimeError(
                "503 Service Unavailable"
            ),
            FakeGeminiResponse(
                "Recovered response. [1]"
            ),
        ],
    )

    sleeps = []

    monkeypatch.setattr(
        llm.time,
        "sleep",
        lambda seconds: sleeps.append(
            seconds
        ),
    )

    result = llm.generate_with_gemini(
        "What is machine learning?",
        _sample_ledger(),
    )

    assert result == (
        "Recovered response. [1]"
    )

    client = (
        FakeGeminiClient.instances[0]
    )

    assert len(
        client.interactions.calls
    ) == 2

    assert sleeps == [
        0.01,
    ]


def test_generate_with_gemini_does_not_retry_auth_failure(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv(
        "GEMINI_API_KEY",
        "test-key",
    )

    _replace_settings(
        monkeypatch,
        cache_dir=str(tmp_path),
        cache_enabled=True,
        gemini_cache_enabled=False,
        gemini_max_retries=3,
    )

    _patch_gemini(
        monkeypatch,
        [
            RuntimeError(
                "401 invalid API key"
            ),
        ],
    )

    sleeps = []

    monkeypatch.setattr(
        llm.time,
        "sleep",
        lambda seconds: sleeps.append(
            seconds
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="Gemini synthesis failed",
    ):
        llm.generate_with_gemini(
            "What is machine learning?",
            _sample_ledger(),
        )

    client = (
        FakeGeminiClient.instances[0]
    )

    assert len(
        client.interactions.calls
    ) == 1

    assert sleeps == []


def test_generate_with_gemini_exhausts_retries(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv(
        "GEMINI_API_KEY",
        "test-key",
    )

    _replace_settings(
        monkeypatch,
        cache_dir=str(tmp_path),
        cache_enabled=True,
        gemini_cache_enabled=False,
        gemini_max_retries=2,
        retry_base_seconds=0.01,
    )

    _patch_gemini(
        monkeypatch,
        [
            RuntimeError(
                "503 Service Unavailable"
            ),
            RuntimeError(
                "503 Service Unavailable"
            ),
            RuntimeError(
                "503 Service Unavailable"
            ),
        ],
    )

    sleeps = []

    monkeypatch.setattr(
        llm.time,
        "sleep",
        lambda seconds: sleeps.append(
            seconds
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="Gemini synthesis failed",
    ):
        llm.generate_with_gemini(
            "What is machine learning?",
            _sample_ledger(),
        )

    client = (
        FakeGeminiClient.instances[0]
    )

    assert len(
        client.interactions.calls
    ) == 3

    assert sleeps == [
        0.01,
        0.02,
    ]


def test_generate_with_gemini_does_not_cache_failed_response(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv(
        "GEMINI_API_KEY",
        "test-key",
    )

    _replace_settings(
        monkeypatch,
        cache_dir=str(tmp_path),
        cache_enabled=True,
        gemini_cache_enabled=True,
        gemini_max_retries=0,
    )

    _patch_gemini(
        monkeypatch,
        [
            RuntimeError(
                "503 Service Unavailable"
            ),
        ],
    )

    with pytest.raises(
        RuntimeError,
        match="Gemini synthesis failed",
    ):
        llm.generate_with_gemini(
            "What is machine learning?",
            _sample_ledger(),
        )

    cache = llm._get_gemini_cache()

    context = llm.build_gemini_context(
        _sample_ledger()
    )

    key = llm._build_gemini_cache_key(
        "What is machine learning?",
        "definition",
        context,
    )

    assert cache.get(key) is None


# ============================================================
# FALLBACK
# ============================================================


def test_fallback_synthesis_uses_definition():
    ledger = _sample_ledger()

    ledger["_sources"] = [
        {
            "id": 1,
            "title": "Example Source",
            "url": "https://example.com",
        }
    ]

    result = llm.fallback_synthesis(
        "What is machine learning?",
        ledger,
    )

    assert "RESEARCH SUMMARY" in result
    assert "Definition" in result
    assert "Machine learning is a branch" in result
    assert "[1]" in result
    assert "https://example.com" in result


def test_generate_answer_falls_back_when_gemini_fails(
    monkeypatch,
):
    sources = [
        {
            "id": 1,
            "title": "Example Source",
            "url": "https://example.com",
            "quality_score": 0.9,
            "evidence": [
                {
                    "text": (
                        "Machine learning is a branch of "
                        "artificial intelligence that learns "
                        "from data."
                    ),
                    "agreement_count": 2,
                    "confidence": 0.9,
                }
            ],
        }
    ]

    def fail_gemini(
        query,
        ledger,
    ):
        raise RuntimeError(
            "Gemini unavailable"
        )

    monkeypatch.setattr(
        llm,
        "generate_with_gemini",
        fail_gemini,
    )

    result = llm.generate_answer(
        "What is machine learning?",
        sources,
    )

    assert "RESEARCH SUMMARY" in result
    assert "Definition" in result
    assert "Machine learning is a branch" in result
