from research.processor import (
    add_cross_source_agreement,
    clean_evidence_artifacts,
    evidence_similarity,
    evidence_specificity_score,
    extract_evidence,
    prepare_llm_evidence,
)


def test_rephrased_evidence_has_meaningful_similarity():
    first = (
        "Machine learning algorithms learn patterns "
        "from data and use those patterns to make predictions."
    )

    second = (
        "Machine learning systems identify patterns "
        "in data that can then be used for prediction."
    )

    similarity = evidence_similarity(
        first,
        second,
    )

    assert similarity >= 0.60


def test_unrelated_evidence_does_not_match():
    first = (
        "Machine learning models learn patterns "
        "from training data."
    )

    second = (
        "The Pacific Ocean is the largest ocean "
        "on Earth."
    )

    similarity = evidence_similarity(
        first,
        second,
    )

    assert similarity < 0.60


def test_cross_source_agreement_populates_supporting_sources():
    sources = [
        {
            "id": 1,
            "quality_score": 0.95,
            "evidence": [
                {
                    "text": (
                        "Machine learning algorithms "
                        "learn patterns from data "
                        "to make predictions."
                    ),
                    "confidence": 0.80,
                    "cleanliness_score": 0.95,
                }
            ],
        },
        {
            "id": 2,
            "quality_score": 0.90,
            "evidence": [
                {
                    "text": (
                        "Machine learning systems "
                        "identify patterns in data "
                        "and use them for prediction."
                    ),
                    "confidence": 0.80,
                    "cleanliness_score": 0.95,
                }
            ],
        },
    ]

    result = add_cross_source_agreement(
        sources
    )

    first = result[0]["evidence"][0]
    second = result[1]["evidence"][0]

    assert first["agreement_count"] == 2
    assert first["supporting_sources"] == [2]

    assert second["agreement_count"] == 2
    assert second["supporting_sources"] == [1]


def test_same_source_evidence_does_not_count_as_agreement():
    sources = [
        {
            "id": 1,
            "evidence": [
                {
                    "text": (
                        "Machine learning learns "
                        "patterns from data."
                    ),
                    "confidence": 0.80,
                    "cleanliness_score": 0.95,
                },
                {
                    "text": (
                        "Machine learning identifies "
                        "patterns in data."
                    ),
                    "confidence": 0.80,
                    "cleanliness_score": 0.95,
                },
            ],
        },
    ]

    result = add_cross_source_agreement(
        sources
    )

    for evidence in result[0]["evidence"]:
        assert evidence["agreement_count"] == 1
        assert evidence["supporting_sources"] == []


def test_cross_source_contradiction_is_flagged_and_penalized():
    sources = [
        {
            "id": 1,
            "evidence": [
                {
                    "text": (
                        "The method does not require labeled "
                        "training data for prediction."
                    ),
                    "confidence": 0.90,
                    "cleanliness_score": 0.95,
                }
            ],
        },
        {
            "id": 2,
            "evidence": [
                {
                    "text": (
                        "The method requires labeled training "
                        "data for prediction."
                    ),
                    "confidence": 0.90,
                    "cleanliness_score": 0.95,
                }
            ],
        },
    ]

    result = add_cross_source_agreement(sources)

    first = result[0]["evidence"][0]
    second = result[1]["evidence"][0]

    assert first["contradicting_sources"] == [2]
    assert second["contradicting_sources"] == [1]
    assert first["contradiction_count"] == 1
    assert first["risk_flag"] == "cross_source_contradiction"
    assert first["confidence"] < 0.90


def test_marketing_artifact_is_removed():
    text = (
        "****Learn more about the datascience@berkeley "
        "curriculum** and start your journey today."
    )

    cleaned = clean_evidence_artifacts(
        text
    )

    assert cleaned == ""


def test_table_source_artifact_is_removed():
    text = (
        "Source: Bard | Text-to-image | "
        "Generative AI | Example"
    )

    cleaned = clean_evidence_artifacts(
        text
    )

    assert cleaned == ""


def test_navigation_artifact_is_removed():
    text = (
        "Home Menu Search Login Subscribe"
    )

    cleaned = clean_evidence_artifacts(
        text
    )

    assert cleaned == ""


def test_specificity_score_rewards_specific_claims():
    generic = (
        "Machine learning is useful in many systems."
    )

    specific = (
        "Machine learning algorithms learn patterns "
        "from labeled training data to predict known outputs."
    )

    assert (
        evidence_specificity_score(specific)
        > evidence_specificity_score(generic)
    )


def test_extract_evidence_rejects_short_list_fragments():
    search_result = {
        "content": (
            "1. Association: Discover rules in data. "
            "2. Clustering: Group similar observations."
        )
    }

    evidence = extract_evidence(
        search_result
    )

    assert evidence == []


def test_prepare_llm_evidence_preserves_citation_metadata():
    evidence = [
        {
            "text": (
                "Machine learning systems identify "
                "patterns in data for prediction."
            ),
            "confidence": 0.90,
            "cleanliness_score": 0.95,
            "agreement_count": 2,
            "supporting_sources": [2],
            "possible_supporting_sources": [],
            "classification": "general_fact",
            "risk_flag": None,
        }
    ]

    selected = prepare_llm_evidence(
        evidence,
        source_quality=0.95,
    )

    assert len(selected) == 1
    assert selected[0]["supporting_sources"] == [2]
    assert selected[0]["agreement_count"] == 2
    assert selected[0]["rank_score"] > 0