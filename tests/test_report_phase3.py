from research.report import (
    build_evidence_fallback_report,
    build_report,
    validate_citation_ids,
)


def _sources():
    return [
        {
            "id": 1,
            "title": "Primary source",
            "url": "https://example.edu/article",
            "score": 0.9,
            "quality_score": 0.9,
            "quality_category": "academic",
            "research_step": "overview",
            "research_query": "example topic",
            "evidence": [
                {
                    "text": (
                        "Example topic is a defined research subject."
                    ),
                    "confidence": 0.9,
                    "agreement_count": 2,
                    "agreement_score": 0.5,
                    "rank_score": 0.85,
                }
            ],
        },
        {
            "id": 2,
            "title": "Secondary source",
            "url": "https://example.gov/report",
            "score": 0.8,
            "quality_score": 0.95,
            "quality_category": "government",
            "research_step": "mechanism",
            "research_query": (
                "example topic how it works"
            ),
            "evidence": [
                {
                    "text": (
                        "The subject is used for practical analysis."
                    ),
                    "confidence": 0.85,
                    "agreement_count": 1,
                    "agreement_score": 0.25,
                    "rank_score": 0.75,
                }
            ],
        },
        {
            "id": 3,
            "title": "Technical source",
            "url": "https://example.org/technical",
            "score": 0.85,
            "quality_score": 0.88,
            "quality_category": "technical",
            "research_step": "overview",
            "research_query": "example topic definition",
            "evidence": [
                {
                    "text": (
                        "The topic can be described using a technical "
                        "definition."
                    ),
                    "confidence": 0.88,
                    "agreement_count": 1,
                    "agreement_score": 0.25,
                    "rank_score": 0.8,
                }
            ],
        },
    ]


def test_valid_citations_are_accepted():
    assert validate_citation_ids(
        "The subject is defined this way [1].",
        _sources(),
    ) == (1,)


def test_unsupported_citation_ids_are_rejected():
    try:
        validate_citation_ids(
            "This statement cites a missing source [99].",
            _sources(),
        )

    except ValueError as error:
        assert "99" in str(error)

    else:
        raise AssertionError(
            "Expected unsupported citation ID to fail."
        )


def test_build_report_contains_claims_and_source_metadata():
    report = build_report(
        "What is the example topic?",
        "The subject is a defined research subject [1].",
        _sources(),
        mode="quick",
    )

    assert report["schema_version"] == "3.0"

    assert (
        report["citation_integrity"]["valid"]
        is True
    )

    assert (
        report["citation_integrity"][
            "referenced_source_ids"
        ]
        == [1]
    )

    assert (
        report["key_findings"][0]["citation_ids"]
        == [1]
    )

    assert (
        report["key_findings"][0]["supported"]
        is True
    )

    assert (
        report["key_findings"][0]["confidence"]
        > 0
    )

    assert (
        report["sources"][0]["domain"]
        == "example.edu"
    )

    assert (
        report["sources"][0]["source_category"]
        == "academic"
    )


def test_build_report_records_uncited_claims_as_limitation():
    report = build_report(
        "Example topic",
        (
            "The subject is a defined research subject [1]. "
            "This is an uncited statement."
        ),
        _sources(),
    )

    assert (
        report["citation_integrity"][
            "uncited_claim_count"
        ]
        == 1
    )

    assert any(
        "not associated with an explicit source citation"
        in item
        for item in report["limitations"]
    )


def test_fallback_report_is_citation_safe():
    report = build_evidence_fallback_report(
        "Example topic",
        _sources(),
        mode="quick",
    )

    assert (
        report["citation_integrity"]["valid"]
        is True
    )

    assert report["key_findings"]

    assert all(
        finding["supported"]
        for finding in report["key_findings"]
    )

    assert all(
        finding["citation_ids"]
        for finding in report["key_findings"]
    )


def test_build_report_does_not_split_hard_wrapped_sentences():
    answer = (
        "In basic terms, machine learning (ML) is the process of "
        "training a piece of software, called a model, to make useful "
        "predictions or generate content (such as text, images, audio, "
        "or video) from data [3].\n"
        "The central premise of machine learning is that optimizing a "
        "model's performance on a dataset of tasks that adequately "
        "resemble real-world problems—through a process called model "
        "training—enables the model to make accurate predictions on new "
        "data in its ultimate use case [2].\n"
        "Additionally, performance in machine learning is typically "
        "evaluated based on the ability to reproduce known knowledge [1]."
    )

    report = build_report(
        "what is machine learning?",
        answer,
        _sources(),
        mode="quick",
    )

    findings = report["key_findings"]

    assert len(findings) == 3

    assert findings[0]["citation_ids"] == [3]
    assert findings[1]["citation_ids"] == [2]
    assert findings[2]["citation_ids"] == [1]

    assert all(
        finding["supported"]
        for finding in findings
    )

    assert all(
        finding["confidence"] > 0
        for finding in findings
    )

    assert (
        report["citation_integrity"]["uncited_claim_count"]
        == 0
    )


def test_build_report_ignores_fallback_report_structure():
    answer = (
        "RESEARCH SUMMARY\n"
        "\n"
        "Definition\n"
        "\n"
        "The central premise of machine learning is that "
        "optimizing a model's performance on a dataset of "
        "tasks that adequately resemble real-world problems "
        "enables the model to make accurate predictions on "
        "new data in its ultimate use case. [2]\n"
        "\n"
        "How It Works\n"
        "\n"
        "The central premise of machine learning is that "
        "optimizing a model's performance on a dataset of "
        "tasks that adequately resemble real-world problems "
        "enables the model to make accurate predictions on "
        "new data in its ultimate use case. [2]\n"
        "\n"
        "Sources\n"
        "\n"
        "[2] What is Machine Learning? | IBM\n"
        "https://www.ibm.com/think/topics/machine-learning"
    )

    report = build_report(
        "what is machine learning?",
        answer,
        _sources(),
        mode="quick",
    )

    findings = report["key_findings"]

    assert len(findings) == 2

    assert all(
        finding["citation_ids"] == [2]
        for finding in findings
    )

    assert all(
        finding["supported"]
        for finding in findings
    )

    assert (
        report["citation_integrity"][
            "uncited_claim_count"
        ]
        == 0
    )

    assert not any(
        "RESEARCH SUMMARY" in finding["text"]
        for finding in findings
    )

    assert not any(
        "Definition" in finding["text"]
        for finding in findings
    )

    assert not any(
        "How It Works" in finding["text"]
        for finding in findings
    )

    assert not any(
        "Sources" in finding["text"]
        for finding in findings
    )

    assert not any(
        "https://" in finding["text"]
        for finding in findings
    )