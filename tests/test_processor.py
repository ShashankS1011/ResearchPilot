import pytest

from research.processor import (
    normalize_whitespace,
    clean_scraped_markup,
    clean_evidence_artifacts,
    is_noise_line,
    looks_like_navigation,
    looks_like_metadata,
    looks_like_instructional_meta,
    split_extraction_sections,
    split_into_sentences,
    sentence_quality_score,
    classify_evidence,
    evidence_hash,
    similarity_key,
    is_duplicate_evidence,
    domain_category,
    source_quality_score,
    evidence_similarity,
    add_cross_source_agreement,
    prepare_llm_evidence,
    prepare_sources,
)


def test_normalize_whitespace():
    text = "  This   is\n\n a   test.  "

    result = normalize_whitespace(text)

    assert isinstance(result, str)
    assert result.strip()
    assert "This" in result
    assert "test." in result


def test_clean_scraped_markup_removes_basic_markup():
    text = "<p>Hello <strong>world</strong></p>"

    result = clean_scraped_markup(text)

    assert "Hello" in result
    assert "world" in result
    assert "<p>" not in result
    assert "<strong>" not in result


def test_clean_evidence_artifacts_returns_string():
    text = (
        "This is useful evidence.\n"
        "Cookie Policy\n"
        "Privacy Policy\n"
        "This is more useful evidence."
    )

    result = clean_evidence_artifacts(text)

    assert isinstance(result, str)
    assert "This is useful evidence." in result


def test_noise_line_returns_boolean():
    result = is_noise_line("Cookie Policy")

    assert isinstance(result, bool)


def test_noise_line_returns_boolean_for_normal_content():
    result = is_noise_line(
        "Machine learning algorithms learn patterns from data."
    )

    assert isinstance(result, bool)


def test_navigation_detection_returns_boolean():
    result = looks_like_navigation(
        "Home | About | Contact"
    )

    assert isinstance(result, bool)


def test_navigation_detection_for_normal_content():
    result = looks_like_navigation(
        "Machine learning is used in many applications."
    )

    assert isinstance(result, bool)


def test_metadata_detection_returns_boolean():
    result = looks_like_metadata(
        "Published: January 2025"
    )

    assert isinstance(result, bool)


def test_metadata_detection_for_normal_content():
    result = looks_like_metadata(
        "Machine learning models can classify images."
    )

    assert isinstance(result, bool)


def test_instructional_metadata_detection_returns_boolean():
    result = looks_like_instructional_meta(
        "Click here to read more."
    )

    assert isinstance(result, bool)


def test_split_extraction_sections():
    text = (
        "Introduction\n"
        "Machine learning uses data.\n\n"
        "Applications\n"
        "It is used in many fields."
    )

    sections = split_extraction_sections(text)

    assert isinstance(sections, list)
    assert len(sections) >= 1


def test_split_into_sentences():
    text = (
        "Machine learning uses data. "
        "It can identify patterns. "
        "It is widely used."
    )

    sentences = split_into_sentences(text)

    assert isinstance(sentences, list)
    assert len(sentences) >= 2


def test_sentence_quality_score_returns_numeric_value():
    sentence = (
        "Machine learning algorithms identify "
        "patterns from data."
    )

    score = sentence_quality_score(sentence)

    assert isinstance(score, (int, float))
    assert score >= 0


def test_classify_evidence_returns_string():
    sentence = (
        "Supervised learning uses labeled training data."
    )

    classification = classify_evidence(sentence)

    assert isinstance(classification, str)
    assert classification.strip()


def test_evidence_hash_is_stable():
    text = "Machine learning uses data."

    first = evidence_hash(text)
    second = evidence_hash(text)

    assert first == second
    assert isinstance(first, str)
    assert first


def test_similarity_key_is_stable():
    text = "Machine learning uses data."

    first = similarity_key(text)
    second = similarity_key(text)

    assert first == second
    assert isinstance(first, str)


def test_duplicate_evidence_detects_identical_text():
    existing = [
        {
            "text": "Machine learning uses data."
        }
    ]

    assert is_duplicate_evidence(
        "Machine learning uses data.",
        existing,
    )


def test_duplicate_evidence_accepts_different_text():
    existing = [
        {
            "text": "Machine learning uses data."
        }
    ]

    assert not is_duplicate_evidence(
        "Machine learning can classify images.",
        existing,
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://www.geeksforgeeks.org/machine-learning/",
        "https://www.coursera.org/articles/what-is-machine-learning",
        "https://www.youtube.com/watch?v=example",
    ],
)
def test_domain_category_returns_string(url):
    category = domain_category(url)

    assert isinstance(category, str)
    assert category.strip()


def test_source_quality_score_returns_numeric_value():
    result = {
        "url": (
            "https://www.geeksforgeeks.org/"
            "machine-learning/"
        ),
        "title": "Machine Learning",
        "content": (
            "Machine learning is a field of artificial "
            "intelligence that uses algorithms to learn "
            "patterns from data."
        ),
    }

    extracted_content = (
        "Machine learning is a field of artificial "
        "intelligence that uses algorithms to learn "
        "patterns from data."
    )

    score = source_quality_score(
        result,
        extracted_content,
    )

    assert isinstance(score, (int, float))
    assert score >= 0


def test_evidence_similarity_identical_text_is_high():
    text = (
        "Machine learning algorithms identify "
        "patterns in data."
    )

    similarity = evidence_similarity(
        text,
        text,
    )

    assert isinstance(similarity, (int, float))
    assert similarity >= 0.9


def test_evidence_similarity_different_text_is_lower():
    first = (
        "Machine learning algorithms identify "
        "patterns in data."
    )

    second = (
        "The solar system contains planets "
        "and other celestial objects."
    )

    similarity = evidence_similarity(
        first,
        second,
    )

    assert isinstance(similarity, (int, float))
    assert similarity < 0.9


def test_add_cross_source_agreement():
    sources = [
        {
            "id": 1,
            "evidence": [
                {
                    "text": (
                        "Machine learning uses "
                        "training data."
                    )
                }
            ],
        },
        {
            "id": 2,
            "evidence": [
                {
                    "text": (
                        "Machine learning uses "
                        "training data."
                    )
                }
            ],
        },
    ]

    result = add_cross_source_agreement(
        sources
    )

    assert result is not None
    assert isinstance(result, list)


def test_prepare_llm_evidence_returns_collection():
    source = {
        "id": 1,
        "title": "Test Source",
        "url": "https://example.com",
        "evidence": [
            {
                "text": (
                    "Machine learning uses "
                    "training data."
                ),
                "confidence": 0.9,
                "classification": "fact",
            }
        ],
    }

    result = prepare_llm_evidence(
        [source]
    )

    assert isinstance(result, list)


def test_prepare_sources_handles_empty_inputs():
    result = prepare_sources(
        "machine learning",
        [],
        [],
    )

    assert isinstance(result, list)
    assert result == []