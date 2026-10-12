
import logging
import os
import re
import textwrap
import time
import inspect

from dotenv import load_dotenv
from google import genai

from research.cache import CacheStore
from research.config import SETTINGS
from research.reliability import (
    retry_after_seconds,
    retry_budget_allows,
    retry_delay,
    sanitize_error_message,
)

load_dotenv()


logger = logging.getLogger("researchpilot.llm")

_GEMINI_CLIENT = None
_GEMINI_CLIENT_FACTORY = None

OUTPUT_WIDTH = 100


# ============================================================
# OUTPUT FORMATTING
# ============================================================

def wrap_text(text, width=OUTPUT_WIDTH):
    """
    Wrap normal prose at a readable terminal width.

    Existing paragraphs are preserved instead of putting every
    sentence or word on a separate line.
    """

    if not text:
        return ""

    lines = []

    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()

        if not paragraph:
            lines.append("")
            continue

        if (
            paragraph.startswith("#")
            or paragraph.startswith("•")
            or paragraph.startswith("- ")
            or paragraph.startswith("* ")
        ):
            lines.extend(
                textwrap.wrap(
                    paragraph,
                    width=width,
                    subsequent_indent="  ",
                )
            )
            continue

        lines.extend(
            textwrap.wrap(
                paragraph,
                width=width,
            )
        )

    return "\n".join(lines)


def clean_text(text):
    """
    Normalize text while preserving readable sentence structure.
    """

    if not text:
        return ""

    text = str(text)

    text = text.replace(
        "\\*",
        "*",
    )
    text = text.replace(
        "\\_",
        "_",
    )
    text = text.replace(
        "\\#",
        "#",
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


def clean_evidence_fragment(text):
    """
    Clean a scraped evidence fragment.
    """

    if not text:
        return ""

    text = str(text)

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    text = re.sub(
        r"^\s*[-•*]\s*",
        "",
        text,
    )

    return text.strip()


def normalize_sentence(text):
    """
    Normalize a sentence without changing its substantive meaning.
    """

    text = clean_evidence_fragment(text)

    if not text:
        return ""

    text = re.sub(
        r"\s+([,.!?;:])",
        r"\1",
        text,
    )

    text = re.sub(
        r"([,.!?;:])([A-Za-z])",
        r"\1 \2",
        text,
    )

    return text.strip()


def clean_gemini_markdown(text):
    """
    Clean Gemini output while retaining useful Markdown structure.
    """

    if not text:
        return ""

    text = str(text)

    text = text.replace(
        "\\*",
        "*",
    )
    text = text.replace(
        "\\_",
        "_",
    )
    text = text.replace(
        "\\#",
        "#",
    )

    text = re.sub(
        r"^\s*```(?:markdown|text)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s*```\s*$",
        "",
        text,
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    lines = []

    for line in text.splitlines():
        stripped = line.strip()

        if re.fullmatch(
            r"#{1,6}\s*answer\s*#*",
            stripped,
            flags=re.IGNORECASE,
        ):
            continue

        if not stripped:
            if lines and lines[-1] != "":
                lines.append("")
            continue

        lines.append(stripped)

    return "\n".join(lines).strip()


def sentence_words(text):
    """
    Return normalized word tokens from a sentence.
    """

    if not text:
        return []

    return re.findall(
        r"\b[a-zA-Z][a-zA-Z0-9'-]*\b",
        text.lower(),
    )


def query_keywords(query):
    """
    Extract useful keywords from a research query.
    """

    stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "their",
        "this",
        "to",
        "what",
        "when",
        "where",
        "which",
        "why",
        "with",
    }

    return [
        word
        for word in sentence_words(query)
        if word not in stopwords
    ]


# ============================================================
# QUERY INTENT
# ============================================================

def detect_query_intent(query):
    """
    Detect the dominant research intent of a query.
    """

    lower = (query or "").lower()

    if any(
        phrase in lower
        for phrase in (
            "what is",
            "what are",
            "define",
            "definition of",
            "meaning of",
        )
    ):
        return "definition"

    if any(
        phrase in lower
        for phrase in (
            "how does",
            "how do",
            "how is",
            "how are",
            "how it works",
            "how does it work",
            "mechanism",
        )
    ):
        return "mechanism"

    if any(
        phrase in lower
        for phrase in (
            "types of",
            "type of",
            "categories of",
            "category of",
            "kinds of",
            "classification of",
        )
    ):
        return "types"

    if any(
        phrase in lower
        for phrase in (
            "applications of",
            "application of",
            "uses of",
            "use cases",
            "used for",
            "examples of",
        )
    ):
        return "applications"

    if any(
        phrase in lower
        for phrase in (
            "advantages",
            "disadvantages",
            "pros and cons",
            "benefits",
            "limitations",
        )
    ):
        return "comparison"

    return "general"


# ============================================================
# CLAIM QUALITY
# ============================================================

def is_noise_claim(text):
    """
    Detect obvious low-value or navigation-like text.
    """

    if not text:
        return True

    lower = text.lower().strip()

    noise_patterns = (
        "click here",
        "read more",
        "sign up",
        "subscribe",
        "cookie policy",
        "privacy policy",
        "terms of service",
        "all rights reserved",
        "table of contents",
        "share this article",
        "follow us",
        "advertisement",
        "advertising",
    )

    return any(
        pattern in lower
        for pattern in noise_patterns
    )


def is_scraped_fragment(text):
    """
    Detect fragments that are unlikely to be complete evidence claims.
    """

    if not text:
        return True

    stripped = text.strip()

    if len(stripped) < 20:
        return True

    if stripped.startswith(
        (
            "http://",
            "https://",
        )
    ):
        return True

    return False


def classify_claim(text):
    """
    Classify a claim into a broad evidence category.
    """

    if not text:
        return "unknown"

    lower = text.lower()

    if is_noise_claim(text):
        return "noise"

    if (
        "defined as" in lower
        or "refers to" in lower
        or "is a" in lower
        or "is an" in lower
        or lower.startswith("machine learning is")
    ):
        return "definition"

    if (
        "works by" in lower
        or "works through" in lower
        or "process involves" in lower
        or "consists of" in lower
        or "uses" in lower
    ):
        return "mechanism"

    if (
        "types include" in lower
        or "types of" in lower
        or "categories include" in lower
        or "categorized as" in lower
        or "classified as" in lower
        or "classified into" in lower
    ):
        return "types"

    if (
        "used for" in lower
        or "applications include" in lower
        or "applications of" in lower
        or "used in" in lower
    ):
        return "applications"

    return "supporting"


def detect_claim_risk(text):
    """
    Detect language indicating uncertainty or potentially risky claims.
    """

    if not text:
        return False

    lower = text.lower()

    risk_markers = (
        "may",
        "might",
        "could",
        "possibly",
        "approximately",
        "estimated",
        "reportedly",
        "according to some",
        "it is believed",
        "allegedly",
    )

    return any(
        marker in lower
        for marker in risk_markers
    )


def base_confidence_for_agreement(agreement_count):
    """
    Estimate confidence from source agreement.
    """

    if agreement_count <= 1:
        return 0.55

    if agreement_count == 2:
        return 0.72

    if agreement_count == 3:
        return 0.84

    return 0.92


def calculate_confidence(
    quality_score=0.0,
    agreement_count=1,
    risk_flag=False,
):
    """
    Calculate a bounded confidence score.
    """

    quality_score = max(
        0.0,
        min(
            1.0,
            float(quality_score),
        ),
    )

    agreement_score = base_confidence_for_agreement(
        agreement_count
    )

    confidence = (
        0.55 * quality_score
        + 0.45 * agreement_score
    )

    if risk_flag:
        confidence -= 0.15

    return max(
        0.0,
        min(
            1.0,
            confidence,
        ),
    )


def classify_source(url):
    """
    Classify a source based on its hostname.
    """

    if not url:
        return "unknown"

    lower = url.lower()

    if ".gov" in lower:
        return "government"

    if ".edu" in lower:
        return "academic"

    if (
        "nature.com" in lower
        or "science.org" in lower
        or "sciencedirect.com" in lower
        or "springer.com" in lower
        or "arxiv.org" in lower
    ):
        return "academic"

    if (
        "wikipedia.org" in lower
        or "britannica.com" in lower
    ):
        return "reference"

    if (
        "medium.com" in lower
        or "blogspot." in lower
        or "wordpress." in lower
    ):
        return "blog"

    return "web"


# ============================================================
# EVIDENCE SCORING
# ============================================================

def relevance_score(
    text,
    query,
):
    """
    Score lexical relevance between evidence and query.
    """

    if not text or not query:
        return 0.0

    text_words = set(
        sentence_words(text)
    )

    keywords = set(
        query_keywords(query)
    )

    if not keywords:
        return 0.0

    overlap = text_words.intersection(
        keywords
    )

    return len(overlap) / len(keywords)


def similarity_score(
    first,
    second,
):
    """
    Calculate simple token-based similarity.
    """

    first_words = set(
        sentence_words(first)
    )

    second_words = set(
        sentence_words(second)
    )

    if not first_words or not second_words:
        return 0.0

    intersection = first_words.intersection(
        second_words
    )

    union = first_words.union(
        second_words
    )

    if not union:
        return 0.0

    return len(intersection) / len(union)


def is_redundant(
    text,
    existing,
    threshold=0.82,
):
    """
    Determine whether evidence is substantially duplicated.
    """

    if not text:
        return True

    for item in existing:
        candidate = (
            item.get("text", "")
            if isinstance(item, dict)
            else str(item)
        )

        if similarity_score(
            text,
            candidate,
        ) >= threshold:
            return True

    return False


def evidence_score(
    text,
    query="",
    quality_score=0.0,
    agreement_count=1,
):
    """
    Combine evidence quality, relevance, and agreement.
    """

    relevance = relevance_score(
        text,
        query,
    )

    agreement = base_confidence_for_agreement(
        agreement_count
    )

    return (
        0.40 * float(quality_score)
        + 0.35 * relevance
        + 0.25 * agreement
    )


# ============================================================
# DEFINITION EVIDENCE
# ============================================================

def is_definition_evidence(text):
    """
    Determine whether text is a useful definition.
    """

    if not text:
        return False

    lower = text.lower()

    patterns = (
        r"\bis defined as\b",
        r"\brefers to\b",
        r"\bis a\b",
        r"\bis an\b",
        r"\bmeans\b",
        r"\bcan be defined as\b",
        r"\bdescribes\b",
    )

    return any(
        re.search(
            pattern,
            lower,
        )
        for pattern in patterns
    )


def choose_best_definition(evidence):
    """
    Choose the strongest definition evidence.
    """

    candidates = [
        item
        for item in evidence
        if is_definition_evidence(
            item.get("text", "")
        )
        and not item.get("risk_flag")
    ]

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda item: (
            item.get(
                "quality_score",
                0.0,
            ),
            item.get(
                "confidence",
                0.0,
            ),
            item.get(
                "agreement_count",
                1,
            ),
        ),
    )


# ============================================================
# MECHANISM EVIDENCE
# ============================================================

def mechanism_strength(text):
    """
    Score how strongly text explains a mechanism.
    """

    if not text:
        return 0

    lower = text.lower()

    markers = (
        "works by",
        "works through",
        "process",
        "steps",
        "first",
        "then",
        "finally",
        "input",
        "output",
        "training",
        "algorithm",
        "learns from",
    )

    return sum(
        1
        for marker in markers
        if marker in lower
    )


def is_mechanism_evidence(text):
    return mechanism_strength(text) > 0


def choose_best_mechanism(evidence):
    """
    Choose the strongest mechanism evidence.
    """

    candidates = [
        item
        for item in evidence
        if is_mechanism_evidence(
            item.get("text", "")
        )
        and not item.get("risk_flag")
    ]

    if not candidates:
        return None

    return max(
        candidates,
        key=lambda item: (
            mechanism_strength(
                item.get("text", "")
            ),
            item.get(
                "quality_score",
                0.0,
            ),
            item.get(
                "confidence",
                0.0,
            ),
            item.get(
                "agreement_count",
                1,
            ),
        ),
    )


# ============================================================
# MACHINE-LEARNING TYPES
# ============================================================

def is_types_evidence(text):
    """
    Detect evidence describing categories or types of machine learning.
    """

    if not text:
        return False

    lower = text.lower()

    patterns = [
        r"\bml systems can be categorized as\b",
        r"\bmachine learning systems can be categorized as\b",
        r"\bmachine learning.*categorized as\b",
        r"\bmachine learning.*classified as\b",
        r"\btypes?\s+of\s+machine learning\b",
        r"\btypes?\s+of\s+ml\b",
        r"\bcategories?\s+of\s+machine learning\b",
        r"\bcategories?\s+of\s+ml\b",
        r"\bcommon types of machine learning\b",
        r"\btypes?\s+include\b",
        r"\bcategories?\s+include\b",
        r"\bclassified into\b",
        r"\bcategorized into\b",
        r"\bclassification includes\b",
        r"\bcan be divided into\b",
        r"\bml\s+systems.*\bsupervised\b.*\bunsupervised\b",
    ]

    return any(
        re.search(
            pattern,
            lower,
        )
        for pattern in patterns
    )


def is_supervised_evidence(text):
    """Detect supervised-learning evidence without substring leakage."""

    if not text:
        return False

    lower = text.lower()

    return (
        re.search(
            r"\bsupervised learning\b.*?\buses?\b.*?\blabeled(?:\s+\w+){0,2}\s+data\b",
            lower,
        )
        is not None
        or (
            re.search(
                r"\bsupervised learning\b",
                lower,
            )
            is not None
            and (
                "labeled data" in lower
                or "labelled data" in lower
                or "labeled dataset" in lower
                or "labelled dataset" in lower
                or "labeled training data" in lower
                or "labelled training data" in lower
            )
        )
    )

def is_unsupervised_evidence(text):
    """
    Detect unsupervised-learning evidence.
    """

    if not text:
        return False

    lower = text.lower()

    if "unsupervised learning" not in lower:
        return False

    return any(
        phrase in lower
        for phrase in (
            "unlabeled data",
            "unlabelled data",
            "without labels",
            "without labeled",
            "without labelled",
            "discover patterns",
            "find patterns",
            "clusters",
            "hidden structure",
        )
    )


def is_reinforcement_evidence(text):
    """
    Detect reinforcement-learning evidence.
    """

    if not text:
        return False

    lower = text.lower()

    if "reinforcement learning" not in lower:
        return False

    return any(
        phrase in lower
        for phrase in (
            "reward",
            "rewards",
            "penalty",
            "penalties",
            "environment",
            "agent",
            "actions",
        )
    )


def is_generative_evidence(text):
    """
    Detect generative-learning evidence.
    """

    if not text:
        return False

    lower = text.lower()

    if "generative" not in lower:
        return False

    return any(
        phrase in lower
        for phrase in (
            "generate",
            "generates",
            "create",
            "creates",
            "new data",
            "new content",
            "content generation",
        )
    )


def _split_type_evidence(text):
    """Split numbered multi-type evidence into individual statements."""

    if not text:
        return []

    cleaned = clean_evidence_fragment(text)

    parts = re.split(
        r"(?:^|(?<=[.!?]))\s*(?=\d+\.\s*)",
        cleaned,
    )

    parts = [
        re.sub(r"^\d+\.\s*", "", part).strip()
        for part in parts
        if part.strip()
    ]

    return parts or [cleaned]


def extract_type_evidence(evidence):
    """Group evidence by machine-learning type."""

    types = {
        "supervised": [],
        "unsupervised": [],
        "reinforcement": [],
        "generative": [],
    }

    for item in evidence:
        if not isinstance(item, dict):
            continue

        text = item.get("text", "")

        for fragment in _split_type_evidence(text):
            candidate = dict(item)
            candidate["text"] = fragment

            if is_supervised_evidence(fragment):
                types["supervised"].append(candidate)

            if is_unsupervised_evidence(fragment):
                types["unsupervised"].append(candidate)

            if is_reinforcement_evidence(fragment):
                types["reinforcement"].append(candidate)

            if is_generative_evidence(fragment):
                types["generative"].append(candidate)

    return types

def choose_best_types(evidence):
    """Select category evidence and the strongest description for each type."""

    valid_evidence = [
        item
        for item in evidence
        if isinstance(item, dict)
    ]

    category_candidates = [
        item
        for item in valid_evidence
        if is_types_evidence(item.get("text", ""))
    ]

    category_source = None

    if category_candidates:
        category_source = max(
            category_candidates,
            key=lambda item: (
                item.get("quality_score", 0.0),
                item.get("confidence", 0.0),
                item.get("agreement_count", 1),
            ),
        )

    grouped = extract_type_evidence(valid_evidence)
    type_descriptions = {}

    for type_label, candidates in grouped.items():
        if not candidates:
            continue

        best = max(
            candidates,
            key=lambda item: (
                item.get("quality_score", 0.0),
                item.get("confidence", 0.0),
                item.get("agreement_count", 1),
            ),
        )
        type_descriptions[type_label] = best

    if category_source is None and not type_descriptions:
        return None

    return {
        "category": category_source,
        "descriptions": type_descriptions,
    }

def choose_best_applications(evidence):
    """
    Choose evidence describing applications.
    """

    candidates = []

    for item in evidence:
        text = item.get(
            "text",
            "",
        )

        lower = text.lower()

        if any(
            phrase in lower
            for phrase in (
                "applications include",
                "used for",
                "used in",
                "applications of",
                "use cases",
            )
        ):
            candidates.append(item)

    candidates.sort(
        key=lambda item: (
            item.get(
                "quality_score",
                0.0,
            ),
            item.get(
                "confidence",
                0.0,
            ),
            item.get(
                "agreement_count",
                1,
            ),
        ),
        reverse=True,
    )

    return candidates


# ============================================================
# EVIDENCE FILTERING
# ============================================================

def is_low_value_for_definition(text):
    """
    Determine whether evidence is unsuitable for a definition.
    """

    if not text:
        return True

    lower = text.lower()

    return any(
        phrase in lower
        for phrase in (
            "according to",
            "for example",
            "such as",
            "applications include",
            "used in",
            "history of",
        )
    )


def should_send_to_llm(
    evidence,
    query,
):
    """
    Determine whether there is enough structured evidence
    to justify Gemini synthesis.
    """

    if not evidence:
        return False

    useful = [
        item
        for item in evidence
        if isinstance(
            item,
            dict,
        )
        and item.get(
            "text"
        )
        and not is_noise_claim(
            item.get(
                "text",
                "",
            )
        )
    ]

    if not useful:
        return False

    return True


# ============================================================
# EVIDENCE LEDGER
# ============================================================

def build_evidence_ledger(
    evidence,
    query="",
    intent=None,
):
    """
    Build a structured evidence ledger.
    """

    if intent is None:
        intent = detect_query_intent(
            query
        )

    definition = choose_best_definition(
        evidence
    )

    mechanism = choose_best_mechanism(
        evidence
    )

    types = choose_best_types(
        evidence
    )

    applications = choose_best_applications(
        evidence
    )

    supporting = []

    for item in evidence:
        if not isinstance(
            item,
            dict,
        ):
            continue

        text = normalize_sentence(
            item.get(
                "text",
                "",
            )
        )

        if not text:
            continue

        if is_noise_claim(text):
            continue

        if (
            definition
            and text
            == definition.get(
                "text",
                "",
            )
        ):
            continue

        if (
            mechanism
            and text
            == mechanism.get(
                "text",
                "",
            )
        ):
            continue

        supporting.append(item)

    return {
        "query": query,
        "intent": intent,
        "definition": definition,
        "mechanism": mechanism,
        "types": types,
        "applications": applications,
        "supporting": supporting,
    }


# ============================================================
# GEMINI CONTEXT
# ============================================================

def _sample_ledger(ledger):
    """
    Produce a compact representative view of the ledger.
    """

    sample = {
        "definition": ledger.get(
            "definition"
        ),
        "mechanism": ledger.get(
            "mechanism"
        ),
        "types": ledger.get(
            "types"
        ),
        "applications": ledger.get(
            "applications"
        ),
        "supporting": ledger.get(
            "supporting",
            [],
        )[:5],
    }

    return sample


def build_gemini_context(ledger):
    """
    Build evidence context for Gemini.
    """

    if not ledger:
        return ""

    sections = []

    definition = ledger.get(
        "definition"
    )

    if definition:
        text = normalize_sentence(
            definition.get(
                "text",
                "",
            )
        )

        source_id = definition.get(
            "source_id"
        )

        if text:
            sections.append(
                f"[{source_id}] {text}"
            )

    mechanism = ledger.get(
        "mechanism"
    )

    if mechanism:
        text = normalize_sentence(
            mechanism.get(
                "text",
                "",
            )
        )

        source_id = mechanism.get(
            "source_id"
        )

        if text:
            sections.append(
                f"[{source_id}] {text}"
            )

    types = ledger.get(
        "types"
    )

    if types:
        category = types.get(
            "category"
        )

        if category:
            text = normalize_sentence(
                category.get(
                    "text",
                    "",
                )
            )

            source_id = category.get(
                "source_id"
            )

            if text:
                sections.append(
                    f"[{source_id}] {text}"
                )

        descriptions = types.get(
            "descriptions",
            {},
        )

        for type_label, item in descriptions.items():
            text = normalize_sentence(
                item.get(
                    "text",
                    "",
                )
            )

            source_id = item.get(
                "source_id"
            )

            if text:
                sections.append(
                    f"[{source_id}] {text}"
                )

    applications = ledger.get(
        "applications",
        [],
    )

    for item in applications:
        text = normalize_sentence(
            item.get(
                "text",
                "",
            )
        )

        source_id = item.get(
            "source_id"
        )

        if text:
            sections.append(
                f"[{source_id}] {text}"
            )

    supporting = ledger.get(
        "supporting",
        [],
    )

    for item in supporting[:5]:
        text = normalize_sentence(
            item.get(
                "text",
                "",
            )
        )

        source_id = item.get(
            "source_id"
        )

        if text:
            sections.append(
                f"[{source_id}] {text}"
            )

    deduplicated = []

    for section in sections:
        if section not in deduplicated:
            deduplicated.append(
                section
            )

    return "\n".join(
        deduplicated
    )


# ============================================================
# GEMINI RETRY / CACHE HELPERS
# ============================================================

def _sanitize_error_message(error):
    """
    Redact configured secrets and credential-like values from error text.
    """

    return sanitize_error_message(error)


def _is_retryable_gemini_error(error):
    """
    Determine whether a Gemini error is likely temporary.
    """

    message = str(error).lower()

    retry_markers = (
        "429",
        "500",
        "502",
        "503",
        "504",
        "timeout",
        "timed out",
        "temporarily unavailable",
        "connection",
        "rate limit",
        "rate_limit",
        "server error",
        "internal server",
    )

    return any(
        marker in message
        for marker in retry_markers
    )


def _retry_delay(attempt):
    """
    Calculate bounded exponential backoff.
    """

    return retry_delay(
        attempt,
        SETTINGS.retry_base_seconds,
        SETTINGS.retry_max_seconds,
        SETTINGS.retry_jitter_seconds,
    )


def _gemini_request_kwargs():
    """Return request options supported by the installed Gemini client."""

    try:
        parameters = inspect.signature(
            _GEMINI_CLIENT.interactions.create
        ).parameters
    except (
        TypeError,
        ValueError,
        AttributeError,
    ):
        return {}

    if (
        "timeout" not in parameters
        and not any(
            parameter.kind
            is inspect.Parameter.VAR_KEYWORD
            for parameter in parameters.values()
        )
    ):
        return {}

    return {
        "timeout": SETTINGS.request_timeout_seconds,
    }


def _get_gemini_cache():
    """
    Return the Gemini cache using the current application settings.
    """

    return CacheStore(
        directory=SETTINGS.cache_dir,
        ttl_seconds=SETTINGS.cache_ttl_seconds,
        enabled=(
            SETTINGS.cache_enabled
            and SETTINGS.gemini_cache_enabled
        ),
    )


def _build_gemini_cache_key(
    query,
    intent,
    context,
):
    """
    Build a deterministic Gemini cache key.
    """

    return CacheStore.build_key(
        "gemini",
        {
            "model": SETTINGS.gemini_model,
            "query": query,
            "intent": intent,
            "context": context,
        },
    )


def _get_gemini_client(api_key):
    """
    Return a reusable Gemini client.

    The factory identity is tracked so monkeypatched test clients
    automatically invalidate the previous client instance.
    """

    global _GEMINI_CLIENT
    global _GEMINI_CLIENT_FACTORY

    factory = genai.Client

    if (
        _GEMINI_CLIENT is None
        or _GEMINI_CLIENT_FACTORY is not factory
    ):
        _GEMINI_CLIENT = factory(
            api_key=api_key
        )
        _GEMINI_CLIENT_FACTORY = factory

    return _GEMINI_CLIENT


# ============================================================
# GEMINI SYNTHESIS
# ============================================================

def generate_with_gemini(
    query,
    ledger,
):
    """
    Generate the final answer using Gemini.

    Successful responses can be cached when Gemini caching is enabled.
    Failed responses are never cached.

    When Gemini caching is disabled, the reusable Gemini client is
    deliberately cleared before each request. This prevents stale
    client state from affecting uncached calls and also makes mocked
    calls deterministic in tests.
    """

    global _GEMINI_CLIENT
    global _GEMINI_CLIENT_FACTORY

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY was not found in .env"
        )

    context = build_gemini_context(
        ledger
    )

    if not context:
        raise ValueError(
            "No usable evidence was available for Gemini synthesis."
        )

    intent = ledger.get(
        "intent",
        "general",
    )

    prompt = f"""
You are ResearchPilot, an evidence-grounded research assistant.

Research question:
{query}

Question intent:
{intent}

Use ONLY the evidence supplied below.

Do not use outside knowledge.

Every factual statement must be directly supported by the supplied
evidence.

Rules:

1. Do not invent facts.
2. Do not infer unsupported facts.
3. Match the answer structure to the question intent.
4. If the question asks for types, focus on the types and their
   descriptions. Do not add unrelated definition, history,
   mechanism, or application material.
5. If the question asks how something works, focus on the mechanism.
6. If the question asks for applications or examples, focus on those.
7. Do not treat an arbitrary supporting sentence as a definition.
8. Remove duplicate claims.
9. Do not add unsupported statistics or historical claims.
10. Keep citations in [number] form.
11. Output normal readable text.
12. Do not escape punctuation with backslashes.
13. Do not put every word or sentence on a separate line.
14. Keep the answer concise but informative.
15. If the evidence does not support a requested point, say so rather
    than filling the gap with outside knowledge.

Evidence:

{context}
"""

    cache_enabled = (
        SETTINGS.cache_enabled
        and SETTINGS.gemini_cache_enabled
    )

    cache = (
        _get_gemini_cache()
        if cache_enabled
        else None
    )

    cache_key = (
        _build_gemini_cache_key(
            query,
            intent,
            context,
        )
        if cache_enabled
        else None
    )

    if (
        cache is not None
        and cache_key is not None
    ):
        cached = cache.get(
            cache_key
        )

        if isinstance(cached, str) and cached.strip():
            logger.info(
                "Gemini cache hit."
            )

            return clean_gemini_markdown(
                cached
            )

        if cached is not None:
            logger.warning(
                "Gemini cache entry ignored because it was malformed."
            )

        logger.info(
            "Gemini cache miss."
        )

    # Response caching and client reuse are separate concerns.
    #
    # When Gemini response caching is enabled, reuse the module-level
    # client normally. When response caching is disabled, invalidate the
    # reusable client before this synthesis call and then obtain a fresh
    # client through the same factory path. The fresh client is reused
    # across retry attempts within this call.
    #
    # Keeping client construction behind _get_gemini_client() is
    # important because it preserves dependency injection and mocked
    # clients used by the test suite.
    if not cache_enabled:
        _GEMINI_CLIENT = None
        _GEMINI_CLIENT_FACTORY = None

    client = _get_gemini_client(
        api_key
    )

    max_retries = (
        SETTINGS.gemini_max_retries
    )
    retry_started_at = time.monotonic()

    last_error = None

    logger.info(
        "Starting Gemini generation (max_attempts=%s).",
        max_retries + 1,
    )

    for attempt in range(
        max_retries + 1
    ):
        try:
            response = client.interactions.create(
                model=SETTINGS.gemini_model,
                input=prompt,
                **_gemini_request_kwargs(),
            )

            answer = getattr(
                response,
                "output_text",
                None,
            )

            if not answer:
                answer = str(response)

            cleaned = clean_gemini_markdown(
                answer
            )

            if (
                cache is not None
                and cache_key is not None
            ):
                try:
                    cache.set(
                        cache_key,
                        cleaned,
                    )
                except Exception as cache_error:
                    logger.warning(
                        "Gemini response cache write failed: %s",
                        _sanitize_error_message(cache_error),
                    )
                else:
                    logger.info(
                        "Gemini response cached."
                    )

            return cleaned

        except Exception as error:
            last_error = error

            logger.warning(
                "Gemini generation failed "
                "on attempt %s/%s: %s",
                attempt + 1,
                max_retries + 1,
                _sanitize_error_message(error),
            )

            if attempt >= max_retries:
                break

            if not _is_retryable_gemini_error(
                error
            ):
                logger.error(
                    "Gemini error is not retryable."
                )
                break

            delay = retry_delay(
                attempt,
                SETTINGS.retry_base_seconds,
                SETTINGS.retry_max_seconds,
                SETTINGS.retry_jitter_seconds,
                retry_after_seconds(error),
            )

            if not retry_budget_allows(
                retry_started_at,
                delay,
                SETTINGS.retry_budget_seconds,
            ):
                logger.warning(
                    "Gemini retry budget exhausted."
                )
                break

            logger.info(
                "Retrying Gemini generation "
                "in %.2f seconds.",
                delay,
            )

            time.sleep(
                delay
            )

    safe_error_message = _sanitize_error_message(
        last_error
    )

    logger.error(
        "Gemini generation exhausted all retry attempts."
    )

    raise RuntimeError(
        "Gemini synthesis failed: "
        f"{safe_error_message}"
    ) from None


# ============================================================
# TYPE DESCRIPTION CLEANING
# ============================================================

def clean_type_description(
    text,
    type_label,
):
    """
    Clean scraped type headings while preserving the actual claim.
    """

    text = normalize_sentence(
        text
    )

    if not text or not type_label:
        return text

    label = type_label.strip()

    base = re.sub(
        r"\s+Learning$",
        "",
        label,
        flags=re.IGNORECASE,
    ).strip()

    heading_patterns = [
        rf"^\s*{re.escape(base)}\s+Machine\s+Learning\s*",
        rf"^\s*{re.escape(label)}\s*[-:–—]\s*",
    ]

    for pattern in heading_patterns:
        match = re.match(
            pattern,
            text,
            flags=re.IGNORECASE,
        )

        if not match:
            continue

        remainder = text[
            match.end():
        ].strip()

        if not remainder:
            continue

        return remainder

    return text


# ============================================================
# FALLBACK HELPERS
# ============================================================

def choose_best_supporting_evidence(
    ledger,
    limit=5,
):
    """
    Return the strongest remaining evidence for fallback output.
    """

    supporting = ledger.get(
        "supporting",
        [],
    )

    if not supporting:
        return []

    candidates = [
        item
        for item in supporting
        if isinstance(
            item,
            dict,
        )
        and normalize_sentence(
            item.get(
                "text",
                "",
            )
        )
    ]

    candidates.sort(
        key=lambda item: (
            evidence_score(
                item.get(
                    "text",
                    "",
                ),
                ledger.get(
                    "query",
                    "",
                ),
                item.get(
                    "quality_score",
                    0.0,
                ),
                item.get(
                    "agreement_count",
                    1,
                ),
            ),
            item.get(
                "confidence",
                0.0,
            ),
        ),
        reverse=True,
    )

    return candidates[:limit]


def add_sources_to_fallback(
    sections,
    evidence,
):
    """
    Add source references to fallback sections.
    """

    source_ids = set()

    for item in evidence:
        source_id = item.get(
            "source_id"
        )

        if source_id:
            source_ids.add(
                source_id
            )

    if source_ids:
        sections.append(
            "Sources: "
            + ", ".join(
                sorted(
                    source_ids
                )
            )
        )


def fallback_synthesis(
    query,
    ledger,
):
    """
    Build a deterministic evidence-based fallback answer.
    """

    intent = ledger.get(
        "intent",
        detect_query_intent(query),
    )

    sections = [
        "RESEARCH SUMMARY",
    ]
    source_ids = set()
    content_added = False

    # ========================================================
    # DEFINITION
    # ========================================================

    if intent == "definition":

        definition = ledger.get(
            "definition"
        )

        if definition:

            text = normalize_sentence(
                definition.get(
                    "text",
                    "",
                )
            )

            source_id = definition.get(
                "source_id"
            )

            if text:

                sections.extend(
                    [
                        "Definition",
                        "",
                        f"{text} [{source_id}]",
                    ]
                )

                source_ids.add(
                    source_id
                )

                content_added = True

        mechanism = ledger.get(
            "mechanism"
        )

        if mechanism:

            text = normalize_sentence(
                mechanism.get(
                    "text",
                    "",
                )
            )

            source_id = mechanism.get(
                "source_id"
            )

            if text:

                sections.extend(
                    [
                        "",
                        "How It Works",
                        "",
                        f"{text} [{source_id}]",
                    ]
                )

                source_ids.add(
                    source_id
                )

                content_added = True

    # ========================================================
    # MECHANISM
    # ========================================================

    elif intent == "mechanism":

        mechanism = ledger.get(
            "mechanism"
        )

        if mechanism:

            text = normalize_sentence(
                mechanism.get(
                    "text",
                    "",
                )
            )

            source_id = mechanism.get(
                "source_id"
            )

            if text:

                sections.extend(
                    [
                        "How It Works",
                        "",
                        f"{text} [{source_id}]",
                    ]
                )

                source_ids.add(
                    source_id
                )

                content_added = True

        definition = ledger.get(
            "definition"
        )

        if definition:

            text = normalize_sentence(
                definition.get(
                    "text",
                    "",
                )
            )

            source_id = definition.get(
                "source_id"
            )

            if text:

                sections.extend(
                    [
                        "",
                        "Definition",
                        "",
                        f"{text} [{source_id}]",
                    ]
                )

                source_ids.add(
                    source_id
                )

                content_added = True

    # ========================================================
    # TYPES
    # ========================================================

    elif intent == "types":

        types = ledger.get(
            "types"
        )

        if types:

            category = types.get(
                "category"
            )

            if category:

                category_text = normalize_sentence(
                    category.get(
                        "text",
                        "",
                    )
                )

                category_source = category.get(
                    "source_id"
                )

                if category_text:

                    sections.extend(
                        [
                            "Types",
                            "",
                            f"{category_text} "
                            f"[{category_source}]",
                        ]
                    )

                    source_ids.add(
                        category_source
                    )

                    content_added = True

            descriptions = types.get(
                "descriptions",
                {},
            )

            if descriptions:

                if content_added:
                    sections.extend(
                        [
                            "",
                            "Descriptions",
                            "",
                        ]
                    )
                else:
                    sections.extend(
                        [
                            "Types",
                            "",
                        ]
                    )

                for (
                    type_label,
                    item,
                ) in descriptions.items():

                    text = clean_type_description(
                        item.get(
                            "text",
                            "",
                        ),
                        type_label,
                    )

                    source_id = item.get(
                        "source_id"
                    )

                    if not text:
                        continue

                    sections.append(
                        f"- {type_label}: "
                        f"{text} [{source_id}]"
                    )

                    source_ids.add(
                        source_id
                    )

                    content_added = True

    # ========================================================
    # APPLICATIONS
    # ========================================================

    elif intent == "applications":

        applications = ledger.get(
            "applications",
            [],
        )

        if applications:

            sections.extend(
                [
                    "Applications",
                    "",
                ]
            )

            for item in applications:

                text = normalize_sentence(
                    item.get(
                        "text",
                        "",
                    )
                )

                source_id = item.get(
                    "source_id"
                )

                if not text:
                    continue

                sections.append(
                    f"- {text} [{source_id}]"
                )

                source_ids.add(
                    source_id
                )

                content_added = True

    # ========================================================
    # GENERAL
    # ========================================================

    if not content_added:

        supporting = choose_best_supporting_evidence(
            ledger
        )

        if supporting:

            sections.extend(
                [
                    "Evidence",
                    "",
                ]
            )

            for item in supporting:

                text = normalize_sentence(
                    item.get(
                        "text",
                        "",
                    )
                )

                source_id = item.get(
                    "source_id"
                )

                if not text:
                    continue

                sections.append(
                    f"- {text} [{source_id}]"
                )

                source_ids.add(
                    source_id
                )

                content_added = True

    if not content_added:
        return (
            "The available evidence does not "
            "support a reliable answer."
        )

    if source_ids:
        source_metadata = {}

        for source in ledger.get(
            "_sources",
            [],
        ):
            if not isinstance(source, dict):
                continue

            source_id = source.get(
                "id"
            )

            if source_id is None:
                continue

            source_metadata[
                str(source_id)
            ] = source

        source_lines = []

        normalized_source_ids = {
            str(source_id)
            for source_id in source_ids
        }

        for source_id in sorted(
            normalized_source_ids
        ):
            source = source_metadata.get(
                source_id
            )

            if source is None:
                for candidate in ledger.get(
                    "_sources",
                    [],
                ):
                    if not isinstance(
                        candidate,
                        dict,
                    ):
                        continue

                    if str(
                        candidate.get("id")
                    ) == source_id:
                        source = candidate
                        break

            if source:
                title = source.get(
                    "title",
                    "",
                ).strip()
                url = source.get(
                    "url",
                    "",
                ).strip()

                if title and url:
                    source_lines.append(
                        f"[{source_id}] {title} — {url}"
                    )
                elif url:
                    source_lines.append(
                        f"[{source_id}] {url}"
                    )
                elif title:
                    source_lines.append(
                        f"[{source_id}] {title}"
                    )
                else:
                    source_lines.append(
                        f"[{source_id}]"
                    )
            else:
                source_lines.append(
                    source_id
                )

        sections.extend(
            [
                "",
                "Sources",
                "",
                "\n".join(source_lines),
            ]
        )

    return "\n".join(
        sections
    )


# ============================================================
# FINAL ANSWER
# ============================================================

def generate_answer(
    query,
    ledger,
    use_gemini=True,
):
    """Generate the final research answer."""

    if isinstance(ledger, list):
        ledger_sources = list(ledger)
        all_evidence = []

        for source in ledger_sources:
            if not isinstance(source, dict):
                continue

            source_id = source.get("id")
            source_title = source.get("title", "")
            source_url = source.get("url", "")
            source_quality = source.get("quality_score", 0.0)

            for evidence_item in source.get("evidence", []):
                if not isinstance(evidence_item, dict):
                    continue

                item = dict(evidence_item)
                item["source_id"] = source_id
                item["source_title"] = source_title
                item["source_url"] = source_url
                item["quality_score"] = source_quality
                all_evidence.append(item)

        ledger = build_evidence_ledger(
            all_evidence,
            query=query,
        )

        ledger["_sources"] = [
            {
                "id": source.get("id"),
                "title": source.get("title", ""),
                "url": source.get("url", ""),
            }
            for source in ledger_sources
            if isinstance(source, dict)
        ]

    if use_gemini and should_send_to_llm(
        ledger.get("supporting", []),
        query,
    ):
        try:
            return generate_with_gemini(
                query,
                ledger,
            )
        except Exception as error:
            logger.warning(
                "Gemini synthesis failed; using evidence-based fallback: %s",
                _sanitize_error_message(error),
            )

    return fallback_synthesis(
        query,
        ledger,
    )
