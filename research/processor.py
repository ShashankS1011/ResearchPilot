import hashlib
import re
from urllib.parse import urlparse

from .config import SETTINGS
from .logging_config import logger


# ============================================================
# GENERAL CLEANING
# ============================================================

def normalize_whitespace(text):
    if text is None:
        return ""

    text = str(text)

    text = text.replace(
        "\xa0",
        " ",
    )

    text = text.replace(
        "\r\n",
        "\n",
    )

    text = text.replace(
        "\r",
        "\n",
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

    return text.strip()


def clean_scraped_markup(text):
    """
    Remove common extraction artifacts without trying to
    rewrite the actual source meaning.
    """

    if not text:
        return ""

    text = str(text)

    # Remove Markdown links while retaining visible text.
    text = re.sub(
        r"\[([^\]]+)\]\((?:https?://|www\.)[^)]+\)",
        r"\1",
        text,
        flags=re.IGNORECASE,
    )

    # Remove standalone URLs.
    text = re.sub(
        r"https?://\S+",
        " ",
        text,
        flags=re.IGNORECASE,
    )

    # Remove HTML tags.
    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

    # Common Markdown heading markers.
    text = re.sub(
        r"(?m)^\s{0,3}#{1,6}\s*",
        "",
        text,
    )

    # Horizontal rules.
    text = re.sub(
        r"(?m)^\s*[-*_]{3,}\s*$",
        "",
        text,
    )

    # Common scraped escape artifacts.
    replacements = {
        "\\=": "=",
        "\\-": "-",
        "\\_": "_",
        "\\*": "*",
        "&nbsp;": " ",
        "&amp;": "&",
        "&quot;": '"',
        "&#39;": "'",
    }

    for old, new in replacements.items():
        text = text.replace(
            old,
            new,
        )

    return normalize_whitespace(text)


def clean_evidence_artifacts(text):
    """
    Remove obvious navigation, metadata, marketing, table, and
    extraction artifacts while preserving useful evidence lines.
    """

    if text is None:
        return ""

    raw_text = str(text).replace(
        "\\r\\n",
        "\\n",
    ).replace(
        "\\r",
        "\\n",
    )

    if not raw_text.strip():
        return ""

    lines = []

    # Process original lines before whitespace normalization. This is
    # important because clean_scraped_markup() intentionally collapses
    # whitespace, which would otherwise merge useful evidence with
    # cookie/navigation lines.
    raw_lines = raw_text.split("\n")

    if not raw_lines:
        raw_lines = [raw_text]

    for raw_line in raw_lines:
        line = clean_scraped_markup(
            raw_line
        ).strip()

        if not line:
            continue

        if (
            is_noise_line(line)
            or looks_like_navigation(line)
            or looks_like_marketing(line)
            or looks_like_table_artifact(line)
            or looks_like_metadata(line)
            or looks_like_instructional_meta(line)
        ):
            continue

        lines.append(line)

    text = " ".join(lines)

    if not text:
        return ""

    # Repair common heading punctuation artifacts introduced by extraction,
    # e.g. "main. types" -> "main types". Keep this deliberately narrow.
    text = re.sub(
        r"\b(main|major|minor|different|several|two|three|four|five)\.\s+(?=types\b)",
        r"\1 ",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"\s{2,}",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# NOISE DETECTION
# ============================================================

NAVIGATION_PATTERNS = [
    r"^menu$",
    r"^home$",
    r"^search$",
    r"^login$",
    r"^sign in$",
    r"^sign up$",
    r"^subscribe$",
    r"^share$",
    r"^follow us$",
    r"^advertisement$",
    r"^advertising$",
    r"^skip to content$",
    r"^table of contents$",
    r"^contents$",
    r"^read more$",
    r"^learn more$",
    r"^click here$",
    r"^related articles$",
    r"^related posts$",
]


def is_noise_line(text):
    if not text:
        return True

    value = text.strip()

    if len(value) < 3:
        return True

    lower = value.lower()

    for pattern in NAVIGATION_PATTERNS:
        if re.fullmatch(
            pattern,
            lower,
            flags=re.IGNORECASE,
        ):
            return True

    if re.fullmatch(
        r"[\W_]+",
        value,
    ):
        return True

    # Cookie/privacy notices are rarely useful evidence.
    if (
        "cookie policy" in lower
        and len(value) < 250
    ):
        return True

    if (
        "privacy policy" in lower
        and len(value) < 250
    ):
        return True

    if (
        "terms of service" in lower
        and len(value) < 250
    ):
        return True

    # Obvious UI counters.
    if re.fullmatch(
        r"(share|likes?|comments?)\s*[:\-]?\s*\d+",
        lower,
    ):
        return True

    return False


def looks_like_navigation(text):
    if not text:
        return True

    lower = text.lower()

    navigation_words = (
        "home",
        "menu",
        "search",
        "login",
        "sign in",
        "subscribe",
        "contact us",
        "privacy policy",
        "terms of service",
        "cookie policy",
    )

    matches = sum(
        word in lower
        for word in navigation_words
    )

    return matches >= 2 and len(text) < 250


def looks_like_metadata(text):
    if not text:
        return True

    lower = text.lower()

    metadata_markers = (
        "published on",
        "updated on",
        "last updated",
        "author:",
        "written by",
        "copyright",
        "all rights reserved",
    )

    return any(
        marker in lower
        for marker in metadata_markers
    ) and len(text) < 300


def looks_like_instructional_meta(text):
    if not text:
        return True

    lower = text.lower()

    patterns = (
        "in this article",
        "in this video",
        "in this tutorial",
        "in this lesson",
        "watch the video",
        "click the link",
        "subscribe to",
        "follow us on",
    )

    return (
        any(
            pattern in lower
            for pattern in patterns
        )
        and len(text) < 300
    )


# ============================================================
# TEXT SEGMENTATION
# ============================================================





# ============================================================
# EVIDENCE QUALITY
# ============================================================

def sentence_quality_score(text):
    if not text:
        return 0.0

    score = 0.5

    length = len(text)

    if 50 <= length <= 500:
        score += 0.2

    if 100 <= length <= 350:
        score += 0.1

    if re.search(
        r"[.!?]",
        text,
    ):
        score += 0.05

    if re.search(
        r"\b(is|are|means|refers|defined|used|includes|allows|provides|requires|consists)\b",
        text,
        flags=re.IGNORECASE,
    ):
        score += 0.1

    if looks_like_metadata(
        text
    ):
        score -= 0.3

    if looks_like_instructional_meta(
        text
    ):
        score -= 0.2

    if looks_like_navigation(
        text
    ):
        score -= 0.4

    if len(text) < 25:
        score -= 0.3

    if len(text) > 1000:
        score -= 0.2

    return max(
        0.0,
        min(
            1.0,
            score,
        ),
    )


def classify_evidence(text):
    if not text:
        return "unknown"

    lower = text.lower()

    if any(
        word in lower
        for word in (
            "according to",
            "research shows",
            "study found",
            "study reports",
            "survey found",
        )
    ):
        return "reported_finding"

    if any(
        word in lower
        for word in (
            "defined as",
            "refers to",
            "is a",
            "is an",
            "means",
            "are types",
        )
    ):
        return "definition"

    if any(
        word in lower
        for word in (
            "example",
            "for instance",
            "such as",
        )
    ):
        return "example"

    if any(
        word in lower
        for word in (
            "benefit",
            "advantage",
            "limitation",
            "disadvantage",
            "drawback",
        )
    ):
        return "evaluation"

    if any(
        word in lower
        for word in (
            "can",
            "allows",
            "enables",
            "used to",
            "helps",
            "supports",
        )
    ):
        return "capability"

    return "general_fact"


def evidence_hash(text):
    normalized = re.sub(
        r"\W+",
        " ",
        text.lower(),
    ).strip()

    return hashlib.sha256(
        normalized.encode(
            "utf-8"
        )
    ).hexdigest()


def similarity_key(text):
    text = re.sub(
        r"\W+",
        " ",
        text.lower(),
    )

    words = [
        word
        for word in text.split()
        if len(word) > 2
    ]

    return " ".join(
        words[:60]
    )


def is_duplicate_evidence(
    text,
    existing,
):
    current_key = similarity_key(
        text
    )

    if not current_key:
        return True

    for item in existing:
        previous = item.get(
            "text",
            "",
        )

        previous_key = similarity_key(
            previous
        )

        if current_key == previous_key:
            return True

        current_words = set(
            current_key.split()
        )

        previous_words = set(
            previous_key.split()
        )

        if not current_words:
            continue

        intersection = (
            current_words
            & previous_words
        )

        union = (
            current_words
            | previous_words
        )

        if union:
            jaccard = (
                len(intersection)
                / len(union)
            )

            if jaccard >= 0.85:
                return True

    return False


# ============================================================
# SOURCE QUALITY
# ============================================================



def source_quality_score(
    result,
    extracted_content,
):
    score = 0.45

    url = str(
        result.get(
            "url",
            "",
        )
        or ""
    )

    category = domain_category(
        url
    )

    category_bonus = {
        "academic": 0.25,
        "government": 0.25,
        "authoritative": 0.25,
        "organization": 0.15,
        "reference": 0.10,
        "video": 0.02,
        "general": 0.00,
        "unknown": -0.05,
    }

    score += category_bonus.get(
        category,
        0,
    )

    tavily_score = result.get(
        "score"
    )

    if isinstance(
        tavily_score,
        (int, float),
    ):
        score += min(
            max(
                float(tavily_score),
                0,
            ),
            1,
        ) * 0.15

    if extracted_content:
        score += 0.10

    if len(extracted_content) >= 500:
        score += 0.05

    return max(
        0.0,
        min(
            1.0,
            score,
        ),
    )


# ============================================================
# EVIDENCE EXTRACTION
# ============================================================



# ============================================================
# CROSS-SOURCE AGREEMENT
# ============================================================







# ============================================================
# LLM EVIDENCE SELECTION
# ============================================================



# ============================================================
# MAIN PROCESSOR
# ============================================================


# ============================================================
# ROBUST EVIDENCE PROCESSING
# ============================================================

from urllib.parse import urlparse, urlunparse


def _canonical_url(url):
    """Normalize a URL for deterministic duplicate detection."""
    value = str(url or "").strip()
    if not value:
        return ""

    try:
        parsed = urlparse(value)
    except ValueError:
        return value.lower().rstrip("/")

    hostname = (parsed.hostname or "").lower()
    if hostname.startswith("www."):
        hostname = hostname[4:]

    if not hostname:
        return value.lower().rstrip("/")

    path = parsed.path.rstrip("/") or "/"
    return urlunparse(
        (
            parsed.scheme.lower() or "https",
            hostname,
            path,
            "",
            parsed.query,
            "",
        )
    )


def _hostname(url):
    """Return a normalized hostname."""
    try:
        hostname = (urlparse(str(url or "")).hostname or "").lower()
    except ValueError:
        return ""

    if hostname.startswith("www."):
        hostname = hostname[4:]
    return hostname


def _domain_matches(hostname, domain):
    """Return True only for a domain or one of its subdomains."""
    return hostname == domain or hostname.endswith("." + domain)


def domain_category(url):
    """Classify a source domain without substring-based false positives."""
    hostname = _hostname(url)
    if not hostname:
        return "unknown"

    if hostname.endswith(".edu") or ".edu." in hostname:
        return "academic"
    if hostname.endswith(".gov") or ".gov." in hostname:
        return "government"

    authoritative_domains = (
        "nature.com",
        "science.org",
        "arxiv.org",
        "nih.gov",
        "nasa.gov",
        "who.int",
        "mit.edu",
        "stanford.edu",
        "harvard.edu",
    )
    if any(_domain_matches(hostname, domain) for domain in authoritative_domains):
        return "authoritative"

    reference_domains = (
        "wikipedia.org",
        "britannica.com",
        "coursera.org",
        "geeksforgeeks.org",
    )
    if any(_domain_matches(hostname, domain) for domain in reference_domains):
        return "reference"

    if _domain_matches(hostname, "youtube.com") or hostname == "youtu.be":
        return "video"
    if hostname.endswith(".org") or ".org." in hostname:
        return "organization"
    return "general"


def looks_like_marketing(text):
    """Detect short promotional fragments that should not become evidence."""
    if not text:
        return True
    lower = str(text).lower()
    patterns = (
        "learn more about",
        "get started today",
        "start your free",
        "book a demo",
        "request a demo",
        "contact our team",
        "talk to an expert",
        "download our",
        "sign up for",
        "subscribe to our",
        "explore our",
        "discover our",
        "join our community",
        "ready to get started",
    )
    return len(text) < 350 and any(pattern in lower for pattern in patterns)


def looks_like_table_artifact(text):
    """Detect broken table/source extraction remnants."""
    if not text:
        return True
    value = str(text).strip()
    patterns = (
        r"^\s*source\s*:",
        r"^\s*source\s*\|",
        r"^\s*\|.*\|.*\|",
        r"^\s*[-:]+\s*\|",
    )
    return any(re.search(pattern, value, re.IGNORECASE) for pattern in patterns)


def looks_like_image_metadata(text):
    """Detect short image/figure metadata."""
    if not text:
        return True
    lower = str(text).lower()
    markers = (
        "image:", "alt text:", "alt=", "caption:", "figure:", "thumbnail:"
    )
    return len(text) < 250 and any(marker in lower for marker in markers)


def looks_like_list_fragment(text):
    """Detect numbered or bulleted fragments that are unlikely to stand alone."""
    if not text:
        return True

    value = str(text).strip()

    numbered = re.match(
        r"^\(?\d{1,3}[.)]\s+",
        value,
    )
    if numbered:
        return True

    bullet = re.match(
        r"^[•▪◦*-]\s+",
        value,
    )
    return bool(bullet)


def evidence_cleanliness_score(text):
    """Return a deterministic 0..1 cleanliness estimate."""
    if not text:
        return 0.0

    score = 1.0
    if looks_like_navigation(text):
        score -= 0.45
    if looks_like_metadata(text):
        score -= 0.30
    if looks_like_instructional_meta(text):
        score -= 0.25
    if looks_like_marketing(text):
        score -= 0.30
    if looks_like_table_artifact(text):
        score -= 0.35
    if looks_like_image_metadata(text):
        score -= 0.30
    if len(text) < 30:
        score -= 0.25
    if len(text) > 800:
        score -= 0.15
    return max(0.0, min(1.0, score))


def split_extraction_sections(text):
    """Split extracted text into paragraph-like sections."""
    text = clean_scraped_markup(text)
    if not text:
        return []

    sections = re.split(r"\n\s*\n+", text)
    cleaned = []
    for section in sections:
        section = clean_evidence_artifacts(section)
        if not section:
            continue
        if looks_like_navigation(section) or looks_like_marketing(section):
            continue
        cleaned.append(section)
    return cleaned


def split_into_sentences(text):
    """Conservatively split evidence while protecting common abbreviations."""
    text = clean_evidence_artifacts(text)
    if not text:
        return []

    protected = text
    abbreviations = {
        "e.g.": "EGTOKEN",
        "i.e.": "IETOKEN",
        "etc.": "ETCTOKEN",
        "U.S.": "USTOKEN",
        "U.K.": "UKTOKEN",
        "Dr.": "DRTOKEN",
        "Mr.": "MRTOKEN",
        "Ms.": "MSTOKEN",
    }
    for original, token in abbreviations.items():
        protected = protected.replace(original, token)

    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9“\"'])", protected)
    reverse = {token: original for original, token in abbreviations.items()}
    restored = []
    pending_list_marker = ""

    for part in parts:
        part = part.strip()
        for token, original in reverse.items():
            part = part.replace(token, original)

        if not part:
            continue

        # Sentence splitting can isolate numbered-list markers such as
        # ``1.`` from the text that follows. Keep the marker attached so
        # the list-fragment detector can reject the complete item.
        if re.fullmatch(r"\d{1,3}[.)]", part):
            pending_list_marker = part
            continue

        if pending_list_marker:
            part = f"{pending_list_marker} {part}"
            pending_list_marker = ""

        restored.append(part)

    if pending_list_marker:
        restored.append(pending_list_marker)

    return restored


def _token_set(text):
    """Return informative normalized tokens."""
    words = re.findall(
        r"\b[a-zA-Z0-9][a-zA-Z0-9_-]*\b",
        str(text or "").lower(),
    )
    stopwords = {
        "the", "a", "an", "and", "or", "of", "to", "in", "is", "are",
        "for", "on", "with", "that", "this", "it", "as", "by", "from",
        "can", "be", "used", "use", "using", "their", "they", "these",
        "those", "which", "into", "than", "such", "its", "has", "have",
        "also",
    }
    normalized = []
    for word in words:
        if word in stopwords or len(word) <= 2:
            continue
        # Lightweight normalization for common inflectional variants.
        # This keeps the similarity metric deterministic and dependency-free.
        if len(word) > 5 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 5 and word.endswith("s"):
            word = word[:-1]
        normalized.append(word)
    return set(normalized)


def _ordered_content_tokens(text):
    """Return informative tokens in source order for deterministic n-grams."""
    words = re.findall(
        r"\b[a-zA-Z0-9][a-zA-Z0-9_-]*\b",
        str(text or "").lower(),
    )
    stopwords = {
        "the", "a", "an", "and", "or", "of", "to", "in", "is", "are",
        "for", "on", "with", "that", "this", "it", "as", "by", "from",
        "can", "be", "used", "use", "using", "their", "they", "these",
        "those", "which", "into", "than", "such", "its", "has", "have",
        "also",
    }
    normalized = []
    for word in words:
        if word in stopwords or len(word) <= 2:
            continue
        if len(word) > 5 and word.endswith("ies"):
            word = word[:-3] + "y"
        elif len(word) > 5 and word.endswith("s"):
            word = word[:-1]
        normalized.append(word)
    return normalized


def _content_terms(text):
    tokens = _token_set(text)
    specific = {token for token in tokens if len(token) >= 5}
    return specific or tokens


def _token_jaccard(first, second):
    first_tokens = _content_terms(first)
    second_tokens = _content_terms(second)
    if not first_tokens or not second_tokens:
        return 0.0
    union = first_tokens | second_tokens
    return len(first_tokens & second_tokens) / len(union) if union else 0.0


def _containment_similarity(first, second):
    first_tokens = _content_terms(first)
    second_tokens = _content_terms(second)
    if not first_tokens or not second_tokens:
        return 0.0
    shared = first_tokens & second_tokens
    return max(
        len(shared) / len(first_tokens),
        len(shared) / len(second_tokens),
    )


def _key_phrase_overlap(first, second):
    first_tokens = _ordered_content_tokens(first)
    second_tokens = _ordered_content_tokens(second)
    if len(first_tokens) < 2 or len(second_tokens) < 2:
        return 0.0

    first_phrases = {
        (first_tokens[i], first_tokens[i + 1])
        for i in range(len(first_tokens) - 1)
    }
    second_phrases = {
        (second_tokens[i], second_tokens[i + 1])
        for i in range(len(second_tokens) - 1)
    }
    union = first_phrases | second_phrases
    if not union:
        return 0.0
    return len(first_phrases & second_phrases) / len(union)


def evidence_similarity(first, second):
    """Calculate conservative semantic-ish similarity between two claims."""
    first = normalize_whitespace(first)
    second = normalize_whitespace(second)
    if not first or not second:
        return 0.0
    if similarity_key(first) == similarity_key(second):
        return 1.0

    shared = _content_terms(first) & _content_terms(second)
    if len(shared) < 3:
        return 0.0

    score = max(
        _token_jaccard(first, second),
        _containment_similarity(first, second) * 0.95,
        _key_phrase_overlap(first, second) * 1.10,
    )
    return min(1.0, score)


def extract_evidence(search_result, extracted_result=None):
    """Produce cleaned, deduplicated evidence from Tavily search/extraction data."""
    candidates = []
    snippet = search_result.get("content", "") if isinstance(search_result, dict) else ""
    if snippet:
        candidates.append(("search_result", snippet))

    if isinstance(extracted_result, dict):
        raw_content = extracted_result.get("raw_content", "")
        if raw_content:
            candidates.append(("extracted_content", raw_content))

    evidence = []
    seen_hashes = set()

    for origin, content in candidates:
        for section in split_extraction_sections(content):
            for sentence in split_into_sentences(section):
                cleaned = clean_evidence_artifacts(sentence)
                if not cleaned or len(cleaned) < 30:
                    continue
                if len(cleaned) > 800:
                    cleaned = cleaned[:800].rsplit(" ", 1)[0]

                if (
                    looks_like_navigation(cleaned)
                    or looks_like_metadata(cleaned)
                    or looks_like_instructional_meta(cleaned)
                    or looks_like_marketing(cleaned)
                    or looks_like_table_artifact(cleaned)
                    or looks_like_image_metadata(cleaned)
                    or looks_like_list_fragment(cleaned)
                ):
                    continue

                cleanliness = evidence_cleanliness_score(cleaned)
                if cleanliness < 0.60:
                    continue

                quality = sentence_quality_score(cleaned)
                if quality < 0.35:
                    continue

                item_hash = evidence_hash(cleaned)
                if item_hash in seen_hashes or is_duplicate_evidence(cleaned, evidence):
                    continue

                seen_hashes.add(item_hash)
                evidence.append(
                    {
                        "text": cleaned,
                        "confidence": round(quality, 3),
                        "cleanliness_score": round(cleanliness, 3),
                        "classification": classify_evidence(cleaned),
                        "origin": origin,
                        "risk_flag": None,
                        "supporting_sources": [],
                        "possible_supporting_sources": [],
                        "agreement_count": 1,
                    }
                )

    evidence.sort(
        key=lambda item: (
            item.get("confidence", 0.0),
            item.get("cleanliness_score", 0.0),
            len(item.get("text", "")),
        ),
        reverse=True,
    )
    return evidence[:SETTINGS.max_evidence_per_source]


def _classify_relationship(similarity):
    """Map similarity into conservative cross-source relationships."""
    if similarity >= 0.60:
        return "agreement"
    if similarity >= 0.50:
        return "possible_agreement"
    return "unrelated"


def _has_negation_marker(text):
    """Return whether a claim contains an explicit negation."""
    return bool(
        re.search(
            r"\b(?:not|no|never|cannot|can't|doesn't|does not|"
            r"without|fails?|lack(?:s|ing)?)\b",
            str(text or ""),
            flags=re.IGNORECASE,
        )
    )


def _has_numeric_disagreement(first, second):
    """Return whether closely related claims cite different numbers."""
    first_numbers = set(
        re.findall(r"\b\d+(?:\.\d+)?%?\b", str(first or ""))
    )
    second_numbers = set(
        re.findall(r"\b\d+(?:\.\d+)?%?\b", str(second or ""))
    )
    return bool(first_numbers and second_numbers and first_numbers != second_numbers)


def _is_contradiction(first, second, similarity):
    """Detect only explicit, high-overlap disagreement signals."""
    if similarity < 0.50:
        return False

    return (
        _has_negation_marker(first) != _has_negation_marker(second)
        or (
            similarity >= 0.60
            and _has_numeric_disagreement(first, second)
        )
    )


def add_cross_source_agreement(sources):
    """Attach consensus and conservative contradiction metadata."""
    all_items = []
    for source in sources:
        for evidence in source.get("evidence", []):
            if not isinstance(evidence, dict):
                continue
            all_items.append((source, evidence))

    for source, evidence in all_items:
        text = evidence.get("text", "")
        supporting = []
        possible = []
        contradicting = []

        for other_source, other_evidence in all_items:
            if other_source is source:
                continue
            other_id = other_source.get("id")
            if other_id is None:
                continue

            similarity = evidence_similarity(text, other_evidence.get("text", ""))
            relationship = _classify_relationship(similarity)
            other_text = other_evidence.get("text", "")
            if (
                _is_contradiction(text, other_text, similarity)
                and other_id not in contradicting
            ):
                contradicting.append(other_id)
            elif relationship == "agreement" and other_id not in supporting:
                supporting.append(other_id)
            elif relationship == "possible_agreement" and other_id not in possible:
                possible.append(other_id)

        evidence["supporting_sources"] = sorted(supporting)
        evidence["possible_supporting_sources"] = sorted(possible)
        evidence["contradicting_sources"] = sorted(contradicting)
        evidence["agreement_count"] = 1 + len(supporting)
        evidence["agreement_score"] = round(min(1.0, evidence["agreement_count"] / 4.0), 3)
        evidence["contradiction_count"] = len(contradicting)

        if contradicting:
            evidence["risk_flag"] = "cross_source_contradiction"
        elif supporting:
            evidence["risk_flag"] = None
        elif possible:
            evidence["risk_flag"] = "possible_cross_source_support"
        else:
            evidence["risk_flag"] = "single_source"

        base = float(evidence.get("confidence", 0.0))
        bonus = min(0.20, len(supporting) * 0.07)
        penalty = min(0.20, len(contradicting) * 0.10)
        evidence["confidence"] = round(
            max(0.0, min(1.0, base + bonus - penalty)),
            3,
        )

    return sources


def evidence_specificity_score(text):
    tokens = _content_terms(text)
    if not tokens:
        return 0.0
    score = 0.40
    if len(tokens) >= 5:
        score += 0.15
    if len(tokens) >= 8:
        score += 0.10
    if len(tokens) >= 12:
        score += 0.05
    if re.search(r"\b\d+(?:\.\d+)?%?\b", text):
        score += 0.10
    if re.search(r"\b(?:because|therefore|which|that|means|defined|refers|includes|consists|allows|enables)\b", text, re.I):
        score += 0.10
    return max(0.0, min(1.0, score))


def evidence_rank_score(evidence, source_quality=0.0):
    confidence = float(evidence.get("confidence", 0.0))
    cleanliness = float(evidence.get("cleanliness_score", 0.0))
    agreement = float(evidence.get("agreement_score", 0.0))
    specificity = evidence_specificity_score(evidence.get("text", ""))
    score = (
        confidence * 0.30
        + cleanliness * 0.20
        + min(max(source_quality, 0.0), 1.0) * 0.20
        + agreement * 0.20
        + specificity * 0.10
    )
    return round(max(0.0, min(1.0, score)), 3)


def prepare_llm_evidence(evidence, source_quality=0.0):
    """Return a compact ranked evidence list for Gemini."""
    ranked = []
    for item in evidence or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        copied = dict(item)
        copied["rank_score"] = evidence_rank_score(copied, source_quality)
        ranked.append(copied)

    ranked.sort(
        key=lambda item: (
            item.get("rank_score", 0.0),
            item.get("agreement_count", 1),
            item.get("confidence", 0.0),
            len(item.get("text", "")),
        ),
        reverse=True,
    )

    selected = []
    for item in ranked[:5]:
        selected.append(
            {
                "text": item.get("text", ""),
                "confidence": item.get("confidence", 0.0),
                "cleanliness_score": item.get("cleanliness_score", 0.0),
                "classification": item.get("classification", "unknown"),
                "agreement_count": item.get("agreement_count", 1),
                "supporting_sources": item.get("supporting_sources", []),
                "possible_supporting_sources": item.get("possible_supporting_sources", []),
                "contradicting_sources": item.get("contradicting_sources", []),
                "contradiction_count": item.get("contradiction_count", 0),
                "risk_flag": item.get("risk_flag"),
                "rank_score": item.get("rank_score", 0.0),
            }
        )
    return selected


def prepare_sources(query, search_results, extracted_results):
    """Normalize, clean, rank, deduplicate, and annotate research sources."""
    extracted_by_url = {}
    for item in extracted_results or []:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url", "") or "").strip()
        canonical = _canonical_url(url)
        if canonical:
            extracted_by_url[canonical] = item

    sources = []
    seen_urls = set()

    for result in search_results or []:
        if not isinstance(result, dict):
            continue
        url = str(result.get("url", "") or "").strip()
        canonical = _canonical_url(url)
        if not canonical or canonical in seen_urls:
            continue
        seen_urls.add(canonical)

        extracted = extracted_by_url.get(canonical)
        extracted_content = str(extracted.get("raw_content", "") or "") if extracted else ""
        evidence = extract_evidence(result, extracted)
        quality = source_quality_score(result, extracted_content)

        research_query = str(
            result.get("research_query", query) or query or ""
        ).strip()

        sources.append(
            {
                "id": len(sources) + 1,
                "title": str(result.get("title", "") or "Untitled source").strip(),
                "url": url,
                "score": result.get("score"),
                "quality_score": round(quality, 3),
                "quality_category": domain_category(url),
                "evidence": evidence,
                "llm_evidence": [],
                "query": query,
                "research_query": research_query,
            }
        )

    sources = add_cross_source_agreement(sources)

    for source in sources:
        source["llm_evidence"] = prepare_llm_evidence(
            source.get("evidence", []),
            source_quality=source.get("quality_score", 0.0),
        )

    sources.sort(
        key=lambda source: (
            source.get("quality_score", 0.0),
            source.get("score", 0.0) if isinstance(source.get("score"), (int, float)) else 0.0,
        ),
        reverse=True,
    )

    for index, source in enumerate(sources, start=1):
        source["id"] = index

    # IDs changed after sorting, so agreement references must be rebuilt.
    sources = add_cross_source_agreement(sources)
    for source in sources:
        source["llm_evidence"] = prepare_llm_evidence(
            source.get("evidence", []),
            source_quality=source.get("quality_score", 0.0),
        )

    logger.info("Prepared %s research sources.", len(sources))
    return sources
