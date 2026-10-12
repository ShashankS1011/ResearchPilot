"""Phase 3 research report generation and citation integrity."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


_CITATION_PATTERN = re.compile(
    r"\[(\d+(?:\s*,\s*\d+)*)\]"
)

_SENTENCE_PATTERN = re.compile(
    r"(?<=[.!?])\s+"
    r"(?!\[\d+(?:\s*,\s*\d+)*\])"
)

_BULLET_PATTERN = re.compile(
    r"^(?:[•*-]|\d+[.)])\s+"
)

_REPORT_HEADINGS = {
    "research summary",
    "definition",
    "overview",
    "background",
    "how it works",
    "key points",
    "key findings",
    "findings",
    "analysis",
    "conclusion",
    "limitations",
    "limitations / uncertainty",
    "sources",
    "source",
    "references",
}


@dataclass(frozen=True)
class Claim:
    """A report claim and its citation support."""

    claim_id: str
    text: str
    citation_ids: tuple[int, ...]
    confidence: float
    confidence_label: str
    supported: bool


@dataclass(frozen=True)
class SourceMetadata:
    """Structured metadata for a research source."""

    source_id: int
    title: str
    url: str
    domain: str
    category: str
    quality_score: float
    tavily_score: float
    research_step: str
    research_query: str
    evidence_count: int


@dataclass(frozen=True)
class ResearchReport:
    """Structured Phase 3 research report."""

    schema_version: str
    query: str
    mode: str
    generated_at: str
    executive_answer: str
    key_findings: tuple[Claim, ...]
    limitations: tuple[str, ...]
    citation_integrity: dict[str, Any]
    sources: tuple[SourceMetadata, ...]

    def to_dict(self) -> dict[str, Any]:
        """Convert the report into JSON-serializable data."""
        return {
            "schema_version": self.schema_version,
            "query": self.query,
            "mode": self.mode,
            "generated_at": self.generated_at,
            "executive_answer": self.executive_answer,
            "key_findings": [
                {
                    "claim_id": claim.claim_id,
                    "text": claim.text,
                    "citation_ids": list(
                        claim.citation_ids
                    ),
                    "confidence": claim.confidence,
                    "confidence_label": (
                        claim.confidence_label
                    ),
                    "supported": claim.supported,
                }
                for claim in self.key_findings
            ],
            "limitations": list(
                self.limitations
            ),
            "citation_integrity": (
                self.citation_integrity
            ),
            "sources": [
                {
                    "source_id": source.source_id,
                    "title": source.title,
                    "url": source.url,
                    "domain": source.domain,
                    "category": source.category,
                    "source_category": source.category,
                    "quality_score": (
                        source.quality_score
                    ),
                    "tavily_score": (
                        source.tavily_score
                    ),
                    "research_step": (
                        source.research_step
                    ),
                    "research_query": (
                        source.research_query
                    ),
                    "evidence_count": (
                        source.evidence_count
                    ),
                }
                for source in self.sources
            ],
        }


def _utc_now() -> str:
    """Return the current UTC timestamp."""
    return datetime.now(
        timezone.utc
    ).isoformat()


def _domain(url: str) -> str:
    """Extract a normalized domain from a URL."""
    value = str(
        url or ""
    ).strip()

    if not value:
        return ""

    value = re.sub(
        r"^https?://",
        "",
        value,
        flags=re.IGNORECASE,
    )

    value = value.split(
        "/",
        1,
    )[0]

    value = value.split(
        "?",
        1,
    )[0]

    value = value.split(
        "#",
        1,
    )[0]

    if value.startswith("www."):
        value = value[4:]

    return value.lower()


def _citation_ids(
    text: str,
) -> tuple[int, ...]:
    """Extract unique citation IDs in source order."""
    if not text:
        return ()

    citation_ids: list[int] = []

    for match in _CITATION_PATTERN.finditer(
        text
    ):
        values = match.group(1).split(",")

        for value in values:
            source_id = int(
                value.strip()
            )

            if source_id not in citation_ids:
                citation_ids.append(
                    source_id
                )

    return tuple(citation_ids)


def _remove_citations(
    text: str,
) -> str:
    """Remove citation markers from claim text."""
    cleaned = _CITATION_PATTERN.sub(
        "",
        text,
    )

    cleaned = re.sub(
        r"[ \t]+",
        " ",
        cleaned,
    )

    return cleaned.strip()


def _confidence_label(
    confidence: float,
) -> str:
    """Convert a numeric confidence score into a label."""
    if confidence >= 0.80:
        return "high"

    if confidence >= 0.60:
        return "medium"

    return "low"


def _source_support_confidence(
    source: dict[str, Any],
) -> float:
    """Estimate confidence contributed by one source."""
    evidence = source.get(
        "evidence",
        [],
    )

    if not isinstance(
        evidence,
        list,
    ):
        evidence = []

    confidences: list[float] = []

    for item in evidence:
        if not isinstance(
            item,
            dict,
        ):
            continue

        value = item.get(
            "confidence"
        )

        try:
            confidence = float(
                value
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        confidences.append(
            max(
                0.0,
                min(
                    1.0,
                    confidence,
                ),
            )
        )

    if confidences:
        return max(
            confidences
        )

    value = source.get(
        "quality_score",
        source.get(
            "source_quality",
            0.0,
        ),
    )

    try:
        return max(
            0.0,
            min(
                1.0,
                float(value),
            ),
        )
    except (
        TypeError,
        ValueError,
    ):
        return 0.0


def _claim_confidence(
    citation_ids: tuple[int, ...],
    source_map: dict[int, dict[str, Any]],
) -> float:
    """Calculate confidence for a cited claim."""
    if not citation_ids:
        return 0.0

    source_scores: list[float] = []

    for source_id in citation_ids:
        source = source_map.get(
            source_id
        )

        if not source:
            continue

        evidence_confidence = (
            _source_support_confidence(
                source
            )
        )

        evidence_items = source.get(
            "evidence",
            [],
        )

        if not isinstance(
            evidence_items,
            list,
        ):
            evidence_items = []

        best_evidence: dict[str, Any] | None = None

        for evidence in evidence_items:
            if not isinstance(
                evidence,
                dict,
            ):
                continue

            if best_evidence is None:
                best_evidence = evidence
                continue

            current_rank = evidence.get(
                "rank_score",
                evidence.get(
                    "evidence_rank_score",
                    0.0,
                ),
            )

            best_rank = best_evidence.get(
                "rank_score",
                best_evidence.get(
                    "evidence_rank_score",
                    0.0,
                ),
            )

            try:
                current_rank = float(
                    current_rank
                )
            except (
                TypeError,
                ValueError,
            ):
                current_rank = 0.0

            try:
                best_rank = float(
                    best_rank
                )
            except (
                TypeError,
                ValueError,
            ):
                best_rank = 0.0

            if current_rank > best_rank:
                best_evidence = evidence

        agreement_count = 1
        rank_score = 0.0
        agreement_score = 0.0

        if best_evidence:
            try:
                agreement_count = max(
                    1,
                    int(
                        best_evidence.get(
                            "agreement_count",
                            1,
                        )
                    ),
                )
            except (
                TypeError,
                ValueError,
            ):
                agreement_count = 1

            try:
                rank_score = float(
                    best_evidence.get(
                        "rank_score",
                        best_evidence.get(
                            "evidence_rank_score",
                            0.0,
                        ),
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                rank_score = 0.0

            try:
                agreement_score = float(
                    best_evidence.get(
                        "agreement_score",
                        0.0,
                    )
                )
            except (
                TypeError,
                ValueError,
            ):
                agreement_score = 0.0

        agreement_bonus = min(
            0.10,
            max(
                0,
                agreement_count - 1,
            )
            * 0.05,
        )

        rank_bonus = min(
            0.05,
            max(
                0.0,
                rank_score * 0.05,
            ),
        )

        agreement_score_bonus = min(
            0.05,
            max(
                0.0,
                agreement_score * 0.05,
            ),
        )

        quality = source.get(
            "quality_score",
            source.get(
                "source_quality",
                0.0,
            ),
        )

        try:
            quality = float(
                quality
            )
        except (
            TypeError,
            ValueError,
        ):
            quality = 0.0

        quality_bonus = min(
            0.05,
            max(
                0.0,
                quality * 0.05,
            ),
        )

        score = min(
            1.0,
            evidence_confidence
            + agreement_bonus
            + rank_bonus
            + agreement_score_bonus
            + quality_bonus,
        )

        source_scores.append(
            score
        )

    if not source_scores:
        return 0.0

    confidence = max(
        source_scores
    )

    if len(source_scores) >= 2:
        confidence = min(
            1.0,
            confidence + 0.05,
        )

    return round(
        confidence,
        2,
    )


def _is_report_heading(
    line: str,
) -> bool:
    """Return True when a line is a structural report heading."""
    normalized = re.sub(
        r"\s+",
        " ",
        str(line or "").strip(),
    ).strip(
        " :"
    ).lower()

    if not normalized:
        return False

    if normalized in _REPORT_HEADINGS:
        return True

    if (
        len(normalized) <= 50
        and normalized.upper() == normalized
        and re.search(
            r"[A-Z]",
            normalized,
        )
    ):
        return True

    return False


def _split_claim_segments(
    answer: str,
) -> list[str]:
    """Split answer text into logical claim segments.

    Hard-wrapped lines belonging to the same sentence are joined.
    Structured report headings and source metadata are ignored.
    Bullet/list items remain separate claims.
    """
    text = str(
        answer or ""
    ).strip()

    if not text:
        return []

    text = text.replace(
        "\r\n",
        "\n",
    ).replace(
        "\r",
        "\n",
    )

    lines = text.split(
        "\n"
    )

    paragraphs: list[str] = []
    current_lines: list[str] = []
    in_sources_section = False

    def flush_current() -> None:
        if not current_lines:
            return

        paragraph = " ".join(
            part.strip()
            for part in current_lines
            if part.strip()
        ).strip()

        if paragraph:
            paragraphs.append(
                paragraph
            )

        current_lines.clear()

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            flush_current()
            continue

        if _is_report_heading(
            line
        ):
            flush_current()

            heading = line.rstrip(
                " :"
            ).strip().lower()

            if heading in {
                "sources",
                "source",
                "references",
            }:
                in_sources_section = True

            continue

        if in_sources_section:
            continue

        bullet_match = _BULLET_PATTERN.match(
            line
        )

        if bullet_match:
            flush_current()

            bullet_text = line[
                bullet_match.end():
            ].strip()

            if bullet_text:
                paragraphs.append(
                    bullet_text
                )

            continue

        current_lines.append(
            line
        )

    flush_current()

    segments: list[str] = []

    for paragraph in paragraphs:
        sentence_parts = _SENTENCE_PATTERN.split(
            paragraph
        )

        for part in sentence_parts:
            cleaned = part.strip(
                " \t•-"
            )

            if cleaned:
                segments.append(
                    cleaned
                )

    return segments


def _extract_claims(
    answer: str,
    source_map: dict[int, dict[str, Any]],
) -> tuple[
    tuple[dict[str, Any], ...],
    set[int],
    list[str],
]:
    """Extract claims and their citation support."""
    claims: list[dict[str, Any]] = []
    referenced_ids: set[int] = set()
    uncited_claims: list[str] = []

    raw_parts = _split_claim_segments(
        answer
    )

    for raw_part in raw_parts:
        text = raw_part.strip(
            " \t•-"
        )

        if not text:
            continue

        citation_ids = _citation_ids(
            text
        )

        claim_text = _remove_citations(
            text
        ).strip(
            " \t•-"
        )

        if not claim_text:
            continue

        if (
            len(claim_text) < 25
            and not re.search(
                r"\b(?:is|are|was|were|can|may|has|have)\b",
                claim_text,
                re.IGNORECASE,
            )
        ):
            continue

        if citation_ids:
            referenced_ids.update(
                citation_ids
            )
        else:
            uncited_claims.append(
                claim_text
            )

        confidence = _claim_confidence(
            citation_ids,
            source_map,
        )

        supported = (
            bool(citation_ids)
            and all(
                source_id in source_map
                for source_id in citation_ids
            )
        )

        claims.append(
            {
                "claim_id": (
                    f"claim-{len(claims) + 1}"
                ),
                "text": claim_text,
                "citation_ids": list(
                    citation_ids
                ),
                "confidence": confidence,
                "confidence_label": (
                    _confidence_label(
                        confidence
                    )
                ),
                "supported": supported,
            }
        )

    return (
        tuple(claims),
        referenced_ids,
        uncited_claims,
    )


def validate_citation_ids(
    answer: str,
    sources: list[dict[str, Any]],
) -> tuple[int, ...]:
    """Validate that every citation ID exists in the sources."""
    valid_ids = {
        int(source["id"])
        for source in sources
        if (
            isinstance(source, dict)
            and str(
                source.get(
                    "id",
                    "",
                )
            ).isdigit()
        )
    }

    referenced = _citation_ids(
        answer
    )

    invalid = sorted(
        source_id
        for source_id in referenced
        if source_id not in valid_ids
    )

    if invalid:
        raise ValueError(
            "Answer contains unsupported citation IDs: "
            + ", ".join(
                str(value)
                for value in invalid
            )
        )

    return referenced


def _source_category(
    source: dict[str, Any],
) -> str:
    """Resolve the category from the ResearchPilot source schema."""
    direct_keys = (
        "quality_category",
        "source_category",
        "category",
        "source_type",
        "type",
    )

    for key in direct_keys:
        value = source.get(
            key
        )

        if value is None:
            continue

        value = str(
            value
        ).strip()

        if value:
            return value

    nested_source = source.get(
        "source"
    )

    if isinstance(
        nested_source,
        dict,
    ):
        for key in (
            "quality_category",
            "source_category",
            "category",
            "source_type",
            "type",
        ):
            value = nested_source.get(
                key
            )

            if value is None:
                continue

            value = str(
                value
            ).strip()

            if value:
                return value

    return ""


def _source_metadata(
    source: dict[str, Any],
) -> dict[str, Any]:
    """Build normalized source metadata."""
    source_id = source.get(
        "id",
        0,
    )

    try:
        source_id = int(
            source_id
        )
    except (
        TypeError,
        ValueError,
    ):
        source_id = 0

    title = str(
        source.get(
            "title",
            source.get(
                "name",
                "",
            ),
        )
        or ""
    ).strip()

    url = str(
        source.get(
            "url",
            "",
        )
        or ""
    ).strip()

    category = _source_category(
        source
    )

    quality_score = source.get(
        "quality_score",
        source.get(
            "source_quality",
            0.0,
        ),
    )

    try:
        quality_score = round(
            float(
                quality_score
            ),
            2,
        )
    except (
        TypeError,
        ValueError,
    ):
        quality_score = 0.0

    tavily_score = source.get(
        "score",
        source.get(
            "tavily_score",
            0.0,
        ),
    )

    try:
        tavily_score = float(
            tavily_score
        )
    except (
        TypeError,
        ValueError,
    ):
        tavily_score = 0.0

    research_step = str(
        source.get(
            "research_step",
            "",
        )
        or ""
    ).strip()

    research_query = str(
        source.get(
            "research_query",
            "",
        )
        or ""
    ).strip()

    evidence = source.get(
        "evidence",
        [],
    )

    if not isinstance(
        evidence,
        list,
    ):
        evidence = []

    return {
        "source_id": source_id,
        "title": title,
        "url": url,
        "domain": _domain(url),
        "category": category,
        "source_category": category,
        "quality_score": quality_score,
        "tavily_score": tavily_score,
        "research_step": research_step,
        "research_query": research_query,
        "evidence_count": len(
            evidence
        ),
    }


def build_report(
    query: str,
    answer: str,
    sources: list[dict[str, Any]],
    mode: str = "quick",
) -> dict[str, Any]:
    """Build a structured research report.

    Raises:
        ValueError: If the answer contains unsupported citation IDs.
    """
    query = str(
        query or ""
    ).strip()

    answer = str(
        answer or ""
    ).strip()

    normalized_sources = [
        source
        for source in sources
        if isinstance(
            source,
            dict,
        )
    ]

    referenced_ids = validate_citation_ids(
        answer,
        normalized_sources,
    )

    source_map = {
        int(source["id"]): source
        for source in normalized_sources
        if str(
            source.get(
                "id",
                "",
            )
        ).isdigit()
    }

    (
        claims,
        claim_references,
        uncited_claims,
    ) = _extract_claims(
        answer,
        source_map,
    )

    referenced_ids = tuple(
        sorted(
            set(referenced_ids)
            | claim_references
        )
    )

    limitations: list[str] = []

    if uncited_claims:
        limitations.append(
            "Some answer statements were not "
            "associated with an explicit source citation."
        )

    if not referenced_ids:
        limitations.append(
            "The answer did not contain explicit "
            "source citations."
        )

    if not claims:
        limitations.append(
            "No substantive claims could be extracted "
            "from the answer."
        )

    if not normalized_sources:
        limitations.append(
            "No source metadata was available."
        )

    citation_integrity = {
        "valid": True,
        "referenced_source_ids": list(
            referenced_ids
        ),
        "uncited_claim_count": len(
            uncited_claims
        ),
        "unsupported_claim_count": sum(
            1
            for claim in claims
            if not claim["supported"]
        ),
        "claim_count": len(
            claims
        ),
    }

    metadata = tuple(
        _source_metadata(
            source
        )
        for source in normalized_sources
    )

    report = ResearchReport(
        schema_version="3.0",
        query=query,
        mode=str(
            mode or "quick"
        ).strip().lower(),
        generated_at=_utc_now(),
        executive_answer=answer,
        key_findings=tuple(
            Claim(
                claim_id=claim[
                    "claim_id"
                ],
                text=claim[
                    "text"
                ],
                citation_ids=tuple(
                    claim[
                        "citation_ids"
                    ]
                ),
                confidence=claim[
                    "confidence"
                ],
                confidence_label=claim[
                    "confidence_label"
                ],
                supported=claim[
                    "supported"
                ],
            )
            for claim in claims
        ),
        limitations=tuple(
            limitations
        ),
        citation_integrity=(
            citation_integrity
        ),
        sources=tuple(
            SourceMetadata(
                source_id=source[
                    "source_id"
                ],
                title=source[
                    "title"
                ],
                url=source[
                    "url"
                ],
                domain=source[
                    "domain"
                ],
                category=source[
                    "category"
                ],
                quality_score=source[
                    "quality_score"
                ],
                tavily_score=source[
                    "tavily_score"
                ],
                research_step=source[
                    "research_step"
                ],
                research_query=source[
                    "research_query"
                ],
                evidence_count=source[
                    "evidence_count"
                ],
            )
            for source in metadata
        ),
    )

    return report.to_dict()


def build_evidence_fallback_report(
    query: str,
    sources: list[dict[str, Any]],
    mode: str = "quick",
) -> dict[str, Any]:
    """Build a citation-safe report directly from top evidence."""
    normalized_sources = [
        source
        for source in sources
        if isinstance(
            source,
            dict,
        )
    ]

    evidence_items: list[
        tuple[
            float,
            dict[str, Any],
            dict[str, Any],
        ]
    ] = []

    for source in normalized_sources:
        evidence = source.get(
            "evidence",
            [],
        )

        if not isinstance(
            evidence,
            list,
        ):
            continue

        for item in evidence:
            if not isinstance(
                item,
                dict,
            ):
                continue

            text = str(
                item.get(
                    "text",
                    item.get(
                        "content",
                        "",
                    ),
                )
                or ""
            ).strip()

            if not text:
                continue

            rank_score = item.get(
                "rank_score",
                item.get(
                    "evidence_rank_score",
                    0.0,
                ),
            )

            try:
                rank_score = float(
                    rank_score
                )
            except (
                TypeError,
                ValueError,
            ):
                rank_score = 0.0

            evidence_items.append(
                (
                    rank_score,
                    source,
                    item,
                )
            )

    evidence_items.sort(
        key=lambda value: value[0],
        reverse=True,
    )

    selected: list[
        tuple[
            dict[str, Any],
            dict[str, Any],
        ]
    ] = []

    seen_text: set[str] = set()

    for (
        _,
        source,
        evidence,
    ) in evidence_items:
        text = str(
            evidence.get(
                "text",
                evidence.get(
                    "content",
                    "",
                ),
            )
            or ""
        ).strip()

        normalized_text = re.sub(
            r"\s+",
            " ",
            text,
        ).lower()

        if normalized_text in seen_text:
            continue

        seen_text.add(
            normalized_text
        )

        selected.append(
            (
                source,
                evidence,
            )
        )

        if len(selected) >= 2:
            break

    if not selected:
        answer = (
            "No sufficiently strong evidence was "
            "available to construct a fallback answer."
        )
    else:
        sections: list[str] = [
            "RESEARCH SUMMARY",
            "",
        ]

        for index, (
            source,
            evidence,
        ) in enumerate(
            selected,
            start=1,
        ):
            text = str(
                evidence.get(
                    "text",
                    evidence.get(
                        "content",
                        "",
                    ),
                )
                or ""
            ).strip()

            source_id = source.get(
                "id",
                index,
            )

            try:
                source_id = int(
                    source_id
                )
            except (
                TypeError,
                ValueError,
            ):
                source_id = index

            heading = (
                "Definition"
                if index == 1
                else "How It Works"
            )

            sections.extend(
                [
                    heading,
                    "",
                    f"{text} [{source_id}]",
                    "",
                ]
            )

        sections.extend(
            [
                "Sources",
                "",
            ]
        )

        for source, _ in selected:
            source_id = source.get(
                "id",
                "",
            )

            title = str(
                source.get(
                    "title",
                    "",
                )
                or ""
            ).strip()

            url = str(
                source.get(
                    "url",
                    "",
                )
                or ""
            ).strip()

            sections.append(
                f"[{source_id}] {title}"
            )

            if url:
                sections.append(
                    url
                )

            sections.append("")

        answer = "\n".join(
            sections
        ).strip()

    return build_report(
        query=query,
        answer=answer,
        sources=normalized_sources,
        mode=mode,
    )