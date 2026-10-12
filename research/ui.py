"""Terminal presentation helpers for ResearchPilot Phase 4 UX."""

from __future__ import annotations

import os
import shutil
import sys
import textwrap
from typing import Any, Iterable, Mapping


DEFAULT_TERMINAL_WIDTH = 80
MIN_TERMINAL_WIDTH = 60
MAX_TERMINAL_WIDTH = 120

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"


def terminal_width() -> int:
    """Return a safe terminal width for formatted output."""
    try:
        terminal_size = shutil.get_terminal_size(
            fallback=(DEFAULT_TERMINAL_WIDTH, 24)
        )

        if hasattr(terminal_size, "columns"):
            width = terminal_size.columns
        else:
            width = terminal_size[0]
    except (AttributeError, IndexError, OSError, TypeError):
        width = DEFAULT_TERMINAL_WIDTH

    try:
        width = int(width)
    except (TypeError, ValueError):
        width = DEFAULT_TERMINAL_WIDTH

    return max(
        MIN_TERMINAL_WIDTH,
        min(width, MAX_TERMINAL_WIDTH),
    )


def supports_color() -> bool:
    """Return whether ANSI color output should be enabled."""
    if os.getenv("NO_COLOR") is not None:
        return False

    if os.getenv("TERM", "").lower() == "dumb":
        return False

    if not hasattr(sys.stdout, "isatty"):
        return False

    try:
        return bool(sys.stdout.isatty())
    except OSError:
        return False


def style(
    text: str,
    code: str,
    enabled: bool | None = None,
) -> str:
    """Apply ANSI styling when terminal color is enabled."""
    if enabled is None:
        enabled = supports_color()

    if not enabled:
        return text

    return f"{code}{text}{RESET}"


def wrap_text(
    text: Any,
    width: int | None = None,
    subsequent_indent: str = "",
) -> list[str]:
    """Wrap text into terminal-safe lines."""
    value = str(text or "").strip()

    if not value:
        return []

    if width is None:
        width = terminal_width()

    available_width = max(
        20,
        width - len(subsequent_indent),
    )

    return textwrap.wrap(
        value,
        width=available_width,
        break_long_words=False,
        break_on_hyphens=False,
        replace_whitespace=True,
    )


def print_wrapped(
    text: Any,
    indent: str = "",
    width: int | None = None,
) -> None:
    """Print text wrapped to the available terminal width."""
    lines = wrap_text(
        text,
        width=max(
            20,
            (width or terminal_width()) - len(indent),
        ),
    )

    for line in lines:
        print(f"{indent}{line}")


def divider(
    character: str = "=",
    width: int | None = None,
) -> str:
    """Return a terminal-width-aware divider."""
    if not character:
        character = "="

    return character * (width or terminal_width())


def section_header(
    title: str,
    width: int | None = None,
    color: bool | None = None,
) -> str:
    """Return a consistently formatted section header."""
    heading = str(title or "").strip().upper()
    line = divider("=", width)

    return "\n".join(
        (
            line,
            style(heading, BOLD, color),
            line,
        )
    )


def subsection_header(
    title: str,
    width: int | None = None,
    color: bool | None = None,
) -> str:
    """Return a consistently formatted subsection header."""
    heading = str(title or "").strip().upper()

    return "\n".join(
        (
            "",
            style(heading, BOLD, color),
            divider("-", width),
        )
    )


def status_message(
    message: str,
    level: str = "info",
    color: bool | None = None,
) -> str:
    """Return a formatted status message."""
    normalized = str(level or "info").strip().lower()

    symbols = {
        "info": ("→", CYAN),
        "success": ("✓", GREEN),
        "warning": ("!", YELLOW),
        "error": ("✗", RED),
    }

    encoding = str(getattr(sys.stdout, "encoding", "") or "").lower()
    if encoding not in {"utf-8", "utf8"}:
        symbols["info"] = (">", CYAN)
        symbols["success"] = ("OK", GREEN)
        symbols["error"] = ("X", RED)

    symbol, color_code = symbols.get(
        normalized,
        symbols["info"],
    )

    return style(
        f"{symbol} {str(message or '').strip()}",
        color_code,
        color,
    )


def print_status(
    message: str,
    level: str = "info",
    color: bool | None = None,
) -> None:
    """Print a formatted status message."""
    print(
        status_message(
            message,
            level,
            color,
        )
    )


def print_stage(
    number: int,
    total: int,
    title: str,
    detail: str = "",
    color: bool | None = None,
) -> None:
    """Print a deterministic research pipeline stage."""
    stage = f"[{number}/{total}] {str(title or '').strip()}"

    print(
        style(
            stage,
            BOLD,
            color,
        )
    )

    if detail:
        print_wrapped(
            detail,
            indent="      ",
        )


def format_score(value: Any) -> str:
    """Format a numeric score defensively."""
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "N/A"


def clean_display_url(url: Any) -> str:
    """Remove common Markdown URL wrappers for terminal display."""
    value = str(url or "").strip()

    if not value:
        return ""

    if value.startswith("[") and "](" in value:
        closing_label = value.find("](")
        closing_url = value.rfind(")")

        if closing_label > 0 and closing_url > closing_label:
            candidate = value[
                closing_label + 2 : closing_url
            ].strip()

            if candidate.startswith(
                ("http://", "https://")
            ):
                return candidate

    if value.startswith("[") and value.endswith("]"):
        candidate = value[1:-1].strip()

        if candidate.startswith(
            ("http://", "https://")
        ):
            return candidate

    return value


def print_mode_header(
    mode: str,
    question: str,
    color: bool | None = None,
) -> None:
    """Display the active research mode and question."""
    labels = {
        "quick": "QUICK RESEARCH",
        "deep": "DEEP RESEARCH",
        "compare": "COMPARE TOPICS",
        "followup": "FOLLOW-UP RESEARCH",
    }

    label = labels.get(
        str(mode or "").strip().lower(),
        "RESEARCH",
    )

    print(
        section_header(
            f"RESEARCH MODE: {label}",
            color=color,
        )
    )

    print_wrapped(
        f"Question: {question}",
    )


def print_source(
    source: Mapping[str, Any],
    color: bool | None = None,
) -> None:
    """Display one research source and its evidence."""
    source_id = source.get("id", "?")
    title = source.get(
        "title",
        "Untitled source",
    )

    print(
        f"\n[{source_id}] "
        f"{style(str(title), BOLD, color)}"
    )

    url = clean_display_url(
        source.get("url", "")
    )

    if url:
        print_wrapped(
            f"URL: {url}",
            indent="    ",
        )

    print(
        f"    Tavily score: "
        f"{format_score(source.get('score'))}"
    )

    print(
        f"    Quality signal: "
        f"{format_score(source.get('quality_score', 0.0))}"
    )

    category = source.get(
        "quality_category",
        source.get(
            "source_category",
            "unknown",
        ),
    )

    print(
        f"    Source category: {category}"
    )

    evidence = source.get(
        "evidence",
        [],
    )

    if not isinstance(evidence, list):
        evidence = []

    print(
        f"    Evidence items: {len(evidence)}"
    )

    for item in evidence:
        if not isinstance(item, Mapping):
            print_wrapped(
                item,
                indent="      - ",
            )
            continue

        agreement = item.get(
            "agreement_count",
            1,
        )

        supporting_sources = item.get(
            "supporting_sources",
            [],
        )

        print(
            f"      - Agreement: "
            f"{agreement} source(s) "
            f"{supporting_sources}"
        )

        print_wrapped(
            item.get("text", ""),
            indent="        ",
        )

        print(
            f"        Confidence: "
            f"{format_score(item.get('confidence', 0.0))}"
        )

        print(
            "        Classification: "
            f"{item.get('classification', 'unknown')}"
        )

        risk_flag = item.get("risk_flag")

        if risk_flag:
            print(
                f"        Risk flag: {risk_flag}"
            )

    llm_evidence = source.get(
        "llm_evidence",
        [],
    )

    if not isinstance(llm_evidence, list):
        llm_evidence = []

    print(
        f"    Evidence sent to Gemini: "
        f"{len(llm_evidence)}"
    )

    filtered_count = max(
        0,
        len(evidence) - len(llm_evidence),
    )

    if filtered_count:
        print(
            f"      Filtered from Gemini: "
            f"{filtered_count}"
        )


def print_sources(
    sources: Iterable[Mapping[str, Any]],
    color: bool | None = None,
) -> None:
    """Display all prepared research sources."""
    source_list = list(sources or [])

    print(
        section_header(
            f"RESEARCH SOURCES ({len(source_list)})",
            color=color,
        )
    )

    if not source_list:
        print("No research sources are available.")
        return

    for source in source_list:
        if isinstance(source, Mapping):
            print_source(
                source,
                color=color,
            )


def print_finding(
    index: int,
    finding: Mapping[str, Any],
    color: bool | None = None,
) -> None:
    """Display one structured report finding."""
    text = str(
        finding.get("text", "")
        or ""
    ).strip()

    print(
        f"\n{index}. "
        f"{style(text, BOLD, color)}"
    )

    confidence = format_score(
        finding.get("confidence", 0.0)
    )

    confidence_label = str(
        finding.get(
            "confidence_label",
            "",
        )
        or ""
    )

    confidence_suffix = (
        f" ({confidence_label})"
        if confidence_label
        else ""
    )

    print(
        f"   Confidence: "
        f"{confidence}{confidence_suffix}"
    )

    citation_ids = finding.get(
        "citation_ids",
        [],
    )

    print(
        f"   Citations: {citation_ids}"
    )

    print(
        f"   Supported: "
        f"{finding.get('supported', False)}"
    )


def print_report(
    report: Mapping[str, Any],
    color: bool | None = None,
) -> None:
    """Display a structured Phase 3 research report."""
    if not isinstance(report, Mapping):
        print(
            "\nNo structured report is available."
        )
        return

    print(
        section_header(
            "RESEARCH REPORT",
            color=color,
        )
    )

    executive_answer = str(
        report.get(
            "executive_answer",
            "",
        )
        or ""
    ).strip()

    if executive_answer:
        print(
            subsection_header(
                "Executive Answer",
                color=color,
            )
        )
        print_wrapped(executive_answer)

    findings = report.get(
        "key_findings",
        [],
    )

    print(
        subsection_header(
            "Key Findings",
            color=color,
        )
    )

    if isinstance(findings, list) and findings:
        for index, finding in enumerate(
            findings,
            start=1,
        ):
            if isinstance(finding, Mapping):
                print_finding(
                    index,
                    finding,
                    color=color,
                )
            else:
                print(
                    f"{index}. {finding}"
                )
    else:
        print(
            "No structured findings were produced."
        )

    limitations = report.get(
        "limitations",
        [],
    )

    print(
        subsection_header(
            "Limitations / Uncertainty",
            color=color,
        )
    )

    if isinstance(limitations, list) and limitations:
        for limitation in limitations:
            print_wrapped(
                limitation,
                indent="- ",
            )
    else:
        print(
            "No explicit limitations were recorded."
        )

    citation_integrity = report.get(
        "citation_integrity",
        {},
    )

    print(
        subsection_header(
            "Citation Integrity",
            color=color,
        )
    )

    if isinstance(
        citation_integrity,
        Mapping,
    ):
        print(
            f"Valid: "
            f"{citation_integrity.get('valid', False)}"
        )
        print(
            "Referenced sources: "
            f"{citation_integrity.get('referenced_source_ids', [])}"
        )
        print(
            "Uncited claims: "
            f"{citation_integrity.get('uncited_claim_count', 0)}"
        )
    else:
        print(
            "No citation-integrity information is available."
        )

    sources = report.get(
        "sources",
        [],
    )

    print(
        subsection_header(
            "Source Metadata",
            color=color,
        )
    )

    if isinstance(sources, list) and sources:
        for source in sources:
            if not isinstance(source, Mapping):
                continue

            source_id = source.get(
                "source_id",
                source.get("id", "?"),
            )

            title = source.get(
                "title",
                "Untitled source",
            )

            print(
                f"\n[{source_id}] "
                f"{style(str(title), BOLD, color)}"
            )

            domain = source.get(
                "domain",
                "",
            )

            if domain:
                print(
                    f"    Domain: {domain}"
                )

            category = source.get(
                "source_category",
                source.get(
                    "quality_category",
                    "unknown",
                ),
            )

            print(
                f"    Category: {category}"
            )

            print(
                f"    Quality: "
                f"{format_score(source.get('quality_score', 0.0))}"
            )

            research_step = source.get(
                "research_step",
                "",
            )

            if research_step:
                print(
                    f"    Research step: "
                    f"{research_step}"
                )

            url = clean_display_url(
                source.get("url", "")
            )

            if url:
                print_wrapped(
                    f"URL: {url}",
                    indent="    ",
                )

    else:
        print(
            "No source metadata is available."
        )

    print(
        divider("=")
    )


def print_completion(
    color: bool | None = None,
) -> None:
    """Display a consistent research completion message."""
    print(
        "\n"
        + style(
            divider("="),
            BOLD,
            color,
        )
    )
    print(
        style(
            "RESEARCH COMPLETE",
            BOLD,
            color,
        )
    )
    print(
        style(
            divider("="),
            BOLD,
            color,
        )
    )


def print_history(
    sessions: Iterable[Mapping[str, Any]] = (),
    reports: Iterable[Mapping[str, Any]] = (),
    color: bool | None = None,
) -> None:
    """Display a compact summary of persisted research data."""
    print(section_header("RESEARCH HISTORY", color=color))

    session_items = [item for item in sessions if isinstance(item, Mapping)]
    report_items = [item for item in reports if isinstance(item, Mapping)]

    print(subsection_header("SESSIONS", color=color))
    if not session_items:
        print("No saved sessions are available.")
    else:
        for index, session in enumerate(session_items, start=1):
            query = str(
                session.get(
                    "query",
                    session.get("original_query", "Untitled research"),
                )
            ).strip()
            mode = str(session.get("mode", "unknown")).strip()
            updated = str(session.get("updated_at", "unknown")).strip()
            print(f"{index}. {query or 'Untitled research'}")
            print(f"   Mode: {mode or 'unknown'} | Updated: {updated}")

    print(subsection_header("REPORTS", color=color))
    if not report_items:
        print("No saved reports are available.")
    else:
        for index, report in enumerate(report_items, start=1):
            answer = str(report.get("executive_answer", "")).strip()
            findings = report.get("key_findings", [])
            count = len(findings) if isinstance(findings, list) else 0
            print(f"{index}. Findings: {count}")
            if answer:
                print_wrapped(answer, indent="   ")


def print_configuration(settings: Mapping[str, Any], color: bool | None = None) -> None:
    """Display safe runtime configuration without exposing credentials."""
    print(section_header("RESEARCHPILOT CONFIGURATION", color=color))
    for key, value in settings.items():
        print(f"{key}: {value}")