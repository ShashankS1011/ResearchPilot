"""ResearchPilot command-line application."""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any

from research.extract import extract_content
from research.llm import generate_answer
from research.logging_config import logger
from research.processor import prepare_sources
from research.report import (
    build_evidence_fallback_report,
    build_report,
)
from research.search import search_web
from research.storage import (
    export_report,
    load_session,
    load_report_history,
    load_session_history,
    save_report,
    save_research,
    save_session,
)
from research.config import SETTINGS
from research.ui import (
    print_completion,
    print_history,
    print_configuration,
    print_mode_header,
    print_report,
    print_sources,
    print_stage,
    print_status,
    section_header,
    wrap_text,
)


# ============================================================
# ERROR / LOGGING HELPERS
# ============================================================


def _sanitize_error_message(error: Exception) -> str:
    """
    Remove known API credentials from an exception message.

    External SDKs normally do not include credentials in exception
    messages, but sanitizing them here prevents accidental leakage if
    an SDK or downstream exception ever exposes request details.
    """

    message = str(error)

    secrets = (
        os.getenv("TAVILY_API_KEY", ""),
        os.getenv("GEMINI_API_KEY", ""),
    )

    for secret in secrets:
        if secret:
            message = message.replace(
                secret,
                "[REDACTED]",
            )

    return message


# ============================================================
# DISPLAY CLEANING
# ============================================================


def clean_url_for_display(url: Any) -> str:
    """Return a plain URL even if wrapped in Markdown."""

    if not url:
        return ""

    value = str(url).strip()

    if value.startswith("[") and "](" in value:
        closing_label = value.find("](")
        closing_url = value.rfind(")")

        if (
            closing_label > 0
            and closing_url > closing_label
        ):
            candidate = value[
                closing_label + 2 : closing_url
            ].strip()

            if candidate.startswith(
                ("http://", "https://"),
            ):
                return candidate

    if value.startswith("[") and value.endswith("]"):
        candidate = value[1:-1].strip()

        if candidate.startswith(
            ("http://", "https://"),
        ):
            return candidate

    return value


def clean_answer_display(answer: Any) -> str:
    """
    Remove accidental Markdown URL wrappers and escaped
    punctuation from the final terminal display.
    """

    text = str(answer or "")

    text = text.replace(
        "\\=",
        "=",
    )

    text = text.replace(
        "\\-",
        "-",
    )

    text = re.sub(
        r"\[([^\]]+)\]\((https?://[^)]+)\)",
        r"\2",
        text,
    )

    return text.strip()


# ============================================================
# GRACEFUL FAILURE HELPERS
# ============================================================


def _extract_sources_gracefully(
    search_results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Extract source content while isolating individual failures.

    Normal behavior uses one batch Tavily extraction request because
    it is efficient and preserves the existing extraction/cache path.

    If the batch request fails, each URL is attempted independently.
    This allows successful sources to continue through the research
    pipeline even when one URL cannot be extracted.

    Failed URLs are intentionally omitted from the extracted results.
    The processor can still use the Tavily search snippets as evidence.
    """

    urls = [
        str(result.get("url", "")).strip()
        for result in search_results
        if isinstance(result, dict)
        and result.get("url")
    ]

    if not urls:
        logger.warning(
            "No source URLs were available for extraction."
        )
        return []

    logger.info(
        "Starting batch extraction for %s source URLs.",
        len(urls),
    )

    try:
        extracted_response = extract_content(
            urls
        )

        if not isinstance(
            extracted_response,
            dict,
        ):
            logger.warning(
                "Batch extraction returned an unexpected "
                "response type; falling back to per-URL extraction."
            )
            extracted_response = {}

        extracted_results = extracted_response.get(
            "results",
            [],
        )

        if not isinstance(
            extracted_results,
            list,
        ):
            logger.warning(
                "Batch extraction returned an invalid results "
                "collection; falling back to per-URL extraction."
            )
            extracted_results = []

        logger.info(
            "Batch extraction succeeded: %s/%s sources.",
            len(extracted_results),
            len(urls),
        )

        return [
            item
            for item in extracted_results
            if isinstance(item, dict)
        ]

    except Exception as error:
        safe_error = _sanitize_error_message(
            error
        )

        logger.warning(
            "Batch Tavily extraction failed: %s",
            safe_error,
        )

        logger.info(
            "Entering degraded extraction mode: "
            "retrying %s source URLs independently.",
            len(urls),
        )

    recovered_results: list[dict[str, Any]] = []

    for index, url in enumerate(
        urls,
        start=1,
    ):
        try:
            response = extract_content(
                [url]
            )

            if not isinstance(
                response,
                dict,
            ):
                logger.warning(
                    "Individual extraction returned an invalid "
                    "response for source %s/%s.",
                    index,
                    len(urls),
                )
                continue

            results = response.get(
                "results",
                [],
            )

            if not isinstance(
                results,
                list,
            ):
                logger.warning(
                    "Individual extraction returned an invalid "
                    "results collection for source %s/%s.",
                    index,
                    len(urls),
                )
                continue

            valid_results = [
                item
                for item in results
                if isinstance(item, dict)
            ]

            recovered_results.extend(
                valid_results
            )

            logger.info(
                "Individual extraction succeeded for source "
                "%s/%s.",
                index,
                len(urls),
            )

        except Exception as error:
            safe_error = _sanitize_error_message(
                error
            )

            logger.warning(
                "Individual extraction failed for source "
                "%s/%s (%s): %s",
                index,
                len(urls),
                url,
                safe_error,
            )

            continue

    logger.info(
        "Degraded extraction completed: %s/%s sources recovered.",
        len(recovered_results),
        len(urls),
    )

    return recovered_results


def _search_gracefully(
    query: str,
    max_results: int,
) -> list[dict[str, Any]] | None:
    """
    Execute a web search without allowing a provider failure to
    crash the CLI workflow.

    Returns:
        A list of search results on success.
        None when the search provider failed.
    """

    try:
        results = search_web(
            query,
            max_results=max_results,
        )

    except Exception as error:
        safe_error = _sanitize_error_message(
            error
        )

        logger.error(
            "Research search failed: query=%r error=%s",
            query,
            safe_error,
        )

        print_status(
            "Web search failed. Research could not continue.",
            "error",
        )

        print_status(
            f"Details: {safe_error}",
            "warning",
        )

        return None

    if not isinstance(
        results,
        list,
    ):
        logger.error(
            "Search returned an unexpected result type: %s",
            type(results).__name__,
        )

        print_status(
            "Web search returned an invalid response.",
            "error",
        )

        return None

    return [
        result
        for result in results
        if isinstance(result, dict)
    ]


# ============================================================
# SEARCH
# ============================================================


def get_search_limit(mode: str) -> int:
    """Return the source limit for a research mode."""

    if mode == "deep":
        return 8

    if mode == "compare":
        return 6

    return 3


def perform_research(
    query: str,
    mode: str = "quick",
) -> list[dict[str, Any]] | None:
    """
    Run the complete search -> extraction -> processing pipeline.

    Search failures stop the current run cleanly.

    Extraction failures use degraded per-source recovery so that
    successfully extracted sources remain available. Sources for
    which extraction fails can still contribute their Tavily search
    snippets to evidence processing.
    """

    print_mode_header(
        mode,
        query,
    )

    print_stage(
        1,
        3,
        "Searching the web",
        "Finding and ranking sources for the research question.",
    )

    search_results = _search_gracefully(
        query,
        max_results=get_search_limit(mode),
    )

    if search_results is None:
        return None

    print_status(
        f"Found {len(search_results)} sources.",
        "success" if search_results else "warning",
    )

    if not search_results:
        print_status(
            "No web sources were returned. Research stopped.",
            "warning",
        )
        return None

    print_stage(
        2,
        3,
        "Extracting source content",
        (
            f"Collecting content from "
            f"{len(search_results)} source URLs."
        ),
    )

    extracted_results = _extract_sources_gracefully(
        search_results
    )

    print_status(
        (
            f"Extracted content from "
            f"{len(extracted_results)} sources."
        ),
        "success" if extracted_results else "warning",
    )

    if not extracted_results:
        print_status(
            "No source pages could be extracted. "
            "Continuing with available search-result evidence.",
            "warning",
        )

    print_stage(
        3,
        3,
        "Processing evidence",
        (
            "Cleaning, ranking, classifying, and "
            "preparing evidence for synthesis."
        ),
    )

    sources = prepare_sources(
        query,
        search_results,
        extracted_results,
    )

    print_status(
        f"Prepared {len(sources)} research sources.",
        "success" if sources else "warning",
    )

    if not sources:
        print_status(
            "No usable research sources remained after processing.",
            "warning",
        )
        return None

    print_sources(
        sources
    )

    return sources


# ============================================================
# PHASE 3 REPORT GENERATION
# ============================================================


def build_and_save_report(
    query: str,
    answer: str,
    sources: list[dict[str, Any]],
    mode: str = "quick",
    force_fallback: bool = False,
) -> dict[str, Any]:
    """
    Build a structured Phase 3 report and persist it.

    When force_fallback is True, the Gemini answer is intentionally
    bypassed and the evidence-backed fallback report is used
    directly. This is required when Gemini synthesis itself failed.

    Otherwise Gemini-generated answers are validated through the
    report layer. If the answer contains unsupported citation IDs or
    otherwise cannot produce a safe structured report, an
    evidence-backed fallback report is generated instead.
    """

    cleaned_answer = clean_answer_display(
        answer
    )

    if force_fallback:
        logger.warning(
            "Using evidence-based fallback report because "
            "answer synthesis failed."
        )

        report = build_evidence_fallback_report(
            query,
            sources,
            mode=mode,
        )

        if not isinstance(
            report,
            dict,
        ):
            raise TypeError(
                "Evidence fallback report must be a dictionary."
            )

        save_report(
            report
        )

        return report

    try:
        report = build_report(
            query,
            cleaned_answer,
            sources,
            mode=mode,
        )

    except (
        ValueError,
        TypeError,
        KeyError,
    ) as error:
        safe_error = _sanitize_error_message(
            error
        )

        logger.warning(
            "Structured report validation failed: %s",
            safe_error,
        )

        print_status(
            (
                "Structured Gemini report validation failed: "
                f"{safe_error}"
            ),
            "warning",
        )

        print_status(
            "Building a citation-safe evidence fallback report.",
            "info",
        )

        report = build_evidence_fallback_report(
            query,
            sources,
            mode=mode,
        )

    if not isinstance(
        report,
        dict,
    ):
        logger.error(
            "Structured report generation returned "
            "an invalid result type."
        )

        print_status(
            "Structured report generation returned an invalid result.",
            "warning",
        )

        report = build_evidence_fallback_report(
            query,
            sources,
            mode=mode,
        )

    save_report(
        report
    )

    return report


# ============================================================
# ANSWER GENERATION
# ============================================================


def generate_and_display_answer(
    query: str,
    sources: list[dict[str, Any]],
    mode: str = "quick",
) -> str:
    """
    Generate, validate, save, and display the research answer.

    If Gemini synthesis fails unexpectedly, the function does not
    attempt to treat an empty answer as a normal structured answer.
    It explicitly forces evidence-based fallback report generation.
    """

    if mode == "followup":
        heading = "GENERATING FOLLOW-UP ANSWER"
    else:
        heading = "GENERATING RESEARCH ANSWER"

    print(
        section_header(heading)
    )

    print_stage(
        1,
        2,
        "Synthesizing research",
        "Sending ranked evidence to Gemini.",
    )

    answer_generation_failed = False

    try:
        answer = generate_answer(
            query,
            sources,
        )

    except Exception as error:
        safe_error = _sanitize_error_message(
            error
        )

        logger.error(
            "Answer generation failed at CLI boundary: %s",
            safe_error,
        )

        print_status(
            (
                "Gemini synthesis failed unexpectedly. "
                "Building an evidence-based fallback report."
            ),
            "warning",
        )

        answer = ""
        answer_generation_failed = True

    answer = clean_answer_display(
        answer
    )

    print_status(
        "Answer synthesis completed.",
        "success" if answer else "warning",
    )

    print_stage(
        2,
        2,
        "Building structured report",
        (
            "Validating claims, citations, confidence, "
            "and source metadata."
        ),
    )

    print(
        section_header(
            "RESEARCHPILOT ANSWER"
        )
    )

    if answer:
        for line in answer.splitlines():
            if line.strip():
                print(
                    "\n".join(
                        wrap_text(line)
                    )
                )
            else:
                print()

    else:
        print(
            "No synthesized answer was returned. "
            "The report will use evidence-based fallback synthesis."
        )

    report = build_and_save_report(
        query,
        answer,
        sources,
        mode=mode,
        force_fallback=answer_generation_failed,
    )

    print_report(
        report
    )

    return report.get(
        "executive_answer",
        answer,
    )


# ============================================================
# FOLLOW-UP TOPIC HELPERS
# ============================================================


def extract_research_topic(
    session: dict[str, Any] | None,
) -> str:
    """
    Extract a reusable research topic from the previous session.

    Examples:
        "what is machine learning?"
            -> "machine learning"

        "what are neural networks?"
            -> "neural networks"

        "how does machine learning work?"
            -> "machine learning"

        "Compare Python and Java"
            -> "Compare Python and Java"
    """

    if not isinstance(
        session,
        dict,
    ):
        return ""

    workflow = session.get(
        "workflow"
    )

    if isinstance(
        workflow,
        dict,
    ):
        workflow_topic = str(
            workflow.get(
                "topic",
                "",
            )
            or ""
        ).strip()

        if workflow_topic:
            return workflow_topic

    topic = str(
        session.get(
            "topic",
            "",
        )
        or ""
    ).strip()

    query = str(
        session.get(
            "query",
            "",
        )
        or ""
    ).strip()

    if not topic:
        topic = query

    if not topic:
        return ""

    topic = re.sub(
        r"[?!.]+$",
        "",
        topic.strip(),
    ).strip()

    if re.search(
        r"\bvs\.?\b|\bversus\b|\bcompare\b",
        topic,
        flags=re.IGNORECASE,
    ):
        return topic

    patterns = [
        r"^what\s+is\s+(.+)$",
        r"^what\s+are\s+(.+)$",
        r"^who\s+is\s+(.+)$",
        r"^who\s+are\s+(.+)$",
        r"^where\s+is\s+(.+)$",
        r"^where\s+are\s+(.+)$",
        r"^how\s+does\s+(.+?)\s+work$",
        r"^how\s+do\s+(.+?)\s+work$",
        r"^how\s+is\s+(.+?)\s+used$",
        r"^how\s+are\s+(.+?)\s+used$",
        r"^explain\s+(.+)$",
        r"^tell\s+me\s+about\s+(.+)$",
    ]

    for pattern in patterns:
        match = re.match(
            pattern,
            topic,
            flags=re.IGNORECASE,
        )

        if match:
            extracted = match.group(1).strip()

            if extracted:
                return extracted

    return topic


def resolve_followup_query(
    followup_question: str,
    session: dict[str, Any] | None,
) -> str:
    """
    Resolve pronouns and short follow-up questions using the
    previous research topic.
    """

    previous_topic = extract_research_topic(
        session
    )

    if not previous_topic:
        return followup_question

    lower = followup_question.lower().strip()

    contextual_markers = [
        "it",
        "its",
        "they",
        "their",
        "this",
        "that",
        "these",
        "those",
        "the same",
        "above",
        "previous",
        "earlier",
    ]

    needs_context = any(
        re.search(
            rf"\b{re.escape(marker)}\b",
            lower,
        )
        for marker in contextual_markers
    )

    short_question = (
        len(
            followup_question.split()
        )
        <= 10
    )

    if needs_context or short_question:
        return (
            f"{previous_topic} "
            f"{followup_question}"
        ).strip()

    return followup_question


# ============================================================
# FOLLOW-UP MODE
# ============================================================


def run_followup() -> None:
    """Run a follow-up research turn."""

    session = load_session()

    if not session:
        print_status(
            "No previous research session was found.",
            "warning",
        )

        print(
            "Run Quick Research or Deep Research first."
        )

        return

    previous_query = session.get(
        "query",
        "",
    )

    previous_topic = extract_research_topic(
        session
    )

    print(
        section_header(
            "FOLLOW-UP RESEARCH"
        )
    )

    print(
        "\n".join(
            wrap_text(
                "Previous research topic: "
                f"{previous_topic or previous_query}"
            )
        )
    )

    followup_question = input(
        "\nFollow-up question: "
    ).strip()

    if not followup_question:
        print_status(
            "No follow-up question was provided.",
            "warning",
        )
        return

    effective_query = resolve_followup_query(
        followup_question,
        session,
    )

    print_stage(
        1,
        3,
        "Resolving follow-up context",
        effective_query,
    )

    search_results = _search_gracefully(
        effective_query,
        max_results=3,
    )

    if search_results is None:
        return

    print_status(
        f"Found {len(search_results)} sources.",
        "success" if search_results else "warning",
    )

    if not search_results:
        print_status(
            "No sources were found for the follow-up question.",
            "warning",
        )
        return

    print_stage(
        2,
        3,
        "Extracting and processing sources",
        (
            f"Preparing {len(search_results)} "
            "sources for the follow-up answer."
        ),
    )

    extracted_results = _extract_sources_gracefully(
        search_results
    )

    print_status(
        (
            f"Extracted content from "
            f"{len(extracted_results)} sources."
        ),
        "success" if extracted_results else "warning",
    )

    sources = prepare_sources(
        effective_query,
        search_results,
        extracted_results,
    )

    print_status(
        f"Prepared {len(sources)} research sources.",
        "success" if sources else "warning",
    )

    if not sources:
        print_status(
            "No usable research sources remained after processing.",
            "warning",
        )
        return

    print_sources(
        sources
    )

    save_research(
        sources
    )

    save_session(
        query=previous_query,
        topic=previous_topic or previous_query,
        last_question=followup_question,
        effective_query=effective_query,
        mode="followup",
    )

    print_stage(
        3,
        3,
        "Generating follow-up answer",
    )

    generate_and_display_answer(
        effective_query,
        sources,
        mode="followup",
    )

    print_completion()


# ============================================================
# COMPARE MODE
# ============================================================


def run_compare() -> None:
    """Run comparison research."""

    print(
        section_header(
            "COMPARE TOPICS"
        )
    )

    topic_a = input(
        "\nFirst topic: "
    ).strip()

    if not topic_a:
        print_status(
            "No first topic was provided.",
            "warning",
        )
        return

    topic_b = input(
        "Second topic: "
    ).strip()

    if not topic_b:
        print_status(
            "No second topic was provided.",
            "warning",
        )
        return

    query = (
        f"Compare {topic_a} and {topic_b}, "
        "including their definitions, key differences, "
        "advantages, limitations, and typical applications."
    )

    sources = perform_research(
        query,
        mode="compare",
    )

    if not sources:
        return

    save_research(
        sources
    )

    save_session(
        query=query,
        topic=f"{topic_a} vs {topic_b}",
        last_question="",
        effective_query=query,
        mode="compare",
    )

    generate_and_display_answer(
        query,
        sources,
        mode="compare",
    )

    print_completion()


# ============================================================
# QUICK / DEEP MODE
# ============================================================


def run_standard_research(
    query: str,
    mode: str = "quick",
) -> None:
    """Run standard quick or deep research."""

    sources = perform_research(
        query,
        mode=mode,
    )

    if not sources:
        return

    save_research(
        sources
    )

    save_session(
        query=query,
        topic=query,
        last_question="",
        effective_query=query,
        mode=mode,
    )

    generate_and_display_answer(
        query,
        sources,
        mode=mode,
    )

    print_completion()


# ============================================================
# MENU
# ============================================================


def show_menu() -> None:
    """Display the interactive ResearchPilot menu."""

    print(
        section_header(
            "RESEARCHPILOT"
        )
    )

    print(
        "\n1. Quick Research"
    )

    print(
        "   Fast research using 3 sources."
    )

    print(
        "\n2. Deep Research"
    )

    print(
        "   Broader research using up to 8 sources."
    )

    print(
        "\n3. Compare Topics"
    )

    print(
        "   Research comparison questions using multiple sources."
    )

    print(
        "\n4. Follow-up"
    )

    print(
        "   Continue the current research session."
    )

    print("\n5. View History")
    print("   Review persisted session and report summaries.")
    print("\n6. Exit")


def run_history() -> None:
    """Display persisted session and report summaries."""
    try:
        print_history(
            sessions=load_session_history(),
            reports=load_report_history(),
        )
    except (OSError, RuntimeError, ValueError) as error:
        print_status(
            f"Could not load research history: {error}",
            "error",
        )


def run_configuration() -> None:
    """Display non-secret runtime settings."""
    print_configuration(
        {
            "search_depth": SETTINGS.tavily_search_depth,
            "request_timeout_seconds": SETTINGS.request_timeout_seconds,
            "tavily_max_retries": SETTINGS.tavily_max_retries,
            "gemini_max_retries": SETTINGS.gemini_max_retries,
            "cache_enabled": SETTINGS.cache_enabled,
            "persistence_enabled": SETTINGS.persistence_enabled,
            "persistence_db": SETTINGS.persistence_db,
        }
    )


def run_menu() -> None:
    """Run the interactive menu loop."""

    while True:
        show_menu()

        choice = input(
            "\nSelect an option: "
        ).strip()

        if choice == "1":
            query = input(
                "\nWhat do you want to research? "
            ).strip()

            if query:
                run_standard_research(
                    query,
                    mode="quick",
                )
            else:
                print_status(
                    "No research question was provided.",
                    "warning",
                )

        elif choice == "2":
            query = input(
                "\nWhat do you want to research deeply? "
            ).strip()

            if query:
                run_standard_research(
                    query,
                    mode="deep",
                )
            else:
                print_status(
                    "No research question was provided.",
                    "warning",
                )

        elif choice == "3":
            run_compare()

        elif choice == "4":
            run_followup()

        elif choice == "5":
            run_history()

        elif choice == "6":
            print_status(
                "Exiting ResearchPilot.",
                "info",
            )
            break

        else:
            print_status(
                "Invalid option. Please select 1-6.",
                "warning",
            )


# ============================================================
# COMMAND-LINE ENTRY
# ============================================================


def main() -> None:
    """Run the ResearchPilot command-line application."""

    parser = argparse.ArgumentParser(
        description="ResearchPilot AI research assistant."
    )

    parser.add_argument(
        "query",
        nargs="*",
        help="Research question.",
    )

    parser.add_argument(
        "--deep",
        action="store_true",
        help="Run deep research using up to 8 sources.",
    )

    parser.add_argument(
        "--history",
        action="store_true",
        help="Display persisted research history.",
    )

    parser.add_argument(
        "--config",
        action="store_true",
        help="Display safe runtime configuration.",
    )

    parser.add_argument(
        "--export",
        metavar="PATH",
        help="Export the latest structured report to PATH.",
    )

    parser.add_argument(
        "--format",
        choices=("json", "markdown", "html", "csv"),
        default="json",
        help="Format used with --export.",
    )

    parser.add_argument(
        "--serve",
        action="store_true",
        help="Run the authenticated HTTP interface.",
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="HTTP bind host used with --serve.",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="HTTP port used with --serve.",
    )

    args = parser.parse_args()

    if args.serve:
        from research.web import serve

        serve(args.host, args.port)
        return

    if args.history:
        run_history()
        return

    if args.config:
        run_configuration()
        return

    if args.export:
        try:
            with open("research_report.json", "r", encoding="utf-8") as file:
                report = json.load(file)
            export_report(report, args.export, args.format)
            print_status(f"Report exported to {args.export}.", "success")
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            print_status(f"Could not export report: {error}", "error")
        return

    if args.query:
        query = " ".join(
            args.query
        ).strip()

        if not query:
            print_status(
                "No research question was provided.",
                "warning",
            )
            return

        mode = (
            "deep"
            if args.deep
            else "quick"
        )

        run_standard_research(
            query,
            mode=mode,
        )

        return

    run_menu()


if __name__ == "__main__":
    main()