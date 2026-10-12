from __future__ import annotations

from io import StringIO
from unittest.mock import patch

import research.ui as ui


def test_terminal_width_has_safe_bounds():
    with patch(
        "research.ui.shutil.get_terminal_size",
        return_value=(200, 24),
    ):
        width = ui.terminal_width()

    assert width == ui.MAX_TERMINAL_WIDTH


def test_terminal_width_uses_minimum_bound():
    with patch(
        "research.ui.shutil.get_terminal_size",
        return_value=(20, 24),
    ):
        width = ui.terminal_width()

    assert width == ui.MIN_TERMINAL_WIDTH


def test_wrap_text_preserves_complete_words():
    text = (
        "This is a deliberately long sentence that "
        "should be wrapped for terminal presentation."
    )

    lines = ui.wrap_text(
        text,
        width=40,
    )

    assert len(lines) > 1
    assert "terminal presentation." in " ".join(lines)


def test_wrap_text_empty_input():
    assert ui.wrap_text("") == []
    assert ui.wrap_text(None) == []


def test_divider_has_requested_width():
    assert ui.divider("-", width=25) == "-" * 25


def test_section_header_contains_title():
    result = ui.section_header(
        "Research Report",
        width=30,
        color=False,
    )

    assert "RESEARCH REPORT" in result
    assert "=" * 30 in result


def test_subsection_header_contains_title():
    result = ui.subsection_header(
        "Key Findings",
        width=30,
        color=False,
    )

    assert "KEY FINDINGS" in result
    assert "-" * 30 in result


def test_status_message_has_no_ansi_when_color_disabled():
    result = ui.status_message(
        "Research completed.",
        level="success",
        color=False,
    )

    assert result == "✓ Research completed."
    assert "\033[" not in result


def test_style_adds_ansi_when_enabled():
    result = ui.style(
        "test",
        ui.BOLD,
        enabled=True,
    )

    assert result.startswith(ui.BOLD)
    assert result.endswith(ui.RESET)


def test_clean_display_url_handles_markdown_link():
    result = ui.clean_display_url(
        "[Example](https://example.com/page)"
    )

    assert result == "https://example.com/page"


def test_clean_display_url_handles_plain_url():
    result = ui.clean_display_url(
        "https://example.com/page"
    )

    assert result == "https://example.com/page"


def test_format_score_handles_numeric_value():
    assert ui.format_score(0.8765) == "0.88"


def test_format_score_handles_invalid_value():
    assert ui.format_score("not-a-number") == "N/A"


def test_print_stage_outputs_stage_number():
    output = StringIO()

    with patch(
        "research.ui.print",
        side_effect=output.write,
    ):
        ui.print_stage(
            2,
            3,
            "Extracting source content",
            color=False,
        )

    assert "[2/3] Extracting source content" in output.getvalue()


def test_print_sources_displays_source_metadata(capsys):
    sources = [
        {
            "id": 1,
            "title": "Example source",
            "url": "https://example.com",
            "score": 0.91,
            "quality_score": 0.88,
            "quality_category": "technical",
            "evidence": [
                {
                    "text": "Example evidence.",
                    "confidence": 0.90,
                    "agreement_count": 2,
                    "supporting_sources": [1, 2],
                    "classification": "strong",
                }
            ],
            "llm_evidence": [
                {
                    "text": "Example evidence.",
                }
            ],
        }
    ]

    ui.print_sources(
        sources,
        color=False,
    )

    captured = capsys.readouterr().out

    assert "RESEARCH SOURCES (1)" in captured
    assert "Example source" in captured
    assert "https://example.com" in captured
    assert "0.91" in captured
    assert "0.88" in captured
    assert "technical" in captured
    assert "Example evidence." in captured


def test_print_report_displays_structured_sections(capsys):
    report = {
        "executive_answer": (
            "Machine learning uses data to learn patterns."
        ),
        "key_findings": [
            {
                "text": "The model learns from examples. [1]",
                "confidence": 0.91,
                "confidence_label": "high",
                "citation_ids": [1],
                "supported": True,
            }
        ],
        "limitations": [
            "The report is limited by the available sources."
        ],
        "citation_integrity": {
            "valid": True,
            "referenced_source_ids": [1],
            "uncited_claim_count": 0,
        },
        "sources": [
            {
                "source_id": 1,
                "title": "Example source",
                "domain": "example.com",
                "source_category": "technical",
                "quality_score": 0.90,
                "research_step": "overview",
                "url": "https://example.com",
            }
        ],
    }

    ui.print_report(
        report,
        color=False,
    )

    captured = capsys.readouterr().out

    assert "RESEARCH REPORT" in captured
    assert "EXECUTIVE ANSWER" in captured
    assert "KEY FINDINGS" in captured
    assert "LIMITATIONS / UNCERTAINTY" in captured
    assert "CITATION INTEGRITY" in captured
    assert "SOURCE METADATA" in captured
    assert "Machine learning uses data" in captured
    assert "The model learns from examples." in captured
    assert "Example source" in captured


def test_print_report_handles_empty_report(capsys):
    ui.print_report(
        {
            "executive_answer": "",
            "key_findings": [],
            "limitations": [],
            "citation_integrity": {},
            "sources": [],
        },
        color=False,
    )

    captured = capsys.readouterr().out

    assert "RESEARCH REPORT" in captured
    assert "No structured findings were produced." in captured
    assert "No explicit limitations were recorded." in captured
    assert "No source metadata is available." in captured


def test_print_completion_outputs_completion_marker(capsys):
    ui.print_completion(
        color=False,
    )

    captured = capsys.readouterr().out

    assert "RESEARCH COMPLETE" in captured


def test_print_history_displays_sessions_and_reports(capsys):
    ui.print_history(
        sessions=[
            {
                "query": "machine learning",
                "mode": "deep",
                "updated_at": "2026-10-12T00:00:00+00:00",
            }
        ],
        reports=[
            {
                "executive_answer": "A concise research answer.",
                "key_findings": [{"text": "Finding"}],
            }
        ],
        color=False,
    )

    captured = capsys.readouterr().out

    assert "RESEARCH HISTORY" in captured
    assert "machine learning" in captured
    assert "Mode: deep" in captured
    assert "A concise research answer." in captured


def test_print_history_handles_empty_data(capsys):
    ui.print_history(color=False)

    captured = capsys.readouterr().out

    assert "No saved sessions are available." in captured
    assert "No saved reports are available." in captured


def test_print_configuration_hides_nothing_beyond_supplied_values(capsys):
    ui.print_configuration(
        {"cache_enabled": True, "request_timeout_seconds": 45},
        color=False,
    )

    captured = capsys.readouterr().out

    assert "cache_enabled: True" in captured
    assert "request_timeout_seconds: 45" in captured


def test_status_message_uses_ascii_symbols_for_legacy_console(monkeypatch):
    class LegacyStdout:
        encoding = "cp1252"

    monkeypatch.setattr(ui.sys, "stdout", LegacyStdout())

    assert ui.status_message("done", "success", color=False) == "OK done"