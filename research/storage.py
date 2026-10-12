"""Persistence helpers for research output and the active session."""

import json
import os
import csv
import html
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import SETTINGS
from .persistence import SQLiteStore


RESEARCH_FILE = "research_output.json"
REPORT_FILE = "research_report.json"
SESSION_FILE = "research_session.json"


def export_report(
    report: dict[str, Any],
    filename: str,
    format: str = "json",
) -> None:
    """Export a structured report as JSON, Markdown, HTML, or CSV."""
    if not isinstance(report, dict):
        raise TypeError("report must be a dictionary.")

    normalized = str(format or "json").strip().lower()
    if normalized == "json":
        _save_json(report, filename, "Could not export report")
        return

    if normalized != "markdown":
        if normalized == "html":
            answer = html.escape(str(report.get("executive_answer", "")).strip())
            findings = report.get("key_findings", [])
            finding_items = ""
            if isinstance(findings, list):
                rendered_findings = []
                for item in findings:
                    text = item.get("text", "") if isinstance(item, dict) else item
                    if str(text).strip():
                        rendered_findings.append(
                            f"<li>{html.escape(str(text).strip())}</li>"
                        )
                finding_items = "".join(rendered_findings)
            content = (
                "<!doctype html><html><body><h1>Research Report</h1>"
                f"<h2>Executive Answer</h2><p>{answer}</p>"
                f"<h2>Key Findings</h2><ul>{finding_items}</ul>"
                "</body></html>"
            )
            try:
                Path(filename).write_text(content, encoding="utf-8")
            except OSError as error:
                raise RuntimeError(f"Could not export report: {error}") from error
            return

        if normalized == "csv":
            findings = report.get("key_findings", [])
            try:
                with open(filename, "w", encoding="utf-8", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["section", "text"])
                    writer.writerow(["executive_answer", report.get("executive_answer", "")])
                    for item in findings if isinstance(findings, list) else []:
                        text = item.get("text", "") if isinstance(item, dict) else item
                        writer.writerow(["key_finding", text])
            except OSError as error:
                raise RuntimeError(f"Could not export report: {error}") from error
            return

        raise ValueError("Report format must be json, markdown, html, or csv.")

    lines = ["# Research Report", ""]
    answer = str(report.get("executive_answer", "")).strip()
    if answer:
        lines.extend(["## Executive Answer", "", answer, ""])

    findings = report.get("key_findings", [])
    if isinstance(findings, list) and findings:
        lines.extend(["## Key Findings", ""])
        for finding in findings:
            if isinstance(finding, dict):
                text = str(finding.get("text", "")).strip()
            else:
                text = str(finding).strip()
            if text:
                lines.append(f"- {text}")
        lines.append("")

    limitations = report.get("limitations", [])
    if isinstance(limitations, list) and limitations:
        lines.extend(["## Limitations", ""])
        lines.extend(f"- {str(item).strip()}" for item in limitations if str(item).strip())
        lines.append("")

    try:
        with open(filename, "w", encoding="utf-8") as file:
            file.write("\n".join(lines).rstrip() + "\n")
    except OSError as error:
        raise RuntimeError(f"Could not export report: {error}") from error


def _store_for(filename: str | os.PathLike[str]) -> SQLiteStore:
    """Return the configured durable store, or a file-local test store."""

    if str(filename) == SESSION_FILE:
        return SQLiteStore(SETTINGS.persistence_db)

    return SQLiteStore(
        Path(filename).with_suffix(".researchpilot.db")
    )


def _session_key(filename: str | os.PathLike[str]) -> str:
    return str(Path(filename).resolve())


def _save_json(
    data: Any,
    filename: str,
    error_prefix: str,
) -> None:
    """Save JSON data with consistent error handling."""

    try:
        with open(
            filename,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                indent=2,
                ensure_ascii=False,
            )

    except OSError as error:
        raise RuntimeError(
            f"{error_prefix}: {error}"
        ) from error


def save_research(
    sources,
    filename=RESEARCH_FILE,
):
    """Save normalized research sources."""

    _save_json(
        sources,
        filename,
        "Could not save research report",
    )

    if SETTINGS.persistence_enabled:
        _store_for(filename).save_artifact(
            "research_runs",
            _session_key(filename),
            sources,
            datetime.now(timezone.utc).isoformat(),
        )

    print(
        f"\nResearch report saved to: "
        f"{filename}"
    )


def save_report(
    report,
    filename=REPORT_FILE,
):
    """Save the structured Phase 3 research report."""

    if not isinstance(
        report,
        dict,
    ):
        raise TypeError(
            "report must be a dictionary."
        )

    _save_json(
        report,
        filename,
        "Could not save structured research report",
    )

    if SETTINGS.persistence_enabled:
        _store_for(filename).save_artifact(
            "reports",
            _session_key(filename),
            report,
            datetime.now(timezone.utc).isoformat(),
        )

    print(
        f"\nStructured report saved to: "
        f"{filename}"
    )


def save_session(
    query,
    topic=None,
    last_question="",
    effective_query=None,
    mode="quick",
    workflow=None,
    filename=SESSION_FILE,
):
    """Save the current ResearchPilot session.

    ``workflow`` is optional so existing callers and saved sessions
    remain backward compatible while Phase 2 can persist structured
    workflow state.
    """

    session: dict[str, Any] = {
        "query": str(
            query or ""
        ).strip(),

        "topic": str(
            topic or query or ""
        ).strip(),

        "last_question": str(
            last_question or ""
        ).strip(),

        "effective_query": str(
            effective_query or query or ""
        ).strip(),

        "mode": str(
            mode or "quick"
        ).strip().lower(),

        "updated_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    if workflow is not None:
        if hasattr(
            workflow,
            "to_dict",
        ):
            workflow = workflow.to_dict()

        if not isinstance(
            workflow,
            dict,
        ):
            raise TypeError(
                "workflow must be a dictionary "
                "or expose to_dict()."
            )

        session["workflow"] = workflow

    _save_json(
        session,
        filename,
        "Could not save session",
    )

    if SETTINGS.persistence_enabled:
        _store_for(filename).save_session(
            _session_key(filename),
            session,
            session["updated_at"],
        )

    print(
        "\nSession updated."
    )


def load_session(
    filename=SESSION_FILE,
):
    """Load the current session or return an empty dictionary."""

    try:
        with open(
            filename,
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)

    except (
        FileNotFoundError,
        json.JSONDecodeError,
        OSError,
    ):
        if SETTINGS.persistence_enabled:
            stored = _store_for(filename).load_session(
                _session_key(filename)
            )
            return stored or {}
        return {}

    return (
        data
        if isinstance(
            data,
            dict,
        )
        else {}
    )


def clear_session(
    filename=SESSION_FILE,
):
    """Delete the saved research session if it exists."""

    try:
        os.remove(
            filename
        )

    except FileNotFoundError:
        pass

    if SETTINGS.persistence_enabled:
        _store_for(filename).delete_session(
            _session_key(filename)
        )


def load_research_history(
    filename=RESEARCH_FILE,
) -> list[Any]:
    """Load durable research runs associated with an export filename."""

    if not SETTINGS.persistence_enabled:
        return []

    return _store_for(filename).list_artifacts(
        "research_runs",
        _session_key(filename),
    )


def load_report_history(
    filename=REPORT_FILE,
) -> list[Any]:
    """Load durable structured reports associated with an export filename."""

    if not SETTINGS.persistence_enabled:
        return []

    return _store_for(filename).list_artifacts(
        "reports",
        _session_key(filename),
    )


def load_session_history(
    filename=SESSION_FILE,
) -> list[dict[str, Any]]:
    """Return the current durable session as a history-compatible list."""
    if not SETTINGS.persistence_enabled:
        return []

    return _store_for(filename).list_sessions()