import json

from research.storage import export_report


def test_export_report_writes_markdown(tmp_path):
    filename = tmp_path / "report.md"

    export_report(
        {
            "executive_answer": "A concise answer.",
            "key_findings": [{"text": "A supported finding. [1]"}],
            "limitations": ["Limited source coverage."],
        },
        str(filename),
        "markdown",
    )

    content = filename.read_text(encoding="utf-8")

    assert "# Research Report" in content
    assert "A concise answer." in content
    assert "- A supported finding. [1]" in content


def test_export_report_writes_json(tmp_path):
    filename = tmp_path / "report.json"
    report = {"executive_answer": "Answer"}

    export_report(report, str(filename), "json")

    assert json.loads(filename.read_text(encoding="utf-8")) == report
