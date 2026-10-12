import csv

from research.storage import export_report


def test_export_report_writes_html(tmp_path):
    filename = tmp_path / "report.html"
    export_report({"executive_answer": "<safe>"}, str(filename), "html")

    content = filename.read_text(encoding="utf-8")
    assert "<h1>Research Report</h1>" in content
    assert "&lt;safe&gt;" in content


def test_export_report_writes_csv(tmp_path):
    filename = tmp_path / "report.csv"
    export_report(
        {
            "executive_answer": "Answer",
            "key_findings": [{"text": "Finding"}],
        },
        str(filename),
        "csv",
    )

    with filename.open(encoding="utf-8", newline="") as file:
        rows = list(csv.reader(file))
    assert rows[0] == ["section", "text"]
    assert rows[1] == ["executive_answer", "Answer"]
    assert rows[2] == ["key_finding", "Finding"]
