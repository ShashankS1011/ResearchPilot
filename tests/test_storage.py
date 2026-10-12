import json

from research.storage import (
    clear_session,
    load_session,
    load_report_history,
    load_research_history,
    load_session_history,
    save_research,
    save_report,
    save_session,
)


def test_save_and_load_research(tmp_path):
    filename = tmp_path / "research.json"

    sources = [
        {
            "id": 1,
            "title": "Test Source",
            "url": "https://example.com",
            "evidence": [
                {
                    "text": "Test evidence",
                    "confidence": 0.9,
                }
            ],
        }
    ]

    save_research(
        sources,
        filename=str(filename),
    )

    with open(
        filename,
        "r",
        encoding="utf-8",
    ) as file:
        saved_data = json.load(file)

    assert saved_data == sources


def test_save_and_load_session(tmp_path):
    filename = tmp_path / "session.json"

    save_session(
        query="machine learning",
        topic="machine learning",
        last_question="What are its main types?",
        effective_query=(
            "machine learning "
            "What are its main types?"
        ),
        mode="follow-up",
        filename=str(filename),
    )

    session = load_session(
        filename=str(filename)
    )

    assert session["query"] == "machine learning"
    assert session["topic"] == "machine learning"
    assert (
        session["last_question"]
        == "What are its main types?"
    )
    assert (
        session["effective_query"]
        == (
            "machine learning "
            "What are its main types?"
        )
    )
    assert session["mode"] == "follow-up"
    assert "updated_at" in session


def test_load_missing_session_returns_empty_dict(tmp_path):
    filename = tmp_path / "does_not_exist.json"

    session = load_session(
        filename=str(filename)
    )

    assert session == {}


def test_load_invalid_json_returns_empty_dict(tmp_path):
    filename = tmp_path / "invalid.json"

    filename.write_text(
        "{not valid json",
        encoding="utf-8",
    )

    session = load_session(
        filename=str(filename)
    )

    assert session == {}


def test_clear_session(tmp_path):
    filename = tmp_path / "session.json"

    save_session(
        query="test",
        filename=str(filename),
    )

    assert filename.exists()

    clear_session(
        filename=str(filename)
    )

    assert not filename.exists()


def test_clear_missing_session_does_not_fail(tmp_path):
    filename = tmp_path / "missing.json"

    clear_session(
        filename=str(filename)
    )


def test_storage_keeps_durable_histories(tmp_path):
    research_filename = tmp_path / "research.json"
    report_filename = tmp_path / "report.json"

    save_research([{"url": "https://example.com"}], str(research_filename))
    save_report({"answer": "durable"}, str(report_filename))

    assert load_research_history(str(research_filename)) == [
        [{"url": "https://example.com"}]
    ]
    assert load_report_history(str(report_filename)) == [
        {"answer": "durable"}
    ]


def test_storage_exposes_durable_session_history(tmp_path):
    filename = tmp_path / "session.json"

    save_session("machine learning", filename=str(filename))

    assert load_session_history(str(filename))[0]["query"] == (
        "machine learning"
    )