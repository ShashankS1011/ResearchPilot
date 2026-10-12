import json
import os
from datetime import datetime


DEFAULT_SESSION_FILE = "research_session.json"


def create_session(
    original_query="",
    mode="quick",
):
    """Create a new research session."""

    return {
        "session_id": datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        ),
        "created_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "updated_at": datetime.now().isoformat(
            timespec="seconds"
        ),
        "mode": mode,
        "original_query": original_query,
        "turns": [],
        "research_runs": 0,
    }


def load_session(
    filename=DEFAULT_SESSION_FILE,
):
    """Load the most recent research session."""

    if not os.path.exists(filename):
        return None

    try:
        with open(
            filename,
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    except (
        json.JSONDecodeError,
        OSError,
    ):
        return None


def save_session(
    session,
    filename=DEFAULT_SESSION_FILE,
):
    """Persist the current research session."""

    session["updated_at"] = datetime.now().isoformat(
        timespec="seconds"
    )

    with open(
        filename,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            session,
            file,
            indent=2,
            ensure_ascii=False,
        )


def add_turn(
    session,
    question,
    answer,
    mode,
):
    """Add a question/answer pair to the session."""

    session.setdefault(
        "turns",
        [],
    )

    session["turns"].append(
        {
            "timestamp": datetime.now().isoformat(
                timespec="seconds"
            ),
            "question": question,
            "answer": answer,
            "mode": mode,
        }
    )

    session["research_runs"] = (
        int(
            session.get(
                "research_runs",
                0,
            )
        )
        + 1
    )

    session["updated_at"] = datetime.now().isoformat(
        timespec="seconds"
    )


def get_recent_context(
    session,
    max_turns=3,
):
    """
    Return recent conversation context.

    We deliberately limit the context so that follow-up
    questions do not create an enormous Gemini prompt.
    """

    turns = session.get(
        "turns",
        [],
    )

    if not turns:
        return ""

    recent = turns[-max_turns:]

    context_parts = []

    for index, turn in enumerate(
        recent,
        start=1,
    ):
        question = str(
            turn.get(
                "question",
                "",
            )
        ).strip()

        answer = str(
            turn.get(
                "answer",
                "",
            )
        ).strip()

        if not question:
            continue

        # Keep historical answers bounded.
        answer = answer[:4000]

        context_parts.append(
            f"Previous question {index}: {question}\n"
            f"Previous answer {index}:\n{answer}"
        )

    return "\n\n".join(
        context_parts
    )


def build_follow_up_query(
    session,
    current_question,
):
    """
    Build a synthesis query containing recent context.

    The actual Tavily search should still use only the
    current question. This function is used only for
    answer synthesis.
    """

    context = get_recent_context(
        session
    )

    current_question = str(
        current_question
    ).strip()

    if not context:
        return current_question

    return (
        "Use the previous research conversation as context "
        "when answering the current question.\n\n"
        "PREVIOUS RESEARCH CONTEXT\n"
        "--------------------------\n"
        f"{context}\n\n"
        "CURRENT QUESTION\n"
        "----------------\n"
        f"{current_question}"
    )