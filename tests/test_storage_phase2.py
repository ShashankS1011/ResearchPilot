from research.planner import (
    build_research_plan,
)
from research.storage import (
    load_session,
    save_session,
)
from research.workflow import (
    create_workflow,
)


def test_save_session_persists_workflow_state(
    tmp_path,
):
    filename = (
        tmp_path
        / "session.json"
    )

    workflow = create_workflow(
        build_research_plan(
            "machine learning"
        )
    )

    workflow.mark_completed(
        "overview"
    )

    save_session(
        "machine learning",
        topic="machine learning",
        mode="quick",
        workflow=workflow,
        filename=str(filename),
    )

    session = load_session(
        str(filename)
    )

    assert (
        session["workflow"][
            "original_query"
        ]
        == "machine learning"
    )

    assert (
        session["workflow"][
            "completed_steps"
        ]
        == ["overview"]
    )

    assert len(
        session["workflow"][
            "plan"
        ]["steps"]
    ) == 4