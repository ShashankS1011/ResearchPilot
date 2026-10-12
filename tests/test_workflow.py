from research.planner import (
    build_research_plan,
)
from research.workflow import (
    WorkflowState,
    create_workflow,
    execute_workflow,
    extract_research_topic,
    resolve_followup_query,
)


def test_workflow_advances_in_dependency_order():
    state = create_workflow(
        build_research_plan(
            "What is machine learning?"
        )
    )

    assert (
        state.current_step.step_id
        == "definition"
    )

    state.mark_completed(
        "definition"
    )

    assert (
        state.current_step.step_id
        == "concepts"
    )

    state.mark_completed(
        "concepts"
    )

    assert (
        state.current_step.step_id
        == "mechanism"
    )


def test_failed_step_does_not_look_completed():
    state = create_workflow(
        build_research_plan(
            "What is machine learning?"
        )
    )

    state.mark_failed(
        "definition"
    )

    assert (
        state.current_step.step_id
        == "definition"
    )

    assert (
        "definition"
        in state.failed_steps
    )

    assert (
        "definition"
        not in state.completed_steps
    )


def test_workflow_round_trips_to_dict():
    state = create_workflow(
        build_research_plan(
            "machine learning"
        )
    )

    state.mark_completed(
        "overview"
    )

    state.add_follow_up(
        "What are its main types?",
        (
            "machine learning "
            "What are its main types?"
        ),
    )

    restored = WorkflowState.from_dict(
        state.to_dict()
    )

    assert (
        restored.original_query
        == "machine learning"
    )

    assert (
        restored.completed_steps
        == ["overview"]
    )

    assert (
        restored.follow_ups[0].question
        == "What are its main types?"
    )

    assert (
        restored.plan.steps[0].step_id
        == "overview"
    )


def test_followup_resolution_uses_topic():
    assert resolve_followup_query(
        "What are its main types?",
        "machine learning",
    ) == (
        "machine learning "
        "What are its main types?"
    )


def test_followup_resolution_preserves_specific_question():
    question = (
        "How did transformer architectures "
        "influence modern language models "
        "after large-scale pretraining became common?"
    )

    assert (
        resolve_followup_query(
            question,
            "machine learning",
        )
        == question
    )


def test_extract_topic_prefers_structured_workflow_topic():
    session = {
        "query": "what is machine learning?",
        "topic": "machine learning",
        "workflow": {
            "original_query": "machine learning",
        },
    }

    assert (
        extract_research_topic(
            session
        )
        == "machine learning"
    )


def test_execute_workflow_runs_steps_in_dependency_order():
    state = create_workflow(
        build_research_plan(
            "machine learning"
        )
    )

    calls = []

    def fake_search(
        query,
        max_results,
    ):
        calls.append(query)

        return [
            {
                "url": (
                    "https://example.com/"
                    f"{len(calls)}"
                ),
                "title": query,
            }
        ]

    results = execute_workflow(
        state,
        fake_search,
        max_results=4,
    )

    assert [
        item.step_id
        for item in results
    ] == [
        "overview",
        "mechanism",
        "approaches",
        "tradeoffs",
    ]

    assert state.is_complete
    assert len(calls) == 4


def test_execute_workflow_stops_when_cancelled():
    state = create_workflow(
        build_research_plan("machine learning")
    )
    state.cancel()

    assert execute_workflow(
        state,
        lambda query, max_results: [],
        max_results=4,
    ) == ()
    assert not state.is_complete