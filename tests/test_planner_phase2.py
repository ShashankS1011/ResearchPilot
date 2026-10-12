from research.planner import (
    ResearchStep,
    build_research_plan,
)


def test_plan_contains_dependency_aware_steps():
    plan = build_research_plan(
        "What is machine learning?",
        mode="quick",
    )

    assert len(plan.steps) == 4

    assert all(
        isinstance(
            step,
            ResearchStep,
        )
        for step in plan.steps
    )

    assert plan.queries == tuple(
        step.query
        for step in plan.steps
    )

    assert (
        plan.steps[0].step_id
        == "definition"
    )

    assert (
        plan.steps[-1].depends_on
        == ("mechanism",)
    )


def test_comparison_plan_has_distinct_research_angles():
    plan = build_research_plan(
        "Python vs Java",
        mode="compare",
    )

    assert [
        step.step_id
        for step in plan.steps
    ] == [
        "overview",
        "differences",
        "applications",
        "tradeoffs",
    ]

    assert (
        plan.steps[3].depends_on
        == (
            "differences",
            "applications",
        )
    )


def test_empty_query_is_rejected():
    try:
        build_research_plan("   ")

    except ValueError as error:
        assert (
            "cannot be empty"
            in str(error)
        )

    else:
        raise AssertionError(
            "Expected ValueError"
        )