from research.planner import build_research_plan


def test_definition_question_is_decomposed_into_research_angles():
    plan = build_research_plan(
        "What is machine learning?",
        mode="quick",
    )

    assert (
        plan.original_query
        == "What is machine learning"
    )

    assert len(plan.queries) == 4

    assert any(
        "definition" in query.lower()
        for query in plan.queries
    )

    assert any(
        "types" in query.lower()
        for query in plan.queries
    )

    assert any(
        "applications" in query.lower()
        for query in plan.queries
    )


def test_general_question_gets_focused_queries():
    plan = build_research_plan(
        "How does solar energy work?",
        mode="deep",
    )

    assert len(plan.queries) == 4

    assert plan.mode == "deep"

    assert len(
        {
            query.casefold()
            for query in plan.queries
        }
    ) == 4


def test_compare_plan_contains_comparison_dimensions():
    plan = build_research_plan(
        "Python vs Rust",
        mode="compare",
    )

    assert len(plan.queries) == 4

    assert any(
        "differences" in query.lower()
        for query in plan.queries
    )

    assert any(
        "performance" in query.lower()
        for query in plan.queries
    )

    assert any(
        "limitations" in query.lower()
        for query in plan.queries
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