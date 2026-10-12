"""Deterministic research planning and query decomposition."""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ResearchStep:
    """One focused step in a research investigation."""

    step_id: str
    purpose: str
    query: str
    depends_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResearchPlan:
    """A deterministic multi-step research plan."""

    original_query: str
    steps: tuple[ResearchStep, ...]
    mode: str

    @property
    def queries(self) -> tuple[str, ...]:
        """Backward-compatible access to planned search queries."""

        return tuple(
            step.query
            for step in self.steps
        )


def _clean_query(query: str) -> str:
    """Normalize whitespace and terminal punctuation."""

    return re.sub(
        r"\s+",
        " ",
        str(query or "").strip(),
    ).strip(" ?!.")


def _unique_steps(
    steps: list[ResearchStep],
) -> tuple[ResearchStep, ...]:
    """Remove duplicate queries while preserving step order."""

    seen: set[str] = set()
    result: list[ResearchStep] = []

    for step in steps:
        query = _clean_query(step.query)
        key = query.casefold()

        if not query or key in seen:
            continue

        seen.add(key)

        result.append(
            ResearchStep(
                step_id=step.step_id,
                purpose=step.purpose,
                query=query,
                depends_on=step.depends_on,
            )
        )

    return tuple(result)


def _comparison_plan(
    query: str,
) -> list[ResearchStep]:
    """Build independent research angles for a comparison."""

    return [
        ResearchStep(
            "overview",
            "Establish the baseline for the comparison",
            query,
        ),
        ResearchStep(
            "differences",
            "Identify key similarities and differences",
            f"{query} key differences and similarities",
            ("overview",),
        ),
        ResearchStep(
            "applications",
            "Compare performance, use cases, and applications",
            f"{query} performance use cases and applications",
            ("overview",),
        ),
        ResearchStep(
            "tradeoffs",
            "Identify advantages, limitations, and tradeoffs",
            f"{query} advantages limitations and tradeoffs",
            ("differences", "applications"),
        ),
    ]


def _general_plan(
    query: str,
) -> list[ResearchStep]:
    """Build focused research angles for a general question."""

    return [
        ResearchStep(
            "overview",
            "Establish the core facts and scope",
            query,
        ),
        ResearchStep(
            "mechanism",
            "Understand how the subject works",
            f"{query} definition and how it works",
            ("overview",),
        ),
        ResearchStep(
            "approaches",
            "Identify major types, categories, or approaches",
            f"{query} types categories and approaches",
            ("overview",),
        ),
        ResearchStep(
            "tradeoffs",
            "Identify applications, benefits, and limitations",
            (
                f"{query} applications use cases "
                "benefits and limitations"
            ),
            ("mechanism", "approaches"),
        ),
    ]


def _definition_plan(
    query: str,
) -> list[ResearchStep]:
    """Build focused research angles for definition-style questions."""

    return [
        ResearchStep(
            "definition",
            "Establish the definition and scope",
            query,
        ),
        ResearchStep(
            "concepts",
            "Identify the core concepts",
            f"{query} definition and core concepts",
            ("definition",),
        ),
        ResearchStep(
            "mechanism",
            "Explain how it works and its main types",
            f"{query} how it works and main types",
            ("definition",),
        ),
        ResearchStep(
            "applications",
            "Identify applications, benefits, and limitations",
            f"{query} applications benefits and limitations",
            ("mechanism",),
        ),
    ]


def _looks_like_definition_question(
    query: str,
) -> bool:
    """Return whether a query primarily asks what something is."""

    return bool(
        re.search(
            r"^\s*"
            r"(what\s+(?:is|are)|define|definition\s+of|explain)"
            r"\b",
            query,
            flags=re.IGNORECASE,
        )
    )


def build_research_plan(
    query: str,
    mode: str = "quick",
) -> ResearchPlan:
    """Create a deterministic, dependency-aware research plan."""

    original = _clean_query(query)

    if not original:
        raise ValueError(
            "Research query cannot be empty."
        )

    normalized_mode = str(
        mode or "quick"
    ).strip().lower()

    if (
        normalized_mode == "compare"
        or re.search(
            r"\b(?:vs\.?|versus|compare)\b",
            original,
            flags=re.IGNORECASE,
        )
    ):
        steps = _comparison_plan(original)

    elif _looks_like_definition_question(original):
        steps = _definition_plan(original)

    else:
        steps = _general_plan(original)

    return ResearchPlan(
        original_query=original,
        steps=_unique_steps(steps),
        mode=normalized_mode,
    )