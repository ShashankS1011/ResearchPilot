"""Research workflow state and follow-up orchestration."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
import re

from .planner import ResearchPlan, ResearchStep


def _utc_now() -> str:
    """Return the current UTC timestamp."""

    return datetime.now(
        timezone.utc
    ).isoformat()


@dataclass
class FollowUpTurn:
    """A single follow-up question and its resolved research query."""

    question: str
    effective_query: str
    created_at: str = field(
        default_factory=_utc_now
    )


@dataclass
class WorkflowState:
    """Serializable state for one active research investigation."""

    original_query: str
    mode: str
    plan: ResearchPlan
    completed_steps: list[str] = field(
        default_factory=list
    )
    failed_steps: list[str] = field(
        default_factory=list
    )
    follow_ups: list[FollowUpTurn] = field(
        default_factory=list
    )
    created_at: str = field(
        default_factory=_utc_now
    )
    updated_at: str = field(
        default_factory=_utc_now
    )
    cancelled: bool = False

    @property
    def pending_steps(
        self,
    ) -> tuple[ResearchStep, ...]:
        """Return steps that have not completed successfully."""

        completed = set(
            self.completed_steps
        )

        return tuple(
            step
            for step in self.plan.steps
            if step.step_id not in completed
        )

    @property
    def current_step(
        self,
    ) -> ResearchStep | None:
        """Return the next dependency-ready step."""

        completed = set(
            self.completed_steps
        )

        for step in self.plan.steps:
            if step.step_id in completed:
                continue

            if all(
                dependency in completed
                for dependency in step.depends_on
            ):
                return step

        return None

    @property
    def is_complete(self) -> bool:
        """Return True when every planned step has completed."""

        return len(
            self.completed_steps
        ) >= len(self.plan.steps)

    def cancel(self) -> None:
        """Request cooperative cancellation of this workflow."""

        self.cancelled = True
        self.updated_at = _utc_now()

    def resume(self) -> None:
        """Resume a previously cancelled workflow."""

        self.cancelled = False
        self.updated_at = _utc_now()

    def mark_completed(
        self,
        step_id: str,
    ) -> None:
        """Mark one step complete."""

        valid_ids = {
            step.step_id
            for step in self.plan.steps
        }

        if step_id not in valid_ids:
            raise ValueError(
                f"Unknown research step: {step_id}"
            )

        if step_id not in self.completed_steps:
            self.completed_steps.append(
                step_id
            )

        if step_id in self.failed_steps:
            self.failed_steps.remove(
                step_id
            )

        self.updated_at = _utc_now()

    def mark_failed(
        self,
        step_id: str,
    ) -> None:
        """Record a failed step without marking it complete."""

        valid_ids = {
            step.step_id
            for step in self.plan.steps
        }

        if step_id not in valid_ids:
            raise ValueError(
                f"Unknown research step: {step_id}"
            )

        if step_id not in self.failed_steps:
            self.failed_steps.append(
                step_id
            )

        self.updated_at = _utc_now()

    def add_follow_up(
        self,
        question: str,
        effective_query: str,
    ) -> None:
        """Record a follow-up turn."""

        question = str(
            question or ""
        ).strip()

        effective_query = str(
            effective_query or ""
        ).strip()

        if not question:
            raise ValueError(
                "Follow-up question cannot be empty."
            )

        if not effective_query:
            raise ValueError(
                "Effective follow-up query cannot be empty."
            )

        self.follow_ups.append(
            FollowUpTurn(
                question=question,
                effective_query=effective_query,
            )
        )

        self.updated_at = _utc_now()

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Serialize workflow state to JSON-compatible data."""

        data = asdict(self)

        data["plan"]["steps"] = [
            asdict(step)
            for step in self.plan.steps
        ]

        data["follow_ups"] = [
            asdict(turn)
            for turn in self.follow_ups
        ]

        return data

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
    ) -> "WorkflowState":
        """Restore workflow state from persisted JSON data."""

        if not isinstance(
            data,
            dict,
        ):
            raise ValueError(
                "Workflow state must be a dictionary."
            )

        plan_data = data.get(
            "plan"
        )

        if not isinstance(
            plan_data,
            dict,
        ):
            raise ValueError(
                "Workflow state is missing a valid plan."
            )

        raw_steps = plan_data.get(
            "steps",
            [],
        )

        steps = tuple(
            ResearchStep(
                step_id=str(
                    step.get(
                        "step_id",
                        "",
                    )
                ).strip(),
                purpose=str(
                    step.get(
                        "purpose",
                        "",
                    )
                ).strip(),
                query=str(
                    step.get(
                        "query",
                        "",
                    )
                ).strip(),
                depends_on=tuple(
                    step.get(
                        "depends_on",
                        (),
                    )
                ),
            )
            for step in raw_steps
            if isinstance(
                step,
                dict,
            )
        )

        plan = ResearchPlan(
            original_query=str(
                plan_data.get(
                    "original_query",
                    "",
                )
            ).strip(),
            steps=steps,
            mode=str(
                plan_data.get(
                    "mode",
                    "quick",
                )
            ).strip().lower(),
        )

        follow_ups = [
            FollowUpTurn(
                question=str(
                    item.get(
                        "question",
                        "",
                    )
                ).strip(),
                effective_query=str(
                    item.get(
                        "effective_query",
                        "",
                    )
                ).strip(),
                created_at=str(
                    item.get(
                        "created_at",
                        _utc_now(),
                    )
                ),
            )
            for item in data.get(
                "follow_ups",
                [],
            )
            if isinstance(
                item,
                dict,
            )
        ]

        return cls(
            original_query=str(
                data.get(
                    "original_query",
                    "",
                )
            ).strip(),
            mode=str(
                data.get(
                    "mode",
                    "quick",
                )
            ).strip().lower(),
            plan=plan,
            completed_steps=list(
                data.get(
                    "completed_steps",
                    [],
                )
            ),
            failed_steps=list(
                data.get(
                    "failed_steps",
                    [],
                )
            ),
            follow_ups=follow_ups,
            created_at=str(
                data.get(
                    "created_at",
                    _utc_now(),
                )
            ),
            updated_at=str(
                data.get(
                    "updated_at",
                    _utc_now(),
                )
            ),
            cancelled=bool(data.get("cancelled", False)),
        )


@dataclass(frozen=True)
class StepSearchResult:
    """Search output associated with one workflow step."""

    step_id: str
    purpose: str
    query: str
    results: tuple[dict[str, Any], ...]


def create_workflow(
    plan: ResearchPlan,
) -> WorkflowState:
    """Create workflow state from a research plan."""

    return WorkflowState(
        original_query=plan.original_query,
        mode=plan.mode,
        plan=plan,
    )


def execute_workflow(
    state: WorkflowState,
    search_fn,
    max_results: int,
    cancellation_token: Any = None,
) -> tuple[StepSearchResult, ...]:
    """Execute dependency-ready research steps sequentially.

    ``search_fn`` is injected so workflow execution is easy to test
    without making real Tavily requests.
    """

    requested = max(
        int(max_results),
        1,
    )

    step_count = max(
        len(state.plan.steps),
        1,
    )

    per_step = max(
        1,
        (
            requested
            + step_count
            - 1
        )
        // step_count,
    )

    outputs: list[StepSearchResult] = []

    while not state.is_complete and not state.cancelled:
        if (
            cancellation_token is not None
            and getattr(cancellation_token, "is_cancelled", False)
        ):
            state.cancel()
            break
        step = state.current_step

        if step is None:
            break

        try:
            raw_results = search_fn(
                step.query,
                max_results=per_step,
            )

            results = tuple(
                item
                for item in raw_results
                if isinstance(
                    item,
                    dict,
                )
            )

            outputs.append(
                StepSearchResult(
                    step_id=step.step_id,
                    purpose=step.purpose,
                    query=step.query,
                    results=results,
                )
            )

            if results:
                state.mark_completed(
                    step.step_id
                )
            else:
                state.mark_failed(
                    step.step_id
                )
                break

        except Exception:
            state.mark_failed(
                step.step_id
            )
            raise

    return tuple(outputs)


def extract_research_topic(
    session: dict[str, Any],
) -> str:
    """Extract a stable topic from a saved session."""

    if not isinstance(
        session,
        dict,
    ):
        return ""

    workflow = session.get(
        "workflow"
    )

    if isinstance(
        workflow,
        dict,
    ):
        topic = str(
            workflow.get(
                "original_query",
                "",
            )
        ).strip()

        if topic:
            return topic.rstrip(
                "?!. "
            )

    topic = str(
        session.get("topic")
        or session.get("query")
        or ""
    ).strip()

    topic = topic.rstrip(
        "?!. "
    )

    if re.search(
        r"\b(?:vs\.?|versus|compare)\b",
        topic,
        flags=re.IGNORECASE,
    ):
        return topic

    patterns = (
        r"^what\s+is\s+(.+)$",
        r"^what\s+are\s+(.+)$",
        r"^who\s+is\s+(.+)$",
        r"^who\s+are\s+(.+)$",
        r"^how\s+does\s+(.+?)\s+work$",
        r"^how\s+do\s+(.+?)\s+work$",
        r"^explain\s+(.+)$",
        r"^tell\s+me\s+about\s+(.+)$",
    )

    for pattern in patterns:
        match = re.match(
            pattern,
            topic,
            flags=re.IGNORECASE,
        )

        if match:
            return match.group(
                1
            ).strip()

    return topic


def resolve_followup_query(
    question: str,
    topic: str,
) -> str:
    """Resolve contextual or short follow-ups against the active topic."""

    question = str(
        question or ""
    ).strip()

    topic = str(
        topic or ""
    ).strip()

    if not question:
        raise ValueError(
            "Follow-up question cannot be empty."
        )

    if not topic:
        return question

    lower = question.casefold()

    markers = (
        "it",
        "its",
        "they",
        "their",
        "this",
        "that",
        "these",
        "those",
        "the same",
        "above",
        "previous",
        "earlier",
    )

    needs_context = any(
        re.search(
            rf"\b{re.escape(marker)}\b",
            lower,
        )
        for marker in markers
    )

    short_question = (
        len(
            question.split()
        ) <= 10
    )

    if needs_context or short_question:
        return (
            f"{topic} {question}"
        ).strip()

    return question