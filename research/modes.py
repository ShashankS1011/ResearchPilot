from dataclasses import dataclass

from .config import SETTINGS


@dataclass(frozen=True)
class ResearchMode:
    name: str
    description: str
    max_results: int
    follow_up_enabled: bool = True


MODES = {
    "quick": ResearchMode(
        name="Quick Research",
        description=(
            "Fast research using a small "
            "number of sources."
        ),
        max_results=SETTINGS.quick_sources,
        follow_up_enabled=True,
    ),

    "deep": ResearchMode(
        name="Deep Research",
        description=(
            "Broader research using more "
            "sources and evidence."
        ),
        max_results=SETTINGS.deep_sources,
        follow_up_enabled=True,
    ),

    "compare": ResearchMode(
        name="Compare Topics",
        description=(
            "Research a comparison question "
            "using multiple sources."
        ),
        max_results=SETTINGS.compare_sources,
        follow_up_enabled=True,
    ),
}


def get_mode(
    mode_name,
):
    """
    Return a configured research mode.

    Unknown modes fall back to quick research.
    """

    mode_name = str(
        mode_name or ""
    ).strip().lower()

    return MODES.get(
        mode_name,
        MODES["quick"],
    )


def show_modes():
    """
    Display available research modes.
    """

    print(
        "\n" + "=" * 70
    )

    print(
        "RESEARCH MODES"
    )

    print(
        "=" * 70
    )

    for index, mode_name in enumerate(
        (
            "quick",
            "deep",
            "compare",
        ),
        start=1,
    ):
        mode = MODES[mode_name]

        print(
            f"\n{index}. {mode.name}"
        )

        print(
            f"   {mode.description}"
        )

        print(
            f"   Sources: {mode.max_results}"
        )

    print(
        "\n4. Follow-up"
    )

    print(
        "   Continue the current "
        "research session."
    )

    print(
        "\n5. Exit"
    )