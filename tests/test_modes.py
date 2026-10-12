from research.modes import MODES, ResearchMode, get_mode


def test_all_expected_modes_exist():
    assert "quick" in MODES
    assert "deep" in MODES
    assert "compare" in MODES


def test_modes_are_research_mode_instances():
    for mode in MODES.values():
        assert isinstance(mode, ResearchMode)


def test_mode_source_counts_are_positive():
    for mode in MODES.values():
        assert mode.max_results >= 1


def test_get_mode_returns_requested_mode():
    assert get_mode("quick") is MODES["quick"]
    assert get_mode("deep") is MODES["deep"]
    assert get_mode("compare") is MODES["compare"]


def test_get_mode_normalizes_input():
    assert get_mode(" QUICK ") is MODES["quick"]
    assert get_mode("Deep") is MODES["deep"]


def test_unknown_mode_falls_back_to_quick():
    assert get_mode("not-a-real-mode") is MODES["quick"]
    assert get_mode("") is MODES["quick"]
    assert get_mode(None) is MODES["quick"]