import json
from pathlib import Path

import pytest

from app.problem_focus import compile_problem_focus_targets


FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "problem-focus-cases.json"
)


def problem_focus_cases():
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "case",
    problem_focus_cases(),
    ids=lambda case: case["name"],
)
def test_problem_focus_targets_match_shared_explicit_delimiter_cases(case):
    if "code_points" in case:
        assert [ord(character) for character in case["source"]] == (
            case["code_points"]
        )

    targets = compile_problem_focus_targets(case["source"])

    assert [target.math_text for target in targets] == case["math"]
    assert [target.target_id for target in targets] == [
        f"problem-math-{index:03d}"
        for index in range(1, len(case["math"]) + 1)
    ]
    if "display_modes" in case:
        assert [
            target.display_mode for target in targets
        ] == case["display_modes"]


def test_problem_focus_targets_include_display_mode_and_stable_ordinal():
    targets = compile_problem_focus_targets(
        r"先看\(x=2\)，再看\[\frac{1}{2}\]"
    )

    assert [
        (target.display_mode, target.ordinal)
        for target in targets
    ] == [(False, 1), (True, 2)]


def test_problem_focus_targets_reject_over_rendering_budget():
    source = "$x$" + ("a" * 4094)

    assert len(source) == 4097
    assert compile_problem_focus_targets(source) == []


def test_problem_focus_targets_count_unicode_code_points_for_budget():
    within_budget = ("😀" * 4091) + r"\(x\)"
    over_budget = ("😀" * 4092) + r"\(x\)"

    assert len(within_budget) == 4096
    assert [
        target.math_text
        for target in compile_problem_focus_targets(within_budget)
    ] == ["x"]
    assert len(over_budget) == 4097
    assert compile_problem_focus_targets(over_budget) == []


def test_problem_focus_targets_enforce_64_target_boundary():
    at_limit = r"\(x=1\)" * 64
    over_limit = r"\(x=1\)" * 65

    targets = compile_problem_focus_targets(at_limit)

    assert len(targets) == 64
    assert targets[-1].target_id == "problem-math-064"
    assert targets[-1].ordinal == 64
    assert compile_problem_focus_targets(over_limit) == []


@pytest.mark.parametrize(
    "source",
    [
        r"混合$x=1\)",
        r"混合\(x=1$",
        r"多余结束符号\(x=1\)\]",
    ],
)
def test_problem_focus_targets_reject_mixed_or_unmatched_delimiters(source):
    assert compile_problem_focus_targets(source) == []
