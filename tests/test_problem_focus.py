import json
from pathlib import Path

import pytest

from app.problem_focus import (
    compile_problem_focus_targets,
    required_lead_emphasis,
)


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


def test_required_lead_emphasis_uses_first_of_multiple_inline_targets():
    targets = compile_problem_focus_targets(
        r"若\(2n\)是方程\(x^2-2mx+2n=0\)的根"
    )

    requirement = required_lead_emphasis(targets)

    assert requirement is not None
    assert requirement.target_id == "problem-math-001"
    assert requirement.spoken_token == "2n"


def test_required_lead_emphasis_accepts_simple_power_token():
    targets = compile_problem_focus_targets(
        r"已知\(x^2\)与\(x\)的关系"
    )

    requirement = required_lead_emphasis(targets)

    assert requirement is not None
    assert requirement.target_id == "problem-math-001"
    assert requirement.spoken_token == "x^2"


def test_required_lead_emphasis_skips_single_target():
    targets = compile_problem_focus_targets(r"解方程\(x^2-5x+6=0\)")

    assert required_lead_emphasis(targets) is None


def test_required_lead_emphasis_skips_display_first_target():
    targets = compile_problem_focus_targets(
        r"\[x^2-2mx+2n=0\]并且\(n\ne0\)"
    )

    assert len(targets) == 2
    assert required_lead_emphasis(targets) is None


@pytest.mark.parametrize(
    "source",
    (
        r"观察\(x^2-2mx+2n=0\)和\(2n\)",
        r"观察\(x^2-2mx+2n\)和\(2n\)",
        r"利用\(n\ne0\)与\(2n\)",
        r"比较\(n>0\)与\(2n\)",
        r"比较\(n+1\)与\(2n\)",
        r"比较\(n-1\)与\(2n\)",
        r"比较\(abcdefghijklm\)与\(2n\)",
    ),
    ids=(
        "whole-equation",
        "polynomial",
        "latex-relation",
        "plain-relation",
        "addition",
        "subtraction",
        "over-12-codepoints",
    ),
)
def test_required_lead_emphasis_rejects_non_atomic_first_target(source):
    targets = compile_problem_focus_targets(source)

    assert len(targets) == 2
    assert required_lead_emphasis(targets) is None


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
