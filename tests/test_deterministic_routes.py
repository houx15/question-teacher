import asyncio

import pytest

from app.generation import LessonGenerationService, LessonQualityError
from app.math_engine import MathEngine
from app.schemas import ProblemInput
from app.schemas import MathRouteDraft
from tests.generation_fakes import FakeClient


def complete_square_problem() -> ProblemInput:
    return ProblemInput(
        problem_text="用配方法解方程：x^2-6x+5=0",
        reference_answer="x=1 或 x=5",
        required_method="complete_the_square",
    )



def test_complete_square_repeated_root_uses_one_zero_branch_and_exact_reason():
    source = ProblemInput(
        problem_text="用配方法解方程：x^2-6*x+9=0",
        reference_answer="x=3",
        required_method="complete_the_square",
    )
    verified = LessonGenerationService(
        FakeClient([]),
        MathEngine(),
    )._create_deterministic_route(source)

    assert verified is not None
    root_step = next(
        step
        for step in verified.thaw().math_steps
        if step.operation == "take_square_root_both_sides"
    )
    assert root_step.state_after == ["x - 3=0"]
    assert "只能等于 0" in root_step.reason
    assert "正、负" not in root_step.reason


@pytest.mark.parametrize(
    ("source", "expected_family", "expected_operations"),
    [
        (
            ProblemInput(
                problem_text="用公式法解方程：2*x^2-3*x-2=0",
                reference_answer="x=-1/2 或 x=2",
                required_method="quadratic_formula",
            ),
            "quadratic_formula",
            ["quadratic_formula"],
        ),
        (
            ProblemInput(
                problem_text="解方程：x^2-5*x+6=0",
                reference_answer="x=2 或 x=3",
            ),
            "quadratic_formula",
            ["quadratic_formula"],
        ),
        (
            ProblemInput(
                problem_text="解方程：2*x+3=7",
                reference_answer="x=2",
            ),
            "basic_equation_operations",
            ["subtract_both_sides", "divide_both_sides"],
        ),
    ],
)
def test_deterministic_route_families_pass_existing_hard_validation(
    source,
    expected_family,
    expected_operations,
):
    service = LessonGenerationService(FakeClient([]), MathEngine())

    route = service._create_deterministic_route(source)

    assert route is not None
    assert route.method_family == expected_family
    assert route.source == "deterministic"
    assert [
        step.operation for step in route.thaw().math_steps
    ] == expected_operations
    assert (
        service._validate_math_route_draft(
            source,
            route.thaw(),
            MathEngine().validate_problem(
                source.problem_text,
                source.reference_answer,
            ).equation_degree,
        )
        == expected_family
    )



def test_invalid_deterministic_candidate_fails_closed_without_agent_fallback():
    class InvalidDeterministicPlanner:
        def plan(self, problem, equation_degree, solution_strings):
            return MathRouteDraft.model_validate(
                {
                    "math_steps": [
                        {
                            "purpose": "错误路线",
                            "operation": "complete_the_square",
                            "operands": ["9"],
                            "state_before": ["x^2-6*x+5=0"],
                            "state_after": ["(x-3)^2=99"],
                            "reason": "故意构造一个不能通过硬校验的候选。",
                        }
                    ]
                }
            )

    client = FakeClient([])
    service = LessonGenerationService(
        client,
        MathEngine(),
        deterministic_route_planner=InvalidDeterministicPlanner(),
    )

    with pytest.raises(LessonQualityError, match="数学路线未通过验证"):
        asyncio.run(service.generate(complete_square_problem()))

    assert client.route_calls == []


def test_verified_route_thaw_is_deep_copy_and_fingerprint_protected():
    service = LessonGenerationService(FakeClient([]), MathEngine())
    verified = service._create_deterministic_route(
        complete_square_problem()
    )
    assert verified is not None

    first = verified.thaw()
    first.math_steps[0].state_after[0] = "x=999"
    second = verified.thaw()

    assert second.math_steps[0].state_after != ["x=999"]
    assert verified.fingerprint
