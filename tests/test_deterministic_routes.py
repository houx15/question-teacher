import asyncio
import copy
import json

import pytest

from app.generation import LessonGenerationService, LessonQualityError
from app.math_engine import MathEngine
from app.prompts import (
    DIRECTOR_SYSTEM,
    MATERIALS_SYSTEM,
    MATH_ROUTE_SYSTEM,
    REVIEWER_SYSTEM,
)
from app.schemas import ProblemInput
from app.schemas import MathRouteDraft
from tests.generation_fakes import FakeClient
from tests.test_generation import approved_review
from tests.test_generation_agents import materials_payload, narrative_payload


def complete_square_problem() -> ProblemInput:
    return ProblemInput(
        problem_text="用配方法解方程：x^2-6x+5=0",
        reference_answer="x=1 或 x=5",
        required_method="complete_the_square",
    )


def complete_square_narrative():
    payload = narrative_payload()
    payload["method_introduction"] = {
        "method_name": "配方法",
        "student_definition": "把含未知数的部分凑成一个完全平方。",
        "target_form": r"\((x-3)^2=4\)",
        "why_it_helps": "开平方后就能得到两个一次方程。",
    }
    return payload


def test_complete_square_demo_uses_deterministic_verified_route():
    client = FakeClient(
        [
            complete_square_narrative(),
            materials_payload(),
            approved_review(),
        ]
    )

    lesson = asyncio.run(
        LessonGenerationService(client, MathEngine()).generate(
            complete_square_problem()
        )
    )

    assert client.route_calls == []
    assert [call[0] for call in client.all_calls] == [
        DIRECTOR_SYSTEM,
        MATERIALS_SYSTEM,
        REVIEWER_SYSTEM,
    ]
    verified = LessonGenerationService(
        FakeClient([]),
        MathEngine(),
    )._create_deterministic_route(complete_square_problem())
    assert verified is not None
    assert [
        step.operation for step in verified.thaw().math_steps
    ] == [
        "subtract_both_sides",
        "complete_the_square",
        "take_square_root_both_sides",
        "subtract_both_sides",
    ]
    assert verified.thaw().math_steps[-1].state_after == ["x=1", "x=5"]
    assert lesson.validation_report["math_route_source"] == "deterministic"
    assert (
        lesson.validation_report["math_route_method_family"]
        == "complete_the_square"
    )


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


def test_factor_route_is_explicitly_unsupported_and_falls_back_to_agent():
    source = ProblemInput(
        problem_text="用因式分解法解方程：x^2-5*x+6=0",
        reference_answer="x=2 或 x=3",
        required_method="factor",
    )
    route_payload = {
        "math_steps": [
            {
                "purpose": "因式分解",
                "operation": "factor",
                "operands": [],
                "state_before": ["x^2-5*x+6=0"],
                "state_after": ["(x-2)*(x-3)=0"],
                "reason": "把二次式写成两个一次因式的乘积。",
            }
        ]
    }
    client = FakeClient(
        [
            route_payload,
            narrative_payload(),
            materials_payload(),
            approved_review(),
        ]
    )

    lesson = asyncio.run(
        LessonGenerationService(client, MathEngine()).generate(source)
    )

    assert len(client.route_calls) == 1
    assert client.route_calls[0][0] == MATH_ROUTE_SYSTEM
    assert lesson.validation_report["math_route_source"] == "agent"


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

    client = FakeClient([copy.deepcopy(materials_payload())])
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


def test_unspecified_quadratic_resolved_method_reaches_all_teaching_agents():
    source = ProblemInput(
        problem_text="解方程：x^2-5*x+6=0",
        reference_answer="x=2 或 x=3",
    )
    narrative = narrative_payload()
    narrative["method_introduction"]["method_name"] = "公式法"
    client = FakeClient(
        [narrative, materials_payload(), approved_review()]
    )

    asyncio.run(
        LessonGenerationService(client, MathEngine()).generate(source)
    )

    director = json.loads(client.all_calls[0][1])
    materials = json.loads(client.all_calls[1][1])
    reviewer = json.loads(client.all_calls[2][1])
    assert director["resolved_method"] == {
        "family": "quadratic_formula",
        "display_name": "公式法",
    }
    assert materials["resolved_method"] == director["resolved_method"]
    assert reviewer["resolved_method"] == director["resolved_method"]
    assert materials["output_contract"]["transfer_item"][
        "method_profile"
    ]["required_method"] == "quadratic_formula"


@pytest.mark.parametrize("wrong_method", ["因式分解法", "配方法"])
def test_resolved_formula_route_rejects_other_method_narrative(
    wrong_method,
):
    source = ProblemInput(
        problem_text="解方程：x^2-5*x+6=0",
        reference_answer="x=2 或 x=3",
    )
    invalid = narrative_payload()
    invalid["method_introduction"]["method_name"] = wrong_method
    client = FakeClient([invalid, copy.deepcopy(invalid)])

    with pytest.raises(
        LessonQualityError,
        match="已验证数学路线",
    ):
        asyncio.run(
            LessonGenerationService(client, MathEngine()).generate(source)
        )

    assert [call[0] for call in client.all_calls] == [
        DIRECTOR_SYSTEM,
        DIRECTOR_SYSTEM,
    ]


def test_unspecified_linear_resolved_method_reaches_reviewer_and_transfer():
    source = ProblemInput(
        problem_text="解方程：2*x+3=7",
        reference_answer="x=2",
    )
    narrative = narrative_payload()
    narrative["method_introduction"] = {
        "method_name": "等式基本变形",
        "student_definition": "等式两边做同一种运算，等式仍然成立。",
        "target_form": r"\(x=2\)",
        "why_it_helps": "逐步把未知数单独留在一边。",
    }
    client = FakeClient(
        [narrative, materials_payload(), approved_review()]
    )

    asyncio.run(
        LessonGenerationService(client, MathEngine()).generate(source)
    )

    materials = json.loads(client.all_calls[1][1])
    reviewer = json.loads(client.all_calls[2][1])
    profile = materials["output_contract"]["transfer_item"][
        "method_profile"
    ]
    assert profile["resolved_method_family"] == "basic_equation_operations"
    assert profile["required_method"] is None
    assert profile["equation_template"] == "a*x+b=0"
    assert reviewer["resolved_method"] == {
        "family": "basic_equation_operations",
        "display_name": "等式基本变形",
    }
