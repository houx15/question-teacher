import asyncio
import copy
import json

import pytest
from pydantic import ValidationError

from app.generation import (
    LessonGenerationService,
    LessonQualityError,
    _RouteValidationError,
)
from app.llm_client import ModelResponseError
from app.math_engine import MathEngine
from app.prompts import MATH_ROUTE_SYSTEM, math_route_prompt
from app.schemas import MathRouteDraft
from tests.test_generation import problem, valid_draft


def route_payload(source=None):
    return {
        "math_steps": copy.deepcopy(
            (source or valid_draft())["math_steps"]
        )
    }


class AgentClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    async def complete_json(self, system, user):
        self.calls.append((system, user))
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return copy.deepcopy(response)


class UnsupportedDeterministicPlanner:
    def plan(self, problem, equation_degree, solution_strings):
        return None


def agent_service(client, math_engine=None):
    return LessonGenerationService(
        client,
        math_engine or MathEngine(),
        deterministic_route_planner=UnsupportedDeterministicPlanner(),
    )


def create_route(service, source=None, solutions=None, degree=2):
    return asyncio.run(
        service._create_validated_route(
            source or problem(),
            solutions or ["2", "3"],
            degree,
        )
    )


def test_math_route_schema_contains_only_bounded_math_steps():
    schema = MathRouteDraft.model_json_schema()

    assert set(schema["properties"]) == {"math_steps"}
    assert schema["properties"]["math_steps"]["minItems"] == 1
    assert schema["properties"]["math_steps"]["maxItems"] == 16
    with pytest.raises(ValidationError):
        MathRouteDraft.model_validate(
            {**route_payload(), "teaching_assets": ["private"]}
        )


def test_route_prompt_is_minimal_private_and_typed():
    raw_marker = "RAW_REFERENCE_PRIVATE_MARKER"
    source = problem(reference_solution_text=raw_marker)
    payload = json.loads(
        math_route_prompt(
            source,
            ["2", "3"],
            equation_degree=2,
            previous_validation_code="route_step_invalid",
        )
    )
    serialized = json.dumps(payload, ensure_ascii=False)

    assert payload["problem"] == {"problem_text": source.problem_text}
    assert payload["independent_solutions"] == ["2", "3"]
    assert payload["previous_validation_code"] == "route_step_invalid"
    assert payload["output_contract"]["schema"] == (
        MathRouteDraft.model_json_schema()
    )
    assert raw_marker not in serialized
    assert "reference_solution_text" not in serialized
    assert "route_step_invalid" in MATH_ROUTE_SYSTEM


def test_factor_route_contract_allows_only_verified_factored_terminal():
    assert "因式分解方法族" in MATH_ROUTE_SYSTEM
    assert "因式乘积方程" in MATH_ROUTE_SYSTEM
    state_rules = " ".join(
        json.loads(
            math_route_prompt(problem(), ["2", "3"], equation_degree=2)
        )["output_contract"]["operation_contract"]["state_rules"]
    )
    assert "factor" in state_rules
    assert "verified factored product equation" in state_rules


def test_factor_route_must_end_at_the_verified_factor_operation():
    route = route_payload()
    route["math_steps"].append(
        {
            "purpose": "再做一次等价变形",
            "operation": "multiply_both_sides",
            "operands": ["2"],
            "state_before": ["(x-2)(x-3)=0"],
            "state_after": ["2*(x-2)(x-3)=0"],
            "reason": "等式两边同时乘二。",
        }
    )

    with pytest.raises(_RouteValidationError):
        agent_service(AgentClient([]))._validate_math_route_draft(
            problem(),
            MathRouteDraft.model_validate(route),
            equation_degree=2,
        )


def test_route_rejects_an_unrelated_first_state_with_typed_code():
    unrelated = route_payload()
    unrelated["math_steps"][0].update(
        {
            "state_before": ["x^2-9=0"],
            "state_after": ["(x-3)*(x+3)=0"],
        }
    )

    with pytest.raises(_RouteValidationError) as captured:
        agent_service(AgentClient([]))._validate_math_route_draft(
            problem(),
            MathRouteDraft.model_validate(unrelated),
            equation_degree=2,
        )

    assert captured.value.code == "route_first_state_mismatch"


def test_route_rejects_disconnected_consecutive_states_with_typed_code():
    disconnected = {
        "math_steps": [
            {
                "purpose": "两边减六",
                "operation": "subtract_both_sides",
                "operands": ["6"],
                "state_before": ["x^2-5*x+6=0"],
                "state_after": ["x^2-5*x=-6"],
                "reason": "等式两边同时减六。",
            },
            route_payload()["math_steps"][0],
        ]
    }

    with pytest.raises(_RouteValidationError) as captured:
        agent_service(AgentClient([]))._validate_math_route_draft(
            problem(),
            MathRouteDraft.model_validate(disconnected),
            equation_degree=2,
        )

    assert captured.value.code == "route_disconnected"


def test_route_rejects_a_final_solution_mismatch_with_typed_code():
    class PermissiveStepMathEngine(MathEngine):
        def validate_step(self, step):
            del step

    mismatch = route_payload()
    mismatch["math_steps"][0]["state_after"] = ["x=99"]

    with pytest.raises(_RouteValidationError) as captured:
        agent_service(
            AgentClient([]),
            math_engine=PermissiveStepMathEngine(),
        )._validate_math_route_draft(
            problem(),
            MathRouteDraft.model_validate(mismatch),
            equation_degree=2,
        )

    assert captured.value.code == "route_final_solution_mismatch"


def test_route_accepts_contiguous_states_after_notation_normalization():
    contiguous = {
        "math_steps": [
            {
                "purpose": "两边减六",
                "operation": "subtract_both_sides",
                "operands": ["6"],
                "state_before": ["x^2 - 5*x + 6 = 0"],
                "state_after": ["x^2-5*x=-6"],
                "reason": "等式两边同时减六。",
            },
            {
                "purpose": "两边加六",
                "operation": "add_both_sides",
                "operands": ["6"],
                "state_before": ["x² − 5x = -6"],
                "state_after": ["x^2-5*x+6=0"],
                "reason": "等式两边同时加六。",
            },
            route_payload()["math_steps"][0],
        ]
    }

    method = agent_service(AgentClient([]))._validate_math_route_draft(
        problem(),
        MathRouteDraft.model_validate(contiguous),
        equation_degree=2,
    )

    assert method == "factor"


@pytest.mark.parametrize(
    ("source", "route", "expected_method"),
    [
        (
            problem(required_method="quadratic_formula"),
            {
                "math_steps": [
                    {
                        "purpose": "使用求根公式",
                        "operation": "quadratic_formula",
                        "operands": [],
                        "state_before": ["x^2-5*x+6=0"],
                        "state_after": ["x=3", "x=2"],
                        "reason": "代入求根公式得到两个根。",
                    }
                ]
            },
            "quadratic_formula",
        ),
        (
            problem(required_method="quadratic_formula").model_copy(
                update={
                    "problem_text": "用求根公式解方程：x^2+1=0",
                    "reference_answer": "无实数解",
                }
            ),
            {
                "math_steps": [
                    {
                        "purpose": "使用求根公式",
                        "operation": "quadratic_formula",
                        "operands": [],
                        "state_before": ["x^2+1=0"],
                        "state_after": ["无实数解"],
                        "reason": "判别式小于零。",
                    }
                ]
            },
            "quadratic_formula",
        ),
    ],
)
def test_route_accepts_complete_multi_branch_and_empty_real_solution_sets(
    source,
    route,
    expected_method,
):
    method = agent_service(AgentClient([]))._validate_math_route_draft(
        source,
        MathRouteDraft.model_validate(route),
        equation_degree=2,
    )

    assert method == expected_method


def test_route_requires_the_explicitly_requested_method_family():
    with pytest.raises(_RouteValidationError) as captured:
        agent_service(AgentClient([]))._validate_math_route_draft(
            problem(required_method="complete_the_square"),
            MathRouteDraft.model_validate(route_payload()),
            equation_degree=2,
        )

    assert captured.value.code == "route_required_method_missing"


def test_route_retries_one_logical_validation_failure_with_typed_code():
    invalid = route_payload()
    invalid["math_steps"][0]["state_after"] = ["(x-1)*(x-6)=0"]
    client = AgentClient([invalid, route_payload()])

    verified = create_route(agent_service(client))

    assert verified.method_family == "factor"
    assert [system for system, _ in client.calls] == [
        MATH_ROUTE_SYSTEM,
        MATH_ROUTE_SYSTEM,
    ]
    retry = json.loads(client.calls[1][1])
    assert retry["previous_validation_code"] == "route_step_invalid"
    assert "(x-1)" not in client.calls[1][1]


def test_route_schema_failure_uses_typed_retry_and_stops_after_two_attempts():
    invalid = {"math_steps": "private-bad-value", "private": "secret"}
    client = AgentClient([invalid, invalid])

    with pytest.raises(LessonQualityError, match="数学路线结构无效"):
        create_route(agent_service(client))

    assert len(client.calls) == 2
    retry = json.loads(client.calls[1][1])
    assert retry["previous_validation_code"] == "route_schema_invalid"
    assert "private-bad-value" not in client.calls[1][1]


def test_route_provider_errors_are_retried_and_final_identity_is_preserved():
    first = ModelResponseError("temporary")
    final = ModelResponseError("provider-safe-category")
    client = AgentClient([first, final])

    with pytest.raises(ModelResponseError) as caught:
        create_route(agent_service(client))

    assert caught.value is final
    assert len(client.calls) == 2
    assert client.calls[0] == client.calls[1]


def test_route_cancellation_is_not_retried_or_reclassified():
    cancellation = asyncio.CancelledError("cancel route")
    client = AgentClient([cancellation])

    with pytest.raises(asyncio.CancelledError):
        create_route(agent_service(client))

    assert len(client.calls) == 1


def test_unrequested_quadratic_route_rejects_mixed_method_families():
    source = problem(required_method=None)
    mixed = route_payload()
    mixed["math_steps"].append(
        {
            "purpose": "再套公式",
            "operation": "quadratic_formula",
            "operands": [],
            "state_before": ["(x-2)(x-3)=0"],
            "state_after": ["x=2", "x=3"],
            "reason": "错误地混用另一方法族。",
        }
    )
    client = AgentClient([mixed, copy.deepcopy(mixed)])

    with pytest.raises(LessonQualityError, match="数学路线未通过验证"):
        create_route(agent_service(client), source=source)

    assert json.loads(client.calls[1][1])[
        "previous_validation_code"
    ] == "route_method_family_conflict"


def test_linear_route_allows_basic_equation_operations():
    source = problem().model_copy(
        update={
            "problem_text": "解方程：2*x+3=7",
            "reference_answer": "x=2",
            "required_method": None,
        }
    )
    route = {
        "math_steps": [
            {
                "purpose": "移去常数项",
                "operation": "subtract_both_sides",
                "operands": ["3"],
                "state_before": ["2*x+3=7"],
                "state_after": ["2*x=4"],
                "reason": "等式两边同时减去三。",
            },
            {
                "purpose": "求出未知数",
                "operation": "divide_both_sides",
                "operands": ["2"],
                "state_before": ["2*x=4"],
                "state_after": ["x=2"],
                "reason": "等式两边同时除以二。",
            },
        ]
    }
    verified = create_route(
        agent_service(AgentClient([route])),
        source=source,
        solutions=["2"],
        degree=1,
    )

    assert verified.method_family == "basic_equation_operations"


def test_linear_route_rejects_a_nonbasic_operation_before_step_validation():
    source = problem().model_copy(
        update={
            "problem_text": "解方程：2*x+3=7",
            "reference_answer": "x=2",
            "required_method": None,
        }
    )
    invalid = {
        "math_steps": [
            {
                "purpose": "错误地开平方",
                "operation": "take_square_root_both_sides",
                "operands": [],
                "state_before": ["2*x+3=7"],
                "state_after": ["x=2"],
                "reason": "一次方程不应使用开平方。",
            }
        ]
    }

    with pytest.raises(_RouteValidationError) as captured:
        agent_service(AgentClient([]))._validate_math_route_draft(
            source,
            MathRouteDraft.model_validate(invalid),
            equation_degree=1,
        )

    assert captured.value.code == "route_method_family_conflict"
