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
from app.prompts import (
    DIRECTOR_SYSTEM,
    MATERIALS_SYSTEM,
    MATH_ROUTE_SYSTEM,
    REFERENCE_AUDITOR_SYSTEM,
    REVIEWER_SYSTEM,
    REVISION_SYSTEM,
    math_route_prompt,
)
from app.schemas import MathRouteDraft, NarrativeDraft
from tests.test_generation import (
    approved_audit,
    approved_review,
    problem,
    revision_review,
    valid_draft,
)
from tests.test_generation_agents import materials_payload, narrative_payload


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

    async def complete_json(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
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


def test_math_route_schema_contains_only_bounded_math_steps():
    schema = MathRouteDraft.model_json_schema()

    assert set(schema["properties"]) == {"math_steps"}
    assert schema["properties"]["math_steps"]["minItems"] == 1
    assert schema["properties"]["math_steps"]["maxItems"] == 16
    with pytest.raises(ValidationError):
        MathRouteDraft.model_validate(
            {**route_payload(), "teaching_assets": ["private"]}
        )


def test_narrative_schema_cannot_generate_math_steps():
    schema = NarrativeDraft.model_json_schema()

    assert "math_steps" not in schema["properties"]
    invalid = narrative_payload()
    invalid["math_steps"] = route_payload()["math_steps"]
    with pytest.raises(ValidationError):
        NarrativeDraft.model_validate(invalid)


def test_route_prompt_is_minimal_private_and_typed():
    raw_marker = "RAW_REFERENCE_PRIVATE_MARKER"
    audit = approved_audit()
    audit["teaching_assets"] = ["PRIVATE_TEACHING_ASSET"]
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
    assert payload["equation_degree"] == 2
    assert payload["required_method"] == "factor"
    assert payload["previous_validation_code"] == "route_step_invalid"
    assert payload["output_contract"]["schema"] == (
        MathRouteDraft.model_json_schema()
    )
    assert raw_marker not in serialized
    assert "PRIVATE_TEACHING_ASSET" not in serialized
    assert "reference_solution_text" not in serialized
    assert "teaching_assets" not in serialized
    assert "route_step_invalid" in MATH_ROUTE_SYSTEM
    state_rules = " ".join(
        payload["output_contract"]["operation_contract"]["state_rules"]
    )
    assert "positive" in state_rules and "two explicit branches" in state_rules
    assert "zero" in state_rules and "one explicit zero branch" in state_rules


def test_factor_route_contract_allows_only_verified_factored_terminal():
    assert "因式分解方法族" in MATH_ROUTE_SYSTEM
    assert "因式乘积方程" in MATH_ROUTE_SYSTEM
    assert "MathEngine" in MATH_ROUTE_SYSTEM
    assert "independent_solutions" in DIRECTOR_SYSTEM
    assert "零乘积性质" in DIRECTOR_SYSTEM
    assert "independent_solutions" in REVIEWER_SYSTEM
    assert "零乘积性质" in REVIEWER_SYSTEM
    payload = json.loads(
        math_route_prompt(problem(), ["2", "3"], equation_degree=2)
    )
    state_rules = " ".join(
        payload["output_contract"]["operation_contract"]["state_rules"]
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
    service = agent_service(AgentClient([]))

    with pytest.raises(
        _RouteValidationError,
        match="数学路线未通过验证",
    ):
        service._validate_math_route_draft(
            problem(),
            MathRouteDraft.model_validate(route),
            equation_degree=2,
        )


def test_successful_agent_pipeline_has_explicit_call_order():
    client = AgentClient(
        [
            route_payload(),
            narrative_without_route(),
            materials_payload(),
            approved_review(),
        ]
    )

    lesson = asyncio.run(
        agent_service(client).generate(problem())
    )

    assert [system for system, _ in client.calls] == [
        MATH_ROUTE_SYSTEM,
        DIRECTOR_SYSTEM,
        MATERIALS_SYSTEM,
        REVIEWER_SYSTEM,
    ]
    assert lesson.validation_report["math_route_status"] == "verified"


def test_reference_audit_precedes_route_and_raw_text_stays_in_auditor():
    raw_marker = "RAW_REFERENCE_PRIVATE_MARKER"
    client = AgentClient(
        [
            approved_audit(),
            route_payload(),
            narrative_without_route(),
            materials_payload(),
            approved_review(),
        ]
    )

    asyncio.run(
        agent_service(client).generate(
            problem(reference_solution_text=raw_marker)
        )
    )

    assert [system for system, _ in client.calls] == [
        REFERENCE_AUDITOR_SYSTEM,
        MATH_ROUTE_SYSTEM,
        DIRECTOR_SYSTEM,
        MATERIALS_SYSTEM,
        REVIEWER_SYSTEM,
    ]
    assert raw_marker in client.calls[0][1]
    assert all(raw_marker not in prompt for _, prompt in client.calls[1:])


def test_route_retries_one_logical_validation_failure_with_typed_code():
    invalid = route_payload()
    invalid["math_steps"][0]["state_after"] = ["(x-1)*(x-6)=0"]
    client = AgentClient(
        [
            invalid,
            route_payload(),
            narrative_without_route(),
            materials_payload(),
            approved_review(),
        ]
    )

    asyncio.run(
        agent_service(client).generate(problem())
    )

    assert [system for system, _ in client.calls[:2]] == [
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
        asyncio.run(
            agent_service(client).generate(problem())
        )

    assert len(client.calls) == 2
    retry = json.loads(client.calls[1][1])
    assert retry["previous_validation_code"] == "route_schema_invalid"
    assert "private-bad-value" not in client.calls[1][1]
    assert "secret" not in client.calls[1][1]


def test_route_provider_errors_are_retried_and_final_identity_is_preserved():
    first = ModelResponseError("temporary")
    final = ModelResponseError("provider-safe-category")
    client = AgentClient([first, final])

    with pytest.raises(ModelResponseError) as caught:
        asyncio.run(
            agent_service(client).generate(problem())
        )

    assert caught.value is final
    assert len(client.calls) == 2
    assert client.calls[0] == client.calls[1]


def test_route_cancellation_is_not_retried_or_reclassified():
    cancellation = asyncio.CancelledError("cancel route")
    client = AgentClient([cancellation])

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            agent_service(client).generate(problem())
        )

    assert len(client.calls) == 1


def test_revision_never_regenerates_or_accepts_math_route():
    revised = narrative_without_route()
    revised["opening"] = "先观察条件，再选择因数分解的切入点。"
    client = AgentClient(
        [
            route_payload(),
            narrative_without_route(),
            materials_payload(),
            revision_review(),
            revised,
            materials_payload(),
            approved_review(),
        ]
    )

    asyncio.run(
        agent_service(client).generate(problem())
    )

    assert [system for system, _ in client.calls] == [
        MATH_ROUTE_SYSTEM,
        DIRECTOR_SYSTEM,
        MATERIALS_SYSTEM,
        REVIEWER_SYSTEM,
        REVISION_SYSTEM,
        MATERIALS_SYSTEM,
        REVIEWER_SYSTEM,
    ]
    revision = json.loads(client.calls[4][1])
    assert "verified_math_route" in revision
    assert "math_steps" not in revision["output_contract"]["schema"][
        "properties"
    ]


def test_route_is_deep_copied_and_injected_after_agents_finish():
    route_source = route_payload()
    client = AgentClient(
        [
            route_source,
            narrative_without_route(),
            materials_payload(),
            approved_review(),
        ]
    )
    service = agent_service(client)

    lesson = asyncio.run(service.generate(problem()))
    reviewer = json.loads(client.calls[-1][1])
    reviewer["whole_lesson"]["math_steps"][0]["state_after"][0] = "x=99"

    assert lesson.validation_report["math_route_fingerprint"]
    assert route_source["math_steps"][0]["state_after"] == [
        "(x-2)(x-3)=0"
    ]


def test_unrequested_quadratic_route_derives_exactly_one_method_family():
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
        asyncio.run(
            agent_service(client).generate(source)
        )

    retry = json.loads(client.calls[1][1])
    assert retry["previous_validation_code"] == (
        "route_method_family_conflict"
    )


def test_linear_route_allows_basic_equation_operations_without_named_family():
    source = problem_linear()
    route = {
        "math_steps": [
            {
                "purpose": "移去常数项",
                "operation": "subtract_both_sides",
                "operands": ["3"],
                "state_before": ["2*x+3=7"],
                "state_after": ["2*x=4"],
                "reason": "等式两边同时减去 3。",
            },
            {
                "purpose": "求出未知数",
                "operation": "divide_both_sides",
                "operands": ["2"],
                "state_before": ["2*x=4"],
                "state_after": ["x=2"],
                "reason": "等式两边同时除以 2。",
            },
        ]
    }
    client = AgentClient(
        [
            route,
            narrative_without_route(method="等式基本变形"),
            linear_materials(),
            approved_review(),
        ]
    )

    lesson = asyncio.run(
        agent_service(client).generate(source)
    )

    assert lesson.validation_report["math_route_method_family"] == (
        "basic_equation_operations"
    )


def test_linear_route_rejects_nonbasic_operation_before_step_validation():
    source = problem_linear()
    invalid = {
        "math_steps": [
            {
                "purpose": "错误地开平方",
                "operation": "take_square_root_both_sides",
                "operands": [],
                "state_before": ["2*x+3=7"],
                "state_after": ["x=2"],
                "reason": "一次方程不应使用开平方操作。",
            }
        ]
    }
    client = AgentClient([invalid, copy.deepcopy(invalid)])

    with pytest.raises(LessonQualityError, match="数学路线未通过验证"):
        asyncio.run(
            agent_service(client).generate(source)
        )

    retry = json.loads(client.calls[1][1])
    assert retry["previous_validation_code"] == (
        "route_method_family_conflict"
    )


def test_complete_square_route_fixture_passes_required_method_gate():
    source = complete_square_problem()
    route = complete_square_route()
    narrative = narrative_without_route(method="配方法")
    narrative["method_introduction"] = {
        "method_name": "配方法",
        "student_definition": "把二次式整理成完全平方，再利用平方关系求解。",
        "target_form": r"\((x-a)^2=b\)",
        "why_it_helps": "平方形式能直接连接到两个一次方程。",
    }
    client = AgentClient(
        [
            route,
            narrative,
            complete_square_materials(),
            approved_review(),
        ]
    )

    lesson = asyncio.run(
        agent_service(client).generate(source)
    )

    assert lesson.validation_report["math_route_method_family"] == (
        "complete_the_square"
    )


def narrative_without_route(method="因式分解法"):
    payload = narrative_payload()
    payload.pop("math_steps", None)
    payload["method_introduction"]["method_name"] = method
    return payload


def problem_linear():
    from app.schemas import ProblemInput

    return ProblemInput(
        problem_text="解方程：2*x+3=7",
        reference_answer="x=2",
    )


def linear_materials():
    payload = materials_payload()
    payload["transfer_item"] = {
        "problem_text": "解方程：3*x+2=8",
        "expected_answer": "x=2",
        "method_signal": "等式两边做相同运算。",
        "options": [
            {
                "option_id": "two",
                "canonical_answer": "x=2",
                "feedback": "正确。",
            },
            {
                "option_id": "negative-two",
                "canonical_answer": "x=-2",
                "feedback": "移项时符号处理错误。",
            },
            {
                "option_id": "three",
                "canonical_answer": "x=3",
                "feedback": "除法计算不正确。",
            },
        ],
        "correct_option_id": "two",
    }
    return payload


def complete_square_problem():
    from app.schemas import ProblemInput

    return ProblemInput(
        problem_text="用配方法解方程：x^2-6*x+5=0",
        reference_answer="x=1 或 x=5",
        required_method="complete_the_square",
    )


def complete_square_route():
    return {
        "math_steps": [
            {
                "purpose": "移去常数项",
                "operation": "subtract_both_sides",
                "operands": ["5"],
                "state_before": ["x^2-6*x+5=0"],
                "state_after": ["x^2-6*x=-5"],
                "reason": "等式两边同时减去 5。",
            },
            {
                "purpose": "配成完全平方",
                "operation": "complete_the_square",
                "operands": ["9"],
                "state_before": ["x^2-6*x=-5"],
                "state_after": ["(x-3)^2=4"],
                "reason": "一次项系数一半的平方是 9。",
            },
            {
                "purpose": "转成两个分支",
                "operation": "take_square_root_both_sides",
                "operands": [],
                "state_before": ["(x-3)^2=4"],
                "state_after": ["x-3=2", "x-3=-2"],
                "reason": "平方等于 4，所以底数等于 2 或 -2。",
            },
            {
                "purpose": "求出两个根",
                "operation": "add_both_sides",
                "operands": ["3"],
                "state_before": ["x-3=2", "x-3=-2"],
                "state_after": ["x=5", "x=1"],
                "reason": "两个分支都在等式两边加 3。",
            },
        ]
    }


def complete_square_materials():
    payload = materials_payload()
    payload["transfer_item"] = {
        "problem_text": "用配方法解方程：x^2-8*x+12=0",
        "expected_answer": "x=2 或 x=6",
        "method_signal": "先把一次项系数取一半再平方。",
        "options": [
            {
                "option_id": "both",
                "canonical_answer": "x=2 或 x=6",
                "feedback": "两个根都正确。",
            },
            {
                "option_id": "two",
                "canonical_answer": "x=2",
                "feedback": "遗漏了另一个分支。",
            },
            {
                "option_id": "six",
                "canonical_answer": "x=6",
                "feedback": "遗漏了另一个分支。",
            },
        ],
        "correct_option_id": "both",
    }
    return payload


def test_complete_square_agent_fallback_accepts_one_repeated_root_branch():
    from app.schemas import ProblemInput

    source = ProblemInput(
        problem_text="用配方法解方程：4*x^2-4*x+1=0",
        reference_answer="x=1/2",
        required_method="complete_the_square",
    )
    route = {
        "math_steps": [
            {
                "purpose": "先把二次项系数化为一",
                "operation": "divide_both_sides",
                "operands": ["4"],
                "state_before": ["4*x^2-4*x+1=0"],
                "state_after": ["x^2-x+1/4=0"],
                "reason": "等式两边同时除以 4。",
            },
            {
                "purpose": "把常数项移到右边",
                "operation": "subtract_both_sides",
                "operands": ["1/4"],
                "state_before": ["x^2-x+1/4=0"],
                "state_after": ["x^2-x=-1/4"],
                "reason": "等式两边同时减去四分之一。",
            },
            {
                "purpose": "配成完全平方",
                "operation": "complete_the_square",
                "operands": ["1/4"],
                "state_before": ["x^2-x=-1/4"],
                "state_after": ["(x-1/2)^2=0"],
                "reason": "一次项系数一半的平方是四分之一。",
            },
            {
                "purpose": "得到唯一零分支",
                "operation": "take_square_root_both_sides",
                "operands": [],
                "state_before": ["(x-1/2)^2=0"],
                "state_after": ["x-1/2=0"],
                "reason": "底数的平方为零，底数只能为零。",
            },
            {
                "purpose": "求出重根",
                "operation": "add_both_sides",
                "operands": ["1/2"],
                "state_before": ["x-1/2=0"],
                "state_after": ["x=1/2"],
                "reason": "等式两边同时加上二分之一。",
            },
        ]
    }
    narrative = narrative_without_route(method="配方法")
    narrative["method_introduction"]["why_it_helps"] = (
        "平方形式能直接连接到所有实数根分支。"
    )
    client = AgentClient(
        [
            route,
            narrative,
            complete_square_materials(),
            approved_review(),
        ]
    )

    lesson = asyncio.run(
        LessonGenerationService(client, MathEngine()).generate(source)
    )

    assert client.calls[0][0] == MATH_ROUTE_SYSTEM
    assert lesson.validation_report["math_route_source"] == "agent"
    assert lesson.validation_report["math_route_method_family"] == (
        "complete_the_square"
    )
