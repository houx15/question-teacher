import hashlib
import json
from dataclasses import replace

import pytest

from app.claim_checker import ClaimChecker, ClaimStatus
from app.generation import LessonQualityError, _VerifiedMathRoute
from app.schemas import MathRouteDraft, ReferenceGroundingBrief
from app.teaching_route import (
    TeachingRouteConsistency,
    TeachingRouteEvidenceError,
    TeachingRouteMode,
    freeze_grounded_route,
    freeze_symbolic_route,
)


REFERENCE_ANSWER = r"\(m-n=\frac12\)"


def grounding_brief(
    check_requests=None,
) -> ReferenceGroundingBrief:
    return ReferenceGroundingBrief.model_validate(
        {
            "task_summary": "把已知根代回方程，求m-n",
            "target": r"\(m-n\)",
            "assumptions": [
                {"assumption_id": "nonzero-n", "expression": r"\(n\ne0\)"},
                {"assumption_id": "given-root", "expression": r"\(x=2n\)"},
            ],
            "reference_conclusion": REFERENCE_ANSWER,
            "method_name": "代入法",
            "reasoning_steps": [
                {
                    "step_id": "substitute-root",
                    "statement_before": r"\(x^2-2mx+2n=0\)",
                    "operation_kind": "substitute",
                    "operands": ["x=2n"],
                    "statement_after": r"\(4n^2-4mn+2n=0\)",
                    "assumption_ids_used": ["given-root"],
                },
                {
                    "step_id": "use-nonzero",
                    "statement_before": r"\(2n(2n-2m+1)=0\)",
                    "operation_kind": "divide",
                    "operands": ["2n"],
                    "statement_after": r"\(2n-2m+1=0\)",
                    "assumption_ids_used": ["nonzero-n"],
                },
            ],
            "check_requests": check_requests or [],
            "audit_notes": [],
        },
        context={"reference_answer": REFERENCE_ANSWER},
    )


def checked_brief(
    status: ClaimStatus,
    *,
    conclusion_linked: bool = True,
    check_id: str = "back-check",
) -> tuple:
    expressions = {
        ClaimStatus.PASSED: ("2*n-2*m+1", "2*n-2*m+1"),
        ClaimStatus.FAILED: ("1", "2"),
        ClaimStatus.UNSUPPORTED: ("n+1", "1"),
    }
    expression, expected = expressions[status]
    request = {
        "check_id": check_id,
        "kind": (
            "nonzero_division"
            if status == ClaimStatus.UNSUPPORTED
            else "equivalence"
        ),
        "expression": expression,
        "expected": expected,
        "substitutions": {},
        "nonzero_symbols": (
            ["n"] if status == ClaimStatus.UNSUPPORTED else []
        ),
        "conclusion_linked": conclusion_linked,
    }
    brief = grounding_brief([request])
    return brief, ClaimChecker().check(brief.check_requests[0])


def passed_brief_with_expression(expression: str) -> tuple:
    brief = grounding_brief(
        [
            {
                "check_id": "same-check-id",
                "kind": "equivalence",
                "expression": expression,
                "expected": expression,
                "substitutions": {},
                "nonzero_symbols": [],
                "conclusion_linked": True,
            }
        ]
    )
    return brief, ClaimChecker().check(brief.check_requests[0])


def symbolic_route() -> _VerifiedMathRoute:
    return _VerifiedMathRoute.freeze(
        MathRouteDraft.model_validate(
            {
                "math_steps": [
                    {
                        "purpose": "因式分解",
                        "operation": "factor",
                        "operands": [],
                        "state_before": ["x^2-5x+6=0"],
                        "state_after": ["(x-2)(x-3)=0"],
                        "reason": "乘积为零",
                    }
                ]
            }
        ),
        "factor",
        source="deterministic",
    )


def test_grounded_route_preserves_assumptions_and_conclusion():
    brief, result = checked_brief(ClaimStatus.PASSED)
    route = freeze_grounded_route(brief, [result])
    payload = route.to_prompt_payload()

    assert route.mode == TeachingRouteMode.MODEL_CROSS_CHECKED
    assert route.consistency == TeachingRouteConsistency.CONSISTENT
    assert route.method_name == "代入法"
    assert route.final_conclusion == REFERENCE_ANSWER
    assert payload["assumptions"] == [
        {"assumption_id": "nonzero-n", "expression": r"\(n\ne0\)"},
        {"assumption_id": "given-root", "expression": r"\(x=2n\)"},
    ]
    assert payload["steps"][0]["evidence_status"] == "reference_only"
    assert payload["check_evidence"] == [
        {
            "check_id": "back-check",
            "conclusion_linked": True,
            "reason_code": "equivalent",
            "request_fingerprint": result.request_fingerprint,
            "status": "passed",
        }
    ]
    assert route.fingerprint


def test_grounder_free_prose_is_not_frozen_but_typed_changes_are():
    first = grounding_brief()
    free_prose_change = first.model_copy(
        update={
            "task_summary": "任意的私有引用摘要",
            "method_name": "任意的模型方法名",
            "audit_notes": ["任意的审计备注"],
        }
    )
    changed_step = first.reasoning_steps[0].model_copy(
        update={"operation_kind": "derive", "operands": []}
    )
    typed_change = first.model_copy(
        update={
            "reasoning_steps": [changed_step, *first.reasoning_steps[1:]]
        }
    )

    baseline = freeze_grounded_route(first, [])
    assert freeze_grounded_route(free_prose_change, []).fingerprint == (
        baseline.fingerprint
    )
    assert freeze_grounded_route(typed_change, []).fingerprint != (
        baseline.fingerprint
    )


def test_unchecked_grounded_route_is_reference_grounded():
    route = freeze_grounded_route(grounding_brief(), [])

    assert route.mode == TeachingRouteMode.REFERENCE_GROUNDED
    assert route.consistency == TeachingRouteConsistency.CONSISTENT
    assert route.to_prompt_payload()["steps"][0]["evidence_status"] == (
        "reference_only"
    )


def test_passed_unlinked_check_does_not_upgrade_grounded_mode():
    brief, result = checked_brief(
        ClaimStatus.PASSED,
        conclusion_linked=False,
    )
    route = freeze_grounded_route(brief, [result])

    assert route.mode == TeachingRouteMode.REFERENCE_GROUNDED


def test_model_proposed_failed_linked_check_creates_warning():
    brief, result = checked_brief(ClaimStatus.FAILED)
    route = freeze_grounded_route(brief, [result])

    assert route.mode == TeachingRouteMode.REFERENCE_GROUNDED
    assert route.consistency == TeachingRouteConsistency.WARNING


@pytest.mark.parametrize(
    ("status", "conclusion_linked"),
    [
        (ClaimStatus.UNSUPPORTED, True),
        (ClaimStatus.FAILED, False),
    ],
)
def test_nonblocking_check_problem_creates_warning(
    status,
    conclusion_linked,
):
    brief, result = checked_brief(
        status,
        conclusion_linked=conclusion_linked,
    )
    route = freeze_grounded_route(brief, [result])

    assert route.mode == TeachingRouteMode.REFERENCE_GROUNDED
    assert route.consistency == TeachingRouteConsistency.WARNING


@pytest.mark.parametrize(
    "results_factory",
    [
        lambda result: [],
        lambda result: [result, result],
        lambda result: [replace(result, check_id="unknown-check")],
    ],
    ids=["missing", "duplicate", "unknown"],
)
def test_grounded_route_rejects_non_bijective_check_results(
    results_factory,
):
    brief, result = checked_brief(ClaimStatus.PASSED)

    with pytest.raises(TeachingRouteEvidenceError):
        freeze_grounded_route(brief, results_factory(result))


def test_grounded_route_rejects_result_supplied_linkage():
    brief, result = checked_brief(ClaimStatus.PASSED)
    forged = replace(result, conclusion_linked=False)

    with pytest.raises(
        TeachingRouteEvidenceError,
        match="linkage",
    ):
        freeze_grounded_route(brief, [forged])


def test_grounded_route_rejects_result_reused_for_same_id_in_other_brief():
    first_brief, first_result = passed_brief_with_expression("x")
    second_brief, _ = passed_brief_with_expression("m")

    with pytest.raises(
        TeachingRouteEvidenceError,
        match="fingerprint",
    ):
        freeze_grounded_route(second_brief, [first_result])


def test_check_evidence_changes_route_fingerprint():
    brief, result = checked_brief(ClaimStatus.PASSED)

    original = freeze_grounded_route(brief, [result])
    changed = freeze_grounded_route(
        brief,
        [replace(result, reason_code="different-reason")],
    )

    assert original.fingerprint != changed.fingerprint
    assert json.loads(original.canonical_json)["check_evidence"] != (
        json.loads(changed.canonical_json)["check_evidence"]
    )


def test_thawed_route_cannot_mutate_frozen_fingerprint():
    route = freeze_grounded_route(grounding_brief(), [])
    thawed = route.to_prompt_payload()

    thawed["steps"][0]["statement_after"] = "mutated"
    thawed["check_evidence"].append({"status": "mutated"})

    assert route.to_prompt_payload()["steps"][0]["statement_after"] != (
        "mutated"
    )
    assert route.to_prompt_payload()["check_evidence"] == []
    assert route.fingerprint == hashlib.sha256(
        route.canonical_json.encode("utf-8")
    ).hexdigest()


def test_equivalent_grounded_routes_have_canonical_fingerprint():
    first = freeze_grounded_route(grounding_brief(), [])
    second = freeze_grounded_route(grounding_brief(), [])

    assert first.fingerprint == second.fingerprint
    assert json.loads(first.canonical_json) == json.loads(
        second.canonical_json
    )


def test_symbolic_route_adapts_verified_math_route():
    verified = symbolic_route()

    route = freeze_symbolic_route(verified)
    payload = route.to_prompt_payload()

    assert route.mode == TeachingRouteMode.SYMBOLIC_VERIFIED
    assert route.consistency == TeachingRouteConsistency.CONSISTENT
    assert route.method_name == "factor"
    assert payload["target"] == "x"
    assert route.final_conclusion == "(x-2)(x-3)=0"
    assert payload["assumptions"] == []
    assert payload["steps"][0]["evidence_status"] == "checked"
    assert payload["check_evidence"] == []
    assert route.symbolic_math_route_json == verified.canonical_json


def test_symbolic_route_uses_existing_integrity_validation():
    verified = symbolic_route()
    tampered = _VerifiedMathRoute(
        canonical_json=verified.canonical_json.replace(
            "(x-2)(x-3)=0",
            "x=99",
        ),
        fingerprint=verified.fingerprint,
        method_family=verified.method_family,
        source=verified.source,
    )

    with pytest.raises(LessonQualityError, match="完整性检查失败"):
        freeze_symbolic_route(tampered)
