import hashlib
import json

import pytest

from app.claim_checker import ClaimCheckResult, ClaimStatus
from app.generation import LessonQualityError, _VerifiedMathRoute
from app.schemas import MathRouteDraft, ReferenceGroundingBrief
from app.teaching_route import (
    TeachingRouteConsistency,
    TeachingRouteContradiction,
    TeachingRouteMode,
    freeze_grounded_route,
    freeze_symbolic_route,
)


REFERENCE_ANSWER = r"\(m-n=\frac12\)"


def grounding_brief() -> ReferenceGroundingBrief:
    return ReferenceGroundingBrief.model_validate(
        {
            "task_summary": "把已知根代回方程，求m-n",
            "target": r"\(m-n\)",
            "assumptions": [
                r"\(n\ne0\)",
                r"\(x=2n\)是原方程的根",
            ],
            "reference_conclusion": REFERENCE_ANSWER,
            "method_name": "代入法",
            "reasoning_steps": [
                {
                    "step_id": "substitute-root",
                    "statement_before": r"\(x^2-2mx+2n=0\)",
                    "operation_explanation": "把已知根x=2n代入原方程",
                    "statement_after": r"\(4n^2-4mn+2n=0\)",
                },
                {
                    "step_id": "use-nonzero",
                    "statement_before": r"\(2n(2n-2m+1)=0\)",
                    "operation_explanation": "利用n不为0约去2n",
                    "statement_after": r"\(2n-2m+1=0\)",
                },
            ],
            "check_requests": [],
            "audit_notes": [],
        },
        context={"reference_answer": REFERENCE_ANSWER},
    )


def check_result(
    status: ClaimStatus,
    *,
    conclusion_linked: bool = True,
    check_id: str = "back-check",
) -> ClaimCheckResult:
    return ClaimCheckResult(
        check_id=check_id,
        status=status,
        conclusion_linked=conclusion_linked,
        reason_code="test-result",
    )


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
    route = freeze_grounded_route(
        grounding_brief(),
        [check_result(ClaimStatus.PASSED)],
    )

    assert route.mode == TeachingRouteMode.MODEL_CROSS_CHECKED
    assert route.consistency == TeachingRouteConsistency.CONSISTENT
    assert route.method_name == "代入法"
    assert route.final_conclusion == REFERENCE_ANSWER
    assert route.to_prompt_payload()["assumptions"] == [
        r"\(n\ne0\)",
        r"\(x=2n\)是原方程的根",
    ]
    assert route.to_prompt_payload()["steps"][0]["evidence_status"] == (
        "cross_checked"
    )
    assert route.fingerprint


def test_unchecked_grounded_route_is_reference_grounded():
    route = freeze_grounded_route(grounding_brief(), [])

    assert route.mode == TeachingRouteMode.REFERENCE_GROUNDED
    assert route.consistency == TeachingRouteConsistency.CONSISTENT
    assert route.to_prompt_payload()["steps"][0]["evidence_status"] == (
        "reference_only"
    )


def test_passed_unlinked_check_does_not_upgrade_grounded_mode():
    route = freeze_grounded_route(
        grounding_brief(),
        [
            check_result(
                ClaimStatus.PASSED,
                conclusion_linked=False,
            )
        ],
    )

    assert route.mode == TeachingRouteMode.REFERENCE_GROUNDED


def test_failed_conclusion_linked_check_is_contradiction():
    with pytest.raises(
        TeachingRouteContradiction,
        match="参考材料中的推导存在明确矛盾",
    ):
        freeze_grounded_route(
            grounding_brief(),
            [check_result(ClaimStatus.FAILED)],
        )


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
    route = freeze_grounded_route(
        grounding_brief(),
        [
            check_result(
                status,
                conclusion_linked=conclusion_linked,
            )
        ],
    )

    assert route.mode == TeachingRouteMode.REFERENCE_GROUNDED
    assert route.consistency == TeachingRouteConsistency.WARNING


def test_thawed_route_cannot_mutate_frozen_fingerprint():
    route = freeze_grounded_route(grounding_brief(), [])
    thawed = route.to_prompt_payload()

    thawed["steps"][0]["statement_after"] = "mutated"

    assert route.to_prompt_payload()["steps"][0]["statement_after"] != (
        "mutated"
    )
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
    assert route.final_conclusion == "(x-2)(x-3)=0"
    assert payload["assumptions"] == []
    assert payload["steps"][0]["evidence_status"] == "checked"
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
