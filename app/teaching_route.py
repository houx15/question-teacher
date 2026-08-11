import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Optional, Protocol

from app.claim_checker import (
    ClaimChecker,
    ClaimCheckResult,
    ClaimStatus,
)
from app.math_expression import (
    StrictMathExpressionError,
    allowed_gap_codes_for_operation,
    deterministic_method_name,
    validate_operation_operands,
    validate_strict_math_expression,
)
from app.schemas import (
    GroundingCheckRequest,
    MathRouteDraft,
    ReferenceGroundingBrief,
)


class TeachingRouteMode(str, Enum):
    SYMBOLIC_VERIFIED = "symbolic_verified"
    MODEL_CROSS_CHECKED = "model_cross_checked"
    REFERENCE_GROUNDED = "reference_grounded"


class TeachingRouteConsistency(str, Enum):
    CONSISTENT = "consistent"
    WARNING = "warning"


class TeachingRouteEvidenceError(ValueError):
    """Check evidence is incomplete or bound to another request."""


class TeachingRouteIntegrityError(ValueError):
    """The frozen teaching route no longer matches its fingerprint."""


class _VerifiedMathRouteLike(Protocol):
    canonical_json: str
    fingerprint: str
    method_family: str

    def thaw(self) -> MathRouteDraft:
        ...


@dataclass(frozen=True)
class FrozenTeachingRoute:
    mode: TeachingRouteMode
    consistency: TeachingRouteConsistency
    method_name: str
    target_expression: str
    final_conclusion: str
    assumptions_json: str
    steps_json: str
    check_evidence_json: str
    canonical_json: str
    fingerprint: str
    symbolic_math_route_json: Optional[str] = None
    symbolic_context_json: Optional[str] = None

    def to_prompt_payload(self) -> dict:
        assumptions = json.loads(self.assumptions_json)
        steps = json.loads(self.steps_json)
        check_evidence = json.loads(self.check_evidence_json)
        canonical_json = _canonical_json(
            _route_content(
                mode=self.mode,
                consistency=self.consistency,
                method_name=self.method_name,
                target_expression=self.target_expression,
                final_conclusion=self.final_conclusion,
                assumptions=assumptions,
                steps=steps,
                check_evidence=check_evidence,
                symbolic_math_route_json=self.symbolic_math_route_json,
                symbolic_context=(
                    json.loads(self.symbolic_context_json)
                    if self.symbolic_context_json is not None
                    else None
                ),
            )
        )
        actual_fingerprint = hashlib.sha256(
            canonical_json.encode("utf-8")
        ).hexdigest()
        if (
            canonical_json != self.canonical_json
            or actual_fingerprint != self.fingerprint
        ):
            raise TeachingRouteIntegrityError(
                "冻结教学路线的完整性检查失败。"
            )
        payload = {
            "verification_mode": self.mode.value,
            "consistency_status": self.consistency.value,
            "method_name": self.method_name,
            "target": self.target_expression,
            "final_conclusion": self.final_conclusion,
            "assumptions": assumptions,
            "steps": steps,
            "check_evidence": check_evidence,
        }
        if self.symbolic_context_json is not None:
            payload["symbolic_context"] = json.loads(
                self.symbolic_context_json
            )
        return payload


def freeze_grounded_route(
    grounding_brief: ReferenceGroundingBrief,
    check_results: Iterable[ClaimCheckResult],
) -> FrozenTeachingRoute:
    check_evidence = _normalize_check_evidence(
        grounding_brief.check_requests,
        check_results,
    )
    has_passed_conclusion_check = any(
        evidence["status"] == ClaimStatus.PASSED.value
        and evidence["conclusion_linked"]
        for evidence in check_evidence
    )
    mode = (
        TeachingRouteMode.MODEL_CROSS_CHECKED
        if has_passed_conclusion_check
        else TeachingRouteMode.REFERENCE_GROUNDED
    )
    consistency = (
        TeachingRouteConsistency.WARNING
        if any(
            evidence["status"]
            in {
                ClaimStatus.UNSUPPORTED.value,
                ClaimStatus.FAILED.value,
            }
            for evidence in check_evidence
        )
        else TeachingRouteConsistency.CONSISTENT
    )
    evidence_by_step = {
        step.step_id: [
            evidence
            for evidence in check_evidence
            if evidence["source_step_id"] == step.step_id
        ]
        for step in grounding_brief.reasoning_steps
    }

    def evidence_status(step_id: str) -> str:
        bound = evidence_by_step[step_id]
        if not bound:
            return "reference_only"
        if all(
            item["status"] == ClaimStatus.PASSED.value
            for item in bound
        ):
            return "checked"
        return "check_warning"

    steps = [
        {
            **step.model_dump(),
            "evidence_status": evidence_status(step.step_id),
        }
        for step in grounding_brief.reasoning_steps
    ]
    return _freeze_route(
        mode=mode,
        consistency=consistency,
        method_name=deterministic_method_name(
            [step.operation_kind for step in grounding_brief.reasoning_steps]
        ),
        target_expression=grounding_brief.target,
        final_conclusion=grounding_brief.reference_conclusion,
        assumptions=[
            assumption.model_dump(mode="python")
            for assumption in grounding_brief.assumptions
        ],
        steps=steps,
        check_evidence=check_evidence,
    )


def freeze_symbolic_route(
    verified_route: _VerifiedMathRouteLike,
    *,
    method_name: Optional[str] = None,
    equation_degree: Optional[int] = None,
    independent_solutions: Optional[list] = None,
) -> FrozenTeachingRoute:
    route = verified_route.thaw()
    operation_kinds = {
        "simplify": "simplify",
        "add_both_sides": "add",
        "subtract_both_sides": "subtract",
        "multiply_both_sides": "multiply",
        "divide_both_sides": "divide",
        "expand": "expand",
        "factor": "factor",
        "combine_like_terms": "combine_like_terms",
        "take_square_root_both_sides": "take_square_root",
        "split_plus_minus": "split_cases",
        "complete_the_square": "complete_square",
        "quadratic_formula": "quadratic_formula",
    }
    steps = [
        {
            "step_id": f"symbolic-step-{index}",
            "statement_before": _join_states(step.state_before),
            "operation_kind": operation_kinds[step.operation],
            "operands": list(step.operands),
            "statement_after": _join_states(step.state_after),
            "evidence_status": "checked",
        }
        for index, step in enumerate(route.math_steps, start=1)
    ]
    return _freeze_route(
        mode=TeachingRouteMode.SYMBOLIC_VERIFIED,
        consistency=TeachingRouteConsistency.CONSISTENT,
        method_name=method_name or verified_route.method_family,
        target_expression=_infer_symbolic_target(route),
        final_conclusion=_join_states(
            route.math_steps[-1].state_after
        ),
        assumptions=[],
        steps=steps,
        symbolic_math_route_json=verified_route.canonical_json,
        symbolic_context={
            "equation_degree": equation_degree,
            "independent_solutions": list(independent_solutions or []),
            "math_steps": route.model_dump()["math_steps"],
            "method_family": verified_route.method_family,
        },
    )


def _freeze_route(
    *,
    mode: TeachingRouteMode,
    consistency: TeachingRouteConsistency,
    method_name: str,
    final_conclusion: str,
    assumptions: list,
    steps: list,
    target_expression: Optional[str] = None,
    check_evidence: Optional[list] = None,
    symbolic_math_route_json: Optional[str] = None,
    symbolic_context: Optional[dict] = None,
) -> FrozenTeachingRoute:
    target = target_expression or final_conclusion
    try:
        target = validate_strict_math_expression(target)
        final_conclusion = validate_strict_math_expression(final_conclusion)
        assumptions = [
            {
                "assumption_id": item["assumption_id"],
                "expression": validate_strict_math_expression(
                    item["expression"]
                ),
                "source_kind": item.get("source_kind", "solution"),
            }
            for item in assumptions
        ]
        steps = [
            {
                **step,
                "statement_before": validate_strict_math_expression(
                    step["statement_before"]
                ),
                "operands": [
                    validate_strict_math_expression(item)
                    for item in step.get("operands", [])
                ],
                "statement_after": validate_strict_math_expression(
                    step["statement_after"]
                ),
                "assumption_ids_used": list(
                    step.get("assumption_ids_used", [])
                ),
                "allowed_reasoning_gap_codes": (
                    allowed_gap_codes_for_operation(
                        step["operation_kind"]
                    )
                ),
                "source_kind": "verified_route",
            }
            for step in steps
        ]
        for step in steps:
            validate_operation_operands(
                step["operation_kind"], step["operands"]
            )
        assumption_ids = [
            item["assumption_id"] for item in assumptions
        ]
        if len(assumption_ids) != len(set(assumption_ids)):
            raise ValueError("duplicate grounded assumption ids")
        known_assumption_ids = set(assumption_ids)
        if any(
            not set(step["assumption_ids_used"])
            <= known_assumption_ids
            for step in steps
        ):
            raise ValueError("unknown grounded assumption id")
    except (KeyError, StrictMathExpressionError, ValueError):
        raise TeachingRouteIntegrityError(
            "冻结教学路线包含无效数学结构。"
        ) from None
    normalized_check_evidence = check_evidence or []
    assumptions_json = _canonical_json(assumptions)
    steps_json = _canonical_json(steps)
    check_evidence_json = _canonical_json(normalized_check_evidence)
    symbolic_context_json = (
        _canonical_json(symbolic_context)
        if symbolic_context is not None
        else None
    )
    canonical_json = _canonical_json(
        _route_content(
            mode=mode,
            consistency=consistency,
            method_name=method_name,
            target_expression=target,
            final_conclusion=final_conclusion,
            assumptions=json.loads(assumptions_json),
            steps=json.loads(steps_json),
            check_evidence=json.loads(check_evidence_json),
            symbolic_math_route_json=symbolic_math_route_json,
            symbolic_context=symbolic_context,
        )
    )
    fingerprint = hashlib.sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()
    return FrozenTeachingRoute(
        mode=mode,
        consistency=consistency,
        method_name=method_name,
        target_expression=target,
        final_conclusion=final_conclusion,
        assumptions_json=assumptions_json,
        steps_json=steps_json,
        check_evidence_json=check_evidence_json,
        canonical_json=canonical_json,
        fingerprint=fingerprint,
        symbolic_math_route_json=symbolic_math_route_json,
        symbolic_context_json=symbolic_context_json,
    )


def _route_content(
    *,
    mode: TeachingRouteMode,
    consistency: TeachingRouteConsistency,
    method_name: str,
    target_expression: str,
    final_conclusion: str,
    assumptions: list,
    steps: list,
    check_evidence: list,
    symbolic_math_route_json: Optional[str],
    symbolic_context: Optional[dict],
) -> dict:
    return {
        "assumptions": assumptions,
        "consistency_status": consistency.value,
        "final_conclusion": final_conclusion,
        "method_name": method_name,
        "target": target_expression,
        "steps": steps,
        "check_evidence": check_evidence,
        "symbolic_math_route_json": symbolic_math_route_json,
        "symbolic_context": symbolic_context,
        "verification_mode": mode.value,
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _normalize_check_evidence(
    requests: Iterable[GroundingCheckRequest],
    results: Iterable[ClaimCheckResult],
) -> list:
    request_items = tuple(requests)
    result_items = tuple(results)
    request_by_id = {
        request.check_id: request for request in request_items
    }
    if len(request_by_id) != len(request_items):
        raise TeachingRouteEvidenceError(
            "check request ids must be unique"
        )

    result_by_id = {}
    for result in result_items:
        if result.check_id not in request_by_id:
            raise TeachingRouteEvidenceError(
                f"unknown check result id: {result.check_id}"
            )
        if result.check_id in result_by_id:
            raise TeachingRouteEvidenceError(
                f"duplicate check result id: {result.check_id}"
            )
        result_by_id[result.check_id] = result

    missing_ids = set(request_by_id) - set(result_by_id)
    if missing_ids:
        missing = ",".join(sorted(missing_ids))
        raise TeachingRouteEvidenceError(
            f"missing check result ids: {missing}"
        )

    evidence = []
    for check_id in sorted(request_by_id):
        request = request_by_id[check_id]
        result = result_by_id[check_id]
        expected_fingerprint = ClaimChecker.request_fingerprint(
            request
        )
        if result.request_fingerprint != expected_fingerprint:
            raise TeachingRouteEvidenceError(
                f"check request fingerprint mismatch: {check_id}"
            )
        if result.conclusion_linked != request.conclusion_linked:
            raise TeachingRouteEvidenceError(
                f"check conclusion linkage mismatch: {check_id}"
            )
        evidence.append(
            {
                "check_id": check_id,
                "conclusion_linked": request.conclusion_linked,
                "reason_code": result.reason_code,
                "request_fingerprint": expected_fingerprint,
                "source_step_id": request.source_step_id,
                "status": result.status.value,
            }
        )
    return evidence


def _join_states(states: Iterable[str]) -> str:
    return "；".join(states)


def _infer_symbolic_target(route: MathRouteDraft) -> str:
    symbols = {
        match.group()
        for step in route.math_steps
        for state in (*step.state_before, *step.state_after)
        for match in re.finditer(
            r"(?<![A-Za-z])[A-Za-z](?![A-Za-z])",
            state,
        )
    }
    if len(symbols) == 1:
        return next(iter(symbols))
    return _join_states(route.math_steps[-1].state_after)
