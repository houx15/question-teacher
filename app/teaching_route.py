import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Optional, Protocol

from app.claim_checker import ClaimCheckResult, ClaimStatus
from app.schemas import MathRouteDraft, ReferenceGroundingBrief


class TeachingRouteMode(str, Enum):
    SYMBOLIC_VERIFIED = "symbolic_verified"
    MODEL_CROSS_CHECKED = "model_cross_checked"
    REFERENCE_GROUNDED = "reference_grounded"


class TeachingRouteConsistency(str, Enum):
    CONSISTENT = "consistent"
    WARNING = "warning"


class TeachingRouteContradiction(ValueError):
    """A reproducible conclusion-linked contradiction in reference material."""


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
    final_conclusion: str
    assumptions_json: str
    steps_json: str
    canonical_json: str
    fingerprint: str
    symbolic_math_route_json: Optional[str] = None

    def to_prompt_payload(self) -> dict:
        assumptions = json.loads(self.assumptions_json)
        steps = json.loads(self.steps_json)
        canonical_json = _canonical_json(
            _route_content(
                mode=self.mode,
                consistency=self.consistency,
                method_name=self.method_name,
                final_conclusion=self.final_conclusion,
                assumptions=assumptions,
                steps=steps,
                symbolic_math_route_json=self.symbolic_math_route_json,
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
        return {
            "verification_mode": self.mode.value,
            "consistency_status": self.consistency.value,
            "method_name": self.method_name,
            "final_conclusion": self.final_conclusion,
            "assumptions": assumptions,
            "steps": steps,
        }


def freeze_grounded_route(
    grounding_brief: ReferenceGroundingBrief,
    check_results: Iterable[ClaimCheckResult],
) -> FrozenTeachingRoute:
    results = tuple(check_results)
    if any(
        result.status == ClaimStatus.FAILED
        and result.conclusion_linked
        for result in results
    ):
        raise TeachingRouteContradiction(
            "参考材料中的推导存在明确矛盾。"
        )

    has_passed_conclusion_check = any(
        result.status == ClaimStatus.PASSED
        and result.conclusion_linked
        for result in results
    )
    mode = (
        TeachingRouteMode.MODEL_CROSS_CHECKED
        if has_passed_conclusion_check
        else TeachingRouteMode.REFERENCE_GROUNDED
    )
    consistency = (
        TeachingRouteConsistency.WARNING
        if any(
            result.status == ClaimStatus.UNSUPPORTED
            or (
                result.status == ClaimStatus.FAILED
                and not result.conclusion_linked
            )
            for result in results
        )
        else TeachingRouteConsistency.CONSISTENT
    )
    evidence_status = (
        "cross_checked"
        if has_passed_conclusion_check
        else "reference_only"
    )
    steps = [
        {
            **step.model_dump(),
            "evidence_status": evidence_status,
        }
        for step in grounding_brief.reasoning_steps
    ]
    return _freeze_route(
        mode=mode,
        consistency=consistency,
        method_name=grounding_brief.method_name,
        final_conclusion=grounding_brief.reference_conclusion,
        assumptions=list(grounding_brief.assumptions),
        steps=steps,
    )


def freeze_symbolic_route(
    verified_route: _VerifiedMathRouteLike,
) -> FrozenTeachingRoute:
    route = verified_route.thaw()
    steps = [
        {
            "step_id": f"symbolic-step-{index}",
            "statement_before": _join_states(step.state_before),
            "operation_explanation": step.reason,
            "statement_after": _join_states(step.state_after),
            "evidence_status": "checked",
        }
        for index, step in enumerate(route.math_steps, start=1)
    ]
    return _freeze_route(
        mode=TeachingRouteMode.SYMBOLIC_VERIFIED,
        consistency=TeachingRouteConsistency.CONSISTENT,
        method_name=verified_route.method_family,
        final_conclusion=_join_states(
            route.math_steps[-1].state_after
        ),
        assumptions=[],
        steps=steps,
        symbolic_math_route_json=verified_route.canonical_json,
    )


def _freeze_route(
    *,
    mode: TeachingRouteMode,
    consistency: TeachingRouteConsistency,
    method_name: str,
    final_conclusion: str,
    assumptions: list,
    steps: list,
    symbolic_math_route_json: Optional[str] = None,
) -> FrozenTeachingRoute:
    assumptions_json = _canonical_json(assumptions)
    steps_json = _canonical_json(steps)
    canonical_json = _canonical_json(
        _route_content(
            mode=mode,
            consistency=consistency,
            method_name=method_name,
            final_conclusion=final_conclusion,
            assumptions=json.loads(assumptions_json),
            steps=json.loads(steps_json),
            symbolic_math_route_json=symbolic_math_route_json,
        )
    )
    fingerprint = hashlib.sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()
    return FrozenTeachingRoute(
        mode=mode,
        consistency=consistency,
        method_name=method_name,
        final_conclusion=final_conclusion,
        assumptions_json=assumptions_json,
        steps_json=steps_json,
        canonical_json=canonical_json,
        fingerprint=fingerprint,
        symbolic_math_route_json=symbolic_math_route_json,
    )


def _route_content(
    *,
    mode: TeachingRouteMode,
    consistency: TeachingRouteConsistency,
    method_name: str,
    final_conclusion: str,
    assumptions: list,
    steps: list,
    symbolic_math_route_json: Optional[str],
) -> dict:
    return {
        "assumptions": assumptions,
        "consistency_status": consistency.value,
        "final_conclusion": final_conclusion,
        "method_name": method_name,
        "steps": steps,
        "symbolic_math_route_json": symbolic_math_route_json,
        "verification_mode": mode.value,
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _join_states(states: Iterable[str]) -> str:
    return "；".join(states)
