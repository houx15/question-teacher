"""Server-owned boundary for untrusted reference-solution prose."""

import re
from dataclasses import dataclass
from typing import Any, FrozenSet, Iterable, Mapping, Sequence, Tuple

from pydantic import BaseModel

from app.preparation_models import SolutionTrace
from app.schemas import ProblemInput, ReferenceGroundingBrief
from app.teaching_route import FrozenTeachingRoute


REFERENCE_PROSE_FINGERPRINT_LENGTH = 8
_MAX_WALK_NODES = 20_000
_MAX_AGGREGATE_TEXT_CHARS = 2_000_000
_EXPLICIT_MATH = re.compile(
    r"\$\$.*?\$\$|\$[^$\r\n]*\$|\\\([^\r\n]*?\\\)|\\\[.*?\\\]",
    re.DOTALL,
)
_ASCII_MATH_RUN = re.compile(r"[A-Za-z0-9\\{}()\[\]^_+*/=<>.\-]+")
_LATEX_MATH_COMMAND = re.compile(
    r"\\(?:frac|sqrt|ne|neq|times|cdot|pm|le|ge|left|right)\b"
)
_STRUCTURAL_FIELDS = frozenset(
    {
        "action",
        "artifact_type",
        "criterion",
        "evidence_status",
        "invalidated_downstream_artifacts",
        "kind",
        "layer",
        "mode",
        "operation",
        "persistence",
        "responsible_role",
        "retained_artifacts",
        "role",
        "severity",
        "source_kind",
        "status",
        "surface",
        "trajectory_type",
        "type",
    }
)
_LONG_ALPHABETIC_WORD = re.compile(r"[A-Za-z]{3,}")
_CJK_CHARACTER = re.compile(r"[\u3400-\u9fff]")
_CONTROLLED_SAFE_NORMALIZED = frozenset(
    {
        "根据题意可以得到",
        "由条件可得然后继续整理",
    }
)
_CONTROLLED_EQUATION_ACTION = re.compile(
    r"^(?:在)?(?:方程|等式)两边(?:同时|都|同)?"
    r"(?:加上?|减去?|乘以|除以)[A-Za-z0-9零一二三四五六七八九十分之]{1,24}$"
)


class ReferenceContentSafetyError(ValueError):
    """Raised without echoing the sensitive literal."""


def _looks_like_math(run: str) -> bool:
    if _LATEX_MATH_COMMAND.search(run) is not None:
        return True
    if (
        _LONG_ALPHABETIC_WORD.search(run) is not None
        or _CJK_CHARACTER.search(run) is not None
    ):
        return False
    strong_operator = any(char in run for char in "=+*/^<>")
    subtraction = (
        "-" in run
        and any(char.isalnum() for char in run)
    )
    return (
        strong_operator
        or subtraction
    )


def _replace_explicit_math(match: re.Match) -> str:
    source = match.group()
    if source.startswith("$$"):
        inner = source[2:-2]
    elif source.startswith("$"):
        inner = source[1:-1]
    else:
        inner = source[2:-2]
    stripped = inner.strip()
    simple_math = (
        _ASCII_MATH_RUN.fullmatch(stripped) is not None
        and re.search(r"[A-Za-z]{3,}", stripped) is None
    )
    return "" if _looks_like_math(stripped) or simple_math else inner


def _normalized_prose(value: str) -> str:
    without_explicit_math = _EXPLICIT_MATH.sub(
        _replace_explicit_math,
        value,
    )
    pieces = []
    cursor = 0
    for match in _ASCII_MATH_RUN.finditer(without_explicit_math):
        pieces.append(without_explicit_math[cursor : match.start()])
        run = match.group()
        pieces.append(
            "" if _looks_like_math(run) else run
        )
        cursor = match.end()
    pieces.append(without_explicit_math[cursor:])
    return "".join(
        char.casefold() for char in "".join(pieces) if char.isalnum()
    )


def _fingerprints(value: str) -> Iterable[str]:
    size = REFERENCE_PROSE_FINGERPRINT_LENGTH
    normalized = _normalized_prose(value)
    for index in range(len(normalized) - size + 1):
        yield normalized[index : index + size]


def _is_structural_field(field_name: str) -> bool:
    return (
        field_name in _STRUCTURAL_FIELDS
        or field_name.endswith("_id")
        or field_name.endswith("_ids")
    )


def _is_controlled_safe_text(value: str) -> bool:
    normalized = _normalized_prose(value)
    return (
        normalized in _CONTROLLED_SAFE_NORMALIZED
        or _CONTROLLED_EQUATION_ACTION.fullmatch(normalized) is not None
    )


def _bounded_strings(value: Any) -> Iterable[Tuple[str, bool]]:
    remaining = _MAX_WALK_NODES
    aggregate_chars = 0
    stack = [(value, "")]
    while stack and remaining:
        current, field_name = stack.pop()
        remaining -= 1
        if isinstance(current, str):
            aggregate_chars += len(current)
            if aggregate_chars > _MAX_AGGREGATE_TEXT_CHARS:
                raise ReferenceContentSafetyError(
                    "reference safety text bound exceeded"
                )
            yield current, _is_structural_field(field_name)
        elif isinstance(current, BaseModel):
            stack.append((current.model_dump(mode="python"), field_name))
        elif isinstance(current, Mapping):
            stack.extend(
                (item, str(key)) for key, item in current.items()
            )
        elif isinstance(current, Sequence) and not isinstance(
            current, (bytes, bytearray)
        ):
            stack.extend((item, field_name) for item in current)
    if stack:
        raise ReferenceContentSafetyError(
            "reference safety traversal exceeded its bound"
        )


@dataclass(frozen=True)
class ReferenceSafetyPolicy:
    """Reject raw-only prose fingerprints while allowing mathematics."""

    sensitive_fingerprints: FrozenSet[str]

    @classmethod
    def from_problem(cls, problem: ProblemInput) -> "ReferenceSafetyPolicy":
        raw = problem.reference_solution_text or ""
        public = problem.problem_text + "\n" + problem.reference_answer
        public_fingerprints = frozenset(_fingerprints(public))
        return cls(
            sensitive_fingerprints=(
                frozenset(_fingerprints(raw)) - public_fingerprints
            ),
        )

    def ensure_safe(self, value: Any) -> None:
        if not self.sensitive_fingerprints:
            return
        for text, structural in _bounded_strings(value):
            if structural:
                continue
            if _is_controlled_safe_text(text):
                continue
            prose_leak = any(
                item in self.sensitive_fingerprints
                for item in _fingerprints(text)
            )
            if prose_leak:
                raise ReferenceContentSafetyError(
                    "reference-only content crossed the safe boundary"
                )

    def sanitize_solution_trace(
        self,
        trace: SolutionTrace,
        teaching_route: FrozenTeachingRoute,
    ) -> SolutionTrace:
        """Project untrusted analyst output onto the frozen route contract.

        The analyst is allowed to inspect raw reference prose, but none of its
        free-form semantic text becomes a downstream authority.  The server
        keeps only the fact that a structurally valid trace was returned and
        rebuilds the trace from the already guarded, frozen teaching route.
        """
        del trace
        route = teaching_route.to_prompt_payload()
        assumption_ids = [
            "route-assumption-%03d" % index
            for index in range(1, len(route["assumptions"]) + 1)
        ]
        payload = {
            "task_target": "按既定方法完成题目并得到参考结论",
            "reference_conclusion": route["final_conclusion"],
            "assumptions": [
                {
                    "assumption_id": assumption_id,
                    "content": content,
                    "source_anchor": {
                        "source_kind": "problem",
                        "source_id": "problem-assumption-%03d" % index,
                        "excerpt": "题目结构依据",
                    },
                }
                for index, (assumption_id, content) in enumerate(
                    zip(assumption_ids, route["assumptions"]),
                    start=1,
                )
            ],
            "source_steps": [
                {
                    "source_step_id": step["step_id"],
                    "source_anchor": {
                        "source_kind": "verified_route",
                        "source_id": step["step_id"],
                        "excerpt": "已验证路线结构依据",
                    },
                    "state_before": step["statement_before"],
                    "mathematical_action": step["operation_explanation"],
                    "justification": "根据已验证教学路线保留这一步的数学依赖",
                    "state_after": step["statement_after"],
                    "new_information": step["statement_after"],
                    "assumption_ids_used": list(assumption_ids),
                    "omitted_reasoning": [],
                    "evidence_status": "verified_route",
                }
                for step in route["steps"]
            ],
            "audit_notes": [],
        }
        sanitized = SolutionTrace.model_validate(payload)
        self.ensure_safe(sanitized)
        return sanitized

    def sanitize_grounding_brief(
        self,
        brief: ReferenceGroundingBrief,
        reference_answer: str,
    ) -> ReferenceGroundingBrief:
        payload = brief.model_dump(mode="python")
        payload["audit_notes"] = []
        sanitized = ReferenceGroundingBrief.validate_for_reference_answer(
            payload,
            reference_answer,
        )
        self.ensure_safe(sanitized)
        return sanitized
