"""Server-owned boundary for untrusted reference-solution prose."""

import re
from dataclasses import dataclass, field
from typing import Any, FrozenSet, Iterable, Mapping, Sequence, Tuple

from pydantic import BaseModel

from app.math_expression import (
    StrictMathText,
    is_strict_math_expression,
    long_numeric_literals,
    math_identifiers,
    render_typed_math_action,
    render_typed_math_justification,
)
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
_ASCII_ALNUM_RUN = re.compile(r"[A-Za-z0-9]+")
_SKELETON_MATH_COMMANDS = frozenset(
    {
        "angle",
        "circ",
        "cong",
        "cos",
        "dfrac",
        "frac",
        "mathbb",
        "overline",
        "parallel",
        "perp",
        "sim",
        "sin",
        "sqrt",
        "tan",
        "tfrac",
        "triangle",
    }
)
_CONTROL_SKELETON_TERMS = (
    "confidential",
    "hidden",
    "ignore",
    "private",
    "prompt",
    "rules",
    "secret",
    "system",
    "token",
)
_MIN_SHORT_SKELETON_LENGTH = 6
_STRUCTURAL_FIELDS = frozenset(
    {
        "action",
        "artifact_type",
        "criterion",
        "evidence_status",
        "gap_code",
        "invalidated_downstream_artifacts",
        "kind",
        "layer",
        "mode",
        "operation",
        "operation_kind",
        "persistence",
        "responsible_role",
        "reasoning_gap_codes",
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


class ReferenceContentSafetyError(ValueError):
    """Raised without echoing the sensitive literal."""


def _replace_explicit_math(match: re.Match) -> str:
    source = match.group()
    if source.startswith("$$"):
        inner = source[2:-2]
    elif source.startswith("$"):
        inner = source[1:-1]
    else:
        inner = source[2:-2]
    stripped = inner.strip()
    return "" if is_strict_math_expression(stripped) else inner


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
            "" if is_strict_math_expression(run) else run
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


def _ascii_skeleton_candidate(value: str) -> str:
    tokens = [
        token.casefold()
        for token in _ASCII_ALNUM_RUN.findall(value)
        if token.casefold() not in _SKELETON_MATH_COMMANDS
    ]
    if not tokens:
        return ""
    skeleton = "".join(tokens)
    has_control_term = any(
        term in skeleton for term in _CONTROL_SKELETON_TERMS
    )
    has_contiguous_opaque = any(len(token) >= 7 for token in tokens)
    has_split_opaque = (
        len(tokens) >= 4
        and sum(
            token.isalpha() and len(token) <= 2 for token in tokens
        ) >= 4
    )
    if not (has_control_term or has_contiguous_opaque or has_split_opaque):
        return ""
    return skeleton


def _skeleton_fingerprints(value: str) -> Iterable[str]:
    candidate = _ascii_skeleton_candidate(value)
    size = REFERENCE_PROSE_FINGERPRINT_LENGTH
    for index in range(len(candidate) - size + 1):
        yield candidate[index : index + size]


def _short_skeleton(value: str) -> str:
    candidate = _ascii_skeleton_candidate(value)
    if _MIN_SHORT_SKELETON_LENGTH <= len(candidate) < (
        REFERENCE_PROSE_FINGERPRINT_LENGTH
    ):
        return candidate
    return ""


def _is_structural_field(
    field_name: str,
    *,
    check_identifiers: bool,
) -> bool:
    if field_name in _STRUCTURAL_FIELDS:
        return True
    is_identifier = (
        field_name.endswith("_id") or field_name.endswith("_ids")
    )
    return is_identifier and not check_identifiers


def _bounded_strings(
    value: Any,
    *,
    check_identifiers: bool = False,
) -> Iterable[Tuple[str, bool]]:
    remaining = _MAX_WALK_NODES
    aggregate_chars = 0
    stack = [(value, "")]
    while stack and remaining:
        current, field_name = stack.pop()
        remaining -= 1
        if isinstance(current, StrictMathText):
            continue
        if isinstance(current, str):
            aggregate_chars += len(current)
            if aggregate_chars > _MAX_AGGREGATE_TEXT_CHARS:
                raise ReferenceContentSafetyError(
                    "reference safety text bound exceeded"
                )
            yield current, _is_structural_field(
                field_name,
                check_identifiers=check_identifiers,
            )
        elif isinstance(current, BaseModel):
            stack.extend(
                (getattr(current, name), name)
                for name in type(current).model_fields
            )
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


def _bounded_typed_math(value: Any) -> Iterable[StrictMathText]:
    remaining = _MAX_WALK_NODES
    aggregate_chars = 0
    stack = [value]
    while stack and remaining:
        current = stack.pop()
        remaining -= 1
        if isinstance(current, StrictMathText):
            aggregate_chars += len(current)
            if aggregate_chars > _MAX_AGGREGATE_TEXT_CHARS:
                raise ReferenceContentSafetyError(
                    "reference safety text bound exceeded"
                )
            yield current
        elif isinstance(current, BaseModel):
            stack.extend(
                getattr(current, name)
                for name in type(current).model_fields
            )
        elif isinstance(current, Mapping):
            stack.extend(current.values())
        elif isinstance(current, Sequence) and not isinstance(
            current, (str, bytes, bytearray)
        ):
            stack.extend(current)
    if stack:
        raise ReferenceContentSafetyError(
            "reference safety traversal exceeded its bound"
        )


def _embedded_math_expressions(value: str) -> Iterable[str]:
    if is_strict_math_expression(value):
        yield value
        return
    for match in _EXPLICIT_MATH.finditer(value):
        source = match.group()
        inner = (
            source[2:-2]
            if source.startswith(("$$", r"\(", r"\["))
            else source[1:-1]
        )
        if is_strict_math_expression(inner.strip()):
            yield inner.strip()
    for match in _ASCII_MATH_RUN.finditer(value):
        run = match.group()
        has_math_signal = any(char.isdigit() for char in run) or any(
            marker in run
            for marker in (
                "\\",
                "=",
                "<",
                ">",
                "+",
                "-",
                "*",
                "/",
                "^",
                "_",
            )
        )
        if has_math_signal and is_strict_math_expression(run):
            yield run


@dataclass
class ReferenceSafetyPolicy:
    """Reject raw-only prose fingerprints while allowing mathematics."""

    sensitive_fingerprints: FrozenSet[str]
    sensitive_skeleton_fingerprints: FrozenSet[str]
    sensitive_short_skeletons: FrozenSet[str]
    authorized_identifiers: FrozenSet[str]
    authorized_long_numbers: FrozenSet[str]
    authorized_projection_fingerprints: set = field(default_factory=set)
    authorized_projection_skeleton_fingerprints: set = field(
        default_factory=set
    )
    authorized_projection_short_skeletons: set = field(default_factory=set)

    @classmethod
    def from_problem(cls, problem: ProblemInput) -> "ReferenceSafetyPolicy":
        raw = problem.reference_solution_text or ""
        public = problem.problem_text + "\n" + problem.reference_answer
        public_fingerprints = frozenset(_fingerprints(public))
        public_skeleton_fingerprints = frozenset(
            _skeleton_fingerprints(public)
        )
        public_short_skeleton = _short_skeleton(public)
        raw_short_skeleton = _short_skeleton(raw)
        return cls(
            sensitive_fingerprints=(
                frozenset(_fingerprints(raw)) - public_fingerprints
            ),
            sensitive_skeleton_fingerprints=(
                frozenset(_skeleton_fingerprints(raw))
                - public_skeleton_fingerprints
            ),
            sensitive_short_skeletons=frozenset(
                {raw_short_skeleton}
                - ({public_short_skeleton} if public_short_skeleton else set())
                - {""}
            ),
            authorized_identifiers=frozenset(math_identifiers(public)),
            authorized_long_numbers=frozenset(long_numeric_literals(public)),
        )

    def ensure_safe(
        self,
        value: Any,
        *,
        check_identifiers: bool = False,
    ) -> None:
        for expression in _bounded_typed_math(value):
            self._ensure_authorized_math(expression)
        for text, structural in _bounded_strings(
            value,
            check_identifiers=check_identifiers,
        ):
            if structural:
                continue
            for expression in _embedded_math_expressions(text):
                self._ensure_authorized_math(expression)
            prose_leak = any(
                item in self.sensitive_fingerprints
                and item not in self.authorized_projection_fingerprints
                for item in _fingerprints(text)
            )
            if prose_leak:
                raise ReferenceContentSafetyError(
                    "reference-only content crossed the safe boundary"
                )
            skeleton_leak = any(
                item in self.sensitive_skeleton_fingerprints
                and item
                not in self.authorized_projection_skeleton_fingerprints
                for item in _skeleton_fingerprints(text)
            )
            short_skeleton = _short_skeleton(text)
            short_leak = (
                short_skeleton in self.sensitive_short_skeletons
                and short_skeleton
                not in self.authorized_projection_short_skeletons
            )
            if skeleton_leak or short_leak:
                raise ReferenceContentSafetyError(
                    "reference-only content crossed the safe boundary"
                )

    def _ensure_authorized_math(self, expression: str) -> None:
        if not math_identifiers(expression).issubset(
            self.authorized_identifiers
        ) or not long_numeric_literals(expression).issubset(
            self.authorized_long_numbers
        ):
            raise ReferenceContentSafetyError(
                "unverified math content crossed the safe boundary"
            )

    def authorize_server_projection(self, value: Any) -> None:
        for text, structural in _bounded_strings(value):
            if not structural:
                self.authorized_projection_fingerprints.update(
                    _fingerprints(text)
                )
                self.authorized_projection_skeleton_fingerprints.update(
                    _skeleton_fingerprints(text)
                )
                short_skeleton = _short_skeleton(text)
                if short_skeleton:
                    self.authorized_projection_short_skeletons.add(
                        short_skeleton
                    )

    def sanitize_solution_trace(
        self,
        trace: SolutionTrace,
        teaching_route: FrozenTeachingRoute,
    ) -> SolutionTrace:
        """Retain typed analyst decisions and rebuild all free prose."""
        payload = trace.model_dump(mode="python")
        payload["audit_notes"] = []
        counters = {}
        anchors = [
            item["source_anchor"] for item in payload["assumptions"]
        ] + [item["source_anchor"] for item in payload["source_steps"]]
        for anchor in anchors:
            source_kind = anchor["source_kind"]
            counters[source_kind] = counters.get(source_kind, 0) + 1
            if source_kind != "verified_route":
                anchor["source_id"] = "%s-anchor-%03d" % (
                    source_kind,
                    counters[source_kind],
                )
            anchor["excerpt"] = (
                "已验证路线结构依据"
                if source_kind == "verified_route"
                else "题目与参考材料的结构依据"
            )
        for step in payload["source_steps"]:
            step["mathematical_action"] = render_typed_math_action(
                step["operation_kind"], step["operands"]
            )
            step["justification"] = render_typed_math_justification(
                step["operation_kind"]
            )
            step["new_information"] = step["state_after"]
        sanitized = SolutionTrace.model_validate(payload)
        self.authorize_server_projection(sanitized)
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
        self.ensure_safe(sanitized, check_identifiers=True)
        return sanitized
