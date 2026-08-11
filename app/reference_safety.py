"""Server-owned boundary for untrusted reference-solution prose."""

import re
from dataclasses import dataclass
from typing import Any, FrozenSet, Iterable, Mapping, Sequence, Tuple

from pydantic import BaseModel

from app.preparation_models import SolutionTrace
from app.schemas import ProblemInput, ReferenceGroundingBrief


REFERENCE_PROSE_FINGERPRINT_LENGTH = 8
_OPAQUE_FINGERPRINT_LENGTH = 12
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
_FRAGMENT_BOUNDARY = re.compile(r"[\x00\r\n]+")
_OPAQUE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{7,}")
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


class ReferenceContentSafetyError(ValueError):
    """Raised without echoing the sensitive literal."""


def _looks_like_math(run: str) -> bool:
    strong_operator = any(char in run for char in "=+*/^<>")
    subtraction = (
        "-" in run
        and any(char.isdigit() for char in run)
        and re.search(r"[A-Za-z]{3,}", run) is None
    )
    return (
        strong_operator
        or subtraction
        or _LATEX_MATH_COMMAND.search(run) is not None
    )


def _looks_like_opaque_token(run: str) -> bool:
    return len(run) >= REFERENCE_PROSE_FINGERPRINT_LENGTH and any(
        char.isdigit() or char in "_-" for char in run
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
    return "\x00" if _looks_like_math(stripped) or simple_math else inner


def _prose_fragments(value: str) -> Iterable[str]:
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
            "\x00"
            if _looks_like_math(run) or _looks_like_opaque_token(run)
            else run
        )
        cursor = match.end()
    pieces.append(without_explicit_math[cursor:])
    for fragment in _FRAGMENT_BOUNDARY.split("".join(pieces)):
        normalized = "".join(
            char.casefold() for char in fragment if char.isalnum()
        )
        if normalized:
            yield normalized


def _fingerprints(value: str) -> Iterable[str]:
    size = REFERENCE_PROSE_FINGERPRINT_LENGTH
    for fragment in _prose_fragments(value):
        for index in range(len(fragment) - size + 1):
            yield fragment[index : index + size]


def _is_structural_field(field_name: str) -> bool:
    return (
        field_name in _STRUCTURAL_FIELDS
        or field_name.endswith("_id")
        or field_name.endswith("_ids")
    )


def _normalized_opaque_tokens(value: str) -> Iterable[str]:
    for token in _OPAQUE_TOKEN.findall(value):
        if (
            not _looks_like_math(token)
            and any(char.isdigit() or char in "_-" for char in token)
        ):
            yield token.casefold()


def _opaque_fingerprints(value: str) -> Iterable[str]:
    for token in _normalized_opaque_tokens(value):
        if len(token) < _OPAQUE_FINGERPRINT_LENGTH:
            yield token
            continue
        for index in range(len(token) - _OPAQUE_FINGERPRINT_LENGTH + 1):
            yield token[index : index + _OPAQUE_FINGERPRINT_LENGTH]


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
    sensitive_opaque_fingerprints: FrozenSet[str]

    @classmethod
    def from_problem(cls, problem: ProblemInput) -> "ReferenceSafetyPolicy":
        raw = problem.reference_solution_text or ""
        public = problem.problem_text + "\n" + problem.reference_answer
        public_fingerprints = frozenset(_fingerprints(public))
        public_opaque_fingerprints = frozenset(_opaque_fingerprints(public))
        return cls(
            sensitive_fingerprints=(
                frozenset(_fingerprints(raw)) - public_fingerprints
            ),
            sensitive_opaque_fingerprints=(
                frozenset(_opaque_fingerprints(raw))
                - public_opaque_fingerprints
            ),
        )

    def ensure_safe(self, value: Any) -> None:
        if (
            not self.sensitive_fingerprints
            and not self.sensitive_opaque_fingerprints
        ):
            return
        for text, structural in _bounded_strings(value):
            opaque_leak = any(
                item in self.sensitive_opaque_fingerprints
                for item in _opaque_fingerprints(text)
            )
            prose_leak = not structural and any(
                item in self.sensitive_fingerprints
                for item in _fingerprints(text)
            )
            if opaque_leak or prose_leak:
                raise ReferenceContentSafetyError(
                    "reference-only content crossed the safe boundary"
                )

    def sanitize_solution_trace(self, trace: SolutionTrace) -> SolutionTrace:
        payload = trace.model_dump(mode="python")
        payload["audit_notes"] = []
        anchors = [
            item["source_anchor"] for item in payload["assumptions"]
        ] + [item["source_anchor"] for item in payload["source_steps"]]
        counters = {}
        labels = {
            "problem": "题目结构依据",
            "answer": "参考答案结构依据",
            "solution": "参考解析结构依据",
            "verified_route": "已验证路线结构依据",
        }
        for anchor in anchors:
            source_kind = anchor["source_kind"]
            counters[source_kind] = counters.get(source_kind, 0) + 1
            if source_kind != "verified_route":
                anchor["source_id"] = "%s-anchor-%03d" % (
                    source_kind,
                    counters[source_kind],
                )
            anchor["excerpt"] = labels[source_kind]
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
