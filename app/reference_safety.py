"""Server-owned boundary for untrusted reference-solution prose."""

import re
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, FrozenSet, Iterable, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel

from app.math_expression import (
    StrictMathText,
    deterministic_method_name,
    geometry_identifiers,
    is_strict_math_expression,
    math_identifiers,
    render_typed_math_action,
    render_typed_math_justification,
)
from app.math_content import (
    contains_cross_artifact_math_identity,
    normalize_cross_artifact_math_identity,
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
_LATEX_ENVIRONMENT_FRAGMENT = re.compile(
    r"\\begin\{(?P<environment>[A-Za-z]+)\}.*?"
    r"\\end\{(?P=environment)\}",
    re.DOTALL,
)
_LATEX_ENVIRONMENT_TOKEN = re.compile(
    r"\\(?:begin|end)\{(?P<environment>[A-Za-z]+)\}"
)
_ASCII_MATH_RUN = re.compile(r"[A-Za-z0-9\\{}()\[\]^_+*/=<>.\-]+")
_ASCII_ALNUM_RUN = re.compile(r"[A-Za-z0-9]+")
_SKELETON_MATH_COMMANDS = frozenset(
    {
        "angle",
        "because",
        "circ",
        "cong",
        "cos",
        "dfrac",
        "frac",
        "mathbb",
        "odot",
        "overline",
        "overrightarrow",
        "parallel",
        "perp",
        "sim",
        "sin",
        "sqrt",
        "tan",
        "therefore",
        "tfrac",
        "triangle",
        "vec",
        "widehat",
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
_UNICODE_DIGIT_TRANSLATION = str.maketrans(
    "⁰¹²³⁴⁵⁶⁷⁸⁹₀₁₂₃₄₅₆₇₈₉",
    "01234567890123456789",
)
_SERVER_GROUND_ID = re.compile(
    r"ground-(?:assumption|step|check)-\d{3}"
)
_PLAIN_MATH_FUNCTIONS = frozenset(
    {
        "cos",
        "cot",
        "csc",
        "exp",
        "gcd",
        "lcm",
        "ln",
        "log",
        "max",
        "min",
        "mod",
        "sec",
        "sin",
        "tan",
    }
)
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


def _canonical_math_streams(value: str) -> Tuple[str, str, str]:
    without_environments = value
    if is_strict_math_expression(value):
        without_environments = _LATEX_ENVIRONMENT_TOKEN.sub("", value)
    without_commands = re.sub(
        r"\\[A-Za-z]+",
        "",
        without_environments,
    )
    normalized = without_commands.translate(_UNICODE_DIGIT_TRANSLATION)
    alpha = []
    digits = []
    alnum = []
    for match in _ASCII_ALNUM_RUN.finditer(normalized):
        token = match.group().casefold()
        if token.isalpha() and token in _PLAIN_MATH_FUNCTIONS:
            continue
        for char in token:
            if char.isalpha():
                alpha.append(char)
                alnum.append(char)
            elif char.isdigit():
                digits.append(char)
                alnum.append(char)
    return "".join(alpha), "".join(digits), "".join(alnum)


def _malformed_environment_streams(value: str) -> Tuple[str, str, str]:
    names = [
        match.group("environment").casefold()
        for match in _LATEX_ENVIRONMENT_TOKEN.finditer(
            _candidate_source(value)
        )
    ]
    skeleton = "".join(names)
    return skeleton, "", skeleton


def _candidate_kind(token: str) -> str:
    if token.isalpha():
        return "alpha"
    if token.isdigit():
        return "digits"
    return "alnum"


def _candidate_is_long_enough(kind: str, skeleton: str) -> bool:
    return len(skeleton) >= (6 if kind == "digits" else 5)


def _is_complete_math_fragment(value: str) -> bool:
    stripped = value.strip()
    if not is_strict_math_expression(stripped):
        return False
    return (
        "\\" in stripped
        or any(char in stripped for char in "=<>+-*/^_()[]{}&")
        or any(not char.isascii() and not char.isalnum() for char in stripped)
    )


def _candidate_source(value: str) -> str:
    normalized = value.translate(_UNICODE_DIGIT_TRANSLATION).replace(
        "，", ","
    )
    if _is_complete_math_fragment(normalized):
        return ""

    def replace_fragment(match: re.Match) -> str:
        source = match.group()
        if source.startswith("$$"):
            inner = source[2:-2]
        elif source.startswith("$"):
            inner = source[1:-1]
        elif source.startswith((r"\(", r"\[")):
            inner = source[2:-2]
        else:
            inner = source
        return "界" if is_strict_math_expression(inner.strip()) else inner

    without_environments = _LATEX_ENVIRONMENT_FRAGMENT.sub(
        replace_fragment,
        normalized,
    )
    without_explicit_math = _EXPLICIT_MATH.sub(
        replace_fragment,
        without_environments,
    )
    pieces = []
    cursor = 0
    for match in _ASCII_MATH_RUN.finditer(without_explicit_math):
        pieces.append(without_explicit_math[cursor : match.start()])
        run = match.group()
        pieces.append("界" if _is_complete_math_fragment(run) else run)
        cursor = match.end()
    pieces.append(without_explicit_math[cursor:])
    return "".join(pieces)


def _sensitive_math_candidates(
    value: str,
    *,
    public_authorization: bool = False,
) -> Counter:
    candidates = Counter()
    source = _candidate_source(value)
    matches = list(_ASCII_ALNUM_RUN.finditer(source))
    for match in matches:
        token = match.group().casefold()
        kind = _candidate_kind(token)
        if _candidate_is_long_enough(kind, token):
            candidates["%s:%s" % (kind, token)] += 1

    group_tokens = []

    def flush_group() -> None:
        nonlocal group_tokens
        if len(group_tokens) > 1 and (
            not public_authorization
            or any(len(token) >= 3 for token in group_tokens)
        ):
            for start in range(len(group_tokens)):
                skeleton = ""
                for offset, token in enumerate(
                    group_tokens[start:],
                    start=start,
                ):
                    skeleton += token
                    if offset == start:
                        continue
                    kind = _candidate_kind(skeleton)
                    if _candidate_is_long_enough(
                        kind,
                        skeleton,
                    ):
                        candidates[
                            "%s:%s" % (kind, skeleton)
                        ] += 1
                        break
        group_tokens = []

    previous_end = 0
    for match in matches:
        token = match.group().casefold()
        separator = source[previous_end : match.start()]
        if any(char.isalnum() for char in separator):
            flush_group()
        group_tokens.append(token)
        previous_end = match.end()
    flush_group()
    return candidates


def _stream_positions(stream: str) -> dict:
    positions = {}
    for index, char in enumerate(stream):
        positions.setdefault(char, []).append(index)
    return positions


def _is_ordered_subsequence(candidate: str, positions: dict) -> bool:
    current = -1
    for char in candidate:
        choices = positions.get(char, [])
        choice_index = bisect_right(choices, current)
        if choice_index == len(choices):
            return False
        current = choices[choice_index]
    return True


def _contains_sensitive_math_candidate(
    streams: Tuple[str, str, str],
    sensitive_candidates: FrozenSet[str],
) -> bool:
    alpha, digits, alnum = streams
    positions_by_kind = {
        "alpha": _stream_positions(alpha),
        "digits": _stream_positions(digits),
        "alnum": _stream_positions(alnum),
    }
    for candidate in sensitive_candidates:
        kind, skeleton = candidate.split(":", 1)
        if _is_ordered_subsequence(
            skeleton,
            positions_by_kind[kind],
        ):
            return True
    return False


def _is_structural_field(
    field_name: str,
    value: str,
    *,
    check_identifiers: bool,
) -> bool:
    if field_name in _STRUCTURAL_FIELDS:
        return True
    is_identifier = (
        re.search(r"(?:^|_)ids?(?:_|$)", field_name) is not None
    )
    return is_identifier and (
        not check_identifiers
        or _SERVER_GROUND_ID.fullmatch(value) is not None
    )


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
                current,
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
                reversed(
                    [
                        getattr(current, name)
                        for name in type(current).model_fields
                    ]
                )
            )
        elif isinstance(current, Mapping):
            stack.extend(reversed(list(current.values())))
        elif isinstance(current, Sequence) and not isinstance(
            current, (str, bytes, bytearray)
        ):
            stack.extend(reversed(current))
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


def _problem_premise_text(problem_text: str) -> str:
    known_position = problem_text.find("已知")
    if known_position >= 0:
        boundary = len(problem_text)
        search_start = known_position + len("已知")
        for marker in ("则", "求", "问", "选项"):
            position = problem_text.find(marker, search_start)
            if position >= 0:
                boundary = min(boundary, position)
        return problem_text[known_position:boundary]
    boundary = len(problem_text)
    for marker in ("则", "求", "问", "下列", "选项"):
        position = problem_text.find(marker)
        if position >= 0:
            boundary = min(boundary, position)
    return problem_text[:boundary]


def _numeric_literals(value: str) -> set:
    normalized = normalize_cross_artifact_math_identity(value)
    return set(re.findall(r"\d+(?:\.\d+)?", normalized))


def _assumption_source_kind(problem_text: str, expression: str) -> str:
    premise = _problem_premise_text(problem_text)
    if contains_cross_artifact_math_identity(premise, expression):
        return "problem"
    normalized = normalize_cross_artifact_math_identity(expression)
    if normalized.count("=") != 1 or any(
        marker in normalized for marker in ("!=", ">=", "<=")
    ):
        return "solution"
    left, _ = normalized.split("=", 1)
    if re.fullmatch(r"[A-Za-z]", left) is None:
        return "solution"
    if not math_identifiers(expression).issubset(
        math_identifiers(premise)
    ):
        return "solution"
    if not _numeric_literals(expression).issubset(
        _numeric_literals(premise)
    ):
        return "solution"
    return "problem_derived"


@dataclass
class ReferenceSafetyPolicy:
    """Reject raw-only prose fingerprints while allowing mathematics."""

    sensitive_fingerprints: FrozenSet[str]
    sensitive_skeleton_fingerprints: FrozenSet[str]
    sensitive_short_skeletons: FrozenSet[str]
    sensitive_math_candidates: FrozenSet[str]
    authorized_geometry_identifiers: FrozenSet[str]
    authorized_math_identifiers: FrozenSet[str]
    public_problem_text: str
    authorized_helper_identifier: Optional[str] = None
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
        sensitive_math_candidates = (
            _sensitive_math_candidates(raw)
            - _sensitive_math_candidates(
                public,
                public_authorization=True,
            )
        )
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
            sensitive_math_candidates=frozenset(
                item
                for item, count in sensitive_math_candidates.items()
                if count > 0
            ),
            authorized_geometry_identifiers=frozenset(
                geometry_identifiers(public)
            ),
            authorized_math_identifiers=frozenset(
                math_identifiers(public)
            ),
            public_problem_text=problem.problem_text,
        )

    def ensure_safe(
        self,
        value: Any,
        *,
        check_identifiers: bool = False,
        downstream_of_sanitized_trace: bool = False,
    ) -> None:
        math_streams = [[], [], []]
        helper_identifier = self.authorized_helper_identifier
        for expression in _bounded_typed_math(value):
            if not downstream_of_sanitized_trace:
                helper_identifier = self._ensure_authorized_math(
                    expression,
                    helper_identifier,
                )
            for index, stream in enumerate(
                _canonical_math_streams(expression)
            ):
                math_streams[index].append(stream)
        for text, structural in _bounded_strings(
            value,
            check_identifiers=check_identifiers,
        ):
            if structural:
                continue
            for index, stream in enumerate(
                _malformed_environment_streams(text)
            ):
                math_streams[index].append(stream)
            for expression in _embedded_math_expressions(text):
                if not downstream_of_sanitized_trace:
                    helper_identifier = self._ensure_authorized_math(
                        expression,
                        helper_identifier,
                    )
                for index, stream in enumerate(
                    _canonical_math_streams(expression)
                ):
                    math_streams[index].append(stream)
            prose_leak = not downstream_of_sanitized_trace and any(
                item in self.sensitive_fingerprints
                and item not in self.authorized_projection_fingerprints
                for item in _fingerprints(text)
            )
            if prose_leak:
                raise ReferenceContentSafetyError(
                    "reference-only content crossed the safe boundary"
                )
            if downstream_of_sanitized_trace:
                downstream_skeleton = _ascii_skeleton_candidate(text)
                skeleton_leak = any(
                    term in downstream_skeleton
                    for term in _CONTROL_SKELETON_TERMS
                )
                short_leak = False
            else:
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
        aggregate_math_streams = tuple(
            "".join(items) for items in math_streams
        )
        if not downstream_of_sanitized_trace and _contains_sensitive_math_candidate(
            aggregate_math_streams,
            self.sensitive_math_candidates,
        ):
            raise ReferenceContentSafetyError(
                "unverified math content crossed the safe boundary"
            )
        self.authorized_helper_identifier = helper_identifier

    def _ensure_authorized_math(
        self,
        expression: str,
        helper_identifier: Optional[str],
    ) -> Optional[str]:
        if not geometry_identifiers(expression).issubset(
            self.authorized_geometry_identifiers
        ):
            raise ReferenceContentSafetyError(
                "unverified math content crossed the safe boundary"
            )
        novel_identifiers = (
            math_identifiers(expression)
            - self.authorized_math_identifiers
        )
        if helper_identifier is not None:
            novel_identifiers.discard(helper_identifier)
        if len(novel_identifiers) > 1:
            raise ReferenceContentSafetyError(
                "unverified math content crossed the safe boundary"
            )
        if novel_identifiers:
            helper = next(iter(novel_identifiers))
            if helper_identifier is not None:
                raise ReferenceContentSafetyError(
                    "unverified math content crossed the safe boundary"
                )
            helper_identifier = helper
        return helper_identifier

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
        payload["task_summary"] = "结构化数学路线"
        payload["method_name"] = deterministic_method_name(
            [
                item["operation_kind"]
                for item in payload["reasoning_steps"]
            ]
        )
        payload["audit_notes"] = []
        projected = ReferenceGroundingBrief.validate_for_reference_answer(
            payload,
            reference_answer,
        )
        self.ensure_safe(projected)
        payload = projected.model_dump(mode="python")
        assumption_id_map = {
            item["assumption_id"]: "ground-assumption-%03d" % index
            for index, item in enumerate(
                payload["assumptions"], start=1
            )
        }
        step_id_map = {
            item["step_id"]: "ground-step-%03d" % index
            for index, item in enumerate(
                payload["reasoning_steps"], start=1
            )
        }
        for item in payload["assumptions"]:
            item["assumption_id"] = assumption_id_map[
                item["assumption_id"]
            ]
            item["source_kind"] = _assumption_source_kind(
                self.public_problem_text,
                item["expression"],
            )
        for item in payload["reasoning_steps"]:
            item["step_id"] = step_id_map[item["step_id"]]
            item["assumption_ids_used"] = [
                assumption_id_map[assumption_id]
                for assumption_id in item["assumption_ids_used"]
            ]
        for index, item in enumerate(payload["check_requests"], start=1):
            item["check_id"] = "ground-check-%03d" % index
            item["source_step_id"] = step_id_map[
                item["source_step_id"]
            ]
        sanitized = ReferenceGroundingBrief.validate_for_reference_answer(
            payload,
            reference_answer,
        )
        self.ensure_safe(sanitized, check_identifiers=True)
        return sanitized
