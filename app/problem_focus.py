import re
from dataclasses import dataclass
from typing import Optional, Sequence

from app.schemas import ProblemFocusTarget


MAX_PROBLEM_FOCUS_SOURCE_LENGTH = 4096
MAX_PROBLEM_FOCUS_TARGETS = 64
MAX_REQUIRED_LEAD_EMPHASIS_TOKEN_LENGTH = 12
_DELIMITER_TOKEN = re.compile(r"\\(\(|\)|\[|\])|\$+")
_SPEAKABLE_ATOMIC_EXPRESSION = re.compile(
    r"-?(?:"
    r"\d+(?:\.\d+)?"
    r"|"
    r"(?:\d+(?:\.\d+)?)?(?:[A-Za-z](?:\^\d+)?)+"
    r")\Z"
)


@dataclass(frozen=True)
class _OpenDelimiter:
    closing_token: str
    content_start: int
    display_mode: bool


@dataclass(frozen=True)
class _MathToken:
    math_text: str
    display_mode: bool


@dataclass(frozen=True)
class RequiredLeadEmphasis:
    target_id: str
    spoken_token: str


def compile_problem_focus_targets(
    source: str,
) -> list[ProblemFocusTarget]:
    if (
        not isinstance(source, str)
        or len(source) > MAX_PROBLEM_FOCUS_SOURCE_LENGTH
    ):
        return []

    tokens: list[_MathToken] = []
    active: Optional[_OpenDelimiter] = None

    for match in _DELIMITER_TOKEN.finditer(source):
        raw_token = match.group(0)
        if raw_token.startswith("$"):
            offset = 0
            while offset < len(raw_token):
                dollar_index = match.start() + offset
                if _is_escaped_dollar(source, dollar_index):
                    offset += 1
                    continue

                remaining = len(raw_token) - offset

                if active is not None:
                    if active.closing_token not in {"$", "$$"}:
                        return []
                    delimiter_width = len(active.closing_token)
                    if remaining < delimiter_width:
                        return []
                    math_text = source[
                        active.content_start : dollar_index
                    ].strip()
                    if not math_text:
                        return []
                    tokens.append(
                        _MathToken(
                            math_text=math_text,
                            display_mode=active.display_mode,
                        )
                    )
                    if len(tokens) > MAX_PROBLEM_FOCUS_TARGETS:
                        return []
                    active = None
                    offset += delimiter_width
                    continue

                delimiter_width = 2 if remaining >= 2 else 1
                delimiter = "$" * delimiter_width
                active = _OpenDelimiter(
                    closing_token=delimiter,
                    content_start=dollar_index + delimiter_width,
                    display_mode=delimiter_width == 2,
                )
                offset += delimiter_width
            continue

        token = match.group(1)
        if active is not None:
            if token != active.closing_token:
                return []
            math_text = source[
                active.content_start : match.start()
            ].strip()
            if not math_text:
                return []
            tokens.append(
                _MathToken(
                    math_text=math_text,
                    display_mode=active.display_mode,
                )
            )
            if len(tokens) > MAX_PROBLEM_FOCUS_TARGETS:
                return []
            active = None
            continue

        if token in {")", "]"}:
            return []
        active = _OpenDelimiter(
            closing_token=(
                ")" if token == "(" else "]" if token == "[" else token
            ),
            content_start=match.end(),
            display_mode=token in {"[", "$$"},
        )

    if active is not None:
        return []

    return [
        ProblemFocusTarget(
            target_id=f"problem-math-{ordinal:03d}",
            math_text=token.math_text,
            display_mode=token.display_mode,
            ordinal=ordinal,
        )
        for ordinal, token in enumerate(tokens, start=1)
    ]


def required_lead_emphasis(
    targets: Sequence[ProblemFocusTarget],
) -> Optional[RequiredLeadEmphasis]:
    """Require only a first inline atomic token of at most 12 codepoints."""
    if len(targets) < 2 or targets[0].display_mode:
        return None
    token = targets[0].math_text.strip()
    if (
        not token
        or len(token) > MAX_REQUIRED_LEAD_EMPHASIS_TOKEN_LENGTH
        or "\\" in token
        or any(
            relation in token
            for relation in ("=", "<", ">", "≤", "≥", "≠", "!=")
        )
        or "+" in token
        or "-" in token[1:]
        or _SPEAKABLE_ATOMIC_EXPRESSION.fullmatch(token) is None
    ):
        return None
    return RequiredLeadEmphasis(
        target_id=targets[0].target_id,
        spoken_token=token,
    )


def required_lead_emphasis_target(
    targets: Sequence[ProblemFocusTarget],
) -> Optional[str]:
    requirement = required_lead_emphasis(targets)
    return requirement.target_id if requirement is not None else None


def _is_escaped_dollar(source: str, index: int) -> bool:
    preceding_backslashes = 0
    index -= 1
    while index >= 0 and source[index] == "\\":
        preceding_backslashes += 1
        index -= 1
    return preceding_backslashes % 2 == 1
