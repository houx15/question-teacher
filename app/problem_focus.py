import re
from dataclasses import dataclass
from typing import Optional

from app.schemas import ProblemFocusTarget


MAX_PROBLEM_FOCUS_SOURCE_LENGTH = 4096
MAX_PROBLEM_FOCUS_TARGETS = 64
_DELIMITER_TOKEN = re.compile(r"\\(\(|\)|\[|\])|\$+")


@dataclass(frozen=True)
class _OpenDelimiter:
    closing_token: str
    content_start: int
    display_mode: bool


@dataclass(frozen=True)
class _MathToken:
    math_text: str
    display_mode: bool


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


def _is_escaped_dollar(source: str, index: int) -> bool:
    preceding_backslashes = 0
    index -= 1
    while index >= 0 and source[index] == "\\":
        preceding_backslashes += 1
        index -= 1
    return preceding_backslashes % 2 == 1
