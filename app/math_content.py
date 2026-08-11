"""Pure normalization and bounded math/display-content validation helpers."""

import re


_INLINE_MATH_SEGMENT = re.compile(r"\\\((.*?)\\\)")
_BLOCK_MATH_SEGMENT = re.compile(r"\\\[(.*?)\\\]")
MAX_GENERATED_DISPLAY_CONTENT_LENGTH = 500
_INTERNAL_CONTROL_SYNTAX = re.compile(
    r"(?:"
    r"\[\[|\]\]|"
    r"\{\{\s*(?:highlight|target|focus|emphasis|board)\b|"
    r"\\(?:htmlClass|htmlId|htmlStyle|htmlData|href|url|includegraphics)\b|"
    r"\b(?:class|style|data-[A-Za-z0-9_-]+)\s*=|"
    r"\.(?:is-highlighted|is-active|focus-target)\b|"
    r"#(?:board|problem|target|cue)[A-Za-z0-9_-]*\b|"
    r"\[data-[A-Za-z0-9_-]+\]"
    r")",
    re.IGNORECASE,
)
_MALFORMED_HTML_CONTROL_TAGS = {"div", "em", "mark", "span", "strong"}
_ASCII_TOKEN_CHARACTER = re.compile(r"[A-Za-z0-9_]")


def contains_math_markup(value: str) -> bool:
    """Preserve the legacy spoken-text math-markup boundary."""
    return "$" in value or bool(
        re.search(r"\\(?:[()[\]]|[A-Za-z]+)", value)
    )


def contains_internal_control_syntax(value: str) -> bool:
    """Return whether student-visible text contains runtime-only syntax."""
    return (
        _INTERNAL_CONTROL_SYNTAX.search(value) is not None
        or _contains_html_control_syntax(value)
    )


def _contains_html_control_syntax(value: str) -> bool:
    """Scan HTML-like controls in linear time without parsing general prose."""
    position = 0
    length = len(value)
    while position < length:
        opening = value.find("<", position)
        if opening < 0:
            return False
        cursor = opening + 1
        if cursor < length and value[cursor] == "/":
            cursor += 1
        if (
            cursor >= length
            or not value[cursor].isascii()
            or not value[cursor].isalpha()
        ):
            position = opening + 1
            continue
        name_start = cursor
        cursor += 1
        while cursor < length and (
            value[cursor].isascii()
            and (value[cursor].isalnum() or value[cursor] in ":-")
        ):
            cursor += 1
        if cursor < length and not (
            value[cursor].isspace() or value[cursor] in "/>"
        ):
            position = opening + 1
            continue
        tag_name = value[name_start:cursor].lower()
        if tag_name in _MALFORMED_HTML_CONTROL_TAGS:
            return True
        while cursor < length and value[cursor] not in "<>":
            cursor += 1
        if cursor < length and value[cursor] == ">":
            return True
        if cursor >= length:
            return False
        position = cursor
    return False


def normalize_choice_option_label(label: str) -> str:
    """Approximate browser display normalization for choice-label equality."""
    normalized = " ".join(label.split())
    normalized = _INLINE_MATH_SEGMENT.sub(
        lambda match: r"\(" + re.sub(r"\s+", "", match.group(1)) + r"\)",
        normalized,
    )
    return _BLOCK_MATH_SEGMENT.sub(
        lambda match: r"\[" + re.sub(r"\s+", "", match.group(1)) + r"\]",
        normalized,
    )


def normalize_grounded_choice_option_label(label: str) -> str:
    """Normalize visually equivalent, bounded KaTeX choice labels."""
    normalized = label.translate(
        str.maketrans(
            {
                "−": "-",
                "－": "-",
                "–": "-",
                "—": "-",
                "＋": "+",
                "×": "*",
                "·": "*",
                "÷": "/",
                "＝": "=",
                "≠": r"\ne",
                "²": "^2",
                "³": "^3",
            }
        )
    )
    for command, replacement in (
        (r"\left", ""),
        (r"\right", ""),
        (r"\dfrac", r"\frac"),
        (r"\tfrac", r"\frac"),
        (r"\times", "*"),
        (r"\cdot", "*"),
        (r"\div", "/"),
        (r"\neq", r"\ne"),
    ):
        normalized = normalized.replace(command, replacement)
    for delimiter in (r"\(", r"\)", r"\[", r"\]", "$"):
        normalized = normalized.replace(delimiter, "")
    for spacing in (
        r"\,",
        r"\;",
        r"\:",
        r"\!",
        r"\quad",
        r"\qquad",
        r"\enspace",
        r"\thinspace",
        r"\medspace",
        r"\thickspace",
    ):
        normalized = normalized.replace(spacing, "")
    normalized = re.sub(r"\s+", "", normalized)
    redundant_script_braces = re.compile(
        r"([_^])\{([A-Za-z0-9]|\\[A-Za-z]+)\}"
    )
    while True:
        reduced = redundant_script_braces.sub(r"\1\2", normalized)
        if reduced == normalized:
            return normalized
        normalized = reduced


def normalize_answer_leak_text(value: str) -> str:
    """Normalize only the bounded forms used by the leakage gate."""
    normalized = value.lower()
    normalized = re.sub(
        r"\\(?:left|right|text|mathrm|mathbf)",
        "",
        normalized,
    )
    return re.sub(
        r"[\s$\\()\[\]{}，。；：、,.!?！？;:]",
        "",
        normalized,
    )


def contains_bounded_answer_token(
    visible_value: str,
    private_value: str,
) -> bool:
    """Match a private value without treating prefixes as equal tokens."""
    visible = _normalize_bounded_answer_text(visible_value)
    private = _normalize_bounded_answer_text(private_value).strip()
    if not private:
        return False
    position = visible.find(private)
    while position >= 0:
        end = position + len(private)
        before_is_token = (
            position > 0
            and _ASCII_TOKEN_CHARACTER.fullmatch(visible[position - 1])
            is not None
            and _ASCII_TOKEN_CHARACTER.fullmatch(private[0]) is not None
        )
        after_is_token = (
            end < len(visible)
            and _ASCII_TOKEN_CHARACTER.fullmatch(visible[end]) is not None
            and _ASCII_TOKEN_CHARACTER.fullmatch(private[-1]) is not None
        )
        if not before_is_token and not after_is_token:
            return True
        position = visible.find(private, position + 1)
    return False


def _normalize_bounded_answer_text(value: str) -> str:
    normalized = value.lower()
    normalized = re.sub(
        r"\\(?:left|right|text|mathrm|mathbf)",
        "",
        normalized,
    )
    normalized = re.sub(
        r"\\(?:[()\[\]]|[A-Za-z]+)",
        lambda item: item.group()[1:],
        normalized,
    )
    normalized = re.sub(r"[$()\[\]{}]", "", normalized)
    normalized = re.sub(r"[\s，。；：、,.!?！？;:]+", " ", normalized)
    normalized = re.sub(r"\s*([=+\-*/^<>])\s*", r"\1", normalized)
    normalized = re.sub(r"\s*(和|或)\s*", r"\1", normalized)
    return normalized


def contains_explicit_choice_answer_leak(
    visible_value: str,
    correct_option_id: str,
    correct_label: str,
) -> bool:
    """Apply the legacy bounded choice-answer announcement detector."""
    visible = normalize_answer_leak_text(visible_value)
    option_id = normalize_answer_leak_text(correct_option_id)
    cue = r"(?:正确答案|答案|应选|选择)"
    if len(option_id) >= 4:
        explicit_option_id = option_id in visible
    else:
        option_id_pattern = (
            r"(?<![a-z0-9_+\-*/^=×÷−－–—＋])"
            + re.escape(option_id)
            + r"(?![a-z0-9_+\-*/^=×÷−－–—＋])"
        )
        explicit_option_id = re.search(
            rf"(?:"
            rf"(?:选择|应选|选){option_id_pattern}|"
            rf"(?:正确答案|答案)"
            rf"(?:就是|应该是|应为|是|为){option_id_pattern}|"
            rf"{option_id_pattern}(?:选项|项)|"
            rf"{option_id_pattern}"
            rf"(?:就是|应该是|应为|是|为)"
            rf"(?:正确答案|正确选项)"
            rf")",
            visible,
        ) is not None
        if not explicit_option_id:
            explicit_option_id = re.search(
                r"(?:correct|answer)(?:is)?"
                + re.escape(option_id)
                + r"(?![a-z0-9_+\-*/^=×÷−－–—＋])",
                visible,
            ) is not None
    if option_id and explicit_option_id:
        return True

    label = normalize_answer_leak_text(correct_label)
    if not label or not contains_bounded_answer_token(
        visible_value,
        correct_label,
    ):
        return False
    label_pattern = re.escape(label)
    return re.search(
        rf"(?:{cue}.{{0,32}}{label_pattern}|"
        rf"{label_pattern}.{{0,32}}{cue})",
        visible,
    ) is not None


def normalize_reference_text(value: str) -> str:
    """Normalize bounded math/reference text without semantic guessing."""
    normalized = value.translate(
        str.maketrans(
            {
                "×": "*",
                "÷": "/",
                "−": "-",
                "－": "-",
                "–": "-",
                "—": "-",
                "＋": "+",
            }
        )
    )
    normalized = re.sub(
        r"\\(?:left|right|,|;|!|quad|qquad)",
        "",
        normalized,
    )
    normalized = re.sub(
        r"\\(?:dfrac|tfrac|frac)"
        r"\{([A-Za-z0-9.+\-]+)\}"
        r"\{([A-Za-z0-9.+\-]+)\}",
        r"\1/\2",
        normalized,
    )
    normalized = re.sub(
        r"\\(?:dfrac|tfrac|frac)"
        r"([A-Za-z0-9])([A-Za-z0-9])",
        r"\1/\2",
        normalized,
    )
    superscript_digits = str.maketrans(
        "⁰¹²³⁴⁵⁶⁷⁸⁹",
        "0123456789",
    )
    normalized = re.sub(
        r"[⁰¹²³⁴⁵⁶⁷⁸⁹]+",
        lambda match: "^" + match.group().translate(superscript_digits),
        normalized,
    )
    return re.sub(
        r"[\s$\\()\[\]{}，。；：,;:]",
        "",
        normalized,
    )


def normalize_cross_artifact_math_identity(value: str) -> str:
    """Normalize presentation while preserving mathematical structure."""
    normalized = value.translate(
        str.maketrans(
            {
                "×": "*",
                "÷": "/",
                "−": "-",
                "－": "-",
                "–": "-",
                "—": "-",
                "＋": "+",
                "＝": "=",
                "，": ",",
                "；": ";",
                "：": ":",
                "≠": r"\ne",
            }
        )
    )
    for delimiter in (r"\(", r"\)", r"\[", r"\]", "$$", "$"):
        normalized = normalized.replace(delimiter, "")
    normalized = re.sub(
        r"\\(?:left|right|,|;|:|!|quad|qquad|enspace|thinspace|medspace|thickspace)",
        "",
        normalized,
    )
    normalized = normalized.replace(r"\{", "{").replace(r"\}", "}")
    normalized = re.sub(r"\\(?:dfrac|tfrac)", r"\\frac", normalized)
    normalized = re.sub(r"\s+", "", normalized)
    simple_fraction = re.compile(
        r"\\frac\{([A-Za-z0-9.]+)\}\{([A-Za-z0-9.]+)\}"
    )
    while True:
        reduced = simple_fraction.sub(r"\1/\2", normalized)
        if reduced == normalized:
            break
        normalized = reduced
    superscript_digits = str.maketrans(
        "⁰¹²³⁴⁵⁶⁷⁸⁹",
        "0123456789",
    )
    return re.sub(
        r"[⁰¹²³⁴⁵⁶⁷⁸⁹]+",
        lambda match: "^" + match.group().translate(superscript_digits),
        normalized,
    )


def contains_cross_artifact_math_identity(
    container_value: str,
    candidate_value: str,
) -> bool:
    """Bounded containment for a declared math target within a reference."""
    container = normalize_cross_artifact_math_identity(container_value)
    candidate = normalize_cross_artifact_math_identity(candidate_value)
    if not candidate:
        return False
    position = container.find(candidate)
    while position >= 0:
        end = position + len(candidate)
        before_is_token = (
            position > 0
            and _ASCII_TOKEN_CHARACTER.fullmatch(container[position - 1])
            is not None
            and _ASCII_TOKEN_CHARACTER.fullmatch(candidate[0]) is not None
        )
        after_is_token = (
            end < len(container)
            and _ASCII_TOKEN_CHARACTER.fullmatch(container[end]) is not None
            and _ASCII_TOKEN_CHARACTER.fullmatch(candidate[-1]) is not None
        )
        if not before_is_token and not after_is_token:
            return True
        position = container.find(candidate, position + 1)
    return False


def _balanced_pair(value: str, opening: str, closing: str) -> bool:
    position = 0
    inside = False
    while position < len(value):
        if value.startswith(opening, position):
            if inside:
                return False
            inside = True
            position += len(opening)
            continue
        if value.startswith(closing, position):
            if not inside:
                return False
            inside = False
            position += len(closing)
            continue
        position += 1
    return not inside


def _balanced_braces(value: str) -> bool:
    depth = 0
    for index, character in enumerate(value):
        if index and value[index - 1] == "\\":
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth < 0:
                return False
    return depth == 0


def is_valid_generated_display_content(value: str) -> bool:
    """Conservatively validate generated KaTeX/display text.

    This checks delimiters, braces, and internal runtime syntax. It is not a
    general LaTeX parser and deliberately does not judge mathematical truth.
    """
    if len(value) > MAX_GENERATED_DISPLAY_CONTENT_LENGTH:
        return False
    if contains_internal_control_syntax(value):
        return False

    families = []
    for opening, closing, name in (
        (r"\(", r"\)", "paren"),
        (r"\[", r"\]", "bracket"),
    ):
        present = opening in value or closing in value
        if present:
            families.append(name)
            if not _balanced_pair(value, opening, closing):
                return False

    without_display_dollars = value.replace("$$", "")
    has_display_dollars = "$$" in value
    has_inline_dollars = "$" in without_display_dollars
    if value.count("$$") % 2:
        return False
    if without_display_dollars.count("$") % 2:
        return False
    if has_display_dollars:
        families.append("display-dollar")
    if has_inline_dollars:
        families.append("inline-dollar")
    if len(families) > 1:
        return False
    if not _balanced_braces(value):
        return False
    return True
