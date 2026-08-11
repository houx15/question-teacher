"""Strict, bounded declassification for model-authored mathematics."""

import re
from typing import Literal

from pydantic_core import core_schema

from app.math_content import contains_internal_control_syntax


MAX_STRICT_MATH_EXPRESSION_LENGTH = 500

MathOperationKind = Literal[
    "identify",
    "substitute",
    "eliminate",
    "expand",
    "factor",
    "combine_like_terms",
    "simplify",
    "rearrange",
    "add",
    "subtract",
    "multiply",
    "divide",
    "apply_identity",
    "complete_square",
    "quadratic_formula",
    "back_substitute",
    "square",
    "take_square_root",
    "split_cases",
    "compare",
    "derive",
    "conclude",
]
ReasoningGapCode = Literal[
    "missing_justification",
    "implicit_substitution",
    "implicit_equivalence",
    "implicit_nonzero_condition",
    "implicit_case_split",
    "implicit_identity",
    "implicit_domain_restriction",
    "nonzero_condition_required",
    "domain_condition_required",
    "branch_completeness_required",
    "back_substitution_required",
    "reference_omits_step",
]

_MATH_COMMANDS = frozenset(
    {
        "alpha",
        "angle",
        "beta",
        "because",
        "cdot",
        "circ",
        "cong",
        "cos",
        "cot",
        "csc",
        "delta",
        "div",
        "dfrac",
        "epsilon",
        "exp",
        "frac",
        "gamma",
        "ge",
        "geq",
        "in",
        "infty",
        "lambda",
        "le",
        "left",
        "leq",
        "ln",
        "log",
        "max",
        "mathbb",
        "min",
        "mp",
        "mu",
        "ne",
        "neq",
        "notin",
        "odot",
        "omega",
        "overline",
        "overrightarrow",
        "parallel",
        "perp",
        "phi",
        "pi",
        "pm",
        "psi",
        "qquad",
        "quad",
        "rho",
        "right",
        "sec",
        "sigma",
        "sim",
        "sin",
        "sqrt",
        "square",
        "subset",
        "subseteq",
        "sum",
        "tan",
        "tau",
        "theta",
        "therefore",
        "tfrac",
        "times",
        "triangle",
        "union",
        "varphi",
        "vec",
        "widehat",
    }
)
_PLAIN_FUNCTIONS = frozenset(
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
_LATEX_ENVIRONMENTS = (
    "aligned",
    "array",
    "bmatrix",
    "cases",
    "matrix",
    "pmatrix",
)
_LATEX_ENVIRONMENT_TOKEN = re.compile(
    r"\\(begin|end)\{([A-Za-z]+)\}"
)
_LEFT_RIGHT_TOKEN = re.compile(r"\\(left|right)\b")
_CJK_MATH_TOKENS = (
    ("大于等于", ">="),
    ("小于等于", "<="),
    ("不等于", "!="),
    ("大于", ">"),
    ("小于", "<"),
    ("或", "|"),
    ("且", "&"),
)
_ALLOWED_ASCII_PUNCTUATION = frozenset(
    "+-*/^_=<>!.,;:&|%()[]{}'"
)
_ALLOWED_UNICODE_MATH = frozenset(
    "±¹²³·×÷ΓΔΠΣΦΨΩαβγδεθλμπρστφψω"
    "⁰ⁱ⁴⁵⁶⁷⁸⁹⁺⁻ⁿ"
    "₀₁₂₃₄₅₆₇₈₉₊₋"
    "→⇒∅∈∉∑−∓√∞∩∪≈≠≤≥⊂⊆"
    "∠°⊥∥△≅∽∵∴⊙□▱"
)
_ASCII_WORD = re.compile(r"[A-Za-z]+")
_ASCII_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_OPAQUE_MIXED_ALNUM = re.compile(r"[A-Za-z0-9]{7,}")
_GEOMETRY_COMMAND = re.compile(
    r"\\(?:angle|overline|overrightarrow|parallel|perp|triangle|"
    r"cong|sim|square|vec|widehat)\b"
)
_GEOMETRY_RELATION_MARKERS = frozenset(
    "=/:\u2220⊥∥△≅∽∵∴⊙□▱"
)
_CONTROL_SKELETON_TERMS = (
    "ignore",
    "rules",
    "secret",
    "hidden",
    "confidential",
    "private",
    "prompt",
    "system",
    "token",
)
_EXACTLY_ONE_OPERAND = frozenset(
    {
        "add",
        "subtract",
        "multiply",
        "divide",
        "apply_identity",
        "complete_square",
    }
)
_ONE_TO_FOUR_OPERANDS = frozenset(
    {"substitute", "eliminate", "compare", "back_substitute"}
)
_ZERO_OPERANDS = frozenset(
    {
        "identify",
        "expand",
        "combine_like_terms",
        "simplify",
        "rearrange",
        "quadratic_formula",
        "square",
        "take_square_root",
        "split_cases",
        "derive",
        "conclude",
    }
)


class StrictMathExpressionError(ValueError):
    """Raised without including rejected model-authored content."""


def _strip_math_delimiters(value: str) -> str:
    stripped = value.strip()
    delimiters = (("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]"))
    changed = True
    while changed:
        changed = False
        for opening, closing in delimiters:
            if (
                stripped.startswith(opening)
                and stripped.endswith(closing)
                and len(stripped) > len(opening) + len(closing)
            ):
                stripped = stripped[len(opening) : -len(closing)].strip()
                changed = True
                break
    return stripped


def _normalized_for_scan(value: str) -> str:
    normalized = _strip_math_delimiters(value)
    for environment in _LATEX_ENVIRONMENTS:
        normalized = normalized.replace(r"\begin{%s}" % environment, "")
        normalized = normalized.replace(r"\end{%s}" % environment, "")
    for source, target in _CJK_MATH_TOKENS:
        normalized = normalized.replace(source, target)
    return normalized.replace("，", ",").replace("；", ";")


def _validate_latex_structure(value: str) -> None:
    inner = _strip_math_delimiters(value)
    if "$" in inner or any(
        marker in inner for marker in (r"\(", r"\)", r"\[", r"\]")
    ):
        raise StrictMathExpressionError("invalid strict math expression")

    environment_stack = []
    for match in _LATEX_ENVIRONMENT_TOKEN.finditer(value):
        direction, environment = match.groups()
        if environment not in _LATEX_ENVIRONMENTS:
            raise StrictMathExpressionError("invalid strict math expression")
        if direction == "begin":
            environment_stack.append(environment)
        elif not environment_stack or environment_stack.pop() != environment:
            raise StrictMathExpressionError("invalid strict math expression")
    if environment_stack:
        raise StrictMathExpressionError("invalid strict math expression")

    left_right_depth = 0
    for match in _LEFT_RIGHT_TOKEN.finditer(inner):
        if match.group(1) == "left":
            left_right_depth += 1
        elif left_right_depth == 0:
            raise StrictMathExpressionError("invalid strict math expression")
        else:
            left_right_depth -= 1
    if left_right_depth:
        raise StrictMathExpressionError("invalid strict math expression")


def _ascii_letter_tokens(source: str) -> list:
    return [
        token
        for token in _ASCII_WORD.findall(source)
        if token.casefold() not in _MATH_COMMANDS
        and token.casefold() not in _PLAIN_FUNCTIONS
    ]


def _reject_split_control_tokens(source: str) -> None:
    tokens = _ascii_letter_tokens(source)
    skeleton = "".join(tokens).casefold()
    if any(term in skeleton for term in _CONTROL_SKELETON_TERMS):
        raise StrictMathExpressionError("invalid strict math expression")
    if sum(
        len(token) == 2 and token.islower() for token in tokens
    ) >= 4:
        raise StrictMathExpressionError("invalid strict math expression")


def _is_geometry_point_name(word: str, source: str) -> bool:
    has_geometry_context = (
        any(marker in source for marker in _GEOMETRY_RELATION_MARKERS)
        or _GEOMETRY_COMMAND.search(source) is not None
    )
    return (
        has_geometry_context
        and word.isupper()
        and (
            2 <= len(word) <= 3
            or (
                len(word) == 4
                and any(
                    marker in source
                    for marker in (r"\square", "□", "▱")
                )
            )
        )
    )


def _balanced_group_end(
    source: str,
    start: int,
    opening: str,
    closing: str,
) -> int:
    if start >= len(source) or source[start] != opening:
        raise StrictMathExpressionError("invalid strict math expression")
    depth = 0
    index = start
    while index < len(source):
        if source[index] == opening:
            depth += 1
        elif source[index] == closing:
            depth -= 1
            if depth == 0:
                return index + 1
        index += 1
    raise StrictMathExpressionError("invalid strict math expression")


def _validate_command_arguments(
    source: str,
    command: str,
    command_end: int,
) -> None:
    cursor = command_end
    while cursor < len(source) and source[cursor].isspace():
        cursor += 1
    if command in {"frac", "dfrac", "tfrac"}:
        if cursor < len(source) and source[cursor] == "{":
            first_end = _balanced_group_end(source, cursor, "{", "}")
            if not source[cursor + 1 : first_end - 1].strip():
                raise StrictMathExpressionError(
                    "invalid strict math expression"
                )
            cursor = first_end
            while cursor < len(source) and source[cursor].isspace():
                cursor += 1
            second_end = _balanced_group_end(source, cursor, "{", "}")
            if not source[cursor + 1 : second_end - 1].strip():
                raise StrictMathExpressionError(
                    "invalid strict math expression"
                )
            return
        legacy = source[cursor : cursor + 2]
        if len(legacy) != 2 or not all(
            item.isascii() and item.isalnum() for item in legacy
        ):
            raise StrictMathExpressionError(
                "invalid strict math expression"
            )
    elif command in {
        "sqrt",
        "overline",
        "overrightarrow",
        "vec",
        "widehat",
    }:
        if cursor < len(source) and source[cursor] == "[":
            cursor = _balanced_group_end(source, cursor, "[", "]")
            while cursor < len(source) and source[cursor].isspace():
                cursor += 1
        end = _balanced_group_end(source, cursor, "{", "}")
        if not source[cursor + 1 : end - 1].strip():
            raise StrictMathExpressionError(
                "invalid strict math expression"
            )
    elif command == "mathbb":
        end = _balanced_group_end(source, cursor, "{", "}")
        if source[cursor + 1 : end - 1].strip() not in {
            "C",
            "N",
            "Q",
            "R",
            "Z",
        }:
            raise StrictMathExpressionError(
                "invalid strict math expression"
            )
    elif command in {"left", "right"}:
        direct_delimiters = (
            "([{|."
            if command == "left"
            else ")]}|."
        )
        if cursor >= len(source):
            raise StrictMathExpressionError(
                "invalid strict math expression"
            )
        if source[cursor] == "\\":
            if (
                cursor + 1 >= len(source)
                or source[cursor + 1] not in "{}|"
            ):
                raise StrictMathExpressionError(
                    "invalid strict math expression"
                )
        elif source[cursor] not in direct_delimiters:
            raise StrictMathExpressionError(
                "invalid strict math expression"
            )
    elif command == "sum":
        if cursor >= len(source) or source[cursor] not in "_^":
            raise StrictMathExpressionError(
                "invalid strict math expression"
            )
        cursor += 1
        if cursor >= len(source):
            raise StrictMathExpressionError(
                "invalid strict math expression"
            )
        if source[cursor] == "{":
            _balanced_group_end(source, cursor, "{", "}")
        elif not source[cursor].isalnum():
            raise StrictMathExpressionError(
                "invalid strict math expression"
            )


def validate_strict_math_expression(value: str) -> str:
    """Return a bounded math expression or fail without echoing it."""
    if not isinstance(value, str):
        raise StrictMathExpressionError("strict math expression must be text")
    stripped = value.strip()
    if (
        not stripped
        or len(stripped) > MAX_STRICT_MATH_EXPRESSION_LENGTH
        or contains_internal_control_syntax(stripped)
    ):
        raise StrictMathExpressionError("invalid strict math expression")
    _validate_latex_structure(stripped)
    source = _normalized_for_scan(stripped)
    if (
        "://" in source
        or ".." in source
        or re.search(r"\d\s+\.\d", source)
        or re.search(r"(?:\+\+|\*\*|//|\^\^|==|[+*/^][+*/^])", source)
    ):
        raise StrictMathExpressionError("invalid strict math expression")
    _reject_split_control_tokens(source)
    if any(
        any(char.isalpha() for char in match.group())
        and any(char.isdigit() for char in match.group())
        for match in _OPAQUE_MIXED_ALNUM.finditer(source)
    ):
        raise StrictMathExpressionError("invalid strict math expression")
    stack = []
    saw_math_atom = False
    index = 0
    while index < len(source):
        char = source[index]
        if char.isspace() or char == "$":
            index += 1
            continue
        if char.isascii() and char.isdigit():
            saw_math_atom = True
            match = _ASCII_NUMBER.match(source, index)
            assert match is not None
            index = match.end()
            if index < len(source) and source[index] == ".":
                raise StrictMathExpressionError(
                    "invalid strict math expression"
                )
            continue
        if char.isascii() and char.isalpha():
            match = _ASCII_WORD.match(source, index)
            assert match is not None
            word = match.group()
            if not (
                len(word) == 1
                or word in _PLAIN_FUNCTIONS
                or (len(word) == 2 and word.islower())
                or _is_geometry_point_name(word, source)
            ):
                raise StrictMathExpressionError("invalid strict math expression")
            saw_math_atom = True
            index = match.end()
            continue
        if char == "\\":
            if index + 1 >= len(source):
                raise StrictMathExpressionError("invalid strict math expression")
            following = source[index + 1]
            if following in "\\()[]{} ,;!":
                index += 2
                continue
            match = _ASCII_WORD.match(source, index + 1)
            if match is None or match.group() not in _MATH_COMMANDS:
                raise StrictMathExpressionError("invalid strict math expression")
            _validate_command_arguments(
                source,
                match.group(),
                match.end(),
            )
            saw_math_atom = True
            if match.group() in {"left", "right"}:
                index = match.end()
                while index < len(source) and source[index].isspace():
                    index += 1
                index += 2 if source[index] == "\\" else 1
            else:
                index = match.end()
            continue
        if char in "([{":
            stack.append(char)
            index += 1
            continue
        if char in ")]}":
            expected = {")": "(", "]": "[", "}": "{"}[char]
            if not stack or stack.pop() != expected:
                raise StrictMathExpressionError("invalid strict math expression")
            index += 1
            continue
        if char in _ALLOWED_ASCII_PUNCTUATION:
            index += 1
            continue
        if char in _ALLOWED_UNICODE_MATH:
            saw_math_atom = True
            index += 1
            continue
        raise StrictMathExpressionError("invalid strict math expression")
    if stack or not saw_math_atom:
        raise StrictMathExpressionError("invalid strict math expression")
    return stripped


def validate_operation_operands(
    operation_kind: MathOperationKind,
    operands: list,
) -> None:
    count = len(operands)
    if (
        operation_kind in _EXACTLY_ONE_OPERAND
        and count != 1
    ) or (
        operation_kind in _ONE_TO_FOUR_OPERANDS
        and not 1 <= count <= 4
    ) or (
        operation_kind in _ZERO_OPERANDS
        and count != 0
    ):
        raise ValueError("operation operands do not match operation kind")


def allowed_gap_codes_for_operation(
    operation_kind: MathOperationKind,
) -> list:
    baseline = ["missing_justification", "reference_omits_step"]
    mapping = {
        "substitute": ["implicit_substitution"],
        "eliminate": ["implicit_equivalence"],
        "expand": ["implicit_equivalence"],
        "factor": ["implicit_equivalence"],
        "combine_like_terms": ["implicit_equivalence"],
        "simplify": ["implicit_equivalence"],
        "rearrange": ["implicit_equivalence"],
        "add": ["implicit_equivalence"],
        "subtract": ["implicit_equivalence"],
        "multiply": ["implicit_equivalence"],
        "divide": [
            "implicit_nonzero_condition",
            "nonzero_condition_required",
        ],
        "apply_identity": ["implicit_identity"],
        "complete_square": ["implicit_identity"],
        "take_square_root": [
            "implicit_domain_restriction",
            "domain_condition_required",
            "branch_completeness_required",
        ],
        "split_cases": [
            "implicit_case_split",
            "branch_completeness_required",
        ],
        "back_substitute": ["back_substitution_required"],
    }
    return baseline + list(mapping.get(operation_kind, []))


def is_strict_math_expression(value: str) -> bool:
    try:
        validate_strict_math_expression(value)
    except StrictMathExpressionError:
        return False
    return True


def math_identifiers(value: str) -> set:
    source = _normalized_for_scan(value)
    identifiers = set()
    for token in _ascii_letter_tokens(source):
        if len(token) == 1:
            identifiers.add(token)
        elif len(token) == 2 and token.islower():
            identifiers.update(token)
        elif token.isupper() and 2 <= len(token) <= 3:
            identifiers.update(token)
    return identifiers


def geometry_identifiers(value: str) -> set:
    source = _normalized_for_scan(value)
    return {
        token
        for token in _ascii_letter_tokens(source)
        if _is_geometry_point_name(token, source)
    }


def long_numeric_literals(value: str) -> set:
    return {
        item
        for item in re.findall(r"(?<![A-Za-z0-9])\d{8,}(?![A-Za-z0-9])", value)
    }


class StrictMathText(str):
    """Nominal string marking content admitted by the strict math lexer."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        del source_type, handler
        return core_schema.no_info_after_validator_function(
            cls._validate,
            core_schema.str_schema(
                strip_whitespace=True,
                min_length=1,
                max_length=MAX_STRICT_MATH_EXPRESSION_LENGTH,
            ),
        )

    @classmethod
    def _validate(cls, value: str) -> "StrictMathText":
        return cls(validate_strict_math_expression(value))


StrictMathExpression = StrictMathText


_OPERATION_LABELS = {
    "identify": "识别数学结构",
    "substitute": "代入已知数学量",
    "eliminate": "消去中间数学量",
    "expand": "展开数学式",
    "factor": "因式分解",
    "combine_like_terms": "合并同类项",
    "simplify": "化简数学式",
    "rearrange": "整理数学关系",
    "add": "在等式两边加上同一个数学量",
    "subtract": "在等式两边减去同一个数学量",
    "multiply": "在关系两边乘以同一个数学量",
    "divide": "在等式两边除以同一个非零数学量",
    "apply_identity": "应用数学恒等式",
    "complete_square": "配成完全平方",
    "quadratic_formula": "应用一元二次方程求根公式",
    "back_substitute": "回代已得数学量",
    "square": "对数学关系作平方变换",
    "take_square_root": "对平方关系开平方",
    "split_cases": "将数学关系分情况讨论",
    "compare": "比较数学关系",
    "derive": "依据已知数学关系推导",
    "conclude": "回到题目目标并得出结论",
}


def render_typed_math_action(
    operation_kind: MathOperationKind,
    operands: list,
) -> str:
    label = _OPERATION_LABELS[operation_kind]
    if not operands:
        return label
    return "%s：%s" % (label, "，".join(operands))


def render_typed_math_justification(
    operation_kind: MathOperationKind,
) -> str:
    del operation_kind
    return "该操作由类型化数学依赖与前后状态支持"


def deterministic_method_name(
    operation_kinds: list,
) -> str:
    priorities = (
        ("eliminate", "消元法"),
        ("substitute", "代入法"),
        ("factor", "因式分解法"),
        ("take_square_root", "开平方法"),
        ("complete_square", "配方法"),
        ("quadratic_formula", "求根公式法"),
        ("apply_identity", "恒等变换"),
    )
    for operation_kind, method_name in priorities:
        if operation_kind in operation_kinds:
            return method_name
    return "结构化推理"
