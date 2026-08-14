"""Deterministic, bounded speech for a deliberately small math subset."""

import re
from typing import List, NamedTuple, Optional, Tuple

from app.math_content import contains_internal_control_syntax
from app.math_expression import (
    MAX_STRICT_MATH_EXPRESSION_LENGTH,
    is_strict_math_expression,
)


MAX_DISPLAY_MATH_SCAN_LENGTH = 2048


class MathSpeechError(ValueError):
    """A stable, content-free failure at the display-to-speech boundary."""

    def __init__(self, code: str = "unsupported_math_speech") -> None:
        super().__init__(code)
        self.code = code


class _Token(NamedTuple):
    kind: str
    value: str


_COMMANDS = {
    "cdot": "*",
    "times": "*",
    "ne": "!=",
    "neq": "!=",
    "ge": ">=",
    "geq": ">=",
    "le": "<=",
    "leq": "<=",
}
_FUNCTION_SPEECH = {
    "cos": "余弦",
    "cot": "余切",
    "csc": "余割",
    "exp": "指数函数",
    "gcd": "最大公约数",
    "lcm": "最小公倍数",
    "ln": "自然对数",
    "log": "对数",
    "max": "最大值",
    "min": "最小值",
    "mod": "模",
    "sec": "正割",
    "sin": "正弦",
    "tan": "正切",
}
_BINARY_PRECEDENCE = {
    "=": 1,
    "!=": 1,
    ">": 1,
    "<": 1,
    ">=": 1,
    "<=": 1,
    "+": 2,
    "-": 2,
    "*": 3,
}
_ATOM_STARTS = {"NUMBER", "VARIABLE", "FUNCTION", "LPAREN", "FRAC"}


def _unsupported() -> MathSpeechError:
    return MathSpeechError("unsupported_math_speech")


def extract_display_math(value: str) -> List[str]:
    """Extract supported, non-nested display math delimiters in one pass."""
    if (
        not isinstance(value, str)
        or len(value) > MAX_DISPLAY_MATH_SCAN_LENGTH
        or contains_internal_control_syntax(value)
    ):
        raise _unsupported()
    segments: List[str] = []
    index = 0
    length = len(value)
    while index < length:
        if value.startswith("$$", index):
            raise _unsupported()
        if value[index] == "$":
            closing = value.find("$", index + 1)
            if closing < 0 or value.startswith("$$", closing):
                raise _unsupported()
            inner = value[index + 1 : closing]
            index = closing + 1
        elif value.startswith(r"\(", index):
            closing = value.find(r"\)", index + 2)
            if closing < 0:
                raise _unsupported()
            inner = value[index + 2 : closing]
            index = closing + 2
        elif value.startswith(r"\[", index):
            closing = value.find(r"\]", index + 2)
            if closing < 0:
                raise _unsupported()
            inner = value[index + 2 : closing]
            index = closing + 2
        elif value.startswith((r"\)", r"\]"), index):
            raise _unsupported()
        else:
            index += 1
            continue
        if (
            not inner.strip()
            or len(inner.strip()) > MAX_STRICT_MATH_EXPRESSION_LENGTH
            or "$" in inner
            or any(marker in inner for marker in (r"\(", r"\)", r"\[", r"\]"))
        ):
            raise _unsupported()
        segments.append(inner.strip())
    return segments


def _single_expression(value: str) -> str:
    if not isinstance(value, str):
        raise _unsupported()
    segments = extract_display_math(value)
    if segments:
        if len(segments) != 1:
            raise _unsupported()
        stripped = value.strip()
        is_single_wrapper = (
            (
                stripped.startswith("$")
                and not stripped.startswith("$$")
                and stripped.endswith("$")
            )
            or (
                stripped.startswith(r"\(")
                and stripped.endswith(r"\)")
            )
            or (
                stripped.startswith(r"\[")
                and stripped.endswith(r"\]")
            )
        )
        if not is_single_wrapper:
            raise _unsupported()
        return segments[0]
    expression = value.strip()
    if len(expression) > MAX_STRICT_MATH_EXPRESSION_LENGTH:
        raise _unsupported()
    return expression


def _tokenize(expression: str) -> List[_Token]:
    tokens: List[_Token] = []
    index = 0
    while index < len(expression):
        char = expression[index]
        if char.isspace():
            index += 1
            continue
        if char.isascii() and char.isdigit():
            end = index + 1
            while end < len(expression) and expression[end].isdigit():
                end += 1
            tokens.append(_Token("NUMBER", expression[index:end]))
            index = end
            continue
        if char.isascii() and char.isalpha():
            end = index + 1
            while end < len(expression) and expression[end].isalpha():
                end += 1
            word = expression[index:end]
            if len(word) == 1:
                tokens.append(_Token("VARIABLE", word))
            elif word in _FUNCTION_SPEECH:
                tokens.append(_Token("FUNCTION", word))
            else:
                raise _unsupported()
            index = end
            continue
        if char == "\\":
            match = re.match(r"\\([A-Za-z]+)", expression[index:])
            if match is None:
                raise _unsupported()
            command = match.group(1)
            index += len(match.group(0))
            if command in _COMMANDS:
                tokens.append(_Token("OP", _COMMANDS[command]))
            elif command in {"frac", "dfrac", "tfrac"}:
                tokens.append(_Token("FRAC", command))
            elif command in _FUNCTION_SPEECH:
                tokens.append(_Token("FUNCTION", command))
            else:
                raise _unsupported()
            continue
        two = expression[index : index + 2]
        if two in {"!=", ">=", "<="}:
            tokens.append(_Token("OP", two))
            index += 2
            continue
        translated = {
            "≠": "!=",
            "≥": ">=",
            "≤": "<=",
            "×": "*",
            "·": "*",
        }.get(char, char)
        if translated in _BINARY_PRECEDENCE or translated == "^":
            tokens.append(_Token("OP", translated))
        elif char == "(":
            tokens.append(_Token("LPAREN", char))
        elif char == ")":
            tokens.append(_Token("RPAREN", char))
        elif char == "{":
            tokens.append(_Token("LBRACE", char))
        elif char == "}":
            tokens.append(_Token("RBRACE", char))
        else:
            raise _unsupported()
        index += 1
    tokens.append(_Token("EOF", ""))
    return tokens


_Node = Tuple[object, ...]


class _Parser:
    def __init__(self, tokens: List[_Token]) -> None:
        self.tokens = tokens
        self.position = 0

    def current(self) -> _Token:
        return self.tokens[self.position]

    def consume(self, kind: str, value: Optional[str] = None) -> _Token:
        token = self.current()
        if token.kind != kind or (value is not None and token.value != value):
            raise _unsupported()
        self.position += 1
        return token

    def parse(self) -> _Node:
        node = self.expression(1)
        if self.current().kind != "EOF":
            raise _unsupported()
        return node

    def expression(self, minimum_precedence: int) -> _Node:
        left = self.unary()
        while True:
            token = self.current()
            implicit = token.kind in _ATOM_STARTS
            operator = "*" if implicit else token.value
            if token.kind != "OP" and not implicit:
                break
            precedence = _BINARY_PRECEDENCE.get(operator)
            if precedence is None or precedence < minimum_precedence:
                break
            if not implicit:
                self.position += 1
            right = self.expression(precedence + 1)
            left = ("binary", operator, left, right, implicit)
        return left

    def unary(self) -> _Node:
        if self.current() == _Token("OP", "-"):
            self.position += 1
            return ("negative", self.unary())
        node = self.primary()
        if self.current() == _Token("OP", "^"):
            self.position += 1
            exponent = self.consume("NUMBER").value
            if exponent not in {"2", "3"}:
                raise _unsupported()
            node = ("power", exponent, node)
        return node

    def primary(self) -> _Node:
        token = self.current()
        if token.kind == "NUMBER":
            self.position += 1
            return ("number", token.value)
        if token.kind == "VARIABLE":
            self.position += 1
            return ("variable", token.value)
        if token.kind == "LPAREN":
            self.position += 1
            node = self.expression(1)
            self.consume("RPAREN")
            return ("parentheses", node)
        if token.kind == "FUNCTION":
            self.position += 1
            argument = self.primary()
            return ("function", token.value, argument)
        if token.kind == "FRAC":
            self.position += 1
            numerator = self.group()
            denominator = self.group()
            return ("fraction", numerator, denominator)
        raise _unsupported()

    def group(self) -> _Node:
        self.consume("LBRACE")
        node = self.expression(1)
        self.consume("RBRACE")
        return node


_DIGITS = "零一二三四五六七八九"


def _integer_to_chinese(value: str) -> str:
    number = int(value)
    if number < 10:
        return _DIGITS[number]
    if number < 100:
        tens, units = divmod(number, 10)
        return (
            ("" if tens == 1 else _DIGITS[tens])
            + "十"
            + ("" if units == 0 else _DIGITS[units])
        )
    return " ".join(_DIGITS[int(char)] for char in value)


def _render(node: _Node) -> str:
    kind = node[0]
    if kind == "number":
        return _integer_to_chinese(str(node[1]))
    if kind == "variable":
        return str(node[1])
    if kind == "negative":
        return "负" + _render(node[1])
    if kind == "parentheses":
        return "括号 " + _render(node[1]) + " 括号"
    if kind == "power":
        base = node[2]
        rendered = _render(base)
        if base[0] == "parentheses":
            rendered = _render(base[1]) + " 整体"
        return rendered + ("的平方" if node[1] == "2" else "的立方")
    if kind == "fraction":
        return _render(node[2]) + "分之" + _render(node[1])
    if kind == "function":
        return _FUNCTION_SPEECH[str(node[1])] + _render(node[2])
    if kind != "binary":
        raise _unsupported()
    operator = str(node[1])
    left, right = node[2], node[3]
    left_text, right_text = _render(left), _render(right)
    if operator == "*":
        if (
            bool(node[4])
            and left[0] in {"number", "variable"}
            and right[0] == "variable"
        ):
            return left_text + " " + right_text
        return left_text + "乘" + right_text
    joiner = {
        "+": "加",
        "-": " 减 ",
        "=": "等于",
        "!=": " 不等于",
        ">": "大于",
        "<": "小于",
        ">=": "大于等于",
        "<=": "小于等于",
    }[operator]
    return left_text + joiner + right_text


def display_math_to_spoken(value: str) -> str:
    """Convert one supported display expression to deterministic Mandarin."""
    try:
        expression = _single_expression(value)
        if not is_strict_math_expression(expression):
            raise _unsupported()
        return _render(_Parser(_tokenize(expression)).parse())
    except MathSpeechError:
        raise
    except Exception:
        raise _unsupported() from None


def _normalized_spoken(value: str) -> str:
    if not isinstance(value, str) or len(value) > MAX_DISPLAY_MATH_SCAN_LENGTH:
        raise _unsupported()
    return re.sub(r"[\s，。；：、,.!?！？;:]+", "", value).casefold()


_CHINESE_NUMERAL_CHARACTERS = frozenset("零一二三四五六七八九十百千万亿两点")


def _reading_pattern(expected: str) -> re.Pattern:
    parts = []
    index = 0
    while index < len(expected):
        char = expected[index]
        if char.isascii() and char.isalpha():
            parts.append(
                r"(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])"
                % re.escape(char)
            )
            index += 1
            continue
        if char in _CHINESE_NUMERAL_CHARACTERS:
            end = index + 1
            while (
                end < len(expected)
                and expected[end] in _CHINESE_NUMERAL_CHARACTERS
            ):
                end += 1
            numeral = expected[index:end]
            numeral_class = "".join(sorted(_CHINESE_NUMERAL_CHARACTERS))
            parts.append(
                r"(?<![%s])%s(?![%s])"
                % (
                    numeral_class,
                    re.escape(numeral),
                    numeral_class,
                )
            )
            index = end
            continue
        parts.append(re.escape(char))
        index += 1
    return re.compile("".join(parts))


def contains_deterministic_math_speech(
    math_value: str,
    spoken_text: str,
) -> bool:
    """Return whether one safe math reading occurs at token boundaries."""
    try:
        expected = _normalized_spoken(display_math_to_spoken(math_value))
        spoken = _normalized_spoken(spoken_text)
        return _reading_pattern(expected).search(spoken) is not None
    except MathSpeechError:
        return False


def validate_display_spoken_alignment(display_text: str, spoken_text: str) -> None:
    """Require every displayed formula's deterministic speech in the narration."""
    try:
        spoken = _normalized_spoken(spoken_text)
        cursor = 0
        for expression in extract_display_math(display_text):
            expected = _normalized_spoken(display_math_to_spoken(expression))
            match = _reading_pattern(expected).search(spoken, cursor)
            if match is None:
                raise MathSpeechError("display_spoken_math_mismatch")
            cursor = match.end()
    except MathSpeechError as error:
        if error.code == "display_spoken_math_mismatch":
            raise
        raise MathSpeechError(error.code) from None
    except Exception:
        raise _unsupported() from None
