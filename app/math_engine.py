import re
from dataclasses import dataclass
from typing import List

from sympy import (
    Eq,
    Expr,
    Float,
    Integer,
    Rational,
    S,
    Symbol,
    default_sort_key,
    simplify,
    solveset,
    sqrt,
    sstr,
)
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from app.schemas import MathStep


class MathValidationError(ValueError):
    """Raised when math input is unsafe, malformed, or unsupported."""


class _ImmutableList(list):
    def _reject_mutation(self, *args, **kwargs):
        raise TypeError("solution_strings is immutable")

    __delitem__ = _reject_mutation
    __iadd__ = _reject_mutation
    __imul__ = _reject_mutation
    __setitem__ = _reject_mutation
    append = _reject_mutation
    clear = _reject_mutation
    extend = _reject_mutation
    insert = _reject_mutation
    pop = _reject_mutation
    remove = _reject_mutation
    reverse = _reject_mutation
    sort = _reject_mutation


@dataclass(frozen=True)
class ProblemValidation:
    solution_strings: List[str]

    def __post_init__(self):
        object.__setattr__(
            self,
            "solution_strings",
            _ImmutableList(self.solution_strings),
        )


class MathEngine:
    _TRANSFORMATIONS = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
    )
    _CHARACTER_PATTERN = re.compile(r"[0-9A-Za-z+\-*/^().\s]+")
    _IDENTIFIER_PATTERN = re.compile(r"[A-Za-z]+")
    _ANSWER_SEPARATOR = re.compile(r"\s*(?:或|(?i:\bor\b))\s*")
    _NORMALIZATION_TABLE = str.maketrans(
        {
            "×": "*",
            "÷": "/",
            "−": "-",
            "－": "-",
            "–": "-",
            "—": "-",
            "＋": "+",
            "＝": "=",
            "²": "^2",
            "Ｘ": "x",
            "ｘ": "x",
            "X": "x",
        }
    )

    def __init__(self) -> None:
        self.x = Symbol("x", real=True)
        self._local_dict = {"x": self.x, "sqrt": sqrt}
        self._global_dict = {
            "Integer": Integer,
            "Float": Float,
            "Rational": Rational,
        }

    def parse_expression(self, text):
        normalized = self._normalize_expression(text)
        try:
            expression = parse_expr(
                normalized,
                local_dict=self._local_dict,
                global_dict=self._global_dict,
                transformations=self._TRANSFORMATIONS,
                evaluate=True,
            )
        except Exception as exc:
            raise MathValidationError("数学表达式格式不正确。") from exc

        if not isinstance(expression, Expr):
            raise MathValidationError("数学表达式格式不正确。")
        if expression.free_symbols - {self.x}:
            raise MathValidationError("数学表达式中只能使用未知数 x。")
        return expression

    def parse_equation(self, text):
        if not isinstance(text, str):
            raise MathValidationError("方程必须是文本。")

        equation_text = self._extract_equation(text).translate(
            self._NORMALIZATION_TABLE
        )
        if equation_text.count("=") != 1:
            raise MathValidationError("方程必须且只能包含一个等号。")

        left_text, right_text = equation_text.split("=")
        if not left_text.strip() or not right_text.strip():
            raise MathValidationError("方程等号两边都不能为空。")

        left = self.parse_expression(left_text)
        right = self.parse_expression(right_text)
        return Eq(left, right, evaluate=False)

    def solution_set(self, equations: List[str]):
        if not isinstance(equations, list) or not equations:
            raise MathValidationError("方程状态不能为空。")

        combined = S.EmptySet
        for equation_text in equations:
            equation = self.parse_equation(equation_text)
            try:
                solutions = solveset(
                    equation.lhs - equation.rhs,
                    self.x,
                    domain=S.Reals,
                )
            except Exception as exc:
                raise MathValidationError("暂不支持求解这个方程。") from exc
            combined = combined.union(solutions)

        if combined.is_finite_set is not True:
            raise MathValidationError("暂仅支持有限实数解集。")
        return combined

    def validate_problem(
        self,
        problem_text,
        reference_answer,
    ) -> ProblemValidation:
        actual_solutions = self.solution_set([problem_text])
        reference_solutions = self._answer_solution_set(reference_answer)
        if actual_solutions != reference_solutions:
            raise MathValidationError("参考答案与题目实际解集不一致。")

        solution_strings = [
            sstr(solution)
            for solution in sorted(actual_solutions, key=default_sort_key)
        ]
        return ProblemValidation(solution_strings=solution_strings)

    def validate_step(self, step: MathStep) -> None:
        before = getattr(step, "state_before", None)
        after = getattr(step, "state_after", None)
        if not isinstance(before, list) or not before:
            raise MathValidationError("变形前的方程状态不能为空。")
        if not isinstance(after, list) or not after:
            raise MathValidationError("变形后的方程状态不能为空。")

        before_solutions = self.solution_set(before)
        after_solutions = self.solution_set(after)
        if before_solutions != after_solutions:
            raise MathValidationError("变形前后的解集不一致。")

    def expressions_equivalent(self, actual, expected) -> bool:
        actual_expression = self.parse_expression(actual)
        expected_expression = self.parse_expression(expected)
        try:
            return simplify(actual_expression - expected_expression) == 0
        except Exception as exc:
            raise MathValidationError("无法比较这两个数学表达式。") from exc

    def answers_equivalent(self, actual, expected) -> bool:
        return self._answer_solution_set(actual) == self._answer_solution_set(
            expected
        )

    def _answer_solution_set(self, answer):
        if not isinstance(answer, str) or not answer.strip():
            raise MathValidationError("参考答案不能为空。")
        branches = self._ANSWER_SEPARATOR.split(answer.strip())
        if not branches or any(not branch for branch in branches):
            raise MathValidationError("参考答案格式不正确。")
        return self.solution_set(branches)

    def _normalize_expression(self, text):
        if not isinstance(text, str):
            raise MathValidationError("数学表达式必须是文本。")

        normalized = text.translate(self._NORMALIZATION_TABLE).strip()
        if not normalized:
            raise MathValidationError("数学表达式不能为空。")
        if self._CHARACTER_PATTERN.fullmatch(normalized) is None:
            raise MathValidationError("数学表达式包含不支持的字符。")
        if re.search(r"\*\s*\*|/\s*/", normalized):
            raise MathValidationError("数学表达式包含不支持的运算符。")

        without_numbers = re.sub(
            r"(?:\d+(?:\.\d*)?|\.\d+)",
            "",
            normalized,
        )
        if "." in without_numbers:
            raise MathValidationError("数学表达式不支持属性访问。")

        identifiers = self._IDENTIFIER_PATTERN.findall(normalized)
        if any(identifier not in {"x", "sqrt"} for identifier in identifiers):
            raise MathValidationError("数学表达式中只能使用未知数 x 和 sqrt。")
        if "sqrt" in identifiers and re.search(
            r"\bsqrt\b(?!\s*\()",
            normalized,
        ):
            raise MathValidationError("sqrt 后必须使用括号。")
        return normalized

    @staticmethod
    def _extract_equation(text):
        stripped = text.strip()
        if not stripped:
            raise MathValidationError("方程不能为空。")
        colon_index = max(stripped.rfind(":"), stripped.rfind("："))
        if colon_index >= 0:
            stripped = stripped[colon_index + 1 :].strip()
        if not stripped:
            raise MathValidationError("冒号后缺少方程。")
        return stripped
