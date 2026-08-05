import re
from dataclasses import dataclass
from typing import List

from sympy import (
    Add,
    Eq,
    Expr,
    Float,
    FiniteSet,
    Integer,
    Mul,
    Poly,
    Pow,
    Rational,
    S,
    Symbol,
    count_ops,
    default_sort_key,
    expand,
    preorder_traversal,
    simplify,
    solveset,
    sqrt,
    sstr,
)
from sympy.core.relational import Equality
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    rationalize,
    standard_transformations,
)
from sympy.polys.polyerrors import PolynomialError
from sympy.sets.sets import Set

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


@dataclass(frozen=True)
class _EquationParts:
    left_raw: Expr
    right_raw: Expr
    left: Expr
    right: Expr


class MathEngine:
    _MAX_EXPRESSION_LENGTH = 256
    _MAX_BRANCHES = 4
    _MAX_NESTING = 12
    _MAX_DIGITS = 12
    _MAX_LITERAL_EXPONENT = 4
    _MAX_OPERATIONS = 64
    _MAX_POLYNOMIAL_DEGREE = 2
    _TRANSFORMATIONS = standard_transformations + (
        implicit_multiplication_application,
        convert_xor,
        rationalize,
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
            "__builtins__": {},
            "Add": Add,
            "Integer": Integer,
            "Float": Float,
            "Mul": Mul,
            "Pow": Pow,
            "Rational": Rational,
        }

    def parse_expression(self, text: str) -> Expr:
        _, expression = self._parse_expression_pair(text)
        return expression

    def parse_equation(self, text: str) -> Equality:
        parts = self._parse_equation_parts(text)
        return Eq(parts.left, parts.right, evaluate=False)

    def solution_set(self, equations: List[str]) -> Set:
        if not isinstance(equations, list) or not equations:
            raise MathValidationError("方程状态不能为空。")
        if len(equations) > self._MAX_BRANCHES:
            raise MathValidationError("方程分支不能超过四个。")

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
        problem_text: str,
        reference_answer: str,
    ) -> ProblemValidation:
        equation_text = self._extract_problem_equation(problem_text)
        actual_solutions = self.solution_set([equation_text])
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
        self._validate_operation(getattr(step, "operation", None), before, after)

    def expressions_equivalent(self, actual: str, expected: str) -> bool:
        actual_expression = self.parse_expression(actual)
        expected_expression = self.parse_expression(expected)
        try:
            return simplify(actual_expression - expected_expression) == 0
        except Exception as exc:
            raise MathValidationError("无法比较这两个数学表达式。") from exc

    def answers_equivalent(self, actual: str, expected: str) -> bool:
        return self._answer_solution_set(actual) == self._answer_solution_set(
            expected
        )

    def _parse_equation_parts(self, text: str) -> _EquationParts:
        if not isinstance(text, str):
            raise MathValidationError("方程必须是文本。")

        equation_text = text.translate(self._NORMALIZATION_TABLE).strip()
        if equation_text.count("=") != 1:
            raise MathValidationError("方程必须且只能包含一个等号。")

        left_text, right_text = equation_text.split("=")
        if not left_text.strip() or not right_text.strip():
            raise MathValidationError("方程等号两边都不能为空。")

        left_raw, left = self._parse_expression_pair(left_text)
        right_raw, right = self._parse_expression_pair(right_text)
        self._validate_equation_polynomial(left, right)
        return _EquationParts(
            left_raw=left_raw,
            right_raw=right_raw,
            left=left,
            right=right,
        )

    def _parse_expression_pair(self, text: str):
        normalized = self._normalize_expression(text)
        try:
            unevaluated = parse_expr(
                normalized,
                local_dict=self._local_dict,
                global_dict=self._global_dict,
                transformations=self._TRANSFORMATIONS,
                evaluate=False,
            )
            self._validate_expression_shape(unevaluated)
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
        return unevaluated, expression

    def _answer_solution_set(self, answer: str) -> FiniteSet:
        if not isinstance(answer, str) or not answer.strip():
            raise MathValidationError("参考答案不能为空。")
        branches = self._ANSWER_SEPARATOR.split(answer.strip())
        if not branches or any(not branch for branch in branches):
            raise MathValidationError("参考答案格式不正确。")
        if len(branches) > self._MAX_BRANCHES:
            raise MathValidationError("参考答案分支不能超过四个。")

        values = []
        for branch in branches:
            match = re.fullmatch(r"\s*x\s*=\s*(.+?)\s*", branch)
            if match is None:
                raise MathValidationError("参考答案必须写成 x=常量。")
            value = self.parse_expression(match.group(1))
            if value.free_symbols or value.is_real is not True:
                raise MathValidationError("参考答案右侧必须是实数常量。")
            values.append(value)
        return FiniteSet(*values)

    def _normalize_expression(self, text: str) -> str:
        if not isinstance(text, str):
            raise MathValidationError("数学表达式必须是文本。")

        normalized = text.translate(self._NORMALIZATION_TABLE).strip()
        if not normalized:
            raise MathValidationError("数学表达式不能为空。")
        if len(normalized) > self._MAX_EXPRESSION_LENGTH:
            raise MathValidationError("数学表达式过长。")
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
        if any(
            len(digits) > self._MAX_DIGITS
            for digits in re.findall(r"\d+", normalized)
        ):
            raise MathValidationError("数字字面量过长。")
        self._validate_parenthesis_nesting(normalized)
        self._validate_literal_exponents(normalized)

        identifiers = self._IDENTIFIER_PATTERN.findall(normalized)
        if any(identifier not in {"x", "sqrt"} for identifier in identifiers):
            raise MathValidationError("数学表达式中只能使用未知数 x 和 sqrt。")
        if "sqrt" in identifiers and re.search(
            r"\bsqrt\b(?!\s*\()",
            normalized,
        ):
            raise MathValidationError("sqrt 后必须使用括号。")
        return normalized

    def _validate_expression_shape(self, expression: Expr) -> None:
        if not isinstance(expression, Expr):
            raise MathValidationError("数学表达式格式不正确。")
        if expression.free_symbols - {self.x}:
            raise MathValidationError("数学表达式中只能使用未知数 x。")
        if count_ops(expression, visual=False) > self._MAX_OPERATIONS:
            raise MathValidationError("数学表达式运算过多。")

        for node in preorder_traversal(expression):
            if not isinstance(node, Pow):
                continue
            exponent = node.exp
            if node.base.has(self.x) and exponent.is_negative:
                raise MathValidationError("暂不支持未知数位于分母。")
            if exponent.is_Integer:
                if abs(int(exponent)) > self._MAX_LITERAL_EXPONENT:
                    raise MathValidationError("指数绝对值不能超过四。")
            elif exponent.is_Rational:
                if node.base.has(self.x):
                    raise MathValidationError("暂不支持未知数的根式幂。")
            else:
                raise MathValidationError("暂不支持这个指数形式。")

    def _validate_operation(
        self,
        operation: str,
        before_texts: List[str],
        after_texts: List[str],
    ) -> None:
        before = [
            self._parse_equation_parts(text) for text in before_texts
        ]
        after = [self._parse_equation_parts(text) for text in after_texts]

        valid = False
        if operation == "split_plus_minus":
            valid = len(before) == 1 and len(after) >= 2
        elif operation in {
            "add_both_sides",
            "subtract_both_sides",
            "complete_the_square",
        }:
            valid = self._same_delta_on_both_sides(before, after)
            if valid and operation == "complete_the_square":
                valid = self._has_squared_binomial(after[0])
        elif operation in {
            "multiply_both_sides",
            "divide_both_sides",
        }:
            valid = self._same_factor_on_both_sides(before, after)
        elif operation == "expand":
            valid = self._is_expansion(before, after)
        elif operation == "factor":
            valid = self._is_factorization(before, after)
        elif operation in {"simplify", "combine_like_terms"}:
            valid = self._is_nonincreasing_simplification(before, after)
        elif operation == "take_square_root_both_sides":
            valid = (
                len(before) == 1
                and bool(after)
                and self._equation_contains_square(before[0])
            )
        elif operation == "quadratic_formula":
            valid = self._is_quadratic_formula_result(before, after)

        if not valid:
            raise MathValidationError("操作标签与代数变形结构不一致。")

    def _same_delta_on_both_sides(
        self,
        before: List[_EquationParts],
        after: List[_EquationParts],
    ) -> bool:
        if len(before) != 1 or len(after) != 1:
            return False
        left_delta = simplify(after[0].left - before[0].left)
        right_delta = simplify(after[0].right - before[0].right)
        return (
            left_delta == right_delta
            and left_delta != 0
            and not left_delta.has(self.x)
        )

    def _same_factor_on_both_sides(
        self,
        before: List[_EquationParts],
        after: List[_EquationParts],
    ) -> bool:
        if len(before) != 1 or len(after) != 1:
            return False

        factors = []
        for before_side, after_side in (
            (before[0].left, after[0].left),
            (before[0].right, after[0].right),
        ):
            if before_side == 0:
                if after_side != 0:
                    return False
                continue
            factors.append(simplify(after_side / before_side))
        if not factors:
            return False
        first = factors[0]
        return (
            first not in (0, 1)
            and not first.has(self.x)
            and all(simplify(factor - first) == 0 for factor in factors[1:])
        )

    def _is_expansion(
        self,
        before: List[_EquationParts],
        after: List[_EquationParts],
    ) -> bool:
        if len(before) != 1 or len(after) != 1:
            return False
        if not self._corresponding_sides_equal(before[0], after[0]):
            return False
        return any(
            simplify(expand(before_raw) - after_value) == 0
            and before_raw != after_raw
            for before_raw, after_raw, after_value in (
                (
                    before[0].left_raw,
                    after[0].left_raw,
                    after[0].left,
                ),
                (
                    before[0].right_raw,
                    after[0].right_raw,
                    after[0].right,
                ),
            )
        )

    def _is_factorization(
        self,
        before: List[_EquationParts],
        after: List[_EquationParts],
    ) -> bool:
        if len(before) != 1 or len(after) != 1:
            return False
        if not self._corresponding_sides_equal(before[0], after[0]):
            return False
        return any(
            before_raw != after_raw and self._has_factored_structure(after_raw)
            for before_raw, after_raw in (
                (before[0].left_raw, after[0].left_raw),
                (before[0].right_raw, after[0].right_raw),
            )
        )

    def _is_nonincreasing_simplification(
        self,
        before: List[_EquationParts],
        after: List[_EquationParts],
    ) -> bool:
        if len(before) != 1 or len(after) != 1:
            return False
        if not self._corresponding_sides_equal(before[0], after[0]):
            return False
        before_ops = count_ops(before[0].left_raw) + count_ops(
            before[0].right_raw
        )
        after_ops = count_ops(after[0].left_raw) + count_ops(
            after[0].right_raw
        )
        return after_ops <= before_ops

    def _is_quadratic_formula_result(
        self,
        before: List[_EquationParts],
        after: List[_EquationParts],
    ) -> bool:
        if len(before) != 1 or not after:
            return False
        try:
            degree = Poly(
                before[0].left - before[0].right,
                self.x,
            ).degree()
        except PolynomialError:
            return False
        return degree == 2 and all(
            equation.left == self.x
            and not equation.right.has(self.x)
            and equation.right.is_real is True
            for equation in after
        )

    def _corresponding_sides_equal(
        self,
        before: _EquationParts,
        after: _EquationParts,
    ) -> bool:
        return (
            simplify(before.left - after.left) == 0
            and simplify(before.right - after.right) == 0
        )

    def _has_squared_binomial(self, equation: _EquationParts) -> bool:
        return any(
            self._is_squared_binomial(node)
            for expression in (equation.left_raw, equation.right_raw)
            for node in preorder_traversal(expression)
        )

    def _is_squared_binomial(self, expression: Expr) -> bool:
        if not isinstance(expression, Pow) or expression.exp != 2:
            return False
        try:
            return (
                expression.base.has(self.x)
                and Poly(expression.base, self.x).degree() == 1
                and isinstance(expression.base, Add)
            )
        except PolynomialError:
            return False

    def _equation_contains_square(self, equation: _EquationParts) -> bool:
        return any(
            isinstance(node, Pow)
            and node.exp == 2
            and node.base.has(self.x)
            for expression in (equation.left_raw, equation.right_raw)
            for node in preorder_traversal(expression)
        )

    def _has_factored_structure(self, expression: Expr) -> bool:
        if isinstance(expression, Pow):
            return expression.exp.is_Integer and expression.exp > 1
        if not isinstance(expression, Mul):
            return False
        dependent_factors = [
            factor for factor in expression.args if factor.has(self.x)
        ]
        return len(dependent_factors) >= 2 or any(
            isinstance(factor, Add) for factor in dependent_factors
        )

    def _validate_equation_polynomial(
        self,
        left: Expr,
        right: Expr,
    ) -> None:
        try:
            polynomial = Poly(left - right, self.x)
            degree = polynomial.degree()
        except (PolynomialError, TypeError, ValueError) as exc:
            raise MathValidationError("暂仅支持一元多项式方程。") from exc
        if degree is not S.NegativeInfinity and (
            degree > self._MAX_POLYNOMIAL_DEGREE
        ):
            raise MathValidationError("暂仅支持一次或二次方程。")

    def _validate_parenthesis_nesting(self, text: str) -> None:
        nesting = 0
        for character in text:
            if character == "(":
                nesting += 1
                if nesting > self._MAX_NESTING:
                    raise MathValidationError("括号嵌套过深。")
            elif character == ")":
                nesting -= 1
                if nesting < 0:
                    raise MathValidationError("括号不匹配。")
        if nesting != 0:
            raise MathValidationError("括号不匹配。")

    def _validate_literal_exponents(self, text: str) -> None:
        for match in re.finditer(r"\^\s*([+-]?)\s*(\d+)", text):
            sign, digits = match.groups()
            value = int(f"{sign}{digits}")
            if abs(value) > self._MAX_LITERAL_EXPONENT:
                raise MathValidationError("指数绝对值不能超过四。")

    @staticmethod
    def _extract_problem_equation(text: str) -> str:
        if not isinstance(text, str):
            raise MathValidationError("题目必须是文本。")
        stripped = text.strip()
        if not stripped:
            raise MathValidationError("题目不能为空。")
        colon_index = max(stripped.rfind(":"), stripped.rfind("："))
        if colon_index >= 0:
            stripped = stripped[colon_index + 1 :].strip()
        if not stripped:
            raise MathValidationError("冒号后缺少方程。")
        return stripped
