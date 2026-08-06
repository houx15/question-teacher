import re
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple

from sympy import (
    Add,
    Eq,
    Expr,
    Float,
    FiniteSet,
    Integer,
    latex,
    Mul,
    Poly,
    Pow,
    Rational,
    S,
    Symbol,
    collect,
    count_ops,
    default_sort_key,
    expand,
    factor,
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


class MathValidationReason(str, Enum):
    INVALID_INPUT = "invalid_input"
    UNSUPPORTED = "unsupported"
    CONTRADICTION = "contradiction"


class MathValidationError(ValueError):
    """Raised when math input is unsafe, malformed, or unsupported."""

    def __init__(
        self,
        message: str,
        reason: MathValidationReason = MathValidationReason.INVALID_INPUT,
    ) -> None:
        super().__init__(message)
        self.reason = reason


class _ImmutableList(list):
    def _reject_mutation(self, *args: object, **kwargs: object) -> None:
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
    equation_degree: int

    def __post_init__(self) -> None:
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
    _NO_REAL_SOLUTION = "无实数解"
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
    _CHARACTER_OR_ABSOLUTE_VALUE_PATTERN = re.compile(
        r"[0-9A-Za-z+\-*/^().|\s]+"
    )
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

        empty_state_flags = [
            self._is_no_real_solution_token(text) for text in equations
        ]
        if any(empty_state_flags):
            if len(equations) == 1 and empty_state_flags[0]:
                return S.EmptySet
            raise MathValidationError("无实数解状态不能与其他方程混用。")

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
                raise MathValidationError(
                    "暂不支持求解这个方程。",
                    MathValidationReason.UNSUPPORTED,
                ) from exc
            combined = combined.union(solutions)

        if combined.is_finite_set is not True:
            raise MathValidationError(
                "暂仅支持有限实数解集。",
                MathValidationReason.UNSUPPORTED,
            )
        return combined

    def validate_problem(
        self,
        problem_text: str,
        reference_answer: str,
    ) -> ProblemValidation:
        equation_text = self._extract_problem_equation(problem_text)
        equation_parts = self._parse_equation_parts(equation_text)
        degree = Poly(
            equation_parts.left - equation_parts.right,
            self.x,
        ).degree()
        actual_solutions = self.solution_set([equation_text])
        reference_solutions = self._answer_solution_set(reference_answer)
        if actual_solutions != reference_solutions:
            raise MathValidationError(
                "参考答案与题目实际解集不一致。",
                MathValidationReason.CONTRADICTION,
            )

        solution_strings = [
            sstr(solution)
            for solution in sorted(actual_solutions, key=default_sort_key)
        ]
        return ProblemValidation(
            solution_strings=solution_strings,
            equation_degree=(
                0 if degree is S.NegativeInfinity else int(degree)
            ),
        )

    def try_validate_supported_problem(
        self,
        problem_text: str,
        reference_answer: str,
    ) -> ProblemValidation:
        """Run the unchanged strict validator for a supported problem."""
        return self.validate_problem(problem_text, reference_answer)

    def extract_problem_equation(self, problem_text: str) -> str:
        """Return the single validated equation segment from a problem."""
        equation_text = self._extract_problem_equation(problem_text)
        self.parse_equation(equation_text)
        return equation_text

    def validate_step(self, step: MathStep) -> None:
        before = getattr(step, "state_before", None)
        after = getattr(step, "state_after", None)
        if not isinstance(before, list) or not before:
            raise MathValidationError("变形前的方程状态不能为空。")
        if not isinstance(after, list) or not after:
            raise MathValidationError("变形后的方程状态不能为空。")

        self._validate_operation(
            getattr(step, "operation", None),
            getattr(step, "operands", None),
            before,
            after,
        )
        before_solutions = self.solution_set(before)
        after_solutions = self.solution_set(after)
        if before_solutions != after_solutions:
            raise MathValidationError("变形前后的解集不一致。")

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

    def format_answer_label(self, answer: str) -> str:
        """Format a validated answer as its public math-choice label."""
        solution_set = self._answer_solution_set(answer)
        if solution_set == S.EmptySet:
            return r"\(\text{无实数解}\)"
        if not isinstance(solution_set, FiniteSet):
            raise MathValidationError("参考答案格式不正确。")
        return " 或 ".join(
            rf"\(x={latex(value)}\)"
            for value in sorted(solution_set, key=default_sort_key)
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

    def _parse_expression_pair(self, text: str) -> Tuple[Expr, Expr]:
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
        except MathValidationError:
            raise
        except Exception as exc:
            raise MathValidationError("数学表达式格式不正确。") from exc

        if not isinstance(expression, Expr):
            raise MathValidationError("数学表达式格式不正确。")
        if expression.free_symbols - {self.x}:
            raise MathValidationError(
                "数学表达式中只能使用未知数 x。",
                MathValidationReason.UNSUPPORTED,
            )
        return unevaluated, expression

    def _answer_solution_set(self, answer: str) -> Set:
        if not isinstance(answer, str) or not answer.strip():
            raise MathValidationError("参考答案不能为空。")
        normalized = answer.strip()
        if self._is_no_real_solution_token(normalized):
            return S.EmptySet

        branches = self._ANSWER_SEPARATOR.split(normalized)
        if not branches or any(not branch for branch in branches):
            raise MathValidationError("参考答案格式不正确。")
        if len(branches) > self._MAX_BRANCHES:
            raise MathValidationError("参考答案分支不能超过四个。")
        if any(self._is_no_real_solution_token(branch) for branch in branches):
            raise MathValidationError("无实数解不能与其他答案分支混用。")

        values: List[Expr] = []
        for branch in branches:
            match = re.fullmatch(r"\s*x\s*=\s*(.+?)\s*", branch)
            if match is None:
                raise MathValidationError("参考答案必须写成 x=常量。")
            raw_value, value = self._parse_expression_pair(match.group(1))
            if raw_value.free_symbols:
                raise MathValidationError("参考答案右侧不能包含 x 或其他符号。")
            if value.free_symbols or value.is_real is not True:
                raise MathValidationError("参考答案右侧必须是实数常量。")
            values.append(value)
        return FiniteSet(*values)

    def _is_no_real_solution_token(self, text: object) -> bool:
        return isinstance(text, str) and text.strip() == self._NO_REAL_SOLUTION

    def _normalize_expression(self, text: str) -> str:
        if not isinstance(text, str):
            raise MathValidationError("数学表达式必须是文本。")

        normalized = text.translate(self._NORMALIZATION_TABLE).strip()
        if not normalized:
            raise MathValidationError("数学表达式不能为空。")
        if len(normalized) > self._MAX_EXPRESSION_LENGTH:
            raise MathValidationError("数学表达式过长。")
        if self._CHARACTER_PATTERN.fullmatch(normalized) is None:
            reason = (
                MathValidationReason.UNSUPPORTED
                if self._CHARACTER_OR_ABSOLUTE_VALUE_PATTERN.fullmatch(
                    normalized
                )
                else MathValidationReason.INVALID_INPUT
            )
            raise MathValidationError(
                "数学表达式包含不支持的字符。",
                reason,
            )
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
            raise MathValidationError(
                "数学表达式中只能使用未知数 x 和 sqrt。",
                MathValidationReason.UNSUPPORTED,
            )
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
            raise MathValidationError(
                "数学表达式中只能使用未知数 x。",
                MathValidationReason.UNSUPPORTED,
            )
        if count_ops(expression, visual=False) > self._MAX_OPERATIONS:
            raise MathValidationError("数学表达式运算过多。")

        for node in preorder_traversal(expression):
            if not isinstance(node, Pow):
                continue
            exponent = node.exp
            if node.base.has(self.x) and exponent.is_negative:
                raise MathValidationError(
                    "暂不支持未知数位于分母。",
                    MathValidationReason.UNSUPPORTED,
                )
            if exponent.is_Integer:
                if abs(int(exponent)) > self._MAX_LITERAL_EXPONENT:
                    raise MathValidationError("指数绝对值不能超过四。")
            elif exponent.is_Rational:
                if node.base.has(self.x):
                    raise MathValidationError(
                        "暂不支持未知数的根式幂。",
                        MathValidationReason.UNSUPPORTED,
                    )
            else:
                raise MathValidationError(
                    "暂不支持这个指数形式。",
                    MathValidationReason.UNSUPPORTED,
                )
        if expression.is_real is not True:
            raise MathValidationError("数学表达式必须在实数范围内。")

    def _validate_operation(
        self,
        operation: Optional[str],
        operand_texts: Optional[List[str]],
        before_texts: List[str],
        after_texts: List[str],
    ) -> None:
        if any(
            self._is_no_real_solution_token(text) for text in before_texts
        ):
            raise MathValidationError("操作前必须提供方程状态。")

        empty_after_flags = [
            self._is_no_real_solution_token(text) for text in after_texts
        ]
        after_is_empty_state = (
            len(after_texts) == 1 and empty_after_flags[0]
        )
        if any(empty_after_flags) and not after_is_empty_state:
            raise MathValidationError("无实数解状态不能与其他方程混用。")

        before = [
            self._parse_equation_parts(text) for text in before_texts
        ]
        after = (
            []
            if after_is_empty_state
            else [
                self._parse_equation_parts(text) for text in after_texts
            ]
        )

        valid = False
        if operation in {
            "split_plus_minus",
            "take_square_root_both_sides",
        }:
            valid = self._is_square_root_branch_transition(
                operation,
                before,
                after,
                after_is_empty_state,
            )
        elif operation in {
            "add_both_sides",
            "subtract_both_sides",
            "multiply_both_sides",
            "divide_both_sides",
            "complete_the_square",
        }:
            valid = self._is_declared_operand_operation(
                operation,
                operand_texts,
                before,
                after,
            )
        elif operation in {
            "simplify",
            "combine_like_terms",
            "expand",
            "factor",
        }:
            valid = self._is_canonical_operation(operation, before, after)
        elif operation == "quadratic_formula":
            valid = self._is_quadratic_formula_result(
                before,
                after,
                after_is_empty_state,
            )

        if not valid:
            raise MathValidationError("操作标签与代数变形结构不一致。")

    def _is_declared_operand_operation(
        self,
        operation: str,
        operand_texts: Optional[List[str]],
        before: List[_EquationParts],
        after: List[_EquationParts],
    ) -> bool:
        if (
            not isinstance(operand_texts, list)
            or len(operand_texts) != 1
            or not before
            or len(before) != len(after)
            or (operation == "complete_the_square" and len(before) != 1)
        ):
            return False

        operand = self.parse_expression(operand_texts[0])
        if operand == 0:
            return False
        if operation in {
            "multiply_both_sides",
            "divide_both_sides",
            "complete_the_square",
        } and operand.has(self.x):
            return False
        if (
            operation == "complete_the_square"
            and operand.is_positive is not True
        ):
            return False

        for before_part, after_part in zip(before, after):
            if operation in {"add_both_sides", "complete_the_square"}:
                expected_left = before_part.left + operand
                expected_right = before_part.right + operand
            elif operation == "subtract_both_sides":
                expected_left = before_part.left - operand
                expected_right = before_part.right - operand
            elif operation == "multiply_both_sides":
                expected_left = before_part.left * operand
                expected_right = before_part.right * operand
            else:
                expected_left = before_part.left / operand
                expected_right = before_part.right / operand

            if (
                simplify(after_part.left - expected_left) != 0
                or simplify(after_part.right - expected_right) != 0
                or self._corresponding_sides_equal(
                    before_part,
                    after_part,
                )
            ):
                return False
            if (
                operation == "complete_the_square"
                and not self._newly_introduces_squared_binomial(
                    before_part,
                    after_part,
                )
            ):
                return False
        return True

    def _is_canonical_operation(
        self,
        operation: str,
        before: List[_EquationParts],
        after: List[_EquationParts],
    ) -> bool:
        if len(before) != 1 or len(after) != 1:
            return False

        transformer = {
            "simplify": simplify,
            "combine_like_terms": lambda expression: collect(
                expand(expression),
                self.x,
            ),
            "expand": expand,
            "factor": factor,
        }[operation]
        expected_left = transformer(before[0].left_raw)
        expected_right = transformer(before[0].right_raw)

        if operation == "combine_like_terms":
            if (
                after[0].left_raw != expected_left
                or after[0].right_raw != expected_right
            ):
                return False
        elif (
            after[0].left != expected_left
            or after[0].right != expected_right
        ):
            return False

        if operation == "simplify":
            before_ops = count_ops(before[0].left_raw) + count_ops(
                before[0].right_raw
            )
            expected_ops = count_ops(expected_left) + count_ops(expected_right)
            after_ops = count_ops(after[0].left_raw) + count_ops(
                after[0].right_raw
            )
            return expected_ops < before_ops and after_ops <= expected_ops
        if operation == "combine_like_terms":
            return (
                before[0].left_raw != expected_left
                or before[0].right_raw != expected_right
            )
        return (
            before[0].left != expected_left
            or before[0].right != expected_right
        )

    def _is_quadratic_formula_result(
        self,
        before: List[_EquationParts],
        after: List[_EquationParts],
        after_is_empty_state: bool = False,
    ) -> bool:
        if len(before) != 1:
            return False
        try:
            degree = Poly(
                before[0].left - before[0].right,
                self.x,
            ).degree()
        except PolynomialError:
            return False
        if degree != 2:
            return False

        try:
            expected_solutions = solveset(
                before[0].left - before[0].right,
                self.x,
                domain=S.Reals,
            )
        except Exception:
            return False
        if expected_solutions.is_finite_set is not True:
            return False
        if expected_solutions == S.EmptySet:
            return after_is_empty_state
        if after_is_empty_state or not after:
            return False

        actual_values: List[Expr] = []
        for equation in after:
            if (
                equation.left != self.x
                or equation.right.has(self.x)
                or equation.right.is_real is not True
            ):
                return False
            actual_values.append(equation.right)
        unique_values = FiniteSet(*actual_values)
        return (
            len(actual_values) == len(unique_values)
            and len(actual_values) == len(expected_solutions)
            and unique_values == expected_solutions
        )

    def _is_square_root_branch_transition(
        self,
        operation: str,
        before: List[_EquationParts],
        after: List[_EquationParts],
        after_is_empty_state: bool = False,
    ) -> bool:
        if len(before) != 1:
            return False

        square_parts = self._explicit_square_and_constant(before[0])
        if square_parts is None:
            return False
        base, constant = square_parts
        if constant.is_negative is True:
            return (
                operation == "take_square_root_both_sides"
                and after_is_empty_state
            )
        if constant.is_nonnegative is not True or after_is_empty_state:
            return False

        root = sqrt(constant)
        expected_values = FiniteSet(root, -root)
        expected_count = len(expected_values)
        if operation == "split_plus_minus":
            required_count = 2
            if expected_count != required_count:
                return False
        else:
            required_count = expected_count
        if len(after) != required_count:
            return False

        actual_values = []
        for equation in after:
            value = self._assigned_value_for_base(equation, base)
            if value is None:
                return False
            actual_values.append(value)

        return (
            len(FiniteSet(*actual_values)) == required_count
            and FiniteSet(*actual_values) == expected_values
            and all(
                not (
                    equation.left_raw == before[0].left_raw
                    and equation.right_raw == before[0].right_raw
                )
                for equation in after
            )
        )

    def _explicit_square_and_constant(
        self,
        equation: _EquationParts,
    ) -> Optional[Tuple[Expr, Expr]]:
        for square_side, constant_side in (
            (equation.left_raw, equation.right),
            (equation.right_raw, equation.left),
        ):
            if (
                isinstance(square_side, Pow)
                and square_side.exp == 2
                and square_side.base.has(self.x)
                and not constant_side.has(self.x)
                and constant_side.is_real is True
            ):
                return square_side.base, constant_side
        return None

    def _assigned_value_for_base(
        self,
        equation: _EquationParts,
        base: Expr,
    ) -> Optional[Expr]:
        for candidate_base, candidate_value in (
            (equation.left, equation.right),
            (equation.right, equation.left),
        ):
            if (
                simplify(candidate_base - base) == 0
                and not candidate_value.has(self.x)
                and candidate_value.is_real is True
            ):
                return candidate_value
        return None

    def _corresponding_sides_equal(
        self,
        before: _EquationParts,
        after: _EquationParts,
    ) -> bool:
        return (
            simplify(before.left - after.left) == 0
            and simplify(before.right - after.right) == 0
        )

    def _newly_introduces_squared_binomial(
        self,
        before: _EquationParts,
        after: _EquationParts,
    ) -> bool:
        return any(
            self._is_scaled_squared_binomial(factor(after_side))
            and not self._contains_or_factors_to_squared_binomial(
                before_raw,
                before_side,
            )
            for before_raw, before_side, after_side in (
                (before.left_raw, before.left, after.left),
                (before.right_raw, before.right, after.right),
            )
        )

    def _contains_or_factors_to_squared_binomial(
        self,
        raw_expression: Expr,
        expression: Expr,
    ) -> bool:
        return self._is_scaled_squared_binomial(factor(expression)) or any(
            self._is_scaled_squared_binomial(node)
            for node in preorder_traversal(raw_expression)
        )

    def _is_scaled_squared_binomial(self, expression: Expr) -> bool:
        scalar, dependent = expression.as_independent(
            self.x,
            as_Add=False,
        )
        return (
            scalar != 0
            and not scalar.has(self.x)
            and scalar.is_real is True
            and self._is_squared_binomial(dependent)
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

    def _validate_equation_polynomial(
        self,
        left: Expr,
        right: Expr,
    ) -> None:
        try:
            polynomial = Poly(left - right, self.x)
            degree = polynomial.degree()
        except (PolynomialError, TypeError, ValueError) as exc:
            raise MathValidationError(
                "暂仅支持一元多项式方程。",
                MathValidationReason.UNSUPPORTED,
            ) from exc
        if degree is not S.NegativeInfinity and (
            degree > self._MAX_POLYNOMIAL_DEGREE
        ):
            raise MathValidationError(
                "暂仅支持一次或二次方程。",
                MathValidationReason.UNSUPPORTED,
            )
        if any(
            coefficient.is_real is not True
            for coefficient in polynomial.all_coeffs()
        ):
            raise MathValidationError("方程系数必须是实数。")

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
