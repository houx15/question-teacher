from typing import List, Optional

from sympy import Poly, default_sort_key, expand, simplify, sqrt, sstr

from app.math_engine import MathEngine, MathValidationError
from app.schemas import MathRouteDraft, NarrativeMathStep, ProblemInput


class DeterministicRoutePlanner:
    """Build a small allow-listed set of algebra routes without model calls."""

    _MAX_INTEGER_COEFFICIENT = 100

    def __init__(self, math_engine: MathEngine) -> None:
        self.math_engine = math_engine

    @staticmethod
    def _text(expression) -> str:
        return sstr(expression).replace("**", "^")

    def plan(
        self,
        problem: ProblemInput,
        equation_degree: int,
        solution_strings: List[str],
    ) -> Optional[MathRouteDraft]:
        if problem.required_method == "complete_the_square":
            route = self._complete_the_square(problem)
        elif (
            problem.required_method == "quadratic_formula"
            or (
                problem.required_method is None
                and equation_degree == 2
            )
        ):
            route = self._quadratic_formula(
                problem,
                solution_strings,
            )
        elif (
            problem.required_method is None
            and equation_degree == 1
        ):
            route = self._linear_equation(problem)
        else:
            # Factor routes need a separate zero-product transition that
            # the current operation vocabulary cannot state faithfully.
            return None
        if route is None:
            return None
        # The service performs the authoritative step-by-step hard validation
        # before freezing. Returning a draft here never marks it verified.
        return route.model_copy(deep=True)

    def _quadratic_formula(
        self,
        problem: ProblemInput,
        solution_strings: List[str],
    ) -> MathRouteDraft:
        original = self.math_engine.extract_problem_equation(
            problem.problem_text
        )
        if not solution_strings:
            final_state = ["无实数解"]
            reason = "判别式小于零，因此方程没有实数根。"
        else:
            final_state = [
                f"x={solution}"
                for solution in solution_strings
            ]
            reason = "把系数代入求根公式，分别计算全部实数根。"
        return MathRouteDraft(
            math_steps=[
                NarrativeMathStep(
                    purpose="使用求根公式",
                    operation="quadratic_formula",
                    operands=[],
                    state_before=[original],
                    state_after=final_state,
                    reason=reason,
                )
            ]
        )

    def _complete_the_square(
        self,
        problem: ProblemInput,
    ) -> Optional[MathRouteDraft]:
        original = self.math_engine.extract_problem_equation(
            problem.problem_text
        )
        equation = self.math_engine.parse_equation(original)
        if simplify(equation.rhs) != 0:
            return None
        polynomial = Poly(
            expand(equation.lhs - equation.rhs),
            self.math_engine.x,
        )
        if polynomial.degree() != 2 or polynomial.LC() != 1:
            return None
        _, linear, constant = polynomial.all_coeffs()
        if not (
            linear.is_Integer
            and constant.is_Integer
            and 0 < abs(int(linear)) <= self._MAX_INTEGER_COEFFICIENT
            and abs(int(constant)) <= self._MAX_INTEGER_COEFFICIENT
        ):
            return None

        x = self.math_engine.x
        offset = linear / 2
        square_amount = offset**2
        steps = []
        current = original
        isolated_right = -constant

        if constant != 0:
            isolated = (
                f"{self._text(x**2 + linear*x)}="
                f"{self._text(isolated_right)}"
            )
            steps.append(
                NarrativeMathStep(
                    purpose="把常数项移到等号右边",
                    operation="subtract_both_sides",
                    operands=[self._text(constant)],
                    state_before=[current],
                    state_after=[isolated],
                    reason="等式两边同时减去常数项，等式仍然成立。",
                )
            )
            current = isolated

        completed_right = simplify(isolated_right + square_amount)
        completed_left = f"({self._text(x + offset)})^2"
        completed = f"{completed_left}={self._text(completed_right)}"
        steps.append(
            NarrativeMathStep(
                purpose="两边加上同一个数配成完全平方",
                operation="complete_the_square",
                operands=[self._text(square_amount)],
                state_before=[current],
                state_after=[completed],
                reason="一次项系数一半的平方，能把左边配成完全平方。",
            )
        )

        if completed_right.is_negative is True:
            steps.append(
                NarrativeMathStep(
                    purpose="判断实数解",
                    operation="take_square_root_both_sides",
                    operands=[],
                    state_before=[completed],
                    state_after=["无实数解"],
                    reason="实数的平方不能是负数，所以没有实数解。",
                )
            )
            return MathRouteDraft(math_steps=steps)
        if completed_right.is_nonnegative is not True:
            return None

        root_values = sorted(
            {sqrt(completed_right), -sqrt(completed_right)},
            key=default_sort_key,
        )
        base = self._text(x + offset)
        branches = [
            f"{base}={self._text(value)}"
            for value in root_values
        ]
        repeated_root = len(root_values) == 1
        steps.append(
            NarrativeMathStep(
                purpose=(
                    "利用平方为零得到唯一分支"
                    if repeated_root
                    else "等式两边开平方并写出全部分支"
                ),
                operation="take_square_root_both_sides",
                operands=[],
                state_before=[completed],
                state_after=branches,
                reason=(
                    "一个式子的平方等于零时，这个式子只能等于 0。"
                    if repeated_root
                    else "平方等于正数时，要分别考虑正、负两个平方根。"
                ),
            )
        )

        if offset != 0:
            final_state = [
                f"x={self._text(simplify(value - offset))}"
                for value in root_values
            ]
            steps.append(
                NarrativeMathStep(
                    purpose="解出未知数",
                    operation="subtract_both_sides",
                    operands=[self._text(offset)],
                    state_before=branches,
                    state_after=final_state,
                    reason="每个分支的等式两边同时减去同一个数。",
                )
            )
        return MathRouteDraft(math_steps=steps)

    def _linear_equation(
        self,
        problem: ProblemInput,
    ) -> Optional[MathRouteDraft]:
        original = self.math_engine.extract_problem_equation(
            problem.problem_text
        )
        equation = self.math_engine.parse_equation(original)
        if equation.rhs.has(self.math_engine.x):
            return None

        steps = []
        current = original
        expanded_left = expand(equation.lhs)
        expanded_right = expand(equation.rhs)
        expanded_state = (
            f"{self._text(expanded_left)}={self._text(expanded_right)}"
        )
        if expanded_state != original.replace(" ", ""):
            candidate = NarrativeMathStep(
                purpose="展开括号",
                operation="expand",
                operands=[],
                state_before=[current],
                state_after=[expanded_state],
                reason="先展开括号，看清一次项和常数项。",
            )
            try:
                self.math_engine.validate_step(candidate)
            except MathValidationError:
                pass
            else:
                steps.append(candidate)
                current = expanded_state

        equation = self.math_engine.parse_equation(current)
        polynomial = Poly(equation.lhs, self.math_engine.x)
        if polynomial.degree() != 1 or equation.rhs.has(self.math_engine.x):
            return None
        coefficient, constant = polynomial.all_coeffs()
        right = equation.rhs

        if constant != 0:
            after_right = simplify(right - constant)
            after = (
                f"{self._text(coefficient*self.math_engine.x)}="
                f"{self._text(after_right)}"
            )
            steps.append(
                NarrativeMathStep(
                    purpose="移去常数项",
                    operation="subtract_both_sides",
                    operands=[self._text(constant)],
                    state_before=[current],
                    state_after=[after],
                    reason="等式两边同时减去常数项。",
                )
            )
            current = after
            right = after_right

        if coefficient != 1:
            after = f"x={self._text(simplify(right/coefficient))}"
            steps.append(
                NarrativeMathStep(
                    purpose="把未知数的系数化为一",
                    operation="divide_both_sides",
                    operands=[self._text(coefficient)],
                    state_before=[current],
                    state_after=[after],
                    reason="等式两边同时除以未知数的非零系数。",
                )
            )

        if not steps:
            return None
        return MathRouteDraft(math_steps=steps)
