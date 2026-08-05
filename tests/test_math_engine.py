from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from app.math_engine import MathEngine, MathValidationError
from app.schemas import MathStep


@pytest.fixture
def engine():
    return MathEngine()


def make_step(operation, before, after):
    return MathStep(
        purpose="验证代数操作",
        operation=operation,
        state_before=before,
        state_after=after,
        reason="测试操作标签与变形结构一致。",
    )


def test_validate_problem_returns_sorted_quadratic_solutions(engine):
    validation = engine.validate_problem(
        "x^2-6x+5=0",
        "x=1 或 x=5",
    )

    assert validation.solution_strings == ["1", "5"]


def test_problem_validation_is_immutable(engine):
    validation = engine.validate_problem("x=1", "x=1")

    with pytest.raises(FrozenInstanceError):
        validation.solution_strings = ["2"]
    with pytest.raises(TypeError):
        validation.solution_strings.append("2")


def test_validate_problem_rejects_conflicting_reference_answer(engine):
    with pytest.raises(MathValidationError, match="参考答案"):
        engine.validate_problem("2x+3=7", "x=3")


def test_validate_step_accepts_equivalent_split_branches(engine):
    step = MathStep(
        purpose="拆分正负分支",
        operation="split_plus_minus",
        state_before=["(x-3)^2=4"],
        state_after=["x-3=2", "x-3=-2"],
        reason="平方等于四时，底数等于正二或负二。",
    )

    engine.validate_step(step)


def test_validate_step_rejects_changed_solution_set(engine):
    step = MathStep(
        purpose="移项",
        operation="subtract_both_sides",
        state_before=["2x+3=7"],
        state_after=["2x=10"],
        reason="错误示例。",
    )

    with pytest.raises(MathValidationError, match="解集不一致"):
        engine.validate_step(step)


def test_validate_step_rejects_mislabeled_factor_operation(engine):
    step = make_step("factor", ["x=1"], ["2x=2"])

    with pytest.raises(MathValidationError):
        engine.validate_step(step)


def test_validate_step_accepts_factoring(engine):
    step = make_step(
        "factor",
        ["x^2-5x+6=0"],
        ["(x-2)(x-3)=0"],
    )

    engine.validate_step(step)


def test_validate_step_accepts_expansion(engine):
    step = make_step(
        "expand",
        ["(x-2)(x-3)=0"],
        ["x^2-5x+6=0"],
    )

    engine.validate_step(step)


@pytest.mark.parametrize(
    ("operation", "before", "after"),
    (
        ("add_both_sides", ["x=1"], ["x+3=4"]),
        ("multiply_both_sides", ["x+1=2"], ["2(x+1)=4"]),
        ("divide_both_sides", ["2x+2=4"], ["x+1=2"]),
    ),
)
def test_validate_step_accepts_same_operation_on_both_sides(
    engine,
    operation,
    before,
    after,
):
    engine.validate_step(make_step(operation, before, after))


def test_validate_step_rejects_different_changes_on_equation_sides(engine):
    step = make_step("add_both_sides", ["x=1"], ["2x=2"])

    with pytest.raises(MathValidationError):
        engine.validate_step(step)


def test_validate_step_accepts_completing_the_square(engine):
    step = make_step(
        "complete_the_square",
        ["x^2-6x=7"],
        ["(x-3)^2=16"],
    )

    engine.validate_step(step)


def test_validate_step_accepts_combining_like_terms(engine):
    step = make_step(
        "combine_like_terms",
        ["x+x=2"],
        ["2x=2"],
    )

    engine.validate_step(step)


def test_validate_step_rejects_simplification_that_increases_operations(
    engine,
):
    step = make_step(
        "simplify",
        ["2x=2"],
        ["x+x=1+1"],
    )

    with pytest.raises(MathValidationError):
        engine.validate_step(step)


def test_validate_step_accepts_taking_square_roots(engine):
    step = make_step(
        "take_square_root_both_sides",
        ["x^2=4"],
        ["x=2", "x=-2"],
    )

    engine.validate_step(step)


def test_validate_step_accepts_quadratic_formula_branches(engine):
    step = make_step(
        "quadratic_formula",
        ["x^2-5x+6=0"],
        ["x=2", "x=3"],
    )

    engine.validate_step(step)


def test_validate_step_rejects_quadratic_formula_without_solved_branches(
    engine,
):
    step = make_step(
        "quadratic_formula",
        ["x^2-5x+6=0"],
        ["(x-2)(x-3)=0"],
    )

    with pytest.raises(MathValidationError):
        engine.validate_step(step)


def test_expressions_equivalent_after_expansion(engine):
    assert engine.expressions_equivalent(
        "(x-3)^2",
        "x^2-6x+9",
    )


def test_answers_equivalent_ignores_branch_order(engine):
    assert engine.answers_equivalent(
        "x=5 or x=1",
        "x=1 or x=5",
    )


def test_decimal_coefficients_are_solved_exactly(engine):
    validation = engine.validate_problem("0.1*x=0.3", "x=3")

    assert validation.solution_strings == ["3"]


def test_decimal_reference_answers_are_exact(engine):
    assert engine.answers_equivalent("x=3", "x=3.0")


def test_chinese_prompt_and_unicode_symbols_are_normalized(engine):
    validation = engine.validate_problem(
        "请解方程：X²－5×X＋6＝0",
        "x=2 或 x=3",
    )

    assert validation.solution_strings == ["2", "3"]


@pytest.mark.parametrize(
    ("problem", "reference", "expected"),
    (
        ("2x+1=0", "x=-1/2", ["-1/2"]),
        ("x^2=2", "x=-sqrt(2) 或 x=sqrt(2)", ["-sqrt(2)", "sqrt(2)"]),
    ),
)
def test_fraction_and_radical_roots_parse_and_compare(
    engine,
    problem,
    reference,
    expected,
):
    validation = engine.validate_problem(problem, reference)

    assert validation.solution_strings == expected


@pytest.mark.parametrize(
    "equation",
    (
        "x/x=x+1",
        "(x-1)/(x-1)=x",
    ),
)
def test_equations_with_variable_denominators_are_rejected(
    engine,
    equation,
):
    with pytest.raises(MathValidationError):
        engine.solution_set([equation])


@pytest.mark.parametrize(
    "unsafe_expression",
    (
        "__import__('os')",
        "y+1",
        "x.real",
        "x.sqrt",
        "x[0]",
        "{x}",
        "sin(x)",
        "sqrt.__call__(x)",
        "sqrt",
        "x//2",
        "x**2",
    ),
)
def test_parse_expression_rejects_unsafe_or_unknown_syntax(
    engine,
    unsafe_expression,
):
    with pytest.raises(MathValidationError):
        engine.parse_expression(unsafe_expression)


@pytest.mark.parametrize(
    "malformed_equation",
    (
        "x==1",
        "x=1=2",
        "x+1",
        "=1",
        "x=",
    ),
)
def test_parse_equation_rejects_malformed_input(engine, malformed_equation):
    with pytest.raises(MathValidationError):
        engine.parse_equation(malformed_equation)


def test_parse_equation_rejects_prompt_prefix_and_colon(engine):
    with pytest.raises(MathValidationError):
        engine.parse_equation("请解方程：x=1")


def test_validate_problem_alone_extracts_final_colon_suffix(engine):
    validation = engine.validate_problem(
        "课堂题目: 请计算：x+1=2",
        "x=1",
    )

    assert validation.solution_strings == ["1"]


@pytest.mark.parametrize(
    "reference",
    (
        "x^2-1=0",
        "x=1 or 1=2",
        "x=1 or",
        "x=1 trailing",
    ),
)
def test_reference_answers_require_anchored_solved_x_branches(
    engine,
    reference,
):
    with pytest.raises(MathValidationError):
        engine.answers_equivalent(reference, "x=1")


def test_solution_set_rejects_empty_raw_state_list(engine):
    with pytest.raises(MathValidationError):
        engine.solution_set([])


def test_solution_set_accepts_an_equation_with_no_real_solutions(engine):
    assert engine.solution_set(["x^2+1=0"]).is_empty


def test_solution_set_rejects_more_than_four_branches(engine):
    with pytest.raises(MathValidationError):
        engine.solution_set(["x=1", "x=2", "x=3", "x=4", "x=5"])


@pytest.mark.parametrize(
    "expression",
    (
        "1" * 257,
        "9^9^9",
        "1234567890123",
        "(" * 13 + "x" + ")" * 13,
        "x" + "+1" * 65,
    ),
)
def test_parse_expression_rejects_excessive_complexity(engine, expression):
    with pytest.raises(MathValidationError):
        engine.parse_expression(expression)


@pytest.mark.parametrize(
    "equation",
    (
        "x^5=1",
        "sqrt(x)=1",
    ),
)
def test_parse_equation_rejects_unsupported_nonquadratic_shapes(
    engine,
    equation,
):
    with pytest.raises(MathValidationError):
        engine.parse_equation(equation)


def test_radical_coefficients_remain_supported(engine):
    validation = engine.validate_problem(
        "sqrt(2)*x=2",
        "x=sqrt(2)",
    )

    assert validation.solution_strings == ["sqrt(2)"]


def test_validate_step_fails_safely_for_raw_empty_state(engine):
    step = SimpleNamespace(state_before=[], state_after=["x=1"])

    with pytest.raises(MathValidationError):
        engine.validate_step(step)


@pytest.mark.parametrize(
    ("problem", "reference"),
    (
        ("x=x", "x=0"),
        ("0=0", "x=0"),
    ),
)
def test_validate_problem_rejects_non_finite_solution_sets(
    engine,
    problem,
    reference,
):
    with pytest.raises(MathValidationError, match="有限"):
        engine.validate_problem(problem, reference)


def test_invalid_expression_equivalence_raises_instead_of_returning_false(
    engine,
):
    with pytest.raises(MathValidationError):
        engine.expressions_equivalent("x+1", "malicious(x)")
