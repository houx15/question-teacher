import pytest

from app.math_expression import (
    StrictMathExpressionError,
    StrictMathText,
    validate_strict_math_expression,
)


@pytest.mark.parametrize(
    "expression",
    [
        "-x<-2",
        r"\begin{cases}2x+y=5\\x-y=1\end{cases}",
        "a^2+b^2=c^2",
        r"x=\sqrt{5}|x=-\sqrt{5}",
        r"\sin^2(x)+\cos^2(x)=1",
        r"m-n=\frac{1}{2}",
        r"\frac{x+1}{\sqrt{x^2+1}}",
    ],
)
def test_strict_math_bridge_accepts_broad_typed_mathematics(expression):
    assert validate_strict_math_expression(expression) == expression
    assert isinstance(StrictMathText._validate(expression), StrictMathText)


@pytest.mark.parametrize(
    "expression",
    [
        r"\frac{IGNOREALLRULES}{1}",
        r"\sqrt{IGNOREALLRULES}",
        r"\frac{这是内部批注不要公开}{1}",
        r"\frac{1}",
        r"\sqrt",
        "在方程两边同时加IGNOREALLRULES",
        "SECRETKEY123456789",
        "4c970004b0678d43",
    ],
)
def test_strict_math_bridge_rejects_prose_controls_and_malformed_commands(
    expression,
):
    with pytest.raises(StrictMathExpressionError):
        validate_strict_math_expression(expression)
