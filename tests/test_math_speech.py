import pytest

from app.math_speech import (
    MathSpeechError,
    display_math_to_spoken,
    extract_display_math,
    validate_display_spoken_alignment,
)


@pytest.mark.parametrize(
    ("display", "spoken"),
    (
        ("$m-n$", "m 减 n"),
        (r"\(n\ne0\)", "n 不等于零"),
        (r"\[(2n)^2\]", "二 n 整体的平方"),
        (r"$\frac{1}{2}$", "二分之一"),
        ("$-4(m-n)+2=0$", "负四乘括号 m 减 n 括号加二等于零"),
    ),
)
def test_display_math_to_spoken_has_deterministic_required_readings(display, spoken):
    assert display_math_to_spoken(display) == spoken


def test_display_math_to_spoken_accepts_bounded_delimiter_whitespace():
    assert display_math_to_spoken("$ m-n $") == "m 减 n"


@pytest.mark.parametrize(
    ("display", "spoken"),
    (
        (r"$\sin(m)\ge0$", "正弦括号 m 括号大于等于零"),
        (r"$m\cdot n$", "m乘n"),
        (r"$m\times n$", "m乘n"),
        (r"$n^3<8$", "n的立方小于八"),
        (r"$n\le8$", "n小于等于八"),
        (r"$gcd(m)$", "最大公约数括号 m 括号"),
        (r"$lcm(n)$", "最小公倍数括号 n 括号"),
        (r"$mod(n)$", "模括号 n 括号"),
    ),
)
def test_display_math_to_spoken_supports_approved_functions_and_operators(display, spoken):
    assert display_math_to_spoken(display) == spoken


@pytest.mark.parametrize(
    ("display", "spoken"),
    (
        (r"$x^{2}$", "x的平方"),
        ("$x²$", "x的平方"),
        ("$x³$", "x的立方"),
        ("$x−1=0$", "x 减 一等于零"),
    ),
)
def test_display_math_to_spoken_supports_standard_strict_display_forms(
    display,
    spoken,
):
    assert display_math_to_spoken(display) == spoken


@pytest.mark.parametrize("display", (r"$x^{4}$", r"$x^{23}$", "$x⁴$"))
def test_display_math_to_spoken_rejects_unsupported_exponents(display):
    with pytest.raises(MathSpeechError) as captured:
        display_math_to_spoken(display)
    assert captured.value.code == "unsupported_math_speech"


def test_extract_display_math_supports_only_balanced_supported_delimiters():
    assert extract_display_math(r"由 $m-n$ 且 \(n\ne0\) 得 \[m-n=\frac{1}{2}\]") == [
        "m-n",
        r"n\ne0",
        r"m-n=\frac{1}{2}",
    ]

    for malformed in ("$m-n", r"\(m-n\]", "$$m-n$$", r"\[m-\(n\)\]"):
        with pytest.raises(MathSpeechError) as captured:
            extract_display_math(malformed)
        assert captured.value.code == "unsupported_math_speech"


@pytest.mark.parametrize(
    "display",
    (
        r"$\href{x}{m-n}$",
        r"$\unknown{m}$",
        r"$m_{n}$",
        "$https://example.com$",
        "$<span>m</span>$",
        "$m+(n$",
        "$m+{}$",
    ),
)
def test_display_math_to_spoken_fails_closed_for_unsupported_content(display):
    with pytest.raises(MathSpeechError) as captured:
        display_math_to_spoken(display)
    assert captured.value.code == "unsupported_math_speech"


def test_display_math_to_spoken_is_bounded_and_total_safe():
    with pytest.raises(MathSpeechError) as captured:
        display_math_to_spoken("$" + "m" * 501 + "$")
    assert captured.value.code == "unsupported_math_speech"

    for value in (None, 1, [], {}):
        with pytest.raises(MathSpeechError) as captured:
            display_math_to_spoken(value)  # type: ignore[arg-type]
        assert captured.value.code == "unsupported_math_speech"


def test_display_spoken_alignment_requires_every_deterministic_reading():
    validate_display_spoken_alignment(
        r"已知 \(n\ne0\)，板书 $m-n=\frac{1}{2}$。",
        "因为 n 不等于零，所以 m 减 n 等于二分之一。",
    )

    with pytest.raises(MathSpeechError) as captured:
        validate_display_spoken_alignment(
            r"已知 \(n\ne0\)，板书 $m-n=\frac{1}{2}$。",
            "所以 m 减 n 等于二分之一。",
        )
    assert captured.value.code == "display_spoken_math_mismatch"


@pytest.mark.parametrize(
    ("display", "spoken"),
    (
        ("$m$ 再看 $n$", "先说 n，再说 m。"),
        ("$m$ 与 $m$", "只说一次 m。"),
        ("$m$", "math 方法"),
        ("$1$", "结果是十一。"),
        ("$1$ 与 $11$", "结果是十一。"),
    ),
)
def test_display_spoken_alignment_rejects_reordered_overlapping_or_partial_tokens(
    display,
    spoken,
):
    with pytest.raises(MathSpeechError) as captured:
        validate_display_spoken_alignment(display, spoken)
    assert captured.value.code == "display_spoken_math_mismatch"


def test_display_spoken_alignment_keeps_natural_sentence_matches():
    validate_display_spoken_alignment(
        "$m$ 与 $1$",
        "变量 m 的当前取值是一。",
    )
