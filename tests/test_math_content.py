import pytest

from app.math_content import (
    contains_explicit_choice_answer_leak,
    contains_internal_control_syntax,
    contains_math_markup,
    is_valid_generated_display_content,
    normalize_answer_leak_text,
    normalize_choice_option_label,
    normalize_grounded_choice_option_label,
    normalize_reference_text,
)


@pytest.mark.parametrize(
    "value",
    ("$x=1$", "$$x=1$$", r"\(x=1\)", r"\[x=1\]"),
)
def test_contains_math_markup_detects_supported_delimiters(value):
    assert contains_math_markup(value)


def test_contains_math_markup_preserves_legacy_backslash_detection():
    assert contains_math_markup(r"使用 \frac{1}{2} 计算")
    assert not contains_math_markup("先观察等式两边。")


@pytest.mark.parametrize(
    "value",
    (
        "先代入，再得到结果。",
        r"代入后得到 \(4n^2-4mn+2n=0\)。",
        r"整理为 \[m-n=\dfrac{1}{2}\]。",
        r"路径字面量 C:\\temp 与公式 \(x=1\)",
        "$x=1$",
        "$$x=1$$",
    ),
)
def test_generated_display_validator_accepts_natural_chinese_and_valid_math(value):
    assert is_valid_generated_display_content(value)


@pytest.mark.parametrize(
    "value",
    (
        "$x=1",
        "$$x=1$",
        r"\(x=1$",
        r"\[x=1\)",
        r"\(x=1\) and $y=2$",
        r"\(x=\frac{1}{2\)",
    ),
)
def test_generated_display_validator_rejects_unbalanced_or_mixed_markup(value):
    assert not is_valid_generated_display_content(value)


@pytest.mark.parametrize(
    "value",
    (
        "[[target:problem-math-001]]",
        "[[red]]",
        "{{highlight:board-1}}",
        '<mark data-target="board-1">x</mark>',
        "data-target=board-1",
    ),
)
def test_internal_target_and_highlight_syntax_is_rejected(value):
    assert contains_internal_control_syntax(value)
    assert not is_valid_generated_display_content(value)


def test_grounded_choice_normalizer_collapses_bounded_katex_equivalents():
    equivalents = (
        r"\(\dfrac{1}{2}\)",
        r"\( \frac {1}{2} \)",
        r"$\tfrac{1}{2}$",
    )
    assert len({normalize_grounded_choice_option_label(item) for item in equivalents}) == 1
    assert normalize_grounded_choice_option_label("２－１＝１") == "２-１=１"
    assert normalize_grounded_choice_option_label(r"\(x^{2}\)") == "x^2"


def test_legacy_normalizers_keep_their_exact_results():
    value = r"  选项   \( x + 1 = 2 \)  "
    assert normalize_choice_option_label(value) == r"选项 \(x+1=2\)"
    assert normalize_grounded_choice_option_label(value) == "选项x+1=2"
    assert normalize_answer_leak_text(r"\(\left(x^{2}+1\right)\)") == "x^2+1"
    assert normalize_reference_text(r"\(\dfrac{1}{2}\)") == "1/2"


def test_bounded_answer_leak_detection_requires_explicit_short_answer_context():
    assert contains_explicit_choice_answer_leak("正确答案是A", "A", "x=1")
    assert not contains_explicit_choice_answer_leak("A出现在代数式中", "A", "x=1")
    assert contains_explicit_choice_answer_leak(
        r"答案应为 \(x=1\)", "option-a", r"\(x=1\)"
    )
