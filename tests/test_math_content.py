import pytest
from pydantic import ValidationError

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
from app.schemas import LessonMoment, NarrativeSyncCue


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


@pytest.mark.parametrize(
    "value",
    (
        '<span class="is-highlighted">x</span>',
        ".is-highlighted",
        "#board-target",
        "[data-highlight]",
        "[[red]",
        "[[",
        "]]",
        "{{highlight",
        "<mark",
    ),
)
def test_malformed_html_selector_and_highlight_fragments_are_internal_control_syntax(value):
    assert contains_internal_control_syntax(value)
    assert not is_valid_generated_display_content(value)


@pytest.mark.parametrize(
    "command",
    (
        "url",
        "href",
        "includegraphics",
        "htmlClass",
        "htmlId",
        "htmlStyle",
        "htmlData",
    ),
)
def test_full_katex_trust_command_group_is_internal_control_syntax(command):
    value = r"\(\%s{payload}{x}\)" % command
    assert contains_internal_control_syntax(value)
    assert not is_valid_generated_display_content(value)


@pytest.mark.parametrize(
    "value",
    (
        '<img src="lesson.png">',
        '<img src="lesson.png" />',
        '<DIV class="lesson">content</DIV>',
        '<section data-target="board-1"   >content</section   >',
    ),
)
def test_generic_html_tags_are_internal_control_syntax(value):
    assert contains_internal_control_syntax(value)
    assert not is_valid_generated_display_content(value)


@pytest.mark.parametrize(
    "value",
    (
        "x<2",
        "a>b",
        "用<角括号表达但不闭合",
        "用 <说明 文字> 表达普通内容",
        "0.25",
        "观察[已知条件]",
        "语义标识 board-1 与 problem-root",
        "由（1）得到结论",
        r"\(x<2\)",
        r"\({{x}}+1\)",
        r"\[m-n=\frac{1}{2}\]",
    ),
)
def test_control_syntax_detector_preserves_ordinary_math_and_chinese(value):
    assert not contains_internal_control_syntax(value)
    assert is_valid_generated_display_content(value)


def test_legacy_spoken_cue_uses_shared_internal_control_syntax_boundary():
    with pytest.raises(ValidationError, match="natural speech"):
        NarrativeSyncCue(
            cue_id="cue-internal-control",
            spoken_text='<span class="is-highlighted">重点</span>',
        )


@pytest.mark.parametrize(
    "spoken_text",
    (
        r"\(\htmlClass{is-highlighted}{x}\)",
        r"\(\htmlId{board-target}{x}\)",
        r"\[\htmlStyle{color:red}{x}\]",
        r"\(\htmlData{target=board-1}{x}\)",
        '<img src="lesson.png">',
        '<IMG SRC="lesson.png" />',
        '<section data-target="board-1">重点</section>',
    ),
)
def test_legacy_spoken_cue_rejects_generic_html_tags(spoken_text):
    with pytest.raises(ValidationError, match="natural speech"):
        NarrativeSyncCue(
            cue_id="cue-generic-html",
            spoken_text=spoken_text,
        )


def test_legacy_narration_conversion_keeps_compatibility_error_for_internal_control():
    with pytest.raises(
        ValidationError,
        match="legacy narration is not compatible",
    ):
        LessonMoment.model_validate(
            {
                "purpose": "检查旧格式边界",
                "narration": "[[highlight:board-1]]",
                "board_actions": [],
                "layer": "base",
            }
        )


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
