import pytest

from app.compiler import LessonCompileError, LessonCompiler
from app.schemas import LessonDraft
from tests.test_generation import problem, valid_draft


def compile_lesson(compiler=None):
    return (compiler or LessonCompiler()).compile(
        problem(),
        LessonDraft.model_validate(valid_draft()),
        {"review_status": "approved"},
    )


def test_compiler_creates_ordered_beats_with_stable_navigation():
    lesson = compile_lesson(
        LessonCompiler(lesson_id_factory=lambda: "lesson-fixed")
    )

    assert lesson.lesson_id == "lesson-fixed"
    assert len(lesson.beats) == len(valid_draft()["moments"]) + 4
    assert [beat.beat_id for beat in lesson.beats] == [
        "beat-001",
        "beat-002",
        "beat-003",
        "beat-004",
        "beat-005",
        "beat-006",
    ]
    assert [beat.next_beat_id for beat in lesson.beats] == [
        "beat-002",
        "beat-003",
        "beat-004",
        "beat-005",
        "beat-006",
        None,
    ]


def test_compiler_opens_by_writing_the_original_problem():
    lesson = compile_lesson()
    opening = lesson.beats[0]

    assert opening.purpose == "进入问题"
    assert opening.narration == valid_draft()["opening"]
    assert len(opening.board_actions) == 1
    assert opening.board_actions[0].model_dump(exclude_none=True) == {
        "type": "write",
        "target": "original_problem",
        "content": problem().problem_text,
    }


@pytest.mark.parametrize(
    ("student_definition", "why_it_helps", "expected_narration"),
    [
        (
            "把二次式写成两个一次因式的乘积",
            "这样能把二次方程拆成一次方程",
            "今天用因式分解法。把二次式写成两个一次因式的乘积。这样能把二次方程拆成一次方程。",
        ),
        (
            "把二次式写成两个一次因式的乘积。",
            "这样能把二次方程拆成一次方程！",
            "今天用因式分解法。把二次式写成两个一次因式的乘积。这样能把二次方程拆成一次方程！",
        ),
    ],
)
def test_compiler_introduces_method_before_first_manuscript_moment(
    student_definition,
    why_it_helps,
    expected_narration,
):
    draft = valid_draft()
    draft["method_introduction"]["student_definition"] = (
        student_definition
    )
    draft["method_introduction"]["why_it_helps"] = why_it_helps
    lesson = LessonCompiler().compile(
        problem(),
        LessonDraft.model_validate(draft),
        {"review_status": "approved"},
    )
    introduction = draft["method_introduction"]
    method_beat = lesson.beats[1]

    assert method_beat.purpose == "先认识方法"
    assert method_beat.layer == "micro_explanation"
    assert method_beat.narration.startswith(
        f"今天用{introduction['method_name']}。"
    )
    assert method_beat.narration == expected_narration
    assert introduction["student_definition"] in method_beat.narration
    assert introduction["why_it_helps"] in method_beat.narration
    assert introduction["target_form"] not in method_beat.narration
    assert r"\(" not in method_beat.narration
    assert [
        action.model_dump(exclude_none=True)
        for action in method_beat.board_actions
    ] == [
        {
            "type": "write",
            "target": "method_name",
            "content": introduction["method_name"],
        },
        {"type": "focus", "target": "method_name"},
        {
            "type": "write",
            "target": "method_target_form",
            "content": introduction["target_form"],
        },
    ]


def test_compiler_preserves_manuscript_moments_in_order():
    draft = valid_draft()
    lesson = compile_lesson()

    manuscript_beats = lesson.beats[2 : 2 + len(draft["moments"])]
    assert [beat.purpose for beat in manuscript_beats] == [
        moment["purpose"] for moment in draft["moments"]
    ]
    assert manuscript_beats[0].interaction.interaction_id == "find-factor-pair"
    assert manuscript_beats[1].board_actions[0].type == "transform"


def test_compiler_appends_summary_then_transfer_interaction():
    draft = valid_draft()
    lesson = compile_lesson()
    summary = lesson.beats[-2]
    transfer = lesson.beats[-1]

    assert summary.purpose == "压缩方法"
    assert summary.board_actions[0].type == "write"
    assert summary.board_actions[0].target == "method_summary"
    assert summary.board_actions[0].content == draft["summary"]
    assert transfer.purpose == "完成近迁移"
    assert transfer.layer == "interaction"
    assert transfer.interaction.kind == "choice"
    assert transfer.interaction.prompt == draft["transfer_item"]["problem_text"]
    assert (
        transfer.interaction.expected_answer
        == draft["transfer_item"]["correct_option_id"]
    )
    assert [
        option.model_dump(exclude_none=True)
        for option in transfer.interaction.options
    ] == [
        {
            key: option[key]
            for key in ("option_id", "label", "feedback")
        }
        for option in draft["transfer_item"]["options"]
    ]
    assert transfer.interaction.hints == [
        draft["transfer_item"]["method_signal"]
    ]


def test_compiler_preserves_legacy_text_transfer_without_options():
    draft = valid_draft()
    draft["transfer_item"]["options"] = []
    draft["transfer_item"]["correct_option_id"] = None
    lesson = LessonCompiler().compile(
        problem(),
        LessonDraft.model_validate(draft),
        {"review_status": "approved"},
    )

    transfer = lesson.beats[-1].interaction
    assert transfer.kind == "transfer"
    assert transfer.expected_answer == draft["transfer_item"]["expected_answer"]
    assert transfer.options == []


def test_compiler_rejects_missing_canonicalized_transfer_label_safely():
    draft = valid_draft()
    private_option_id = "private-option-without-label"
    draft["transfer_item"]["options"][0]["option_id"] = private_option_id
    draft["transfer_item"]["correct_option_id"] = private_option_id
    draft["transfer_item"]["options"][0].pop("label")

    with pytest.raises(LessonCompileError) as exc_info:
        LessonCompiler().compile(
            problem(),
            LessonDraft.model_validate(draft),
            {"review_status": "approved"},
        )

    assert str(exc_info.value) == "近迁移选项缺少已规范化的显示标签。"
    assert private_option_id not in str(exc_info.value)


def test_compiler_rejects_overlong_method_spoken_narration():
    draft = valid_draft()
    private_narration = "私有方法说明" * 20
    draft["method_introduction"]["student_definition"] = private_narration
    draft["method_introduction"]["why_it_helps"] = "便于求根"

    with pytest.raises(LessonCompileError) as exc_info:
        LessonCompiler().compile(
            problem(),
            LessonDraft.model_validate(draft),
            {"review_status": "approved"},
        )

    assert str(exc_info.value) == "方法介绍的口语讲稿过长。"
    assert private_narration not in str(exc_info.value)


@pytest.mark.parametrize(
    "unsafe_id",
    ["", "   ", "../lesson", "lesson/other", "lesson\nother", "x" * 129],
)
def test_compiler_rejects_unsafe_injected_lesson_id(unsafe_id):
    compiler = LessonCompiler(lesson_id_factory=lambda: unsafe_id)

    with pytest.raises(LessonCompileError, match="lesson id"):
        compile_lesson(compiler)


def test_compiler_hides_lesson_id_factory_errors():
    def broken_factory():
        raise RuntimeError("private-id-detail")

    compiler = LessonCompiler(lesson_id_factory=broken_factory)

    with pytest.raises(LessonCompileError) as exc_info:
        compile_lesson(compiler)

    assert "private-id-detail" not in str(exc_info.value)
