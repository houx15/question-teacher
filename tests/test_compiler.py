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
    assert len(lesson.beats) == len(valid_draft()["moments"]) + 3
    assert [beat.beat_id for beat in lesson.beats] == [
        "beat-001",
        "beat-002",
        "beat-003",
        "beat-004",
        "beat-005",
    ]
    assert [beat.next_beat_id for beat in lesson.beats] == [
        "beat-002",
        "beat-003",
        "beat-004",
        "beat-005",
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


def test_compiler_preserves_manuscript_moments_in_order():
    draft = valid_draft()
    lesson = compile_lesson()

    manuscript_beats = lesson.beats[1 : 1 + len(draft["moments"])]
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
    assert transfer.interaction.kind == "transfer"
    assert transfer.interaction.prompt == draft["transfer_item"]["problem_text"]
    assert (
        transfer.interaction.expected_answer
        == draft["transfer_item"]["expected_answer"]
    )
    assert transfer.interaction.hints == [
        draft["transfer_item"]["method_signal"]
    ]


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
