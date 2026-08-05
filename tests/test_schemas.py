import pytest
from pydantic import ValidationError

from app.schemas import (
    BoardAction,
    GenerationJob,
    Interaction,
    LessonDraft,
    LessonMoment,
    MathStep,
    ProblemInput,
    ReviewDecision,
    RuntimeBeat,
    RuntimeLesson,
    TransferItem,
)


def test_problem_input_accepts_valid_example_and_defaults_to_standard():
    problem = ProblemInput(
        problem_text="解方程 x² - 5x + 6 = 0",
        reference_answer="x = 2 或 x = 3",
        required_method="factor",
    )

    assert problem.lesson_length == "standard"
    assert problem.required_method == "factor"


def test_board_action_uses_semantic_target_without_coordinates():
    action = BoardAction(
        type="focus",
        target="factorized_equation",
        content="(x - 2)(x - 3) = 0",
    )

    dumped = action.model_dump()

    assert dumped["target"] == "factorized_equation"
    assert "x" not in dumped
    assert "y" not in dumped


def test_interaction_requires_expected_answer():
    with pytest.raises(ValidationError):
        Interaction(
            interaction_id="identify_factors",
            kind="expression",
            prompt="哪两个数的积是 6、和是 -5？",
        )


def test_lesson_moment_combines_micro_explanation_actions_and_interaction():
    moment = LessonMoment(
        purpose="解释因式分解的两个因式",
        narration="先找乘积为六、和为负五的两个数，再写成两个一次因式。",
        layer="micro_explanation",
        board_actions=[
            BoardAction(type="focus", target="constant_and_linear_terms"),
            BoardAction(
                type="annotate",
                target="factor_pair",
                annotation="circle",
                content="-2, -3",
            ),
        ],
        interaction=Interaction(
            interaction_id="factor_pair",
            kind="expression",
            prompt="请写出这两个数。",
            expected_answer="-2, -3",
        ),
    )

    assert len(moment.narration) <= 90
    assert moment.layer == "micro_explanation"
    assert [action.type for action in moment.board_actions] == [
        "focus",
        "annotate",
    ]
    assert moment.board_actions[1].annotation == "circle"
    assert moment.interaction is not None
    assert moment.interaction.kind == "expression"


def test_lesson_moment_rejects_narration_longer_than_90_characters():
    with pytest.raises(ValidationError):
        LessonMoment(
            purpose="过长讲解",
            narration="讲" * 91,
        )


def test_review_decision_requires_fixes_only_when_revision_is_required():
    with pytest.raises(ValidationError):
        ReviewDecision(
            status="revision_required",
            overall_assessment="需要补充方法选择的说明。",
        )

    approved = ReviewDecision(
        status="approved",
        overall_assessment="数学过程与讲解结构一致。",
    )

    assert approved.must_fix == []


def test_runtime_beat_and_interaction_accept_audio_urls():
    interaction = Interaction(
        interaction_id="check_roots",
        kind="choice",
        prompt="哪个根满足原方程？",
        expected_answer="both",
        explanation_after_correct="两个根代回原方程都成立。",
        hint_audio_urls=[
            "https://media.example/hints/check-one.mp3",
            "https://media.example/hints/check-two.mp3",
        ],
        correct_audio_url="https://media.example/feedback/correct.mp3",
    )
    beat = RuntimeBeat(
        beat_id="beat-check-roots",
        purpose="检查解",
        narration="把两个根分别代回原方程。",
        board_actions=[
            BoardAction(type="focus", target="solution_set"),
        ],
        layer="interaction",
        interaction=interaction,
        audio_url="https://media.example/narration/check-roots.mp3",
    )

    assert beat.audio_url == "https://media.example/narration/check-roots.mp3"
    assert interaction.hint_audio_urls == [
        "https://media.example/hints/check-one.mp3",
        "https://media.example/hints/check-two.mp3",
    ]
    assert (
        interaction.correct_audio_url
        == "https://media.example/feedback/correct.mp3"
    )


def test_lesson_and_job_contracts_support_nested_runtime_data():
    math_step = MathStep(
        purpose="得到两个一次因式",
        operation="factor",
        state_before=["x² - 5x + 6 = 0"],
        state_after=["(x - 2)(x - 3) = 0"],
        reason="乘积为六且和为负五的两个数是负二和负三。",
    )
    moment = LessonMoment(
        purpose="展示因式分解",
        narration="把二次式分解成两个一次因式。",
        board_actions=[
            BoardAction(
                type="transform",
                source="original_equation",
                target="factorized_equation",
            ),
        ],
    )
    transfer_item = TransferItem(
        problem_text="解方程 x² - 7x + 12 = 0",
        expected_answer="x = 3 或 x = 4",
        method_signal="寻找乘积为常数项、和为一次项系数的两个数。",
    )
    draft = LessonDraft(
        title="用因式分解法解一元二次方程",
        learning_goal="能根据系数找到因式并求根。",
        opening="观察常数项和一次项系数。",
        method_rationale="整式容易分解，因式分解法步骤最短。",
        math_steps=[math_step],
        moments=[moment],
        summary="先分解，再令每个因式等于零。",
        transfer_item=transfer_item,
    )
    runtime = RuntimeLesson(
        lesson_id="lesson-factor-001",
        problem="x² - 5x + 6 = 0",
        title=draft.title,
        learning_goal=draft.learning_goal,
        beats=[
            RuntimeBeat(
                beat_id="beat-factor",
                purpose=moment.purpose,
                narration=moment.narration,
                board_actions=moment.board_actions,
                layer=moment.layer,
            ),
        ],
        summary=draft.summary,
        transfer_item=transfer_item,
        validation_report={"math_valid": True},
    )
    job = GenerationJob(
        job_id="job-001",
        status="completed",
        stage="runtime_compiled",
        lesson_id=runtime.lesson_id,
    )

    assert draft.math_steps[0].operation == "factor"
    assert runtime.validation_report == {"math_valid": True}
    assert job.status == "completed"
