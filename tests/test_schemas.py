import pytest
from pydantic import ValidationError

from app.schemas import (
    BoardAction,
    GenerationJob,
    Interaction,
    InteractionOption,
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
        problem_text="  解方程 x² - 5x + 6 = 0  ",
        reference_answer="  x = 2 或 x = 3  ",
        required_method="factor",
    )

    assert problem.lesson_length == "standard"
    assert problem.required_method == "factor"
    assert problem.problem_text == "解方程 x² - 5x + 6 = 0"
    assert problem.reference_answer == "x = 2 或 x = 3"


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


@pytest.mark.parametrize("coordinate", ("x", "y"))
def test_board_action_rejects_unknown_coordinate_fields(coordinate):
    with pytest.raises(ValidationError):
        BoardAction.model_validate(
            {
                "type": "focus",
                "target": "factorized_equation",
                coordinate: 20,
            }
        )


def test_top_level_and_nested_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        ProblemInput(
            problem_text="解方程 x + 1 = 0",
            reference_answer="x = -1",
            lesson_lenght="concise",
        )

    with pytest.raises(ValidationError):
        LessonMoment(
            purpose="停顿",
            narration="先想一想。",
            board_actions=[
                {
                    "type": "pause",
                    "unexpected_duration": 2,
                }
            ],
        )


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        (
            ProblemInput,
            {
                "problem_text": "  x  ",
                "reference_answer": "x = 1",
            },
        ),
        (
            ProblemInput,
            {
                "problem_text": "解方程 x = 1",
                "reference_answer": "   ",
            },
        ),
        (
            InteractionOption,
            {
                "option_id": "   ",
                "label": "选项一",
            },
        ),
        (
            RuntimeBeat,
            {
                "beat_id": "   ",
                "purpose": "停顿",
                "narration": "先想一想。",
                "board_actions": [],
                "layer": "base",
            },
        ),
        (
            GenerationJob,
            {
                "job_id": "job-001",
                "status": "queued",
                "stage": "   ",
            },
        ),
    ],
)
def test_required_strings_reject_whitespace_only_values(model, payload):
    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_optional_board_content_is_stripped_and_rejects_whitespace():
    action = BoardAction(
        type="write",
        target="  solution_line  ",
        content="  x = 2  ",
    )

    assert action.target == "solution_line"
    assert action.content == "x = 2"

    with pytest.raises(ValidationError):
        BoardAction(type="focus", target="   ")


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "write", "target": "line_1", "content": "x + 1 = 0"},
        {
            "type": "transform",
            "target": "line_2",
            "content": "x = -1",
        },
        {"type": "focus", "target": "line_2"},
        {"type": "mask", "target": "answer"},
        {"type": "reveal", "target": "answer"},
        {"type": "fade", "target": "line_1"},
        {
            "type": "annotate",
            "target": "coefficient",
            "annotation": "circle",
        },
        {
            "type": "annotate",
            "target": "root",
            "annotation": "label",
            "content": "方程的根",
        },
        {
            "type": "annotate",
            "target": "term_left",
            "annotation": "arrow",
            "relation_target": "term_right",
        },
        {
            "type": "compare",
            "target": "method_factor",
            "relation_target": "method_formula",
        },
        {"type": "pause"},
        {"type": "clear"},
        {"type": "clear", "target": "scratch_area"},
    ],
)
def test_board_action_accepts_executable_payloads(payload):
    action = BoardAction.model_validate(payload)

    assert action.type == payload["type"]


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "write", "content": "x = 1"},
        {"type": "write", "target": "line_1"},
        {"type": "transform", "content": "x = 1"},
        {"type": "transform", "target": "line_1"},
        {"type": "focus"},
        {"type": "mask"},
        {"type": "reveal"},
        {"type": "fade"},
        {"type": "annotate", "annotation": "circle"},
        {"type": "annotate", "target": "term"},
        {
            "type": "annotate",
            "target": "term",
            "annotation": "label",
        },
        {
            "type": "annotate",
            "target": "term",
            "annotation": "arrow",
        },
        {"type": "compare", "relation_target": "method_formula"},
        {"type": "compare", "target": "method_factor"},
    ],
)
def test_board_action_rejects_non_executable_payloads(payload):
    with pytest.raises(ValidationError):
        BoardAction.model_validate(payload)


def test_math_operation_and_lesson_layer_are_restricted():
    with pytest.raises(ValidationError):
        MathStep(
            purpose="求解",
            operation="solve",
            state_before=["x + 1 = 0"],
            state_after=["x = -1"],
            reason="移项",
        )

    with pytest.raises(ValidationError):
        RuntimeBeat(
            beat_id="beat-001",
            purpose="讲解",
            narration="把一移到等号右边。",
            board_actions=[],
            layer="custom_layer",
        )


def test_operand_required_operation_rejects_missing_or_multiple_operands():
    base_payload = {
        "purpose": "等式两边同时加三",
        "operation": "add_both_sides",
        "state_before": ["x = 1"],
        "state_after": ["x + 3 = 4"],
        "reason": "等式两边做相同运算。",
    }

    with pytest.raises(ValidationError):
        MathStep.model_validate(base_payload)

    with pytest.raises(ValidationError):
        MathStep.model_validate({**base_payload, "operands": ["3", "4"]})


def test_operand_free_operation_rejects_operands():
    with pytest.raises(ValidationError):
        MathStep(
            purpose="因式分解",
            operation="factor",
            operands=["6"],
            state_before=["x² - 5x + 6 = 0"],
            state_after=["(x - 2)(x - 3) = 0"],
            reason="寻找乘积为六且和为负五的两个数。",
        )


@pytest.mark.parametrize(
    ("operation", "operand"),
    [
        ("add_both_sides", "3"),
        ("subtract_both_sides", "x"),
        ("complete_the_square", "9"),
    ],
)
def test_structured_operand_operations_accept_one_operand(
    operation,
    operand,
):
    step = MathStep(
        purpose="执行等式变形",
        operation=operation,
        operands=[operand],
        state_before=["x = 1"],
        state_after=["x = 1"],
        reason="记录结构化操作数。",
    )

    assert step.operands == [operand]


def test_math_step_rejects_whitespace_operand():
    with pytest.raises(ValidationError):
        MathStep(
            purpose="等式两边同时加三",
            operation="add_both_sides",
            operands=["   "],
            state_before=["x = 1"],
            state_after=["x + 3 = 4"],
            reason="等式两边做相同运算。",
        )


@pytest.mark.parametrize("empty_field", ("state_before", "state_after"))
def test_math_step_rejects_empty_state_lists(empty_field):
    payload = {
        "purpose": "移项",
        "operation": "subtract_both_sides",
        "operands": ["1"],
        "state_before": ["x + 1 = 0"],
        "state_after": ["x = -1"],
        "reason": "等式两边同时减一。",
    }
    payload[empty_field] = []

    with pytest.raises(ValidationError):
        MathStep.model_validate(payload)


@pytest.mark.parametrize("empty_field", ("math_steps", "moments"))
def test_lesson_draft_rejects_empty_execution_collections(empty_field):
    payload = {
        "title": "移项解方程",
        "learning_goal": "能用等式性质完成移项。",
        "opening": "观察未知数所在的位置。",
        "method_rationale": "通过等式两边同减一来隔离未知数。",
        "math_steps": [
            {
                "purpose": "移项",
                "operation": "subtract_both_sides",
                "operands": ["1"],
                "state_before": ["x + 1 = 0"],
                "state_after": ["x = -1"],
                "reason": "等式两边同时减一。",
            }
        ],
        "moments": [
            {
                "purpose": "解释移项",
                "narration": "等式两边同时减一。",
            }
        ],
        "summary": "对等式两边做相同运算。",
        "transfer_item": {
            "problem_text": "解方程 x + 2 = 0",
            "expected_answer": "x = -2",
            "method_signal": "等式两边同时减二。",
        },
    }
    payload[empty_field] = []

    with pytest.raises(ValidationError):
        LessonDraft.model_validate(payload)


def test_runtime_lesson_requires_beats_but_beat_action_lists_may_be_empty():
    moment = LessonMoment(
        purpose="口头总结",
        narration="等式两边必须做相同运算。",
    )
    beat = RuntimeBeat(
        beat_id="beat-summary",
        purpose="口头总结",
        narration="等式两边必须做相同运算。",
        board_actions=[],
        layer="base",
    )

    assert moment.board_actions == []
    assert beat.board_actions == []

    with pytest.raises(ValidationError):
        RuntimeLesson(
            lesson_id="lesson-empty-beats",
            problem=ProblemInput(
                problem_text="解方程 x + 1 = 0",
                reference_answer="x = -1",
            ),
            title="移项解方程",
            learning_goal="能用等式性质完成移项。",
            beats=[],
            summary="对等式两边做相同运算。",
            transfer_item=TransferItem(
                problem_text="解方程 x + 2 = 0",
                expected_answer="x = -2",
                method_signal="等式两边同时减二。",
            ),
            validation_report={},
        )


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
    with pytest.raises(ValidationError):
        ReviewDecision(
            status="revision_required",
            overall_assessment="需要补充方法选择的说明。",
            must_fix=["   "],
        )
    with pytest.raises(ValidationError):
        ReviewDecision(
            status="approved",
            overall_assessment="数学过程与讲解结构一致。",
            must_fix=["补充一个步骤"],
        )
    with pytest.raises(ValidationError):
        ReviewDecision(
            status="approved",
            overall_assessment="数学过程与讲解结构一致。",
            evidence=["   "],
        )

    approved = ReviewDecision(
        status="approved",
        overall_assessment="数学过程与讲解结构一致。",
    )
    revision = ReviewDecision(
        status="revision_required",
        overall_assessment="需要补充方法选择的说明。",
        must_fix=["  解释为何选择因式分解法  "],
    )

    assert approved.must_fix == []
    assert revision.must_fix == ["解释为何选择因式分解法"]


def test_runtime_beat_and_interaction_accept_audio_urls():
    interaction = Interaction(
        interaction_id="check_roots",
        kind="choice",
        prompt="哪个根满足原方程？",
        expected_answer="both",
        explanation_after_correct="两个根代回原方程都成立。",
        hint_audio_urls=[
            "/audio/hints/check-one.mp3",
            "/audio/hints/check-two.mp3",
        ],
        correct_audio_url="/audio/feedback/correct.mp3",
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
        audio_url="/audio/narration/check-roots.mp3",
    )

    assert beat.audio_url == "/audio/narration/check-roots.mp3"
    assert interaction.hint_audio_urls == [
        "/audio/hints/check-one.mp3",
        "/audio/hints/check-two.mp3",
    ]
    assert interaction.correct_audio_url == "/audio/feedback/correct.mp3"


def test_optional_interaction_feedback_is_normalized_when_provided():
    interaction = Interaction(
        interaction_id="feedback-normalization",
        kind="free_text",
        prompt="请说明理由。",
        expected_answer="等式两边做相同运算。",
        explanation_after_correct="  对，两边必须做相同运算。  ",
    )

    assert interaction.explanation_after_correct == "对，两边必须做相同运算。"

    with pytest.raises(ValidationError):
        Interaction(
            interaction_id="blank-feedback",
            kind="free_text",
            prompt="请说明理由。",
            expected_answer="等式两边做相同运算。",
            explanation_after_correct="   ",
        )


def test_runtime_lesson_rejects_invalid_problem_payload():
    with pytest.raises(ValidationError):
        RuntimeLesson(
            lesson_id="lesson-invalid-problem",
            problem={
                "problem_text": "x",
                "reference_answer": "",
            },
            title="无效题目",
            learning_goal="验证题目输入",
            beats=[],
            summary="无",
            transfer_item=TransferItem(
                problem_text="解方程 x + 1 = 0",
                expected_answer="x = -1",
                method_signal="移项",
            ),
            validation_report={},
        )


@pytest.mark.parametrize(
    "payload",
    [
        {
            "job_id": "job-completed",
            "status": "completed",
            "stage": "runtime_compiled",
        },
        {
            "job_id": "job-failed",
            "status": "failed",
            "stage": "generation",
        },
        {
            "job_id": "job-queued-lesson",
            "status": "queued",
            "stage": "queued",
            "lesson_id": "lesson-001",
        },
        {
            "job_id": "job-queued-error",
            "status": "queued",
            "stage": "queued",
            "error": "not started",
        },
        {
            "job_id": "job-running-lesson",
            "status": "running",
            "stage": "drafting",
            "lesson_id": "lesson-001",
        },
        {
            "job_id": "job-running-error",
            "status": "running",
            "stage": "drafting",
            "error": "still running",
        },
        {
            "job_id": "job-completed-error",
            "status": "completed",
            "stage": "runtime_compiled",
            "lesson_id": "lesson-001",
            "error": "stale error",
        },
        {
            "job_id": "job-failed-lesson",
            "status": "failed",
            "stage": "generation",
            "lesson_id": "lesson-001",
            "error": "model unavailable",
        },
    ],
)
def test_generation_job_rejects_state_inconsistent_payloads(payload):
    with pytest.raises(ValidationError):
        GenerationJob.model_validate(payload)


@pytest.mark.parametrize(
    "payload",
    [
        {"job_id": "job-queued", "status": "queued", "stage": "queued"},
        {"job_id": "job-running", "status": "running", "stage": "drafting"},
        {
            "job_id": "job-completed",
            "status": "completed",
            "stage": "runtime_compiled",
            "lesson_id": "lesson-001",
        },
        {
            "job_id": "job-failed",
            "status": "failed",
            "stage": "generation",
            "error": "model unavailable",
        },
    ],
)
def test_generation_job_accepts_state_consistent_payloads(payload):
    job = GenerationJob.model_validate(payload)

    assert job.status == payload["status"]


def test_mutable_defaults_are_isolated_between_models():
    first = Interaction(
        interaction_id="interaction-first",
        kind="free_text",
        prompt="请说明理由。",
        expected_answer="移项后等式仍然成立。",
    )
    second = Interaction(
        interaction_id="interaction-second",
        kind="free_text",
        prompt="请说明理由。",
        expected_answer="等式两边做相同运算。",
    )

    first.hints.append("观察等号两边。")
    first.options.append(
        InteractionOption(option_id="option-1", label="同时减一")
    )

    assert second.hints == []
    assert second.options == []


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
                content="(x - 2)(x - 3) = 0",
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
        problem=ProblemInput(
            problem_text="解方程 x² - 5x + 6 = 0",
            reference_answer="x = 2 或 x = 3",
            required_method="factor",
        ),
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
    assert runtime.problem.required_method == "factor"
    assert runtime.problem.reference_answer == "x = 2 或 x = 3"
    assert runtime.validation_report == {"math_valid": True}
    assert job.status == "completed"
