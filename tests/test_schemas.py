import pytest
from pydantic import ValidationError

from app.schemas import (
    BoardAction,
    GenerationJob,
    GeneratedTransferOption,
    Interaction,
    InteractionOption,
    LessonDraft,
    LessonMoment,
    MathStep,
    MethodIntroduction,
    NarrativeSyncCue,
    ProblemInput,
    ReferenceGroundingBrief,
    ReferenceMaterialAudit,
    ReviewDecision,
    RuntimeBeat,
    RuntimeLesson,
    SyncVisualAction,
    TransferItem,
    TransferOption,
)


def grounding_brief_payload():
    return {
        "task_summary": "把已知根代回方程，求m-n",
        "target": r"\(m-n\)",
        "assumptions": [r"\(n\ne0\)", r"\(x=2n\)是原方程的根"],
        "reference_conclusion": r"\(m-n=\frac12\)",
        "method_name": "代入法",
        "reasoning_steps": [
            {
                "step_id": "substitute-root",
                "statement_before": r"\(x^2-2mx+2n=0\)",
                "operation_explanation": "把已知根x=2n代入原方程",
                "statement_after": r"\(4n^2-4mn+2n=0\)",
            },
            {
                "step_id": "use-nonzero",
                "statement_before": r"\(2n(2n-2m+1)=0\)",
                "operation_explanation": "利用n不为0约去2n",
                "statement_after": r"\(2n-2m+1=0\)",
            },
        ],
        "check_requests": [
            {
                "check_id": "check-substitution",
                "kind": "substitution",
                "expression": "x^2-2*m*x+2*n",
                "expected": "4*n^2-4*m*n+2*n",
                "substitutions": {"x": "2*n"},
                "nonzero_symbols": [],
                "conclusion_linked": True,
            }
        ],
        "audit_notes": [],
    }


def test_reference_grounding_brief_is_bounded_and_structured():
    brief = ReferenceGroundingBrief.model_validate(
        grounding_brief_payload(),
        context={"reference_answer": r"\(m-n=\frac12\)"},
    )

    assert brief.method_name == "代入法"
    assert brief.check_requests[0].conclusion_linked is True


def test_reference_grounding_brief_accepts_all_and_only_local_check_kinds():
    payload = grounding_brief_payload()
    payload["check_requests"] = [
        {
            **payload["check_requests"][0],
            "check_id": f"check-{kind}",
            "kind": kind,
        }
        for kind in (
            "substitution",
            "equivalence",
            "nonzero_division",
            "back_substitution",
        )
    ]

    brief = ReferenceGroundingBrief.model_validate(
        payload,
        context={"reference_answer": r"\(m-n=\frac12\)"},
    )
    assert [request.kind for request in brief.check_requests] == [
        "substitution",
        "equivalence",
        "nonzero_division",
        "back_substitution",
    ]

    payload["check_requests"][0]["kind"] = "execute_python"
    with pytest.raises(ValidationError):
        ReferenceGroundingBrief.model_validate(
            payload,
            context={"reference_answer": r"\(m-n=\frac12\)"},
        )


def test_reference_grounding_brief_requires_unique_check_request_ids():
    payload = grounding_brief_payload()
    payload["check_requests"] = [
        payload["check_requests"][0],
        {
            **payload["check_requests"][0],
            "kind": "back_substitution",
        },
    ]

    with pytest.raises(ValidationError, match="check request ids"):
        ReferenceGroundingBrief.model_validate(
            payload,
            context={"reference_answer": r"\(m-n=\frac12\)"},
        )


@pytest.mark.parametrize(
    ("collection", "limit"),
    [
        ("reasoning_steps", 12),
        ("check_requests", 8),
        ("assumptions", 8),
        ("audit_notes", 8),
    ],
)
def test_reference_grounding_brief_rejects_oversized_collections(
    collection,
    limit,
):
    payload = grounding_brief_payload()
    source = payload[collection][0] if payload[collection] else "需要补充说明"
    payload[collection] = [source for _ in range(limit + 1)]

    with pytest.raises(ValidationError):
        ReferenceGroundingBrief.model_validate(
            payload,
            context={"reference_answer": r"\(m-n=\frac12\)"},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("task_summary", " "),
        ("target", ""),
        ("reference_conclusion", "\n"),
        ("method_name", " "),
        ("task_summary", "过" * 181),
        ("target", "x" * 161),
    ],
)
def test_reference_grounding_brief_rejects_blank_or_oversized_text(
    field,
    value,
):
    payload = grounding_brief_payload()
    payload[field] = value

    with pytest.raises(ValidationError):
        ReferenceGroundingBrief.model_validate(
            payload,
            context={"reference_answer": r"\(m-n=\frac12\)"},
        )


def test_reference_grounding_brief_forbids_extra_fields_at_every_level():
    payload = grounding_brief_payload()
    payload["run_command"] = "trusted"
    with pytest.raises(ValidationError):
        ReferenceGroundingBrief.model_validate(
            payload,
            context={"reference_answer": r"\(m-n=\frac12\)"},
        )

    payload = grounding_brief_payload()
    payload["reasoning_steps"][0]["unexpected"] = "field"
    with pytest.raises(ValidationError):
        ReferenceGroundingBrief.model_validate(
            payload,
            context={"reference_answer": r"\(m-n=\frac12\)"},
        )

    payload = grounding_brief_payload()
    payload["check_requests"][0]["tool"] = "shell"
    with pytest.raises(ValidationError):
        ReferenceGroundingBrief.model_validate(
            payload,
            context={"reference_answer": r"\(m-n=\frac12\)"},
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("substitutions", {"root": "2*n"}),
        ("substitutions", {"x1": "2*n"}),
        ("nonzero_symbols", ["number"]),
        ("nonzero_symbols", ["n1"]),
    ],
)
def test_reference_grounding_brief_requires_single_letter_symbol_keys(
    field,
    value,
):
    payload = grounding_brief_payload()
    payload["check_requests"][0][field] = value

    with pytest.raises(ValidationError):
        ReferenceGroundingBrief.model_validate(
            payload,
            context={"reference_answer": r"\(m-n=\frac12\)"},
        )


def test_reference_conclusion_agrees_with_supplied_answer_after_text_normalization():
    payload = grounding_brief_payload()
    payload["reference_conclusion"] = r" \( m-n = \frac12 \) "

    brief = ReferenceGroundingBrief.model_validate(
        payload,
        context={"reference_answer": r"$\frac12$"},
    )
    assert brief.reference_conclusion == r"\( m-n = \frac12 \)"


def test_reference_conclusion_rejects_mismatch_with_supplied_answer():
    with pytest.raises(ValidationError, match="reference_conclusion"):
        ReferenceGroundingBrief.model_validate(
            grounding_brief_payload(),
            context={"reference_answer": r"$\frac13$"},
        )


@pytest.mark.parametrize(
    "reference_conclusion",
    [
        r"\(m+n=\frac12\)",
        r"\(\text{错误目标}=\frac12\)",
        r"\(m-n=q=\frac12\)",
    ],
)
def test_reference_conclusion_rejects_wrong_lhs_or_multiple_equal_signs(
    reference_conclusion,
):
    payload = grounding_brief_payload()
    payload["reference_conclusion"] = reference_conclusion

    with pytest.raises(ValidationError, match="reference_conclusion"):
        ReferenceGroundingBrief.validate_for_reference_answer(
            payload,
            r"$\frac12$",
        )


def test_reference_conclusion_accepts_pure_rhs_through_context_helper():
    payload = grounding_brief_payload()
    payload["reference_conclusion"] = r"\(\frac12\)"

    brief = ReferenceGroundingBrief.validate_for_reference_answer(
        payload,
        r"$\frac12$",
    )

    assert brief.reference_conclusion == r"\(\frac12\)"


def test_equation_reference_answer_requires_whole_conclusion_match():
    payload = grounding_brief_payload()
    payload["reference_conclusion"] = r"\(m-n=\frac12\)"

    brief = ReferenceGroundingBrief.validate_for_reference_answer(
        payload,
        r"$m-n=\frac12$",
    )
    assert brief.reference_conclusion == r"\(m-n=\frac12\)"

    for invalid_conclusion in (r"\(\frac12\)", r"\(m+n=\frac12\)"):
        payload["reference_conclusion"] = invalid_conclusion
        with pytest.raises(ValidationError, match="reference_conclusion"):
            ReferenceGroundingBrief.validate_for_reference_answer(
                payload,
                r"$m-n=\frac12$",
            )


@pytest.mark.parametrize(
    "context",
    [
        None,
        {},
        {"reference_answer": None},
        {"reference_answer": 1},
        {"reference_answer": " \n "},
    ],
)
def test_reference_grounding_brief_requires_explicit_reference_answer_context(
    context,
):
    with pytest.raises(ValidationError, match="reference_answer context"):
        ReferenceGroundingBrief.model_validate(
            grounding_brief_payload(),
            context=context,
        )


def valid_transfer_options():
    return [
        TransferOption(
            option_id="both-roots",
            label="x = 3 或 x = 4",
            canonical_answer="x = 3 或 x = 4",
            feedback="两个根都能使原方程成立。",
        ),
        TransferOption(
            option_id="only-three",
            label="x = 3",
            canonical_answer="x = 3",
            feedback="还遗漏了另一个根。",
        ),
        TransferOption(
            option_id="only-four",
            label="x = 4",
            canonical_answer="x = 4",
            feedback="还遗漏了另一个根。",
        ),
    ]


def test_method_introduction_accepts_student_facing_complete_square_contract():
    introduction = MethodIntroduction(
        method_name="配方法",
        student_definition="把二次式整理成完全平方，再开平方求根。",
        target_form=r"\((x-a)^2=b\)",
        why_it_helps="完全平方能把二次方程转成直接可开平方的形式。",
    )

    assert introduction.method_name == "配方法"
    assert introduction.target_form == r"\((x-a)^2=b\)"


def test_method_introduction_field_budgets_keep_spoken_narration_bounded():
    introduction = MethodIntroduction(
        method_name="方" * 8,
        student_definition="定义" * 18,
        target_form="t" * 80,
        why_it_helps="作用" * 16,
    )

    assert len(introduction.method_name) == 8
    assert len(introduction.student_definition) == 36
    assert len(introduction.target_form) == 80
    assert len(introduction.why_it_helps) == 32
    assert len(introduction.spoken_narration) <= 90


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("method_name", "方" * 9),
        ("student_definition", "定义" * 18 + "多"),
        ("target_form", "t" * 81),
        ("why_it_helps", "作用" * 16 + "多"),
    ],
)
def test_method_introduction_rejects_fields_over_their_budget(
    field,
    value,
):
    payload = {
        "method_name": "配方法",
        "student_definition": "把二次式整理成完全平方。",
        "target_form": r"\((x-a)^2=b\)",
        "why_it_helps": "这样可以直接开平方求根。",
    }
    payload[field] = value

    with pytest.raises(ValidationError) as exc_info:
        MethodIntroduction.model_validate(payload)

    assert exc_info.value.errors()[0]["type"] == "string_too_long"
    assert exc_info.value.errors()[0]["loc"] == (field,)


@pytest.mark.parametrize(
    ("student_definition", "why_it_helps", "expected_narration"),
    [
        (
            "把二次式整理成完全平方",
            "这样就能直接开平方求根",
            "今天用配方法。把二次式整理成完全平方。这样就能直接开平方求根。",
        ),
        (
            "把二次式整理成完全平方。",
            "这样就能直接开平方求根！",
            "今天用配方法。把二次式整理成完全平方。这样就能直接开平方求根！",
        ),
        (
            "把二次式整理成完全平方？",
            "这样就能直接开平方求根!",
            "今天用配方法。把二次式整理成完全平方？这样就能直接开平方求根!",
        ),
    ],
)
def test_method_introduction_builds_spoken_narration_with_sentence_boundaries(
    student_definition,
    why_it_helps,
    expected_narration,
):
    introduction = MethodIntroduction(
        method_name="配方法",
        student_definition=student_definition,
        target_form=r"\((x-a)^2=b\)",
        why_it_helps=why_it_helps,
    )

    narration = introduction.spoken_narration

    assert narration == expected_narration
    assert introduction.target_form not in narration
    assert r"\(" not in narration


def test_interaction_option_keeps_legacy_defaults_and_accepts_diagnostic_feedback():
    legacy_option = InteractionOption(option_id="factor", label="因式分解")
    diagnostic_option = InteractionOption(
        option_id="square",
        label="配方法",
        feedback="先观察这个式子能否直接分解。",
        feedback_audio_url="https://audio.example/diagnostic.mp3",
    )

    assert legacy_option.feedback is None
    assert legacy_option.feedback_audio_url is None
    assert diagnostic_option.feedback == "先观察这个式子能否直接分解。"
    assert diagnostic_option.feedback_audio_url == "https://audio.example/diagnostic.mp3"


def test_transfer_item_accepts_three_diagnostic_options():
    item = TransferItem(
        problem_text="用因式分解法解方程：x^2-7x+12=0",
        expected_answer="x=3 或 x=4",
        method_signal="寻找乘积为 12、和为 -7 的两个数。",
        options=valid_transfer_options(),
        correct_option_id="both-roots",
    )

    assert item.correct_option_id == "both-roots"
    assert len(item.options) == 3


def test_generated_transfer_option_requires_student_visible_label():
    payload = {
        "option_id": "both-roots",
        "canonical_answer": "x=3 或 x=4",
        "feedback": "两个根都能使原方程成立。",
    }

    with pytest.raises(ValidationError):
        GeneratedTransferOption.model_validate(payload)


def test_transfer_item_does_not_treat_untrusted_derived_labels_as_identity():
    options = valid_transfer_options()
    for option in options:
        option.label = "model placeholder"

    item = TransferItem(
        problem_text="用因式分解法解方程：x^2-7x+12=0",
        expected_answer="x=3 或 x=4",
        method_signal="寻找乘积为 12、和为 -7 的两个数。",
        options=options,
        correct_option_id="both-roots",
    )

    assert len(item.options) == 3


def test_transfer_item_keeps_legacy_empty_options_compatible():
    item = TransferItem(
        problem_text="解方程 x + 2 = 0",
        expected_answer="x = -2",
        method_signal="等式两边同时减二。",
    )

    assert item.options == []
    assert item.correct_option_id is None


def test_transfer_item_rejects_invalid_diagnostic_option_contract():
    def invalid_item(options, correct_option_id):
        return TransferItem(
            problem_text="用因式分解法解方程：x^2-7x+12=0",
            expected_answer="x=3 或 x=4",
            method_signal="寻找乘积为 12、和为 -7 的两个数。",
            options=options,
            correct_option_id=correct_option_id,
        )

    duplicate_ids = valid_transfer_options()
    duplicate_ids[1] = TransferOption(
        option_id="both-roots",
        label="x = 3",
        canonical_answer="x = 3",
        feedback="还遗漏了另一个根。",
    )
    for options, correct_option_id in [
        (duplicate_ids, "both-roots"),
        (valid_transfer_options()[:2], "both-roots"),
        (valid_transfer_options(), None),
        (valid_transfer_options(), "unknown"),
    ]:
        with pytest.raises(ValidationError):
            invalid_item(options, correct_option_id)

    with pytest.raises(ValidationError):
        TransferItem(
            problem_text="用因式分解法解方程：x^2-7x+12=0",
            expected_answer="x=3 或 x=4",
            method_signal="寻找乘积为 12、和为 -7 的两个数。",
            options=valid_transfer_options(),
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


def test_problem_input_preserves_multiline_reference_solution():
    problem = ProblemInput(
        problem_text="解方程 x^2-6x+5=0",
        reference_answer="x=1 或 x=5",
        reference_solution_text=(
            "\n  解：移项，得 x^2-6x=-5。\n\n"
            "两边同时加 9，得 (x-3)^2=4。\n"
            "所以 x=1 或 x=5。  \n"
        ),
    )

    assert problem.reference_solution_text == (
        "解：移项，得 x^2-6x=-5。\n\n"
        "两边同时加 9，得 (x-3)^2=4。\n"
        "所以 x=1 或 x=5。"
    )


def test_problem_input_allows_missing_reference_solution():
    problem = ProblemInput(
        problem_text="解方程 x=1",
        reference_answer="x=1",
    )

    assert problem.reference_solution_text is None


def test_problem_input_rejects_oversized_reference_solution():
    with pytest.raises(ValidationError):
        ProblemInput(
            problem_text="解方程 x=1",
            reference_answer="x=1",
            reference_solution_text="甲" * 12001,
        )


def test_reference_material_audit_accepts_approved_review():
    audit = ReferenceMaterialAudit(
        status="approved",
        claimed_answer="x=1 或 x=5",
        method_summary="配方法",
        key_steps=[
            {
                "purpose": "配方",
                "operation": "complete_the_square",
                "operands": ["9"],
                "state_before": ["x^2-6x=-5"],
                "state_after": ["(x-3)^2=4"],
                "reason": "两边同时加 9 并构造完全平方。",
            }
        ],
        teaching_assets=["先让学生观察一次项系数的一半。"],
        warnings=[],
        blocking_issues=[],
        evidence=["所以 x=1 或 x=5。"],
    )

    assert audit.status == "approved"
    assert audit.key_steps[0].operation == "complete_the_square"


def test_rejected_reference_material_audit_requires_issue_and_evidence():
    with pytest.raises(ValidationError):
        ReferenceMaterialAudit(
            status="rejected",
            claimed_answer="x=1",
            method_summary=None,
            key_steps=[],
            teaching_assets=[],
            warnings=[],
            blocking_issues=[],
            evidence=[],
        )


def test_approved_reference_material_audit_rejects_blocking_issues():
    with pytest.raises(ValidationError):
        ReferenceMaterialAudit(
            status="approved",
            claimed_answer=None,
            method_summary=None,
            key_steps=[],
            teaching_assets=[],
            warnings=[],
            blocking_issues=["步骤改变了解集。"],
            evidence=["由 x=1 得 x=2。"],
        )


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


def sync_visual_action_payload(**overrides):
    return {
        "surface": "board",
        "type": "focus",
        "target": "solution-line-001",
        **overrides,
    }


def test_narrative_sync_cue_accepts_voice_only_cue():
    cue = NarrativeSyncCue(
        cue_id="cue-voice-only",
        spoken_text="先观察这个式子的结构。",
    )

    assert cue.lead_actions == []
    assert cue.start_actions == []
    assert cue.end_actions == []


def test_sync_visual_action_accepts_problem_emphasis_with_semantic_target():
    action = SyncVisualAction(
        surface="problem",
        type="emphasize",
        target="problem-math-001",
        emphasis_style="underline",
        persistence="trace",
    )

    assert action.target == "problem-math-001"
    assert action.emphasis_style == "underline"
    assert action.persistence == "trace"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("target", "problem-math-001[data-secret]"),
        ("target", "problem-math-001:nth-child(2)"),
        ("emphasis_style", "background:url(javascript:alert(1))"),
        ("emphasis_style", "[[red]]"),
    ],
)
def test_sync_visual_action_rejects_selectors_and_inline_styles(
    field,
    value,
):
    payload = {
        "surface": "problem",
        "type": "emphasize",
        "target": "problem-math-001",
        "emphasis_style": "underline",
        "persistence": "trace",
    }
    payload[field] = value

    with pytest.raises(ValidationError):
        SyncVisualAction.model_validate(payload)


def test_sync_visual_action_accepts_math_comparison_content():
    action = SyncVisualAction(
        surface="board",
        type="write",
        target="solution-line-001",
        content="a < b > c",
    )

    assert action.content == "a < b > c"


@pytest.mark.parametrize(
    "payload",
    [
        sync_visual_action_payload(
            surface="problem",
            type="write",
            content="x = 1",
        ),
        sync_visual_action_payload(
            surface="problem",
            type="transform",
            content="x = 1",
        ),
        sync_visual_action_payload(
            surface="problem",
            type="annotate",
            annotation="underline",
        ),
    ],
)
def test_sync_visual_action_rejects_problem_surface_mutation(payload):
    with pytest.raises(ValidationError):
        SyncVisualAction.model_validate(payload)


@pytest.mark.parametrize("action_type", ("write", "transform"))
def test_sync_visual_action_requires_content_for_mutation(action_type):
    with pytest.raises(ValidationError):
        SyncVisualAction(
            surface="board",
            type=action_type,
            target="solution-line-001",
        )


@pytest.mark.parametrize(
    "payload",
    [
        sync_visual_action_payload(type="emphasize"),
        sync_visual_action_payload(persistence="transient"),
        sync_visual_action_payload(content="x = 1"),
        sync_visual_action_payload(
            type="write",
            content="x = 1",
            source="source-line-001",
        ),
        sync_visual_action_payload(type="annotate", annotation="label"),
        sync_visual_action_payload(type="annotate", annotation="arrow"),
        sync_visual_action_payload(
            type="annotate",
            annotation="underline",
            relation_target="source-line-001",
        ),
    ],
)
def test_sync_visual_action_rejects_irrelevant_or_incomplete_fields(
    payload,
):
    with pytest.raises(ValidationError):
        SyncVisualAction.model_validate(payload)


@pytest.mark.parametrize(
    ("action_field", "action_count"),
    [
        ("lead_actions", 7),
        ("start_actions", 9),
        ("end_actions", 7),
    ],
)
def test_narrative_sync_cue_rejects_oversized_action_lists(
    action_field,
    action_count,
):
    payload = {
        "cue_id": "cue-oversized",
        "spoken_text": "观察这一行。",
        action_field: [
            sync_visual_action_payload(
                target=f"solution-line-{index}",
            )
            for index in range(action_count)
        ],
    }

    with pytest.raises(ValidationError):
        NarrativeSyncCue.model_validate(payload)


@pytest.mark.parametrize("spoken_text", ("", "讲" * 91))
def test_narrative_sync_cue_rejects_invalid_spoken_text(spoken_text):
    with pytest.raises(ValidationError):
        NarrativeSyncCue(
            cue_id="cue-invalid-spoken-text",
            spoken_text=spoken_text,
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
        "method_introduction": {
            "method_name": "等式性质",
            "student_definition": "等式两边做相同运算，等式仍然成立。",
            "target_form": "x = a",
            "why_it_helps": "可以逐步把未知数单独留在等号一边。",
        },
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


def test_choice_interaction_requires_executable_unique_options():
    base = {
        "interaction_id": "choose-method",
        "kind": "choice",
        "prompt": "选择正确方法。",
        "expected_answer": "factor",
    }

    with pytest.raises(ValidationError):
        Interaction.model_validate(base)
    with pytest.raises(ValidationError):
        Interaction.model_validate(
            {
                **base,
                "options": [
                    {"option_id": "factor", "label": "因式分解"},
                    {"option_id": "factor", "label": "重复选项"},
                ],
            }
        )
    with pytest.raises(ValidationError):
        Interaction.model_validate(
            {
                **base,
                "expected_answer": "missing",
                "options": [
                    {"option_id": "factor", "label": "因式分解"},
                ],
            }
        )

    interaction = Interaction.model_validate(
        {
            **base,
            "options": [
                {"option_id": "factor", "label": "因式分解"},
                {"option_id": "formula", "label": "求根公式"},
            ],
        }
    )

    assert interaction.expected_answer == "factor"


def test_non_choice_interaction_rejects_options():
    with pytest.raises(ValidationError):
        Interaction(
            interaction_id="write-expression",
            kind="expression",
            prompt="写出因式分解结果。",
            expected_answer="(x-2)(x-3)",
            options=[
                InteractionOption(
                    option_id="unused",
                    label="不应出现",
                )
            ],
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
                annotation="underline",
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
    assert moment.board_actions[1].annotation == "underline"
    assert moment.interaction is not None
    assert moment.interaction.kind == "expression"

    with pytest.raises(
        ValueError,
        match=(
            "legacy action is not losslessly representable as "
            "SyncVisualAction"
        ),
    ):
        LessonMoment(
            purpose="拒绝有损旧标注",
            narration="观察这一步。",
            board_actions=[
                BoardAction(
                    type="annotate",
                    target="factor_pair",
                    annotation="circle",
                )
            ],
        )


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
        options=[
            InteractionOption(option_id="both", label="两个根都满足"),
        ],
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
    assert beat.sync_cues == []
    assert interaction.hint_audio_urls == [
        "/audio/hints/check-one.mp3",
        "/audio/hints/check-two.mp3",
    ]
    assert interaction.correct_audio_url == "/audio/feedback/correct.mp3"


def test_legacy_runtime_lesson_defaults_new_runtime_collections_to_empty():
    lesson = RuntimeLesson(
        lesson_id="lesson-legacy-runtime",
        problem=ProblemInput(
            problem_text="解方程 x + 1 = 0",
            reference_answer="x = -1",
        ),
        title="移项解方程",
        learning_goal="能用等式性质完成移项。",
        beats=[
            RuntimeBeat(
                beat_id="beat-legacy",
                purpose="保持旧课程兼容",
                narration="等式两边同时减一。",
                board_actions=[],
                layer="base",
                audio_url="/audio/lesson-legacy-runtime/beat-legacy.mp3",
            )
        ],
        summary="等式两边做相同运算。",
        transfer_item=TransferItem(
            problem_text="解方程 x + 2 = 0",
            expected_answer="x = -2",
            method_signal="等式两边同时减二。",
        ),
        validation_report={},
    )

    assert lesson.beats[0].sync_cues == []
    assert lesson.beats[0].audio_url.endswith("beat-legacy.mp3")
    assert lesson.problem_focus_targets == []


def test_runtime_lesson_rejects_duplicate_cue_ids_across_beats():
    def runtime_beat(beat_id, spoken_text):
        return RuntimeBeat(
            beat_id=beat_id,
            purpose="讲解一步",
            narration=spoken_text,
            board_actions=[],
            layer="base",
            sync_cues=[
                {
                    "cue_id": "duplicate-runtime-cue",
                    "spoken_text": spoken_text,
                }
            ],
        )

    with pytest.raises(ValidationError, match="runtime cue ids"):
        RuntimeLesson(
            lesson_id="lesson-duplicate-runtime-cues",
            problem=ProblemInput(
                problem_text="解方程 x + 1 = 0",
                reference_answer="x = -1",
            ),
            title="移项解方程",
            learning_goal="能用等式性质完成移项。",
            beats=[
                runtime_beat("beat-one", "先观察等号两边。"),
                runtime_beat("beat-two", "再进行相同运算。"),
            ],
            summary="等式两边做相同运算。",
            transfer_item=TransferItem(
                problem_text="解方程 x + 2 = 0",
                expected_answer="x = -2",
                method_signal="等式两边同时减二。",
            ),
            validation_report={},
        )


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
        options=[
            TransferOption(
                option_id="both-roots",
                label="x = 3 或 x = 4",
                canonical_answer="x = 3 或 x = 4",
                feedback="两个根都能使原方程成立。",
            ),
            TransferOption(
                option_id="only-three",
                label="x = 3",
                canonical_answer="x = 3",
                feedback="还遗漏了另一个根。",
            ),
            TransferOption(
                option_id="only-four",
                label="x = 4",
                canonical_answer="x = 4",
                feedback="还遗漏了另一个根。",
            ),
        ],
        correct_option_id="both-roots",
    )
    draft = LessonDraft(
        title="用因式分解法解一元二次方程",
        learning_goal="能根据系数找到因式并求根。",
        opening="观察常数项和一次项系数。",
        method_rationale="整式容易分解，因式分解法步骤最短。",
        method_introduction=MethodIntroduction(
            method_name="因式分解法",
            student_definition="把二次式写成两个一次因式的乘积，再分别令每个因式为零。",
            target_form="(x-a)(x-b)=0",
            why_it_helps="零乘积性质把二次方程拆成两个一次方程。",
        ),
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
    assert runtime.transfer_item.correct_option_id == "both-roots"
    assert len(runtime.transfer_item.options) == 3
    assert job.status == "completed"
