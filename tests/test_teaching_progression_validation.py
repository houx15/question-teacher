import pytest

from app.preparation_models import ReasoningTrajectory, TeachingProgression
from app.schemas import ProblemFocusTarget
from app.teaching_progression_validation import (
    TeachingProgressionValidationError,
    derive_misconception_vocabulary,
    validate_teaching_progression,
)


def trajectory_payload():
    return {
        "trajectory_type": "hybrid",
        "lesson_purpose": "理解代入与约分的依赖顺序",
        "episodes": [
            {
                "episode_id": "episode-with-a-provider-controlled-long-id-1",
                "sequence_index": 0,
                "mode": "plan",
                "source_step_ids": ["trace-1"],
                "learner_state_before": "知道根条件",
                "attention_targets": ["x=2n"],
                "thinking_question": "根条件怎样变成方程信息？",
                "decision": "把根代入方程",
                "decision_reason": "根必须使原方程成立",
                "mathematical_action": "代入x=2n",
                "action_justification": "根的定义",
                "result": "4n^2-4mn+2n=0",
                "result_meaning": "建立m与n的关系",
                "transition_reason": "已经得到含公因式的关系，下一步判断能否约分。",
                "must_teach": [
                    {
                        "must_teach_id": "must-1",
                        "content": "根必须使原方程成立",
                        "why_it_matters": "这是代入的依据",
                    },
                    {
                        "must_teach_id": "must-2",
                        "content": "代入要替换所有x",
                        "why_it_matters": "避免漏项",
                    },
                ],
                "likely_misconceptions": ["只代入一个x", "不知道根为何能代入"],
            },
            {
                "episode_id": "episode-2",
                "sequence_index": 1,
                "mode": "execute",
                "source_step_ids": ["trace-2"],
                "learner_state_before": "得到可因式分解的关系",
                "attention_targets": ["n!=0"],
                "thinking_question": "什么条件下可以约去2n？",
                "decision": "先使用n!=0再约分",
                "decision_reason": "只有非零因式可以约去",
                "mathematical_action": "两边同除2n",
                "action_justification": "n!=0",
                "result": "2n-2m+1=0",
                "result_meaning": "可以整理出目标",
                "transition_reason": "约分后已只剩目标关系，可以解释结论。",
                "must_teach": [
                    {
                        "must_teach_id": "must-3",
                        "content": "约分前确认因式非零",
                        "why_it_matters": "保证变形合法",
                    }
                ],
                "likely_misconceptions": ["未确认非零就约分"],
            },
        ],
        "method_summary": "代入后使用非零条件约分",
        "error_summary": "不能跳过非零条件",
    }


def progression_payload():
    return {
        "steps": [
            {
                "step_id": "progression-1",
                "sequence_index": 0,
                "episode_ids": ["episode-with-a-provider-controlled-long-id-1"],
                "phase": "construct",
                "student_problem": "根条件怎样变成方程信息？",
                "why_now": "先建立m与n的关系，才能判断后续约分。",
                "evidence_target_ids": ["target-root"],
                "guiding_questions": ["根必须满足什么？"],
                "knowledge_anchor": "根的定义",
                "checkpoint": {
                    "diagnostic_goal": "识别代入遗漏",
                    "misconception_ids": ["misconception-001-001"],
                },
                "reveal": "把x=2n代入原方程",
                "math_action": "展开并合并同类项",
                "directory_question": "如何使用根条件？",
                "directory_label": "用根条件建立参数关系",
                "board_summary": ["代入根后得到：4n^2-4mn+2n=0"],
                "error_tip": "注意替换所有x",
                "transition_question": "这个式子有什么公因式？",
                "must_teach_refs": ["must-1", "must-2"],
            },
            {
                "step_id": "progression-2",
                "sequence_index": 1,
                "episode_ids": ["episode-2"],
                "phase": "execute",
                "student_problem": "什么条件下可以约去2n？",
                "why_now": "当前式子出现2n，必须先核对非零条件再约分。",
                "evidence_target_ids": ["target-nonzero"],
                "guiding_questions": ["n是否可能为0？"],
                "knowledge_anchor": "等式同除非零数",
                "checkpoint": {
                    "diagnostic_goal": "识别非法约分",
                    "misconception_ids": ["misconception-002-001"],
                },
                "reveal": "由n!=0可同除2n",
                "math_action": "约分并整理",
                "directory_question": "什么时候能约分？",
                "directory_label": "核对非零条件后约分",
                "board_summary": ["由n!=0，可约去2n得：2n-2m+1=0"],
                "error_tip": "不能把未知数当作必然非零",
                "transition_question": "怎样整理出m-n？",
                "must_teach_refs": ["must-3"],
            },
        ]
    }


def models(trajectory_value=None, progression_value=None):
    return (
        ReasoningTrajectory.model_validate(trajectory_value or trajectory_payload()),
        TeachingProgression.model_validate(progression_value or progression_payload()),
    )


def targets():
    return [
        ProblemFocusTarget(target_id="target-root", math_text="x=2n", ordinal=1),
        ProblemFocusTarget(target_id="target-nonzero", math_text="n!=0", ordinal=2),
    ]


def assert_code(code, progression_value=None, trajectory_value=None, target_value=None):
    trajectory, progression = models(trajectory_value, progression_value)
    with pytest.raises(TeachingProgressionValidationError) as captured:
        validate_teaching_progression(
            progression,
            trajectory,
            targets() if target_value is None else target_value,
        )
    assert captured.value.code == code
    assert str(captured.value) == "%s:%s" % (
        captured.value.code,
        captured.value.artifact_id,
    )


def test_valid_progression_and_explanatory_board_summaries_pass():
    trajectory, progression = models()
    validate_teaching_progression(progression, trajectory, targets())


def test_misconception_vocabulary_is_stable_bounded_and_ordered():
    trajectory, _ = models()
    assert derive_misconception_vocabulary(trajectory) == [
        {
            "misconception_id": "misconception-001-001",
            "episode_id": "episode-with-a-provider-controlled-long-id-1",
            "description": "只代入一个x",
        },
        {
            "misconception_id": "misconception-001-002",
            "episode_id": "episode-with-a-provider-controlled-long-id-1",
            "description": "不知道根为何能代入",
        },
        {
            "misconception_id": "misconception-002-001",
            "episode_id": "episode-2",
            "description": "未确认非零就约分",
        },
    ]
    assert max(
        len(item["misconception_id"])
        for item in derive_misconception_vocabulary(trajectory)
    ) == len("misconception-001-001")


def test_empty_misconception_vocabulary_requires_empty_checkpoint_ids():
    trajectory_value = trajectory_payload()
    for episode in trajectory_value["episodes"]:
        episode["likely_misconceptions"] = []
    progression_value = progression_payload()
    progression_value["steps"][0]["checkpoint"]["misconception_ids"] = []
    progression_value["steps"][1]["checkpoint"]["misconception_ids"] = []
    trajectory, progression = models(trajectory_value, progression_value)
    assert derive_misconception_vocabulary(trajectory) == []
    validate_teaching_progression(progression, trajectory, targets())


@pytest.mark.parametrize("mutation", ("missing", "duplicate", "reversed"))
def test_episode_coverage_must_exactly_equal_trajectory_order(mutation):
    payload = progression_payload()
    if mutation == "missing":
        payload["steps"] = payload["steps"][:1]
    elif mutation == "duplicate":
        payload["steps"][1]["episode_ids"] = [
            "episode-with-a-provider-controlled-long-id-1"
        ]
    else:
        payload["steps"][0]["episode_ids"], payload["steps"][1]["episode_ids"] = (
            payload["steps"][1]["episode_ids"],
            payload["steps"][0]["episode_ids"],
        )
    assert_code("progression_episode_coverage_invalid", payload)


def test_episode_order_inside_one_step_cannot_reverse_trajectory():
    payload = progression_payload()
    payload["steps"] = [payload["steps"][0]]
    payload["steps"][0]["episode_ids"] = [
        "episode-2",
        "episode-with-a-provider-controlled-long-id-1",
    ]
    payload["steps"][0]["must_teach_refs"] = ["must-1", "must-2", "must-3"]
    assert_code("progression_episode_coverage_invalid", payload)


@pytest.mark.parametrize("mutation", ("missing", "duplicate"))
def test_must_teach_items_are_covered_exactly_once(mutation):
    payload = progression_payload()
    if mutation == "missing":
        payload["steps"][0]["must_teach_refs"] = ["must-1"]
    else:
        payload["steps"][0]["must_teach_refs"] = ["must-1", "must-2", "must-1"]
    assert_code("progression_must_teach_coverage_invalid", payload)


@pytest.mark.parametrize("reference", ("must-unknown", "must-3"))
def test_must_teach_reference_must_exist_and_belong_to_step_episode(reference):
    payload = progression_payload()
    payload["steps"][0]["must_teach_refs"][0] = reference
    assert_code("progression_must_teach_ref_invalid", payload)


def test_evidence_target_must_be_allowlisted():
    payload = progression_payload()
    payload["steps"][0]["evidence_target_ids"] = ["unknown-target"]
    assert_code("progression_evidence_target_invalid", payload)


def test_evidence_targets_cannot_repeat_within_a_step():
    payload = progression_payload()
    payload["steps"][0]["evidence_target_ids"] = ["target-root", "target-root"]
    assert_code("progression_evidence_target_duplicate", payload)


def test_evidence_targets_cannot_omit_all_authoritative_targets():
    payload = progression_payload()
    for step in payload["steps"]:
        step["evidence_target_ids"] = []
    assert_code("progression_evidence_target_coverage_invalid", payload)


def test_evidence_targets_cannot_partially_cover_authoritative_targets():
    payload = progression_payload()
    payload["steps"][1]["evidence_target_ids"] = []
    assert_code("progression_evidence_target_coverage_invalid", payload)


def test_evidence_targets_exactly_cover_authoritative_set_across_steps():
    trajectory, progression = models()
    validate_teaching_progression(progression, trajectory, targets())


def test_empty_authoritative_target_set_requires_empty_progression_evidence():
    payload = progression_payload()
    for step in payload["steps"]:
        step["evidence_target_ids"] = []
    trajectory, progression = models(progression_value=payload)
    validate_teaching_progression(progression, trajectory, [])


def test_directory_labels_are_unique():
    payload = progression_payload()
    payload["steps"][1]["directory_label"] = payload["steps"][0]["directory_label"]
    assert_code("progression_directory_label_duplicate", payload)


@pytest.mark.parametrize(
    "generic_why_now",
    (
        "然后计算",
        "然后进行计算",
        "然后计算一下",
        "接下来计算",
        "接着进行计算",
        "继续进行计算",
        "开始进行运算",
        "下一步整理",
        "再来化简一下",
        "之后继续处理",
        "接下来进行运算",
        "然后算一下",
    ),
)
def test_generic_why_now_is_rejected(generic_why_now):
    payload = progression_payload()
    payload["steps"][0]["why_now"] = generic_why_now
    assert_code("progression_why_not_explanatory", payload)


def test_mutated_empty_why_now_is_rejected_total_safely():
    trajectory, progression = models()
    object.__setattr__(progression.steps[0], "why_now", "")

    with pytest.raises(TeachingProgressionValidationError) as captured:
        validate_teaching_progression(progression, trajectory, targets())

    assert captured.value.code == "progression_structure_invalid"
    assert captured.value.artifact_id == "teaching_progression"


def test_causal_why_now_remains_explanatory():
    payload = progression_payload()
    payload["steps"][0]["why_now"] = (
        "因为根必须使原方程成立，此时代入才能建立m与n的关系。"
    )
    trajectory, progression = models(progression_value=payload)
    validate_teaching_progression(progression, trajectory, targets())


def test_goal_driven_why_now_remains_explanatory():
    payload = progression_payload()
    payload["steps"][0]["why_now"] = (
        "为了把含n的式子变成目标关系，下一步整理等式。"
    )
    trajectory, progression = models(progression_value=payload)
    validate_teaching_progression(progression, trajectory, targets())


def test_board_summary_cannot_only_repeat_the_current_episode_result():
    payload = progression_payload()
    payload["steps"][0]["board_summary"] = ["4n^2-4mn+2n=0"]
    assert_code("progression_board_summary_not_explanatory", payload)


@pytest.mark.parametrize(
    "summaries",
    (
        ["4n^2-4mn+2n=0。"],
        ["4n^2-4mn+2n=0!"],
        ["4n^2-4mn+2n=0。", "4n^2-4mn+2n=0！"],
    ),
)
def test_board_summary_punctuation_and_duplicates_cannot_bypass_result_repeat(
    summaries,
):
    payload = progression_payload()
    payload["steps"][0]["board_summary"] = summaries
    assert_code("progression_board_summary_not_explanatory", payload)


@pytest.mark.parametrize(
    "summary",
    (
        "结果：4n^2-4mn+2n=0。",
        "得到: 4n^2-4mn+2n=0!",
        "当前推理得到：4n^2-4mn+2n=0",
        "（1）4n^2-4mn+2n=0",
        "(1) 结论：4n^2-4mn+2n=0",
    ),
)
def test_board_summary_decorative_wrappers_cannot_bypass_result_repeat(summary):
    payload = progression_payload()
    payload["steps"][0]["board_summary"] = [summary]
    assert_code("progression_board_summary_not_explanatory", payload)


def test_board_summary_rejects_punctuation_only_content():
    payload = progression_payload()
    payload["steps"][0]["board_summary"] = ["【……》“”"]
    assert_code("progression_board_summary_not_explanatory", payload)


def test_multi_episode_board_summaries_cannot_only_repeat_episode_results():
    payload = progression_payload()
    payload["steps"] = [payload["steps"][0]]
    payload["steps"][0]["episode_ids"] = [
        "episode-with-a-provider-controlled-long-id-1",
        "episode-2",
    ]
    payload["steps"][0]["must_teach_refs"] = ["must-1", "must-2", "must-3"]
    payload["steps"][0]["evidence_target_ids"] = [
        "target-root",
        "target-nonzero",
    ]
    payload["steps"][0]["checkpoint"]["misconception_ids"] = [
        "misconception-001-001",
        "misconception-002-001",
    ]
    payload["steps"][0]["board_summary"] = [
        "结果：4n^2-4mn+2n=0",
        "（2）2n-2m+1=0。",
    ]
    assert_code("progression_board_summary_not_explanatory", payload)


@pytest.mark.parametrize(
    "summary",
    (
        "因为x=2n是根，所以代入得：4n^2-4mn+2n=0",
        "x=2n是根 → 4n^2-4mn+2n=0",
    ),
)
def test_board_summary_with_an_explanatory_relation_remains_valid(summary):
    payload = progression_payload()
    payload["steps"][0]["board_summary"] = [summary]
    trajectory, progression = models(progression_value=payload)
    validate_teaching_progression(progression, trajectory, targets())


def test_board_summary_rejects_invalid_display_content():
    payload = progression_payload()
    payload["steps"][0]["board_summary"] = [r"\(4n^2-4mn+2n=0"]
    assert_code("progression_board_content_invalid", payload)


def test_evidence_target_cannot_repeat_across_progression_steps():
    payload = progression_payload()
    payload["steps"][1]["evidence_target_ids"] = ["target-root"]
    assert_code("progression_evidence_target_duplicate", payload)


@pytest.mark.parametrize(
    ("ids", "code"),
    (
        (["misconception-999-999"], "progression_misconception_ref_invalid"),
        (
            ["misconception-001-001", "misconception-001-001"],
            "progression_misconception_ref_duplicate",
        ),
        (["misconception-002-001"], "progression_misconception_ref_invalid"),
    ),
)
def test_checkpoint_misconceptions_are_known_unique_and_owned_by_step(ids, code):
    payload = progression_payload()
    payload["steps"][0]["checkpoint"]["misconception_ids"] = ids
    assert_code(code, payload)


def test_validator_requires_exact_models_and_bounded_target_container():
    trajectory, progression = models()
    with pytest.raises(TypeError, match="exact TeachingProgression"):
        validate_teaching_progression(object(), trajectory, targets())
    with pytest.raises(TypeError, match="exact ReasoningTrajectory"):
        validate_teaching_progression(progression, object(), targets())
    with pytest.raises(TypeError, match="problem_targets"):
        validate_teaching_progression(progression, trajectory, object())


@pytest.mark.parametrize(
    ("field", "mutated_value"),
    (
        ("board_summary", None),
        ("evidence_target_ids", [[]]),
        ("must_teach_refs", [[]]),
    ),
)
def test_mutated_progression_structure_fails_with_stable_validation_error(
    field,
    mutated_value,
):
    trajectory, progression = models()
    object.__setattr__(progression.steps[0], field, mutated_value)

    with pytest.raises(TeachingProgressionValidationError) as captured:
        validate_teaching_progression(progression, trajectory, targets())

    assert captured.value.code == "progression_structure_invalid"
    assert captured.value.artifact_id == "teaching_progression"
    assert str(captured.value) == (
        "progression_structure_invalid:teaching_progression"
    )


def test_error_message_is_stable_and_does_not_echo_generated_detail():
    payload = progression_payload()
    payload["steps"][0]["why_now"] = "然后计算"
    trajectory, progression = models(progression_value=payload)
    with pytest.raises(TeachingProgressionValidationError) as captured:
        validate_teaching_progression(progression, trajectory, targets())
    assert str(captured.value) == "progression_why_not_explanatory:progression-1"
    assert not hasattr(captured.value, "detail")
