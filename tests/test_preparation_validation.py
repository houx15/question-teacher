import copy

import pytest
from pydantic import ValidationError

from app.math_content import contains_internal_control_syntax
from app.pedagogy_rubric import PEDAGOGY_RUBRIC_VERSION
from app.preparation_models import (
    InteractionPlan,
    LessonReviewDecision,
    PerformanceScore,
    PreparedLesson,
    ReasoningTrajectory,
    SimulationReport,
    SolutionTrace,
    TeachingScript,
)
from app.preparation_validation import (
    PreparationValidationError,
    blocking_signature,
    validate_interaction_plan,
    validate_performance_score,
    validate_prepared_lesson,
    validate_reasoning_trajectory,
    validate_review_decision,
    validate_simulation_report,
    validate_solution_trace,
    validate_teaching_script,
)
from app.schemas import ProblemFocusTarget, ReferenceGroundingBrief
from app.teaching_route import freeze_grounded_route


STEP_IDS = (
    "is-root",
    "substitute-root",
    "target-relation",
    "expand",
    "common-factor",
    "use-nonzero",
    "return-target",
)
STATES = (
    "x=2n is a root",
    "x=2n",
    "target m-n",
    "4n^2-4mn+2n=0",
    "2n(2n-2m+1)=0",
    "2n-2m+1=0",
    r"m-n=\dfrac{1}{2}",
)


def route(final_conclusion="m-n=1/2"):
    brief = ReferenceGroundingBrief.validate_for_reference_answer(
        {
            "task_summary": "由参数根求m-n",
            "target": "m-n",
            "assumptions": ["n!=0", "x=2n是原方程的根"],
            "reference_conclusion": final_conclusion,
            "method_name": "代入法",
            "reasoning_steps": [
                {
                    "step_id": step_id,
                    "statement_before": "题目条件" if index == 0 else STATES[index - 1],
                    "operation_explanation": "保留这一步的数学依赖",
                    "statement_after": (
                        final_conclusion
                        if index == len(STEP_IDS) - 1
                        else state
                    ),
                }
                for index, (step_id, state) in enumerate(zip(STEP_IDS, STATES))
            ],
            "check_requests": [],
            "audit_notes": [],
        },
        final_conclusion,
    )
    return freeze_grounded_route(brief, [])


def trace_payload():
    return {
        "task_target": "求关系m-n",
        "reference_conclusion": r"\(m-n=\frac{1}{2}\)",
        "assumptions": [
            {
                "assumption_id": "assumption-nonzero",
                "content": "n!=0",
                "source_anchor": {
                    "source_kind": "problem",
                    "source_id": "problem",
                    "excerpt": "n不等于0",
                },
            }
        ],
        "source_steps": [
            {
                "source_step_id": step_id,
                "source_anchor": {
                    "source_kind": "verified_route",
                    "source_id": step_id,
                    "excerpt": "冻结路线步骤",
                },
                "state_before": "题目条件" if index == 0 else STATES[index - 1],
                "mathematical_action": (
                    "识别根条件" if step_id == "is-root"
                    else "代入x=2n" if step_id == "substitute-root"
                    else "锁定目标关系m-n" if step_id == "target-relation"
                    else "展开" if step_id == "expand"
                    else "观察公因式2n" if step_id == "common-factor"
                    else "利用n!=0后除以2n" if step_id == "use-nonzero"
                    else "回到目标m-n"
                ),
                "justification": "来自题目与冻结路线",
                "state_after": state,
                "new_information": "得到下一步所需信息",
                "assumption_ids_used": (
                    ["assumption-nonzero"] if step_id == "use-nonzero" else []
                ),
                "omitted_reasoning": [],
                "evidence_status": "verified_route",
            }
            for index, (step_id, state) in enumerate(zip(STEP_IDS, STATES))
        ],
        "audit_notes": [],
    }


def episode_payload(index, step_id):
    episode_id = "episode-%d" % (index + 1)
    return {
        "episode_id": episode_id,
        "sequence_index": index,
        "mode": ("understand", "plan", "execute", "monitor", "execute", "revise", "reflect")[index],
        "source_step_ids": [step_id],
        "learner_state_before": "知道前一步",
        "attention_targets": [STATES[index]],
        "thinking_question": "此刻下一步依赖什么？",
        "decision": "执行当前数学动作",
        "decision_reason": "保持数学依赖顺序",
        "mathematical_action": "处理%s" % step_id,
        "action_justification": "冻结路线支持",
        "result": STATES[index],
        "result_meaning": "为下一步提供依据",
        "transition_reason": "继续处理目标关系",
        "must_teach": [
            {
                "must_teach_id": "must-%d" % (index + 1),
                "content": "解释%s" % step_id,
                "why_it_matters": "学生需要理解依赖",
            }
        ],
        "likely_misconceptions": [],
        "interaction_intent": None,
        "visual_intent": "逐步写出关系",
    }


def trajectory_payload():
    return {
        "trajectory_type": "hybrid",
        "lesson_purpose": "理解参数根如何导出目标关系",
        "episodes": [episode_payload(index, step_id) for index, step_id in enumerate(STEP_IDS)],
        "method_summary": "识别根、代入、整理、使用非零条件并回到目标",
        "error_summary": "不能跳过n!=0就除法",
    }


def clause_payload(index, clause_id=None, episode_id=None, math_reference=None):
    return {
        "clause_id": clause_id or "clause-%d" % (index + 1),
        "episode_id": episode_id or "episode-%d" % (index + 1),
        "pedagogical_function": "explain",
        "spoken_text": "我们解释当前这一步为什么成立。",
        "math_references": [math_reference or STATES[index]],
        "learner_gain": "理解当前依赖",
        "answer_exposure": index == len(STEP_IDS) - 1,
        "must_teach_refs": ["must-%d" % (index + 1)],
    }


def script_payload():
    clauses = [clause_payload(index) for index in range(len(STEP_IDS))]
    clauses.insert(
        2,
        clause_payload(
            1,
            clause_id="clause-2-resume",
            episode_id="episode-2",
            math_reference="x=2n",
        ),
    )
    clauses[1]["must_teach_refs"] = ["must-2"]
    clauses[2]["must_teach_refs"] = []
    return {
        "title": "参数根与参数关系",
        "learning_goal": "会把已知根代回并得到目标关系",
        "method_rationale": "根满足原方程",
        "method_introduction": {
            "method_name": "代入法",
            "student_definition": "把已知根代回方程",
            "target_form": "只求m-n",
            "why_it_helps": "直接得到参数关系",
        },
        "opening_clause_ids": ["clause-1"],
        "method_introduction_clause_ids": ["clause-2"],
        "clauses": clauses,
        "closing_summary_clause_ids": ["clause-7"],
    }


def transfer_payload():
    return {
        "problem_text": "另一题仍用参数根求关系",
        "expected_answer": "p-q=1/2",
        "method_signal": "先代入已知根",
        "options": [
            {"option_id": "transfer-a", "label": r"\(p-q=\frac{1}{2}\)", "canonical_answer": "p-q=1/2", "feedback": "正确"},
            {"option_id": "transfer-b", "label": r"\(p-q=2\)", "canonical_answer": "p-q=2", "feedback": "检查除法"},
            {"option_id": "transfer-c", "label": r"\(p+q=\frac{1}{2}\)", "canonical_answer": "p+q=1/2", "feedback": "检查目标"},
        ],
        "correct_option_id": "transfer-a",
    }


def interaction_plan_payload():
    return {
        "interactions": [
            {
                "interaction_id": "interaction-1",
                "episode_id": "episode-2",
                "after_clause_id": "clause-2",
                "diagnostic_target": "是否知道代入已知根",
                "diagnostic_kind": "conception",
                "prompt": "下一步应处理哪个已知条件？",
                "options": [
                    {"option_id": "option-a", "display_text": "代入已知根", "canonical_answer": "substitute-root"},
                    {"option_id": "option-b", "display_text": "分别求m和n", "canonical_answer": "solve-separately", "misconception": "偏离目标"},
                    {"option_id": "option-c", "display_text": "忽略根条件", "canonical_answer": "ignore-root", "misconception": "未用已知"},
                ],
                "correct_option_id": "option-a",
                "correct_feedback": "对，根一定满足原方程。",
                "incorrect_feedback_by_option": {"option-b": "目标只是关系。", "option-c": "先用根条件。"},
                "hint": "想想根的定义。",
                "resume_clause_id": "clause-2-resume",
                "concealed_targets": [],
            }
        ],
        "transfer_item": transfer_payload(),
    }


def score_payload():
    script = script_payload()
    board_objects = [
        {"board_object_id": "board-%d" % (index + 1), "content": STATES[index]}
        for index in range(len(STEP_IDS))
    ]
    cues = []
    written_targets = set()
    for clause in script["clauses"]:
        episode_index = int(clause["episode_id"].split("-")[1]) - 1
        target = "board-%d" % (episode_index + 1)
        if target in written_targets:
            action = {
                "surface": "board",
                "type": "focus",
                "target": target,
            }
        else:
            action = {
                "surface": "board",
                "type": "write",
                "target": target,
                "content": clause["math_references"][0],
            }
            written_targets.add(target)
        cues.append(
            {
                "cue_id": "cue-%s" % clause["clause_id"],
                "clause_ids": [clause["clause_id"]],
                "start_actions": [
                    {
                        "clause_id": clause["clause_id"],
                        "action": action,
                    }
                ],
            }
        )
    return {"cues": cues, "board_objects": board_objects, "overlay_transitions": []}


def simulation_payload():
    return {
        "episode_results": [
            {
                "episode_id": "episode-%d" % (index + 1),
                "learner_profile": "初学者",
                "can_identify_attention_target": True,
                "can_explain_decision": True,
                "can_execute_action": True,
                "can_use_result_to_continue": True,
                "evidence": ["能复述当前步骤"],
            }
            for index in range(len(STEP_IDS))
        ],
        "interaction_results": ["interaction-1: learner selected option-b then retried"],
        "end_of_lesson_recall": "先代入根，再用n不为0，最后回到m-n",
        "blocking_findings": [],
    }


def review_payload():
    return {
        "status": "approved",
        "findings": [
            {
                "finding_id": "finding-polish",
                "severity": "polish",
                "artifact_type": "teaching_script",
                "artifact_id": "clause-3",
                "criterion": "语言简洁",
                "evidence": "句子略长",
                "responsible_role": "script_teacher",
                "requested_change": "下次可精简",
                "invalidated_downstream_artifacts": [],
            }
        ],
        "retained_artifacts": [],
        "approval_summary": "硬门槛通过",
    }


def prepared_payload():
    return {
        "rubric_version": PEDAGOGY_RUBRIC_VERSION,
        "solution_trace": trace_payload(),
        "reasoning_trajectory": trajectory_payload(),
        "teaching_script": script_payload(),
        "interaction_plan": interaction_plan_payload(),
        "performance_score": score_payload(),
        "simulation_report": simulation_payload(),
        "review": review_payload(),
        "repair_count": 0,
        "artifact_history": [
            {"artifact_type": artifact, "version": 1, "responsible_role": role}
            for artifact, role in (
                ("solution_trace", "reference_analyst"),
                ("reasoning_trajectory", "teaching_designer"),
                ("teaching_script", "script_teacher"),
                ("interaction_plan", "interaction_designer"),
                ("performance_score", "classroom_director"),
            )
        ],
    }


def models():
    payload = prepared_payload()
    return (
        SolutionTrace.model_validate(payload["solution_trace"]),
        ReasoningTrajectory.model_validate(payload["reasoning_trajectory"]),
        TeachingScript.model_validate(payload["teaching_script"]),
        InteractionPlan.model_validate(payload["interaction_plan"]),
        PerformanceScore.model_validate(payload["performance_score"]),
        SimulationReport.model_validate(payload["simulation_report"]),
        LessonReviewDecision.model_validate(payload["review"]),
    )


def assert_code(code, call):
    with pytest.raises(PreparationValidationError) as error:
        call()
    assert error.value.code == code
    assert error.value.artifact_id
    assert error.value.detail
    assert "冻结路线步骤" not in error.value.detail


def test_parameter_root_full_traceability_matrix_validates():
    prepared = PreparedLesson.model_validate(prepared_payload())
    targets = [ProblemFocusTarget(target_id="problem-root", math_text="2n", ordinal=1)]
    validate_prepared_lesson(prepared, route(), targets)


def test_solution_trace_rejects_conclusion_mismatch_and_missing_assumption():
    payload = trace_payload()
    payload["reference_conclusion"] = "m-n=2"
    trace = SolutionTrace.model_validate(payload)
    assert_code("trace_conclusion_mismatch", lambda: validate_solution_trace(trace, route()))
    payload = trace_payload()
    payload["source_steps"][0]["assumption_ids_used"] = ["assumption-missing"]
    trace = SolutionTrace.model_validate(payload)
    assert_code("trace_assumption_missing", lambda: validate_solution_trace(trace, route()))


def test_solution_trace_does_not_collapse_set_grouping_into_digits():
    brief = ReferenceGroundingBrief.validate_for_reference_answer(
        {
            "task_summary": "区分集合解与两位数",
            "target": "x",
            "assumptions": [],
            "reference_conclusion": "x=12",
            "method_name": "比较法",
            "reasoning_steps": [
                {
                    "step_id": "is-root",
                    "statement_before": "题目条件",
                    "operation_explanation": "保留集合分隔符",
                    "statement_after": "x=12",
                }
            ],
            "check_requests": [],
            "audit_notes": [],
        },
        "x=12",
    )
    payload = trace_payload()
    payload["reference_conclusion"] = "x={1,2}"
    invalid = SolutionTrace.model_validate(payload)
    assert_code(
        "trace_conclusion_mismatch",
        lambda: validate_solution_trace(
            invalid,
            freeze_grounded_route(brief, []),
        ),
    )


def test_solution_trace_accepts_presentation_only_conclusion_variants():
    payload = trace_payload()
    payload["reference_conclusion"] = (
        r"\(\left m−n\right=\tfrac{1}{2}\)"
    )
    validate_solution_trace(SolutionTrace.model_validate(payload), route())

    payload["reference_conclusion"] = r"m-n=\frac{-1}{2}"
    validate_solution_trace(
        SolutionTrace.model_validate(payload),
        route("m-n=-1/2"),
    )


def test_solution_trace_rejects_structurally_inconsistent_verified_route_anchor():
    payload = trace_payload()
    payload["source_steps"][0]["source_anchor"]["source_id"] = "route-step-missing"
    trace = SolutionTrace.model_validate(payload)
    assert_code("trace_source_anchor_invalid", lambda: validate_solution_trace(trace, route()))


def test_reasoning_trajectory_rejects_missing_and_uncovered_source_steps():
    trace, trajectory, *_ = models()
    payload = trajectory.model_dump()
    payload["episodes"][0]["source_step_ids"] = ["step-missing"]
    invalid = ReasoningTrajectory.model_validate(payload)
    assert_code("episode_source_missing", lambda: validate_reasoning_trajectory(invalid, trace))
    payload = trajectory.model_dump()
    payload["episodes"][-1]["source_step_ids"] = ["use-nonzero"]
    invalid = ReasoningTrajectory.model_validate(payload)
    assert_code("trace_step_uncovered", lambda: validate_reasoning_trajectory(invalid, trace))


def test_reasoning_trajectory_accepts_exact_step_id_audit_note_for_uncovered_step():
    trace, trajectory, *_ = models()
    trace = trace.model_copy(update={"audit_notes": ["return-target deliberately audited"]})
    payload = trajectory.model_dump()
    payload["episodes"][-1]["source_step_ids"] = ["use-nonzero"]
    validate_reasoning_trajectory(ReasoningTrajectory.model_validate(payload), trace)


def test_reasoning_trajectory_rejects_later_step_before_required_earlier_step():
    trace, trajectory, *_ = models()
    payload = trajectory.model_dump()
    payload["episodes"][0]["source_step_ids"] = ["substitute-root", "is-root"]
    invalid = ReasoningTrajectory.model_validate(payload)
    assert_code("episode_source_order_invalid", lambda: validate_reasoning_trajectory(invalid, trace))


def test_teaching_script_rejects_missing_episode_and_must_teach_coverage():
    _, trajectory, script, *_ = models()
    payload = script.model_dump()
    payload["clauses"][0]["episode_id"] = "episode-missing"
    invalid = TeachingScript.model_validate(payload)
    assert_code("clause_episode_missing", lambda: validate_teaching_script(invalid, trajectory))
    payload = script.model_dump()
    payload["clauses"][0]["must_teach_refs"] = []
    invalid = TeachingScript.model_validate(payload)
    assert_code("must_teach_uncovered", lambda: validate_teaching_script(invalid, trajectory))


def test_teaching_script_rejects_invalid_or_cross_episode_must_teach_reference():
    _, trajectory, script, *_ = models()
    payload = script.model_dump()
    payload["clauses"][0]["must_teach_refs"] = ["must-missing"]
    invalid = TeachingScript.model_validate(payload)
    assert_code("must_teach_ref_invalid", lambda: validate_teaching_script(invalid, trajectory))
    payload = script.model_dump()
    payload["clauses"][0]["must_teach_refs"] = ["must-2"]
    invalid = TeachingScript.model_validate(payload)
    assert_code("must_teach_ref_invalid", lambda: validate_teaching_script(invalid, trajectory))


def test_teaching_script_rejects_duplicate_must_teach_ids_across_episodes():
    _, trajectory, script, *_ = models()
    trajectory_value = trajectory.model_dump()
    trajectory_value["episodes"][1]["must_teach"][0]["must_teach_id"] = "must-1"
    trajectory = ReasoningTrajectory.model_validate(trajectory_value)
    script_value = script.model_dump()
    script_value["clauses"][1]["must_teach_refs"] = []
    script = TeachingScript.model_validate(script_value)
    assert_code(
        "must_teach_id_duplicate",
        lambda: validate_teaching_script(script, trajectory),
    )


def test_teaching_script_rejects_episode_reordering_and_spoken_markup():
    _, trajectory, script, *_ = models()
    payload = script.model_dump()
    payload["clauses"][0]["episode_id"], payload["clauses"][1]["episode_id"] = "episode-2", "episode-1"
    payload["clauses"][0]["must_teach_refs"], payload["clauses"][1]["must_teach_refs"] = ["must-2"], ["must-1"]
    invalid = TeachingScript.model_validate(payload)
    assert_code("clause_episode_order_invalid", lambda: validate_teaching_script(invalid, trajectory))
    payload = script.model_dump()
    payload["clauses"][0]["spoken_text"] = "得到 $x=2n$。"
    invalid = TeachingScript.model_validate(payload)
    assert_code("spoken_markup_invalid", lambda: validate_teaching_script(invalid, trajectory))


def test_interaction_plan_rejects_clause_binding_and_answer_leakage():
    _, trajectory, script, plan, *_ = models()
    payload = plan.model_dump()
    payload["interactions"][0]["resume_clause_id"] = "clause-1"
    invalid = InteractionPlan.model_validate(payload)
    assert_code("interaction_clause_invalid", lambda: validate_interaction_plan(invalid, trajectory, script))
    payload = plan.model_dump()
    payload["interactions"][0]["concealed_targets"] = ["option-a"]
    invalid = InteractionPlan.model_validate(payload)
    assert_code("interaction_answer_leakage", lambda: validate_interaction_plan(invalid, trajectory, script))


@pytest.mark.parametrize("field", ("prompt", "hint"))
def test_interaction_plan_rejects_correct_answer_leakage_in_student_visible_text(field):
    _, trajectory, script, plan, *_ = models()
    payload = plan.model_dump()
    payload["interactions"][0][field] = "正确答案是代入已知根"
    invalid = InteractionPlan.model_validate(payload)
    assert_code("interaction_answer_leakage", lambda: validate_interaction_plan(invalid, trajectory, script))


def test_interaction_plan_rejects_correct_answer_leakage_in_wrong_label_and_unknown_concealed_id():
    _, trajectory, script, plan, *_ = models()
    payload = plan.model_dump()
    payload["interactions"][0]["options"][1]["display_text"] = "错误项却写了代入已知根"
    invalid = InteractionPlan.model_validate(payload)
    assert_code("interaction_answer_leakage", lambda: validate_interaction_plan(invalid, trajectory, script))


def test_interaction_plan_does_not_treat_x1_as_substring_of_x10():
    _, trajectory, script, plan, *_ = models()
    payload = plan.model_dump()
    interaction = payload["interactions"][0]
    interaction["options"][0].update(
        display_text="x=1",
        canonical_answer="x=1",
    )
    interaction["options"][1].update(
        display_text="x=10",
        canonical_answer="x=10",
    )
    validate_interaction_plan(
        InteractionPlan.model_validate(payload),
        trajectory,
        script,
    )


@pytest.mark.parametrize("display_text", ("x=1/2", "x=1.5", "x=1+2"))
def test_interaction_plan_accepts_compound_math_distractors(display_text):
    _, trajectory, script, plan, *_ = models()
    payload = plan.model_dump()
    interaction = payload["interactions"][0]
    interaction["options"][0].update(
        display_text="x=1",
        canonical_answer="x=1",
    )
    interaction["options"][1].update(
        display_text=display_text,
        canonical_answer="private-distractor-value",
    )
    validate_interaction_plan(
        InteractionPlan.model_validate(payload), trajectory, script
    )


def test_interaction_plan_ignores_private_wrong_canonical_answer():
    _, trajectory, script, plan, *_ = models()
    payload = plan.model_dump()
    interaction = payload["interactions"][0]
    interaction["options"][0].update(
        display_text="x=1",
        canonical_answer="x=1",
    )
    interaction["options"][1].update(
        display_text="另一个计算过程",
        canonical_answer="x=1",
    )
    validate_interaction_plan(
        InteractionPlan.model_validate(payload), trajectory, script
    )


def test_interaction_plan_still_rejects_exact_displayed_answer_leaks():
    _, trajectory, script, plan, *_ = models()
    payload = plan.model_dump()
    interaction = payload["interactions"][0]
    interaction["options"][0].update(
        display_text="x=1",
        canonical_answer="x=1",
    )
    interaction["options"][1].update(
        display_text="错误项直接写出 x=1",
        canonical_answer="private-wrong-value",
    )
    invalid = InteractionPlan.model_validate(payload)
    assert_code(
        "interaction_answer_leakage",
        lambda: validate_interaction_plan(invalid, trajectory, script),
    )


def test_interaction_plan_short_option_id_requires_answer_announcement():
    _, trajectory, script, plan, *_ = models()
    payload = plan.model_dump()
    interaction = payload["interactions"][0]
    interaction["options"][0]["option_id"] = "a"
    interaction["correct_option_id"] = "a"
    interaction["prompt"] = "a variable appears in ordinary prose"
    validate_interaction_plan(
        InteractionPlan.model_validate(payload), trajectory, script
    )
    interaction["prompt"] = "correct:A"
    invalid = InteractionPlan.model_validate(payload)
    assert_code(
        "interaction_answer_leakage",
        lambda: validate_interaction_plan(invalid, trajectory, script),
    )


@pytest.mark.parametrize("visible_field", ("prompt", "hint", "wrong_option"))
def test_interaction_plan_rejects_same_episode_concealed_semantic_content(visible_field):
    _, trajectory, script, plan, *_ = models()
    trajectory_value = trajectory.model_dump()
    trajectory_value["episodes"][1]["must_teach"][0]["content"] = (
        "尚未公开的检查条件"
    )
    trajectory = ReasoningTrajectory.model_validate(trajectory_value)
    payload = plan.model_dump()
    interaction = payload["interactions"][0]
    interaction["concealed_targets"] = ["must-2"]
    concealed_content = "尚未公开的检查条件"
    if visible_field == "wrong_option":
        interaction["options"][1]["display_text"] = (
            "错误项提前出现%s" % concealed_content
        )
    else:
        interaction[visible_field] = "请判断：%s" % concealed_content
    invalid = InteractionPlan.model_validate(payload)
    assert_code(
        "interaction_answer_leakage",
        lambda: validate_interaction_plan(invalid, trajectory, script),
    )


def test_interaction_plan_concealed_registry_is_same_episode_and_valid_when_content_absent():
    _, trajectory, script, plan, *_ = models()
    payload = plan.model_dump()
    payload["interactions"][0]["concealed_targets"] = ["must-3"]
    invalid = InteractionPlan.model_validate(payload)
    assert_code(
        "interaction_answer_leakage",
        lambda: validate_interaction_plan(invalid, trajectory, script),
    )
    validate_interaction_plan(plan, trajectory, script)
    payload = plan.model_dump()
    payload["interactions"][0]["concealed_targets"] = ["future-id-missing"]
    invalid = InteractionPlan.model_validate(payload)
    assert_code("interaction_answer_leakage", lambda: validate_interaction_plan(invalid, trajectory, script))


def test_interaction_plan_resolves_same_episode_clause_math_as_concealed_content():
    _, trajectory, script, plan, *_ = models()
    payload = plan.model_dump()
    interaction = payload["interactions"][0]
    interaction["concealed_targets"] = ["clause-2"]
    validate_interaction_plan(
        InteractionPlan.model_validate(payload), trajectory, script
    )
    interaction["prompt"] = "题面现在展示x=2n"
    invalid = InteractionPlan.model_validate(payload)
    assert_code(
        "interaction_answer_leakage",
        lambda: validate_interaction_plan(invalid, trajectory, script),
    )


def test_interaction_plan_rejects_katex_equivalent_interaction_and_transfer_labels():
    _, trajectory, script, plan, *_ = models()
    payload = plan.model_dump()
    payload["interactions"][0]["options"][0]["display_text"] = r"\(\dfrac{1}{2}\)"
    payload["interactions"][0]["options"][1]["display_text"] = r"\( \frac{1}{2} \)"
    invalid = InteractionPlan.model_validate(payload)
    assert_code("choice_formula_duplicate", lambda: validate_interaction_plan(invalid, trajectory, script))
    payload = plan.model_dump()
    payload["transfer_item"]["options"][1]["label"] = r"\(p-q=\dfrac{1}{2}\)"
    invalid = InteractionPlan.model_validate(payload)
    assert_code("choice_formula_duplicate", lambda: validate_interaction_plan(invalid, trajectory, script))


def test_performance_score_rejects_clause_coverage_and_nonadjacent_cue():
    _, _, script, plan, score, *_ = models()
    payload = score.model_dump()
    payload["cues"] = payload["cues"][:-1]
    invalid = PerformanceScore.model_validate(payload)
    assert_code("cue_clause_coverage_invalid", lambda: validate_performance_score(invalid, [], script, plan))
    payload = score.model_dump()
    first = payload["cues"].pop(0)
    third = payload["cues"].pop(1)
    first["clause_ids"].append(third["clause_ids"][0])
    first["start_actions"].extend(third["start_actions"])
    payload["cues"].insert(0, first)
    invalid = PerformanceScore.model_validate(payload)
    assert_code("cue_clause_nonadjacent", lambda: validate_performance_score(invalid, [], script, plan))


def test_performance_score_rejects_visual_clause_target_and_phase_errors():
    _, _, script, plan, score, *_ = models()
    payload = score.model_dump()
    payload["cues"][0]["start_actions"][0]["clause_id"] = "clause-2"
    invalid = PerformanceScore.model_validate(payload)
    assert_code("visual_clause_invalid", lambda: validate_performance_score(invalid, [], script, plan))
    payload = score.model_dump()
    payload["cues"][0]["start_actions"][0]["action"]["target"] = "board-missing"
    invalid = PerformanceScore.model_validate(payload)
    assert_code("visual_target_invalid", lambda: validate_performance_score(invalid, [], script, plan))


def test_performance_score_rejects_write_content_that_disagrees_with_declared_object():
    _, _, script, plan, score, *_ = models()
    payload = score.model_dump()
    payload["board_objects"][0]["content"] = "declared different content"
    invalid = PerformanceScore.model_validate(payload)
    assert_code(
        "visual_target_invalid",
        lambda: validate_performance_score(invalid, [], script, plan),
    )
    payload = score.model_dump()
    action = payload["cues"][0]["start_actions"].pop()
    action["action"]["type"] = "reveal"
    action["action"].pop("content")
    payload["cues"][0]["end_actions"] = [action]
    invalid = PerformanceScore.model_validate(payload)
    assert_code("visual_target_invalid", lambda: validate_performance_score(invalid, [], script, plan))


def test_performance_score_preserves_grouping_for_first_introduction_identity():
    _, _, script, plan, score, *_ = models()
    script_value = script.model_dump()
    script_value["clauses"][0]["math_references"] = ["x={1,2}"]
    script = TeachingScript.model_validate(script_value)
    score_value = score.model_dump()
    score_value["board_objects"][0]["content"] = "x=12"
    score_value["cues"][0]["start_actions"][0]["action"]["content"] = "x=12"
    invalid = PerformanceScore.model_validate(score_value)
    assert_code(
        "visual_action_too_early",
        lambda: validate_performance_score(invalid, [], script, plan),
    )


@pytest.mark.parametrize(
    ("reference", "display"),
    (
        (r"\(\left x=\tfrac{1}{2}\right\)", r"$x=\dfrac{1}{2}$"),
        (r"\(x=\frac{-1}{2}\)", "x=-1/2"),
        (r"\(x^{2}\)", "x^2"),
        (r"\(2\times x\)", r"$2\cdot x$"),
    ),
)
def test_performance_score_accepts_presentation_only_first_introduction_variants(
    reference,
    display,
):
    _, _, script, plan, score, *_ = models()
    script_value = script.model_dump()
    script_value["clauses"][0]["math_references"] = [reference]
    script = TeachingScript.model_validate(script_value)
    score_value = score.model_dump()
    score_value["board_objects"][0]["content"] = display
    score_value["cues"][0]["start_actions"][0]["action"]["content"] = display
    validate_performance_score(
        PerformanceScore.model_validate(score_value), [], script, plan
    )


def test_performance_score_requires_exact_problem_target_models():
    _, _, script, plan, score, *_ = models()
    with pytest.raises(TypeError, match="ProblemFocusTarget"):
        validate_performance_score(score, [{"target_id": "problem-root"}], script, plan)


def test_performance_score_rejects_problem_target_and_actions_that_arrive_too_early():
    _, _, script, plan, score, *_ = models()
    payload = score.model_dump()
    payload["cues"][0]["lead_actions"] = [
        {
            "clause_id": "clause-1",
            "action": {"surface": "problem", "type": "focus", "target": "problem-missing"},
        }
    ]
    invalid = PerformanceScore.model_validate(payload)
    targets = [ProblemFocusTarget(target_id="problem-root", math_text="2n", ordinal=1)]
    assert_code("visual_target_invalid", lambda: validate_performance_score(invalid, targets, script, plan))
    payload = score.model_dump()
    payload["cues"][0]["start_actions"][0]["action"]["content"] = STATES[3]
    payload["board_objects"][0]["content"] = STATES[3]
    invalid = PerformanceScore.model_validate(payload)
    assert_code("visual_action_too_early", lambda: validate_performance_score(invalid, [], script, plan))
    payload = score.model_dump()
    payload["cues"][0]["lead_actions"] = [
        {
            "clause_id": "clause-1",
            "action": {"surface": "problem", "type": "focus", "target": "problem-future"},
        }
    ]
    invalid = PerformanceScore.model_validate(payload)
    future_targets = [
        ProblemFocusTarget(target_id="problem-future", math_text=STATES[3], ordinal=1)
    ]
    assert_code(
        "visual_action_too_early",
        lambda: validate_performance_score(invalid, future_targets, script, plan),
    )


def test_visual_action_bound_to_first_clause_cannot_use_later_clause_in_same_cue():
    _, _, script, plan, score, *_ = models()
    payload = score.model_dump()
    first = payload["cues"].pop(0)
    second = payload["cues"].pop(0)
    first["clause_ids"].extend(second["clause_ids"])
    first["start_actions"][0]["action"]["content"] = STATES[1]
    payload["board_objects"][0]["content"] = STATES[1]
    first["start_actions"].extend(second["start_actions"])
    payload["cues"].insert(0, first)
    invalid = PerformanceScore.model_validate(payload)
    assert_code(
        "visual_action_too_early",
        lambda: validate_performance_score(invalid, [], script, plan),
    )


def test_performance_score_rejects_invalid_formula_and_decorative_annotation():
    _, _, script, plan, score, *_ = models()
    payload = score.model_dump()
    payload["board_objects"][0]["content"] = r"\(x=2n$"
    invalid = PerformanceScore.model_validate(payload)
    assert_code("board_formula_invalid", lambda: validate_performance_score(invalid, [], script, plan))
    payload = score.model_dump()
    payload["cues"][0]["start_actions"].append(
        {
            "clause_id": "clause-1",
            "action": {
                "surface": "board",
                "type": "annotate",
                "target": "board-1",
                "annotation": "label",
                "content": STATES[0],
            },
        }
    )
    invalid = PerformanceScore.model_validate(payload)
    assert_code("non_discriminating_emphasis", lambda: validate_performance_score(invalid, [], script, plan))


def test_performance_score_rejects_unbalanced_or_unknown_overlay_transitions():
    _, _, script, plan, score, *_ = models()
    payload = score.model_dump()
    payload["overlay_transitions"] = [
        {"transition_id": "enter-1", "after_clause_id": "clause-missing", "action": "enter", "layer": "comparison"}
    ]
    invalid = PerformanceScore.model_validate(payload)
    assert_code("overlay_transition_invalid", lambda: validate_performance_score(invalid, [], script, plan))


def overlay_score_payload():
    payload = score_payload()
    payload["board_objects"][2]["layer"] = "comparison"
    payload["overlay_transitions"] = [
        {
            "transition_id": "enter-comparison",
            "after_clause_id": "clause-2-resume",
            "action": "enter",
            "layer": "comparison",
        },
        {
            "transition_id": "return-comparison",
            "after_clause_id": "clause-3",
            "action": "return",
            "layer": "comparison",
        },
    ]
    payload["cues"][3]["start_actions"].append(
        {
            "clause_id": "clause-3",
            "action": {
                "surface": "board",
                "type": "focus",
                "target": "board-3",
            },
        }
    )
    return payload


def test_performance_score_accepts_overlay_write_use_return_then_base_resume():
    _, _, script, plan, *_ = models()
    validate_performance_score(
        PerformanceScore.model_validate(overlay_score_payload()),
        [],
        script,
        plan,
    )


@pytest.mark.parametrize("leak_kind", ("focus", "source", "relation"))
def test_overlay_object_cannot_leak_into_base_lifecycle_after_return(leak_kind):
    _, _, script, plan, *_ = models()
    payload = overlay_score_payload()
    base_cue = payload["cues"][4]
    if leak_kind == "focus":
        base_cue["start_actions"] = [
            {
                "clause_id": "clause-4",
                "action": {
                    "surface": "board",
                    "type": "focus",
                    "target": "board-3",
                },
            }
        ]
    elif leak_kind == "source":
        base_cue["start_actions"][0]["action"].update(
            type="transform",
            source="board-3",
        )
    else:
        base_cue["start_actions"].append(
            {
                "clause_id": "clause-4",
                "action": {
                    "surface": "board",
                    "type": "annotate",
                    "target": "board-4",
                    "annotation": "arrow",
                    "relation_target": "board-3",
                },
            }
        )
    invalid = PerformanceScore.model_validate(payload)
    assert_code(
        "visual_target_invalid",
        lambda: validate_performance_score(invalid, [], script, plan),
    )


def test_overlay_transition_must_align_to_cue_boundary_and_cannot_nest():
    _, _, script, plan, score, *_ = models()
    payload = score.model_dump()
    first = payload["cues"].pop(0)
    second = payload["cues"].pop(0)
    first["clause_ids"].extend(second["clause_ids"])
    first["start_actions"].extend(second["start_actions"])
    payload["cues"].insert(0, first)
    payload["overlay_transitions"] = [
        {"transition_id": "enter-1", "after_clause_id": "clause-1", "action": "enter", "layer": "comparison"},
        {"transition_id": "return-1", "after_clause_id": "clause-2", "action": "return", "layer": "comparison"},
    ]
    invalid = PerformanceScore.model_validate(payload)
    assert_code(
        "overlay_transition_invalid",
        lambda: validate_performance_score(invalid, [], script, plan),
    )
    payload = score.model_dump()
    payload["overlay_transitions"] = [
        {"transition_id": "enter-1", "after_clause_id": "clause-1", "action": "enter", "layer": "comparison"},
        {"transition_id": "enter-2", "after_clause_id": "clause-1", "action": "enter", "layer": "micro_explanation"},
        {"transition_id": "return-2", "after_clause_id": "clause-2", "action": "return", "layer": "micro_explanation"},
        {"transition_id": "return-1", "after_clause_id": "clause-3", "action": "return", "layer": "comparison"},
    ]
    invalid = PerformanceScore.model_validate(payload)
    assert_code(
        "overlay_transition_invalid",
        lambda: validate_performance_score(invalid, [], script, plan),
    )
    payload = score.model_dump()
    payload["overlay_transitions"] = [
        {"transition_id": "enter-1", "after_clause_id": "clause-1", "action": "enter", "layer": "comparison"},
        {"transition_id": "return-1", "after_clause_id": "clause-3", "action": "return", "layer": "comparison"},
    ]
    invalid = PerformanceScore.model_validate(payload)
    assert_code("overlay_transition_invalid", lambda: validate_performance_score(invalid, [], script, plan))
    payload = score.model_dump()
    payload["overlay_transitions"] = [
        {"transition_id": "enter-1", "after_clause_id": "clause-2", "action": "enter", "layer": "comparison"}
    ]
    invalid = PerformanceScore.model_validate(payload)
    assert_code("overlay_transition_invalid", lambda: validate_performance_score(invalid, [], script, plan))


def test_overlay_enter_and_return_require_an_intervening_cue():
    _, _, script, plan, score, *_ = models()
    payload = score.model_dump()
    payload["overlay_transitions"] = [
        {
            "transition_id": "enter-1",
            "after_clause_id": "clause-1",
            "action": "enter",
            "layer": "comparison",
        },
        {
            "transition_id": "return-1",
            "after_clause_id": "clause-1",
            "action": "return",
            "layer": "comparison",
        },
    ]
    invalid = PerformanceScore.model_validate(payload)
    assert_code(
        "overlay_transition_invalid",
        lambda: validate_performance_score(invalid, [], script, plan),
    )


def large_performance_models(clause_count):
    script_value = script_payload()
    clauses = []
    board_objects = []
    cues = []
    for index in range(clause_count):
        clause_id = "large-clause-%d" % index
        content = "x=%d" % index
        clauses.append(
            clause_payload(
                0,
                clause_id=clause_id,
                episode_id="episode-1",
                math_reference=content,
            )
        )
        clauses[-1]["must_teach_refs"] = []
        board_id = "large-board-%d" % index
        board_objects.append(
            {"board_object_id": board_id, "content": content}
        )
        cues.append(
            {
                "cue_id": "large-cue-%d" % index,
                "clause_ids": [clause_id],
                "start_actions": [
                    {
                        "clause_id": clause_id,
                        "action": {
                            "surface": "board",
                            "type": "write",
                            "target": board_id,
                            "content": content,
                        },
                    }
                ],
            }
        )
    script_value.update(
        clauses=clauses,
        opening_clause_ids=[clauses[0]["clause_id"]],
        method_introduction_clause_ids=[clauses[1]["clause_id"]],
        closing_summary_clause_ids=[clauses[-1]["clause_id"]],
    )
    return (
        TeachingScript.model_validate(script_value),
        PerformanceScore.model_validate(
            {
                "cues": cues,
                "board_objects": board_objects,
                "overlay_transitions": [],
            }
        ),
    )


def test_performance_score_handles_large_valid_artifact_with_bounded_state():
    _, _, _, plan, *_ = models()
    script, score = large_performance_models(128)
    validate_performance_score(score, [], script, plan)


def test_performance_score_rejects_artifacts_over_explicit_count_bound():
    with pytest.raises(ValidationError, match="at most 256"):
        large_performance_models(257)


def test_performance_score_uses_upstream_problem_target_cap():
    _, _, script, plan, score, *_ = models()
    targets = [
        ProblemFocusTarget(
            target_id="problem-target-%d" % index,
            math_text="x=%d" % index,
            ordinal=(index % 64) + 1,
        )
        for index in range(65)
    ]
    assert_code(
        "artifact_size_invalid",
        lambda: validate_performance_score(
            score,
            targets,
            script,
            plan,
        ),
    )


def test_performance_score_rejects_excessive_math_reference_count():
    _, _, script, *_ = models()
    script_value = script.model_dump()
    script_value["clauses"][0]["math_references"] = [STATES[0]] + [
        "extra-reference-%d" % index for index in range(2048)
    ]
    with pytest.raises(ValidationError, match="at most 2048"):
        TeachingScript.model_validate(script_value)


def test_simulation_report_requires_exact_episode_coverage_and_no_private_answers():
    _, trajectory, _, plan, _, report, _ = models()
    payload = report.model_dump()
    payload["episode_results"] = payload["episode_results"][:-1]
    invalid = SimulationReport.model_validate(payload)
    assert_code("simulation_episode_coverage_invalid", lambda: validate_simulation_report(invalid, trajectory, plan))
    payload = report.model_dump()
    payload["interaction_results"] = ["interaction-1 correct_option_id=option-a"]
    invalid = SimulationReport.model_validate(payload)
    assert_code("simulation_private_answer_invalid", lambda: validate_simulation_report(invalid, trajectory, plan))
    payload = report.model_dump()
    payload["interaction_results"] = ["student recalled canonical answer substitute-root"]
    invalid = SimulationReport.model_validate(payload)
    assert_code("simulation_private_answer_invalid", lambda: validate_simulation_report(invalid, trajectory, plan))


@pytest.mark.parametrize(
    "private_value",
    (
        "p-q=1/2",
        "transfer-a",
        r"\(p-q=\frac{1}{2}\)",
    ),
)
def test_simulation_report_rejects_transfer_private_answer_material(private_value):
    _, trajectory, _, plan, _, report, _ = models()
    payload = report.model_dump()
    payload["interaction_results"] = ["raw transfer result: %s" % private_value]
    invalid = SimulationReport.model_validate(payload)
    assert_code(
        "simulation_private_answer_invalid",
        lambda: validate_simulation_report(invalid, trajectory, plan),
    )


def test_simulation_report_accepts_nonleaking_incorrect_option_observation():
    _, trajectory, _, plan, _, report, _ = models()
    payload = report.model_dump()
    payload["interaction_results"] = [
        "learner selected transfer-b and explained the mistake"
    ]
    validate_simulation_report(
        SimulationReport.model_validate(payload), trajectory, plan
    )


@pytest.mark.parametrize(
    "observed_distractor",
    ("x=10", "x=1/2", "x=1.5", "x=1+2"),
)
def test_simulation_private_answers_use_bounded_math_tokens(
    observed_distractor,
):
    _, trajectory, _, plan, _, report, _ = models()
    plan_value = plan.model_dump()
    transfer = plan_value["transfer_item"]
    transfer["expected_answer"] = "x=1"
    transfer["options"][0].update(
        label=r"\(x=1\)",
        canonical_answer="x=1",
    )
    plan = InteractionPlan.model_validate(plan_value)
    report_value = report.model_dump()
    report_value["interaction_results"] = [
        "learner tested %s first" % observed_distractor
    ]
    validate_simulation_report(
        SimulationReport.model_validate(report_value), trajectory, plan
    )


def test_simulation_short_private_option_id_requires_explicit_announcement():
    _, trajectory, _, plan, _, report, _ = models()
    plan_value = plan.model_dump()
    transfer = plan_value["transfer_item"]
    transfer["options"][0]["option_id"] = "a"
    transfer["correct_option_id"] = "a"
    plan = InteractionPlan.model_validate(plan_value)
    report_value = report.model_dump()
    report_value["interaction_results"] = [
        "learner made a careful choice"
    ]
    validate_simulation_report(
        SimulationReport.model_validate(report_value), trajectory, plan
    )
    report_value["interaction_results"] = ["correct:A"]
    invalid = SimulationReport.model_validate(report_value)
    assert_code(
        "simulation_private_answer_invalid",
        lambda: validate_simulation_report(invalid, trajectory, plan),
    )


@pytest.mark.parametrize(
    "markup",
    (
        '<span class="is-highlighted">重点</span>',
        ".is-highlighted",
        "#board-target",
        "[data-highlight]",
        "[[red]",
        "{{highlight",
        "<mark",
    ),
)
def test_spoken_and_board_content_reject_internal_control_fragments(markup):
    _, trajectory, script, plan, score, *_ = models()
    script_payload_value = script.model_dump()
    script_payload_value["clauses"][0]["spoken_text"] = markup
    invalid_script = TeachingScript.model_validate(script_payload_value)
    assert_code(
        "spoken_markup_invalid",
        lambda: validate_teaching_script(invalid_script, trajectory),
    )
    score_payload_value = score.model_dump()
    score_payload_value["board_objects"][0]["content"] = markup
    invalid_score = PerformanceScore.model_validate(score_payload_value)
    assert_code(
        "board_formula_invalid",
        lambda: validate_performance_score(invalid_score, [], script, plan),
    )


@pytest.mark.parametrize(
    "markup",
    (
        r"\(\htmlClass{is-highlighted}{x}\)",
        r"\(\htmlId{board-target}{x}\)",
        r"\[\htmlStyle{color:red}{x}\]",
        r"\(\htmlData{target=board-1}{x}\)",
        r"\(\href{https://example.com}{x}\)",
        r"\(\url{https://example.com}\)",
        r"\(\includegraphics{lesson.png}\)",
        '<img src="lesson.png">',
        '<IMG SRC="lesson.png" />',
        '<section data-target="board-1">x</section>',
    ),
)
def test_board_content_rejects_dom_url_commands_and_generic_html(markup):
    _, _, script, plan, score, *_ = models()
    payload = score.model_dump()
    payload["board_objects"][0]["content"] = markup
    invalid = PerformanceScore.model_validate(payload)
    assert_code(
        "board_formula_invalid",
        lambda: validate_performance_score(invalid, [], script, plan),
    )


@pytest.mark.parametrize(
    "markup",
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
def test_teaching_script_rejects_shared_internal_controls(markup):
    assert contains_internal_control_syntax(markup)
    _, trajectory, script, *_ = models()
    payload = script.model_dump()
    payload["clauses"][0]["spoken_text"] = markup
    invalid = TeachingScript.model_validate(payload)
    assert_code(
        "spoken_markup_invalid",
        lambda: validate_teaching_script(invalid, trajectory),
    )


def test_review_decision_rechecks_invalid_constructed_approval():
    invalid = LessonReviewDecision.model_construct(
        status="approved",
        findings=[
            LessonReviewDecision.model_fields["findings"].annotation.__args__[0].model_validate(
                {
                    "finding_id": "finding-blocking",
                    "severity": "blocking",
                    "artifact_type": "solution_trace",
                    "artifact_id": "is-root",
                    "criterion": "结论一致",
                    "evidence": "不一致",
                    "responsible_role": "reference_analyst",
                    "requested_change": "修复",
                }
            )
        ],
        retained_artifacts=[],
        approval_summary="错误批准",
    )
    assert_code("review_approval_invalid", lambda: validate_review_decision(invalid))


def test_blocking_signature_uses_only_sorted_material_identity_fields():
    first = LessonReviewDecision.model_validate(
        {
            "status": "revision_required",
            "findings": [
                {
                    "finding_id": "one",
                    "severity": "material",
                    "artifact_type": "teaching_script",
                    "artifact_id": "clause-2",
                    "criterion": "必须解释理由",
                    "evidence": "wording one",
                    "responsible_role": "script_teacher",
                    "requested_change": "change one",
                    "invalidated_downstream_artifacts": ["performance_score"],
                },
                review_payload()["findings"][0],
            ],
            "approval_summary": "revise",
        }
    )
    second_payload = first.model_dump()
    second_payload["findings"].reverse()
    second_payload["findings"][1].update(
        finding_id="two", evidence="wording two", requested_change="change two",
        invalidated_downstream_artifacts=[],
    )
    second = LessonReviewDecision.model_validate(second_payload)
    assert blocking_signature(first) == blocking_signature(second)
    assert len(blocking_signature(first)) == 64
    assert blocking_signature(first).islower()


def test_blocking_signature_ignores_duplicate_semantic_findings():
    finding = {
        "finding_id": "one",
        "severity": "material",
        "artifact_type": "teaching_script",
        "artifact_id": "clause-2",
        "criterion": "必须解释理由",
        "evidence": "wording one",
        "responsible_role": "script_teacher",
        "requested_change": "change one",
    }
    single = LessonReviewDecision.model_validate(
        {
            "status": "revision_required",
            "findings": [finding],
            "approval_summary": "revise",
        }
    )
    duplicate = copy.deepcopy(finding)
    duplicate.update(
        finding_id="two",
        evidence="wording two",
        requested_change="change two",
    )
    repeated = LessonReviewDecision.model_validate(
        {
            "status": "revision_required",
            "findings": [finding, duplicate],
            "approval_summary": "revise",
        }
    )
    assert blocking_signature(single) == blocking_signature(repeated)


def test_prepared_lesson_rejects_misaligned_rubric_version():
    prepared = PreparedLesson.model_validate(prepared_payload()).model_copy(update={"rubric_version": "v1"})
    assert_code(
        "rubric_version_invalid",
        lambda: validate_prepared_lesson(prepared, route(), []),
    )
