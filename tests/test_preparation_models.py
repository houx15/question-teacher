import copy

import pytest
from pydantic import ValidationError

import app.preparation_models as preparation_models
from app.preparation_models import (
    ArtifactRevision,
    ClauseBoundVisualAction,
    GenerationRecord,
    InteractionPlan,
    LessonReviewDecision,
    PlannedInteraction,
    PreparedLesson,
    ReasoningTrajectory,
    RoleCallRecord,
    ScriptClause,
    TeachingScript,
)


def source_anchor(source_id="source-1"):
    return {"source_kind": "solution", "source_id": source_id, "excerpt": "原解答的关键一步"}


def trace_step(step_id="step-1"):
    return {
        "source_step_id": step_id, "source_anchor": source_anchor(step_id),
        "state_before": "x加一等于三", "mathematical_action": "两边同时减一",
        "justification": "等式两边同减相等数仍相等", "state_after": "x等于二",
        "new_information": "未知数的值已确定", "evidence_status": "derived",
    }


def episode(episode_id="episode-1", index=0, mode="understand"):
    return {
        "episode_id": episode_id, "sequence_index": index, "mode": mode,
        "source_step_ids": ["step-1"], "learner_state_before": "知道题目条件",
        "attention_targets": ["等式两边"], "thinking_question": "怎样保留等式？",
        "decision": "两边同减一", "decision_reason": "需要消去常数项",
        "mathematical_action": "两边同时减一", "action_justification": "等式性质",
        "result": "x等于二", "result_meaning": "得到未知数", "transition_reason": "可以检查结果",
        "must_teach": [{"must_teach_id": "teach-%s" % episode_id, "content": "等式性质", "why_it_matters": "保证变形正确"}],
        "interaction_intent": None, "visual_intent": None,
    }


def script_clause(clause_id, episode_id="episode-1"):
    return {
        "clause_id": clause_id, "episode_id": episode_id, "pedagogical_function": "explain",
        "spoken_text": "我们把等式两边同时减一。", "learner_gain": "理解等式性质", "answer_exposure": False,
    }


def transfer_item():
    return {
        "problem_text": "解方程 x加二等于五", "expected_answer": "x等于三", "method_signal": "两边同减二",
        "options": [
            {"option_id": "transfer-a", "label": "x等于三", "canonical_answer": "x等于三", "feedback": "正确"},
            {"option_id": "transfer-b", "label": "x等于七", "canonical_answer": "x等于七", "feedback": "再想想"},
            {"option_id": "transfer-c", "label": "x等于二", "canonical_answer": "x等于二", "feedback": "再想想"},
        ], "correct_option_id": "transfer-a",
    }


def interaction(interaction_id="interaction-1"):
    return {
        "interaction_id": interaction_id, "episode_id": "episode-1", "after_clause_id": "open-1",
        "diagnostic_target": "等式性质", "diagnostic_kind": "conception", "prompt": "下一步怎么做？",
        "options": [
            {"option_id": "option-a", "display_text": "两边同减一", "canonical_answer": "两边同减一"},
            {"option_id": "option-b", "display_text": "只在左边减一", "canonical_answer": "左边减一", "misconception": "单边变形"},
            {"option_id": "option-c", "display_text": "两边同加一", "canonical_answer": "两边加一", "misconception": "方向错误"},
        ], "correct_option_id": "option-a", "correct_feedback": "对。",
        "incorrect_feedback_by_option": {"option-b": "等式两边要同步。", "option-c": "注意消去加一。"},
        "hint": "想想怎样保持等式。", "resume_clause_id": "method-1",
    }


def teaching_script():
    return {
        "title": "一元一次方程", "learning_goal": "掌握等式性质", "method_rationale": "从等式性质出发",
        "method_introduction": {"method_name": "移项法", "student_definition": "把项移到另一边", "target_form": "x等于常数", "why_it_helps": "能求出未知数"},
        "opening_clause_ids": ["open-1"], "method_introduction_clause_ids": ["method-1"],
        "clauses": [script_clause("open-1"), script_clause("method-1"), script_clause("close-1")],
        "closing_summary_clause_ids": ["close-1"],
    }


def prepared_lesson():
    return {
        "rubric_version": "v1", "solution_trace": {"task_target": "求x", "reference_conclusion": "x等于二", "source_steps": [trace_step()]},
        "reasoning_trajectory": {"trajectory_type": "planned", "lesson_purpose": "理解等式性质", "episodes": [episode()], "method_summary": "同减", "error_summary": "避免单边变形"},
        "teaching_script": teaching_script(), "interaction_plan": {"interactions": [], "transfer_item": transfer_item()},
        "performance_score": {"cues": [{"cue_id": "cue-1", "clause_ids": ["open-1"]}]},
        "simulation_report": {"episode_results": [{"episode_id": "episode-1", "learner_profile": "基础学习者", "can_identify_attention_target": True, "can_explain_decision": True, "can_execute_action": True, "can_use_result_to_continue": True, "evidence": ["能说明等式两边同步"]}], "end_of_lesson_recall": "能说出等式性质"},
        "review": {"status": "approved", "findings": [{"finding_id": "finding-1", "severity": "polish", "artifact_type": "teaching_script", "artifact_id": "script-1", "criterion": "语句自然", "evidence": "一个句子可更短", "responsible_role": "script_teacher", "requested_change": "精简句子"}], "approval_summary": "可以使用"},
        "repair_count": 0,
        "artifact_history": [
            {"artifact_type": artifact, "version": 1, "responsible_role": role}
            for artifact, role in [
                ("solution_trace", "reference_analyst"), ("reasoning_trajectory", "teaching_designer"),
                ("teaching_script", "script_teacher"), ("interaction_plan", "interaction_designer"),
                ("performance_score", "classroom_director"),
            ]
        ],
    }


def test_every_artifact_forbids_unknown_fields():
    payload = prepared_lesson()
    payload["solution_trace"]["unexpected"] = "no"
    with pytest.raises(ValidationError):
        PreparedLesson.model_validate(payload)


def test_all_private_artifact_models_reject_unknown_fields():
    models = [
        value for value in vars(preparation_models).values()
        if isinstance(value, type)
        and value.__module__ == preparation_models.__name__
        and issubclass(value, preparation_models.SchemaModel)
    ]
    assert models
    for model in models:
        with pytest.raises(ValidationError) as error:
            model.model_validate({"unknown_field": "no"})
        assert any(item["loc"] == ("unknown_field",) for item in error.value.errors())


def test_ids_are_unique_in_local_artifacts():
    payload = prepared_lesson()
    payload["solution_trace"]["source_steps"].append(trace_step("step-1"))
    with pytest.raises(ValidationError, match="trace step ids"):
        PreparedLesson.model_validate(payload)


def test_generated_ids_are_valid_and_every_local_id_collection_is_unique():
    payload = prepared_lesson()
    payload["solution_trace"]["source_steps"][0]["source_step_id"] = "bad id"
    with pytest.raises(ValidationError):
        PreparedLesson.model_validate(payload)
    payload = prepared_lesson()
    payload["solution_trace"]["assumptions"] = [
        {"assumption_id": "assumption-1", "content": "n不为零", "source_anchor": source_anchor("a")},
        {"assumption_id": "assumption-1", "content": "x是根", "source_anchor": source_anchor("b")},
    ]
    with pytest.raises(ValidationError, match="assumption ids"):
        PreparedLesson.model_validate(payload)
    payload = prepared_lesson()
    first = payload["reasoning_trajectory"]["episodes"][0]
    first["must_teach"].append(copy.deepcopy(first["must_teach"][0]))
    with pytest.raises(ValidationError, match="must-teach ids"):
        PreparedLesson.model_validate(payload)
    payload = prepared_lesson()
    payload["teaching_script"]["clauses"].append(script_clause("open-1"))
    with pytest.raises(ValidationError, match="clause ids"):
        PreparedLesson.model_validate(payload)
    payload = prepared_lesson()
    payload["interaction_plan"]["interactions"] = [interaction(), interaction()]
    with pytest.raises(ValidationError, match="interaction ids"):
        PreparedLesson.model_validate(payload)
    payload = prepared_lesson()
    payload["performance_score"]["cues"].append(copy.deepcopy(payload["performance_score"]["cues"][0]))
    with pytest.raises(ValidationError, match="cue ids"):
        PreparedLesson.model_validate(payload)
    payload = prepared_lesson()
    payload["performance_score"]["board_objects"] = [
        {"board_object_id": "board-1", "content": "x等于二"},
        {"board_object_id": "board-1", "content": "x等于三"},
    ]
    with pytest.raises(ValidationError, match="board object ids"):
        PreparedLesson.model_validate(payload)
    payload = prepared_lesson()
    payload["performance_score"]["overlay_transitions"] = [
        {"transition_id": "transition-1", "after_clause_id": "open-1", "action": "enter", "layer": "comparison"},
        {"transition_id": "transition-1", "after_clause_id": "method-1", "action": "return", "layer": "comparison"},
    ]
    with pytest.raises(ValidationError, match="overlay transition ids"):
        PreparedLesson.model_validate(payload)
    payload = prepared_lesson()
    payload["review"]["findings"].append(copy.deepcopy(payload["review"]["findings"][0]))
    with pytest.raises(ValidationError, match="finding ids"):
        PreparedLesson.model_validate(payload)


def test_reasoning_episodes_accept_interleaved_modes_and_require_contiguous_indexes():
    payload = prepared_lesson()["reasoning_trajectory"]
    payload["episodes"] = [episode("episode-1", 0, "understand"), episode("episode-2", 1, "explore"), episode("episode-3", 2, "monitor")]
    assert [item.mode for item in ReasoningTrajectory.model_validate(payload).episodes] == ["understand", "explore", "monitor"]
    payload["episodes"][2]["sequence_index"] = 4
    with pytest.raises(ValidationError, match="contiguous"):
        ReasoningTrajectory.model_validate(payload)


def test_reasoning_mode_is_closed_literal():
    payload = episode(mode="lecture")
    with pytest.raises(ValidationError):
        ReasoningTrajectory.model_validate({"trajectory_type": "planned", "lesson_purpose": "目标", "episodes": [payload], "method_summary": "方法", "error_summary": "错误"})


def test_teaching_script_owns_nonblank_explanatory_narration_in_clauses():
    payload = script_clause("clause-1")
    payload["spoken_text"] = " "
    with pytest.raises(ValidationError):
        ScriptClause.model_validate(payload)
    assert TeachingScript.model_validate(teaching_script()).clauses[0].spoken_text


def test_interaction_plan_allows_zero_interactions():
    plan = InteractionPlan.model_validate({"interactions": [], "transfer_item": transfer_item()})
    assert plan.interactions == []


def test_interaction_options_require_one_existing_correct_id_and_feedback_for_all_incorrect_options():
    payload = interaction()
    payload["options"][2]["option_id"] = "option-b"
    with pytest.raises(ValidationError, match="option ids"):
        PlannedInteraction.model_validate(payload)
    payload = interaction()
    payload["correct_option_id"] = "option-missing"
    with pytest.raises(ValidationError, match="correct_option_id"):
        PlannedInteraction.model_validate(payload)
    payload = interaction()
    del payload["incorrect_feedback_by_option"]["option-c"]
    with pytest.raises(ValidationError, match="incorrect feedback"):
        PlannedInteraction.model_validate(payload)


def test_script_sections_exist_in_order_and_do_not_overlap():
    payload = teaching_script()
    payload["closing_summary_clause_ids"] = ["method-1"]
    with pytest.raises(ValidationError, match="overlap"):
        TeachingScript.model_validate(payload)
    payload = teaching_script()
    payload["opening_clause_ids"] = ["method-1"]
    payload["method_introduction_clause_ids"] = ["close-1"]
    payload["closing_summary_clause_ids"] = ["open-1"]
    with pytest.raises(ValidationError, match="script order"):
        TeachingScript.model_validate(payload)


def test_review_decision_literals_and_approval_rules():
    payload = prepared_lesson()["review"]
    payload["findings"][0]["severity"] = "blocking"
    with pytest.raises(ValidationError, match="approved"):
        LessonReviewDecision.model_validate(payload)
    payload["status"] = "revision_required"
    assert LessonReviewDecision.model_validate(payload).status == "revision_required"
    payload["findings"] = []
    with pytest.raises(ValidationError, match="blocking or material"):
        LessonReviewDecision.model_validate(payload)
    payload = prepared_lesson()["review"]
    payload["findings"][0]["responsible_role"] = "runtime_operator"
    with pytest.raises(ValidationError):
        LessonReviewDecision.model_validate(payload)


def test_prepared_lesson_requires_rubric_and_complete_history():
    payload = prepared_lesson()
    del payload["rubric_version"]
    with pytest.raises(ValidationError):
        PreparedLesson.model_validate(payload)
    payload = prepared_lesson()
    payload["artifact_history"] = payload["artifact_history"][:4]
    with pytest.raises(ValidationError):
        PreparedLesson.model_validate(payload)


def test_role_call_output_pair_and_nonnegative_token_usage():
    payload = {"role": "lesson_reviewer", "output_artifact_type": "solution_trace", "duration_ms": 0, "retry_count": 0}
    with pytest.raises(ValidationError, match="output type and version"):
        RoleCallRecord.model_validate(payload)
    payload = {"role": "lesson_reviewer", "duration_ms": 0, "retry_count": 0, "token_usage": {"input": -1}}
    with pytest.raises(ValidationError, match="token usage"):
        RoleCallRecord.model_validate(payload)
