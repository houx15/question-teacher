import copy
from typing import Literal

import pytest
from pydantic import ValidationError

import app.preparation_models as preparation_models
from app.preparation_models import (
    ArtifactRevision,
    ClauseBoundVisualAction,
    ExplanationDepth,
    GenerationRecord,
    InteractionPlan,
    LessonReviewDecision,
    PlannedInteraction,
    PreparedLesson,
    ReasoningTrajectory,
    ResponseScript,
    RoleCallRecord,
    ScriptClause,
    SimulationReport,
    TeachingScript,
)


def source_anchor(source_id="source-1"):
    return {"source_kind": "solution", "source_id": source_id, "excerpt": "原解答的关键一步"}


def trace_step(step_id="step-1"):
    return {
        "source_step_id": step_id, "source_anchor": source_anchor(step_id),
        "state_before": "x+1=3", "operation_kind": "subtract",
        "operands": ["1"], "mathematical_action": "两边同时减一",
        "justification": "等式两边同减相等数仍相等", "state_after": "x=2",
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


def teaching_progression_payload():
    return {
        "steps": [
            {
                "step_id": "teaching-step-001",
                "sequence_index": 0,
                "episode_ids": ["episode-001"],
                "phase": "construct",
                "student_problem": "方程的根代表什么？",
                "why_now": "先把题目的关键事实变成可执行条件。",
                "evidence_target_ids": ["problem-focus-001"],
                "guiding_questions": ["把一个数代入方程后会发生什么？"],
                "knowledge_anchor": "方程的根代入后等式成立",
                "checkpoint": None,
                "reveal": "令x=2n",
                "math_action": "把根代入关于x的方程",
                "directory_question": "根代表什么？",
                "directory_label": "第一步：理解方程的根",
                "board_summary": ["方程的根 → 代入后等式成立"],
                "error_tip": "不要把根代给m或n",
                "transition_question": "这个方程是关于谁的？",
                "must_teach_refs": ["must-teach-root"],
            }
        ]
    }


def prepared_lesson():
    return {
        "rubric_version": "v1", "solution_trace": {"task_target": "x", "reference_conclusion": "x=2", "source_steps": [trace_step()]},
        "reasoning_trajectory": {"trajectory_type": "planned", "lesson_purpose": "理解等式性质", "episodes": [episode()], "method_summary": "同减", "error_summary": "避免单边变形"},
        "teaching_script": teaching_script(), "interaction_plan": {"interactions": [], "transfer_item": transfer_item()},
        "performance_score": {"cues": [{"cue_id": "cue-1", "clause_ids": ["open-1"]}]},
        "simulation_report": {"episode_results": [{"episode_id": "episode-1", "learner_profile": "基础学习者", "can_identify_attention_target": True, "can_explain_decision": True, "can_execute_action": True, "can_use_result_to_continue": True, "evidence": ["能说明等式两边同步"]}], "end_of_lesson_recall": "能说出等式性质"},
        "review": {"status": "approved", "findings": [{"finding_id": "finding-1", "severity": "polish", "artifact_type": "teaching_script", "artifact_id": "script-1", "criterion": "learner_follows_why", "evidence": "一个句子可更短", "responsible_role": "script_teacher", "requested_change": "精简句子"}], "approval_summary": "可以使用"},
        "repair_count": 0,
        "artifact_history": [
            {"artifact_type": artifact, "version": 1, "responsible_role": role}
            for artifact, role in [
                ("solution_trace", "reference_analyst"), ("reasoning_trajectory", "teaching_designer"),
                ("teaching_script", "script_teacher"), ("interaction_plan", "interaction_designer"),
                ("performance_score", "classroom_director"),
                ("simulation_report", "student_simulator"),
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
        {"assumption_id": "assumption-1", "content": "n!=0", "source_anchor": source_anchor("a")},
        {"assumption_id": "assumption-1", "content": "x=2n", "source_anchor": source_anchor("b")},
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


def test_reasoning_trajectory_accepts_every_supported_mode_in_order():
    modes = [
        "understand", "plan", "explore", "execute", "monitor", "revise", "reflect",
    ]
    trajectory = ReasoningTrajectory.model_validate({
        "trajectory_type": "hybrid", "lesson_purpose": "完整推理过程",
        "episodes": [episode("episode-%d" % index, index, mode) for index, mode in enumerate(modes)],
        "method_summary": "按需切换推理方式", "error_summary": "留意推理断点",
    })
    assert [item.mode for item in trajectory.episodes] == modes


def test_reasoning_trajectory_rejects_duplicate_episode_ids():
    payload = prepared_lesson()["reasoning_trajectory"]
    payload["episodes"] = [episode("episode-1", 0), episode("episode-1", 1, "plan")]
    with pytest.raises(ValidationError, match="episode ids"):
        ReasoningTrajectory.model_validate(payload)


def test_teaching_progression_accepts_ordered_steps_with_stable_ids_and_valid_phases():
    payload = teaching_progression_payload()
    first_step = payload["steps"][0]
    payload["steps"] = [
        dict(
            first_step,
            step_id="teaching-step-%03d" % (index + 1),
            sequence_index=index,
            phase=phase,
        )
        for index, phase in enumerate(["construct", "explore", "execute", "check"])
    ]

    progression = preparation_models.TeachingProgression.model_validate(payload)

    assert [item.step_id for item in progression.steps] == [
        "teaching-step-001",
        "teaching-step-002",
        "teaching-step-003",
        "teaching-step-004",
    ]
    assert [item.sequence_index for item in progression.steps] == [0, 1, 2, 3]
    assert [item.phase for item in progression.steps] == [
        "construct",
        "explore",
        "execute",
        "check",
    ]


def test_teaching_progression_requires_contiguous_indexes_starting_at_zero():
    payload = teaching_progression_payload()
    payload["steps"][0]["sequence_index"] = 1

    with pytest.raises(
        ValidationError,
        match="teaching step indexes must be contiguous starting at zero",
    ):
        preparation_models.TeachingProgression.model_validate(payload)


def test_teaching_progression_requires_guidance_and_closed_phase_vocabulary():
    payload = teaching_progression_payload()
    payload["steps"][0]["guiding_questions"] = []
    with pytest.raises(ValidationError):
        preparation_models.TeachingProgression.model_validate(payload)

    payload = teaching_progression_payload()
    payload["steps"][0]["phase"] = "lecture"
    with pytest.raises(ValidationError):
        preparation_models.TeachingProgression.model_validate(payload)


def test_teaching_progression_rejects_duplicate_and_invalid_step_ids():
    payload = teaching_progression_payload()
    duplicate = copy.deepcopy(payload["steps"][0])
    duplicate["sequence_index"] = 1
    payload["steps"].append(duplicate)
    with pytest.raises(ValidationError, match="teaching step ids"):
        preparation_models.TeachingProgression.model_validate(payload)

    payload = teaching_progression_payload()
    payload["steps"][0]["step_id"] = "bad id"
    with pytest.raises(ValidationError):
        preparation_models.TeachingProgression.model_validate(payload)


def test_teaching_progression_board_summary_has_explicit_bounds():
    payload = teaching_progression_payload()
    payload["steps"][0]["board_summary"] = [
        "板书%d" % index for index in range(8)
    ]
    assert len(
        preparation_models.TeachingProgression.model_validate(payload)
        .steps[0]
        .board_summary
    ) == 8

    payload["steps"][0]["board_summary"].append("板书超限")
    with pytest.raises(ValidationError, match="at most 8"):
        preparation_models.TeachingProgression.model_validate(payload)

    payload = teaching_progression_payload()
    payload["steps"][0]["board_summary"] = []
    with pytest.raises(ValidationError):
        preparation_models.TeachingProgression.model_validate(payload)


def test_prepared_lesson_contains_first_class_teaching_progression():
    payload = prepared_lesson()
    payload["teaching_progression"] = teaching_progression_payload()

    dumped = PreparedLesson.model_validate(payload).model_dump()
    reparsed = PreparedLesson.model_validate(dumped)

    assert (
        reparsed.teaching_progression.steps[0].directory_label
        == "第一步：理解方程的根"
    )


def test_reasoning_trajectory_episode_count_is_bounded_at_validation_edge():
    maximum = 256
    payload = prepared_lesson()["reasoning_trajectory"]
    payload["episodes"] = [
        episode("episode-%d" % index, index)
        for index in range(maximum)
    ]

    assert len(ReasoningTrajectory.model_validate(payload).episodes) == maximum
    payload["episodes"].append(episode("episode-overflow", maximum))
    with pytest.raises(ValidationError, match="at most 256"):
        ReasoningTrajectory.model_validate(payload)


def test_teaching_script_clause_count_is_bounded_at_validation_edge():
    maximum = 256
    payload = teaching_script()
    payload["clauses"] = [
        script_clause("clause-%d" % index)
        for index in range(maximum)
    ]
    payload["opening_clause_ids"] = ["clause-0"]
    payload["method_introduction_clause_ids"] = ["clause-1"]
    payload["closing_summary_clause_ids"] = ["clause-255"]

    assert len(TeachingScript.model_validate(payload).clauses) == maximum
    payload["clauses"].insert(-1, script_clause("clause-overflow"))
    with pytest.raises(ValidationError, match="at most 256"):
        TeachingScript.model_validate(payload)


def test_performance_score_collection_counts_are_bounded():
    maximum = 256
    payload = {
        "cues": [
            {"cue_id": "cue-%d" % index, "clause_ids": ["clause-1"]}
            for index in range(maximum)
        ],
        "board_objects": [
            {"board_object_id": "board-%d" % index, "content": "x=%d" % index}
            for index in range(maximum)
        ],
        "overlay_transitions": [],
    }

    assert len(
        preparation_models.PerformanceScore.model_validate(payload).cues
    ) == maximum
    payload["cues"].append(
        {"cue_id": "cue-overflow", "clause_ids": ["clause-1"]}
    )
    with pytest.raises(ValidationError, match="at most 256"):
        preparation_models.PerformanceScore.model_validate(payload)


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


def test_script_clause_preserves_legacy_loading_and_orders_new_authoring_fields():
    legacy = ScriptClause.model_validate(script_clause("legacy-clause"))
    assert legacy.lesson_step_id is None
    assert legacy.display_text is None

    properties = list(ScriptClause.model_json_schema()["properties"])
    assert properties.index("lesson_step_id") == properties.index("episode_id") + 1
    assert properties.index("display_text") == properties.index("pedagogical_function") + 1


def test_response_script_has_closed_depth_and_bounded_clauses():
    clause = script_clause("response-clause")
    clause.update(
        lesson_step_id="teaching-step-1",
        display_text=r"因为 \(x=2\)，所以等式成立。",
    )
    payload = {
        "response_id": "response-1",
        "interaction_id": "interaction-1",
        "option_id": "option-a",
        "classification": "correct",
        "error_code": None,
        "depth": "brief",
        "clauses": [clause],
    }
    response = ResponseScript.model_validate(payload)
    assert response.depth == "brief"
    assert response.clauses[0].display_text

    payload["depth"] = "verbose"
    with pytest.raises(ValidationError):
        ResponseScript.model_validate(payload)

    payload["depth"] = "conceptual"
    payload["clauses"] = [
        dict(clause, clause_id="response-clause-%d" % index)
        for index in range(8)
    ]
    assert len(ResponseScript.model_validate(payload).clauses) == 8
    payload["clauses"].append(dict(clause, clause_id="response-overflow"))
    with pytest.raises(ValidationError, match="at most 8"):
        ResponseScript.model_validate(payload)


def test_teaching_script_defaults_response_scripts_to_empty_and_accepts_typed_responses():
    assert TeachingScript.model_validate(teaching_script()).response_scripts == []

    payload = teaching_script()
    clause = script_clause("response-clause")
    clause.update(
        lesson_step_id="teaching-step-1",
        display_text="两边同时减一",
    )
    payload["response_scripts"] = [
        {
            "response_id": "response-1",
            "interaction_id": "interaction-1",
            "option_id": "option-a",
            "classification": "incorrect",
            "error_code": "direction-error",
            "depth": "worked",
            "clauses": [clause],
        }
    ]
    assert TeachingScript.model_validate(payload).response_scripts[0].depth == "worked"


def test_explanation_depth_alias_exposes_exact_supported_vocabulary():
    assert ExplanationDepth == Literal["brief", "conceptual", "worked"]


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


def test_script_sections_must_cover_exact_prefix_adjacent_method_and_suffix():
    payload = teaching_script()
    payload["clauses"].insert(0, script_clause("ordinary-before"))
    with pytest.raises(ValidationError, match="opening prefix"):
        TeachingScript.model_validate(payload)
    payload = teaching_script()
    payload["clauses"].insert(1, script_clause("ordinary-gap"))
    with pytest.raises(ValidationError, match="immediately follow"):
        TeachingScript.model_validate(payload)
    payload = teaching_script()
    payload["clauses"].append(script_clause("ordinary-after"))
    with pytest.raises(ValidationError, match="closing suffix"):
        TeachingScript.model_validate(payload)


def test_script_section_ids_must_exist_in_clauses():
    payload = teaching_script()
    payload["opening_clause_ids"] = ["missing-clause"]
    with pytest.raises(ValidationError, match="must exist in clauses"):
        TeachingScript.model_validate(payload)


def test_review_decision_literals_and_approval_rules():
    payload = prepared_lesson()["review"]
    payload["findings"][0]["severity"] = "blocking"
    with pytest.raises(ValidationError, match="approved"):
        LessonReviewDecision.model_validate(payload)
    payload = prepared_lesson()["review"]
    payload["findings"][0]["severity"] = "material"
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


def test_review_criterion_is_a_stable_bounded_vocabulary():
    payload = prepared_lesson()["review"]
    payload["findings"][0]["criterion"] = "learner_follows_why"
    assert LessonReviewDecision.model_validate(payload).findings[0].criterion == (
        "learner_follows_why"
    )
    payload["findings"][0]["criterion"] = "学生要能跟上"
    with pytest.raises(ValidationError, match="criterion"):
        LessonReviewDecision.model_validate(payload)


def test_simulation_and_review_text_fields_have_explicit_boundaries():
    simulation = prepared_lesson()["simulation_report"]
    simulation["episode_results"][0]["learner_profile"] = "学" * 120
    simulation["episode_results"][0]["evidence"] = ["证" * 800]
    simulation["end_of_lesson_recall"] = "回" * 800
    SimulationReport.model_validate(simulation)

    simulation["episode_results"][0]["evidence"] = ["证" * 1001]
    with pytest.raises(ValidationError):
        SimulationReport.model_validate(simulation)

    review = prepared_lesson()["review"]
    review["findings"][0].update(
        criterion="learner_follows_why",
        evidence="证" * 800,
        requested_change="改" * 800,
    )
    review["approval_summary"] = "结" * 800
    LessonReviewDecision.model_validate(review)
    review["approval_summary"] = "结" * 2_000_000
    with pytest.raises(ValidationError):
        LessonReviewDecision.model_validate(review)


def test_simulation_report_rejects_aggregate_oversize_payload():
    episode_result = prepared_lesson()["simulation_report"][
        "episode_results"
    ][0]
    episode_result["evidence"] = ["证" * 800 for _ in range(16)]
    report = {
        "episode_results": [
            dict(episode_result, episode_id="episode-%d" % index)
            for index in range(20)
        ],
        "end_of_lesson_recall": "回" * 800,
    }

    with pytest.raises(ValidationError, match="serialized byte limit"):
        SimulationReport.model_validate(report)


def test_review_decision_rejects_aggregate_oversize_payload():
    finding = prepared_lesson()["review"]["findings"][0]
    finding.update(
        criterion="learner_follows_why",
        evidence="证" * 800,
        requested_change="改" * 800,
    )
    review = {
        "status": "approved",
        "findings": [
            dict(finding, finding_id="finding-%d" % index)
            for index in range(64)
        ],
        "approval_summary": "结" * 800,
    }

    with pytest.raises(ValidationError, match="serialized byte limit"):
        LessonReviewDecision.model_validate(review)


def test_review_artifact_lists_are_unique():
    review = prepared_lesson()["review"]
    review["findings"][0]["criterion"] = "learner_follows_why"
    review["retained_artifacts"] = ["solution_trace", "solution_trace"]
    with pytest.raises(ValidationError, match="retained artifacts"):
        LessonReviewDecision.model_validate(review)

    review = prepared_lesson()["review"]
    review["findings"][0]["criterion"] = "learner_follows_why"
    review["findings"][0]["invalidated_downstream_artifacts"] = [
        "performance_score",
        "performance_score",
    ]
    with pytest.raises(ValidationError, match="invalidated downstream"):
        LessonReviewDecision.model_validate(review)


def test_prepared_lesson_requires_rubric_and_complete_history():
    payload = prepared_lesson()
    del payload["rubric_version"]
    with pytest.raises(ValidationError):
        PreparedLesson.model_validate(payload)
    payload = prepared_lesson()
    payload["artifact_history"] = payload["artifact_history"][:4]
    with pytest.raises(ValidationError):
        PreparedLesson.model_validate(payload)


def test_prepared_lesson_builder_validates():
    assert PreparedLesson.model_validate(prepared_lesson()).rubric_version == "v1"


def test_simulation_revision_can_record_the_student_simulator_role():
    revision = ArtifactRevision.model_validate(
        {
            "artifact_type": "simulation_report",
            "version": 1,
            "responsible_role": "student_simulator",
        }
    )

    assert revision.responsible_role == "student_simulator"


def test_artifact_revision_rejects_boolean_version_from_raw_mapping():
    with pytest.raises(ValidationError):
        ArtifactRevision.model_validate(
            {
                "artifact_type": "simulation_report",
                "version": True,
                "responsible_role": "student_simulator",
            }
        )


def test_prepared_lesson_rejects_boolean_repair_count_from_raw_mapping():
    payload = prepared_lesson()
    payload["repair_count"] = False

    with pytest.raises(ValidationError):
        PreparedLesson.model_validate(payload)


def test_artifact_revision_cannot_attribute_authoring_to_the_reviewer():
    with pytest.raises(ValidationError):
        ArtifactRevision.model_validate(
            {
                "artifact_type": "simulation_report",
                "version": 1,
                "responsible_role": "lesson_reviewer",
            }
        )


def test_role_call_output_pair_and_nonnegative_token_usage():
    payload = {"role": "lesson_reviewer", "output_artifact_type": "solution_trace", "duration_ms": 0, "retry_count": 0}
    with pytest.raises(ValidationError, match="output type and version"):
        RoleCallRecord.model_validate(payload)
    payload = {"role": "lesson_reviewer", "duration_ms": 0, "retry_count": 0, "token_usage": {"input": -1}}
    with pytest.raises(ValidationError, match="token usage"):
        RoleCallRecord.model_validate(payload)


@pytest.mark.parametrize(
    "token_usage",
    [
        {"api_secret": 1},
        {"prompt_tokens": True},
        {"prompt_tokens": 1_000_000_001},
    ],
)
def test_role_call_token_usage_is_a_bounded_fixed_vocabulary(token_usage):
    with pytest.raises(ValidationError, match="token usage"):
        RoleCallRecord.model_validate(
            {
                "role": "reference_analyst",
                "duration_ms": 0,
                "retry_count": 0,
                "token_usage": token_usage,
            }
        )


def test_role_call_accepts_supported_token_usage_counters():
    record = RoleCallRecord.model_validate(
        {
            "role": "reference_analyst",
            "duration_ms": 0,
            "retry_count": 0,
            "token_usage": {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
                "cached_tokens": 3,
                "reasoning_tokens": 1,
            },
        }
    )

    assert record.token_usage["total_tokens"] == 12


@pytest.mark.parametrize("version", [0, -1])
def test_role_call_input_artifact_versions_must_be_positive(version):
    payload = {
        "role": "lesson_reviewer", "input_artifact_versions": {"solution_trace": version},
        "duration_ms": 0, "retry_count": 0,
    }
    with pytest.raises(ValidationError):
        RoleCallRecord.model_validate(payload)


def test_role_call_rejects_boolean_input_artifact_version_from_raw_mapping():
    payload = {
        "role": "lesson_reviewer",
        "input_artifact_versions": {"solution_trace": True},
        "duration_ms": 0,
        "retry_count": 0,
    }

    with pytest.raises(ValidationError):
        RoleCallRecord.model_validate(payload)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("output_artifact_version", True),
        ("duration_ms", False),
        ("retry_count", True),
    ],
)
def test_role_call_rejects_boolean_private_numeric_fields(field, value):
    payload = {
        "role": "reference_analyst",
        "output_artifact_type": "solution_trace",
        "output_artifact_version": 1,
        "duration_ms": 0,
        "retry_count": 0,
    }
    payload[field] = value

    with pytest.raises(ValidationError):
        RoleCallRecord.model_validate(payload)


def test_interaction_feedback_map_rejects_extra_or_invalid_id_keys():
    payload = interaction()
    payload["incorrect_feedback_by_option"]["option-extra"] = "多余反馈"
    with pytest.raises(ValidationError, match="incorrect feedback"):
        PlannedInteraction.model_validate(payload)
    payload = interaction()
    payload["incorrect_feedback_by_option"] = {"bad key": "反馈", "option-c": "反馈"}
    with pytest.raises(ValidationError):
        PlannedInteraction.model_validate(payload)


def test_full_generation_record_with_seven_role_calls_validates():
    record = GenerationRecord.model_validate({
        "generation_id": "generation-1", "lesson_id": "lesson-1", "route_fingerprint": "route-1",
        "prepared_lesson": prepared_lesson(),
        "role_calls": [
            {
                "role": role, "input_artifact_versions": {"solution_trace": 1},
                "output_artifact_type": "solution_trace", "output_artifact_version": 1,
                "duration_ms": 1, "retry_count": 0,
            }
            for role in [
                "reference_analyst", "teaching_designer", "script_teacher", "interaction_designer",
                "classroom_director", "student_simulator", "lesson_reviewer",
            ]
        ],
        "cue_provenance": [
            {
                "episode_id": "episode-1",
                "clause_id": "open-1",
                "original_performance_cue_id": "cue-1",
                "runtime_cue_id": "runtime-cue-1",
                "spoken_text": "我们把等式两边同时减一。",
            }
        ],
        "created_at": "2026-08-11T10:00:00+08:00",
    })
    assert len(record.role_calls) == 7
    assert record.cue_provenance[0].clause_id == "open-1"
