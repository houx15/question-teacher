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
    TeachingProgression,
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
from app.reference_safety import ReferenceSafetyPolicy
from app.schemas import (
    ProblemFocusTarget,
    ProblemInput,
    ReferenceGroundingBrief,
)
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
    "x^2-2mx+2n=0",
    "x=2n",
    "m-n",
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
            "assumptions": [
                {
                    "assumption_id": "assumption-nonzero",
                    "expression": "n!=0",
                    "source_kind": "problem",
                },
                {
                    "assumption_id": "assumption-root",
                    "expression": "x=2n",
                    "source_kind": "problem",
                },
            ],
            "reference_conclusion": final_conclusion,
            "method_name": "代入法",
            "reasoning_steps": [
                {
                    "step_id": step_id,
                    "statement_before": (
                        "x^2-2mx+2n=0"
                        if index == 0
                        else STATES[index - 1]
                    ),
                    "operation_kind": (
                        "identify" if index in {0, 2, 4}
                        else "substitute" if index == 1
                        else "expand" if index == 3
                        else "divide" if index == 5
                        else "rearrange"
                    ),
                    "operands": (["2n"] if index in {1, 5} else []),
                    "statement_after": (
                        final_conclusion
                        if index == len(STEP_IDS) - 1
                        else state
                    ),
                    "assumption_ids_used": (
                        ["assumption-nonzero"]
                        if step_id == "use-nonzero"
                        else ["assumption-root"]
                        if step_id == "substitute-root"
                        else []
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
        "task_target": "m-n",
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
            },
            {
                "assumption_id": "assumption-root",
                "content": "x=2n",
                "source_anchor": {
                    "source_kind": "problem",
                    "source_id": "problem-root",
                    "excerpt": "2n是根",
                },
            },
        ],
        "source_steps": [
            {
                "source_step_id": step_id,
                "source_anchor": {
                    "source_kind": "verified_route",
                    "source_id": step_id,
                    "excerpt": "冻结路线步骤",
                },
                "state_before": (
                    "x^2-2mx+2n=0"
                    if index == 0
                    else STATES[index - 1]
                ),
                "operation_kind": (
                    "identify" if index in {0, 2, 4}
                    else "substitute" if index == 1
                    else "expand" if index == 3
                    else "divide" if index == 5
                    else "rearrange"
                ),
                "operands": (["2n"] if index in {1, 5} else []),
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
                    ["assumption-nonzero"]
                    if step_id == "use-nonzero"
                    else ["assumption-root"]
                    if step_id == "substitute-root"
                    else []
                ),
                "reasoning_gap_codes": [],
                "evidence_status": "reference_only",
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
        "transition_reason": (
            "根据前一步的结果，必须确认当前条件后再推进目标关系。"
        ),
        "must_teach": [
            {
                "must_teach_id": "must-%d" % (index + 1),
                "content": "解释%s" % step_id,
                "why_it_matters": "学生需要理解依赖",
                "student_display_evidence": "解释当前这一步为什么成立",
                "student_spoken_evidence": "我们解释当前这一步为什么成立。",
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


def progression_payload():
    trajectory = trajectory_payload()
    return {
        "steps": [
            {
                "step_id": "teaching-step-%d" % (index + 1),
                "sequence_index": index,
                "episode_ids": [episode["episode_id"]],
                "phase": "construct" if index == 0 else "execute",
                "student_problem": episode["thinking_question"],
                "why_now": episode["transition_reason"],
                "evidence_target_ids": [],
                "guiding_questions": [episode["thinking_question"]],
                "knowledge_anchor": episode["decision_reason"],
                "checkpoint": (
                    {
                        "diagnostic_goal": "检查学生是否知道先代入已知根",
                        "misconception_ids": [],
                    }
                    if index == 1
                    else None
                ),
                "reveal": episode["decision"],
                "math_action": episode["mathematical_action"],
                "directory_question": episode["thinking_question"],
                "directory_label": "第%d步：处理当前问题" % (index + 1),
                "board_summary": ["由当前条件可推出：%s" % episode["result"]],
                "error_tip": "注意条件使用范围",
                "transition_question": episode["transition_reason"],
                "must_teach_refs": [
                    item["must_teach_id"] for item in episode["must_teach"]
                ],
            }
            for index, episode in enumerate(trajectory["episodes"])
        ]
    }


def clause_payload(index, clause_id=None, episode_id=None, math_reference=None):
    return {
        "clause_id": clause_id or "clause-%d" % (index + 1),
        "episode_id": episode_id or "episode-%d" % (index + 1),
        "lesson_step_id": "teaching-step-%d" % (index + 1),
        "pedagogical_function": "explain",
        "display_text": "解释当前这一步为什么成立",
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
    seen_steps = set()
    for clause in clauses:
        if clause["lesson_step_id"] not in seen_steps:
            clause["pedagogical_function"] = "question"
            seen_steps.add(clause["lesson_step_id"])
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
        "response_scripts": [
            {
                "response_id": "response-%s" % option_id,
                "interaction_id": "interaction-1",
                "option_id": option_id,
                "classification": (
                    "correct" if option_id == "option-a" else "incorrect"
                ),
                "error_code": (
                    None if option_id == "option-a" else "%s-error" % option_id
                ),
                "depth": (
                    "brief" if option_id == "option-a" else "conceptual"
                ),
                "clauses": [
                    {
                        "clause_id": "response-clause-%s" % option_id,
                        "episode_id": "episode-2",
                        "lesson_step_id": "teaching-step-2",
                        "pedagogical_function": (
                            "transition" if option_id == "option-a" else "correct"
                        ),
                        "display_text": (
                            "判断正确"
                            if option_id == "option-a"
                            else (
                                "偏离目标；目标只是关系"
                                if option_id == "option-b"
                                else "未用已知；先用根条件"
                            )
                        ),
                        "spoken_text": (
                            "对。"
                            if option_id == "option-a"
                            else (
                                "这个选择偏离目标，目标只是关系，再回到根条件判断。"
                                if option_id == "option-b"
                                else "这个选择未用已知，先用根条件，再继续判断。"
                            )
                        ),
                        "learner_gain": (
                            "确认判断" if option_id == "option-a" else "理解错误原因"
                        ),
                        "answer_exposure": False,
                        "must_teach_refs": [],
                    }
                ],
            }
            for option_id in ("option-a", "option-b", "option-c")
        ],
        "interaction_scripts": [
            {
                "interaction_id": "interaction-1",
                "prompt": "下一步应处理哪个已知条件？",
                "hint": "想想根的定义。",
                "options": [
                    {"option_id": "option-a", "label": "代入已知根"},
                    {"option_id": "option-b", "label": "分别求m和n"},
                    {"option_id": "option-c", "label": "忽略根条件"},
                ],
            }
        ],
        "transfer_script": {
            "problem_text": "另一题仍用参数根求关系",
            "method_signal": "先代入已知根",
            "options": [
                {"option_id": "transfer-a", "label": r"\(p-q=\frac{1}{2}\)", "feedback": "正确"},
                {"option_id": "transfer-b", "label": r"\(p-q=2\)", "feedback": "检查除法"},
                {"option_id": "transfer-c", "label": r"\(p+q=\frac{1}{2}\)", "feedback": "检查目标"},
            ],
        },
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
                "teaching_step_id": "teaching-step-2",
                "after_clause_id": "clause-2",
                "why_pause": "在这里停下是为了检查学生是否知道先代入已知根。",
                "diagnostic_target": "是否知道代入已知根",
                "diagnostic_kind": "conception",
                "prompt": "下一步应处理哪个已知条件？",
                "options": [
                    {"option_id": "option-a", "display_text": "代入已知根", "canonical_answer": "substitute-root"},
                    {"option_id": "option-b", "display_text": "分别求m和n", "canonical_answer": "solve-separately", "misconception": "偏离目标", "error_code": "option-b-error", "remediation_depth": "conceptual"},
                    {"option_id": "option-c", "display_text": "忽略根条件", "canonical_answer": "ignore-root", "misconception": "未用已知", "error_code": "option-c-error", "remediation_depth": "conceptual"},
                ],
                "correct_option_id": "option-a",
                "correct_feedback": "对，根一定满足原方程。",
                "incorrect_feedback_by_option": {"option-b": "目标只是关系。", "option-c": "先用根条件。"},
                "hint": "想想根的定义。",
                "resume_clause_id": "clause-2-resume",
                "resume_step_id": "teaching-step-2",
                "resume_policy": "continue",
                "concealed_targets": [],
            }
        ],
        "transfer_item": transfer_payload(),
    }


def structured_score_payload():
    script = script_payload()
    progression = progression_payload()
    step_labels = {
        step["step_id"]: step["directory_label"]
        for step in progression["steps"]
    }
    clauses_by_step = {}
    for clause in script["clauses"]:
        clauses_by_step.setdefault(clause["lesson_step_id"], []).append(
            clause["clause_id"]
        )
    board_objects = []
    cues = []
    for clause in script["clauses"]:
        step_id = clause["lesson_step_id"]
        target = "board-%s" % clause["clause_id"]
        role = (
            "knowledge_anchor"
            if clauses_by_step[step_id][0] == clause["clause_id"]
            else "working"
        )
        board_objects.append(
            {
                "board_object_id": target,
                "content": clause["math_references"][0],
                "teaching_step_id": step_id,
                "line_role": role,
            }
        )
        end_actions = []
        if clauses_by_step[step_id][0] == clause["clause_id"]:
            end_actions.extend(
                [
                    {
                        "clause_id": clause["clause_id"],
                        "action": {
                            "surface": "board",
                            "type": "reveal_step_header",
                            "target": step_id,
                            "teaching_step_id": step_id,
                            "step_label": step_labels[step_id],
                        },
                    },
                    {
                        "clause_id": clause["clause_id"],
                        "action": {
                            "surface": "board",
                            "type": "scroll_to_step",
                            "target": step_id,
                            "teaching_step_id": step_id,
                        },
                    },
                ]
            )
        end_actions.append(
            {
                "clause_id": clause["clause_id"],
                "action": {
                    "surface": "board",
                    "type": "write",
                    "target": target,
                    "content": clause["math_references"][0],
                    "teaching_step_id": step_id,
                    "board_role": role,
                },
            }
        )
        if clauses_by_step[step_id][-1] == clause["clause_id"]:
            end_actions.append(
                {
                    "clause_id": clause["clause_id"],
                    "action": {
                        "surface": "board",
                        "type": "complete_step",
                        "target": step_id,
                        "teaching_step_id": step_id,
                    },
                }
            )
        cues.append(
            {
                "cue_id": "cue-%s" % clause["clause_id"],
                "clause_ids": [clause["clause_id"]],
                "end_actions": end_actions,
            }
        )
    for response in script["response_scripts"]:
        for clause in response["clauses"]:
            cue = {
                "cue_id": "cue-%s" % clause["clause_id"],
                "clause_ids": [clause["clause_id"]],
            }
            if response["classification"] == "incorrect":
                step_id = clause["lesson_step_id"]
                target = "support-%s" % response["option_id"]
                board_objects.append(
                    {
                        "board_object_id": target,
                        "content": clause["display_text"],
                        "teaching_step_id": step_id,
                        "line_role": "support",
                    }
                )
                support_target = "support-panel-%s" % response["option_id"]
                cue["start_actions"] = [
                    {
                        "clause_id": clause["clause_id"],
                        "action": {
                            "surface": "board",
                            "type": "open_supporting_explanation",
                            "target": support_target,
                            "teaching_step_id": step_id,
                        },
                    },
                    {
                        "clause_id": clause["clause_id"],
                        "action": {
                            "surface": "board",
                            "type": "write",
                            "target": target,
                            "content": clause["display_text"],
                            "teaching_step_id": step_id,
                            "board_role": "support",
                        },
                    },
                ]
                cue["end_actions"] = [
                    {
                        "clause_id": clause["clause_id"],
                        "action": {
                            "surface": "board",
                            "type": "close_supporting_explanation",
                            "target": support_target,
                            "teaching_step_id": step_id,
                        },
                    },
                    {
                        "clause_id": clause["clause_id"],
                        "action": {
                            "surface": "board",
                            "type": "scroll_to_step",
                            "target": step_id,
                            "teaching_step_id": step_id,
                        },
                    },
                ]
            cues.append(cue)
    return {"cues": cues, "board_objects": board_objects, "overlay_transitions": []}


def score_payload():
    script = script_payload()
    board_objects = [
        {
            "board_object_id": "board-%d" % (index + 1),
            "content": STATES[index],
        }
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
    return {
        "cues": cues,
        "board_objects": board_objects,
        "overlay_transitions": [],
    }


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
                "can_align_display_and_spoken_math": True,
                "can_recover_with_adaptive_support": True,
                "can_locate_current_step": True,
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
                "criterion": "learner_follows_why",
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
        "teaching_progression": progression_payload(),
        "interaction_plan": interaction_plan_payload(),
        "teaching_script": script_payload(),
        "performance_score": structured_score_payload(),
        "simulation_report": simulation_payload(),
        "review": review_payload(),
        "repair_count": 0,
        "artifact_history": [
            {"artifact_type": artifact, "version": 1, "responsible_role": role}
            for artifact, role in (
                ("solution_trace", "reference_analyst"),
                ("reasoning_trajectory", "teaching_designer"),
                ("teaching_progression", "teaching_designer"),
                ("interaction_plan", "interaction_designer"),
                ("teaching_script", "script_teacher"),
                ("performance_score", "classroom_director"),
                ("simulation_report", "student_simulator"),
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
        PerformanceScore.model_validate(score_payload()),
        SimulationReport.model_validate(payload["simulation_report"]),
        LessonReviewDecision.model_validate(payload["review"]),
    )


def validate_current_script(script, trajectory):
    validate_teaching_script(
        script,
        trajectory,
        TeachingProgression.model_validate(progression_payload()),
        InteractionPlan.model_validate(interaction_plan_payload()),
    )


def semantically_anchored_script_payload():
    payload = script_payload()
    anchors = {
        "option-b": ("偏离目标", "目标只是关系"),
        "option-c": ("未用已知", "先用根条件"),
    }
    for response in payload["response_scripts"]:
        if response["option_id"] not in anchors:
            continue
        misconception, correction = anchors[response["option_id"]]
        clause = response["clauses"][0]
        clause["display_text"] = "%s；%s" % (misconception, correction)
        clause["spoken_text"] = (
            "这个选择的问题是%s，%s，再继续判断。"
            % (misconception, correction)
        )
    return payload


def response_anchor_case(misconception, correction, spoken_text):
    script = semantically_anchored_script_payload()
    plan = interaction_plan_payload()
    plan["interactions"][0]["options"][1]["misconception"] = misconception
    plan["interactions"][0]["incorrect_feedback_by_option"][
        "option-b"
    ] = correction
    response = script["response_scripts"][1]
    response["clauses"][0]["display_text"] = "检查当前选择"
    response["clauses"][0]["spoken_text"] = spoken_text
    return script, plan


def test_current_rubric_prepared_lesson_requires_progression_and_seven_history_items():
    payload = prepared_payload()
    payload["teaching_progression"] = None
    payload["artifact_history"] = [
        item
        for item in payload["artifact_history"]
        if item["artifact_type"] != "teaching_progression"
    ]

    assert_history_invalid(payload)


def test_current_rubric_prepared_lesson_runs_progression_semantic_validation():
    payload = prepared_payload()
    payload["teaching_progression"]["steps"][0]["evidence_target_ids"] = [
        "target-not-provided"
    ]
    prepared = PreparedLesson.model_validate(payload)

    with pytest.raises(PreparationValidationError) as captured:
        validate_prepared_lesson(
            prepared,
            route(),
            [
                ProblemFocusTarget(
                    target_id="problem-root",
                    math_text="2n",
                    ordinal=1,
                )
            ],
        )

    assert captured.value.code == "progression_evidence_target_invalid"
    assert captured.value.artifact_id == "teaching-step-1"


def test_review_dependency_uses_artifact_order_for_progression_interaction_and_script():
    trace, trajectory, script, plan, score, report, _ = models()
    progression = TeachingProgression.model_validate(progression_payload())
    progression_finding = {
        "finding_id": "finding-progression",
        "severity": "material",
        "artifact_type": "teaching_progression",
        "artifact_id": "teaching-step-1",
        "criterion": "learner_follows_why",
        "evidence": "推进缺少为什么此刻处理",
        "responsible_role": "teaching_designer",
        "requested_change": "补充 why_now",
        "invalidated_downstream_artifacts": [
            "interaction_plan",
            "teaching_script",
            "performance_score",
            "simulation_report",
        ],
    }
    decision = LessonReviewDecision.model_validate(
        {
            "status": "revision_required",
            "findings": [progression_finding],
            "retained_artifacts": [
                "solution_trace",
                "reasoning_trajectory",
            ],
            "approval_summary": "从教学推进开始修订",
        }
    )

    validate_review_decision(
        decision,
        trace,
        trajectory,
        script,
        plan,
        score,
        report,
        progression=progression,
    )

    interaction_payload = decision.model_dump(mode="python")
    interaction_payload["findings"][0].update(
        finding_id="finding-interaction",
        artifact_type="interaction_plan",
        artifact_id="interaction-1",
        responsible_role="interaction_designer",
        invalidated_downstream_artifacts=[
            "teaching_script",
            "performance_score",
            "simulation_report",
        ],
    )
    interaction_payload["retained_artifacts"] = [
        "solution_trace",
        "reasoning_trajectory",
        "teaching_progression",
    ]
    validate_review_decision(
        LessonReviewDecision.model_validate(interaction_payload),
        trace,
        trajectory,
        script,
        plan,
        score,
        report,
        progression=progression,
    )


def assert_code(code, call):
    with pytest.raises(PreparationValidationError) as error:
        call()
    assert error.value.code == code
    assert error.value.artifact_id
    assert error.value.detail
    assert "冻结路线步骤" not in error.value.detail


def test_parameter_root_full_traceability_matrix_validates():
    payload = prepared_payload()
    payload["teaching_progression"]["steps"][0]["evidence_target_ids"] = [
        "problem-root"
    ]
    prepared = PreparedLesson.model_validate(payload)
    targets = [
        ProblemFocusTarget(
            target_id="problem-root",
            math_text="2n",
            ordinal=1,
        )
    ]
    validate_prepared_lesson(prepared, route(), targets)


@pytest.mark.parametrize(
    ("mutation", "code"),
    (
        ("missing_display_evidence", "must_teach_evidence_missing"),
        ("missing_interaction_script", "interaction_script_coverage_invalid"),
        ("missing_transfer_script", "transfer_script_missing"),
    ),
)
def test_current_rubric_fails_closed_on_new_student_evidence_ownership(
    mutation,
    code,
):
    payload = prepared_payload()
    if mutation == "missing_display_evidence":
        payload["reasoning_trajectory"]["episodes"][0]["must_teach"][0].pop(
            "student_display_evidence"
        )
    elif mutation == "missing_interaction_script":
        payload["teaching_script"]["interaction_scripts"] = []
    else:
        payload["teaching_script"]["transfer_script"] = None
    prepared = PreparedLesson.model_validate(payload)

    assert_code(
        code,
        lambda: validate_prepared_lesson(prepared, route(), []),
    )


@pytest.mark.parametrize(
    ("field", "unsafe_value", "code"),
    (
        ("problem_text", "{{highlight target}}", "choice_formula_invalid"),
        ("method_signal", r"先看 \(m-n\)", "transfer_script_content_invalid"),
        ("feedback", r"所以 \(m-n=1\)", "transfer_script_content_invalid"),
    ),
)
def test_current_transfer_script_rejects_unsafe_public_text(
    field,
    unsafe_value,
    code,
):
    payload = prepared_payload()
    transfer = payload["teaching_script"]["transfer_script"]
    if field == "feedback":
        transfer["options"][0][field] = unsafe_value
    else:
        transfer[field] = unsafe_value
    prepared = PreparedLesson.model_validate(payload)

    assert_code(
        code,
        lambda: validate_prepared_lesson(prepared, route(), []),
    )


@pytest.mark.parametrize(
    "ability",
    (
        "can_identify_attention_target",
        "can_explain_decision",
        "can_execute_action",
        "can_use_result_to_continue",
        "can_align_display_and_spoken_math",
        "can_recover_with_adaptive_support",
        "can_locate_current_step",
    ),
)
def test_current_rubric_review_cannot_approve_failed_simulation_ability(
    ability,
):
    payload = prepared_payload()
    payload["simulation_report"]["episode_results"][0][ability] = False
    prepared = PreparedLesson.model_validate(payload)

    assert_code(
        "review_non_compensable_gate_invalid",
        lambda: validate_prepared_lesson(prepared, route(), []),
    )


@pytest.mark.parametrize(
    "ability",
    (
        "can_align_display_and_spoken_math",
        "can_recover_with_adaptive_support",
        "can_locate_current_step",
    ),
)
def test_current_rubric_requires_structured_simulation_abilities(ability):
    payload = prepared_payload()
    payload["simulation_report"]["episode_results"][0].pop(ability)
    prepared = PreparedLesson.model_validate(payload)

    assert_code(
        "simulation_structured_ability_missing",
        lambda: validate_prepared_lesson(prepared, route(), []),
    )


def test_structured_performance_covers_main_and_response_lifecycles():
    payload = prepared_payload()
    validate_performance_score(
        PerformanceScore.model_validate(payload["performance_score"]),
        [],
        TeachingProgression.model_validate(payload["teaching_progression"]),
        TeachingScript.model_validate(payload["teaching_script"]),
        InteractionPlan.model_validate(payload["interaction_plan"]),
    )


@pytest.mark.parametrize(
    "mutation",
    ("missing_reveal", "duplicate_complete", "wrong_step_line", "unclosed_support"),
)
def test_structured_performance_rejects_invalid_lifecycle(mutation):
    payload = prepared_payload()
    score = payload["performance_score"]
    if mutation == "missing_reveal":
        cue = score["cues"][0]
        cue["end_actions"] = [
            item
            for item in cue["end_actions"]
            if item["action"]["type"] != "reveal_step_header"
        ]
    elif mutation == "duplicate_complete":
        cue = score["cues"][0]
        cue["end_actions"].append(
            copy.deepcopy(
                next(
                    item
                    for item in cue["end_actions"]
                    if item["action"]["type"] == "complete_step"
                )
            )
        )
    elif mutation == "wrong_step_line":
        score["board_objects"][0]["teaching_step_id"] = "teaching-step-2"
    else:
        response_cue = next(
            cue
            for cue in score["cues"]
            if cue["clause_ids"] == ["response-clause-option-b"]
        )
        response_cue["end_actions"] = [
            item
            for item in response_cue["end_actions"]
            if item["action"]["type"] != "close_supporting_explanation"
        ]

    with pytest.raises(PreparationValidationError):
        validate_performance_score(
            PerformanceScore.model_validate(score),
            [],
            TeachingProgression.model_validate(payload["teaching_progression"]),
            TeachingScript.model_validate(payload["teaching_script"]),
            InteractionPlan.model_validate(payload["interaction_plan"]),
        )


def test_text_only_geometry_trace_and_route_cross_boundary_safely():
    source = ProblemInput(
        problem_text="已知AB=AC，求角A。",
        reference_answer=r"\angle A=60^\circ",
        reference_solution_text=(
            r"AB=AC，依据已核对的几何关系可得\angle A=60^\circ"
        ),
    )
    policy = ReferenceSafetyPolicy.from_problem(source)
    brief = ReferenceGroundingBrief.validate_for_reference_answer(
        {
            "task_summary": "由等腰三角形条件确定角",
            "target": r"\angle A",
            "assumptions": [
                {"assumption_id": "equal-sides", "expression": "AB=AC"}
            ],
            "reference_conclusion": r"\angle A=60^\circ",
            "method_name": "几何关系",
            "reasoning_steps": [
                {
                    "step_id": "geometry-step",
                    "statement_before": "AB=AC",
                    "operation_kind": "derive",
                    "operands": [],
                    "statement_after": r"\angle A=60^\circ",
                    "assumption_ids_used": ["equal-sides"],
                }
            ],
            "check_requests": [],
            "audit_notes": [],
        },
        r"\angle A=60^\circ",
    )
    brief = policy.sanitize_grounding_brief(
        brief,
        source.reference_answer,
    )
    geometry_route = freeze_grounded_route(brief, [])
    trace = SolutionTrace.model_validate(
        {
            "task_target": r"\angle A",
            "reference_conclusion": r"\angle A=60^\circ",
            "assumptions": [
                {
                    "assumption_id": "ground-assumption-001",
                    "content": "AB=AC",
                    "source_anchor": {
                        "source_kind": "problem",
                        "source_id": "problem-equal-sides",
                        "excerpt": "题目结构依据",
                    },
                }
            ],
            "source_steps": [
                {
                    "source_step_id": "ground-step-001",
                    "source_anchor": {
                        "source_kind": "verified_route",
                        "source_id": "ground-step-001",
                        "excerpt": "已验证路线结构依据",
                    },
                    "state_before": "AB=AC",
                    "operation_kind": "derive",
                    "operands": [],
                    "mathematical_action": "依据等腰关系推导",
                    "justification": "几何条件支持",
                    "state_after": r"\angle A=60^\circ",
                    "new_information": r"\angle A=60^\circ",
                    "assumption_ids_used": ["ground-assumption-001"],
                    "reasoning_gap_codes": [],
                    "evidence_status": "reference_only",
                }
            ],
            "audit_notes": [],
        }
    )

    trace = policy.sanitize_solution_trace(trace, geometry_route)
    validate_solution_trace(trace, geometry_route)
    assert geometry_route.to_prompt_payload()["steps"][0][
        "statement_before"
    ] == "AB=AC"


def test_solution_trace_rejects_conclusion_mismatch_and_missing_assumption():
    payload = trace_payload()
    payload["reference_conclusion"] = "m-n=2"
    trace = SolutionTrace.model_validate(payload)
    assert_code("trace_conclusion_mismatch", lambda: validate_solution_trace(trace, route()))
    payload = trace_payload()
    payload["source_steps"][0]["assumption_ids_used"] = ["assumption-missing"]
    trace = SolutionTrace.model_validate(payload)
    assert_code("trace_assumption_missing", lambda: validate_solution_trace(trace, route()))


@pytest.mark.parametrize(
    "tamper",
    [
        lambda step: step.update(operation_kind="factor", operands=["999"]),
        lambda step: step.update(assumption_ids_used=[]),
        lambda step: step.update(evidence_status="quoted"),
        lambda step: step["source_anchor"].update(source_kind="solution"),
        lambda step: step["source_anchor"].update(source_id="return-target"),
        lambda step: step.update(reasoning_gap_codes=["implicit_identity"]),
    ],
)
def test_solution_trace_rejects_typed_decision_and_provenance_tampering(
    tamper,
):
    payload = trace_payload()
    step = payload["source_steps"][1]
    tamper(step)
    invalid = SolutionTrace.model_validate(payload)

    with pytest.raises(PreparationValidationError):
        validate_solution_trace(invalid, route())


def test_reasoning_trajectory_must_resolve_every_selected_trace_gap():
    trace_data = trace_payload()
    trace_data["source_steps"][1]["reasoning_gap_codes"] = [
        "implicit_substitution"
    ]
    trace = SolutionTrace.model_validate(trace_data)
    trajectory = ReasoningTrajectory.model_validate(trajectory_payload())

    assert_code(
        "trajectory_gap_unresolved",
        lambda: validate_reasoning_trajectory(trajectory, trace),
    )

    trajectory_data = trajectory_payload()
    trajectory_data["episodes"][1]["resolved_gap_refs"] = [
        {
            "source_step_id": "substitute-root",
            "gap_code": "implicit_substitution",
            "must_teach_id": "must-2",
        }
    ]
    resolved = ReasoningTrajectory.model_validate(trajectory_data)
    validate_reasoning_trajectory(resolved, trace)


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
                    "statement_before": "x=12",
                    "operation_kind": "compare",
                    "operands": ["x={1,2}"],
                    "statement_after": "x=12",
                }
            ],
            "check_requests": [],
            "audit_notes": [],
        },
        "x=12",
    )
    payload = trace_payload()
    payload["task_target"] = "x"
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
        r"\(m−n=\tfrac{1}{2}\)"
    )
    validate_solution_trace(SolutionTrace.model_validate(payload), route())

    payload["reference_conclusion"] = r"m-n=\frac{-1}{2}"
    payload["source_steps"][-1]["state_after"] = "m-n=-1/2"
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
    assert_code("clause_episode_missing", lambda: validate_current_script(invalid, trajectory))
    payload = script.model_dump()
    payload["clauses"][0]["must_teach_refs"] = []
    invalid = TeachingScript.model_validate(payload)
    assert_code("must_teach_uncovered", lambda: validate_current_script(invalid, trajectory))


def test_teaching_script_rejects_invalid_or_cross_episode_must_teach_reference():
    _, trajectory, script, *_ = models()
    payload = script.model_dump()
    payload["clauses"][0]["must_teach_refs"] = ["must-missing"]
    invalid = TeachingScript.model_validate(payload)
    assert_code("must_teach_ref_invalid", lambda: validate_current_script(invalid, trajectory))
    payload = script.model_dump()
    payload["clauses"][0]["must_teach_refs"] = ["must-2"]
    invalid = TeachingScript.model_validate(payload)
    assert_code("must_teach_ref_invalid", lambda: validate_current_script(invalid, trajectory))


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
        lambda: validate_current_script(script, trajectory),
    )


def test_teaching_script_rejects_episode_reordering_and_spoken_markup():
    _, trajectory, script, *_ = models()
    payload = script.model_dump()
    payload["clauses"][0]["episode_id"], payload["clauses"][1]["episode_id"] = "episode-2", "episode-1"
    payload["clauses"][0]["lesson_step_id"], payload["clauses"][1]["lesson_step_id"] = "teaching-step-2", "teaching-step-1"
    payload["clauses"][0]["must_teach_refs"], payload["clauses"][1]["must_teach_refs"] = ["must-2"], ["must-1"]
    invalid = TeachingScript.model_validate(payload)
    assert_code("clause_episode_order_invalid", lambda: validate_current_script(invalid, trajectory))
    payload = script.model_dump()
    payload["clauses"][0]["spoken_text"] = "得到 $x=2n$。"
    invalid = TeachingScript.model_validate(payload)
    assert_code("spoken_markup_invalid", lambda: validate_current_script(invalid, trajectory))


def test_current_teaching_script_requires_step_and_display_on_every_clause():
    _, trajectory, script, *_ = models()
    validate_current_script(script, trajectory)

    for field, code in (
        ("lesson_step_id", "clause_lesson_step_missing"),
        ("display_text", "clause_display_missing"),
    ):
        payload = script.model_dump()
        payload["clauses"][0][field] = None
        assert_code(
            code,
            lambda payload=payload: validate_current_script(
                TeachingScript.model_validate(payload), trajectory
            ),
        )

        payload = script.model_dump()
        payload["response_scripts"][0]["clauses"][0][field] = None
        assert_code(
            code,
            lambda payload=payload: validate_current_script(
                TeachingScript.model_validate(payload), trajectory
            ),
        )


def test_current_teaching_script_requires_ordered_progression_coverage_and_episode_binding():
    _, trajectory, script, *_ = models()
    payload = script.model_dump()
    payload["clauses"][0]["lesson_step_id"] = "teaching-step-2"
    invalid = TeachingScript.model_validate(payload)
    assert_code(
        "clause_episode_step_mismatch",
        lambda: validate_current_script(invalid, trajectory),
    )

    progression = TeachingProgression.model_validate(progression_payload())
    progression_payload_value = progression.model_dump()
    progression_payload_value["steps"].append(
        dict(
            progression_payload_value["steps"][-1],
            step_id="teaching-step-uncovered",
            sequence_index=len(progression_payload_value["steps"]),
        )
    )
    assert_code(
        "progression_step_uncovered",
        lambda: validate_teaching_script(
            script,
            trajectory,
            TeachingProgression.model_validate(progression_payload_value),
            InteractionPlan.model_validate(interaction_plan_payload()),
        ),
    )


def test_response_scripts_cover_each_interaction_option_exactly_once():
    _, trajectory, script, *_ = models()
    payload = script.model_dump()
    payload["response_scripts"].pop()
    assert_code(
        "response_script_coverage_invalid",
        lambda: validate_current_script(
            TeachingScript.model_validate(payload), trajectory
        ),
    )

    payload = script.model_dump()
    payload["response_scripts"][1]["interaction_id"] = "interaction-missing"
    assert_code(
        "response_script_binding_invalid",
        lambda: validate_current_script(
            TeachingScript.model_validate(payload), trajectory
        ),
    )


def test_response_scripts_enforce_classification_depth_and_correct_error_contract():
    _, trajectory, script, *_ = models()
    cases = (
        (0, "classification", "incorrect", "response_classification_invalid"),
        (0, "depth", "conceptual", "response_depth_invalid"),
        (0, "error_code", "invented-error", "response_error_code_invalid"),
        (1, "classification", "correct", "response_classification_invalid"),
        (1, "depth", "brief", "response_depth_invalid"),
        (1, "depth", "worked", "response_depth_invalid"),
        (1, "error_code", "other-error", "response_error_code_invalid"),
    )
    for index, field, value, code in cases:
        payload = script.model_dump()
        payload["response_scripts"][index][field] = value
        assert_code(
            code,
            lambda payload=payload: validate_current_script(
                TeachingScript.model_validate(payload), trajectory
            ),
        )


def test_incorrect_response_depth_depends_on_distinct_semantic_units_not_length():
    _, trajectory, script, *_ = models()
    payload = script.model_dump()
    correct_clause = payload["response_scripts"][0]["clauses"][0]
    correct_clause["display_text"] = "请回到根的定义检查这一步的条件"
    correct_clause["spoken_text"] = "请回到根的定义，检查这一步是否真的用上了已知条件。"
    wrong_clause = payload["response_scripts"][1]["clauses"][0]
    wrong_clause["display_text"] = "偏离目标；目标只是关系"
    wrong_clause["spoken_text"] = "偏离目标，目标只是关系。"

    validate_current_script(
        TeachingScript.model_validate(payload), trajectory
    )


def test_incorrect_response_rejects_identical_semantic_units_despite_filler():
    _, trajectory, _, *_ = models()
    payload = semantically_anchored_script_payload()
    plan_payload = interaction_plan_payload()
    plan_payload["interactions"][0]["options"][1]["misconception"] = "回到目标"
    plan_payload["interactions"][0]["incorrect_feedback_by_option"][
        "option-b"
    ] = "回到目标"
    response = payload["response_scripts"][1]
    response["clauses"][0]["display_text"] = "回到目标"
    response["clauses"][0]["spoken_text"] = "回到目标" + "啊" * 60

    assert_code(
        "response_remediation_insufficient",
        lambda: validate_teaching_script(
            TeachingScript.model_validate(payload),
            trajectory,
            TeachingProgression.model_validate(progression_payload()),
            InteractionPlan.model_validate(plan_payload),
        ),
    )


def test_incorrect_response_accepts_distinct_semantic_units_with_filler():
    _, trajectory, _, *_ = models()
    payload = semantically_anchored_script_payload()
    payload["response_scripts"][1]["clauses"][0]["spoken_text"] += (
        "啊" * 40
    )

    validate_current_script(
        TeachingScript.model_validate(payload), trajectory
    )


def test_incorrect_response_rejects_one_nested_anchor_occurrence():
    _, trajectory, _, *_ = models()
    script, plan = response_anchor_case(
        "偏离目标",
        "偏离目标关系",
        "这里只说了偏离目标关系。",
    )

    assert_code(
        "response_semantic_anchor_missing",
        lambda: validate_teaching_script(
            TeachingScript.model_validate(script),
            trajectory,
            TeachingProgression.model_validate(progression_payload()),
            InteractionPlan.model_validate(plan),
        ),
    )


def test_incorrect_response_rejects_fused_overlapping_anchor_occurrences():
    _, trajectory, _, *_ = models()
    script, plan = response_anchor_case(
        "目标关系",
        "关系错误",
        "这里只说了目标关系错误。",
    )

    assert_code(
        "response_semantic_anchor_missing",
        lambda: validate_teaching_script(
            TeachingScript.model_validate(script),
            trajectory,
            TeachingProgression.model_validate(progression_payload()),
            InteractionPlan.model_validate(plan),
        ),
    )


def test_incorrect_response_accepts_separate_natural_anchor_occurrences():
    _, trajectory, _, *_ = models()
    script, plan = response_anchor_case(
        "偏离目标",
        "回到目标关系",
        "这个选择偏离目标，需要回到目标关系重新检查。",
    )

    validate_teaching_script(
        TeachingScript.model_validate(script),
        trajectory,
        TeachingProgression.model_validate(progression_payload()),
        InteractionPlan.model_validate(plan),
    )


def test_incorrect_response_accepts_contained_anchor_when_repeated_separately():
    _, trajectory, _, *_ = models()
    script, plan = response_anchor_case(
        "偏离目标",
        "偏离目标关系",
        "这里是偏离目标关系，另外说明它确实偏离目标。",
    )

    validate_teaching_script(
        TeachingScript.model_validate(script),
        trajectory,
        TeachingProgression.model_validate(progression_payload()),
        InteractionPlan.model_validate(plan),
    )


def test_incorrect_response_deduplicates_identical_fused_display_and_spoken_units():
    _, trajectory, _, *_ = models()
    script, plan = response_anchor_case(
        "目标关系",
        "关系错误",
        "目标关系错误",
    )
    script["response_scripts"][1]["clauses"][0][
        "display_text"
    ] = "目标关系错误"

    assert_code(
        "response_semantic_anchor_missing",
        lambda: validate_teaching_script(
            TeachingScript.model_validate(script),
            trajectory,
            TeachingProgression.model_validate(progression_payload()),
            InteractionPlan.model_validate(plan),
        ),
    )


def test_incorrect_response_accepts_reason_and_correction_in_distinct_units():
    _, trajectory, _, *_ = models()
    script, plan = response_anchor_case(
        "偏离目标",
        "回到目标关系",
        "需要回到目标关系重新检查。",
    )
    script["response_scripts"][1]["clauses"][0][
        "display_text"
    ] = "这个选择偏离目标"

    validate_teaching_script(
        TeachingScript.model_validate(script),
        trajectory,
        TeachingProgression.model_validate(progression_payload()),
        InteractionPlan.model_validate(plan),
    )


def test_incorrect_response_rejects_containing_wrapper_with_only_fused_evidence():
    _, trajectory, _, *_ = models()
    script, plan = response_anchor_case(
        "目标关系",
        "关系错误",
        "这个选择出现目标关系错误，请重新检查。",
    )
    script["response_scripts"][1]["clauses"][0][
        "display_text"
    ] = "目标关系错误"

    assert_code(
        "response_semantic_anchor_missing",
        lambda: validate_teaching_script(
            TeachingScript.model_validate(script),
            trajectory,
            TeachingProgression.model_validate(progression_payload()),
            InteractionPlan.model_validate(plan),
        ),
    )


def test_incorrect_response_keeps_noncontaining_natural_prefix_suffix_units():
    _, trajectory, _, *_ = models()
    script, plan = response_anchor_case(
        "偏离目标",
        "回到目标关系",
        "请回到目标关系重新检查。",
    )
    script["response_scripts"][1]["clauses"][0][
        "display_text"
    ] = "这个选择确实偏离目标"

    validate_teaching_script(
        TeachingScript.model_validate(script),
        trajectory,
        TeachingProgression.model_validate(progression_payload()),
        InteractionPlan.model_validate(plan),
    )


def test_incorrect_response_accepts_independent_occurrence_inside_wrapper_unit():
    _, trajectory, _, *_ = models()
    script, plan = response_anchor_case(
        "目标关系",
        "关系错误",
        "这个选择出现目标关系错误，再单独说明关系错误。",
    )
    script["response_scripts"][1]["clauses"][0][
        "display_text"
    ] = "目标关系错误"

    validate_teaching_script(
        TeachingScript.model_validate(script),
        trajectory,
        TeachingProgression.model_validate(progression_payload()),
        InteractionPlan.model_validate(plan),
    )


def test_incorrect_response_rejects_two_distinct_wrappers_with_only_fused_evidence():
    _, trajectory, _, *_ = models()
    script, plan = response_anchor_case(
        "目标关系",
        "关系错误",
        "这里出现目标关系错误请重试。",
    )
    script["response_scripts"][1]["clauses"][0][
        "display_text"
    ] = "判断目标关系错误"

    assert_code(
        "response_semantic_anchor_missing",
        lambda: validate_teaching_script(
            TeachingScript.model_validate(script),
            trajectory,
            TeachingProgression.model_validate(progression_payload()),
            InteractionPlan.model_validate(plan),
        ),
    )


def test_current_interaction_requires_misconception_on_every_wrong_option():
    _, trajectory, _, *_ = models()
    script = TeachingScript.model_validate(semantically_anchored_script_payload())
    plan_payload = interaction_plan_payload()
    plan_payload["interactions"][0]["options"][1]["misconception"] = None
    plan = InteractionPlan.model_validate(plan_payload)

    assert_code(
        "response_misconception_missing",
        lambda: validate_teaching_script(
            script,
            trajectory,
            TeachingProgression.model_validate(progression_payload()),
            plan,
        ),
    )


@pytest.mark.parametrize(
    "unrelated_response",
    (
        "啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊啊",
        "这是未用已知，所以先用根条件，再继续判断。",
    ),
)
def test_wrong_response_requires_its_own_misconception_and_correction_anchors(
    unrelated_response,
):
    _, trajectory, _, *_ = models()
    payload = semantically_anchored_script_payload()
    response = next(
        item for item in payload["response_scripts"]
        if item["option_id"] == "option-b"
    )
    response["clauses"][0]["display_text"] = unrelated_response
    response["clauses"][0]["spoken_text"] = unrelated_response

    assert_code(
        "response_semantic_anchor_missing",
        lambda: validate_current_script(
            TeachingScript.model_validate(payload), trajectory
        ),
    )


def test_wrong_response_accepts_direct_misconception_and_correction_anchors():
    _, trajectory, _, *_ = models()
    validate_current_script(
        TeachingScript.model_validate(semantically_anchored_script_payload()),
        trajectory,
    )


def test_response_clause_requires_interaction_episode_step_and_safe_aligned_language():
    _, trajectory, script, *_ = models()
    payload = script.model_dump()
    payload["response_scripts"][1]["clauses"][0]["episode_id"] = "episode-3"
    assert_code(
        "response_clause_step_invalid",
        lambda: validate_current_script(
            TeachingScript.model_validate(payload), trajectory
        ),
    )

    payload = script.model_dump()
    payload["response_scripts"][1]["clauses"][0]["display_text"] = "<span>提示</span>"
    assert_code(
        "display_content_invalid",
        lambda: validate_current_script(
            TeachingScript.model_validate(payload), trajectory
        ),
    )

    payload = script.model_dump()
    payload["clauses"][0]["display_text"] = r"由 \(m-n\) 开始"
    payload["clauses"][0]["spoken_text"] = "我们从当前目标开始。"
    assert_code(
        "display_spoken_math_mismatch",
        lambda: validate_current_script(
            TeachingScript.model_validate(payload), trajectory
        ),
    )


@pytest.mark.parametrize("private_answer", ("solve-separately", "substitute-root"))
def test_incorrect_response_does_not_expose_any_private_canonical_answer(private_answer):
    _, trajectory, script, *_ = models()
    payload = semantically_anchored_script_payload()
    payload["response_scripts"][1]["clauses"][0]["display_text"] = (
        "内部答案 %s" % private_answer
    )
    assert_code(
        "response_private_answer_leakage",
        lambda: validate_current_script(
            TeachingScript.model_validate(payload), trajectory
        ),
    )


@pytest.mark.parametrize(
    ("canonical_answer", "visible_equivalent"),
    (
        ("solve-separately", "s o l v e - s e p a r a t e l y"),
        ("x=1", "x 等于一"),
        (r"\frac{1}{2}", "二分之一"),
    ),
)
def test_incorrect_response_rejects_compact_or_spoken_canonical_equivalents(
    canonical_answer,
    visible_equivalent,
):
    _, trajectory, _, *_ = models()
    payload = semantically_anchored_script_payload()
    plan_payload = interaction_plan_payload()
    plan_payload["interactions"][0]["options"][1][
        "canonical_answer"
    ] = canonical_answer
    response = payload["response_scripts"][1]
    response["clauses"][0]["spoken_text"] += " %s" % visible_equivalent

    assert_code(
        "response_private_answer_leakage",
        lambda: validate_teaching_script(
            TeachingScript.model_validate(payload),
            trajectory,
            TeachingProgression.model_validate(progression_payload()),
            InteractionPlan.model_validate(plan_payload),
        ),
    )


@pytest.mark.parametrize(
    ("canonical_answer", "spoken_leak"),
    (
        ("1/2", "二分之一"),
        ("p-q=1/2", "p 减 q 等于二分之一"),
        ("1 / 2", "二分之一"),
        ("p-q = 1 / 2", "p 减 q 等于二分之一"),
    ),
)
def test_incorrect_response_rejects_slash_fraction_spoken_canonical_leakage(
    canonical_answer,
    spoken_leak,
):
    _, trajectory, _, *_ = models()
    script, plan = response_anchor_case(
        "偏离目标",
        "目标只是关系",
        "这个选择偏离目标，目标只是关系，%s。" % spoken_leak,
    )
    plan["interactions"][0]["options"][1][
        "canonical_answer"
    ] = canonical_answer

    assert_code(
        "response_private_answer_leakage",
        lambda: validate_teaching_script(
            TeachingScript.model_validate(script),
            trajectory,
            TeachingProgression.model_validate(progression_payload()),
            InteractionPlan.model_validate(plan),
        ),
    )


@pytest.mark.parametrize("canonical_answer", ("1/0", "1 / 0"))
def test_slash_fraction_spoken_canonical_fails_closed_for_zero_denominator(
    canonical_answer,
):
    _, trajectory, _, *_ = models()
    script, plan = response_anchor_case(
        "偏离目标",
        "目标只是关系",
        "这个选择偏离目标，目标只是关系，不能说零分之一。",
    )
    plan["interactions"][0]["options"][1][
        "canonical_answer"
    ] = canonical_answer

    validate_teaching_script(
        TeachingScript.model_validate(script),
        trajectory,
        TeachingProgression.model_validate(progression_payload()),
        InteractionPlan.model_validate(plan),
    )


def test_incorrect_response_does_not_treat_x1_as_spoken_prefix_of_x10():
    _, trajectory, _, *_ = models()
    payload = semantically_anchored_script_payload()
    plan_payload = interaction_plan_payload()
    plan_payload["interactions"][0]["options"][1]["canonical_answer"] = "x=1"
    payload["response_scripts"][1]["clauses"][0]["spoken_text"] += (
        " 例如 x 等于十。"
    )

    validate_teaching_script(
        TeachingScript.model_validate(payload),
        trajectory,
        TeachingProgression.model_validate(progression_payload()),
        InteractionPlan.model_validate(plan_payload),
    )


def test_incorrect_response_allows_canonical_already_public_in_option_display():
    _, trajectory, _, *_ = models()
    script, plan = response_anchor_case(
        "偏离目标",
        "回到目标关系",
        "这个选择偏离目标，需要回到目标关系。",
    )
    option = plan["interactions"][0]["options"][1]
    option["display_text"] = "偏离目标"
    option["canonical_answer"] = "偏离目标"

    validate_teaching_script(
        TeachingScript.model_validate(script),
        trajectory,
        TeachingProgression.model_validate(progression_payload()),
        InteractionPlan.model_validate(plan),
    )


def test_incorrect_response_allows_public_frac_and_slash_canonical_identity():
    _, trajectory, _, *_ = models()
    script, plan = response_anchor_case(
        "偏离目标",
        "回到目标关系",
        "这个选择偏离目标，需要回到目标关系，公开值是二分之一。",
    )
    option = plan["interactions"][0]["options"][1]
    option["display_text"] = r"\(\frac{1}{2}\)"
    option["canonical_answer"] = "1 / 2"

    validate_teaching_script(
        TeachingScript.model_validate(script),
        trajectory,
        TeachingProgression.model_validate(progression_payload()),
        InteractionPlan.model_validate(plan),
    )


def test_incorrect_response_still_rejects_opaque_canonical_not_public_in_display():
    _, trajectory, _, *_ = models()
    script, plan = response_anchor_case(
        "偏离目标",
        "回到目标关系",
        "这个选择偏离目标，需要回到目标关系，不能暴露 opaque-route。",
    )
    option = plan["interactions"][0]["options"][1]
    option["display_text"] = "偏离目标"
    option["canonical_answer"] = "opaque-route"

    assert_code(
        "response_private_answer_leakage",
        lambda: validate_teaching_script(
            TeachingScript.model_validate(script),
            trajectory,
            TeachingProgression.model_validate(progression_payload()),
            InteractionPlan.model_validate(plan),
        ),
    )


@pytest.mark.parametrize(
    ("field", "value", "code"),
    (
        ("spoken_text", None, "teaching_script_content_invalid"),
        ("display_text", None, "clause_display_missing"),
        ("lesson_step_id", None, "clause_lesson_step_missing"),
    ),
)
def test_teaching_script_entry_revalidates_mutated_clause_content(
    field,
    value,
    code,
):
    _, trajectory, script, *_ = models()
    object.__setattr__(script.clauses[0], field, value)
    assert_code(code, lambda: validate_current_script(script, trajectory))


def test_teaching_script_entry_maps_wrong_type_to_content_free_error():
    _, trajectory, _, *_ = models()
    assert_code(
        "teaching_script_content_invalid",
        lambda: validate_teaching_script(
            object(),
            trajectory,
            TeachingProgression.model_validate(progression_payload()),
            InteractionPlan.model_validate(interaction_plan_payload()),
        ),
    )


@pytest.mark.parametrize(
    ("artifact", "code", "artifact_id", "detail"),
    (
        (
            "trajectory",
            "reasoning_trajectory_content_invalid",
            "reasoning_trajectory",
            "Reasoning trajectory content failed defensive model validation.",
        ),
        (
            "progression",
            "teaching_progression_content_invalid",
            "teaching_progression",
            "Teaching progression content failed defensive model validation.",
        ),
        (
            "interaction_plan",
            "interaction_plan_content_invalid",
            "interaction_plan",
            "Interaction plan content failed defensive model validation.",
        ),
    ),
)
def test_teaching_script_entry_defensively_revalidates_upstream_models(
    artifact,
    code,
    artifact_id,
    detail,
):
    _, trajectory, script, plan, *_ = models()
    progression = TeachingProgression.model_validate(progression_payload())
    if artifact == "trajectory":
        object.__setattr__(trajectory.episodes[0], "sequence_index", None)
    elif artifact == "progression":
        object.__setattr__(progression.steps[0], "sequence_index", None)
    else:
        object.__setattr__(plan.interactions[0].options[0], "display_text", None)

    with pytest.raises(PreparationValidationError) as captured:
        validate_teaching_script(script, trajectory, progression, plan)
    assert captured.value.code == code
    assert captured.value.artifact_id == artifact_id
    assert captured.value.detail == detail


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


@pytest.mark.parametrize(
    ("mutation", "code"),
    (
        (lambda item: item.update(episode_id=None), "interaction_intent_missing"),
        (lambda item: item.update(teaching_step_id=None), "interaction_intent_missing"),
        (lambda item: item.update(why_pause=None), "interaction_intent_missing"),
        (lambda item: item.update(resume_step_id=None), "interaction_intent_missing"),
        (lambda item: item.update(resume_policy="retry"), "interaction_resume_policy_invalid"),
        (lambda item: item.update(teaching_step_id="missing-step"), "interaction_step_invalid"),
        (lambda item: item.update(resume_step_id="teaching-step-3"), "interaction_step_invalid"),
        (lambda item: item.update(episode_id="episode-3"), "interaction_step_invalid"),
        (lambda item: item.update(why_pause="这里停一下检查。"), "interaction_why_pause_invalid"),
    ),
)
def test_current_interaction_plan_requires_checkpoint_bound_same_step_intent(
    mutation,
    code,
):
    payload = interaction_plan_payload()
    mutation(payload["interactions"][0])
    assert_code(
        code,
        lambda: validate_interaction_plan(
            InteractionPlan.model_validate(payload),
            TeachingProgression.model_validate(progression_payload()),
        ),
    )


def test_current_interaction_plan_requires_a_declared_checkpoint():
    progression = progression_payload()
    progression["steps"][1]["checkpoint"] = None
    assert_code(
        "interaction_checkpoint_missing",
        lambda: validate_interaction_plan(
            InteractionPlan.model_validate(interaction_plan_payload()),
            TeachingProgression.model_validate(progression),
        ),
    )


@pytest.mark.parametrize(
    ("option_index", "field", "value"),
    (
        (0, "error_code", "correct-error"),
        (0, "remediation_depth", "conceptual"),
        (0, "misconception", "正确项不应有误区"),
        (1, "error_code", None),
        (1, "remediation_depth", None),
        (1, "misconception", None),
    ),
)
def test_current_interaction_plan_requires_private_diagnosis_per_option(
    option_index,
    field,
    value,
):
    payload = interaction_plan_payload()
    payload["interactions"][0]["options"][option_index][field] = value
    assert_code(
        "interaction_option_diagnosis_invalid",
        lambda: validate_interaction_plan(
            InteractionPlan.model_validate(payload),
            TeachingProgression.model_validate(progression_payload()),
        ),
    )


def test_current_interaction_plan_requires_unique_wrong_error_codes():
    payload = interaction_plan_payload()
    payload["interactions"][0]["options"][2]["error_code"] = "option-b-error"
    assert_code(
        "interaction_error_code_duplicate",
        lambda: validate_interaction_plan(
            InteractionPlan.model_validate(payload),
            TeachingProgression.model_validate(progression_payload()),
        ),
    )


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
        lambda: validate_current_script(invalid_script, trajectory),
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
        lambda: validate_current_script(invalid, trajectory),
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
                    "criterion": "authoritative_math_alignment",
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
                    "criterion": "learner_follows_why",
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
        "criterion": "learner_follows_why",
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


def test_review_requires_each_finding_to_cite_an_existing_artifact_id():
    trace, trajectory, script, plan, score, report, _ = models()
    decision = LessonReviewDecision.model_validate(
        {
            "status": "revision_required",
            "findings": [
                {
                    "finding_id": "finding-missing-artifact",
                    "severity": "material",
                    "artifact_type": "teaching_script",
                    "artifact_id": "clause-does-not-exist",
                    "criterion": "learner_follows_why",
                    "evidence": "审核声称该句缺失理由",
                    "responsible_role": "script_teacher",
                    "requested_change": "补充理由",
                    "invalidated_downstream_artifacts": [
                        "interaction_plan",
                        "performance_score",
                        "simulation_report",
                    ],
                }
            ],
            "retained_artifacts": [
                "solution_trace",
                "reasoning_trajectory",
            ],
            "approval_summary": "需修订",
        }
    )

    assert_code(
        "review_evidence_invalid",
        lambda: validate_review_decision(
            decision, trace, trajectory, script, plan, score, report
        ),
    )


@pytest.mark.parametrize(
    "artifact_id",
    ("response-option-b", "response-clause-option-b"),
)
def test_review_can_cite_response_and_response_clause_ids(artifact_id):
    trace, trajectory, script, plan, score, report, _ = models()
    payload = review_payload()
    payload["findings"][0]["artifact_id"] = artifact_id
    decision = LessonReviewDecision.model_validate(payload)

    validate_review_decision(
        decision,
        trace,
        trajectory,
        script,
        plan,
        score,
        report,
        progression=TeachingProgression.model_validate(progression_payload()),
    )


def test_review_can_cite_a_concrete_nested_source_anchor_id():
    trace, trajectory, script, plan, score, report, _ = models()
    anchor_id = trace.assumptions[0].source_anchor.source_id
    decision = LessonReviewDecision.model_validate(
        {
            "status": "revision_required",
            "findings": [
                {
                    "finding_id": "finding-source-anchor",
                    "severity": "material",
                    "artifact_type": "solution_trace",
                    "artifact_id": anchor_id,
                    "criterion": "authoritative_math_alignment",
                    "evidence": "%s 的条件用途未说明" % anchor_id,
                    "responsible_role": "reference_analyst",
                    "requested_change": "标明条件的数学用途",
                    "invalidated_downstream_artifacts": [
                        "reasoning_trajectory",
                        "teaching_progression",
                        "interaction_plan",
                        "teaching_script",
                        "performance_score",
                        "simulation_report",
                    ],
                }
            ],
            "retained_artifacts": [],
            "approval_summary": "需修订",
        }
    )

    validate_review_decision(
        decision, trace, trajectory, script, plan, score, report
    )


def test_review_cannot_assign_a_role_later_than_the_cited_artifact_owner():
    trace, trajectory, script, plan, score, report, _ = models()
    decision = LessonReviewDecision.model_validate(
        {
            "status": "revision_required",
            "findings": [
                {
                    "finding_id": "finding-late-role",
                    "severity": "material",
                    "artifact_type": "teaching_script",
                    "artifact_id": "clause-1",
                    "criterion": "learner_follows_why",
                    "evidence": "clause-1 缺少决定理由",
                    "responsible_role": "classroom_director",
                    "requested_change": "补充理由",
                    "invalidated_downstream_artifacts": [
                        "simulation_report"
                    ],
                }
            ],
            "retained_artifacts": [
                "solution_trace",
                "reasoning_trajectory",
                "teaching_script",
                "interaction_plan",
            ],
            "approval_summary": "需修订",
        }
    )

    assert_code(
        "review_responsibility_invalid",
        lambda: validate_review_decision(
            decision, trace, trajectory, script, plan, score, report
        ),
    )


def test_interaction_finding_rejects_later_script_teacher_responsibility():
    trace, trajectory, script, plan, score, report, _ = models()
    progression = TeachingProgression.model_validate(progression_payload())
    decision = LessonReviewDecision.model_validate(
        {
            "status": "revision_required",
            "findings": [
                {
                    "finding_id": "finding-interaction-late-role",
                    "severity": "material",
                    "artifact_type": "interaction_plan",
                    "artifact_id": "interaction-1",
                    "criterion": "interaction_no_answer_leak",
                    "evidence": "interaction-1 的选项提前泄露了答案",
                    "responsible_role": "script_teacher",
                    "requested_change": "重写选项以诊断理解且不泄露答案",
                    "invalidated_downstream_artifacts": [
                        "teaching_script",
                        "performance_score",
                        "simulation_report",
                    ],
                }
            ],
            "retained_artifacts": [
                "solution_trace",
                "reasoning_trajectory",
                "teaching_progression",
            ],
            "approval_summary": "从互动方案开始修订",
        }
    )

    assert_code(
        "review_responsibility_invalid",
        lambda: validate_review_decision(
            decision,
            trace,
            trajectory,
            script,
            plan,
            score,
            report,
            progression=progression,
        ),
    )


def test_script_finding_rejects_non_owner_interaction_designer_responsibility():
    trace, trajectory, script, plan, score, report, _ = models()
    progression = TeachingProgression.model_validate(progression_payload())
    decision = LessonReviewDecision.model_validate(
        {
            "status": "revision_required",
            "findings": [
                {
                    "finding_id": "finding-script-earlier-role",
                    "severity": "material",
                    "artifact_type": "teaching_script",
                    "artifact_id": "clause-1",
                    "criterion": "learner_follows_why",
                    "evidence": "clause-1 没有覆盖互动后的恢复路径",
                    "responsible_role": "interaction_designer",
                    "requested_change": "补齐互动结果到讲稿的恢复路径",
                    "invalidated_downstream_artifacts": [
                        "performance_score",
                        "simulation_report",
                    ],
                }
            ],
            "retained_artifacts": [
                "solution_trace",
                "reasoning_trajectory",
                "teaching_progression",
                "interaction_plan",
            ],
            "approval_summary": "从讲稿开始修订",
        }
    )

    assert_code(
        "review_responsibility_invalid",
        lambda: validate_review_decision(
            decision,
            trace,
            trajectory,
            script,
            plan,
            score,
            report,
            progression=progression,
        ),
    )


def test_simulation_finding_accepts_exact_student_simulator_owner():
    trace, trajectory, script, plan, score, report, _ = models()
    progression = TeachingProgression.model_validate(progression_payload())
    decision = LessonReviewDecision.model_validate(
        {
            "status": "revision_required",
            "findings": [
                {
                    "finding_id": "finding-simulation-owner",
                    "severity": "material",
                    "artifact_type": "simulation_report",
                    "artifact_id": "episode-1",
                    "criterion": "learner_follows_why",
                    "evidence": "episode-1 的模拟证据不足。",
                    "responsible_role": "student_simulator",
                    "requested_change": "重新模拟该课堂结果。",
                    "invalidated_downstream_artifacts": [],
                }
            ],
            "retained_artifacts": [
                "solution_trace",
                "reasoning_trajectory",
                "teaching_progression",
                "interaction_plan",
                "teaching_script",
                "performance_score",
            ],
            "approval_summary": "仅重新模拟学生表现",
        }
    )

    validate_review_decision(
        decision,
        trace,
        trajectory,
        script,
        plan,
        score,
        report,
        progression=progression,
    )


def test_review_retained_and_invalidated_artifacts_follow_dependency_order():
    trace, trajectory, script, plan, score, report, _ = models()
    base = {
        "status": "revision_required",
        "findings": [
            {
                "finding_id": "finding-script-dependency",
                "severity": "material",
                "artifact_type": "teaching_script",
                "artifact_id": "clause-1",
                "criterion": "learner_follows_why",
                "evidence": "clause-1 没有交代决定理由",
                "responsible_role": "script_teacher",
                "requested_change": "补充决定理由",
                "invalidated_downstream_artifacts": [
                    "performance_score",
                    "simulation_report",
                ],
            }
        ],
        "retained_artifacts": [
            "solution_trace",
            "reasoning_trajectory",
            "teaching_progression",
            "interaction_plan",
        ],
        "approval_summary": "从讲稿开始修订",
    }
    valid = LessonReviewDecision.model_validate(base)
    validate_review_decision(
        valid, trace, trajectory, script, plan, score, report
    )

    director_base = copy.deepcopy(base)
    director_base["findings"][0].update(
        {
            "finding_id": "finding-director-dependency",
            "artifact_type": "performance_score",
            "artifact_id": "cue-clause-1",
            "criterion": "visual_action_alignment",
            "responsible_role": "classroom_director",
            "invalidated_downstream_artifacts": ["simulation_report"],
        }
    )
    director_base["retained_artifacts"] = [
        "solution_trace",
        "reasoning_trajectory",
        "teaching_progression",
        "interaction_plan",
        "teaching_script",
    ]
    validate_review_decision(
        LessonReviewDecision.model_validate(director_base),
        trace,
        trajectory,
        script,
        plan,
        score,
        report,
    )

    invalidated_upstream = copy.deepcopy(base)
    invalidated_upstream["findings"][0][
        "invalidated_downstream_artifacts"
    ] = ["solution_trace"]
    assert_code(
        "review_dependency_invalid",
        lambda: validate_review_decision(
            LessonReviewDecision.model_validate(invalidated_upstream),
            trace,
            trajectory,
            script,
            plan,
            score,
            report,
        ),
    )

    retained_downstream = copy.deepcopy(base)
    retained_downstream["retained_artifacts"].append("teaching_script")
    assert_code(
        "review_dependency_invalid",
        lambda: validate_review_decision(
            LessonReviewDecision.model_validate(retained_downstream),
            trace,
            trajectory,
            script,
            plan,
            score,
            report,
        ),
    )


@pytest.mark.parametrize(
    "invalidated",
    (
        None,
        [],
        ["interaction_plan", "performance_score"],
        ["simulation_report", "performance_score", "interaction_plan"],
    ),
)
def test_material_review_requires_exact_ordered_invalidated_suffix(invalidated):
    trace, trajectory, script, plan, score, report, _ = models()
    finding = {
        "finding_id": "finding-script-exact-suffix",
        "severity": "material",
        "artifact_type": "teaching_script",
        "artifact_id": "clause-1",
        "criterion": "learner_follows_why",
        "evidence": "clause-1 没有交代决定理由",
        "responsible_role": "script_teacher",
        "requested_change": "补充决定理由",
    }
    if invalidated is not None:
        finding["invalidated_downstream_artifacts"] = invalidated
    payload = {
        "status": "revision_required",
        "findings": [finding],
        "retained_artifacts": ["solution_trace", "reasoning_trajectory"],
        "approval_summary": "从讲稿开始修订",
    }

    assert_code(
        "review_dependency_invalid",
        lambda: validate_review_decision(
            LessonReviewDecision.model_validate(payload),
            trace,
            trajectory,
            script,
            plan,
            score,
            report,
        ),
    )


def test_material_review_requires_exact_retained_prefix_for_earliest_finding():
    trace, trajectory, script, plan, score, report, _ = models()
    payload = {
        "status": "revision_required",
        "findings": [
            {
                "finding_id": "finding-director-exact-metadata",
                "severity": "material",
                "artifact_type": "performance_score",
                "artifact_id": "cue-clause-1",
                "criterion": "visual_action_alignment",
                "evidence": "视觉动作错位",
                "responsible_role": "classroom_director",
                "requested_change": "对齐子句",
                "invalidated_downstream_artifacts": ["simulation_report"],
            },
            {
                "finding_id": "finding-script-earliest-metadata",
                "severity": "blocking",
                "artifact_type": "teaching_script",
                "artifact_id": "clause-1",
                "criterion": "learner_follows_why",
                "evidence": "未说明决定理由",
                "responsible_role": "script_teacher",
                "requested_change": "补充理由",
                "invalidated_downstream_artifacts": [
                    "performance_score",
                    "simulation_report",
                ],
            },
        ],
        "retained_artifacts": [],
        "approval_summary": "从最早责任角色修订",
    }

    assert_code(
        "review_dependency_invalid",
        lambda: validate_review_decision(
            LessonReviewDecision.model_validate(payload),
            trace,
            trajectory,
            script,
            plan,
            score,
            report,
        ),
    )
    payload["retained_artifacts"] = [
        "solution_trace",
        "reasoning_trajectory",
        "teaching_progression",
        "interaction_plan",
    ]
    validate_review_decision(
        LessonReviewDecision.model_validate(payload),
        trace,
        trajectory,
        script,
        plan,
        score,
        report,
    )


@pytest.mark.parametrize(
    "retained",
    (
        None,
        [],
        ["reasoning_trajectory", "solution_trace"],
    ),
)
def test_material_review_rejects_omitted_incomplete_or_reordered_retained_prefix(
    retained,
):
    trace, trajectory, script, plan, score, report, _ = models()
    payload = {
        "status": "revision_required",
        "findings": [
            {
                "finding_id": "finding-script-retained-prefix",
                "severity": "material",
                "artifact_type": "teaching_script",
                "artifact_id": "clause-1",
                "criterion": "learner_follows_why",
                "evidence": "未说明决定理由",
                "responsible_role": "script_teacher",
                "requested_change": "补充理由",
                "invalidated_downstream_artifacts": [
                    "performance_score",
                    "simulation_report",
                ],
            }
        ],
        "approval_summary": "从讲稿开始修订",
    }
    if retained is not None:
        payload["retained_artifacts"] = retained

    assert_code(
        "review_dependency_invalid",
        lambda: validate_review_decision(
            LessonReviewDecision.model_validate(payload),
            trace,
            trajectory,
            script,
            plan,
            score,
            report,
        ),
    )


def test_polish_only_approval_uses_empty_repair_metadata():
    trace, trajectory, script, plan, score, report, _ = models()
    payload = review_payload()
    validate_review_decision(
        LessonReviewDecision.model_validate(payload),
        trace,
        trajectory,
        script,
        plan,
        score,
        report,
    )

    retained = copy.deepcopy(payload)
    retained["retained_artifacts"] = ["solution_trace"]
    assert_code(
        "review_dependency_invalid",
        lambda: validate_review_decision(
            LessonReviewDecision.model_validate(retained),
            trace,
            trajectory,
            script,
            plan,
            score,
            report,
        ),
    )

    invalidated = copy.deepcopy(payload)
    invalidated["findings"][0]["invalidated_downstream_artifacts"] = [
        "simulation_report"
    ]
    assert_code(
        "review_dependency_invalid",
        lambda: validate_review_decision(
            LessonReviewDecision.model_validate(invalidated),
            trace,
            trajectory,
            script,
            plan,
            score,
            report,
        ),
    )


@pytest.mark.parametrize(
    "failed_ability",
    (
        "can_identify_attention_target",
        "can_explain_decision",
        "can_execute_action",
        "can_use_result_to_continue",
    ),
)
def test_review_cannot_approve_when_a_non_compensable_novice_gate_fails(
    failed_ability,
):
    trace, trajectory, script, plan, score, report, decision = models()
    report.episode_results[0] = report.episode_results[0].model_copy(
        update={failed_ability: False}
    )

    assert_code(
        "review_non_compensable_gate_invalid",
        lambda: validate_review_decision(
            decision, trace, trajectory, script, plan, score, report
        ),
    )


def test_prepared_lesson_requires_an_approved_review_even_if_artifacts_validate():
    payload = prepared_payload()
    payload["review"] = {
        "status": "revision_required",
        "findings": [
            {
                "finding_id": "finding-script",
                "severity": "material",
                "artifact_type": "teaching_script",
                "artifact_id": "clause-1",
                "criterion": "learner_follows_why",
                "evidence": "clause-1 没有说明理由",
                "responsible_role": "script_teacher",
                "requested_change": "补充理由",
                "invalidated_downstream_artifacts": [
                    "performance_score",
                    "simulation_report",
                ],
            }
        ],
        "retained_artifacts": [
            "solution_trace",
            "reasoning_trajectory",
            "teaching_progression",
            "interaction_plan",
        ],
        "approval_summary": "需修订",
    }
    prepared = PreparedLesson.model_validate(payload)

    assert_code(
        "review_approval_invalid",
        lambda: validate_prepared_lesson(prepared, route(), []),
    )


def assert_history_invalid(payload):
    prepared = PreparedLesson.model_validate(payload)
    assert_code(
        "artifact_history_invalid",
        lambda: validate_prepared_lesson(prepared, route(), []),
    )


def revision(artifact_type, version):
    roles = {
        "solution_trace": "reference_analyst",
        "reasoning_trajectory": "teaching_designer",
        "teaching_progression": "teaching_designer",
        "interaction_plan": "interaction_designer",
        "teaching_script": "script_teacher",
        "performance_score": "classroom_director",
        "simulation_report": "student_simulator",
    }
    return {
        "artifact_type": artifact_type,
        "version": version,
        "responsible_role": roles[artifact_type],
    }


def test_prepared_history_requires_exactly_all_seven_artifact_types():
    payload = prepared_payload()
    payload["artifact_history"] = [
        item
        for item in payload["artifact_history"]
        if item["artifact_type"] != "simulation_report"
    ]
    assert_history_invalid(payload)


@pytest.mark.parametrize("versions", ([1, 1], [1, 3], [2, 1]))
def test_prepared_history_versions_are_unique_contiguous_and_finish_latest(
    versions,
):
    payload = prepared_payload()
    payload["repair_count"] = 1
    payload["artifact_history"] = [
        item
        for item in payload["artifact_history"]
        if item["artifact_type"] != "simulation_report"
    ] + [
        {
            "artifact_type": "simulation_report",
            "version": version,
            "responsible_role": "student_simulator",
        }
        for version in versions
    ]
    assert_history_invalid(payload)


def test_prepared_history_rejects_wrong_artifact_role():
    payload = prepared_payload()
    simulation = next(
        item
        for item in payload["artifact_history"]
        if item["artifact_type"] == "simulation_report"
    )
    simulation["responsible_role"] = "classroom_director"
    assert_history_invalid(payload)


def test_prepared_history_simulation_versions_match_repair_count():
    payload = prepared_payload()
    payload["repair_count"] = 1
    assert_history_invalid(payload)


def test_prepared_history_accepts_dependency_suffix_for_each_repair_cycle():
    payload = prepared_payload()
    payload["repair_count"] = 2
    payload["artifact_history"].extend(
        [
            revision("performance_score", 2),
            revision("simulation_report", 2),
            revision("interaction_plan", 2),
            revision("teaching_script", 2),
            revision("performance_score", 3),
            revision("simulation_report", 3),
        ]
    )

    validate_prepared_lesson(
        PreparedLesson.model_validate(payload),
        route(),
        [],
    )


def test_prepared_history_accepts_simulation_only_repair_suffix():
    payload = prepared_payload()
    payload["repair_count"] = 1
    payload["artifact_history"].append(
        revision("simulation_report", 2)
    )

    validate_prepared_lesson(
        PreparedLesson.model_validate(payload),
        route(),
        [],
        active_versions={
            item["artifact_type"]: (
                2 if item["artifact_type"] == "simulation_report" else 1
            )
            for item in payload["artifact_history"][:7]
        },
    )


@pytest.mark.parametrize(
    "repair_history",
    (
        [revision("solution_trace", 2)],
        [
            revision("performance_score", 2),
            revision("performance_score", 3),
            revision("simulation_report", 2),
        ],
        [
            revision("teaching_script", 2),
            revision("interaction_plan", 2),
            revision("simulation_report", 2),
        ],
        [
            revision("simulation_report", 2),
            revision("performance_score", 2),
        ],
    ),
)
def test_prepared_history_rejects_non_state_machine_chronology(repair_history):
    payload = prepared_payload()
    payload["repair_count"] = 0 if len(repair_history) == 1 else 1
    payload["artifact_history"].extend(copy.deepcopy(repair_history))

    assert_history_invalid(payload)


def test_prepared_history_rejects_reordered_initial_segment():
    payload = prepared_payload()
    payload["artifact_history"][0], payload["artifact_history"][1] = (
        payload["artifact_history"][1],
        payload["artifact_history"][0],
    )

    assert_history_invalid(payload)


def test_prepared_history_rejects_repair_count_over_runtime_budget():
    payload = prepared_payload()
    payload["repair_count"] = 999
    assert_history_invalid(payload)


def test_prepared_history_must_match_supplied_current_active_versions():
    prepared = PreparedLesson.model_validate(prepared_payload())
    active = {
        item.artifact_type: item.version
        for item in prepared.artifact_history
    }
    validate_prepared_lesson(
        prepared,
        route(),
        [],
        active_versions=active,
    )
    active["simulation_report"] = 2
    assert_code(
        "artifact_history_invalid",
        lambda: validate_prepared_lesson(
            prepared,
            route(),
            [],
            active_versions=active,
        ),
    )
