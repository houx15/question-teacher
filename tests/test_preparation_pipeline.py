import asyncio
import inspect
import json

import httpx
import pytest

import app.preparation_pipeline as preparation_pipeline
from app.config import Settings
from app.llm_client import (
    ModelCompletion,
    ModelResponseError,
    ModelStructureError,
    OpenAICompatibleClient,
)
from app.preparation_models import (
    PreparedLesson,
    ReasoningTrajectory,
    RoleCallRecord,
    SolutionTrace,
)
from app.preparation_pipeline import (
    LessonPreparationPipeline,
    PreparationFailure,
    PreparationState,
)
from app.schemas import ProblemFocusTarget, ProblemInput, ReferenceGroundingBrief
from app.teaching_route import freeze_grounded_route
from tests.preparation_fakes import (
    PreparationFakeClient,
    PreparationFakeResponse,
    role_for_system,
)
from tests.test_preparation_models import prepared_lesson as prepared_lesson_payload


STEP_IDS = (
    "substitute-root",
    "connect-target",
    "use-nonzero",
    "return-target",
)
RAW_REFERENCE_MARKER = "PRIVATE-REFERENCE-ONLY-9f17"


def problem() -> ProblemInput:
    return ProblemInput(
        problem_text=(
            "若2n（n不等于0）是方程x^2-2mx+2n=0的根，"
            "求m-n的值。"
        ),
        reference_answer="m-n=1/2",
        reference_solution_text=(
            RAW_REFERENCE_MARKER
            + "：$x=2n$，$4n^2-4mn+2n=0$，"
            "$2n-2m+1=0$，$m-n=1/2$。"
        ),
    )


def route(final_conclusion="m-n=1/2"):
    statements = (
        ("x^2-2mx+2n=0", "substitute", ["x=2n"], "4n^2-4mn+2n=0"),
        ("4n^2-4mn+2n=0", "factor", ["2n"], "2n(2n-2m+1)=0"),
        ("2n(2n-2m+1)=0", "divide", ["2n"], "2n-2m+1=0"),
        ("2n-2m+1=0", "rearrange", [], final_conclusion),
    )
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
                    "statement_before": before,
                    "operation_kind": operation_kind,
                    "operands": operands,
                    "statement_after": after,
                    "assumption_ids_used": (
                        ["assumption-nonzero"]
                        if step_id == "use-nonzero"
                        else ["assumption-root"]
                        if step_id == "substitute-root"
                        else []
                    ),
                }
                for step_id, (before, operation_kind, operands, after) in zip(
                    STEP_IDS, statements
                )
            ],
            "check_requests": [],
            "audit_notes": [],
        },
        final_conclusion,
    )
    return freeze_grounded_route(brief, [])


def focus_targets():
    return [
        ProblemFocusTarget(target_id="problem-root", math_text="2n", ordinal=1),
        ProblemFocusTarget(target_id="problem-equation", math_text="x^2-2mx+2n=0", ordinal=2),
    ]


def trace_payload(final_conclusion="m-n=1/2"):
    states = (
        ("x^2-2mx+2n=0", "substitute", ["x=2n"], "4n^2-4mn+2n=0"),
        ("4n^2-4mn+2n=0", "factor", ["2n"], "2n(2n-2m+1)=0"),
        ("2n(2n-2m+1)=0", "divide", ["2n"], "2n-2m+1=0"),
        ("2n-2m+1=0", "rearrange", [], final_conclusion),
    )
    return {
        "task_target": "m-n",
        "reference_conclusion": final_conclusion,
        "assumptions": [
            {
                "assumption_id": "assumption-nonzero",
                "content": "n!=0",
                "source_anchor": {
                    "source_kind": "problem",
                    "source_id": "problem-nonzero",
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
                "state_before": before,
                "operation_kind": operation_kind,
                "operands": operands,
                "mathematical_action": "待服务端重建",
                "justification": "保留参考解析与题目条件的数学依赖",
                "state_after": after,
                "new_information": "得到下一步所需关系",
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
            for step_id, (before, operation_kind, operands, after) in zip(
                STEP_IDS, states
            )
        ],
        "audit_notes": [],
    }


def trajectory_payload(
    modes=("plan", "execute", "monitor", "revise", "execute"),
    trajectory_type="hybrid",
):
    step_groups = (
        ["substitute-root"],
        ["connect-target"],
        ["use-nonzero"],
        ["use-nonzero"],
        ["return-target"],
    )
    if len(modes) == 1:
        step_groups = (list(STEP_IDS),)
    elif len(modes) == 2:
        step_groups = (list(STEP_IDS[:2]), list(STEP_IDS[2:]))
    return {
        "trajectory_type": trajectory_type,
        "lesson_purpose": "理解为什么每一步能推进到目标",
        "episodes": [
            {
                "episode_id": "episode-%d" % (index + 1),
                "sequence_index": index,
                "mode": mode,
                "source_step_ids": step_groups[index],
                "learner_state_before": "知道已有条件和前一步结果",
                "attention_targets": ["m-n", "n不等于0"],
                "thinking_question": "此刻应该看什么，为什么？",
                "decision": "按当前信息选择下一个数学动作",
                "decision_reason": "这个动作保留冻结路线的先后依赖",
                "mathematical_action": "执行%s片段" % mode,
                "action_justification": "题目条件与已得结果共同支持",
                "result": "得到可以继续判断的新关系",
                "result_meaning": "离目标m-n更近一步",
                "transition_reason": "用新信息决定是否继续或修订",
                "must_teach": [
                    {
                        "must_teach_id": "must-%d" % (index + 1),
                        "content": "当前决定与依据",
                        "why_it_matters": "学生需要跟上为什么",
                    }
                ],
                "likely_misconceptions": [],
                "interaction_intent": None,
                "visual_intent": "只呈现当前关键信息",
            }
            for index, mode in enumerate(modes)
        ],
        "method_summary": "代入根，连接目标，使用非零条件，回到m-n",
        "error_summary": "约去含字母因式前必须先确认它不为零",
    }


def downstream_script_payload():
    clause_specs = (
        ("clause-open", "episode-1", ["must-1"], "先找到题目要求的关系。", ["m-n"]),
        ("clause-method", "episode-1", [], "根一定满足原方程，所以先代入。", ["x=2n"]),
        ("clause-2", "episode-2", ["must-2"], "代入后先观察它与目标的关系。", ["4n^2-4mn+2n=0"]),
        ("clause-2-resume", "episode-2", [], "现在用刚才的判断继续整理。", ["4n^2-4mn+2n=0"]),
        ("clause-3", "episode-3", ["must-3"], "注意n不等于零，这一步才能约去因式。", ["2n-2m+1=0"]),
        ("clause-4", "episode-4", ["must-4"], "约去之后检查新关系是否能继续。", ["2n-2m+1=0"]),
        ("clause-close", "episode-5", ["must-5"], "最后整理并回到m减n这个目标。", ["m-n=1/2"]),
    )
    return {
        "title": "从参数根到目标关系",
        "learning_goal": "理解代入根与非零条件的作用",
        "method_rationale": "根满足原方程",
        "method_introduction": {
            "method_name": "代入法",
            "student_definition": "把已知根放回原方程",
            "target_form": "m-n",
            "why_it_helps": "直接建立参数关系",
        },
        "opening_clause_ids": ["clause-open"],
        "method_introduction_clause_ids": ["clause-method"],
        "clauses": [
            {
                "clause_id": clause_id,
                "episode_id": episode_id,
                "pedagogical_function": "explain",
                "spoken_text": spoken_text,
                "math_references": math_references,
                "learner_gain": "理解当前一步为什么推进",
                "answer_exposure": clause_id == "clause-close",
                "must_teach_refs": must_teach_refs,
            }
            for (
                clause_id,
                episode_id,
                must_teach_refs,
                spoken_text,
                math_references,
            ) in clause_specs
        ],
        "closing_summary_clause_ids": ["clause-close"],
    }


def downstream_transfer_payload():
    return {
        "problem_text": "若a是另一个方程的根，下一步应做什么？",
        "expected_answer": "代入已知根",
        "method_signal": "使用根的定义",
        "options": [
            {"option_id": "transfer-a", "label": "代入已知根", "canonical_answer": "代入已知根", "feedback": "对。"},
            {"option_id": "transfer-b", "label": "猜测参数", "canonical_answer": "猜测", "feedback": "先用根条件。"},
            {"option_id": "transfer-c", "label": "忽略方程", "canonical_answer": "忽略", "feedback": "根必须满足方程。"},
        ],
        "correct_option_id": "transfer-a",
    }


def downstream_interaction_payload():
    return {"interactions": [], "transfer_item": downstream_transfer_payload()}


def downstream_planned_interaction(interaction_id="interaction-1"):
    return {
        "interaction_id": interaction_id,
        "episode_id": "episode-2",
        "after_clause_id": "clause-2",
        "diagnostic_target": "是否知道要继续整理",
        "diagnostic_kind": "execution",
        "prompt": "下一步应该怎样做？",
        "options": [
            {"option_id": "option-a", "display_text": "继续整理", "canonical_answer": "simplify"},
            {"option_id": "option-b", "display_text": "停在原式", "canonical_answer": "stop", "misconception": "没有推进"},
            {"option_id": "option-c", "display_text": "猜测结论", "canonical_answer": "guess", "misconception": "跳步"},
        ],
        "correct_option_id": "option-a",
        "correct_feedback": "对，继续整理才能推进。",
        "incorrect_feedback_by_option": {
            "option-b": "还需要整理。",
            "option-c": "先完成数学步骤。",
        },
        "hint": "看看当前等式。",
        "resume_clause_id": "clause-2-resume",
        "concealed_targets": [],
    }


def downstream_score_payload():
    script = downstream_script_payload()
    return {
        "cues": [
            {
                "cue_id": "cue-%s" % clause["clause_id"],
                "clause_ids": [clause["clause_id"]],
            }
            for clause in script["clauses"]
        ],
        "board_objects": [],
        "overlay_transitions": [],
    }


def downstream_simulation_payload():
    return {
        "episode_results": [
            {
                "episode_id": "episode-%d" % (index + 1),
                "learner_profile": "初学者",
                "can_identify_attention_target": True,
                "can_explain_decision": True,
                "can_execute_action": True,
                "can_use_result_to_continue": True,
                "evidence": ["能说出当前重点、理由、操作和下一步。"],
            }
            for index in range(5)
        ],
        "interaction_results": [],
        "end_of_lesson_recall": "先代入根，再用非零条件，最后回到m-n。",
        "blocking_findings": [],
    }


REVIEW_ARTIFACT_ORDER = [
    "solution_trace",
    "reasoning_trajectory",
    "teaching_script",
    "interaction_plan",
    "performance_score",
    "simulation_report",
]
REVIEW_ROLE_ARTIFACT = {
    "reference_analyst": "solution_trace",
    "teaching_designer": "reasoning_trajectory",
    "script_teacher": "teaching_script",
    "interaction_designer": "interaction_plan",
    "classroom_director": "performance_score",
}


def downstream_review_payload(status="approved", findings=None):
    review_findings = list(findings or [])
    material = [
        item for item in review_findings if item["severity"] != "polish"
    ]
    retained = []
    if material:
        earliest_artifact = min(
            (REVIEW_ROLE_ARTIFACT[item["responsible_role"]] for item in material),
            key=REVIEW_ARTIFACT_ORDER.index,
        )
        retained = REVIEW_ARTIFACT_ORDER[
            : REVIEW_ARTIFACT_ORDER.index(earliest_artifact)
        ]
    return {
        "status": status,
        "findings": review_findings,
        "retained_artifacts": retained,
        "approval_summary": (
            "核心门槛通过" if status == "approved" else "需要定向修订"
        ),
    }


def review_finding(role, criterion="learner_follows_why"):
    artifact_by_role = {
        "reference_analyst": ("solution_trace", "substitute-root"),
        "teaching_designer": ("reasoning_trajectory", "episode-1"),
        "script_teacher": ("teaching_script", "clause-open"),
        "interaction_designer": ("interaction_plan", "interaction_plan"),
        "classroom_director": ("performance_score", "cue-clause-open"),
    }
    artifact_type, artifact_id = artifact_by_role[role]
    artifact_index = REVIEW_ARTIFACT_ORDER.index(artifact_type)
    return {
        "finding_id": "finding-%s-%s" % (role, criterion),
        "severity": "material",
        "artifact_type": artifact_type,
        "artifact_id": artifact_id,
        "criterion": criterion,
        "evidence": "%s 缺少学生可追踪的理由。" % artifact_id,
        "responsible_role": role,
        "requested_change": "补充当前决定的理由和转移。",
        "invalidated_downstream_artifacts": REVIEW_ARTIFACT_ORDER[
            artifact_index + 1 :
        ],
    }


def prompt_payload(recorded_call):
    body = recorded_call.user.split("<UNTRUSTED_SOURCE_DATA>\n", 1)[1]
    return json.loads(body.split("\n</UNTRUSTED_SOURCE_DATA>", 1)[0])


def client(
    trace=None,
    trajectory=None,
    script=None,
    interaction=None,
    performance=None,
    traces=None,
    trajectories=None,
    scripts=None,
    interactions=None,
    performances=None,
    simulations=None,
    reviews=None,
):
    return PreparationFakeClient(
        {
            "reference_analyst": list(
                traces or [trace or trace_payload()]
            ),
            "teaching_designer": list(
                trajectories or [trajectory or trajectory_payload()]
            ),
            "script_teacher": list(
                scripts or [script or downstream_script_payload()]
            ),
            "interaction_designer": list(
                interactions
                or [interaction or downstream_interaction_payload()]
            ),
            "classroom_director": list(
                performances or [performance or downstream_score_payload()]
            ),
            "student_simulator": list(
                simulations or [downstream_simulation_payload()]
            ),
            "lesson_reviewer": list(
                reviews or [downstream_review_payload()]
            ),
        }
    )


def run_early(pipeline, on_stage=None):
    return asyncio.run(
        pipeline.prepare_early(
            problem(), route(), focus_targets(), on_stage=on_stage
        )
    )


def test_public_prepare_returns_only_an_approved_complete_prepared_lesson():
    result = asyncio.run(
        LessonPreparationPipeline(client()).prepare(
            problem(), route(), focus_targets()
        )
    )

    assert type(result) is PreparedLesson
    assert result.review.status == "approved"
    assert len(result.simulation_report.episode_results) == 5


def test_script_interaction_and_performance_stages_run_in_dependency_order():
    fake = client()
    stages = []

    asyncio.run(
        LessonPreparationPipeline(fake).prepare(
            problem(), route(), focus_targets(), on_stage=stages.append
        )
    )

    assert [call.role for call in fake.calls] == [
        "reference_analyst",
        "teaching_designer",
        "script_teacher",
        "interaction_designer",
        "classroom_director",
        "student_simulator",
        "lesson_reviewer",
    ]
    assert stages == [
        "整理参考解析",
        "设计解题思维轨迹",
        "编写讲稿",
        "设计互动",
        "编排板书与高亮",
        "模拟学生并审核课程",
    ]


@pytest.mark.parametrize(
    ("responsible_role", "expected_versions"),
    [
        (
            "reference_analyst",
            {
                "solution_trace": 2,
                "reasoning_trajectory": 2,
                "teaching_script": 2,
                "interaction_plan": 2,
                "performance_score": 2,
                "simulation_report": 2,
            },
        ),
        (
            "teaching_designer",
            {
                "solution_trace": 1,
                "reasoning_trajectory": 2,
                "teaching_script": 2,
                "interaction_plan": 2,
                "performance_score": 2,
                "simulation_report": 2,
            },
        ),
        (
            "script_teacher",
            {
                "solution_trace": 1,
                "reasoning_trajectory": 1,
                "teaching_script": 2,
                "interaction_plan": 2,
                "performance_score": 2,
                "simulation_report": 2,
            },
        ),
        (
            "interaction_designer",
            {
                "solution_trace": 1,
                "reasoning_trajectory": 1,
                "teaching_script": 1,
                "interaction_plan": 2,
                "performance_score": 2,
                "simulation_report": 2,
            },
        ),
        (
            "classroom_director",
            {
                "solution_trace": 1,
                "reasoning_trajectory": 1,
                "teaching_script": 1,
                "interaction_plan": 1,
                "performance_score": 2,
                "simulation_report": 2,
            },
        ),
    ],
)
def test_each_repair_route_retains_upstream_and_rebuilds_only_downstream(
    responsible_role, expected_versions
):
    finding = review_finding(responsible_role)
    fake = client(
        traces=[trace_payload(), trace_payload()],
        trajectories=[trajectory_payload(), trajectory_payload()],
        scripts=[downstream_script_payload(), downstream_script_payload()],
        interactions=[
            downstream_interaction_payload(),
            downstream_interaction_payload(),
        ],
        performances=[downstream_score_payload(), downstream_score_payload()],
        simulations=[
            downstream_simulation_payload(),
            downstream_simulation_payload(),
        ],
        reviews=[
            downstream_review_payload("revision_required", [finding]),
            downstream_review_payload(),
        ],
    )

    run = asyncio.run(
        LessonPreparationPipeline(fake).prepare_with_audit(
            problem(), route(), focus_targets()
        )
    )

    assert run.audit.versions == expected_versions
    assert run.audit.active_versions == expected_versions
    assert run.prepared_lesson.repair_count == 1
    assert run.prepared_lesson.artifact_history == run.audit.history
    repaired_start = REVIEW_ARTIFACT_ORDER.index(finding["artifact_type"])
    assert [
        item.artifact_type for item in run.audit.history
    ] == REVIEW_ARTIFACT_ORDER + REVIEW_ARTIFACT_ORDER[repaired_start:]
    assert [call.role for call in fake.calls].count("student_simulator") == 2
    assert [call.role for call in fake.calls].count("lesson_reviewer") == 2
    repair_call = [
        call for call in fake.calls if call.role == responsible_role
    ][1]
    repair = prompt_payload(repair_call)["repair_request"]
    artifact_order = [
        "solution_trace",
        "reasoning_trajectory",
        "teaching_script",
        "interaction_plan",
        "performance_score",
    ]
    assert set(repair["retained_artifacts"]) == set(
        artifact_order[: artifact_order.index(finding["artifact_type"]) + 1]
    )
    assert repair["finding_ids"] == [finding["finding_id"]]
    repaired_revision = next(
        revision
        for revision in reversed(run.audit.history)
        if revision.responsible_role == responsible_role
    )
    assert repaired_revision.finding_ids == [finding["finding_id"]]


def test_repair_prompt_contains_current_and_upstream_but_no_downstream_artifacts():
    finding = review_finding("script_teacher")
    fake = client(
        scripts=[downstream_script_payload(), downstream_script_payload()],
        interactions=[
            downstream_interaction_payload(),
            downstream_interaction_payload(),
        ],
        performances=[downstream_score_payload(), downstream_score_payload()],
        simulations=[
            downstream_simulation_payload(),
            downstream_simulation_payload(),
        ],
        reviews=[
            downstream_review_payload("revision_required", [finding]),
            downstream_review_payload(),
        ],
    )

    run = asyncio.run(
        LessonPreparationPipeline(fake).prepare_with_audit(
            problem(), route(), focus_targets()
        )
    )

    repair_call = [
        call for call in fake.calls if call.role == "script_teacher"
    ][1]
    repair = prompt_payload(repair_call)["repair_request"]
    assert set(repair["retained_artifacts"]) == {
        "solution_trace",
        "reasoning_trajectory",
        "teaching_script",
    }
    assert "interaction_plan" not in repair_call.user
    assert "performance_score" not in repair_call.user
    repaired_call_record = [
        call for call in run.audit.role_calls if call.role == "script_teacher"
    ][1]
    assert repaired_call_record.input_artifact_versions == {
        "solution_trace": 1,
        "reasoning_trajectory": 1,
        "teaching_script": 1,
    }


@pytest.mark.parametrize(
    ("trajectory_failure", "expected_category"),
    [
        (ModelResponseError("provider unavailable"), "provider_error"),
        (
            {
                **trajectory_payload(),
                "episodes": [
                    {
                        **trajectory_payload()["episodes"][0],
                        "source_step_ids": ["missing-source-step"],
                    },
                    *trajectory_payload()["episodes"][1:],
                ],
            },
            "reasoning_design_failed",
        ),
    ],
)
def test_mid_repair_failure_keeps_issued_history_but_clears_inactive_downstream(
    trajectory_failure, expected_category
):
    finding = review_finding("reference_analyst")
    fake = client(
        traces=[trace_payload(), trace_payload()],
        trajectories=[trajectory_payload(), trajectory_failure],
        simulations=[downstream_simulation_payload()],
        reviews=[downstream_review_payload("revision_required", [finding])],
    )

    with pytest.raises(PreparationFailure) as captured:
        asyncio.run(
            LessonPreparationPipeline(fake).prepare_with_audit(
                problem(), route(), focus_targets()
            )
        )

    assert captured.value.category == expected_category
    assert captured.value.audit.versions == {
        "solution_trace": 2,
        "reasoning_trajectory": 1,
        "teaching_script": 1,
        "interaction_plan": 1,
        "performance_score": 1,
        "simulation_report": 1,
    }
    assert captured.value.audit.active_versions == {"solution_trace": 2}
    assert [
        item.version
        for item in captured.value.audit.history
        if item.artifact_type == "solution_trace"
    ] == [1, 2]
    assert captured.value.audit.role_calls[-1].input_artifact_versions == {
        "solution_trace": 2
    }


@pytest.mark.parametrize(
    "failure_role",
    (
        "reference_analyst",
        "teaching_designer",
        "script_teacher",
        "interaction_designer",
        "classroom_director",
        "student_simulator",
        "lesson_reviewer",
    ),
)
@pytest.mark.parametrize("failure_kind", ("provider", "deterministic"))
def test_every_repair_rebuild_boundary_preserves_truthful_active_versions(
    failure_role, failure_kind
):
    invalid_trace = trace_payload(final_conclusion="m-n=3/2")
    invalid_trajectory = trajectory_payload()
    invalid_trajectory["episodes"][0]["source_step_ids"] = ["missing-step"]
    invalid_script = downstream_script_payload()
    invalid_script["clauses"][3], invalid_script["clauses"][4] = (
        invalid_script["clauses"][4],
        invalid_script["clauses"][3],
    )
    invalid_interaction = {
        "interactions": [
            downstream_planned_interaction("interaction-1"),
            downstream_planned_interaction("interaction-2"),
        ],
        "transfer_item": downstream_transfer_payload(),
    }
    invalid_performance = downstream_score_payload()
    invalid_performance["cues"][0]["lead_actions"] = [
        {
            "clause_id": "clause-open",
            "action": {
                "surface": "problem",
                "type": "focus",
                "target": "missing-problem-target",
            },
        }
    ]
    invalid_simulation = downstream_simulation_payload()
    invalid_simulation["episode_results"] = invalid_simulation[
        "episode_results"
    ][:-1]
    invalid_review = downstream_review_payload(
        "revision_required",
        [
            {
                **review_finding("reference_analyst"),
                "finding_id": "finding-missing-review-artifact",
                "artifact_id": "missing-trace-step",
            }
        ],
    )
    deterministic = {
        "reference_analyst": invalid_trace,
        "teaching_designer": invalid_trajectory,
        "script_teacher": invalid_script,
        "interaction_designer": invalid_interaction,
        "classroom_director": invalid_performance,
        "student_simulator": invalid_simulation,
        "lesson_reviewer": invalid_review,
    }
    failure = (
        ModelResponseError("provider unavailable")
        if failure_kind == "provider"
        else deterministic[failure_role]
    )
    initial_finding = review_finding("reference_analyst")
    fake = client(
        traces=[
            trace_payload(),
            failure if failure_role == "reference_analyst" else trace_payload(),
        ],
        trajectories=[
            trajectory_payload(),
            failure
            if failure_role == "teaching_designer"
            else trajectory_payload(),
        ],
        scripts=[
            downstream_script_payload(),
            failure
            if failure_role == "script_teacher"
            else downstream_script_payload(),
        ],
        interactions=[
            downstream_interaction_payload(),
            failure
            if failure_role == "interaction_designer"
            else downstream_interaction_payload(),
        ],
        performances=[
            downstream_score_payload(),
            failure
            if failure_role == "classroom_director"
            else downstream_score_payload(),
        ],
        simulations=[
            downstream_simulation_payload(),
            failure
            if failure_role == "student_simulator"
            else downstream_simulation_payload(),
        ],
        reviews=[
            downstream_review_payload(
                "revision_required", [initial_finding]
            ),
            failure
            if failure_role == "lesson_reviewer"
            else downstream_review_payload(),
        ],
    )

    with pytest.raises(PreparationFailure) as captured:
        asyncio.run(
            LessonPreparationPipeline(fake).prepare_with_audit(
                problem(), route(), focus_targets()
            )
        )

    dependency_order = [
        "solution_trace",
        "reasoning_trajectory",
        "teaching_script",
        "interaction_plan",
        "performance_score",
        "simulation_report",
    ]
    boundary = dependency_order.index(
        "simulation_report"
        if failure_role == "lesson_reviewer"
        else failure_role.replace("reference_analyst", "solution_trace")
        .replace("teaching_designer", "reasoning_trajectory")
        .replace("script_teacher", "teaching_script")
        .replace("interaction_designer", "interaction_plan")
        .replace("classroom_director", "performance_score")
        .replace("student_simulator", "simulation_report")
    )
    expected_active = {}
    for index, artifact_type in enumerate(dependency_order):
        if index < boundary or failure_role == "lesson_reviewer":
            expected_active[artifact_type] = 2
    if failure_role == "reference_analyst":
        expected_active = {"solution_trace": 1}
    assert captured.value.audit.active_versions == expected_active
    assert set(captured.value.audit.versions) == set(dependency_order)


@pytest.mark.parametrize(
    ("prompt_name", "role"),
    [
        ("student_simulation_prompt", "student_simulator"),
        ("lesson_review_prompt", "lesson_reviewer"),
    ],
)
def test_simulation_and_review_prompt_size_failures_are_safe_and_audited(
    monkeypatch, prompt_name, role
):
    def oversized_prompt(*args, **kwargs):
        del args, kwargs
        raise ValueError("prompt_payload_too_large")

    monkeypatch.setattr(preparation_pipeline, prompt_name, oversized_prompt)

    with pytest.raises(PreparationFailure) as captured:
        asyncio.run(
            LessonPreparationPipeline(client()).prepare_with_audit(
                problem(), route(), focus_targets()
            )
        )

    assert captured.value.category == "prompt_payload_too_large"
    assert captured.value.role == role
    assert captured.value.audit.role_calls[-1].role == role
    assert captured.value.audit.role_calls[-1].failure_category == (
        "prompt_payload_too_large"
    )


def test_repair_projection_limit_failure_is_safe_and_audited(monkeypatch):
    original = preparation_pipeline.teaching_script_prompt

    def bounded_repair_prompt(trajectory, repair=None):
        if repair is not None:
            raise ValueError("repair_request_evidence_text_limit")
        return original(trajectory, repair=repair)

    monkeypatch.setattr(
        preparation_pipeline,
        "teaching_script_prompt",
        bounded_repair_prompt,
    )
    finding = review_finding("script_teacher")
    fake = client(
        reviews=[downstream_review_payload("revision_required", [finding])]
    )

    with pytest.raises(PreparationFailure) as captured:
        asyncio.run(
            LessonPreparationPipeline(fake).prepare_with_audit(
                problem(), route(), focus_targets()
            )
        )

    assert captured.value.category == "prompt_payload_too_large"
    assert captured.value.role == "script_teacher"
    assert captured.value.audit.role_calls[-1].failure_category == (
        "prompt_payload_too_large"
    )
    assert captured.value.audit.active_versions == {
        "solution_trace": 1,
        "reasoning_trajectory": 1,
        "teaching_script": 1,
    }


def test_unknown_prompt_programmer_value_error_propagates(monkeypatch):
    def broken_prompt(*args, **kwargs):
        del args, kwargs
        raise ValueError("programmer contract defect")

    monkeypatch.setattr(
        preparation_pipeline,
        "lesson_review_prompt",
        broken_prompt,
    )

    with pytest.raises(ValueError, match="programmer contract defect"):
        asyncio.run(
            LessonPreparationPipeline(client()).prepare_with_audit(
                problem(), route(), focus_targets()
            )
        )


@pytest.mark.parametrize("role", ("student_simulator", "lesson_reviewer"))
def test_oversized_model_text_fails_as_audited_invalid_structure(role):
    simulations = [downstream_simulation_payload()]
    reviews = [downstream_review_payload()]
    if role == "student_simulator":
        invalid = downstream_simulation_payload()
        invalid["episode_results"][0]["evidence"] = ["证" * 1001]
        simulations = [invalid, invalid]
    else:
        invalid = downstream_review_payload()
        invalid["approval_summary"] = "结" * 2_000_000
        reviews = [invalid, invalid]
    fake = client(simulations=simulations, reviews=reviews)

    with pytest.raises(PreparationFailure) as captured:
        asyncio.run(
            LessonPreparationPipeline(fake).prepare_with_audit(
                problem(), route(), focus_targets()
            )
        )

    assert captured.value.category == "invalid_structure"
    assert captured.value.role == role
    assert captured.value.audit.role_calls[-1].failure_category == (
        "invalid_structure"
    )


def test_multiple_material_findings_repair_from_earliest_responsible_role():
    findings = [
        review_finding("classroom_director", "visual_action_alignment"),
        review_finding("script_teacher", "learner_follows_why"),
    ]
    assert preparation_pipeline.earliest_responsible_role(
        [
            preparation_pipeline.ReviewFinding.model_validate(item)
            for item in findings
        ]
    ) == "script_teacher"
    fake = client(
        scripts=[downstream_script_payload(), downstream_script_payload()],
        interactions=[
            downstream_interaction_payload(),
            downstream_interaction_payload(),
        ],
        performances=[downstream_score_payload(), downstream_score_payload()],
        simulations=[
            downstream_simulation_payload(),
            downstream_simulation_payload(),
        ],
        reviews=[
            downstream_review_payload("revision_required", findings),
            downstream_review_payload(),
        ],
    )

    run = asyncio.run(
        LessonPreparationPipeline(fake).prepare_with_audit(
            problem(), route(), focus_targets()
        )
    )

    assert run.audit.versions["solution_trace"] == 1
    assert run.audit.versions["reasoning_trajectory"] == 1
    assert run.audit.versions["teaching_script"] == 2
    repair_call = [
        call for call in fake.calls if call.role == "script_teacher"
    ][1]
    assert prompt_payload(repair_call)["repair_request"]["finding_ids"] == [
        findings[1]["finding_id"]
    ]


def test_polish_only_review_approves_without_a_repair_cycle():
    polish = review_finding("script_teacher", "learner_follows_why")
    polish["severity"] = "polish"
    polish["invalidated_downstream_artifacts"] = []
    fake = client(
        reviews=[downstream_review_payload("approved", [polish])]
    )

    run = asyncio.run(
        LessonPreparationPipeline(fake).prepare_with_audit(
            problem(), route(), focus_targets()
        )
    )

    assert run.prepared_lesson.review.status == "approved"
    assert run.prepared_lesson.repair_count == 0
    assert [call.role for call in fake.calls].count("script_teacher") == 1


def test_three_repairs_can_converge_without_fixed_two_round_acceptance():
    criteria = [
        "visual_action_alignment",
        "current_emphasis_correct",
        "learner_follows_why",
    ]
    reviews = [
        downstream_review_payload(
            "revision_required",
            [review_finding("classroom_director", criteria[index])],
        )
        for index in range(3)
    ] + [downstream_review_payload()]
    fake = client(
        performances=[downstream_score_payload() for _ in range(4)],
        simulations=[downstream_simulation_payload() for _ in range(4)],
        reviews=reviews,
    )
    stages = []

    run = asyncio.run(
        LessonPreparationPipeline(fake).prepare_with_audit(
            problem(), route(), focus_targets(), on_stage=stages.append
        )
    )

    assert run.prepared_lesson.review.status == "approved"
    assert run.prepared_lesson.repair_count == 3
    assert run.audit.versions["performance_score"] == 4
    assert stages == [
        "整理参考解析",
        "设计解题思维轨迹",
        "编写讲稿",
        "设计互动",
        "编排板书与高亮",
        "模拟学生并审核课程",
        "正在修订完整讲解",
        "编排板书与高亮",
        "模拟学生并审核课程",
        "正在修订完整讲解",
        "编排板书与高亮",
        "模拟学生并审核课程",
        "正在修订完整讲解",
        "编排板书与高亮",
        "模拟学生并审核课程",
    ]


def test_reference_analyst_typed_repair_changes_downstream_projection():
    repaired_trace = trace_payload()
    repaired_trace["source_steps"][0].update(
        reasoning_gap_codes=["implicit_substitution"],
    )
    repaired_trajectory = trajectory_payload()
    repaired_trajectory["episodes"][0]["resolved_gap_refs"] = [
        {
            "source_step_id": "substitute-root",
            "gap_code": "implicit_substitution",
            "must_teach_id": "must-1",
        }
    ]
    fake = client(
        traces=[trace_payload(), repaired_trace],
        trajectories=[trajectory_payload(), repaired_trajectory],
        scripts=[downstream_script_payload(), downstream_script_payload()],
        interactions=[
            downstream_interaction_payload(),
            downstream_interaction_payload(),
        ],
        performances=[
            downstream_score_payload(),
            downstream_score_payload(),
        ],
        simulations=[
            downstream_simulation_payload(),
            downstream_simulation_payload(),
        ],
        reviews=[
            downstream_review_payload(
                "revision_required",
                [review_finding("reference_analyst")],
            ),
            downstream_review_payload(),
        ],
    )

    result = asyncio.run(
        LessonPreparationPipeline(fake).prepare_with_audit(
            problem(), route(), focus_targets()
        )
    )

    first_step = result.prepared_lesson.solution_trace.source_steps[0]
    assert first_step.operation_kind == "substitute"
    assert first_step.assumption_ids_used == ["assumption-root"]
    assert first_step.reasoning_gap_codes == ["implicit_substitution"]
    designer_calls = [
        item for item in fake.calls if item.role == "teaching_designer"
    ]
    repaired_prompt = prompt_payload(designer_calls[1])
    projected_step = repaired_prompt["solution_trace"]["source_steps"][0]
    assert projected_step["operation_kind"] == "substitute"
    assert projected_step["mathematical_action"] == "代入已知数学量：x=2n"
    assert projected_step["reasoning_gap_codes"] == [
        "implicit_substitution"
    ]


def test_eight_unresolved_repairs_fail_safely_without_prepared_lesson():
    cue_ids = [
        "cue-clause-open",
        "cue-clause-method",
        "cue-clause-2",
        "cue-clause-2-resume",
        "cue-clause-3",
        "cue-clause-4",
        "cue-clause-close",
    ]
    findings = []
    for index in range(9):
        finding = review_finding(
            "classroom_director",
            (
                "visual_action_alignment"
                if index < len(cue_ids)
                else "current_emphasis_correct"
            ),
        )
        finding["finding_id"] = "finding-budget-%d" % index
        finding["artifact_id"] = cue_ids[index % len(cue_ids)]
        findings.append(finding)
    reviews = [
        downstream_review_payload(
            "revision_required",
            [findings[index]],
        )
        for index in range(9)
    ]
    fake = client(
        performances=[downstream_score_payload() for _ in range(9)],
        simulations=[downstream_simulation_payload() for _ in range(9)],
        reviews=reviews,
    )

    with pytest.raises(PreparationFailure) as captured:
        asyncio.run(
            LessonPreparationPipeline(fake).prepare_with_audit(
                problem(), route(), focus_targets()
            )
        )

    assert captured.value.category == "review_not_converged"
    assert captured.value.role == "lesson_reviewer"
    assert captured.value.detail == "课程审核未收敛。"
    assert captured.value.audit.versions["performance_score"] == 9
    assert [call.role for call in fake.calls].count("lesson_reviewer") == 9


def test_repeated_signature_uses_fresh_context_then_fails_immediately():
    first = review_finding("classroom_director")
    second = dict(first, finding_id="finding-second", evidence="措辞不同")
    third = dict(first, finding_id="finding-third", requested_change="措辞不同")
    fake = client(
        performances=[downstream_score_payload(), downstream_score_payload()],
        simulations=[
            downstream_simulation_payload(),
            downstream_simulation_payload(),
        ],
        reviews=[
            downstream_review_payload("revision_required", [first]),
            downstream_review_payload("revision_required", [second]),
            downstream_review_payload("revision_required", [third]),
        ],
    )

    with pytest.raises(PreparationFailure) as captured:
        asyncio.run(
            LessonPreparationPipeline(fake).prepare(
                problem(), route(), focus_targets()
            )
        )

    assert captured.value.category == "review_not_converged"
    reviewer_calls = [
        call for call in fake.calls if call.role == "lesson_reviewer"
    ]
    contexts = [prompt_payload(call)["reviewer_context_id"] for call in reviewer_calls]
    assert contexts[0] == contexts[1]
    assert contexts[2] != contexts[1]
    assert [call.role for call in fake.calls].count("classroom_director") == 2


def test_failed_review_never_enters_repair_or_returns_a_lesson():
    finding = review_finding("script_teacher")
    fake = client(
        reviews=[downstream_review_payload("failed", [finding])]
    )

    with pytest.raises(PreparationFailure) as captured:
        asyncio.run(
            LessonPreparationPipeline(fake).prepare(
                problem(), route(), focus_targets()
            )
        )

    assert captured.value.category == "review_not_converged"
    assert [call.role for call in fake.calls].count("script_teacher") == 1


def test_unknown_repair_role_is_rejected_before_state_mutation():
    pipeline = LessonPreparationPipeline(client())
    state = PreparationState()
    before = state.__dict__.copy()
    context_type = preparation_pipeline.PreparationContext
    context = context_type(problem(), route(), focus_targets(), None)

    with pytest.raises(RuntimeError, match="unknown responsible role"):
        asyncio.run(pipeline._repair_from("unknown", state, [], context))

    assert state.__dict__ == before


def test_oversized_valid_upstream_prompt_fails_safely_with_current_audit():
    oversized = trajectory_payload(modes=("plan",))
    private_marker = "OVERSIZED-TRAJECTORY-MARKER"
    oversized["episodes"][0]["decision"] = (
        private_marker + "大" * 300_000
    )
    fake = client(trajectory=oversized)

    with pytest.raises(PreparationFailure) as captured:
        asyncio.run(
            LessonPreparationPipeline(fake).prepare_with_audit(
                problem(), route(), focus_targets()
            )
        )

    failure = captured.value
    assert failure.category == "prompt_payload_too_large"
    assert failure.role == "script_teacher"
    assert failure.detail == "备课内容超出可处理范围。"
    assert private_marker not in failure.detail
    assert [call.role for call in fake.calls] == [
        "reference_analyst",
        "teaching_designer",
    ]
    assert failure.audit is not None
    assert failure.audit.versions == {
        "solution_trace": 1,
        "reasoning_trajectory": 1,
    }
    assert [call.role for call in failure.audit.role_calls] == [
        "reference_analyst",
        "teaching_designer",
        "script_teacher",
    ]
    assert failure.audit.role_calls[-1].failure_category == (
        "prompt_payload_too_large"
    )


def test_zero_interactions_and_cues_without_highlights_are_accepted():
    fake = client(
        interaction=downstream_interaction_payload(),
        performance=downstream_score_payload(),
    )

    asyncio.run(
        LessonPreparationPipeline(fake).prepare(
            problem(), route(), focus_targets()
        )
    )

    assert [call.role for call in fake.calls][-4:] == [
        "interaction_designer",
        "classroom_director",
        "student_simulator",
        "lesson_reviewer",
    ]


def test_script_dependency_reordering_fails_without_structure_retry():
    invalid = downstream_script_payload()
    invalid["clauses"][3], invalid["clauses"][4] = (
        invalid["clauses"][4],
        invalid["clauses"][3],
    )
    fake = client(script=invalid)

    with pytest.raises(PreparationFailure) as captured:
        asyncio.run(
            LessonPreparationPipeline(fake).prepare(
                problem(), route(), focus_targets()
            )
        )

    assert captured.value.category == "teaching_script_failed"
    assert captured.value.role == "script_teacher"
    assert [call.role for call in fake.calls].count("script_teacher") == 1
    assert captured.value.audit.versions == {
        "solution_trace": 1,
        "reasoning_trajectory": 1,
    }


def test_non_discriminating_performance_emphasis_fails_deterministically():
    invalid = downstream_score_payload()
    invalid["board_objects"] = [
        {"board_object_id": "only-board-object", "content": "m-n"}
    ]
    invalid["cues"][0]["start_actions"] = [
        {
            "clause_id": "clause-open",
            "action": {
                "surface": "board",
                "type": "write",
                "target": "only-board-object",
                "content": "m-n",
            },
        },
        {
            "clause_id": "clause-open",
            "action": {
                "surface": "board",
                "type": "annotate",
                "target": "only-board-object",
                "annotation": "label",
                "content": "m-n",
            },
        },
    ]
    fake = client(performance=invalid)

    with pytest.raises(PreparationFailure) as captured:
        asyncio.run(
            LessonPreparationPipeline(fake).prepare(
                problem(), route(), focus_targets()
            )
        )

    assert captured.value.category == "performance_score_failed"
    assert captured.value.role == "classroom_director"
    assert [call.role for call in fake.calls].count("classroom_director") == 1


def test_highlighting_the_only_board_element_is_non_discriminating():
    invalid = downstream_score_payload()
    invalid["board_objects"] = [
        {"board_object_id": "only-board-object", "content": "m-n"}
    ]
    invalid["cues"][0]["start_actions"] = [
        {
            "clause_id": "clause-open",
            "action": {
                "surface": "board",
                "type": "write",
                "target": "only-board-object",
                "content": "m-n",
            },
        },
        {
            "clause_id": "clause-open",
            "action": {
                "surface": "board",
                "type": "emphasize",
                "target": "only-board-object",
                "emphasis_style": "highlight",
            },
        },
    ]
    invalid["cues"][0]["end_actions"] = [
        {
            "clause_id": "clause-open",
            "action": {
                "surface": "board",
                "type": "fade",
                "target": "only-board-object",
            },
        }
    ]

    with pytest.raises(PreparationFailure) as captured:
        asyncio.run(
            LessonPreparationPipeline(client(performance=invalid)).prepare(
                problem(), route(), focus_targets()
            )
        )

    assert captured.value.category == "performance_score_failed"


def test_problem_highlight_must_be_bound_to_a_clause_discussing_the_target():
    script = downstream_script_payload()
    script["clauses"][1]["math_references"] = ["x^2-2mx+2n=0"]
    invalid = downstream_score_payload()
    clause_4_cue = next(
        cue for cue in invalid["cues"] if cue["clause_ids"] == ["clause-4"]
    )
    clause_4_cue["lead_actions"] = [
        {
            "clause_id": "clause-4",
            "action": {
                "surface": "problem",
                "type": "focus",
                "target": "problem-equation",
            },
        }
    ]

    with pytest.raises(PreparationFailure) as captured:
        asyncio.run(
            LessonPreparationPipeline(
                client(script=script, performance=invalid)
            ).prepare(problem(), route(), focus_targets())
        )

    assert captured.value.category == "performance_score_failed"


def test_board_object_can_be_emphasized_and_faded_after_introduction():
    script = downstream_script_payload()
    script["clauses"][0]["math_references"].append("x=2n")
    valid = downstream_score_payload()
    valid["board_objects"] = [
        {"board_object_id": "target-relation", "content": "m-n"},
        {"board_object_id": "given-root", "content": "x=2n"},
    ]
    valid["cues"][0]["start_actions"] = [
        {
            "clause_id": "clause-open",
            "action": {
                "surface": "board",
                "type": "write",
                "target": "target-relation",
                "content": "m-n",
            },
        },
        {
            "clause_id": "clause-open",
            "action": {
                "surface": "board",
                "type": "write",
                "target": "given-root",
                "content": "x=2n",
            },
        },
    ]
    valid["cues"][1]["start_actions"] = [
        {
            "clause_id": "clause-method",
            "action": {
                "surface": "board",
                "type": "emphasize",
                "target": "target-relation",
                "emphasis_style": "underline",
            },
        }
    ]
    valid["cues"][1]["end_actions"] = [
        {
            "clause_id": "clause-method",
            "action": {
                "surface": "board",
                "type": "fade",
                "target": "target-relation",
            },
        }
    ]

    asyncio.run(
        LessonPreparationPipeline(
            client(script=script, performance=valid)
        ).prepare(problem(), route(), focus_targets())
    )


def test_multiple_interactions_in_one_episode_fail_before_performance_stage():
    interaction = downstream_planned_interaction()
    duplicate = downstream_planned_interaction("interaction-2")
    plan = {
        "interactions": [interaction, duplicate],
        "transfer_item": downstream_transfer_payload(),
    }
    fake = client(interaction=plan)

    with pytest.raises(PreparationFailure) as captured:
        asyncio.run(
            LessonPreparationPipeline(fake).prepare(
                problem(), route(), focus_targets()
            )
        )

    assert captured.value.category == "interaction_plan_failed"
    assert captured.value.role == "interaction_designer"
    assert [call.role for call in fake.calls].count("interaction_designer") == 1
    assert "classroom_director" not in [call.role for call in fake.calls]


def test_overlay_history_does_not_hide_sole_base_object_emphasis():
    score = downstream_score_payload()
    score["board_objects"] = [
        {"board_object_id": "base-target", "content": "m-n"},
        {
            "board_object_id": "overlay-target",
            "content": "4n^2-4mn+2n=0",
            "layer": "comparison",
        },
    ]
    score["cues"][0]["start_actions"] = [
        {
            "clause_id": "clause-open",
            "action": {
                "surface": "board",
                "type": "write",
                "target": "base-target",
                "content": "m-n",
            },
        }
    ]
    score["overlay_transitions"] = [
        {
            "transition_id": "enter-comparison",
            "after_clause_id": "clause-method",
            "action": "enter",
            "layer": "comparison",
        },
        {
            "transition_id": "return-comparison",
            "after_clause_id": "clause-2",
            "action": "return",
            "layer": "comparison",
        },
    ]
    clause_2 = next(
        cue for cue in score["cues"] if cue["clause_ids"] == ["clause-2"]
    )
    clause_2["start_actions"] = [
        {
            "clause_id": "clause-2",
            "action": {
                "surface": "board",
                "type": "write",
                "target": "overlay-target",
                "content": "4n^2-4mn+2n=0",
            },
        }
    ]
    resume = next(
        cue
        for cue in score["cues"]
        if cue["clause_ids"] == ["clause-2-resume"]
    )
    resume["start_actions"] = [
        {
            "clause_id": "clause-2-resume",
            "action": {
                "surface": "board",
                "type": "emphasize",
                "target": "base-target",
                "emphasis_style": "highlight",
            },
        }
    ]
    resume["end_actions"] = [
        {
            "clause_id": "clause-2-resume",
            "action": {
                "surface": "board",
                "type": "fade",
                "target": "base-target",
            },
        }
    ]

    with pytest.raises(PreparationFailure) as captured:
        asyncio.run(
            LessonPreparationPipeline(client(performance=score)).prepare(
                problem(), route(), focus_targets()
            )
        )

    assert captured.value.category == "performance_score_failed"


def test_prepare_return_annotation_remains_prepared_lesson():
    signature = inspect.signature(LessonPreparationPipeline.prepare)

    assert signature.return_annotation is PreparedLesson


def test_trace_and_trajectory_stages_run_in_dependency_order():
    fake = client()
    stages = []
    pipeline = LessonPreparationPipeline(fake)

    result = run_early(pipeline, on_stage=stages.append)

    assert [call.role for call in fake.calls] == [
        "reference_analyst",
        "teaching_designer",
    ]
    assert stages == ["整理参考解析", "设计解题思维轨迹"]
    assert isinstance(result.solution_trace, SolutionTrace)
    assert isinstance(result.reasoning_trajectory, ReasoningTrajectory)
    assert result.versions == {
        "solution_trace": 1,
        "reasoning_trajectory": 1,
    }
    assert result.active_versions == result.versions


def test_raw_reference_solution_reaches_only_reference_analyst():
    fake = client()

    run_early(LessonPreparationPipeline(fake))

    analyst_calls = [call for call in fake.calls if call.role == "reference_analyst"]
    designer_calls = [call for call in fake.calls if call.role == "teaching_designer"]
    assert all(RAW_REFERENCE_MARKER in call.user for call in analyst_calls)
    assert all(RAW_REFERENCE_MARKER not in call.user for call in designer_calls)


def test_reference_anchor_excerpt_is_replaced_before_downstream_use():
    trace = trace_payload()
    trace["assumptions"][0]["source_anchor"]["excerpt"] = (
        RAW_REFERENCE_MARKER
    )
    fake = client(trace=trace)

    result = run_early(LessonPreparationPipeline(fake))

    assert RAW_REFERENCE_MARKER not in result.solution_trace.model_dump_json()
    designer_call = next(
        call for call in fake.calls if call.role == "teaching_designer"
    )
    assert RAW_REFERENCE_MARKER not in designer_call.user


def test_reference_only_literal_in_trajectory_is_rejected_before_script():
    trajectory = trajectory_payload()
    trajectory["lesson_purpose"] += " " + RAW_REFERENCE_MARKER
    fake = client(trajectory=trajectory)

    with pytest.raises(PreparationFailure) as captured:
        run_early(LessonPreparationPipeline(fake))

    assert captured.value.category == "reference_content_leak"
    assert RAW_REFERENCE_MARKER not in str(captured.value)
    assert [call.role for call in fake.calls] == [
        "reference_analyst",
        "teaching_designer",
    ]


def test_reference_only_literal_in_script_is_rejected_before_interaction():
    script = downstream_script_payload()
    script["title"] += " " + RAW_REFERENCE_MARKER
    fake = client(script=script)

    with pytest.raises(PreparationFailure) as captured:
        asyncio.run(
            LessonPreparationPipeline(fake).prepare(
                problem(), route(), focus_targets()
            )
        )

    assert captured.value.category == "reference_content_leak"
    assert RAW_REFERENCE_MARKER not in str(captured.value)
    assert [call.role for call in fake.calls] == [
        "reference_analyst",
        "teaching_designer",
        "script_teacher",
    ]


def test_reference_only_literal_in_review_is_rejected_privately():
    review = downstream_review_payload()
    review["approval_summary"] += " " + RAW_REFERENCE_MARKER
    fake = client(reviews=[review])

    with pytest.raises(PreparationFailure) as captured:
        asyncio.run(
            LessonPreparationPipeline(fake).prepare(
                problem(), route(), focus_targets()
            )
        )

    assert captured.value.category == "reference_content_leak"
    assert RAW_REFERENCE_MARKER not in str(captured.value)
    assert captured.value.audit is not None
    assert captured.value.audit.role_calls[-1].failure_category == (
        "reference_content_leak"
    )


def test_each_role_gets_one_structure_retry():
    fake = PreparationFakeClient(
        {
            "reference_analyst": [{"unexpected": "shape"}, trace_payload()],
            "teaching_designer": [[], trajectory_payload()],
        }
    )
    pipeline = LessonPreparationPipeline(fake)

    result = run_early(pipeline)

    assert [call.role for call in fake.calls] == [
        "reference_analyst",
        "reference_analyst",
        "teaching_designer",
        "teaching_designer",
    ]
    assert [record.retry_count for record in result.role_calls] == [1, 1]
    assert [record.failure_category for record in result.role_calls] == [None, None]


def test_every_preparation_role_receives_its_exact_output_schema():
    fake = client()

    asyncio.run(
        LessonPreparationPipeline(fake).prepare(
            problem(), route(), focus_targets()
        )
    )

    required_property_by_role = {
        "reference_analyst": "source_steps",
        "teaching_designer": "episodes",
        "script_teacher": "clauses",
        "interaction_designer": "interactions",
        "classroom_director": "cues",
        "student_simulator": "episode_results",
        "lesson_reviewer": "status",
    }
    assert [call.role for call in fake.calls] == list(required_property_by_role)
    for call in fake.calls:
        schema_text = call.user.split("<OUTPUT_JSON_SCHEMA>\n", 1)[1].split(
            "\n</OUTPUT_JSON_SCHEMA>", 1
        )[0]
        schema = json.loads(schema_text)
        assert required_property_by_role[call.role] in schema["properties"]


def test_structure_retry_keeps_schema_without_echoing_invalid_output():
    private_invalid_marker = "PRIVATE_INVALID_MODEL_OUTPUT"
    fake = PreparationFakeClient(
        {
            "reference_analyst": [
                {"unexpected": private_invalid_marker},
                trace_payload(),
            ],
            "teaching_designer": [trajectory_payload()],
        }
    )

    run_early(LessonPreparationPipeline(fake))

    analyst_calls = [
        call for call in fake.calls if call.role == "reference_analyst"
    ]
    assert len(analyst_calls) == 2
    assert all("<OUTPUT_JSON_SCHEMA>" in call.user for call in analyst_calls)
    assert private_invalid_marker not in analyst_calls[1].user
    assert analyst_calls[1].user.count("<OUTPUT_JSON_SCHEMA>") == 1


def test_preparation_prefers_provider_native_structured_output():
    class NativeStructuredFake(PreparationFakeClient):
        def __init__(self):
            super().__init__(
                {
                    "reference_analyst": [trace_payload()],
                    "teaching_designer": [trajectory_payload()],
                }
            )
            self.model_types = []

        async def complete_model_with_metadata(
            self, system, user, model_type
        ):
            self.model_types.append(model_type)
            return await super().complete_json_with_metadata(system, user)

    fake = NativeStructuredFake()

    run_early(LessonPreparationPipeline(fake))

    assert fake.model_types == [SolutionTrace, ReasoningTrajectory]


@pytest.mark.parametrize(
    ("failing_role", "responses"),
    [
        (
            "reference_analyst",
            {
                "reference_analyst": [{}, {"still": "invalid"}],
                "teaching_designer": [trajectory_payload()],
            },
        ),
        (
            "teaching_designer",
            {
                "reference_analyst": [trace_payload()],
                "teaching_designer": [{}, []],
            },
        ),
    ],
)
def test_second_invalid_structure_fails_with_safe_role(failing_role, responses):
    pipeline = LessonPreparationPipeline(PreparationFakeClient(responses))

    with pytest.raises(PreparationFailure) as captured:
        asyncio.run(
            pipeline.prepare_early(problem(), route(), focus_targets())
        )

    failure = captured.value
    assert failure.category == "invalid_structure"
    assert failure.role == failing_role
    assert failure.detail == "模型输出结构无效。"
    assert RAW_REFERENCE_MARKER not in str(failure)
    assert failure.audit is not None
    assert failure.audit.role_calls[-1].failure_category == "invalid_structure"
    assert failure.audit.role_calls[-1].output_artifact_type is None


def test_invalid_json_response_gets_the_same_single_structure_retry():
    fake = PreparationFakeClient(
        {
            "reference_analyst": [
                ModelStructureError(
                    "invalid_json",
                    token_usage={"prompt_tokens": 4, "total_tokens": 4},
                ),
                PreparationFakeResponse(
                    trace_payload(),
                    {"prompt_tokens": 6, "total_tokens": 6},
                ),
            ],
            "teaching_designer": [trajectory_payload()],
        }
    )
    pipeline = LessonPreparationPipeline(fake)

    result = run_early(pipeline)

    assert [call.role for call in fake.calls].count("reference_analyst") == 2
    assert result.role_calls[0].retry_count == 1
    assert result.role_calls[0].token_usage == {
        "prompt_tokens": 10,
        "total_tokens": 10,
    }


def test_real_openai_client_metadata_usage_reaches_pipeline_records():
    responses = [
        (trace_payload(), {"prompt_tokens": 10, "total_tokens": 10}),
        (trajectory_payload(), {"prompt_tokens": 20, "total_tokens": 20}),
    ]
    request_count = 0

    def handler(request):
        nonlocal request_count
        request_count += 1
        payload, usage = responses.pop(0)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": json.dumps(payload)}}
                ],
                "usage": usage,
            },
        )

    settings = Settings(
        openai_base_url="https://model.example/v1",
        openai_api_key="test-secret-key",
        openai_model="demo-model",
    )
    client_with_metadata = OpenAICompatibleClient(
        settings,
        transport=httpx.MockTransport(handler),
    )

    result = run_early(LessonPreparationPipeline(client_with_metadata))
    asyncio.run(client_with_metadata.close())

    assert request_count == 2
    assert [record.token_usage for record in result.role_calls] == [
        {"prompt_tokens": 10, "total_tokens": 10},
        {"prompt_tokens": 20, "total_tokens": 20},
    ]


def test_provider_failure_is_not_misclassified_or_structure_retried():
    provider_message = "Model request failed with HTTP status 503. secret payload"
    fake = PreparationFakeClient(
        {
            "reference_analyst": [ModelResponseError(provider_message)],
            "teaching_designer": [trajectory_payload()],
        }
    )
    pipeline = LessonPreparationPipeline(fake)

    with pytest.raises(PreparationFailure) as captured:
        asyncio.run(
            pipeline.prepare_early(problem(), route(), focus_targets())
        )

    assert captured.value.category == "provider_error"
    assert captured.value.role == "reference_analyst"
    assert captured.value.detail == "模型服务暂时不可用。"
    assert provider_message not in str(captured.value)
    assert len(fake.calls) == 1
    assert captured.value.audit.role_calls[-1].failure_category == "provider_error"
    returned_calls = captured.value.audit.role_calls
    returned_calls[-1].failure_category = "tampered"
    assert captured.value.audit.role_calls[-1].failure_category == "provider_error"
    assert RAW_REFERENCE_MARKER not in repr(captured.value.audit)


def test_deterministic_trace_failure_stops_before_designer_without_retry():
    fake = client(trace=trace_payload(final_conclusion="m-n=3/2"))
    pipeline = LessonPreparationPipeline(fake)

    with pytest.raises(PreparationFailure) as captured:
        asyncio.run(
            pipeline.prepare_early(problem(), route(), focus_targets())
        )

    assert captured.value.category == "reference_trace_failed"
    assert captured.value.role == "reference_analyst"
    assert [call.role for call in fake.calls] == ["reference_analyst"]
    assert captured.value.audit.versions == {}
    assert captured.value.audit.role_calls[-1].failure_category == "reference_trace_failed"
    assert captured.value.audit.role_calls[-1].output_artifact_version is None


def test_deterministic_trajectory_failure_is_not_a_structure_retry():
    invalid_trajectory = trajectory_payload()
    invalid_trajectory["episodes"][0]["source_step_ids"] = ["missing-step"]
    fake = client(trajectory=invalid_trajectory)
    pipeline = LessonPreparationPipeline(fake)

    with pytest.raises(PreparationFailure) as captured:
        asyncio.run(
            pipeline.prepare_early(problem(), route(), focus_targets())
        )

    assert captured.value.category == "reasoning_design_failed"
    assert captured.value.role == "teaching_designer"
    assert [call.role for call in fake.calls] == [
        "reference_analyst",
        "teaching_designer",
    ]
    assert captured.value.audit.versions == {"solution_trace": 1}
    assert captured.value.audit.role_calls[-1].failure_category == "reasoning_design_failed"


def test_plan_execute_monitor_revise_execute_trajectory_is_accepted():
    pipeline = LessonPreparationPipeline(client())

    result = run_early(pipeline)

    assert [
        episode.mode
        for episode in result.reasoning_trajectory.episodes
    ] == ["plan", "execute", "monitor", "revise", "execute"]


@pytest.mark.parametrize(
    ("trajectory_type", "modes"),
    [
        ("planned", ("plan", "execute")),
        ("exploratory", ("explore", "monitor")),
        ("hybrid", ("execute",)),
    ],
)
def test_trajectory_type_does_not_force_every_reasoning_mode(
    trajectory_type, modes
):
    fake = client(
        trajectory=trajectory_payload(
            modes=modes, trajectory_type=trajectory_type
        )
    )
    pipeline = LessonPreparationPipeline(fake)

    result = run_early(pipeline)

    accepted = result.reasoning_trajectory
    assert accepted.trajectory_type == trajectory_type
    assert tuple(episode.mode for episode in accepted.episodes) == modes


def test_parameter_root_trajectory_preserves_four_indispensable_moves():
    pipeline = LessonPreparationPipeline(client())

    result = run_early(pipeline)

    trajectory = result.reasoning_trajectory
    covered = {
        step_id
        for episode in trajectory.episodes
        for step_id in episode.source_step_ids
    }
    assert covered.issuperset(
        {
            "substitute-root",
            "connect-target",
            "use-nonzero",
            "return-target",
        }
    )
    summary = trajectory.method_summary
    assert all(token in summary for token in ("代入", "目标", "非零", "m-n"))


def test_role_records_are_versioned_only_after_validation_and_are_content_safe():
    fake = PreparationFakeClient(
        {
            "reference_analyst": [
                PreparationFakeResponse(
                    payload=trace_payload(),
                    token_usage={
                        "prompt_tokens": 321,
                        "completion_tokens": 123,
                        "total_tokens": 444,
                    },
                )
            ],
            "teaching_designer": [trajectory_payload()],
        }
    )
    pipeline = LessonPreparationPipeline(fake)

    result = run_early(pipeline)

    analyst, designer = result.role_calls
    assert analyst.input_artifact_versions == {}
    assert analyst.output_artifact_type == "solution_trace"
    assert analyst.output_artifact_version == 1
    assert analyst.token_usage == {
        "prompt_tokens": 321,
        "completion_tokens": 123,
        "total_tokens": 444,
    }
    assert designer.input_artifact_versions == {"solution_trace": 1}
    assert designer.output_artifact_type == "reasoning_trajectory"
    assert designer.output_artifact_version == 1
    serialized = json.dumps(
        [record.model_dump(mode="json") for record in result.role_calls],
        ensure_ascii=False,
    )
    assert RAW_REFERENCE_MARKER not in serialized
    assert "system" not in analyst.model_fields
    assert "prompt" not in analyst.model_fields
    assert "payload" not in analyst.model_fields


def test_concurrent_preparations_do_not_share_active_state():
    class InterleavingClient(PreparationFakeClient):
        def __init__(self):
            super().__init__(
                {
                    "reference_analyst": [trace_payload(), trace_payload()],
                    "teaching_designer": [
                        trajectory_payload(),
                        trajectory_payload(),
                    ],
                }
            )
            self.analyst_arrivals = 0
            self.both_analysts_started = asyncio.Event()

        async def complete_json_with_metadata(self, system, user):
            if "参考材料分析员" in system:
                self.analyst_arrivals += 1
                if self.analyst_arrivals == 2:
                    self.both_analysts_started.set()
                await self.both_analysts_started.wait()
            return await super().complete_json_with_metadata(system, user)

    async def scenario():
        pipeline = LessonPreparationPipeline(InterleavingClient())
        return await asyncio.gather(
            pipeline.prepare_early(problem(), route(), focus_targets()),
            pipeline.prepare_early(problem(), route(), focus_targets()),
        )

    results = asyncio.run(scenario())

    assert len(results) == 2
    assert all(result.versions["reasoning_trajectory"] == 1 for result in results)


def test_reversed_concurrent_completion_keeps_distinct_artifacts_and_usage():
    class ReversedCompletionClient:
        def __init__(self):
            self.release_a = asyncio.Event()
            self.finished = []

        async def complete_json(self, system, user):
            role = role_for_system(system)
            is_a = "REQUEST-A" in user
            if is_a and role == "reference_analyst":
                await self.release_a.wait()
            if role == "reference_analyst":
                conclusion = "m-n=1/2" if is_a else "m-n=3/2"
                usage = 11 if is_a else 21
                return ModelCompletion(
                    trace_payload(final_conclusion=conclusion),
                    {"prompt_tokens": usage, "total_tokens": usage},
                )
            usage = 12 if is_a else 22
            if not is_a:
                self.finished.append("B")
                self.release_a.set()
            else:
                self.finished.append("A")
            return ModelCompletion(
                trajectory_payload(),
                {"prompt_tokens": usage, "total_tokens": usage},
            )

    async def scenario():
        fake = ReversedCompletionClient()
        pipeline = LessonPreparationPipeline(fake)
        problem_a = problem().model_copy(
            update={"problem_text": problem().problem_text + " REQUEST-A"}
        )
        problem_b = problem().model_copy(
            update={
                "problem_text": problem().problem_text + " REQUEST-B",
                "reference_answer": "m-n=3/2",
            }
        )
        results = await asyncio.gather(
            pipeline.prepare_early(
                problem_a, route("m-n=1/2"), focus_targets()
            ),
            pipeline.prepare_early(
                problem_b, route("m-n=3/2"), focus_targets()
            ),
        )
        return fake, results

    fake, (result_a, result_b) = asyncio.run(scenario())

    assert fake.finished == ["B", "A"]
    assert result_a.solution_trace.reference_conclusion == "m-n=1/2"
    assert result_b.solution_trace.reference_conclusion == "m-n=3/2"
    assert [
        call.token_usage["prompt_tokens"] for call in result_a.role_calls
    ] == [11, 12]
    assert [
        call.token_usage["prompt_tokens"] for call in result_b.role_calls
    ] == [21, 22]


def test_retry_usage_is_summed_and_unknown_or_secret_keys_are_omitted():
    fake = PreparationFakeClient(
        {
            "reference_analyst": [
                PreparationFakeResponse(
                    payload={},
                    token_usage={
                        "prompt_tokens": 10,
                        "completion_tokens": 2,
                        "total_tokens": 12,
                        "api_secret": 999,
                    },
                ),
                PreparationFakeResponse(
                    payload=trace_payload(),
                    token_usage={
                        "prompt_tokens": 20,
                        "completion_tokens": 3,
                        "total_tokens": 23,
                        "unknown_counter": 10,
                    },
                ),
            ],
            "teaching_designer": [trajectory_payload()],
        }
    )

    result = run_early(LessonPreparationPipeline(fake))

    assert result.role_calls[0].token_usage == {
        "prompt_tokens": 30,
        "completion_tokens": 5,
        "total_tokens": 35,
    }


def test_provider_failure_does_not_inherit_usage_from_an_earlier_call():
    fake = PreparationFakeClient(
        {
            "reference_analyst": [
                PreparationFakeResponse(
                    trace_payload(),
                    {"prompt_tokens": 11, "total_tokens": 11},
                )
            ],
            "teaching_designer": [ModelResponseError("provider unavailable")],
        }
    )

    with pytest.raises(PreparationFailure) as captured:
        asyncio.run(
            LessonPreparationPipeline(fake).prepare_early(
                problem(), route(), focus_targets()
            )
        )

    analyst, designer = captured.value.audit.role_calls
    assert analyst.token_usage == {"prompt_tokens": 11, "total_tokens": 11}
    assert designer.token_usage is None


@pytest.mark.parametrize(
    "unexpected",
    [AssertionError("fake exhausted"), TypeError("programmer defect")],
)
def test_unexpected_internal_errors_are_not_mapped_to_provider_failure(unexpected):
    fake = PreparationFakeClient(
        {
            "reference_analyst": [unexpected],
            "teaching_designer": [trajectory_payload()],
        }
    )

    with pytest.raises(type(unexpected), match=str(unexpected)):
        asyncio.run(
            LessonPreparationPipeline(fake).prepare_early(
                problem(), route(), focus_targets()
            )
        )


def test_cancellation_propagates_without_a_failure_record():
    fake = PreparationFakeClient(
        {
            "reference_analyst": [asyncio.CancelledError()],
            "teaching_designer": [trajectory_payload()],
        }
    )

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            LessonPreparationPipeline(fake).prepare_early(
                problem(), route(), focus_targets()
            )
        )


def test_run_snapshot_returns_defensive_copies_of_audit_and_artifacts():
    result = run_early(LessonPreparationPipeline(client()))

    first_calls = result.role_calls
    first_calls[0].failure_category = "tampered"
    first_versions = result.versions
    first_versions["solution_trace"] = 99
    first_active_versions = result.active_versions
    first_active_versions["solution_trace"] = 88
    first_trace = result.solution_trace
    first_trace.task_target = "tampered"

    assert result.role_calls[0].failure_category is None
    assert result.versions["solution_trace"] == 1
    assert result.active_versions["solution_trace"] == 1
    assert result.solution_trace.task_target == "m-n"


def test_prepare_with_audit_returns_defensive_approved_lesson_and_full_audit():
    class SuccessfulPipeline(LessonPreparationPipeline):
        async def _continue_preparation(
            self,
            state,
            problem_value,
            teaching_route,
            problem_focus_targets,
            on_stage,
        ):
            del teaching_route, problem_focus_targets, on_stage
            state.role_calls.append(
                RoleCallRecord(
                    role="script_teacher",
                    input_artifact_versions=dict(state.versions),
                    duration_ms=1,
                    retry_count=0,
                    token_usage={"prompt_tokens": 31, "total_tokens": 31},
                )
            )
            payload = prepared_lesson_payload()
            payload["review"]["approval_summary"] = (
                "approved-" + problem_value.problem_text[-1]
            )
            return PreparedLesson.model_validate(payload)

    pipeline = SuccessfulPipeline(client())
    source = problem().model_copy(
        update={"problem_text": problem().problem_text + "A"}
    )

    run = asyncio.run(
        pipeline.prepare_with_audit(source, route(), focus_targets())
    )
    returned = run.prepared_lesson
    returned.review.approval_summary = "tampered"
    returned_calls = run.audit.role_calls
    returned_calls[-1].failure_category = "tampered"

    assert run.prepared_lesson.review.approval_summary == "approved-A"
    assert run.audit.role_calls[-1].role == "script_teacher"
    assert run.audit.role_calls[-1].failure_category is None
    assert len(run.audit.role_calls) == 3


def test_concurrent_full_runs_keep_distinct_lessons_and_downstream_records():
    class ReversedClient:
        def __init__(self):
            self.release_a = asyncio.Event()
            self.finished = []

        async def complete_json_with_metadata(self, system, user):
            role = role_for_system(system)
            is_a = "FULL-A" in user
            if is_a and role == "reference_analyst":
                await self.release_a.wait()
            if role == "reference_analyst":
                conclusion = "m-n=1/2" if is_a else "m-n=3/2"
                usage = 41 if is_a else 51
                return ModelCompletion(
                    trace_payload(final_conclusion=conclusion),
                    {"prompt_tokens": usage, "total_tokens": usage},
                )
            usage = 42 if is_a else 52
            if not is_a:
                self.finished.append("B")
                self.release_a.set()
            else:
                self.finished.append("A")
            return ModelCompletion(
                trajectory_payload(),
                {"prompt_tokens": usage, "total_tokens": usage},
            )

    class SuccessfulPipeline(LessonPreparationPipeline):
        async def _continue_preparation(
            self,
            state,
            source_problem,
            *args,
        ):
            del args
            is_a = source_problem.problem_text.endswith("FULL-A")
            usage = 43 if is_a else 53
            state.role_calls.append(
                RoleCallRecord(
                    role="script_teacher",
                    input_artifact_versions=dict(state.versions),
                    duration_ms=1,
                    retry_count=0,
                    token_usage={
                        "prompt_tokens": usage,
                        "total_tokens": usage,
                    },
                )
            )
            payload = prepared_lesson_payload()
            payload["review"]["approval_summary"] = (
                "full-A" if is_a else "full-B"
            )
            return PreparedLesson.model_validate(payload)

    async def scenario():
        client_value = ReversedClient()
        pipeline = SuccessfulPipeline(client_value)
        problem_a = problem().model_copy(
            update={"problem_text": problem().problem_text + "FULL-A"}
        )
        problem_b = problem().model_copy(
            update={
                "problem_text": problem().problem_text + "FULL-B",
                "reference_answer": "m-n=3/2",
            }
        )
        runs = await asyncio.gather(
            pipeline.prepare_with_audit(
                problem_a, route("m-n=1/2"), focus_targets()
            ),
            pipeline.prepare_with_audit(
                problem_b, route("m-n=3/2"), focus_targets()
            ),
        )
        return client_value, runs

    client_value, (run_a, run_b) = asyncio.run(scenario())

    assert client_value.finished == ["B", "A"]
    assert run_a.prepared_lesson.review.approval_summary == "full-A"
    assert run_b.prepared_lesson.review.approval_summary == "full-B"
    assert [
        call.token_usage["prompt_tokens"] for call in run_a.audit.role_calls
    ] == [41, 42, 43]
    assert [
        call.token_usage["prompt_tokens"] for call in run_b.audit.role_calls
    ] == [51, 52, 53]


def test_real_full_state_machine_reverses_concurrent_runs_without_state_leakage():
    class FullInterleavingClient:
        def __init__(self):
            self.release_a = asyncio.Event()
            self.finished = []

        async def complete_json_with_metadata(self, system, user):
            role = role_for_system(system)
            marker = "RUN-A" if "RUN-A" in user else "RUN-B"
            is_a = marker == "RUN-A"
            if role == "classroom_director" and is_a:
                await self.release_a.wait()

            if role == "reference_analyst":
                payload = trace_payload()
                payload["audit_notes"] = [marker]
            elif role == "teaching_designer":
                payload = trajectory_payload()
                payload["lesson_purpose"] += " " + marker
            elif role == "script_teacher":
                payload = downstream_script_payload()
                payload["title"] += " " + marker
            elif role == "interaction_designer":
                payload = downstream_interaction_payload()
                payload["transfer_item"]["problem_text"] += " " + marker
            elif role == "classroom_director":
                payload = downstream_score_payload()
            elif role == "student_simulator":
                payload = downstream_simulation_payload()
                payload["end_of_lesson_recall"] += " " + marker
            else:
                payload = downstream_review_payload()
                payload["approval_summary"] += " " + marker
                self.finished.append(marker)
                if not is_a:
                    self.release_a.set()
            usage = 101 if is_a else 202
            return ModelCompletion(
                payload,
                {"prompt_tokens": usage, "total_tokens": usage},
            )

    async def scenario():
        fake = FullInterleavingClient()
        pipeline = LessonPreparationPipeline(fake)
        problem_a = problem().model_copy(
            update={"problem_text": problem().problem_text + " RUN-A"}
        )
        problem_b = problem().model_copy(
            update={"problem_text": problem().problem_text + " RUN-B"}
        )
        runs = await asyncio.gather(
            pipeline.prepare_with_audit(
                problem_a, route(), focus_targets()
            ),
            pipeline.prepare_with_audit(
                problem_b, route(), focus_targets()
            ),
        )
        return fake, runs

    fake, (run_a, run_b) = asyncio.run(scenario())

    assert fake.finished == ["RUN-B", "RUN-A"]
    assert run_a.prepared_lesson.teaching_script.title.endswith("RUN-A")
    assert run_b.prepared_lesson.teaching_script.title.endswith("RUN-B")
    assert run_a.prepared_lesson.review.approval_summary.endswith("RUN-A")
    assert run_b.prepared_lesson.review.approval_summary.endswith("RUN-B")
    assert run_a.audit.active_versions == run_a.audit.versions
    assert run_b.audit.active_versions == run_b.audit.versions
    assert {
        call.token_usage["prompt_tokens"] for call in run_a.audit.role_calls
    } == {101}
    assert {
        call.token_usage["prompt_tokens"] for call in run_b.audit.role_calls
    } == {202}


def test_prepare_compatibility_method_returns_only_defensive_prepared_lesson():
    class SuccessfulPipeline(LessonPreparationPipeline):
        async def _continue_preparation(self, state, *args):
            del state, args
            return PreparedLesson.model_validate(prepared_lesson_payload())

    pipeline = SuccessfulPipeline(client())

    lesson = asyncio.run(
        pipeline.prepare(problem(), route(), focus_targets())
    )

    assert type(lesson) is PreparedLesson
    assert not hasattr(lesson, "audit")


def test_downstream_preparation_failure_receives_current_request_audit():
    class FailingPipeline(LessonPreparationPipeline):
        async def _continue_preparation(self, state, *args):
            del args
            state.role_calls.append(
                RoleCallRecord(
                    role="script_teacher",
                    input_artifact_versions=dict(state.versions),
                    duration_ms=1,
                    retry_count=0,
                    failure_category="invalid_structure",
                )
            )
            raise PreparationFailure(
                "invalid_structure",
                "script_teacher",
                "模型输出结构无效。",
            )

    with pytest.raises(PreparationFailure) as captured:
        asyncio.run(
            FailingPipeline(client()).prepare_with_audit(
                problem(), route(), focus_targets()
            )
        )

    assert captured.value.audit is not None
    assert captured.value.audit.versions == {
        "solution_trace": 1,
        "reasoning_trajectory": 1,
    }
    assert [call.role for call in captured.value.audit.role_calls] == [
        "reference_analyst",
        "teaching_designer",
        "script_teacher",
    ]


def test_cumulative_usage_overflow_drops_the_entire_usage_record():
    maximum = 1_000_000_000
    fake = PreparationFakeClient(
        {
            "reference_analyst": [
                PreparationFakeResponse(
                    {},
                    {
                        "prompt_tokens": maximum,
                        "completion_tokens": 2,
                        "total_tokens": maximum,
                    },
                ),
                PreparationFakeResponse(
                    trace_payload(),
                    {
                        "prompt_tokens": 1,
                        "completion_tokens": 3,
                        "total_tokens": 1,
                    },
                ),
            ],
            "teaching_designer": [trajectory_payload()],
        }
    )

    result = run_early(LessonPreparationPipeline(fake))

    assert result.role_calls[0].token_usage is None


def test_accept_artifact_checks_record_before_mutating_state():
    pipeline = LessonPreparationPipeline(client())
    state = PreparationState(
        role_calls=[
            RoleCallRecord(
                role="teaching_designer",
                duration_ms=0,
                retry_count=0,
            )
        ]
    )

    with pytest.raises(RuntimeError, match="does not match"):
        pipeline._accept_artifact(
            state,
            artifact_type="solution_trace",
            responsible_role="reference_analyst",
            artifact=SolutionTrace.model_validate(trace_payload()),
        )

    assert state.solution_trace is None
    assert state.versions == {}
    assert state.history == []
