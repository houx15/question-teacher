import asyncio
import inspect
import json

import httpx
import pytest

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
            + "：将x=2n代入，连接目标m-n；利用n不等于0约去因式，"
            "最后回到m-n=1/2。"
        ),
    )


def route(final_conclusion="m-n=1/2"):
    statements = (
        ("题目给出x=2n是根", "把x=2n代入原方程", "4n^2-4mn+2n=0"),
        ("4n^2-4mn+2n=0", "观察目标只需要m-n的关系", "2n(2n-2m+1)=0"),
        ("2n(2n-2m+1)=0", "利用n不等于0约去2n", "2n-2m+1=0"),
        ("2n-2m+1=0", "整理并回到目标m-n", final_conclusion),
    )
    brief = ReferenceGroundingBrief.validate_for_reference_answer(
        {
            "task_summary": "由参数根求m-n",
            "target": "m-n",
            "assumptions": ["n不等于0", "x=2n是原方程的根"],
            "reference_conclusion": final_conclusion,
            "method_name": "代入法",
            "reasoning_steps": [
                {
                    "step_id": step_id,
                    "statement_before": before,
                    "operation_explanation": action,
                    "statement_after": after,
                }
                for step_id, (before, action, after) in zip(STEP_IDS, statements)
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
        ("题目条件", "代入已知根x=2n", "4n^2-4mn+2n=0"),
        ("4n^2-4mn+2n=0", "连接到目标关系m-n", "2n(2n-2m+1)=0"),
        ("2n(2n-2m+1)=0", "使用n不等于0约去2n", "2n-2m+1=0"),
        ("2n-2m+1=0", "重新回到m-n并整理", "m-n=1/2"),
    )
    return {
        "task_target": "求m-n",
        "reference_conclusion": final_conclusion,
        "assumptions": [
            {
                "assumption_id": "assumption-nonzero",
                "content": "n不等于0",
                "source_anchor": {
                    "source_kind": "problem",
                    "source_id": "problem-nonzero",
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
                "state_before": before,
                "mathematical_action": action,
                "justification": "保留参考解析与题目条件的数学依赖",
                "state_after": after,
                "new_information": "得到下一步所需关系",
                "assumption_ids_used": (
                    ["assumption-nonzero"] if step_id == "use-nonzero" else []
                ),
                "omitted_reasoning": [],
                "evidence_status": "verified_route",
            }
            for step_id, (before, action, after) in zip(STEP_IDS, states)
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


def client(
    trace=None,
    trajectory=None,
    script=None,
    interaction=None,
    performance=None,
):
    return PreparationFakeClient(
        {
            "reference_analyst": [trace or trace_payload()],
            "teaching_designer": [trajectory or trajectory_payload()],
            "script_teacher": [script or downstream_script_payload()],
            "interaction_designer": [
                interaction or downstream_interaction_payload()
            ],
            "classroom_director": [
                performance or downstream_score_payload()
            ],
        }
    )


def run_early(pipeline, on_stage=None):
    return asyncio.run(
        pipeline.prepare_early(
            problem(), route(), focus_targets(), on_stage=on_stage
        )
    )


def test_public_prepare_cannot_return_a_partial_prepared_lesson():
    with pytest.raises(NotImplementedError, match="simulation and review"):
        asyncio.run(
            LessonPreparationPipeline(client()).prepare(
                problem(), route(), focus_targets()
            )
        )


def test_script_interaction_and_performance_stages_run_in_dependency_order():
    fake = client()
    stages = []

    with pytest.raises(NotImplementedError, match="simulation and review"):
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
    ]
    assert stages == [
        "整理参考解析",
        "设计解题思维轨迹",
        "编写讲稿",
        "设计互动",
        "编排板书与高亮",
    ]


def test_zero_interactions_and_cues_without_highlights_are_accepted():
    fake = client(
        interaction=downstream_interaction_payload(),
        performance=downstream_score_payload(),
    )

    with pytest.raises(NotImplementedError, match="simulation and review"):
        asyncio.run(
            LessonPreparationPipeline(fake).prepare(
                problem(), route(), focus_targets()
            )
        )

    assert [call.role for call in fake.calls][-2:] == [
        "interaction_designer",
        "classroom_director",
    ]


def test_script_dependency_reordering_fails_without_structure_retry():
    invalid = downstream_script_payload()
    invalid["clauses"][2], invalid["clauses"][3] = (
        invalid["clauses"][3],
        invalid["clauses"][2],
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
    invalid["cues"][4]["lead_actions"] = [
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

    with pytest.raises(NotImplementedError, match="simulation and review"):
        asyncio.run(
            LessonPreparationPipeline(
                client(script=script, performance=valid)
            ).prepare(problem(), route(), focus_targets())
        )


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


def test_raw_reference_solution_reaches_only_reference_analyst():
    fake = client()

    run_early(LessonPreparationPipeline(fake))

    analyst_calls = [call for call in fake.calls if call.role == "reference_analyst"]
    designer_calls = [call for call in fake.calls if call.role == "teaching_designer"]
    assert all(RAW_REFERENCE_MARKER in call.user for call in analyst_calls)
    assert all(RAW_REFERENCE_MARKER not in call.user for call in designer_calls)


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
    first_trace = result.solution_trace
    first_trace.task_target = "tampered"

    assert result.role_calls[0].failure_category is None
    assert result.versions["solution_trace"] == 1
    assert result.solution_trace.task_target == "求m-n"


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
