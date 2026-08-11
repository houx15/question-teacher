import asyncio
import json

import pytest

from app.llm_client import ModelResponseError
from app.preparation_models import ReasoningTrajectory, SolutionTrace
from app.preparation_pipeline import (
    LessonPreparationPipeline,
    PreparationFailure,
)
from app.schemas import ProblemFocusTarget, ProblemInput, ReferenceGroundingBrief
from app.teaching_route import freeze_grounded_route
from tests.preparation_fakes import (
    PreparationFakeClient,
    PreparationFakeResponse,
)


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


def route():
    statements = (
        ("题目给出x=2n是根", "把x=2n代入原方程", "4n^2-4mn+2n=0"),
        ("4n^2-4mn+2n=0", "观察目标只需要m-n的关系", "2n(2n-2m+1)=0"),
        ("2n(2n-2m+1)=0", "利用n不等于0约去2n", "2n-2m+1=0"),
        ("2n-2m+1=0", "整理并回到目标m-n", "m-n=1/2"),
    )
    brief = ReferenceGroundingBrief.validate_for_reference_answer(
        {
            "task_summary": "由参数根求m-n",
            "target": "m-n",
            "assumptions": ["n不等于0", "x=2n是原方程的根"],
            "reference_conclusion": "m-n=1/2",
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
        "m-n=1/2",
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


def client(trace=None, trajectory=None):
    return PreparationFakeClient(
        {
            "reference_analyst": [trace or trace_payload()],
            "teaching_designer": [trajectory or trajectory_payload()],
        }
    )


def run_early(pipeline, on_stage=None):
    with pytest.raises(NotImplementedError, match="downstream preparation"):
        asyncio.run(
            pipeline.prepare(
                problem(), route(), focus_targets(), on_stage=on_stage
            )
        )


def test_trace_and_trajectory_stages_run_in_dependency_order():
    fake = client()
    stages = []
    pipeline = LessonPreparationPipeline(fake)

    run_early(pipeline, on_stage=stages.append)

    assert [call.role for call in fake.calls] == [
        "reference_analyst",
        "teaching_designer",
    ]
    assert stages == ["整理参考解析", "设计解题思维轨迹"]
    assert pipeline.last_state is not None
    assert isinstance(pipeline.last_state.solution_trace, SolutionTrace)
    assert isinstance(
        pipeline.last_state.reasoning_trajectory, ReasoningTrajectory
    )
    assert pipeline.last_state.versions == {
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

    run_early(pipeline)

    assert [call.role for call in fake.calls] == [
        "reference_analyst",
        "reference_analyst",
        "teaching_designer",
        "teaching_designer",
    ]
    assert [record.retry_count for record in pipeline.role_calls] == [1, 1]
    assert [record.failure_category for record in pipeline.role_calls] == [None, None]


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
            pipeline.prepare(problem(), route(), focus_targets())
        )

    failure = captured.value
    assert failure.category == "invalid_structure"
    assert failure.role == failing_role
    assert failure.detail == "模型输出结构无效。"
    assert RAW_REFERENCE_MARKER not in str(failure)
    assert pipeline.role_calls[-1].failure_category == "invalid_structure"
    assert pipeline.role_calls[-1].output_artifact_type is None


def test_invalid_json_response_gets_the_same_single_structure_retry():
    fake = PreparationFakeClient(
        {
            "reference_analyst": [
                ModelResponseError("Model response content is not valid JSON."),
                trace_payload(),
            ],
            "teaching_designer": [trajectory_payload()],
        }
    )
    pipeline = LessonPreparationPipeline(fake)

    run_early(pipeline)

    assert [call.role for call in fake.calls].count("reference_analyst") == 2
    assert pipeline.role_calls[0].retry_count == 1


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
            pipeline.prepare(problem(), route(), focus_targets())
        )

    assert captured.value.category == "provider_error"
    assert captured.value.role == "reference_analyst"
    assert captured.value.detail == "模型服务暂时不可用。"
    assert provider_message not in str(captured.value)
    assert len(fake.calls) == 1
    assert pipeline.role_calls[-1].failure_category == "provider_error"


def test_deterministic_trace_failure_stops_before_designer_without_retry():
    fake = client(trace=trace_payload(final_conclusion="m-n=3/2"))
    pipeline = LessonPreparationPipeline(fake)

    with pytest.raises(PreparationFailure) as captured:
        asyncio.run(
            pipeline.prepare(problem(), route(), focus_targets())
        )

    assert captured.value.category == "reference_trace_failed"
    assert captured.value.role == "reference_analyst"
    assert [call.role for call in fake.calls] == ["reference_analyst"]
    assert pipeline.last_state.versions == {}
    assert pipeline.role_calls[-1].failure_category == "reference_trace_failed"
    assert pipeline.role_calls[-1].output_artifact_version is None


def test_deterministic_trajectory_failure_is_not_a_structure_retry():
    invalid_trajectory = trajectory_payload()
    invalid_trajectory["episodes"][0]["source_step_ids"] = ["missing-step"]
    fake = client(trajectory=invalid_trajectory)
    pipeline = LessonPreparationPipeline(fake)

    with pytest.raises(PreparationFailure) as captured:
        asyncio.run(
            pipeline.prepare(problem(), route(), focus_targets())
        )

    assert captured.value.category == "reasoning_design_failed"
    assert captured.value.role == "teaching_designer"
    assert [call.role for call in fake.calls] == [
        "reference_analyst",
        "teaching_designer",
    ]
    assert pipeline.last_state.versions == {"solution_trace": 1}
    assert pipeline.role_calls[-1].failure_category == "reasoning_design_failed"


def test_plan_execute_monitor_revise_execute_trajectory_is_accepted():
    pipeline = LessonPreparationPipeline(client())

    run_early(pipeline)

    assert [
        episode.mode
        for episode in pipeline.last_state.reasoning_trajectory.episodes
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

    run_early(pipeline)

    accepted = pipeline.last_state.reasoning_trajectory
    assert accepted.trajectory_type == trajectory_type
    assert tuple(episode.mode for episode in accepted.episodes) == modes


def test_parameter_root_trajectory_preserves_four_indispensable_moves():
    pipeline = LessonPreparationPipeline(client())

    run_early(pipeline)

    trajectory = pipeline.last_state.reasoning_trajectory
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
                    token_usage={"input": 321, "output": 123},
                )
            ],
            "teaching_designer": [trajectory_payload()],
        }
    )
    pipeline = LessonPreparationPipeline(fake)

    run_early(pipeline)

    analyst, designer = pipeline.role_calls
    assert analyst.input_artifact_versions == {}
    assert analyst.output_artifact_type == "solution_trace"
    assert analyst.output_artifact_version == 1
    assert analyst.token_usage == {"input": 321, "output": 123}
    assert designer.input_artifact_versions == {"solution_trace": 1}
    assert designer.output_artifact_type == "reasoning_trajectory"
    assert designer.output_artifact_version == 1
    serialized = json.dumps(
        [record.model_dump(mode="json") for record in pipeline.role_calls],
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

        async def complete_json(self, system, user):
            if "参考材料分析员" in system:
                self.analyst_arrivals += 1
                if self.analyst_arrivals == 2:
                    self.both_analysts_started.set()
                await self.both_analysts_started.wait()
            return await super().complete_json(system, user)

    async def scenario():
        pipeline = LessonPreparationPipeline(InterleavingClient())
        return await asyncio.gather(
            pipeline.prepare(problem(), route(), focus_targets()),
            pipeline.prepare(problem(), route(), focus_targets()),
            return_exceptions=True,
        )

    results = asyncio.run(scenario())

    assert len(results) == 2
    assert all(isinstance(result, NotImplementedError) for result in results)
