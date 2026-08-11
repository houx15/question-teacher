import asyncio
import json

import pytest

from app.generation import (
    GeneratedLessonBundle,
    LessonGenerationService,
    LessonInputError,
    LessonQualityError,
)
from app.math_engine import MathEngine
from app.prompts import REFERENCE_AUDITOR_SYSTEM
from app.preparation_pipeline import (
    LessonPreparationPipeline,
    PreparationFailure,
)
from app.problem_focus import compile_problem_focus_targets
from app.schemas import ProblemInput, ReferenceMaterialAudit
from app.teaching_route import TeachingRouteMode
from tests.generation_fakes import (
    CompositeGenerationClient,
    FakeClient,
)
from tests.preparation_fakes import PreparationFakeClient
from tests.test_preparation_pipeline import (
    downstream_interaction_payload,
    downstream_review_payload,
    downstream_score_payload,
    downstream_script_payload,
    downstream_simulation_payload,
    problem as preparation_problem,
    route as preparation_route,
    trace_payload,
    trajectory_payload,
)


def problem(required_method="factor", reference_solution_text=None):
    return ProblemInput(
        problem_text="用指定方法解方程：x^2-5x+6=0",
        reference_answer="x=2 或 x=3",
        reference_solution_text=reference_solution_text,
        required_method=required_method,
    )


def valid_draft():
    return {
        "title": "把二次式拆成两个一次因式",
        "learning_goal": "理解因式分解如何把二次方程变成两个一次方程。",
        "opening": "先观察原式：哪两个数相乘是 6，相加是 -5？",
        "method_rationale": "首项系数为 1，常数 6 可拆成两个整数的乘积。",
        "method_introduction": {
            "method_name": "因式分解法",
            "student_definition": "把二次式写成两个一次因式的乘积，再分别令每个因式为零。",
            "target_form": r"\((x-a)(x-b)=0\)",
            "why_it_helps": "零乘积性质把一个二次方程拆成两个更容易解的一次方程。",
        },
        "math_steps": [
            {
                "purpose": "因式分解",
                "operation": "factor",
                "operands": [],
                "state_before": ["x^2-5x+6=0"],
                "state_after": ["(x-2)(x-3)=0"],
                "reason": "两个数相乘为 6、相加为 -5。",
            }
        ],
        "moments": [
            {
                "purpose": "寻找因数关系",
                "sync_cues": [
                    {
                        "cue_id": "find-factor-pair-cue",
                        "spoken_text": (
                            "先自己找一找：哪两个数满足乘积与和的条件？"
                        ),
                        "start_actions": [
                            {
                                "surface": "board",
                                "type": "write",
                                "target": "equation",
                                "content": r"\(x^2-5x+6=0\)",
                            },
                            {
                                "surface": "board",
                                "type": "write",
                                "target": "constant_and_linear_terms",
                                "content": "常数项 6，一次项系数 -5",
                            },
                            {
                                "surface": "board",
                                "type": "focus",
                                "target": "constant_and_linear_terms",
                            },
                        ],
                    }
                ],
                "layer": "interaction",
                "interaction": valid_diagnostic_choice(),
            },
            {
                "purpose": "写出因式分解",
                "sync_cues": [
                    {
                        "cue_id": "write-factorization-cue",
                        "spoken_text": (
                            "用刚才找到的两个数，把二次式写成两个一次因式。"
                        ),
                        "start_actions": [
                            {
                                "surface": "board",
                                "type": "transform",
                                "target": "equation",
                                "content": r"\((x-2)(x-3)=0\)",
                            }
                        ],
                    }
                ],
            },
        ],
        "summary": "因式分解后，让每个一次因式分别等于零。",
        "transfer_item": {
            "problem_text": "用因式分解法解方程：x^2-7x+12=0",
            "expected_answer": "x=3 或 x=4",
            "method_signal": "寻找乘积为 12、和为 -7 的两个数。",
            "options": [
                {
                    "option_id": "both-roots",
                    "label": r"\(x=3\) 或 \(x=4\)",
                    "canonical_answer": "x=3 或 x=4",
                    "feedback": "两个根都能使原方程成立。",
                },
                {
                    "option_id": "only-three",
                    "label": r"\(x=3\)",
                    "canonical_answer": "x=3",
                    "feedback": "还遗漏了另一个根。",
                },
                {
                    "option_id": "only-four",
                    "label": r"\(x=4\)",
                    "canonical_answer": "x=4",
                    "feedback": "还遗漏了另一个根。",
                },
            ],
            "correct_option_id": "both-roots",
        },
    }


def valid_diagnostic_choice():
    return {
        "interaction_id": "find-factor-pair",
        "kind": "choice",
        "prompt": "哪一组数同时满足乘积为 6、和为 -5？",
        "expected_answer": "negative-two-negative-three",
        "options": [
            {
                "option_id": "negative-two-negative-three",
                "label": r"\(-2\) 和 \(-3\)",
                "feedback": "这组数的乘积为 6、和为 -5，正好满足条件。",
            },
            {
                "option_id": "two-three",
                "label": r"\(2\) 和 \(3\)",
                "feedback": "乘积是 6，但和是 5，需要注意一次项系数的符号。",
            },
            {
                "option_id": "negative-one-negative-six",
                "label": r"\(-1\) 和 \(-6\)",
                "feedback": "和是 -7，不符合一次项系数。",
            },
        ],
        "hints": ["同时检查乘积和相加结果。"],
        "explanation_after_correct": "这组数同时满足两个条件。",
    }


def reference_audit_payload():
    return {
        "status": "approved",
        "claimed_answer": "x=2 或 x=3",
        "method_summary": "因式分解法",
        "key_steps": [
            {
                "purpose": "因式分解",
                "operation": "factor",
                "operands": [],
                "state_before": ["x^2-5*x+6=0"],
                "state_after": ["(x-2)*(x-3)=0"],
                "reason": "乘积为 6、和为 -5。",
            }
        ],
        "teaching_assets": ["先观察乘积与和的关系。"],
        "warnings": [],
        "blocking_issues": [],
        "evidence": ["所以 x=2 或 x=3。"],
    }


def _approved_preparation_client():
    preparation_client = PreparationFakeClient(
        {
            "reference_analyst": [trace_payload()],
            "teaching_designer": [trajectory_payload()],
            "script_teacher": [downstream_script_payload()],
            "interaction_designer": [downstream_interaction_payload()],
            "classroom_director": [downstream_score_payload()],
            "student_simulator": [downstream_simulation_payload()],
            "lesson_reviewer": [downstream_review_payload()],
        }
    )
    return CompositeGenerationClient(FakeClient([]), preparation_client)


def test_reference_auditor_receives_raw_reference_as_untrusted_input():
    marker = "PRIVATE-AUDIT-MARKER-TASK7"
    source_problem = problem(reference_solution_text=marker)
    client = FakeClient([reference_audit_payload()])
    service = LessonGenerationService(client, MathEngine())

    audit = asyncio.run(service._audit_reference(source_problem, ["2", "3"]))

    assert audit.status == "approved"
    assert client.system_prompts == [REFERENCE_AUDITOR_SYSTEM]
    assert marker in client.user_prompts[0]
    assert "不可信" in REFERENCE_AUDITOR_SYSTEM


@pytest.mark.parametrize("failure_kind", ["rejected", "claim", "step"])
def test_reference_audit_conflicts_block_generation_as_safe_input_errors(
    failure_kind,
):
    payload = reference_audit_payload()
    if failure_kind == "rejected":
        payload.update(
            {
                "status": "rejected",
                "blocking_issues": ["解析内部冲突"],
            }
        )
    elif failure_kind == "claim":
        payload["claimed_answer"] = "x=99"
    else:
        payload["key_steps"][0]["state_after"] = ["(x-1)*(x-6)=0"]
    audit = ReferenceMaterialAudit.model_validate(payload)

    with pytest.raises(LessonInputError, match="数学冲突"):
        LessonGenerationService(
            FakeClient([]),
            MathEngine(),
        )._validate_reference_audit(problem(), audit)


def test_reference_audit_invalid_schema_is_a_quality_error_not_user_blame():
    service = LessonGenerationService(
        FakeClient([{"status": "approved", "private": "bad-shape"}]),
        MathEngine(),
    )

    with pytest.raises(LessonQualityError, match="审阅结构无效"):
        asyncio.run(
            service._audit_reference(
                problem(reference_solution_text="一段参考解析"),
                ["2", "3"],
            )
        )


class _RecordingPreparationPipeline(LessonPreparationPipeline):
    def __init__(self, client, events):
        super().__init__(client)
        self.events = events
        self.received = None

    async def prepare_with_audit(
        self,
        source_problem,
        teaching_route,
        problem_focus_targets,
        on_stage=None,
    ):
        self.events.append("preparation")
        self.received = (
            source_problem,
            teaching_route,
            list(problem_focus_targets),
        )
        return await super().prepare_with_audit(
            source_problem,
            teaching_route,
            problem_focus_targets,
            on_stage,
        )


def test_generate_bundle_uses_approved_preparation_and_keeps_private_evidence_out_of_runtime():
    events = []
    client = _approved_preparation_client()
    pipeline = _RecordingPreparationPipeline(client, events)
    service = LessonGenerationService(
        client,
        MathEngine(),
        preparation_pipeline=pipeline,
    )

    async def grounded_route(source_problem, on_stage):
        events.append("route")
        return preparation_route()

    service._build_grounded_teaching_route = grounded_route
    source_problem = preparation_problem().model_copy(
        update={
            "problem_text": (
                "若$2n$ ($n\\ne 0$)是关于 x的方程 "
                "$x^2-2mx+2n=0$的根，则m-n的值为"
            )
        }
    )
    bundle = asyncio.run(service.generate_bundle(source_problem))

    assert isinstance(bundle, GeneratedLessonBundle)
    assert events[:2] == ["route", "preparation"]
    received_problem, received_route, received_targets = pipeline.received
    assert received_problem == source_problem
    assert received_route.fingerprint == preparation_route().fingerprint
    assert received_targets == compile_problem_focus_targets(
        source_problem.problem_text
    )
    assert [target.math_text for target in received_targets] == [
        "2n",
        "n\\ne 0",
        "x^2-2mx+2n=0",
    ]
    assert bundle.generation_record.lesson_id == bundle.lesson.lesson_id
    assert bundle.generation_record.route_fingerprint == received_route.fingerprint
    assert bundle.generation_record.prepared_lesson.review.status == "approved"
    assert len(bundle.generation_record.role_calls) == 7
    provenance = bundle.generation_record.cue_provenance
    assert [item.clause_id for item in provenance] == [
        clause.clause_id
        for clause in bundle.generation_record.prepared_lesson.teaching_script.clauses
    ]
    assert [item.spoken_text for item in provenance] == [
        clause.spoken_text
        for clause in bundle.generation_record.prepared_lesson.teaching_script.clauses
    ]
    runtime_cues = {
        cue.cue_id: cue
        for beat in bundle.lesson.beats
        for cue in beat.sync_cues
        if cue.cue_id not in {
            "runtime-transfer-intro-cue",
        }
    }
    assert {item.runtime_cue_id for item in provenance} == set(runtime_cues)
    for item in provenance:
        assert item.episode_id
        assert item.original_performance_cue_id
        assert item.spoken_text in runtime_cues[item.runtime_cue_id].spoken_text
    assert bundle.lesson.validation_report == {
        "verification_mode": "reference_grounded",
        "consistency_status": "consistent",
        "teaching_route_fingerprint": received_route.fingerprint,
        "pedagogy_rubric_version": (
            bundle.generation_record.prepared_lesson.rubric_version
        ),
        "artifact_versions": {
            "solution_trace": 1,
            "reasoning_trajectory": 1,
            "teaching_script": 1,
            "interaction_plan": 1,
            "performance_score": 1,
            "simulation_report": 1,
        },
        "repair_count": 0,
        "review_status": "approved",
        "review_assessment": "核心门槛通过",
    }
    serialized_report = json.dumps(
        bundle.lesson.validation_report,
        ensure_ascii=False,
    )
    assert "source_steps" not in serialized_report
    assert "clauses" not in serialized_report
    assert "episode_results" not in serialized_report
    assert bundle.lesson.title == "从参数根到目标关系"
    assert any(
        cue.spoken_text == "根一定满足原方程，所以先代入。"
        for beat in bundle.lesson.beats
        for cue in beat.sync_cues
    )


def test_generate_remains_runtime_lesson_compatibility_wrapper():
    client = _approved_preparation_client()
    service = LessonGenerationService(
        client,
        MathEngine(),
        preparation_pipeline=LessonPreparationPipeline(client),
    )

    async def grounded_route(source_problem, on_stage):
        return preparation_route()

    service._build_grounded_teaching_route = grounded_route
    lesson = asyncio.run(service.generate(preparation_problem()))

    assert lesson.title == "从参数根到目标关系"


def test_symbolic_verified_route_is_frozen_before_preparation():
    class CapturingPipeline:
        def __init__(self):
            self.route = None

        async def prepare_with_audit(
            self,
            source_problem,
            teaching_route,
            problem_focus_targets,
            on_stage=None,
        ):
            del source_problem, problem_focus_targets, on_stage
            self.route = teaching_route
            raise PreparationFailure(
                category="review_not_converged",
                role="lesson_reviewer",
                detail="课程审核未收敛。",
            )

    pipeline = CapturingPipeline()
    source_problem = ProblemInput(
        problem_text="用配方法解方程：x^2-6*x+5=0",
        reference_answer="x=1 或 x=5",
        required_method="complete_the_square",
    )
    service = LessonGenerationService(
        FakeClient([]),
        MathEngine(),
        preparation_pipeline=pipeline,
    )

    with pytest.raises(PreparationFailure, match="课程审核未收敛"):
        asyncio.run(service.generate_bundle(source_problem))

    assert pipeline.route.mode == TeachingRouteMode.SYMBOLIC_VERIFIED
    assert pipeline.route.fingerprint
    assert pipeline.route.to_prompt_payload()["steps"]


def test_nonconverged_preparation_never_reaches_compiler():
    class FailingPipeline:
        async def prepare_with_audit(self, *args, **kwargs):
            raise PreparationFailure(
                category="review_not_converged",
                role="lesson_reviewer",
                detail="课程审核未收敛。",
            )

    class CompilerSpy:
        def __init__(self):
            self.calls = 0

        def compile(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError("compiler must not run")

    compiler = CompilerSpy()
    service = LessonGenerationService(
        FakeClient([]),
        MathEngine(),
        compiler=compiler,
        preparation_pipeline=FailingPipeline(),
    )

    async def grounded_route(source_problem, on_stage):
        return preparation_route()

    service._build_grounded_teaching_route = grounded_route
    with pytest.raises(PreparationFailure, match="课程审核未收敛"):
        asyncio.run(service.generate_bundle(preparation_problem()))

    assert compiler.calls == 0


def test_raw_reference_solution_is_visible_only_to_reference_analysis():
    marker = "PRIVATE-REFERENCE-TASK7-7b39"
    source_problem = preparation_problem().model_copy(
        update={"reference_solution_text": marker}
    )
    client = _approved_preparation_client()
    service = LessonGenerationService(
        client,
        MathEngine(),
        preparation_pipeline=LessonPreparationPipeline(client),
    )

    async def grounded_route(problem_input, on_stage):
        return preparation_route()

    service._build_grounded_teaching_route = grounded_route
    asyncio.run(service.generate_bundle(source_problem))

    calls_with_marker = [
        call.role
        for call in client.preparation_calls
        if marker in call.user
    ]
    assert calls_with_marker == ["reference_analyst"]
