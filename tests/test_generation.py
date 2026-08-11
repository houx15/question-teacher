import asyncio
import json

import pytest
from pydantic import ValidationError

from app.api import public_lesson_payload
from app.claim_checker import ClaimCheckerUnavailableError
from app.compiler import LessonCompileError, LessonCompiler
from app.generation import (
    GeneratedLessonBundle,
    LessonGenerationService,
    LessonInputError,
    LessonQualityError,
)
from app.llm_client import ModelResponseError
from app.math_engine import MathEngine
from app.prompts import REFERENCE_AUDITOR_SYSTEM
from app.preparation_pipeline import (
    LessonPreparationPipeline,
    PreparationFailure,
)
from app.problem_focus import compile_problem_focus_targets
from app.schemas import (
    ProblemInput,
    ReferenceMaterialAudit,
    SyncVisualAction,
)
from app.teaching_route import TeachingRouteMode
from tests.generation_fakes import (
    CompositeGenerationClient,
    FakeClient,
)
from tests.preparation_fakes import PreparationFakeClient
from tests.test_preparation_pipeline import (
    RAW_REFERENCE_MARKER,
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


def grounding_payload(check_requests=None):
    frozen_payload = preparation_route().to_prompt_payload()
    return {
        "task_summary": "由参数根求m-n",
        "target": "m-n",
        "assumptions": [
            {
                "assumption_id": item["assumption_id"],
                "expression": item["expression"],
            }
            for item in frozen_payload["assumptions"]
        ],
        "reference_conclusion": frozen_payload["final_conclusion"],
        "method_name": frozen_payload["method_name"],
        "reasoning_steps": [
            {
                key: value
                for key, value in step.items()
                if key
                not in {
                    "allowed_reasoning_gap_codes",
                    "evidence_status",
                    "source_kind",
                }
            }
            for step in frozen_payload["steps"]
        ],
        "check_requests": list(check_requests or []),
        "audit_notes": [],
    }


def grounded_source_problem(reference_solution_text=None):
    return preparation_problem().model_copy(
        update={
            "problem_text": (
                "若$2n$ ($n\\ne 0$)是关于 x的方程 "
                "$x^2-2mx+2n=0$的根，则m-n的值为"
            ),
            "reference_solution_text": reference_solution_text,
        }
    )


def _approved_preparation_client(route_responses=None, performance=None):
    preparation_client = PreparationFakeClient(
        {
            "reference_analyst": [trace_payload()],
            "teaching_designer": [trajectory_payload()],
            "script_teacher": [downstream_script_payload()],
            "interaction_designer": [downstream_interaction_payload()],
            "classroom_director": [
                performance or downstream_score_payload()
            ],
            "student_simulator": [downstream_simulation_payload()],
            "lesson_reviewer": [downstream_review_payload()],
        }
    )
    return CompositeGenerationClient(
        FakeClient(route_responses or []),
        preparation_client,
    )


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


def test_real_grounder_precedes_preparation_and_freezes_route_and_focus_targets():
    client = _approved_preparation_client([grounding_payload()])
    pipeline = _RecordingPreparationPipeline(client, [])
    service = LessonGenerationService(
        client,
        MathEngine(),
        preparation_pipeline=pipeline,
    )
    source_problem = grounded_source_problem(reference_solution_text=None)

    bundle = asyncio.run(service.generate_bundle(source_problem))

    assert [call.role for call in client.calls] == [
        "reference_grounder",
        "reference_analyst",
        "teaching_designer",
        "script_teacher",
        "interaction_designer",
        "classroom_director",
        "student_simulator",
        "lesson_reviewer",
    ]
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
    assert bundle.lesson.validation_report["verification_mode"] == (
        "reference_grounded"
    )


def test_grounded_raw_reference_reaches_only_grounder_and_reference_analyst():
    marker = "PRIVATE-GROUNDED-REFERENCE-TASK7"
    client = _approved_preparation_client([grounding_payload()])
    service = LessonGenerationService(client, MathEngine())

    asyncio.run(
        service.generate_bundle(
            grounded_source_problem(reference_solution_text=marker)
        )
    )

    assert [call.role for call in client.calls if marker in call.user] == [
        "reference_grounder",
        "reference_analyst",
    ]


def test_grounded_reference_anchor_marker_is_absent_from_all_downstream_outputs():
    trace = trace_payload()
    trace["source_steps"][0]["source_anchor"]["excerpt"] = (
        RAW_REFERENCE_MARKER
    )
    preparation_client = PreparationFakeClient(
        {
            "reference_analyst": [trace],
            "teaching_designer": [trajectory_payload()],
            "script_teacher": [downstream_script_payload()],
            "interaction_designer": [downstream_interaction_payload()],
            "classroom_director": [downstream_score_payload()],
            "student_simulator": [downstream_simulation_payload()],
            "lesson_reviewer": [downstream_review_payload()],
        }
    )
    client = CompositeGenerationClient(
        FakeClient([grounding_payload()]), preparation_client
    )
    source = grounded_source_problem(
        reference_solution_text=(
            RAW_REFERENCE_MARKER + "\n将已知根代入并整理。"
        )
    )

    bundle = asyncio.run(
        LessonGenerationService(client, MathEngine()).generate_bundle(source)
    )

    assert all(
        RAW_REFERENCE_MARKER not in call.user
        for call in client.calls
        if call.role not in {"reference_grounder", "reference_analyst"}
    )
    assert RAW_REFERENCE_MARKER not in json.dumps(
        public_lesson_payload(bundle.lesson), ensure_ascii=False
    )
    assert (
        RAW_REFERENCE_MARKER
        not in bundle.generation_record.prepared_lesson.model_dump_json()
    )


def test_reference_analyst_prose_is_rebuilt_from_frozen_route_before_downstream():
    marker = "这是只供内部审核的批注不要公开"
    trace = trace_payload()
    trace["source_steps"][0]["mathematical_action"] = marker
    preparation_client = PreparationFakeClient(
        {
            "reference_analyst": [trace],
            "teaching_designer": [trajectory_payload()],
            "script_teacher": [downstream_script_payload()],
            "interaction_designer": [downstream_interaction_payload()],
            "classroom_director": [downstream_score_payload()],
            "student_simulator": [downstream_simulation_payload()],
            "lesson_reviewer": [downstream_review_payload()],
        }
    )
    client = CompositeGenerationClient(
        FakeClient([grounding_payload()]), preparation_client
    )
    source = grounded_source_problem(reference_solution_text=marker)

    bundle = asyncio.run(
        LessonGenerationService(client, MathEngine()).generate_bundle(source)
    )

    assert [call.role for call in client.calls if marker in call.user] == [
        "reference_grounder",
        "reference_analyst",
    ]
    assert marker not in bundle.generation_record.prepared_lesson.model_dump_json()
    assert (
        bundle.generation_record.prepared_lesson.solution_trace.source_steps[
            0
        ].mathematical_action
        == "代入已知数学量：x=2n"
    )


def test_grounder_reference_only_literal_is_rejected_before_preparation():
    payload = grounding_payload()
    payload["method_name"] = RAW_REFERENCE_MARKER
    client = _approved_preparation_client([payload])
    source = grounded_source_problem(
        reference_solution_text=RAW_REFERENCE_MARKER
    )

    with pytest.raises(LessonQualityError) as captured:
        asyncio.run(
            LessonGenerationService(client, MathEngine()).generate_bundle(
                source
            )
        )

    assert RAW_REFERENCE_MARKER not in str(captured.value)
    assert [call.role for call in client.calls] == ["reference_grounder"]


def test_checker_unavailability_softly_degrades_the_real_grounded_route():
    check_request = {
        "check_id": "divide-by-n",
        "kind": "nonzero_division",
        "expression": "2*n*(2*n-2*m+1)",
        "expected": "2*n-2*m+1",
        "substitutions": {},
        "nonzero_symbols": ["n"],
        "conclusion_linked": True,
    }

    class UnavailableChecker:
        def check(self, request):
            del request
            raise ClaimCheckerUnavailableError("private checker outage")

    client = _approved_preparation_client(
        [grounding_payload([check_request])]
    )
    lesson = asyncio.run(
        LessonGenerationService(
            client,
            MathEngine(),
            claim_checker=UnavailableChecker(),
        ).generate(grounded_source_problem())
    )

    assert lesson.validation_report["verification_mode"] == (
        "reference_grounded"
    )
    assert lesson.validation_report["consistency_status"] == "warning"


@pytest.mark.parametrize("checker_error", [MemoryError(), PermissionError()])
def test_unexpected_checker_errors_propagate_before_preparation(checker_error):
    check_request = {
        "check_id": "divide-by-n",
        "kind": "nonzero_division",
        "expression": "2*n*(2*n-2*m+1)",
        "expected": "2*n-2*m+1",
        "substitutions": {},
        "nonzero_symbols": ["n"],
        "conclusion_linked": True,
    }

    class BrokenChecker:
        def check(self, request):
            del request
            raise checker_error

    client = _approved_preparation_client(
        [grounding_payload([check_request])]
    )
    service = LessonGenerationService(
        client,
        MathEngine(),
        claim_checker=BrokenChecker(),
    )

    with pytest.raises(type(checker_error)) as captured:
        asyncio.run(service.generate_bundle(grounded_source_problem()))

    assert captured.value is checker_error
    assert [call.role for call in client.calls] == ["reference_grounder"]


def test_reference_audit_provider_failure_retries_before_preparation():
    marker = "PRIVATE-SYMBOLIC-REFERENCE-TASK7"
    audit = reference_audit_payload()
    audit["claimed_answer"] = "x=1 或 x=5"
    audit["key_steps"] = []
    client = _approved_preparation_client(
        [ModelResponseError("temporary provider failure"), audit]
    )
    pipeline = _RecordingPreparationPipeline(client, [])
    source_problem = ProblemInput(
        problem_text="用配方法解方程：x^2-6*x+5=0",
        reference_answer="x=1 或 x=5",
        reference_solution_text=marker,
        required_method="complete_the_square",
    )
    service = LessonGenerationService(
        client,
        MathEngine(),
        preparation_pipeline=pipeline,
    )

    with pytest.raises(PreparationFailure, match="参考解析轨迹"):
        asyncio.run(service.generate_bundle(source_problem))

    assert [call.role for call in client.calls[:3]] == [
        "reference_auditor",
        "reference_auditor",
        "reference_analyst",
    ]
    assert [call.role for call in client.calls if marker in call.user] == [
        "reference_auditor",
        "reference_auditor",
        "reference_analyst",
    ]


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


def test_model_authored_review_summary_remains_private():
    marker = "MODEL-REVIEW-ASSESSMENT-TASK7"
    review = downstream_review_payload()
    review["approval_summary"] = marker
    preparation_client = PreparationFakeClient(
        {
            "reference_analyst": [trace_payload()],
            "teaching_designer": [trajectory_payload()],
            "script_teacher": [downstream_script_payload()],
            "interaction_designer": [downstream_interaction_payload()],
            "classroom_director": [downstream_score_payload()],
            "student_simulator": [downstream_simulation_payload()],
            "lesson_reviewer": [review],
        }
    )
    client = CompositeGenerationClient(FakeClient([]), preparation_client)
    service = LessonGenerationService(client, MathEngine())

    async def grounded_route(source_problem, on_stage):
        del source_problem, on_stage
        return preparation_route()

    service._build_grounded_teaching_route = grounded_route
    bundle = asyncio.run(service.generate_bundle(preparation_problem()))

    assert marker not in json.dumps(
        bundle.lesson.validation_report, ensure_ascii=False
    )
    assert (
        bundle.generation_record.prepared_lesson.review.approval_summary
        == marker
    )


class _MutatingCompiler:
    def __init__(self, mutation):
        self.mutation = mutation

    def compile(self, source_problem, draft, report):
        lesson = LessonCompiler().compile(source_problem, draft, report)
        return self.mutation(lesson)


def _replace_first_authored_cue(lesson, *, spoken_text=None, drop=False):
    beats = list(lesson.beats)
    for beat_index, beat in enumerate(beats):
        cues = list(beat.sync_cues)
        for cue_index, cue in enumerate(cues):
            if cue.cue_id == "runtime-transfer-intro-cue":
                continue
            if drop:
                cues.pop(cue_index)
            else:
                cues[cue_index] = cue.model_copy(
                    update={"spoken_text": spoken_text}
                )
            beats[beat_index] = beat.model_copy(update={"sync_cues": cues})
            return lesson.model_copy(update={"beats": beats})
    raise AssertionError("fixture has no authored cue")


def _merge_first_two_authored_cues(lesson):
    beats = list(lesson.beats)
    locations = []
    for beat_index, beat in enumerate(beats):
        for cue_index, cue in enumerate(beat.sync_cues):
            if cue.cue_id != "runtime-transfer-intro-cue":
                locations.append((beat_index, cue_index, cue))
    first_beat, first_index, first = locations[0]
    second_beat, second_index, second = locations[1]
    first_cues = list(beats[first_beat].sync_cues)
    first_cues[first_index] = first.model_copy(
        update={"spoken_text": first.spoken_text + second.spoken_text}
    )
    beats[first_beat] = beats[first_beat].model_copy(
        update={"sync_cues": first_cues}
    )
    second_cues = list(beats[second_beat].sync_cues)
    second_cues.pop(second_index)
    beats[second_beat] = beats[second_beat].model_copy(
        update={"sync_cues": second_cues}
    )
    return lesson.model_copy(update={"beats": beats})


def _mutate_authored_action(lesson, mode):
    beats = list(lesson.beats)
    if mode == "inject":
        beat = beats[0]
        cues = list(beat.sync_cues)
        cue = cues[0]
        cues[0] = cue.model_copy(
            update={
                "start_actions": [
                    *cue.start_actions,
                    SyncVisualAction(
                        surface="board",
                        type="focus",
                        target="compiler-injected-target",
                    ),
                ]
            }
        )
        beats[0] = beat.model_copy(update={"sync_cues": cues})
        return lesson.model_copy(update={"beats": beats})

    for beat_index, beat in enumerate(beats):
        cues = list(beat.sync_cues)
        for cue_index, cue in enumerate(cues):
            for action_field in (
                "lead_actions",
                "start_actions",
                "end_actions",
            ):
                actions = list(getattr(cue, action_field))
                if not actions:
                    continue
                if mode == "delete":
                    actions.pop(0)
                else:
                    actions[0] = actions[0].model_copy(
                        update={"target": "compiler-mutated-target"}
                    )
                cues[cue_index] = cue.model_copy(
                    update={action_field: actions}
                )
                beats[beat_index] = beat.model_copy(
                    update={"sync_cues": cues}
                )
                return lesson.model_copy(update={"beats": beats})
    raise AssertionError("fixture has no authored action")


def _move_interaction_binding(lesson):
    beats = list(lesson.beats)
    source_index = next(
        index
        for index, beat in enumerate(beats)
        if beat.interaction is not None
    )
    target_index = source_index - 1
    interaction = beats[source_index].interaction
    assert target_index >= 0
    assert beats[target_index].interaction is None
    beats[source_index] = beats[source_index].model_copy(
        update={"interaction": None}
    )
    beats[target_index] = beats[target_index].model_copy(
        update={"interaction": interaction}
    )
    return lesson.model_copy(update={"beats": beats})


def _move_authored_cue_to_another_beat(lesson):
    beats = list(lesson.beats)
    source_index = next(
        index for index, beat in enumerate(beats) if beat.sync_cues
    )
    target_index = source_index + 1
    cue = beats[source_index].sync_cues[0]
    beats[source_index] = beats[source_index].model_copy(
        update={"sync_cues": beats[source_index].sync_cues[1:]}
    )
    beats[target_index] = beats[target_index].model_copy(
        update={"sync_cues": [cue, *beats[target_index].sync_cues]}
    )
    return lesson.model_copy(update={"beats": beats})


def _inject_audio_before_tts(lesson):
    beats = list(lesson.beats)
    beat = beats[0]
    cues = list(beat.sync_cues)
    cues[0] = cues[0].model_copy(
        update={"audio_url": "/audio/compiler-injected.mp3"}
    )
    beats[0] = beat.model_copy(update={"sync_cues": cues})
    return lesson.model_copy(update={"beats": beats})


def _mutate_fixed_transfer_cue(lesson):
    beats = list(lesson.beats)
    beat = beats[-1]
    cue = beat.sync_cues[0]
    assert cue.cue_id == "runtime-transfer-intro-cue"
    beats[-1] = beat.model_copy(
        update={
            "sync_cues": [
                cue.model_copy(update={"spoken_text": "编译器注入的过渡语"})
            ]
        }
    )
    return lesson.model_copy(update={"beats": beats})


@pytest.mark.parametrize(
    "mutation",
    [
        lambda lesson: _mutate_authored_action(lesson, "inject"),
        lambda lesson: _mutate_authored_action(lesson, "delete"),
        lambda lesson: _mutate_authored_action(lesson, "modify"),
        _move_interaction_binding,
        lambda lesson: lesson.model_copy(
            update={
                "beats": [
                    lesson.beats[0].model_copy(update={"layer": "comparison"}),
                    *lesson.beats[1:],
                ]
            }
        ),
        _move_authored_cue_to_another_beat,
        _inject_audio_before_tts,
        _mutate_fixed_transfer_cue,
    ],
)
def test_post_compile_integrity_rejects_any_runtime_semantic_mutation(
    mutation,
):
    score = downstream_score_payload()
    score["board_objects"] = [
        {"board_object_id": "target-relation", "content": "m-n"}
    ]
    score["cues"][0]["start_actions"] = [
        {
            "clause_id": "clause-open",
            "action": {
                "surface": "board",
                "type": "write",
                "target": "target-relation",
                "content": "m-n",
            },
        }
    ]
    client = _approved_preparation_client(performance=score)
    service = LessonGenerationService(
        client,
        MathEngine(),
        compiler=_MutatingCompiler(mutation),
    )

    async def grounded_route(source_problem, on_stage):
        del source_problem, on_stage
        return preparation_route()

    service._build_grounded_teaching_route = grounded_route

    with pytest.raises(LessonQualityError, match="完整性"):
        asyncio.run(service.generate_bundle(preparation_problem()))


class _InPlaceMutatingCompiler:
    def __init__(self, target):
        self.target = target

    def compile(self, source_problem, draft, report):
        if self.target == "problem":
            source_problem.problem_text = "被编译器原地篡改的题目"
        elif self.target == "report":
            report["compiler_injected"] = "private"
        else:
            draft.title = "被编译器原地篡改的标题"
        return LessonCompiler().compile(source_problem, draft, report)


@pytest.mark.parametrize("target", ["problem", "report", "draft"])
def test_compiler_cannot_mutate_its_defensive_inputs_in_place(target):
    client = _approved_preparation_client()
    service = LessonGenerationService(
        client,
        MathEngine(),
        compiler=_InPlaceMutatingCompiler(target),
    )
    source_problem = preparation_problem()
    frozen_source = source_problem.model_dump_json()

    async def grounded_route(received_problem, on_stage):
        del received_problem, on_stage
        return preparation_route()

    service._build_grounded_teaching_route = grounded_route

    with pytest.raises(LessonQualityError, match="完整性"):
        asyncio.run(service.generate_bundle(source_problem))

    assert source_problem.model_dump_json() == frozen_source


@pytest.mark.parametrize(
    "mutation",
    [
        lambda lesson: lesson.model_copy(
            update={
                "problem": ProblemInput(
                    problem_text="另一个完整题目",
                    reference_answer="另一个答案",
                )
            }
        ),
        lambda lesson: lesson.model_copy(
            update={"problem": {"problem_text": "malformed compiler output"}}
        ),
        lambda lesson: lesson.model_copy(
            update={"validation_report": {"review_status": "forged"}}
        ),
        lambda lesson: _replace_first_authored_cue(
            lesson, spoken_text="被编译器改写的讲稿"
        ),
        lambda lesson: _replace_first_authored_cue(lesson, drop=True),
        _merge_first_two_authored_cues,
    ],
)
def test_post_compile_integrity_rejects_mutated_problem_report_or_cues(
    mutation,
):
    client = _approved_preparation_client()
    service = LessonGenerationService(
        client,
        MathEngine(),
        compiler=_MutatingCompiler(mutation),
    )

    async def grounded_route(source_problem, on_stage):
        del source_problem, on_stage
        return preparation_route()

    service._build_grounded_teaching_route = grounded_route

    with pytest.raises(LessonQualityError, match="完整性"):
        asyncio.run(service.generate_bundle(preparation_problem()))


def test_expected_compiler_failure_is_mapped_to_safe_quality_error():
    class FailingCompiler:
        def compile(self, source_problem, draft, report):
            del source_problem, draft, report
            raise LessonCompileError("private compiler detail")

    client = _approved_preparation_client()
    service = LessonGenerationService(
        client, MathEngine(), compiler=FailingCompiler()
    )

    async def grounded_route(source_problem, on_stage):
        del source_problem, on_stage
        return preparation_route()

    service._build_grounded_teaching_route = grounded_route

    with pytest.raises(LessonQualityError) as captured:
        asyncio.run(service.generate_bundle(preparation_problem()))

    assert "private compiler detail" not in str(captured.value)


def test_unexpected_compiler_error_propagates_unchanged():
    programmer_error = RuntimeError("programmer bug")

    class BrokenCompiler:
        def compile(self, source_problem, draft, report):
            del source_problem, draft, report
            raise programmer_error

    client = _approved_preparation_client()
    service = LessonGenerationService(
        client, MathEngine(), compiler=BrokenCompiler()
    )

    async def grounded_route(source_problem, on_stage):
        del source_problem, on_stage
        return preparation_route()

    service._build_grounded_teaching_route = grounded_route

    with pytest.raises(RuntimeError) as captured:
        asyncio.run(service.generate_bundle(preparation_problem()))

    assert captured.value is programmer_error


def test_compiler_cancellation_propagates_unchanged():
    class CancelledCompiler:
        def compile(self, source_problem, draft, report):
            del source_problem, draft, report
            raise asyncio.CancelledError()

    client = _approved_preparation_client()
    service = LessonGenerationService(
        client, MathEngine(), compiler=CancelledCompiler()
    )

    async def grounded_route(source_problem, on_stage):
        del source_problem, on_stage
        return preparation_route()

    service._build_grounded_teaching_route = grounded_route

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(service.generate_bundle(preparation_problem()))


def test_bundle_rejects_a_generation_record_for_another_lesson():
    client = _approved_preparation_client()
    service = LessonGenerationService(client, MathEngine())

    async def grounded_route(source_problem, on_stage):
        del source_problem, on_stage
        return preparation_route()

    service._build_grounded_teaching_route = grounded_route
    bundle = asyncio.run(service.generate_bundle(preparation_problem()))
    forged_record = bundle.generation_record.model_copy(
        update={"lesson_id": "another-lesson"}
    )

    with pytest.raises(ValidationError, match="lesson id mismatch"):
        GeneratedLessonBundle(
            lesson=bundle.lesson,
            generation_record=forged_record,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("episode_id", "forged-episode"),
        ("original_performance_cue_id", "forged-performance-cue"),
    ],
)
def test_bundle_rejects_provenance_detached_from_private_preparation(
    field,
    value,
):
    client = _approved_preparation_client()
    service = LessonGenerationService(client, MathEngine())

    async def grounded_route(source_problem, on_stage):
        del source_problem, on_stage
        return preparation_route()

    service._build_grounded_teaching_route = grounded_route
    bundle = asyncio.run(service.generate_bundle(preparation_problem()))
    provenance = list(bundle.generation_record.cue_provenance)
    provenance[0] = provenance[0].model_copy(update={field: value})
    forged_record = bundle.generation_record.model_copy(
        update={"cue_provenance": provenance}
    )

    with pytest.raises(ValidationError, match="matches preparation"):
        GeneratedLessonBundle(
            lesson=bundle.lesson,
            generation_record=forged_record,
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
            self.problem_focus_targets = None

        async def prepare_with_audit(
            self,
            source_problem,
            teaching_route,
            problem_focus_targets,
            on_stage=None,
        ):
            del source_problem, on_stage
            self.route = teaching_route
            self.problem_focus_targets = list(problem_focus_targets)
            raise PreparationFailure(
                category="review_not_converged",
                role="lesson_reviewer",
                detail="课程审核未收敛。",
            )

    pipeline = CapturingPipeline()
    source_problem = ProblemInput(
        problem_text="关注$x$，用配方法解方程：x^2-6*x+5=0",
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
    assert pipeline.problem_focus_targets == compile_problem_focus_targets(
        source_problem.problem_text
    )
    assert [
        target.math_text for target in pipeline.problem_focus_targets
    ] == ["x"]


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
