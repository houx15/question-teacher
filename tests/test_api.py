import asyncio
from concurrent.futures import ThreadPoolExecutor
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api import (
    _PUBLIC_GENERATION_STAGES,
    run_generation,
    safe_generation_error,
)
from app.audio_manifest import (
    audio_asset_url,
    correct_feedback_asset_id,
    cue_asset_id,
    hint_asset_id,
    option_feedback_asset_id,
    support_cue_asset_id,
)
from app.compiler import LessonCompileError, LessonCompiler
from app.config import Settings
from app.generation import LessonGenerationService, LessonInputError
from app.audio_service import LessonAudioService
from app.preparation_models import GenerationRecord
from app.pedagogy_rubric import PEDAGOGY_RUBRIC_VERSION
from app.problem_focus import compile_problem_focus_targets
from app.main import PROJECT_ROOT, create_app
from app.math_engine import MathEngine
from app.llm_client import ModelResponseError, ModelStructureError
from app.preparation_pipeline import PreparationFailure
from app.tts_client import SpeechGenerationError
from app.schemas import (
    BoardAction,
    Interaction,
    InteractionOption,
    LessonDraft,
    ProblemInput,
    RuntimeBeat,
    RuntimeLesson,
    RuntimeSyncCue,
    SupportSyncCue,
    TransferItem,
    TransferOption,
)
from app.store import MemoryStore
from tests.test_generation import (
    _approved_preparation_client,
    preparation_problem,
    preparation_route,
    problem,
    valid_draft,
)
from tests.test_preparation_models import (
    prepared_lesson,
    teaching_progression_payload,
)
from tests.test_preparation_pipeline import (
    client as preparation_client,
    downstream_interaction_payload,
    downstream_review_payload,
    downstream_score_payload,
    downstream_simulation_payload,
    downstream_script_payload,
    review_finding,
    teaching_progression_payload as parameter_progression_payload,
    trace_payload,
    trajectory_payload,
)
from tests.generation_fakes import CompositeGenerationClient, FakeClient


def problem_input() -> ProblemInput:
    return ProblemInput(
        problem_text="2x+3=7",
        reference_answer="x=2",
        lesson_length="standard",
    )


def runtime_lesson(
    problem: ProblemInput,
    lesson_id: str = "lesson-1",
) -> RuntimeLesson:
    return RuntimeLesson(
        lesson_id=lesson_id,
        problem=problem,
        title="测试课程",
        learning_goal="学会解方程",
        beats=[
            RuntimeBeat(
                beat_id="beat-1",
                purpose="建立等式平衡意识",
                narration="先观察等号两边。",
                board_actions=[],
                layer="base",
            )
        ],
        summary="完成",
        transfer_item={
            "problem_text": "2x=4",
            "expected_answer": "x=2",
            "method_signal": "保持等式平衡",
            "options": [
                {"option_id": "transfer-a", "label": "x=2", "canonical_answer": "x=2", "feedback": "正确"},
                {"option_id": "transfer-b", "label": "x=1", "canonical_answer": "x=1", "feedback": "再算一次"},
                {"option_id": "transfer-c", "label": "x=4", "canonical_answer": "x=4", "feedback": "注意等式两边"},
            ],
            "correct_option_id": "transfer-a",
        },
        validation_report={"math_status": "verified"},
    )


def save_interaction_lesson(
    store,
    *,
    kind,
    expected,
    options=None,
):
    lesson = runtime_lesson(problem_input())
    interaction = Interaction(
        interaction_id="interaction-1",
        kind=kind,
        prompt="请作答。",
        expected_answer=expected,
        options=options or [],
    )
    beat = lesson.beats[0].model_copy(
        update={"interaction": interaction}
    )
    lesson = lesson.model_copy(update={"beats": [beat]})
    store.save_lesson(lesson)
    return lesson, interaction


class FakeGenerator:
    def __init__(self, error=None):
        self.error = error
        self.calls = 0

    async def generate(self, problem, on_stage=None):
        self.calls += 1
        if self.error is not None:
            raise self.error
        stages = ["正在验证数学路线"]
        if problem.reference_solution_text is not None:
            stages.append("正在审阅参考解析")
        stages.extend(
            [
                "正在设计完整讲解",
                "正在进行整篇审稿",
                "正在修订完整讲解",
                "正在编译课堂",
            ]
        )
        for stage in stages:
            if on_stage:
                on_stage(stage)
        return runtime_lesson(problem)


class FakeAudioService:
    def __init__(self):
        self.calls = 0

    async def attach_audio(self, lesson, on_stage=None):
        self.calls += 1
        if on_stage:
            on_stage("正在生成讲解语音")
        for beat in lesson.beats:
            if beat.sync_cues:
                beat.audio_url = None
                for cue in beat.sync_cues:
                    cue.audio_url = audio_asset_url(
                        lesson.lesson_id,
                        cue_asset_id(beat.beat_id, cue.cue_id),
                    )
            else:
                beat.audio_url = audio_asset_url(
                    lesson.lesson_id,
                    beat.beat_id,
                )
            interaction = beat.interaction
            if interaction is None:
                continue
            interaction.hint_audio_urls = [
                audio_asset_url(
                    lesson.lesson_id,
                    hint_asset_id(beat.beat_id, index),
                )
                for index, _hint in enumerate(
                    interaction.hints,
                    start=1,
                )
            ]
            for index, option in enumerate(
                interaction.options,
                start=1,
            ):
                option.feedback_audio_url = (
                    audio_asset_url(
                        lesson.lesson_id,
                        option_feedback_asset_id(beat.beat_id, index),
                    )
                    if option.feedback
                    else None
                )
                for support_cue in option.support_cues:
                    support_cue.audio_url = audio_asset_url(
                        lesson.lesson_id,
                        support_cue_asset_id(
                            beat.beat_id,
                            interaction.interaction_id,
                            option.option_id,
                            support_cue.cue_id,
                        ),
                    )
            interaction.correct_audio_url = (
                audio_asset_url(
                    lesson.lesson_id,
                    correct_feedback_asset_id(beat.beat_id),
                )
                if interaction.explanation_after_correct
                else None
            )
        return lesson


def generation_record_for(lesson):
    prepared = prepared_lesson()
    prepared["rubric_version"] = PEDAGOGY_RUBRIC_VERSION
    prepared["teaching_progression"] = teaching_progression_payload()
    prepared["teaching_script"]["title"] = lesson.title
    prepared["teaching_script"]["learning_goal"] = lesson.learning_goal
    prepared["interaction_plan"]["transfer_item"] = {
        "problem_text": lesson.transfer_item.problem_text,
        "expected_answer": lesson.transfer_item.expected_answer,
        "method_signal": lesson.transfer_item.method_signal,
        "options": [
            {
                "option_id": option.option_id,
                "label": option.label,
                "canonical_answer": option.canonical_answer,
                "feedback": option.feedback,
            }
            for option in lesson.transfer_item.options
        ],
        "correct_option_id": lesson.transfer_item.correct_option_id,
    }
    prepared["teaching_script"]["transfer_script"] = {
        "problem_text": lesson.transfer_item.problem_text,
        "method_signal": lesson.transfer_item.method_signal,
        "options": [
            {
                "option_id": option.option_id,
                "label": option.label,
                "feedback": option.feedback,
            }
            for option in lesson.transfer_item.options
        ],
    }
    for clause in prepared["teaching_script"]["clauses"]:
        clause["lesson_step_id"] = "teaching-step-001"
        clause["display_text"] = "等式两边同时减一"
    artifact_roles = [
        ("solution_trace", "reference_analyst"),
        ("reasoning_trajectory", "teaching_designer"),
        ("teaching_progression", "teaching_designer"),
        ("interaction_plan", "interaction_designer"),
        ("teaching_script", "script_teacher"),
        ("performance_score", "classroom_director"),
        ("simulation_report", "student_simulator"),
    ]
    prepared["artifact_history"] = [
        {
            "artifact_type": artifact_type,
            "version": 1,
            "responsible_role": role,
        }
        for artifact_type, role in artifact_roles
    ]
    prepared["performance_score"] = {
        "cues": [
            {
                "cue_id": f"performance-{index}",
                "clause_ids": [clause_id],
            }
            for index, clause_id in enumerate(
                ["open-1", "method-1", "close-1"],
                start=1,
            )
        ]
    }
    return GenerationRecord.model_validate(
        {
            "generation_id": "generation-api-1",
            "lesson_id": lesson.lesson_id,
            "route_fingerprint": "route-api-1",
            "prepared_lesson": prepared,
            "role_calls": [
                {
                    "role": role,
                    "input_artifact_versions": {"solution_trace": 1},
                    "output_artifact_type": "solution_trace",
                    "output_artifact_version": 1,
                    "duration_ms": 1,
                    "retry_count": 0,
                }
                for role in [
                    "reference_analyst",
                    "teaching_designer",
                    "script_teacher",
                    "interaction_designer",
                    "classroom_director",
                    "student_simulator",
                    "lesson_reviewer",
                ]
            ],
            "cue_provenance": [
                {
                    "episode_id": "episode-1",
                    "lesson_step_id": "teaching-step-001",
                    "clause_id": clause_id,
                    "original_performance_cue_id": f"performance-{index}",
                    "runtime_cue_id": f"runtime-authored-{index}",
                    "display_text": "等式两边同时减一",
                    "spoken_text": "我们把等式两边同时减一。",
                }
                for index, clause_id in enumerate(
                    ["open-1", "method-1", "close-1"],
                    start=1,
                )
            ],
            "created_at": "2026-08-11T10:00:00+08:00",
        }
    )


class BundleGenerator:
    def __init__(self):
        self.bundle_calls = 0
        self.generate_calls = 0

    async def generate_bundle(self, problem, on_stage=None):
        self.bundle_calls += 1
        if on_stage:
            on_stage("正在编译课堂")
        lesson = runtime_lesson(problem, lesson_id="lesson-bundle")
        beat = lesson.beats[0].model_copy(
            update={
                "sync_cues": [
                    RuntimeSyncCue(
                        cue_id=f"runtime-authored-{index}",
                        teaching_step_id="teaching-step-001",
                        display_text="等式两边同时减一",
                        spoken_text="我们把等式两边同时减一。",
                    )
                    for index in range(1, 4)
                ]
            }
        )
        lesson = lesson.model_copy(
            update={
                "beats": [
                    beat.model_copy(
                        update={
                            "narration": (
                                "我们把等式两边同时减一。" * 3
                            )
                        }
                    )
                ],
                "problem_focus_targets": compile_problem_focus_targets(
                    problem.problem_text
                ),
                "summary": "我们把等式两边同时减一。",
                "validation_report": {
                    "teaching_route_fingerprint": "route-api-1",
                    "pedagogy_rubric_version": PEDAGOGY_RUBRIC_VERSION,
                    "artifact_versions": {
                        "solution_trace": 1,
                        "reasoning_trajectory": 1,
                        "teaching_progression": 1,
                        "interaction_plan": 1,
                        "teaching_script": 1,
                        "performance_score": 1,
                        "simulation_report": 1,
                    },
                    "repair_count": 0,
                    "review_status": "approved",
                },
            }
        )
        record = generation_record_for(lesson)

        class Bundle:
            pass

        bundle = Bundle()
        bundle.lesson = lesson
        bundle.generation_record = record
        self.last_bundle = bundle
        return bundle

    async def generate(self, problem, on_stage=None):
        del problem, on_stage
        self.generate_calls += 1
        raise AssertionError("generate_bundle must be preferred")


class PreparationStageGenerator(FakeGenerator):
    async def generate(self, problem, on_stage=None):
        self.calls += 1
        for stage in [
            "正在验证数学路线",
            "整理参考解析",
            "设计解题思维轨迹",
            "设计教学推进",
            "设计互动",
            "编写讲稿",
            "编排板书与高亮",
            "模拟学生并审核课程",
            "正在编译课堂",
        ]:
            if on_stage:
                on_stage(stage)
        return runtime_lesson(problem)


class RepairStageGenerator(FakeGenerator):
    _INITIAL_PREPARATION_STAGES = [
        "正在验证数学路线",
        "整理参考解析",
        "设计解题思维轨迹",
        "设计教学推进",
        "设计互动",
        "编写讲稿",
        "编排板书与高亮",
        "模拟学生并审核课程",
    ]

    def __init__(self, repair_rounds, *, lesson_id="lesson-repaired"):
        super().__init__()
        self.repair_rounds = repair_rounds
        self.lesson_id = lesson_id

    async def generate(self, problem, on_stage=None):
        self.calls += 1
        stages = list(self._INITIAL_PREPARATION_STAGES)
        for repair_round in self.repair_rounds:
            stages.extend(repair_round)
        stages.append("正在编译课堂")
        for stage in stages:
            if on_stage:
                on_stage(stage)
            await asyncio.sleep(0)
        return runtime_lesson(problem, lesson_id=self.lesson_id)


class SequentialIdGenerator(FakeGenerator):
    async def generate(self, problem, on_stage=None):
        lesson = await super().generate(problem, on_stage=on_stage)
        return lesson.model_copy(
            update={"lesson_id": f"lesson-{self.calls}"}
        )


class RecordingStore(MemoryStore):
    def __init__(self):
        super().__init__()
        self.seen_stages = []

    def update_job(self, job_id, **changes):
        job = super().update_job(job_id, **changes)
        if "stage" in changes:
            self.seen_stages.append(job.stage)
        return job


class PerJobRecordingStore(MemoryStore):
    def __init__(self):
        super().__init__()
        self.seen_stages_by_job = {}

    def update_job(self, job_id, **changes):
        job = super().update_job(job_id, **changes)
        if "stage" in changes:
            self.seen_stages_by_job.setdefault(job_id, []).append(job.stage)
        return job


class FailingLessonStore(RecordingStore):
    def save_lesson(self, lesson, generation_record=None):
        del lesson, generation_record
        raise OSError("private database path")


_DETAILED_PUBLIC_STAGES = [
    "正在理解题目",
    "正在核对题目材料",
    "正在整理参考解析",
    "正在设计解题思维轨迹",
    "正在设计课堂推进",
    "正在设计互动",
    "正在编写讲稿",
    "正在编排板书与高亮",
    "正在审核和优化课程",
    "正在编译课程",
    "正在生成语音",
    "正在保存课程",
    "课程已生成",
]


def test_public_preparation_stage_mapping_is_specific_and_stable():
    assert {
        stage: _PUBLIC_GENERATION_STAGES[stage]
        for stage in [
            "整理参考解析",
            "设计解题思维轨迹",
            "设计教学推进",
            "设计互动",
            "编写讲稿",
            "编排板书与高亮",
            "模拟学生并审核课程",
            "正在编译课堂",
            "正在生成讲解语音",
            "正在保存课程",
        ]
    } == {
        "整理参考解析": "正在整理参考解析",
        "设计解题思维轨迹": "正在设计解题思维轨迹",
        "设计教学推进": "正在设计课堂推进",
        "设计互动": "正在设计互动",
        "编写讲稿": "正在编写讲稿",
        "编排板书与高亮": "正在编排板书与高亮",
        "模拟学生并审核课程": "正在审核和优化课程",
        "正在编译课堂": "正在编译课程",
        "正在生成讲解语音": "正在生成语音",
        "正在保存课程": "正在保存课程",
    }


def test_internal_repair_marker_stays_in_review_without_premature_compile():
    assert _PUBLIC_GENERATION_STAGES["正在修订完整讲解"] == (
        "正在审核和优化课程"
    )


def _generation_service_with_pipeline(preparation_fake):
    from tests.generation_fakes import CompositeGenerationClient

    composite = CompositeGenerationClient(FakeClient([]), preparation_fake)
    service = LessonGenerationService(composite, MathEngine())

    async def grounded_route(source_problem, on_stage):
        del source_problem, on_stage
        return preparation_route()

    service._build_grounded_teaching_route = grounded_route
    return service


def test_real_preparation_pipeline_reports_each_public_stage_through_save():
    store = RecordingStore()
    job = store.create_job()
    generator = _generation_service_with_pipeline(
        preparation_client(progression_target_ids=[])
    )

    asyncio.run(
        run_generation(
            job.job_id,
            preparation_problem(),
            store,
            generator,
            FakeAudioService(),
        )
    )

    assert store.seen_stages == _DETAILED_PUBLIC_STAGES


def test_real_review_repair_does_not_repeat_or_regress_public_progress():
    finding = review_finding("classroom_director")
    fake = preparation_client(
        progression_target_ids=[],
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
    store = RecordingStore()
    job = store.create_job()

    asyncio.run(
        run_generation(
            job.job_id,
            preparation_problem(),
            store,
            _generation_service_with_pipeline(fake),
            FakeAudioService(),
        )
    )

    assert store.seen_stages == _DETAILED_PUBLIC_STAGES


@pytest.mark.parametrize(
    ("failure", "expected_category"),
    [
        (
            PreparationFailure("provider_error", "script_teacher", "secret"),
            "provider_error",
        ),
        (
            PreparationFailure("invalid_structure", "script_teacher", "secret"),
            "invalid_structure",
        ),
        (
            ModelStructureError("invalid_json", "secret provider payload"),
            "invalid_structure",
        ),
        (
            PreparationFailure(
                "reference_trace_failed", "reference_analyst", "secret"
            ),
            "reference_trace_failed",
        ),
        (
            PreparationFailure(
                "teaching_script_failed", "script_teacher", "secret"
            ),
            "reasoning_design_failed",
        ),
        (
            PreparationFailure(
                "review_not_converged", "lesson_reviewer", "secret"
            ),
            "review_not_converged",
        ),
    ],
)
def test_generation_failures_record_only_allowlisted_private_categories(
    failure,
    expected_category,
):
    store = MemoryStore()
    job = store.create_job()

    asyncio.run(
        run_generation(
            job.job_id,
            problem_input(),
            store,
            FakeGenerator(failure),
            FakeAudioService(),
        )
    )

    diagnostic = store.get_job_diagnostic(job.job_id)
    assert diagnostic.category == expected_category
    assert diagnostic.model_dump() == {"category": expected_category}
    assert "secret" not in str(diagnostic)
    assert store.get_job(job.job_id).error == "课程生成失败，请稍后重试。"


def test_internal_failure_category_is_absent_from_public_job_response():
    store = MemoryStore()
    failure = PreparationFailure(
        "review_not_converged",
        "lesson_reviewer",
        "private feedback",
    )
    client, _, _ = build_client(
        store=store,
        generator=FakeGenerator(failure),
    )

    response = client.post(
        "/api/lessons/generate",
        json=problem_input().model_dump(),
    )
    job_id = response.json()["job_id"]
    public_job = client.get(f"/api/jobs/{job_id}").json()

    assert "category" not in public_job
    assert "private feedback" not in str(public_job)
    assert store.get_job_diagnostic(job_id).category == "review_not_converged"


def test_tts_and_persistence_failures_have_private_phase_categories():
    class FailingTts:
        async def attach_audio(self, lesson, on_stage=None):
            del lesson, on_stage
            raise SpeechGenerationError("provider payload and key")

    tts_store = MemoryStore()
    tts_job = tts_store.create_job()
    asyncio.run(
        run_generation(
            tts_job.job_id,
            problem_input(),
            tts_store,
            FakeGenerator(),
            FailingTts(),
        )
    )
    persistence_store = FailingLessonStore()
    persistence_job = persistence_store.create_job()
    asyncio.run(
        run_generation(
            persistence_job.job_id,
            problem_input(),
            persistence_store,
            FakeGenerator(),
            FakeAudioService(),
        )
    )

    assert tts_store.get_job_diagnostic(tts_job.job_id).category == "tts_failed"
    assert (
        persistence_store.get_job_diagnostic(persistence_job.job_id).category
        == "persistence_failed"
    )


def test_save_stage_precedes_atomic_save_and_lesson_id_publication():
    class SaveObservingStore(MemoryStore):
        def __init__(self):
            super().__init__()
            self.job_id = None

        def save_lesson(self, lesson, generation_record=None):
            job = self.get_job(self.job_id)
            assert job.stage == "正在保存课程"
            assert job.lesson_id is None
            super().save_lesson(lesson, generation_record)

    store = SaveObservingStore()
    job = store.create_job()
    store.job_id = job.job_id

    asyncio.run(
        run_generation(
            job.job_id,
            problem_input(),
            store,
            FakeGenerator(),
            FakeAudioService(),
        )
    )

    completed = store.get_job(job.job_id)
    assert completed.status == "completed"
    assert completed.lesson_id == "lesson-1"


def test_input_error_remains_correctable_without_internal_failure_category():
    store = MemoryStore()
    job = store.create_job()
    public_message = "题目格式不正确。"

    class TrustedInputValidator:
        async def generate(self, problem, on_stage=None):
            del problem
            if on_stage is not None:
                on_stage("正在验证数学路线")
            raise LessonInputError(public_message)

    asyncio.run(
        run_generation(
            job.job_id,
            problem_input(),
            store,
            TrustedInputValidator(),
            FakeAudioService(),
        )
    )

    assert store.get_job(job.job_id).error == public_message
    assert store.get_job_diagnostic(job.job_id) is None


def test_real_input_validation_contradiction_keeps_correctable_public_message():
    store = MemoryStore()
    job = store.create_job()
    audio = FakeAudioService()
    source_problem = problem_input().model_copy(
        update={"reference_answer": "x=3"}
    )

    asyncio.run(
        run_generation(
            job.job_id,
            source_problem,
            store,
            LessonGenerationService(FakeClient([]), MathEngine()),
            audio,
        )
    )

    failed = store.get_job(job.job_id)
    assert failed.error == "参考答案与题目实际结果不一致。"
    assert store.get_job_diagnostic(job.job_id) is None
    assert audio.calls == 0


def _tainted_input_error():
    error = LessonInputError("题目格式不正确。")
    error.public_message = "private input error api-key=secret"
    error.args = ("private input error api-key=secret",)
    return error


@pytest.mark.parametrize(
    "failure_factory",
    [
        _tainted_input_error,
        lambda: ModelResponseError("private provider api-key=secret"),
        lambda: ModelStructureError(
            "invalid_json", "private structure api-key=secret"
        ),
    ],
    ids=["input-error", "provider-error", "structure-error"],
)
def test_tts_failure_phase_overrides_exception_type_without_public_leak(
    failure_factory,
):
    failure = failure_factory()

    class AdversarialTts:
        async def attach_audio(self, lesson, on_stage=None):
            del lesson, on_stage
            raise failure

    store = MemoryStore()
    job = store.create_job()
    asyncio.run(
        run_generation(
            job.job_id,
            problem_input(),
            store,
            FakeGenerator(),
            AdversarialTts(),
        )
    )

    public_job = store.get_job(job.job_id).model_dump()
    diagnostic = store.get_job_diagnostic(job.job_id)
    assert public_job["error"] == "课程生成失败，请稍后重试。"
    assert diagnostic.model_dump() == {"category": "tts_failed"}
    assert "secret" not in str(public_job)
    assert "secret" not in str(diagnostic)


@pytest.mark.parametrize(
    "failure_factory",
    [
        _tainted_input_error,
        lambda: ModelResponseError("private provider api-key=secret"),
        lambda: ModelStructureError(
            "invalid_json", "private structure api-key=secret"
        ),
    ],
    ids=["input-error", "provider-error", "structure-error"],
)
def test_persistence_failure_phase_overrides_exception_type_without_public_leak(
    failure_factory,
):
    failure = failure_factory()

    class AdversarialStore(MemoryStore):
        def save_lesson(self, lesson, generation_record=None):
            del lesson, generation_record
            raise failure

    store = AdversarialStore()
    job = store.create_job()
    asyncio.run(
        run_generation(
            job.job_id,
            problem_input(),
            store,
            FakeGenerator(),
            FakeAudioService(),
        )
    )

    public_job = store.get_job(job.job_id).model_dump()
    diagnostic = store.get_job_diagnostic(job.job_id)
    assert public_job["error"] == "课程生成失败，请稍后重试。"
    assert diagnostic.model_dump() == {"category": "persistence_failed"}
    assert "secret" not in str(public_job)
    assert "secret" not in str(diagnostic)


@pytest.mark.parametrize("failure_phase", ["tts", "persistence"])
@pytest.mark.parametrize(
    "failure_factory",
    [
        lambda: ModelResponseError("private provider api-key=secret"),
        lambda: LessonInputError("题目格式不正确。"),
    ],
    ids=["provider-error", "allowlisted-input-error"],
)
def test_stale_validation_progress_cannot_regress_later_failure_phase(
    failure_phase,
    failure_factory,
):
    failure = failure_factory()

    class CallbackCapturingGenerator(FakeGenerator):
        async def generate(self, problem, on_stage=None):
            self.stale_callback = on_stage
            return await super().generate(problem, on_stage=on_stage)

    generator = CallbackCapturingGenerator()

    class StaleProgressTts(FakeAudioService):
        async def attach_audio(self, lesson, on_stage=None):
            del lesson
            assert on_stage is not None
            on_stage("正在验证数学路线")
            raise failure

    class StaleProgressStore(MemoryStore):
        def save_lesson(self, lesson, generation_record=None):
            del lesson, generation_record
            assert generator.stale_callback is not None
            generator.stale_callback("正在验证数学路线")
            raise failure

    if failure_phase == "tts":
        store = MemoryStore()
        audio_service = StaleProgressTts()
        expected_category = "tts_failed"
    else:
        store = StaleProgressStore()
        audio_service = FakeAudioService()
        expected_category = "persistence_failed"

    job = store.create_job()
    asyncio.run(
        run_generation(
            job.job_id,
            problem_input(),
            store,
            generator,
            audio_service,
        )
    )

    public_job = store.get_job(job.job_id).model_dump()
    diagnostic = store.get_job_diagnostic(job.job_id)
    assert public_job["error"] == "课程生成失败，请稍后重试。"
    assert diagnostic.model_dump() == {"category": expected_category}
    assert "secret" not in str(public_job)
    assert "secret" not in str(diagnostic)


def test_stale_failure_callbacks_are_isolated_across_concurrent_jobs():
    callbacks = {}

    class CapturingGenerator(FakeGenerator):
        def __init__(self, lesson_id):
            super().__init__()
            self.lesson_id = lesson_id

        async def generate(self, problem, on_stage=None):
            callbacks[self.lesson_id] = on_stage
            lesson = await super().generate(problem, on_stage=on_stage)
            await asyncio.sleep(0)
            return lesson.model_copy(update={"lesson_id": self.lesson_id})

    class ConcurrentFailureStore(MemoryStore):
        def save_lesson(self, lesson, generation_record=None):
            if lesson.lesson_id == "lesson-persistence-stale":
                callback = callbacks[lesson.lesson_id]
                assert callback is not None
                callback("正在验证数学路线")
                raise LessonInputError("题目格式不正确。")
            return super().save_lesson(
                lesson,
                generation_record=generation_record,
            )

    class ConcurrentFailureTts(FakeAudioService):
        async def attach_audio(self, lesson, on_stage=None):
            if lesson.lesson_id == "lesson-tts-stale":
                assert on_stage is not None
                on_stage("正在验证数学路线")
                await asyncio.sleep(0)
                raise ModelResponseError("private provider api-key=secret")
            return await super().attach_audio(lesson, on_stage=on_stage)

    store = ConcurrentFailureStore()
    audio_service = ConcurrentFailureTts()
    tts_job = store.create_job()
    persistence_job = store.create_job()

    async def run_both():
        await asyncio.gather(
            run_generation(
                tts_job.job_id,
                problem_input(),
                store,
                CapturingGenerator("lesson-tts-stale"),
                audio_service,
            ),
            run_generation(
                persistence_job.job_id,
                problem_input(),
                store,
                CapturingGenerator("lesson-persistence-stale"),
                audio_service,
            ),
        )

    asyncio.run(run_both())

    for job, category in [
        (tts_job, "tts_failed"),
        (persistence_job, "persistence_failed"),
    ]:
        public_job = store.get_job(job.job_id).model_dump()
        diagnostic = store.get_job_diagnostic(job.job_id)
        assert public_job["error"] == "课程生成失败，请稍后重试。"
        assert diagnostic.model_dump() == {"category": category}
        assert "secret" not in str(public_job)
        assert "secret" not in str(diagnostic)


def test_arbitrary_lesson_input_error_message_is_rejected():
    with pytest.raises(ValueError, match="unknown input error message"):
        LessonInputError("private adapter output api-key=secret")


def test_tainted_input_error_at_trusted_boundary_still_fails_closed():
    failure = _tainted_input_error()

    class AdversarialInputValidator:
        async def generate(self, problem, on_stage=None):
            del problem
            if on_stage is not None:
                on_stage("正在验证数学路线")
            raise failure

    store = MemoryStore()
    job = store.create_job()
    asyncio.run(
        run_generation(
            job.job_id,
            problem_input(),
            store,
            AdversarialInputValidator(),
            FakeAudioService(),
        )
    )

    public_job = store.get_job(job.job_id).model_dump()
    assert public_job["error"] == "课程生成失败，请稍后重试。"
    assert "secret" not in str(public_job)


@pytest.mark.parametrize(
    "mutation",
    ["delete", [], {"private": "secret"}, object()],
    ids=["deleted", "list", "dict", "object"],
)
def test_mutated_input_error_metadata_never_breaks_safe_failure_handler(
    mutation,
):
    failure = LessonInputError("题目格式不正确。")
    if mutation == "delete":
        del failure.public_message
    else:
        failure.public_message = mutation

    class AdversarialInputValidator:
        async def generate(self, problem, on_stage=None):
            del problem
            if on_stage is not None:
                on_stage("正在验证数学路线")
            raise failure

    store = MemoryStore()
    job = store.create_job()
    asyncio.run(
        run_generation(
            job.job_id,
            problem_input(),
            store,
            AdversarialInputValidator(),
            FakeAudioService(),
        )
    )

    public_job = store.get_job(job.job_id).model_dump()
    assert public_job["status"] == "failed"
    assert public_job["error"] == "课程生成失败，请稍后重试。"
    assert "secret" not in str(public_job)
    diagnostic = store.get_job_diagnostic(job.job_id)
    assert diagnostic is not None
    assert diagnostic.category == "invalid_structure"
    assert "secret" not in str(diagnostic)


def test_mutated_input_error_args_never_invoke_untrusted_equality():
    class ExplosiveEquality:
        def __eq__(self, other):
            del other
            raise AssertionError("secret equality payload")

        def __ne__(self, other):
            del other
            raise AssertionError("secret equality payload")

    failure = LessonInputError("题目格式不正确。")
    failure.args = (ExplosiveEquality(),)

    assert LessonInputError.validated_public_message(failure) is None


def test_run_generation_prefers_bundle_and_keeps_record_private(tmp_path):
    store = MemoryStore(tmp_path / "lessons.sqlite3")
    generator = BundleGenerator()
    client, _, _ = build_client(store=store, generator=generator)

    response = client.post(
        "/api/lessons/generate",
        json=problem_input().model_dump(),
    )
    job = client.get(f"/api/jobs/{response.json()['job_id']}").json()
    lesson_id = job["lesson_id"]
    public = client.get(f"/api/lessons/{lesson_id}")

    assert job["status"] == "completed"
    assert generator.bundle_calls == 1
    assert generator.generate_calls == 0
    assert store.get_generation_record(lesson_id) is not None
    assert "generation_record" not in public.json()
    assert "generation-api-1" not in public.text
    assert "route-api-1" not in public.text


def test_bundle_audio_failure_persists_neither_lesson_nor_record(tmp_path):
    class FailingAudioService(FakeAudioService):
        async def attach_audio(self, lesson, on_stage=None):
            del lesson, on_stage
            self.calls += 1
            raise RuntimeError("private audio failure")

    database_path = tmp_path / "lessons.sqlite3"
    store = MemoryStore(database_path)
    generator = BundleGenerator()
    client, _, _ = build_client(
        store=store,
        generator=generator,
        audio_service=FailingAudioService(),
    )

    response = client.post(
        "/api/lessons/generate",
        json=problem_input().model_dump(),
    )
    job = client.get(f"/api/jobs/{response.json()['job_id']}").json()

    assert job["status"] == "failed"
    assert store.get_lesson("lesson-bundle") is None
    assert store.get_generation_record("lesson-bundle") is None
    assert not database_path.exists()


def test_bundle_save_failure_persists_neither_lesson_nor_record(tmp_path):
    database_path = tmp_path / "lessons.sqlite3"
    store = MemoryStore(database_path)
    seed = runtime_lesson(problem_input(), lesson_id="lesson-seed")
    store.save_lesson(seed)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_api_private_record
            BEFORE INSERT ON lesson_generation_records
            BEGIN
                SELECT RAISE(ABORT, 'private persistence failure');
            END
            """
        )
    generator = BundleGenerator()
    client, _, _ = build_client(store=store, generator=generator)

    response = client.post(
        "/api/lessons/generate",
        json=problem_input().model_dump(),
    )
    job = client.get(f"/api/jobs/{response.json()['job_id']}").json()

    assert job["status"] == "failed"
    assert job["lesson_id"] is None
    assert "private persistence failure" not in str(job)
    assert store.get_lesson("lesson-bundle") is None
    assert store.get_generation_record("lesson-bundle") is None


def test_legacy_generator_without_bundle_remains_supported():
    store = MemoryStore()
    generator = FakeGenerator()
    client, _, _ = build_client(store=store, generator=generator)

    response = client.post(
        "/api/lessons/generate",
        json=problem_input().model_dump(),
    )
    job = client.get(f"/api/jobs/{response.json()['job_id']}").json()

    assert job["status"] == "completed"
    assert generator.calls == 1
    assert store.get_generation_record(job["lesson_id"]) is None


@pytest.mark.parametrize(
    "generator",
    [FakeGenerator(), BundleGenerator()],
    ids=["legacy", "bundle"],
)
def test_precreated_empty_database_initializes_during_generation(
    tmp_path,
    generator,
):
    database_path = tmp_path / "empty.sqlite3"
    sqlite3.connect(database_path).close()
    store = MemoryStore(database_path)
    client, _, audio = build_client(store=store, generator=generator)

    response = client.post(
        "/api/lessons/generate",
        json=problem_input().model_dump(),
    )
    job = client.get(f"/api/jobs/{response.json()['job_id']}").json()

    assert job["status"] == "completed"
    assert audio.calls == 1
    with sqlite3.connect(database_path) as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM lessons"
        ).fetchone()[0] == 1


def test_malformed_lessons_schema_fails_before_tts_without_being_swallowed(
    tmp_path,
):
    database_path = tmp_path / "malformed.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE lessons (wrong_column TEXT)")
    store = MemoryStore(database_path)
    audio = FakeAudioService()
    client, _, _ = build_client(store=store, audio_service=audio)

    response = client.post(
        "/api/lessons/generate",
        json=problem_input().model_dump(),
    )
    job = client.get(f"/api/jobs/{response.json()['job_id']}").json()

    assert job["status"] == "failed"
    assert job["error"] == "课程生成失败，请稍后重试。"
    assert audio.calls == 0


def test_bundle_generation_cancellation_propagates_without_persistence():
    class CancelledBundleGenerator:
        async def generate_bundle(self, problem, on_stage=None):
            del problem, on_stage
            raise asyncio.CancelledError()

    store = MemoryStore()
    job = store.create_job()

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            run_generation(
                job.job_id,
                problem_input(),
                store,
                CancelledBundleGenerator(),
                FakeAudioService(),
            )
        )

    assert store.get_lesson("lesson-bundle") is None
    assert store.get_generation_record("lesson-bundle") is None


@pytest.mark.parametrize(
    "mutation",
    [
        "problem",
        "report",
        "board_actions",
        "layer",
        "cue_text",
        "provenance",
        "bundle_lesson",
    ],
)
def test_audio_cannot_mutate_generated_bundle_semantics(mutation):
    generator = BundleGenerator()

    class MutatingAudio(FakeAudioService):
        async def attach_audio(self, lesson, on_stage=None):
            del on_stage
            self.calls += 1
            if mutation == "problem":
                lesson.problem.problem_text = "forged problem"
            elif mutation == "report":
                lesson.validation_report["teaching_route_fingerprint"] = (
                    "forged-route"
                )
            elif mutation == "board_actions":
                lesson.beats[0].board_actions.append(
                    BoardAction(
                        type="write",
                        target="forged",
                        content="forged",
                    )
                )
            elif mutation == "layer":
                lesson.beats[0].layer = "interaction"
            elif mutation == "cue_text":
                lesson.beats[0].sync_cues[0].spoken_text = "forged cue"
            elif mutation == "provenance":
                generator.last_bundle.generation_record.cue_provenance[
                    0
                ].spoken_text = "forged provenance"
            elif mutation == "bundle_lesson":
                generator.last_bundle.lesson.summary = "forged summary"
            return lesson

    store = MemoryStore()
    job = store.create_job()
    asyncio.run(
        run_generation(
            job.job_id,
            problem_input(),
            store,
            generator,
            MutatingAudio(),
        )
    )

    assert store.get_job(job.job_id).status == "failed"
    assert store.get_lesson("lesson-bundle") is None
    assert store.get_generation_record("lesson-bundle") is None


def test_audio_only_url_changes_preserve_valid_bundle():
    generator = BundleGenerator()

    class AudioOnly(FakeAudioService):
        async def attach_audio(self, lesson, on_stage=None):
            return await super().attach_audio(lesson, on_stage=on_stage)

    store = MemoryStore()
    job = store.create_job()
    asyncio.run(
        run_generation(
            job.job_id,
            problem_input(),
            store,
            generator,
            AudioOnly(),
        )
    )

    assert store.get_job(job.job_id).status == "completed"
    assert store.get_generation_record("lesson-bundle") is not None


def test_all_interaction_audio_url_fields_are_allowed_audio_changes():
    class InteractiveBundleGenerator(BundleGenerator):
        async def generate_bundle(self, problem, on_stage=None):
            bundle = await super().generate_bundle(
                problem,
                on_stage=on_stage,
            )
            interaction = Interaction(
                interaction_id="near-transfer",
                kind="choice",
                prompt="请选择。",
                expected_answer="option-a",
                hints=["先想一想。"],
                explanation_after_correct="正确。",
                options=[
                    InteractionOption(
                        option_id="option-a",
                        label="A",
                        feedback="正确。",
                        support_cues=[
                            SupportSyncCue(
                                cue_id="support-a",
                                display_text=r"\(m-n\)",
                                spoken_text="我们要求的是 m 减 n。",
                            )
                        ],
                    )
                ],
            )
            beat = bundle.lesson.beats[0].model_copy(
                update={"interaction": interaction}
            )
            bundle.lesson = bundle.lesson.model_copy(update={"beats": [beat]})
            return bundle

    class InteractionAudioOnly(FakeAudioService):
        async def attach_audio(self, lesson, on_stage=None):
            return await super().attach_audio(lesson, on_stage=on_stage)

    store = MemoryStore()
    job = store.create_job()
    asyncio.run(
        run_generation(
            job.job_id,
            problem_input(),
            store,
            InteractiveBundleGenerator(),
            InteractionAudioOnly(),
        )
    )

    assert store.get_job(job.job_id).status == "completed"


def test_support_audio_cannot_change_support_cue_semantics():
    class InteractiveBundleGenerator(BundleGenerator):
        async def generate_bundle(self, problem, on_stage=None):
            bundle = await super().generate_bundle(
                problem,
                on_stage=on_stage,
            )
            interaction = Interaction(
                interaction_id="near-transfer",
                kind="choice",
                prompt="请选择。",
                expected_answer="option-a",
                options=[
                    InteractionOption(
                        option_id="option-a",
                        label="A",
                        support_cues=[
                            SupportSyncCue(
                                cue_id="support-a",
                                display_text=r"\(m-n\)",
                                spoken_text="我们要求的是 m 减 n。",
                            )
                        ],
                    )
                ],
            )
            beat = bundle.lesson.beats[0].model_copy(
                update={"interaction": interaction}
            )
            bundle.lesson = bundle.lesson.model_copy(update={"beats": [beat]})
            return bundle

    class MutatingSupportAudio(FakeAudioService):
        async def attach_audio(self, lesson, on_stage=None):
            voiced = await super().attach_audio(
                lesson,
                on_stage=on_stage,
            )
            voiced.beats[0].interaction.options[0].support_cues[
                0
            ].spoken_text = "伪造的支持讲解。"
            return voiced

    store = MemoryStore()
    job = store.create_job()
    asyncio.run(
        run_generation(
            job.job_id,
            problem_input(),
            store,
            InteractiveBundleGenerator(),
            MutatingSupportAudio(),
        )
    )

    assert store.get_job(job.job_id).status == "failed"
    assert store.get_lesson("lesson-bundle") is None


@pytest.mark.parametrize(
    "mutation",
    ["missing", "external", "cross_lesson", "swapped"],
)
def test_bundle_audio_requires_exact_local_cue_manifest(mutation):
    class BadManifestAudio(FakeAudioService):
        async def attach_audio(self, lesson, on_stage=None):
            del on_stage
            cues = lesson.beats[0].sync_cues
            expected = [
                f"/audio/lesson-bundle/beat-1-runtime-authored-{index}.mp3"
                for index in range(1, 4)
            ]
            if mutation == "missing":
                return lesson
            if mutation == "external":
                expected[0] = "https://example.com/private.mp3"
            elif mutation == "cross_lesson":
                expected[0] = (
                    "/audio/another-lesson/beat-1-runtime-authored-1.mp3"
                )
            elif mutation == "swapped":
                expected[0], expected[1] = expected[1], expected[0]
            for cue, url in zip(cues, expected):
                cue.audio_url = url
            return lesson

    store = MemoryStore()
    job = store.create_job()
    asyncio.run(
        run_generation(
            job.job_id,
            problem_input(),
            store,
            BundleGenerator(),
            BadManifestAudio(),
        )
    )

    assert store.get_job(job.job_id).status == "failed"
    assert store.get_lesson("lesson-bundle") is None


def test_legacy_audio_requires_exact_local_beat_manifest():
    class ExternalBeatAudio(FakeAudioService):
        async def attach_audio(self, lesson, on_stage=None):
            del on_stage
            lesson.beats[0].audio_url = "https://example.com/beat.mp3"
            return lesson

    store = MemoryStore()
    job = store.create_job()
    asyncio.run(
        run_generation(
            job.job_id,
            problem_input(),
            store,
            FakeGenerator(),
            ExternalBeatAudio(),
        )
    )

    assert store.get_job(job.job_id).status == "failed"
    assert store.get_lesson("lesson-1") is None


class ByteSpeechClient:
    def __init__(self):
        self.texts = []

    async def synthesize(self, text):
        self.texts.append(text)
        return b"audio"


class PersistenceFailureStore(MemoryStore):
    def save_lesson(self, lesson, generation_record=None):
        del lesson, generation_record
        raise OSError("private persistence failure")


def test_persistence_failure_after_audio_removes_invocation_audio(tmp_path):
    store = PersistenceFailureStore()
    job = store.create_job()
    audio = LessonAudioService(ByteSpeechClient(), tmp_path / "audio")

    asyncio.run(
        run_generation(
            job.job_id,
            problem_input(),
            store,
            FakeGenerator(),
            audio,
        )
    )

    assert store.get_job(job.job_id).status == "failed"
    assert not (tmp_path / "audio" / "lesson-1").exists()


def test_duplicate_lesson_is_rejected_before_tts_and_old_audio_is_untouched(
    tmp_path,
):
    store = MemoryStore()
    lesson = runtime_lesson(problem_input())
    store.save_lesson(lesson)
    old_dir = tmp_path / "audio" / lesson.lesson_id
    old_dir.mkdir(parents=True)
    old_audio = old_dir / "old.mp3"
    old_audio.write_bytes(b"old-audio")
    speech = ByteSpeechClient()
    job = store.create_job()

    asyncio.run(
        run_generation(
            job.job_id,
            problem_input(),
            store,
            FakeGenerator(),
            LessonAudioService(speech, tmp_path / "audio"),
        )
    )

    assert store.get_job(job.job_id).status == "failed"
    assert speech.texts == []
    assert old_audio.read_bytes() == b"old-audio"


def test_cancellation_after_audio_cleans_invocation_audio(tmp_path):
    class CancelledStore(MemoryStore):
        def save_lesson(self, lesson, generation_record=None):
            del lesson, generation_record
            raise asyncio.CancelledError()

    store = CancelledStore()
    job = store.create_job()
    audio = LessonAudioService(ByteSpeechClient(), tmp_path / "audio")

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(
            run_generation(
                job.job_id,
                problem_input(),
                store,
                FakeGenerator(),
                audio,
            )
        )

    assert not (tmp_path / "audio" / "lesson-1").exists()


def test_cleanup_failure_does_not_mask_safe_persistence_failure():
    class CleanupFailureAudio(FakeAudioService):
        def __init__(self):
            super().__init__()
            self.cleanup_calls = 0

        def cleanup_lesson_audio(self, lesson_id):
            del lesson_id
            self.cleanup_calls += 1
            raise OSError("private cleanup path")

    store = PersistenceFailureStore()
    job = store.create_job()
    audio = CleanupFailureAudio()
    asyncio.run(
        run_generation(
            job.job_id,
            problem_input(),
            store,
            FakeGenerator(),
            audio,
        )
    )

    failed = store.get_job(job.job_id)
    assert failed.status == "failed"
    assert failed.error == "课程生成失败，请稍后重试。"
    assert "private cleanup path" not in failed.error
    assert audio.cleanup_calls == 1


def test_never_ending_async_cleanup_is_internally_bounded():
    class HangingCleanupAudio(FakeAudioService):
        async def cleanup_lesson_audio(self, lesson_id):
            del lesson_id
            await asyncio.Event().wait()

    async def scenario():
        store = PersistenceFailureStore()
        job = store.create_job()
        loop = asyncio.get_running_loop()
        started_at = loop.time()
        await asyncio.wait_for(
            run_generation(
                job.job_id,
                problem_input(),
                store,
                FakeGenerator(),
                HangingCleanupAudio(),
            ),
            timeout=0.5,
        )
        return store.get_job(job.job_id), loop.time() - started_at

    failed, elapsed = asyncio.run(scenario())
    assert failed.status == "failed"
    assert failed.error == "课程生成失败，请稍后重试。"
    assert elapsed < 0.3


def test_caller_cancellation_during_async_cleanup_propagates():
    class ObservableCleanupAudio(FakeAudioService):
        def __init__(self):
            super().__init__()
            self.cleanup_started = asyncio.Event()

        async def cleanup_lesson_audio(self, lesson_id):
            del lesson_id
            self.cleanup_started.set()
            await asyncio.Event().wait()

    async def scenario():
        store = PersistenceFailureStore()
        job = store.create_job()
        audio = ObservableCleanupAudio()
        task = asyncio.create_task(
            run_generation(
                job.job_id,
                problem_input(),
                store,
                FakeGenerator(),
                audio,
            )
        )
        await asyncio.wait_for(audio.cleanup_started.wait(), timeout=0.2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(scenario())


def build_client(**overrides):
    generator = overrides.pop("generator", FakeGenerator())
    audio_service = overrides.pop("audio_service", FakeAudioService())
    overrides.setdefault("store", MemoryStore())
    app = create_app(
        generator=generator,
        audio_service=audio_service,
        **overrides,
    )
    return TestClient(app), generator, audio_service


def test_create_app_defaults_to_persistent_lesson_database():
    application = create_app(
        generator=FakeGenerator(),
        audio_service=FakeAudioService(),
    )

    assert application.state.store._database_path == (
        PROJECT_ROOT / "var" / "lessons.sqlite3"
    )


def test_memory_store_revalidates_updates_without_partial_mutation():
    store = MemoryStore()
    job = store.create_job()

    with pytest.raises(ValidationError, match="completed jobs require lesson_id"):
        store.update_job(job.job_id, status="completed")

    assert store.get_job(job.job_id).status == "queued"


def test_memory_store_supports_concurrent_job_creation():
    store = MemoryStore()

    with ThreadPoolExecutor(max_workers=8) as executor:
        jobs = list(executor.map(lambda _: store.create_job(), range(40)))

    assert len({job.job_id for job in jobs}) == 40
    assert all(store.get_job(job.job_id) is not None for job in jobs)


def test_health_reports_configuration_without_exposing_secrets():
    settings = Settings(
        openai_base_url="https://model.example/v1",
        openai_api_key="model-secret",
        openai_model="demo-model",
        tts_base_url="https://speech.example/v1",
        tts_api_key="speech-secret",
        tts_model="demo-voice-model",
        tts_voice="demo-voice",
    )
    client, _, _ = build_client(settings=settings)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_configured": True,
        "voice_configured": True,
    }
    assert "model-secret" not in response.text
    assert "speech-secret" not in response.text


def test_generation_job_completes_with_public_stage_sequence():
    store = RecordingStore()
    client, generator, audio_service = build_client(store=store)

    response = client.post(
        "/api/lessons/generate",
        json=problem_input().model_dump(),
    )

    assert response.status_code == 202
    assert response.json() == {"job_id": response.json()["job_id"]}
    job_id = response.json()["job_id"]
    job = client.get(f"/api/jobs/{job_id}")
    assert job.status_code == 200
    assert job.json()["status"] == "completed"
    assert job.json()["stage"] == "课程已生成"
    lesson = client.get(f"/api/lessons/{job.json()['lesson_id']}")
    assert lesson.status_code == 200
    assert lesson.json()["lesson_id"] == "lesson-1"
    assert generator.calls == 1
    assert audio_service.calls == 1
    assert store.seen_stages == [
        "正在理解题目",
        "正在核对题目材料",
        "正在编写讲稿",
        "正在审核和优化课程",
        "正在编译课程",
        "正在生成语音",
        "正在保存课程",
        "课程已生成",
    ]


def test_preparation_pipeline_stages_advance_public_progress_monotonically():
    store = RecordingStore()
    generator = PreparationStageGenerator()
    client, _, audio_service = build_client(
        store=store,
        generator=generator,
    )

    response = client.post(
        "/api/lessons/generate",
        json=problem_input().model_dump(),
    )

    assert response.status_code == 202
    assert generator.calls == 1
    assert audio_service.calls == 1
    assert store.seen_stages == _DETAILED_PUBLIC_STAGES


def test_script_repair_after_review_does_not_regress_public_progress():
    store = RecordingStore()
    generator = RepairStageGenerator(
        [
            [
                "编写讲稿",
                "设计互动",
                "编排板书与高亮",
                "模拟学生并审核课程",
            ]
        ]
    )
    client, _, _ = build_client(store=store, generator=generator)

    response = client.post(
        "/api/lessons/generate",
        json=problem_input().model_dump(),
    )

    assert response.status_code == 202
    assert store.seen_stages == _DETAILED_PUBLIC_STAGES


@pytest.mark.parametrize(
    "repair_rounds",
    [
        [
            [
                "整理参考解析",
                "设计解题思维轨迹",
                "编写讲稿",
                "设计互动",
                "编排板书与高亮",
                "模拟学生并审核课程",
            ],
            [
                "编排板书与高亮",
                "模拟学生并审核课程",
            ],
        ],
        [
            [
                "设计解题思维轨迹",
                "编写讲稿",
                "设计互动",
                "编排板书与高亮",
                "模拟学生并审核课程",
            ],
            [
                "设计互动",
                "编排板书与高亮",
                "模拟学生并审核课程",
            ],
            [
                "编写讲稿",
                "设计互动",
                "编排板书与高亮",
                "模拟学生并审核课程",
            ],
        ],
    ],
)
def test_multiple_repairs_from_different_roles_keep_public_progress_monotonic(
    repair_rounds,
):
    store = RecordingStore()
    generator = RepairStageGenerator(repair_rounds)
    client, _, _ = build_client(store=store, generator=generator)

    response = client.post(
        "/api/lessons/generate",
        json=problem_input().model_dump(),
    )

    assert response.status_code == 202
    assert store.seen_stages == _DETAILED_PUBLIC_STAGES


def test_concurrent_generation_jobs_keep_independent_progress_state():
    store = PerJobRecordingStore()
    first_job = store.create_job()
    second_job = store.create_job()
    first_generator = _generation_service_with_pipeline(
        preparation_client(progression_target_ids=[])
    )
    finding = review_finding("classroom_director")
    second_generator = _generation_service_with_pipeline(
        preparation_client(
            progression_target_ids=[],
            performances=[
                downstream_score_payload(),
                downstream_score_payload(),
            ],
            simulations=[
                downstream_simulation_payload(),
                downstream_simulation_payload(),
            ],
            reviews=[
                downstream_review_payload("revision_required", [finding]),
                downstream_review_payload(),
            ],
        )
    )

    async def run_both():
        await asyncio.gather(
                run_generation(
                    first_job.job_id,
                    preparation_problem(),
                store,
                first_generator,
                FakeAudioService(),
            ),
                run_generation(
                    second_job.job_id,
                    preparation_problem(),
                store,
                second_generator,
                FakeAudioService(),
            ),
        )

    asyncio.run(run_both())

    assert (
        store.seen_stages_by_job[first_job.job_id]
        == _DETAILED_PUBLIC_STAGES
    )
    assert (
        store.seen_stages_by_job[second_job.job_id]
        == _DETAILED_PUBLIC_STAGES
    )


@pytest.mark.parametrize(
    ("internal_stage", "public_stage"),
    [
        ("正在验证数学路线", "正在核对题目材料"),
        ("正在规划数学路线", "正在核对题目材料"),
        ("正在审阅参考解析", "正在整理参考解析"),
        ("正在整理参考教学路线", "正在整理参考解析"),
    ],
)
def test_capability_and_grounding_use_their_public_generation_stage(
    internal_stage, public_stage,
):
    assert _PUBLIC_GENERATION_STAGES[internal_stage] == public_stage


def test_generation_with_reference_solution_keeps_audit_internal():
    store = RecordingStore()
    client, _, _ = build_client(store=store)
    payload = problem_input().model_copy(
        update={
            "reference_solution_text": (
                "解：2x+3=7，所以 2x=4，最终 x=2。"
            )
        }
    )

    response = client.post(
        "/api/lessons/generate",
        json=payload.model_dump(),
    )

    assert response.status_code == 202
    assert "正在整理参考解析" in store.seen_stages
    assert "正在审阅参考解析" not in store.seen_stages


def test_generation_failure_is_sanitized_and_has_no_lesson():
    private_error = RuntimeError(
        "provider body, api-key=secret, system prompt=private"
    )
    client, _, audio_service = build_client(
        generator=FakeGenerator(private_error)
    )

    response = client.post(
        "/api/lessons/generate",
        json=problem_input().model_dump(),
    )

    assert response.status_code == 202
    job = client.get(f"/api/jobs/{response.json()['job_id']}").json()
    assert job["status"] == "failed"
    assert job["stage"] == "生成失败"
    assert job["error"] == "课程生成失败，请稍后重试。"
    assert "secret" not in str(job)
    assert audio_service.calls == 0
    assert "private" not in str(job)
    assert client.get("/api/lessons/lesson-1").status_code == 404


def test_nonconverged_preparation_never_reaches_compiler_or_audio_service():
    class NonconvergedPipeline:
        async def prepare_with_audit(self, *args, **kwargs):
            raise PreparationFailure(
                category="review_not_converged",
                role="lesson_reviewer",
                detail="private review findings",
            )

    class CompilerSpy:
        def __init__(self):
            self.calls = 0

        def compile(self, *args, **kwargs):
            self.calls += 1
            raise AssertionError("compiler must not run")

    compiler = CompilerSpy()
    generator = LessonGenerationService(
        FakeClient([]),
        MathEngine(),
        compiler=compiler,
        preparation_pipeline=NonconvergedPipeline(),
    )
    audio_service = FakeAudioService()
    store = RecordingStore()
    job = store.create_job()
    source_problem = ProblemInput(
        problem_text="用配方法解方程：x^2-6*x+5=0",
        reference_answer="x=1 或 x=5",
        required_method="complete_the_square",
    )

    asyncio.run(
        run_generation(
            job.job_id,
            source_problem,
            store,
            generator,
            audio_service,
        )
    )

    failed = store.get_job(job.job_id)
    assert failed.status == "failed"
    assert failed.error == "课程生成失败，请稍后重试。"
    assert "private review findings" not in failed.error
    assert compiler.calls == 0
    assert audio_service.calls == 0


def test_compiler_failure_never_reaches_audio_service():
    class FailingCompiler:
        def compile(self, *args, **kwargs):
            del args, kwargs
            raise LessonCompileError("private compiler output")

    generator = _generation_service_with_pipeline(
        preparation_client(progression_target_ids=[])
    )
    generator.compiler = FailingCompiler()
    audio_service = FakeAudioService()
    store = RecordingStore()
    job = store.create_job()

    asyncio.run(
        run_generation(
            job.job_id,
            preparation_problem(),
            store,
            generator,
            audio_service,
        )
    )

    failed = store.get_job(job.job_id)
    assert failed.status == "failed"
    assert failed.error == "课程生成失败，请稍后重试。"
    assert "private compiler output" not in failed.error
    assert store.get_job_diagnostic(job.job_id).category == "compile_failed"
    assert audio_service.calls == 0


def test_generation_does_not_return_id_when_persistence_fails():
    store = FailingLessonStore()
    client, _, _ = build_client(store=store)

    response = client.post(
        "/api/lessons/generate",
        json=problem_input().model_dump(),
    )

    assert response.status_code == 202
    job = client.get(f"/api/jobs/{response.json()['job_id']}").json()
    assert job["status"] == "failed"
    assert job["lesson_id"] is None
    assert job["error"] == "课程生成失败，请稍后重试。"
    assert "private database path" not in str(job)


def test_new_app_instance_reads_persisted_lesson_and_interaction(
    tmp_path,
):
    database = tmp_path / "lessons.sqlite3"
    first_store = MemoryStore(database)
    lesson, interaction = save_interaction_lesson(
        first_store,
        kind="choice",
        expected="o1",
        options=[
            InteractionOption(option_id="o1", label="x=2"),
            InteractionOption(option_id="o2", label="x=5"),
        ],
    )
    second_client, _, _ = build_client(store=MemoryStore(database))

    lesson_response = second_client.get(
        f"/api/lessons/{lesson.lesson_id}"
    )
    evaluation = second_client.post(
        "/api/interactions/evaluate",
        json={
            "lesson_id": lesson.lesson_id,
            "interaction_id": interaction.interaction_id,
            "answer": "o1",
        },
    )

    assert lesson_response.status_code == 200
    assert "reference_answer" not in lesson_response.json()["problem"]
    assert evaluation.json()["classification"] == "correct"


def test_same_problem_payload_generates_distinct_persisted_lessons(
    tmp_path,
):
    database = tmp_path / "lessons.sqlite3"
    generator = SequentialIdGenerator()
    client, _, _ = build_client(
        store=MemoryStore(database),
        generator=generator,
    )

    first = client.post(
        "/api/lessons/generate",
        json=problem_input().model_dump(),
    )
    second = client.post(
        "/api/lessons/generate",
        json=problem_input().model_dump(),
    )
    first_job = client.get(
        f"/api/jobs/{first.json()['job_id']}"
    ).json()
    second_job = client.get(
        f"/api/jobs/{second.json()['job_id']}"
    ).json()

    assert first_job["status"] == "completed"
    assert second_job["status"] == "completed"
    assert first_job["lesson_id"] != second_job["lesson_id"]
    assert generator.calls == 2
    restarted_store = MemoryStore(database)
    assert restarted_store.get_lesson(first_job["lesson_id"]) is not None
    assert restarted_store.get_lesson(second_job["lesson_id"]) is not None


def test_generation_failure_exposes_only_typed_safe_input_errors():
    public_message = "参考解析与题目或参考答案存在数学冲突，请检查后再试。"

    error = LessonInputError(public_message)
    assert safe_generation_error(
        error, trusted_input_validation=True
    ) == public_message
    assert safe_generation_error(error) == "课程生成失败，请稍后重试。"
    assert safe_generation_error(RuntimeError("private-provider-detail")) == (
        "课程生成失败，请稍后重试。"
    )


def test_unsupported_math_does_not_return_math_validation_error():
    error = safe_generation_error(
        LessonInputError("题目格式不正确。"),
        trusted_input_validation=True,
    )

    assert error == "题目格式不正确。"


def test_contradiction_returns_specific_safe_message():
    assert safe_generation_error(
        LessonInputError("参考答案与题目实际结果不一致。"),
        trusted_input_validation=True,
    ) == "参考答案与题目实际结果不一致。"


def test_missing_job_and_lesson_return_404():
    client, _, _ = build_client()

    job_response = client.get("/api/jobs/missing")
    lesson_response = client.get("/api/lessons/missing")

    assert job_response.status_code == 404
    assert lesson_response.status_code == 404


def test_public_lesson_payload_redacts_answers_and_review_internals():
    lesson = RuntimeLesson.model_validate(
        {
            "lesson_id": "lesson-public",
            "problem": {
                "problem_text": "解方程 x^2-6x+9=0",
                "reference_answer": "x=271828",
                "reference_solution_text": (
                    "解：这是不应进入学生课堂的内部参考解析。"
                ),
            },
            "title": "公开课程",
            "learning_goal": "理解完全平方结构。",
            "problem_focus_targets": [
                {
                    "target_id": "problem-math-001",
                    "math_text": "x^2-6x+9=0",
                    "display_mode": False,
                    "ordinal": 1,
                }
            ],
            "beats": [
                {
                    "beat_id": "beat-expression",
                    "purpose": "识别结构",
                    "narration": "请写出对应的完全平方。",
                    "board_actions": [],
                    "layer": "interaction",
                    "audio_url": "/audio/lesson-public/beat-expression.mp3",
                    "sync_cues": [
                        {
                            "cue_id": "identify-square-cue",
                            "spoken_text": "请先观察一次项和常数项。",
                            "start_actions": [
                                {
                                    "surface": "problem",
                                    "type": "focus",
                                    "target": "problem-math-001",
                                }
                            ],
                            "audio_url": (
                                "/audio/lesson-public/"
                                "identify-square-cue.mp3"
                            ),
                        }
                    ],
                    "interaction": {
                        "interaction_id": "expression-check",
                        "kind": "expression",
                        "prompt": "写出完全平方。",
                        "expected_answer": "x^2-6*x+9",
                        "hints": ["观察中间项。"],
                        "hint_audio_urls": [
                            "/audio/lesson-public/hint-1.mp3",
                        ],
                        "correct_audio_url": (
                            "/audio/lesson-public/correct.mp3"
                        ),
                    },
                    "next_beat_id": "beat-choice",
                },
                {
                    "beat_id": "beat-choice",
                    "purpose": "选择方法",
                    "narration": "请选择下一步。",
                    "board_actions": [],
                    "layer": "interaction",
                    "interaction": {
                        "interaction_id": "choice-check",
                        "kind": "choice",
                        "prompt": "选择方法。",
                        "expected_answer": "correct-option",
                        "options": [
                            {
                                "option_id": "correct-option",
                                "label": "配方法",
                            },
                            {
                                "option_id": "other-option",
                                "label": "其他方法",
                                "feedback": "这个式子已经是完全平方。",
                                "feedback_audio_url": (
                                    "/audio/lesson-public/other-option.mp3"
                                ),
                            },
                        ],
                    },
                },
            ],
            "summary": "识别完全平方。",
            "transfer_item": {
                "problem_text": "解方程 x^2-8x+16=0",
                "expected_answer": "x=314159",
                "method_signal": "观察完全平方。",
            },
            "validation_report": {
                "independent_solutions": ["validation-secret"],
                "review_assessment": "private-review",
            },
        }
    )
    store = MemoryStore()
    store.save_lesson(lesson)
    client, _, _ = build_client(store=store)

    response = client.get("/api/lessons/lesson-public")

    assert response.status_code == 200
    payload = response.json()
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "reference_answer" not in payload["problem"]
    assert "reference_solution_text" not in payload["problem"]
    assert "expected_answer" not in payload["transfer_item"]
    assert "validation_report" not in payload
    assert all(
        "expected_answer" not in beat["interaction"]
        for beat in payload["beats"]
    )
    assert "x=271828" not in serialized
    assert "不应进入学生课堂" not in serialized
    assert "x^2-6*x+9" not in serialized
    assert "x=314159" not in serialized
    assert "validation-secret" not in serialized
    assert "private-review" not in serialized
    assert payload["problem"]["problem_text"] == "解方程 x^2-6x+9=0"
    assert payload["problem_focus_targets"] == [
        {
            "target_id": "problem-math-001",
            "math_text": "x^2-6x+9=0",
            "display_mode": False,
            "ordinal": 1,
        }
    ]
    assert payload["beats"][0]["sync_cues"] == [
        {
            "cue_id": "identify-square-cue",
            "spoken_text": "请先观察一次项和常数项。",
            "lead_actions": [],
            "start_actions": [
                {
                    "surface": "problem",
                    "type": "focus",
                    "target": "problem-math-001",
                    "content": None,
                    "source": None,
                    "relation_target": None,
                    "annotation": None,
                    "emphasis_style": None,
                    "persistence": None,
                }
            ],
            "end_actions": [],
            "audio_url": (
                "/audio/lesson-public/identify-square-cue.mp3"
            ),
        }
    ]
    assert payload["beats"][0]["audio_url"].endswith(
        "beat-expression.mp3"
    )
    assert payload["beats"][0]["interaction"]["hints"] == [
        "观察中间项。"
    ]
    assert payload["beats"][0]["interaction"]["hint_audio_urls"]
    assert "explanation_after_correct" not in (
        payload["beats"][0]["interaction"]
    )
    assert "correct_audio_url" not in payload["beats"][0]["interaction"]
    assert payload["beats"][1]["interaction"]["options"] == [
        {"option_id": "correct-option", "label": "配方法"},
        {"option_id": "other-option", "label": "其他方法"},
    ]
    assert "这个式子已经是完全平方" not in serialized
    assert "other-option.mp3" not in serialized
    assert "correct.mp3" not in serialized

    evaluation = client.post(
        "/api/interactions/evaluate",
        json={
            "lesson_id": "lesson-public",
            "interaction_id": "expression-check",
            "answer": "(x-3)^2",
        },
    )
    assert evaluation.json() == {"classification": "correct"}


def test_public_grounded_transfer_redacts_review_evidence_and_feedback():
    lesson = runtime_lesson(problem_input()).model_copy(
        update={
            "transfer_item": TransferItem(
                problem_text=(
                    "若a（a≠0）是方程x^2-px+a=0的根，"
                    "把x=a代入后首先得到哪个等式？"
                ),
                expected_answer="option-substitute",
                method_signal="把已知根代回原方程",
                options=[
                    TransferOption(
                        option_id="option-substitute",
                        label=r"\(a^2-pa+a=0\)",
                        canonical_answer="a^2-p*a+a=0",
                        feedback="对，根代入原方程后等式成立。",
                    ),
                    TransferOption(
                        option_id="option-miss-square",
                        label=r"\(a-pa+a=0\)",
                        canonical_answer="a-p*a+a=0",
                        feedback="代入后x平方应变成a平方。",
                    ),
                    TransferOption(
                        option_id="option-wrong-target",
                        label=r"\(x^2-pa+a=0\)",
                        canonical_answer="x^2-p*a+a=0",
                        feedback="这里还没有把x替换为已知根a。",
                    ),
                ],
                correct_option_id="option-substitute",
            ),
            "validation_report": {
                "verification_mode": "model_cross_checked",
                "review_status": "approved",
                "model_disagreement": "private-model-disagreement",
                "check_requests": ["private-check-request"],
            },
        }
    )
    store = MemoryStore()
    store.save_lesson(lesson)
    client, _, _ = build_client(store=store)

    response = client.get(f"/api/lessons/{lesson.lesson_id}")

    assert response.status_code == 200
    serialized = json.dumps(response.json(), ensure_ascii=False)
    for private_key in (
        "verification_mode",
        "model_disagreement",
        "check_requests",
        "reference_solution_text",
        "private-model-disagreement",
        "private-check-request",
    ):
        assert private_key not in serialized
    transfer_item = response.json()["transfer_item"]
    assert "expected_answer" not in transfer_item
    assert "correct_option_id" not in transfer_item
    assert all(
        "canonical_answer" not in option
        for option in transfer_item["options"]
    )
    assert transfer_item["options"] == [
        {
            "option_id": "option-substitute",
            "label": r"\(a^2-pa+a=0\)",
        },
        {
            "option_id": "option-miss-square",
            "label": r"\(a-pa+a=0\)",
        },
        {
            "option_id": "option-wrong-target",
            "label": r"\(x^2-pa+a=0\)",
        },
    ]


def test_public_compiled_diagnostic_transfer_hides_answer_key():
    draft = valid_draft()
    lesson = LessonCompiler(
        lesson_id_factory=lambda: "lesson-compiled-choice"
    ).compile(
        problem(),
        LessonDraft.model_validate(draft),
        {"review_status": "approved"},
    )
    store = MemoryStore()
    store.save_lesson(lesson)
    client, _, _ = build_client(store=store)

    response = client.get(f"/api/lessons/{lesson.lesson_id}")

    assert response.status_code == 200
    transfer = response.json()["beats"][-1]["interaction"]
    assert transfer["kind"] == "choice"
    assert "expected_answer" not in transfer
    assert transfer["options"] == [
        {
            "option_id": option["option_id"],
            "label": option["label"],
        }
        for option in draft["transfer_item"]["options"]
    ]


def test_choice_evaluation_returns_only_submitted_option_feedback():
    store = MemoryStore()
    correct_option = InteractionOption(
        option_id="correct-option",
        label="正确选项",
        feedback="correct-private-feedback",
        feedback_audio_url="/audio/correct-private.mp3",
        support_cues=[
            SupportSyncCue(
                cue_id="correct-support",
                display_text="correct-support-secret",
                spoken_text="正确后的支持讲解。",
            )
        ],
    )
    wrong_option = InteractionOption(
        option_id="wrong-option",
        label="错误选项",
        feedback="wrong-private-feedback",
        feedback_audio_url="/audio/wrong-private.mp3",
        support_cues=[
            SupportSyncCue(
                cue_id="wrong-support",
                display_text="wrong-support-secret",
                spoken_text="错误后的定向支持讲解。",
            )
        ],
    )
    lesson, interaction = save_interaction_lesson(
        store,
        kind="choice",
        expected="correct-option",
        options=[
            correct_option,
            wrong_option,
        ],
    )
    client, _, _ = build_client(store=store)

    public_response = client.get(f"/api/lessons/{lesson.lesson_id}")
    wrong_response = client.post(
        "/api/interactions/evaluate",
        json={
            "lesson_id": lesson.lesson_id,
            "interaction_id": interaction.interaction_id,
            "answer": "wrong-option",
        },
    )
    correct_response = client.post(
        "/api/interactions/evaluate",
        json={
            "lesson_id": lesson.lesson_id,
            "interaction_id": interaction.interaction_id,
            "answer": "correct-option",
        },
    )
    unselected_response = client.post(
        "/api/interactions/evaluate",
        json={
            "lesson_id": lesson.lesson_id,
            "interaction_id": interaction.interaction_id,
            "answer": "unknown-option",
        },
    )

    public_serialized = json.dumps(
        public_response.json(),
        ensure_ascii=False,
    )
    assert "correct-private-feedback" not in public_serialized
    assert "wrong-private-feedback" not in public_serialized
    assert "correct-private.mp3" not in public_serialized
    assert "wrong-private.mp3" not in public_serialized
    assert "correct-support-secret" not in public_serialized
    assert "wrong-support-secret" not in public_serialized
    assert all(
        "support_cues" not in option
        for option in public_response.json()["beats"][0]["interaction"]["options"]
    )
    assert wrong_response.json() == {
        "classification": "incorrect",
        "feedback": "wrong-private-feedback",
        "feedback_audio_url": "/audio/wrong-private.mp3",
        "support_cues": wrong_option.model_dump(mode="json")["support_cues"],
    }
    assert "correct-private" not in wrong_response.text
    assert correct_response.json() == {
        "classification": "correct",
        "feedback": "correct-private-feedback",
        "feedback_audio_url": "/audio/correct-private.mp3",
        "support_cues": correct_option.model_dump(mode="json")["support_cues"],
    }
    assert "wrong-private" not in correct_response.text
    assert "wrong-support-secret" not in correct_response.text
    assert "correct-support-secret" not in wrong_response.text
    assert unselected_response.json() == {"classification": "incorrect"}


@pytest.mark.parametrize("kind", ["choice", "point_select"])
def test_choice_and_point_select_use_trimmed_exact_comparison(kind):
    store = MemoryStore()
    options = (
        [InteractionOption(option_id="A", label="选项 A")]
        if kind == "choice"
        else []
    )
    lesson, interaction = save_interaction_lesson(
        store,
        kind=kind,
        expected="A",
        options=options,
    )
    client, _, _ = build_client(store=store)

    correct = client.post(
        "/api/interactions/evaluate",
        json={
            "lesson_id": lesson.lesson_id,
            "interaction_id": interaction.interaction_id,
            "answer": " A ",
        },
    )
    incorrect = client.post(
        "/api/interactions/evaluate",
        json={
            "lesson_id": lesson.lesson_id,
            "interaction_id": interaction.interaction_id,
            "answer": "a",
        },
    )

    expected_correct = {"classification": "correct"}
    if kind == "choice":
        expected_correct["support_cues"] = []
    assert correct.json() == expected_correct
    assert incorrect.json() == {"classification": "incorrect"}


def test_expression_interaction_uses_math_equivalence():
    store = MemoryStore()
    lesson, interaction = save_interaction_lesson(
        store,
        kind="expression",
        expected="(x-3)^2",
    )
    client, _, _ = build_client(store=store)

    equivalent = client.post(
        "/api/interactions/evaluate",
        json={
            "lesson_id": lesson.lesson_id,
            "interaction_id": interaction.interaction_id,
            "answer": "x^2-6x+9",
        },
    )
    malformed = client.post(
        "/api/interactions/evaluate",
        json={
            "lesson_id": lesson.lesson_id,
            "interaction_id": interaction.interaction_id,
            "answer": "not math",
        },
    )

    assert equivalent.json() == {"classification": "correct"}
    assert malformed.json() == {"classification": "incorrect"}


def test_transfer_interaction_uses_answer_equivalence():
    store = MemoryStore()
    lesson, interaction = save_interaction_lesson(
        store,
        kind="transfer",
        expected="x=2",
    )
    client, _, _ = build_client(store=store)

    response = client.post(
        "/api/interactions/evaluate",
        json={
            "lesson_id": lesson.lesson_id,
            "interaction_id": interaction.interaction_id,
            "answer": "x=4/2",
        },
    )

    assert response.json() == {"classification": "correct"}


def test_free_text_always_needs_review_without_calling_generator():
    store = MemoryStore()
    lesson, interaction = save_interaction_lesson(
        store,
        kind="free_text",
        expected="等式两边做相同运算。",
    )
    client, generator, _ = build_client(store=store)

    response = client.post(
        "/api/interactions/evaluate",
        json={
            "lesson_id": lesson.lesson_id,
            "interaction_id": interaction.interaction_id,
            "answer": "我的思路包含 private student text",
        },
    )

    assert response.json() == {
        "classification": "needs_review",
        "message": "首版暂不自动判错这类文字回答。",
    }
    assert generator.calls == 0


def test_interaction_evaluation_rejects_forged_expected_answer():
    store = MemoryStore()
    lesson, interaction = save_interaction_lesson(
        store,
        kind="choice",
        expected="A",
        options=[InteractionOption(option_id="A", label="选项 A")],
    )
    client, _, _ = build_client(store=store)

    forged = client.post(
        "/api/interactions/evaluate",
        json={
            "lesson_id": lesson.lesson_id,
            "interaction_id": interaction.interaction_id,
            "answer": "forged",
            "expected": "forged",
        },
    )
    incorrect = client.post(
        "/api/interactions/evaluate",
        json={
            "lesson_id": lesson.lesson_id,
            "interaction_id": interaction.interaction_id,
            "answer": "forged",
        },
    )

    assert forged.status_code == 422
    assert incorrect.json() == {"classification": "incorrect"}


@pytest.mark.parametrize(
    ("lesson_id", "interaction_id"),
    [
        ("missing", "interaction-1"),
        ("lesson-1", "missing"),
    ],
)
def test_interaction_evaluation_returns_safe_404_for_missing_authority(
    lesson_id,
    interaction_id,
):
    store = MemoryStore()
    save_interaction_lesson(
        store,
        kind="free_text",
        expected="参考答案",
    )
    client, _, _ = build_client(store=store)

    response = client.post(
        "/api/interactions/evaluate",
        json={
            "lesson_id": lesson_id,
            "interaction_id": interaction_id,
            "answer": "学生答案",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "课程互动不存在。"


def test_run_generation_records_failed_job_without_raising():
    store = MemoryStore()
    job = store.create_job()

    asyncio.run(
        run_generation(
            job.job_id,
            problem_input(),
            store,
            FakeGenerator(RuntimeError("private")),
            FakeAudioService(),
        )
    )

    assert store.get_job(job.job_id).status == "failed"


def test_lifespan_does_not_close_injected_services():
    class ClosableGenerator(FakeGenerator):
        def __init__(self):
            super().__init__()
            self.close_calls = 0

        async def close(self):
            self.close_calls += 1

    class ClosableAudioService(FakeAudioService):
        def __init__(self):
            super().__init__()
            self.close_calls = 0

        async def close(self):
            self.close_calls += 1

    generator = ClosableGenerator()
    audio_service = ClosableAudioService()
    app = create_app(generator=generator, audio_service=audio_service)

    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200

    assert generator.close_calls == 0
    assert audio_service.close_calls == 0


def _parameter_root_interactions():
    return [
        {
            "interaction_id": "substitution-check",
            "episode_id": "episode-2",
            "teaching_step_id": "teaching-step-2",
            "after_clause_id": "clause-2",
            "why_pause": "此处停下，检查学生是否把2n代入x。",
            "diagnostic_target": "是否把2n代入x",
            "diagnostic_kind": "conception",
            "prompt": "2n应该代入方程中的谁？",
            "options": [
                {
                    "option_id": "sub-correct",
                    "display_text": "代入x",
                    "canonical_answer": "substitute-x",
                },
                {
                    "option_id": "sub-wrong-variable",
                    "display_text": "代入m",
                    "canonical_answer": "substitute-m",
                    "misconception": "把根代给了参数m",
                    "error_code": "substitution-variable-error",
                    "remediation_depth": "conceptual",
                },
                {
                    "option_id": "sub-wrong-nonzero",
                    "display_text": "先约去n",
                    "canonical_answer": "divide-n-first",
                    "misconception": "还没有代入就先约去n",
                    "error_code": "nonzero-condition-error",
                    "remediation_depth": "conceptual",
                },
            ],
            "correct_option_id": "sub-correct",
            "correct_feedback": "对，关于x的方程就代入x。",
            "incorrect_feedback_by_option": {
                "sub-wrong-variable": "关于x的方程只替换x。",
                "sub-wrong-nonzero": "先代入并化简，再判断能否约分。",
            },
            "hint": "看清楚这是关于谁的方程。",
            "resume_clause_id": "clause-2-resume",
            "resume_step_id": "teaching-step-2",
            "resume_policy": "continue",
            "concealed_targets": [],
        },
        {
            "interaction_id": "square-check",
            "episode_id": "episode-3",
            "teaching_step_id": "teaching-step-3",
            "after_clause_id": "clause-3",
            "why_pause": "此处停下，检查学生是否会计算2n的整体平方。",
            "diagnostic_target": "是否会计算2n的整体平方",
            "diagnostic_kind": "execution",
            "prompt": "计算2n的整体平方时，哪一个式子正确？",
            "options": [
                {
                    "option_id": "square-correct",
                    "display_text": "(2n)^2=4n^2",
                    "canonical_answer": "(2n)^2=4n^2",
                },
                {
                    "option_id": "square-wrong",
                    "display_text": "(2n)^2=2n^2",
                    "canonical_answer": "(2n)^2=2n^2",
                    "misconception": "只平方了n，没有平方系数2",
                    "error_code": "square-distribution-error",
                    "remediation_depth": "worked",
                },
                {
                    "option_id": "opposite-wrong",
                    "display_text": "n-m=m-n",
                    "canonical_answer": "same-expression",
                    "misconception": "把n-m直接当成m-n",
                    "error_code": "opposite-expression-error",
                    "remediation_depth": "conceptual",
                },
            ],
            "correct_option_id": "square-correct",
            "correct_feedback": "对，系数2和n都要平方。",
            "incorrect_feedback_by_option": {
                "square-wrong": "中间一步是2的平方乘n的平方，所以得到4n^2。",
                "opposite-wrong": "n-m与m-n互为相反数。",
            },
            "hint": "平方作用于整个2n。",
            "resume_clause_id": "clause-3",
            "resume_step_id": "teaching-step-3",
            "resume_policy": "continue",
            "concealed_targets": [],
        },
    ]


def _parameter_root_preparation_client():
    interactions = _parameter_root_interactions()
    progression = parameter_progression_payload(
        ["problem-math-001", "problem-math-002", "problem-math-003"]
    )
    labels = [
        "第一步：理解方程的根",
        "第二步：代入方程",
        "第三步：展开并化简",
        "第四步：利用n不等于0约去n",
        "第五步：整理出m-n",
    ]
    for step, label in zip(progression["steps"], labels):
        step["directory_label"] = label
    progression["steps"][1]["checkpoint"] = {
        "diagnostic_goal": "检查学生是否把2n代入x",
        "misconception_ids": [
            "misconception-002-001",
            "misconception-002-002",
        ],
    }
    progression["steps"][2]["checkpoint"] = {
        "diagnostic_goal": "检查学生是否会计算2n的整体平方",
        "misconception_ids": [
            "misconception-003-001",
            "misconception-003-002",
        ],
    }
    plan = downstream_interaction_payload()
    plan["interactions"] = interactions
    script = downstream_script_payload(interactions)
    clause_by_id = {
        clause["clause_id"]: clause for clause in script["clauses"]
    }
    clause_by_id["clause-open"].update(
        display_text="方程的根：代入后等式仍成立。",
        spoken_text="方程的根，代入以后等式仍然成立。",
    )
    clause_by_id["clause-method"].update(
        display_text="关于x的方程，只代入x。",
        spoken_text="这是关于 x 的方程，所以只把根代入 x。",
    )
    clause_by_id["clause-2"].update(
        display_text="将x=2n代入原方程。",
        spoken_text="将 x 等于二 n 代入原方程。",
    )
    clause_by_id["clause-3"].update(
        display_text="(2n)^2=4n^2，展开并化简。",
        spoken_text="二 n 的整体平方等于四 n 的平方，然后展开并化简。",
    )
    clause_by_id["clause-4"].update(
        display_text="因为n≠0，可以约去n。",
        spoken_text="因为 n 不等于零，所以可以约去 n。",
    )
    clause_by_id["clause-close"].update(
        display_text=r"所以 $m-n=\frac{1}{2}$。",
        spoken_text="所以 m 减 n 等于二分之一。",
    )
    score = downstream_score_payload(interactions)
    label_by_step = {
        step["step_id"]: step["directory_label"]
        for step in progression["steps"]
    }
    for cue in score["cues"]:
        for phase in ("lead_actions", "start_actions", "end_actions"):
            for binding in cue.get(phase, []):
                action = binding["action"]
                if action["type"] == "reveal_step_header":
                    action["step_label"] = label_by_step[
                        action["teaching_step_id"]
                    ]
    simulation = downstream_simulation_payload()
    simulation["interaction_results"] = [
        "学生能在代入错误后修正并继续。",
        "学生能在平方错误后借助中间步骤修正并继续。",
    ]
    trajectory = trajectory_payload()
    must_teach_contents = (
        "方程的根代入后等式仍成立",
        "将x=2n代入原方程",
        "(2n)^2=4n^2",
        "n≠0，可以约去n",
        "m-n",
    )
    for episode, clause_id, content in zip(
        trajectory["episodes"],
        ("clause-open", "clause-2", "clause-3", "clause-4", "clause-close"),
        must_teach_contents,
    ):
        evidence = episode["must_teach"][0]
        evidence["content"] = content
        evidence["student_display_evidence"] = clause_by_id[clause_id][
            "display_text"
        ]
        evidence["student_spoken_evidence"] = clause_by_id[clause_id][
            "spoken_text"
        ]
    trajectory["episodes"][1]["likely_misconceptions"] = [
        "把根代给参数",
        "还未代入就约分",
    ]
    trajectory["episodes"][2]["likely_misconceptions"] = [
        "只平方字母不平方系数",
        "混淆互为相反数的式子",
    ]
    preparation = preparation_client(
        trace=trace_payload(),
        trajectory=trajectory,
        progression=progression,
        interaction=plan,
        script=script,
        performance=score,
        simulations=[simulation],
        reviews=[downstream_review_payload()],
    )
    return CompositeGenerationClient(FakeClient([]), preparation)


def test_parameter_root_deterministic_pipeline_persistence_and_interactions(
    tmp_path,
):
    source = preparation_problem().model_copy(
        update={
            "problem_text": (
                "若$2n$ ($n\\ne 0$)是关于 x的方程 "
                "$x^2-2mx+2n=0$的根，则m-n的值为"
            ),
            "reference_answer": r"$\frac{1}{2}$",
            "reference_solution_text": (
                "因为2n是关于x的方程的根，所以代入并化简，"
                "由n不等于0约去n，得到m-n等于二分之一。"
            ),
        }
    )
    generator = LessonGenerationService(
        _parameter_root_preparation_client(), MathEngine()
    )

    async def grounded_route(_problem, _on_stage):
        return preparation_route()

    generator._build_grounded_teaching_route = grounded_route
    database = tmp_path / "parameter-root.sqlite3"
    store = MemoryStore(database)
    job = store.create_job()
    asyncio.run(
        run_generation(
            job.job_id,
            source,
            store,
            generator,
            FakeAudioService(),
        )
    )
    completed = store.get_job(job.job_id)
    assert completed.status == "completed"
    lesson_id = completed.lesson_id
    assert lesson_id

    restarted = MemoryStore(database)
    lesson = restarted.get_lesson(lesson_id)
    record = restarted.get_generation_record(lesson_id)
    assert lesson is not None and record is not None
    assert [
        step.directory_label
        for step in record.prepared_lesson.teaching_progression.steps
    ] == [
        "第一步：理解方程的根",
        "第二步：代入方程",
        "第三步：展开并化简",
        "第四步：利用n不等于0约去n",
        "第五步：整理出m-n",
    ]
    authored_cues = [
        cue
        for beat in lesson.beats
        for cue in beat.sync_cues
        if cue.teaching_step_id is not None
    ]
    assert authored_cues
    assert any(
        "m-n" in (cue.display_text or "")
        and "m 减 n" in cue.spoken_text
        for cue in authored_cues
    )
    allowed = {
        "reveal_step_header",
        "scroll_to_step",
        "write",
        "complete_step",
        "open_supporting_explanation",
        "close_supporting_explanation",
    }
    assert all(
        action.type in allowed
        for cue in authored_cues
        for action in (
            *cue.lead_actions,
            *cue.start_actions,
            *cue.end_actions,
        )
        if action.teaching_step_id is not None
    )

    app = create_app(
        generator=generator,
        audio_service=FakeAudioService(),
        store=restarted,
    )
    with TestClient(app) as client:
        public = client.get(f"/api/lessons/{lesson_id}")
        assert public.status_code == 200
        public_payload = public.json()
        public_text = json.dumps(public_payload, ensure_ascii=False)
        for private_key in (
            "correct_option_id",
            "expected_answer",
            "error_code",
            "review",
            "teaching_progression",
            "simulation_report",
        ):
            assert f'"{private_key}"' not in public_text

        interactions = {
            beat.interaction.interaction_id: beat.interaction
            for beat in lesson.beats
            if beat.interaction is not None
            and beat.interaction.interaction_id != "near-transfer"
        }
        substitution = interactions["substitution-check"]
        square = interactions["square-check"]
        correct = client.post(
            "/api/interactions/evaluate",
            json={
                "lesson_id": lesson_id,
                "interaction_id": substitution.interaction_id,
                "answer": "sub-correct",
            },
        )
        wrong = client.post(
            "/api/interactions/evaluate",
            json={
                "lesson_id": lesson_id,
                "interaction_id": square.interaction_id,
                "answer": "square-wrong",
            },
        )
        assert correct.json()["classification"] == "correct"
        assert wrong.json()["classification"] == "incorrect"
        correct_support = correct.json()["support_cues"]
        wrong_support = wrong.json()["support_cues"]
        assert len(correct_support) == 1
        assert any("4n^2" in cue["display_text"] for cue in wrong_support)
        assert sum(len(cue["spoken_text"]) for cue in correct_support) < sum(
            len(cue["spoken_text"]) for cue in wrong_support
        )
        assert substitution.advance_after_response is True
        assert square.advance_after_response is True

    reloaded = MemoryStore(database).get_lesson(lesson_id)
    assert reloaded is not None
    assert all(
        cue.audio_url
        for beat in reloaded.beats
        for cue in beat.sync_cues
    )


def test_production_lifespan_constructs_services_and_closes_owned_clients():
    application = create_app(settings=Settings())

    with TestClient(application):
        services = application.state.services
        model_client = services.generator.client
        speech_client = services.audio_service.client
        assert model_client._client.is_closed is False
        assert speech_client.http.is_closed is False

    assert model_client._client.is_closed is True
    assert speech_client.http.is_closed is True


def test_production_lifespan_selects_and_closes_volcengine_client():
    application = create_app(
        settings=Settings(
            tts_provider="volcengine",
            volcengine_tts_api_key="voice-secret",
            volcengine_tts_resource_id="seed-tts-2.0",
            volcengine_tts_voice="teacher",
        )
    )

    with TestClient(application):
        speech_client = application.state.services.audio_service.client
        assert speech_client.__class__.__name__ == "VolcengineSpeechClient"
        assert speech_client.http.is_closed is False

    assert speech_client.http.is_closed is True


def test_static_and_audio_mounts_and_page_routes_are_registered():
    app = create_app(
        generator=FakeGenerator(),
        audio_service=FakeAudioService(),
    )
    paths = {route.path for route in app.routes}

    assert "/static" in paths
    assert "/audio" in paths
    assert "/" in paths
    assert "/lesson/{lesson_id}" in paths
