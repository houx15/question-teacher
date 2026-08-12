import asyncio
from dataclasses import dataclass
import inspect
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, ConfigDict

from app.config import Settings
from app.audio_manifest import validate_lesson_audio_manifest
from app.generation import (
    GeneratedLessonBundle,
    LessonGenerationFailure,
    LessonInputError,
    LessonQualityError,
)
from app.generation_diagnostics import GenerationFailureCategory
from app.generation_integrity import (
    audio_neutral_lesson_json,
    validate_lesson_generation_pair,
)
from app.math_engine import MathEngine, MathValidationError
from app.preparation_models import GenerationRecord
from app.preparation_pipeline import PreparationFailure
from app.llm_client import ModelResponseError, ModelStructureError
from app.schemas import (
    GenerationJob,
    NonEmptyString,
    ProblemInput,
    RuntimeLesson,
)
from app.store import MemoryStore


_AUDIO_CLEANUP_TIMEOUT_SECONDS = 0.1


class InteractionSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lesson_id: NonEmptyString
    interaction_id: NonEmptyString
    answer: str


@dataclass
class ApiServices:
    settings: Settings
    store: MemoryStore
    math_engine: MathEngine
    generator: Any = None
    audio_service: Any = None


_PUBLIC_GENERATION_STAGES = {
    "正在验证数学路线": "正在整理参考解析",
    "正在规划数学路线": "正在整理参考解析",
    "正在审阅参考解析": "正在整理参考解析",
    "正在整理参考教学路线": "正在整理参考解析",
    "整理参考解析": "正在整理参考解析",
    "设计解题思维轨迹": "正在设计解题思维轨迹",
    "编写讲稿": "正在编写讲稿",
    "设计互动": "正在设计互动",
    "编排板书与高亮": "正在编排板书与高亮",
    "模拟学生并审核课程": "正在审核和优化课程",
    "正在设计完整讲解": "正在编写讲稿",
    "正在准备互动素材": "正在设计互动",
    "正在进行整篇审稿": "正在审核和优化课程",
    "正在修订完整讲解": "正在审核和优化课程",
    "正在编译课堂": "正在编译课堂",
    "正在生成讲解语音": "正在生成讲解语音",
    "正在保存课程": "正在保存课程",
}

_PUBLIC_GENERATION_STAGE_ORDER = (
    "正在理解题目",
    "正在整理参考解析",
    "正在设计解题思维轨迹",
    "正在编写讲稿",
    "正在设计互动",
    "正在编排板书与高亮",
    "正在审核和优化课程",
    "正在编译课堂",
    "正在生成讲解语音",
    "正在保存课程",
)
_PUBLIC_GENERATION_STAGE_ORDINALS = {
    stage: ordinal
    for ordinal, stage in enumerate(_PUBLIC_GENERATION_STAGE_ORDER)
}

_FAILURE_PHASE_ORDINALS = {
    "generation": 0,
    "input_validation": 1,
    "preparation": 2,
    "compile": 3,
    "persistence_preflight": 4,
    "tts": 5,
    "persistence": 6,
}


def safe_generation_error(
    error: Exception,
    *,
    trusted_input_validation: bool = False,
) -> str:
    if trusted_input_validation:
        public_message = LessonInputError.validated_public_message(error)
        if public_message is not None:
            return public_message
    return "课程生成失败，请稍后重试。"


_PREPARATION_FAILURE_CATEGORY_MAP = {
    "provider_error": "provider_error",
    "invalid_structure": "invalid_structure",
    "reference_trace_failed": "reference_trace_failed",
    "reference_content_leak": "reference_trace_failed",
    "reasoning_design_failed": "reasoning_design_failed",
    "teaching_script_failed": "reasoning_design_failed",
    "interaction_plan_failed": "reasoning_design_failed",
    "performance_score_failed": "reasoning_design_failed",
    "simulation_failed": "reasoning_design_failed",
    "prompt_payload_too_large": "invalid_structure",
    "review_not_converged": "review_not_converged",
}


def _private_failure_category(
    error: Exception,
    phase: str,
    stage_ordinal: int,
) -> Optional[GenerationFailureCategory]:
    if phase == "tts":
        return "tts_failed"
    if phase in {"persistence_preflight", "persistence"}:
        return "persistence_failed"
    if phase == "compile":
        if isinstance(
            error,
            (LessonQualityError, ModelResponseError),
        ):
            return "compile_failed"
        return None
    if phase == "input_validation":
        if isinstance(error, LessonInputError):
            if LessonInputError.validated_public_message(error) is not None:
                return None
            return "invalid_structure"
        if isinstance(error, ModelStructureError):
            return "invalid_structure"
        if isinstance(error, ModelResponseError):
            return "provider_error"
        return None
    if isinstance(error, PreparationFailure):
        mapped = _PREPARATION_FAILURE_CATEGORY_MAP.get(error.category)
        if mapped is not None:
            return mapped
        if error.role == "reference_analyst":
            return "reference_trace_failed"
        if error.role == "lesson_reviewer":
            return "review_not_converged"
        return "reasoning_design_failed"
    if isinstance(error, LessonGenerationFailure):
        return error.category
    if isinstance(error, ModelStructureError):
        return "invalid_structure"
    if isinstance(error, ModelResponseError):
        return "provider_error"
    if isinstance(error, LessonQualityError):
        trajectory_ordinal = _PUBLIC_GENERATION_STAGE_ORDINALS[
            "正在设计解题思维轨迹"
        ]
        if stage_ordinal < trajectory_ordinal:
            return "reference_trace_failed"
        return "reasoning_design_failed"
    return None


def public_lesson_payload(lesson: RuntimeLesson) -> dict:
    payload = lesson.model_dump()
    payload["problem"].pop("reference_answer", None)
    payload["problem"].pop("reference_solution_text", None)
    payload["transfer_item"].pop("expected_answer", None)
    payload["transfer_item"].pop("correct_option_id", None)
    for option in payload["transfer_item"].get("options", []):
        option.pop("canonical_answer", None)
        option.pop("feedback", None)
    payload.pop("validation_report", None)
    for beat in payload["beats"]:
        interaction = beat.get("interaction")
        if interaction is not None:
            interaction.pop("expected_answer", None)
            interaction.pop("explanation_after_correct", None)
            interaction.pop("correct_audio_url", None)
            for option in interaction.get("options", []):
                option.pop("feedback", None)
                option.pop("feedback_audio_url", None)
    return payload


def _defensive_model_payload(value: object) -> object:
    model_dump = getattr(value, "model_dump", None)
    if not callable(model_dump):
        return value
    return model_dump(mode="python")


async def _cleanup_failed_audio(audio_service: Any, lesson_id: str) -> None:
    cleanup = getattr(audio_service, "cleanup_lesson_audio", None)
    if not callable(cleanup):
        return
    try:
        result = cleanup(lesson_id)
        if inspect.isawaitable(result):
            await asyncio.wait_for(
                result,
                timeout=_AUDIO_CLEANUP_TIMEOUT_SECONDS,
            )
    except asyncio.CancelledError:
        raise
    except Exception:
        return


async def run_generation(
    job_id: str,
    problem: ProblemInput,
    store: MemoryStore,
    generator: Any,
    audio_service: Any,
) -> None:
    store.update_job(
        job_id,
        status="running",
        stage="正在理解题目",
    )
    current_stage_ordinal = _PUBLIC_GENERATION_STAGE_ORDINALS[
        "正在理解题目"
    ]
    failure_phase = "generation"
    failure_phase_ordinal = _FAILURE_PHASE_ORDINALS[failure_phase]

    def advance_failure_phase(phase: str) -> None:
        nonlocal failure_phase, failure_phase_ordinal
        phase_ordinal = _FAILURE_PHASE_ORDINALS[phase]
        if phase_ordinal <= failure_phase_ordinal:
            return
        failure_phase = phase
        failure_phase_ordinal = phase_ordinal

    def report_stage(stage: str) -> None:
        nonlocal current_stage_ordinal
        if stage in {
            "正在验证数学路线",
            "正在审阅参考解析",
        }:
            advance_failure_phase("input_validation")
        elif stage == "正在编译课堂":
            advance_failure_phase("compile")
        elif stage in _PUBLIC_GENERATION_STAGES:
            advance_failure_phase("preparation")
        public_stage = _PUBLIC_GENERATION_STAGES.get(stage)
        if public_stage is None:
            return
        stage_ordinal = _PUBLIC_GENERATION_STAGE_ORDINALS[public_stage]
        if stage_ordinal <= current_stage_ordinal:
            return
        store.update_job(job_id, stage=public_stage)
        current_stage_ordinal = stage_ordinal

    lesson_id = None
    audio_attached = False
    lesson_saved = False
    try:
        generate_bundle = getattr(generator, "generate_bundle", None)
        if callable(generate_bundle):
            untrusted_bundle = await generate_bundle(
                problem,
                on_stage=report_stage,
            )
            raw_lesson = getattr(untrusted_bundle, "lesson")
            raw_record = getattr(untrusted_bundle, "generation_record")
            bundle = GeneratedLessonBundle.model_validate(
                {
                    "lesson": _defensive_model_payload(raw_lesson),
                    "generation_record": _defensive_model_payload(
                        raw_record
                    ),
                }
            )
            lesson, generation_record = validate_lesson_generation_pair(
                bundle.lesson,
                bundle.generation_record,
            )
            record_snapshot = generation_record.model_dump_json()
        else:
            raw_lesson = await generator.generate(
                problem,
                on_stage=report_stage,
            )
            lesson = RuntimeLesson.model_validate(
                _defensive_model_payload(raw_lesson)
            )
            generation_record = None
            raw_record = None
            record_snapshot = None
        advance_failure_phase("compile")
        raw_lesson_snapshot = RuntimeLesson.model_validate(
            _defensive_model_payload(raw_lesson)
        ).model_dump_json()
        lesson_id = lesson.lesson_id
        advance_failure_phase("persistence_preflight")
        if store.lesson_exists(lesson_id):
            raise ValueError("lesson id already exists")
        advance_failure_phase("tts")
        lesson_snapshot = audio_neutral_lesson_json(lesson)
        voiced_lesson = await audio_service.attach_audio(
            lesson.model_copy(deep=True),
            on_stage=report_stage,
        )
        audio_attached = True
        lesson = RuntimeLesson.model_validate(
            _defensive_model_payload(voiced_lesson)
        )
        validate_lesson_audio_manifest(lesson)
        if audio_neutral_lesson_json(lesson) != lesson_snapshot:
            raise ValueError("audio service changed lesson semantics")
        current_raw_lesson = RuntimeLesson.model_validate(
            _defensive_model_payload(raw_lesson)
        )
        if current_raw_lesson.model_dump_json() != raw_lesson_snapshot:
            raise ValueError("audio service changed generated lesson")
        if generation_record is not None:
            current_raw_record = GenerationRecord.model_validate(
                _defensive_model_payload(raw_record)
            )
            if current_raw_record.model_dump_json() != record_snapshot:
                raise ValueError("audio service changed generation record")
            generation_record = GenerationRecord.model_validate_json(
                record_snapshot
            )
            lesson, generation_record = validate_lesson_generation_pair(
                lesson,
                generation_record,
            )
        # Expose the lesson ID only after its durable save succeeds.
        advance_failure_phase("persistence")
        report_stage("正在保存课程")
        store.save_lesson(lesson, generation_record=generation_record)
        lesson_saved = True
        store.update_job(
            job_id,
            status="completed",
            stage="已完成",
            lesson_id=lesson.lesson_id,
        )
    except BaseException as exc:
        if audio_attached and not lesson_saved and lesson_id is not None:
            await _cleanup_failed_audio(audio_service, lesson_id)
        if isinstance(exc, asyncio.CancelledError):
            raise
        if not isinstance(exc, Exception):
            raise
        category = _private_failure_category(
            exc,
            failure_phase,
            current_stage_ordinal,
        )
        if category is not None:
            store.record_job_diagnostic(job_id, category)
        store.update_job(
            job_id,
            status="failed",
            stage="生成失败",
            error=safe_generation_error(
                exc,
                trusted_input_validation=(
                    failure_phase == "input_validation"
                ),
            ),
        )


def create_api_router(services: ApiServices) -> APIRouter:
    router = APIRouter(prefix="/api")

    @router.get("/health")
    async def health() -> dict:
        return {
            "status": "ok",
            "model_configured": services.settings.model_configured,
            "voice_configured": services.settings.voice_configured,
        }

    @router.post("/lessons/generate", status_code=202)
    async def generate_lesson(
        problem: ProblemInput,
        background_tasks: BackgroundTasks,
    ) -> dict:
        job = services.store.create_job()
        background_tasks.add_task(
            run_generation,
            job.job_id,
            problem,
            services.store,
            services.generator,
            services.audio_service,
        )
        return {"job_id": job.job_id}

    @router.get("/jobs/{job_id}", response_model=GenerationJob)
    async def get_job(job_id: str) -> GenerationJob:
        job = services.store.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="生成任务不存在。")
        return job

    @router.get("/lessons/{lesson_id}")
    async def get_lesson(lesson_id: str) -> dict:
        lesson = services.store.get_lesson(lesson_id)
        if lesson is None:
            raise HTTPException(status_code=404, detail="课程不存在。")
        return public_lesson_payload(lesson)

    @router.post("/interactions/evaluate")
    async def evaluate_interaction(
        submission: InteractionSubmission,
    ) -> dict:
        interaction = services.store.get_interaction(
            submission.lesson_id,
            submission.interaction_id,
        )
        if interaction is None:
            raise HTTPException(
                status_code=404,
                detail="课程互动不存在。",
            )

        selected_option = None
        if interaction.kind == "choice":
            submitted_answer = submission.answer.strip()
            selected_option = next(
                (
                    option
                    for option in interaction.options
                    if option.option_id.strip() == submitted_answer
                ),
                None,
            )
            correct = (
                submitted_answer == interaction.expected_answer.strip()
            )
        elif interaction.kind == "point_select":
            correct = (
                submission.answer.strip()
                == interaction.expected_answer.strip()
            )
        elif interaction.kind == "free_text":
            return {
                "classification": "needs_review",
                "message": "首版暂不自动判错这类文字回答。",
            }
        else:
            try:
                if interaction.kind == "expression":
                    correct = services.math_engine.expressions_equivalent(
                        submission.answer,
                        interaction.expected_answer,
                    )
                else:
                    correct = services.math_engine.answers_equivalent(
                        submission.answer,
                        interaction.expected_answer,
                    )
            except MathValidationError:
                correct = False

        result = {
            "classification": "correct" if correct else "incorrect",
        }
        if selected_option is not None:
            if selected_option.feedback is not None:
                result["feedback"] = selected_option.feedback
            if selected_option.feedback_audio_url is not None:
                result["feedback_audio_url"] = (
                    selected_option.feedback_audio_url
                )
        elif correct and interaction.explanation_after_correct:
            result["feedback"] = interaction.explanation_after_correct
            if interaction.correct_audio_url is not None:
                result["feedback_audio_url"] = (
                    interaction.correct_audio_url
                )
        return result

    return router
