from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, ConfigDict

from app.config import Settings
from app.generation import LessonInputError
from app.math_engine import MathEngine, MathValidationError
from app.schemas import (
    GenerationJob,
    NonEmptyString,
    ProblemInput,
    RuntimeLesson,
)
from app.store import MemoryStore


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
    "正在验证数学路线": "正在验证数学路线",
    "正在审阅参考解析": "正在审阅参考解析",
    "正在设计完整讲解": "正在设计完整讲解",
    "正在进行整篇审稿": "正在进行整篇审稿",
    "正在修订完整讲解": "正在修订并编译课堂",
    "正在编译课堂": "正在修订并编译课堂",
    "正在生成讲解语音": "正在生成讲解语音",
}


def safe_generation_error(error: Exception) -> str:
    if isinstance(error, LessonInputError):
        return error.public_message
    return "课程生成失败，请稍后重试。"


def public_lesson_payload(lesson: RuntimeLesson) -> dict:
    payload = lesson.model_dump()
    payload["problem"].pop("reference_answer", None)
    payload["problem"].pop("reference_solution_text", None)
    payload["transfer_item"].pop("expected_answer", None)
    payload.pop("validation_report", None)
    for beat in payload["beats"]:
        interaction = beat.get("interaction")
        if interaction is not None:
            interaction.pop("expected_answer", None)
    return payload


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
    current_stage = "正在理解题目"

    def report_stage(stage: str) -> None:
        nonlocal current_stage
        public_stage = _PUBLIC_GENERATION_STAGES.get(stage)
        if public_stage is None or public_stage == current_stage:
            return
        store.update_job(job_id, stage=public_stage)
        current_stage = public_stage

    try:
        lesson = await generator.generate(problem, on_stage=report_stage)
        lesson = await audio_service.attach_audio(
            lesson,
            on_stage=report_stage,
        )
        store.save_lesson(lesson)
        store.update_job(
            job_id,
            status="completed",
            stage="已完成",
            lesson_id=lesson.lesson_id,
        )
    except Exception as exc:
        store.update_job(
            job_id,
            status="failed",
            stage="生成失败",
            error=safe_generation_error(exc),
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

        if interaction.kind in {"choice", "point_select"}:
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
        return {
            "classification": "correct" if correct else "incorrect",
        }

    return router
