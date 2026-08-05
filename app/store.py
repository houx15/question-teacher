from threading import RLock
from typing import Dict, Optional
from uuid import uuid4

from app.schemas import GenerationJob, RuntimeLesson


class MemoryStore:
    def __init__(self) -> None:
        self._jobs: Dict[str, GenerationJob] = {}
        self._lessons: Dict[str, RuntimeLesson] = {}
        self._lock = RLock()

    def create_job(self) -> GenerationJob:
        job = GenerationJob(
            job_id=str(uuid4()),
            status="queued",
            stage="等待生成",
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def update_job(self, job_id: str, **changes: object) -> GenerationJob:
        with self._lock:
            current = self._jobs[job_id]
            payload = current.model_dump()
            payload.update(changes)
            updated = GenerationJob.model_validate(payload)
            self._jobs[job_id] = updated
            return updated

    def get_job(self, job_id: str) -> Optional[GenerationJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def save_lesson(self, lesson: RuntimeLesson) -> None:
        with self._lock:
            self._lessons[lesson.lesson_id] = lesson

    def get_lesson(self, lesson_id: str) -> Optional[RuntimeLesson]:
        with self._lock:
            return self._lessons.get(lesson_id)
