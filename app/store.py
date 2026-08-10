import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Dict, Optional, Union
from uuid import uuid4

from pydantic import ValidationError

from app.lesson_ids import is_valid_lesson_id
from app.schemas import GenerationJob, Interaction, RuntimeLesson


class MemoryStore:
    def __init__(
        self,
        database_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self._jobs: Dict[str, GenerationJob] = {}
        self._lessons: Dict[str, RuntimeLesson] = {}
        self._database_path = (
            Path(database_path) if database_path is not None else None
        )
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
        if not is_valid_lesson_id(lesson.lesson_id):
            raise ValueError("invalid lesson id")

        with self._lock:
            if self._database_path is not None:
                self._save_lesson_to_database(lesson)
            self._lessons[lesson.lesson_id] = lesson

    def get_lesson(self, lesson_id: str) -> Optional[RuntimeLesson]:
        if not is_valid_lesson_id(lesson_id):
            return None

        with self._lock:
            cached = self._lessons.get(lesson_id)
            if cached is not None:
                return cached
            if (
                self._database_path is None
                or not self._database_path.is_file()
            ):
                return None

            connection = self._connect()
            try:
                row = connection.execute(
                    "SELECT runtime_json FROM lessons WHERE lesson_id = ?",
                    (lesson_id,),
                ).fetchone()
            finally:
                connection.close()
            if row is None:
                return None
            try:
                lesson = RuntimeLesson.model_validate_json(row[0])
            except ValidationError:
                return None
            if lesson.lesson_id != lesson_id:
                return None
            self._lessons[lesson_id] = lesson
            return lesson

    def get_interaction(
        self,
        lesson_id: str,
        interaction_id: str,
    ) -> Optional[Interaction]:
        lesson = self.get_lesson(lesson_id)
        if lesson is None:
            return None
        for beat in lesson.beats:
            interaction = beat.interaction
            if (
                interaction is not None
                and interaction.interaction_id == interaction_id
            ):
                return interaction
        return None

    def _connect(self) -> sqlite3.Connection:
        if self._database_path is None:
            raise RuntimeError("database path is not configured")
        connection = sqlite3.connect(self._database_path, timeout=5.0)
        try:
            connection.execute("PRAGMA busy_timeout = 5000")
            connection.execute("PRAGMA foreign_keys = ON")
        except Exception:
            connection.close()
            raise
        return connection

    def _save_lesson_to_database(self, lesson: RuntimeLesson) -> None:
        if self._database_path is None:
            raise RuntimeError("database path is not configured")
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect()
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS lessons (
                    lesson_id TEXT PRIMARY KEY,
                    problem_text TEXT NOT NULL,
                    reference_answer TEXT NOT NULL,
                    reference_solution_text TEXT,
                    required_method TEXT,
                    lesson_length TEXT NOT NULL,
                    runtime_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.commit()
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT 1 FROM lessons WHERE lesson_id = ?",
                (lesson.lesson_id,),
            ).fetchone()
            if existing is not None:
                raise ValueError("lesson id already exists")

            required_method = getattr(
                lesson.problem.required_method,
                "value",
                lesson.problem.required_method,
            )
            connection.execute(
                """
                INSERT INTO lessons (
                    lesson_id,
                    problem_text,
                    reference_answer,
                    reference_solution_text,
                    required_method,
                    lesson_length,
                    runtime_json,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    lesson.lesson_id,
                    lesson.problem.problem_text,
                    lesson.problem.reference_answer,
                    lesson.problem.reference_solution_text,
                    required_method,
                    lesson.problem.lesson_length,
                    lesson.model_dump_json(),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
