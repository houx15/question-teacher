import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Dict, Optional, Union
from uuid import uuid4

from pydantic import ValidationError

from app.lesson_ids import is_valid_lesson_id
from app.generation_integrity import validate_lesson_generation_pair
from app.preparation_models import GenerationRecord
from app.schemas import GenerationJob, Interaction, RuntimeLesson


class MemoryStore:
    def __init__(
        self,
        database_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self._jobs: Dict[str, GenerationJob] = {}
        self._lessons: Dict[str, RuntimeLesson] = {}
        self._generation_records: Dict[str, GenerationRecord] = {}
        self._generation_ids: Dict[str, str] = {}
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

    def save_lesson(
        self,
        lesson: RuntimeLesson,
        generation_record: Optional[GenerationRecord] = None,
    ) -> None:
        if not is_valid_lesson_id(lesson.lesson_id):
            raise ValueError("invalid lesson id")
        if generation_record is not None:
            lesson, generation_record = validate_lesson_generation_pair(
                lesson,
                generation_record,
            )

        with self._lock:
            if self._database_path is None:
                if lesson.lesson_id in self._lessons:
                    raise ValueError("lesson id already exists")
                if (
                    generation_record is not None
                    and generation_record.generation_id
                    in self._generation_ids
                ):
                    raise sqlite3.IntegrityError(
                        "UNIQUE constraint failed: "
                        "lesson_generation_records.generation_id"
                    )
            if self._database_path is not None:
                self._save_lesson_to_database(lesson, generation_record)
            self._lessons[lesson.lesson_id] = lesson
            if generation_record is not None:
                self._generation_records[lesson.lesson_id] = (
                    generation_record.model_copy(deep=True)
                )
                self._generation_ids[generation_record.generation_id] = (
                    lesson.lesson_id
                )

    def lesson_exists(self, lesson_id: str) -> bool:
        if not is_valid_lesson_id(lesson_id):
            return False
        with self._lock:
            if lesson_id in self._lessons:
                return True
            if (
                self._database_path is None
                or not self._database_path.is_file()
            ):
                return False
            connection = self._connect()
            try:
                table_exists = connection.execute(
                    """
                    SELECT 1
                    FROM sqlite_master
                    WHERE type = 'table' AND name = 'lessons'
                    """
                ).fetchone()
                if table_exists is None:
                    return False
                return connection.execute(
                    "SELECT 1 FROM lessons WHERE lesson_id = ?",
                    (lesson_id,),
                ).fetchone() is not None
            finally:
                connection.close()

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

    def get_generation_record(
        self,
        lesson_id: str,
    ) -> Optional[GenerationRecord]:
        """Return server-private preparation evidence for one lesson."""
        if not is_valid_lesson_id(lesson_id):
            return None

        with self._lock:
            cached = self._generation_records.get(lesson_id)
            if cached is not None:
                cached_lesson = self._lessons.get(lesson_id)
                if cached_lesson is None:
                    return None
                try:
                    _lesson, cached = validate_lesson_generation_pair(
                        cached_lesson,
                        cached,
                        require_current_rubric=False,
                    )
                except (ValidationError, ValueError):
                    return None
                return cached.model_copy(deep=True)
            if (
                self._database_path is None
                or not self._database_path.is_file()
            ):
                return None

            connection = self._connect()
            try:
                table_exists = connection.execute(
                    """
                    SELECT 1
                    FROM sqlite_master
                    WHERE type = 'table'
                      AND name = 'lesson_generation_records'
                    """
                ).fetchone()
                if table_exists is None:
                    return None
                row = connection.execute(
                    """
                    SELECT
                        generation_id,
                        rubric_version,
                        record_json,
                        created_at
                    FROM lesson_generation_records
                    WHERE lesson_id = ?
                    """,
                    (lesson_id,),
                ).fetchone()
            finally:
                connection.close()
            if row is None:
                return None
            generation_id, rubric_version, record_json, created_at = row
            try:
                record = GenerationRecord.model_validate_json(record_json)
            except ValidationError:
                return None
            if (
                record.lesson_id != lesson_id
                or record.generation_id != generation_id
                or record.prepared_lesson.rubric_version != rubric_version
                or record.created_at != created_at
            ):
                return None
            lesson = self.get_lesson(lesson_id)
            if lesson is None:
                return None
            try:
                _lesson, record = validate_lesson_generation_pair(
                    lesson,
                    record,
                    require_current_rubric=False,
                )
            except (ValidationError, ValueError):
                return None
            self._generation_records[lesson_id] = record.model_copy(deep=True)
            self._generation_ids[record.generation_id] = lesson_id
            return record.model_copy(deep=True)

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

    def _save_lesson_to_database(
        self,
        lesson: RuntimeLesson,
        generation_record: Optional[GenerationRecord],
    ) -> None:
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
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS lesson_generation_records (
                    lesson_id TEXT PRIMARY KEY,
                    generation_id TEXT NOT NULL UNIQUE,
                    rubric_version TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (lesson_id)
                        REFERENCES lessons(lesson_id) ON DELETE CASCADE
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
            if generation_record is not None:
                connection.execute(
                    """
                    INSERT INTO lesson_generation_records (
                        lesson_id,
                        generation_id,
                        rubric_version,
                        record_json,
                        created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        lesson.lesson_id,
                        generation_record.generation_id,
                        generation_record.prepared_lesson.rubric_version,
                        generation_record.model_dump_json(),
                        generation_record.created_at,
                    ),
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()
