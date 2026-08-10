import sqlite3
from datetime import datetime, timezone

import pytest

from app.schemas import (
    Interaction,
    InteractionOption,
    ProblemInput,
    RuntimeBeat,
    RuntimeLesson,
    TransferItem,
)
from app.store import MemoryStore


def runtime_lesson() -> RuntimeLesson:
    return RuntimeLesson(
        lesson_id="lesson-persisted-1",
        problem=ProblemInput(
            problem_text="x²-6x+5=0",
            reference_answer="x=1 或 x=5",
            reference_solution_text=(
                "x²-6x+9=4，所以 (x-3)²=4，x=1 或 x=5。"
            ),
            required_method="complete_the_square",
            lesson_length="standard",
        ),
        title="用配方法解一元二次方程",
        learning_goal="理解配方法中同时加上同一个数的原因。",
        beats=[
            RuntimeBeat(
                beat_id="beat-check",
                purpose="检查配方后的结果",
                narration="请选出配方后的等式。",
                board_actions=[],
                layer="interaction",
                interaction=Interaction(
                    interaction_id="interaction-private-answer",
                    kind="choice",
                    prompt="x²-6x+5=0 配方后应该得到哪个等式？",
                    expected_answer="option-correct",
                    options=[
                        InteractionOption(
                            option_id="option-correct",
                            label="(x-3)²=4",
                            feedback="正确。",
                        ),
                        InteractionOption(
                            option_id="option-wrong",
                            label="(x-3)²=5",
                            feedback="再检查常数项。",
                        ),
                    ],
                ),
            )
        ],
        summary="配方时要保持等式平衡。",
        transfer_item=TransferItem(
            problem_text="x²-4x-5=0",
            expected_answer="x=-1 或 x=5",
            method_signal="先把常数项移到右边，再配方。",
        ),
        validation_report={
            "math_status": "verified",
            "independent_solutions": ["x=1", "x=5"],
        },
    )


def test_sqlite_store_restores_an_equivalent_runtime_lesson(tmp_path):
    database_path = tmp_path / "lessons.sqlite3"
    lesson = runtime_lesson()

    MemoryStore(database_path).save_lesson(lesson)

    restored = MemoryStore(str(database_path)).get_lesson(lesson.lesson_id)
    assert restored == lesson


def test_sqlite_store_restores_private_interaction_answer_after_restart(
    tmp_path,
):
    database_path = tmp_path / "lessons.sqlite3"
    lesson = runtime_lesson()
    MemoryStore(database_path).save_lesson(lesson)

    interaction = MemoryStore(database_path).get_interaction(
        lesson.lesson_id,
        "interaction-private-answer",
    )

    assert interaction is not None
    assert interaction.expected_answer == "option-correct"


def test_sqlite_store_returns_none_for_missing_or_unsafe_lesson_ids(tmp_path):
    database_path = tmp_path / "lessons.sqlite3"
    store = MemoryStore(database_path)

    assert store.get_lesson("missing") is None
    assert store.get_lesson("../lesson") is None
    assert store.get_lesson("lesson/child") is None
    assert store.get_lesson(" lesson") is None


@pytest.mark.parametrize("lesson_id", ["../lesson", "lesson/child", " lesson"])
def test_sqlite_store_rejects_unsafe_lesson_id_before_writing(
    tmp_path,
    lesson_id,
):
    database_path = tmp_path / "nested" / "lessons.sqlite3"
    store = MemoryStore(database_path)
    unsafe_lesson = runtime_lesson().model_copy(
        update={"lesson_id": lesson_id}
    )

    with pytest.raises(ValueError, match="^invalid lesson id$"):
        store.save_lesson(unsafe_lesson)

    assert not database_path.exists()
    assert store._lessons == {}


def test_sqlite_lookup_does_not_create_a_missing_database(tmp_path):
    database_path = tmp_path / "missing" / "lessons.sqlite3"

    assert MemoryStore(database_path).get_lesson("lesson-1") is None
    assert not database_path.exists()


def test_sqlite_store_rejects_duplicate_lesson_id_without_overwriting(
    tmp_path,
):
    database_path = tmp_path / "lessons.sqlite3"
    lesson = runtime_lesson()
    store = MemoryStore(database_path)
    store.save_lesson(lesson)
    replacement = lesson.model_copy(update={"summary": "不应被保存"})

    with pytest.raises(ValueError, match="^lesson id already exists$"):
        store.save_lesson(replacement)

    assert store.get_lesson(lesson.lesson_id) == lesson
    assert MemoryStore(database_path).get_lesson(lesson.lesson_id) == lesson


def test_sqlite_store_does_not_translate_other_integrity_errors(tmp_path):
    database_path = tmp_path / "lessons.sqlite3"
    store = MemoryStore(database_path)
    lesson = runtime_lesson()
    store.save_lesson(lesson)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_lesson_insert
            BEFORE INSERT ON lessons
            BEGIN
                SELECT RAISE(ABORT, 'forced integrity failure');
            END
            """
        )
    other_lesson = lesson.model_copy(update={"lesson_id": "lesson-other"})

    with pytest.raises(sqlite3.IntegrityError, match="forced integrity failure"):
        store.save_lesson(other_lesson)


def test_sqlite_store_does_not_trust_a_spoofed_duplicate_error_message(
    tmp_path,
):
    database_path = tmp_path / "lessons.sqlite3"
    store = MemoryStore(database_path)
    lesson = runtime_lesson()
    store.save_lesson(lesson)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TRIGGER spoof_duplicate_error
            BEFORE INSERT ON lessons
            BEGIN
                SELECT RAISE(
                    ABORT,
                    'UNIQUE constraint failed: lessons.lesson_id'
                );
            END
            """
        )
    other_lesson = lesson.model_copy(update={"lesson_id": "lesson-other"})

    with pytest.raises(
        sqlite3.IntegrityError,
        match=r"UNIQUE constraint failed: lessons\.lesson_id",
    ):
        store.save_lesson(other_lesson)


def test_sqlite_store_fails_closed_when_runtime_json_is_corrupt(tmp_path):
    database_path = tmp_path / "lessons.sqlite3"
    lesson = runtime_lesson()
    MemoryStore(database_path).save_lesson(lesson)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE lessons SET runtime_json = ? WHERE lesson_id = ?",
            ("{not valid json", lesson.lesson_id),
        )

    assert MemoryStore(database_path).get_lesson(lesson.lesson_id) is None


def test_sqlite_store_creates_parent_directories_and_expected_schema(tmp_path):
    database_path = tmp_path / "nested" / "state" / "lessons.sqlite3"
    lesson = runtime_lesson()

    MemoryStore(database_path).save_lesson(lesson)

    assert database_path.is_file()
    with sqlite3.connect(database_path) as connection:
        columns = [
            row[1]
            for row in connection.execute("PRAGMA table_info(lessons)")
        ]
        row = connection.execute(
            """
            SELECT
                problem_text,
                reference_answer,
                reference_solution_text,
                required_method,
                lesson_length,
                created_at
            FROM lessons
            WHERE lesson_id = ?
            """,
            (lesson.lesson_id,),
        ).fetchone()

    assert columns == [
        "lesson_id",
        "problem_text",
        "reference_answer",
        "reference_solution_text",
        "required_method",
        "lesson_length",
        "runtime_json",
        "created_at",
    ]
    assert row is not None
    assert row[:5] == (
        lesson.problem.problem_text,
        lesson.problem.reference_answer,
        lesson.problem.reference_solution_text,
        "complete_the_square",
        "standard",
    )
    created_at = datetime.fromisoformat(row[5])
    assert created_at.tzinfo is not None
    assert created_at.utcoffset() == timezone.utc.utcoffset(created_at)
