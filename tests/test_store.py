import sqlite3
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Barrier

import pytest
from pydantic import ValidationError

from app.schemas import (
    Interaction,
    InteractionOption,
    ProblemInput,
    RuntimeBeat,
    RuntimeLesson,
    RuntimeSyncCue,
    TransferItem,
)
from app.preparation_models import GenerationRecord
from app.pedagogy_rubric import PEDAGOGY_RUBRIC_VERSION
from app.problem_focus import compile_problem_focus_targets
from app.store import MemoryStore
from tests.test_preparation_models import (
    prepared_lesson,
    teaching_progression_payload,
)


def runtime_lesson(*, include_interaction: bool = False) -> RuntimeLesson:
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
                narration="我们把等式两边同时减一。" * 3,
                board_actions=[],
                layer="base",
                sync_cues=[
                    RuntimeSyncCue(
                        cue_id=f"runtime-authored-{index}",
                        teaching_step_id="teaching-step-001",
                        display_text="等式两边同时减一",
                        spoken_text="我们把等式两边同时减一。",
                    )
                    for index in range(1, 4)
                ],
                interaction=(
                    Interaction(
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
                    )
                    if include_interaction
                    else None
                ),
            )
        ],
        problem_focus_targets=compile_problem_focus_targets(
            "x²-6x+5=0"
        ),
        summary="我们把等式两边同时减一。",
        transfer_item=TransferItem(
            problem_text="x²-4x-5=0",
            expected_answer="x=-1 或 x=5",
            method_signal="先把常数项移到右边，再配方。",
        ),
        validation_report={
            "math_status": "verified",
            "independent_solutions": ["x=1", "x=5"],
            "teaching_route_fingerprint": "route-persisted-1",
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
    )


def private_generation_record(
    lesson: RuntimeLesson,
    *,
    generation_id: str = "generation-persisted-1",
) -> GenerationRecord:
    prepared = prepared_lesson()
    prepared["rubric_version"] = PEDAGOGY_RUBRIC_VERSION
    prepared["teaching_progression"] = teaching_progression_payload()
    prepared["teaching_script"]["title"] = lesson.title
    prepared["teaching_script"]["learning_goal"] = lesson.learning_goal
    for clause in prepared["teaching_script"]["clauses"]:
        clause["lesson_step_id"] = "teaching-step-001"
        clause["display_text"] = "等式两边同时减一"
    prepared["artifact_history"] = [
        {
            "artifact_type": artifact_type,
            "version": 1,
            "responsible_role": role,
        }
        for artifact_type, role in [
            ("solution_trace", "reference_analyst"),
            ("reasoning_trajectory", "teaching_designer"),
            ("teaching_progression", "teaching_designer"),
            ("interaction_plan", "interaction_designer"),
            ("teaching_script", "script_teacher"),
            ("performance_score", "classroom_director"),
            ("simulation_report", "student_simulator"),
        ]
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
            "generation_id": generation_id,
            "lesson_id": lesson.lesson_id,
            "route_fingerprint": "route-persisted-1",
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
                    "teaching_designer",
                    "interaction_designer",
                    "script_teacher",
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


def test_sqlite_store_roundtrips_private_generation_record_after_restart(
    tmp_path,
):
    database_path = tmp_path / "lessons.sqlite3"
    lesson = runtime_lesson()
    record = private_generation_record(lesson)

    MemoryStore(database_path).save_lesson(
        lesson,
        generation_record=record,
    )

    restarted = MemoryStore(database_path)
    assert restarted.get_lesson(lesson.lesson_id) == lesson
    assert restarted.get_generation_record(lesson.lesson_id) == record


def test_generation_record_table_has_exact_schema_and_cascade_fk(tmp_path):
    database_path = tmp_path / "lessons.sqlite3"
    lesson = runtime_lesson()
    record = private_generation_record(lesson)
    MemoryStore(database_path).save_lesson(lesson, record)

    with sqlite3.connect(database_path) as connection:
        columns = [
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(lesson_generation_records)"
            )
        ]
        foreign_keys = connection.execute(
            "PRAGMA foreign_key_list(lesson_generation_records)"
        ).fetchall()
        row = connection.execute(
            """
            SELECT lesson_id, generation_id, rubric_version, created_at
            FROM lesson_generation_records
            """
        ).fetchone()
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(
            "DELETE FROM lessons WHERE lesson_id = ?",
            (lesson.lesson_id,),
        )
        remaining = connection.execute(
            "SELECT COUNT(*) FROM lesson_generation_records"
        ).fetchone()[0]

    assert columns == [
        "lesson_id",
        "generation_id",
        "rubric_version",
        "record_json",
        "created_at",
    ]
    assert len(foreign_keys) == 1
    assert foreign_keys[0][2:] == (
        "lessons",
        "lesson_id",
        "lesson_id",
        "NO ACTION",
        "CASCADE",
        "NONE",
    )
    assert row == (
        lesson.lesson_id,
        record.generation_id,
        record.prepared_lesson.rubric_version,
        record.created_at,
    )
    assert remaining == 0


def test_generation_record_insert_failure_atomically_rolls_back_lesson(
    tmp_path,
):
    database_path = tmp_path / "lessons.sqlite3"
    store = MemoryStore(database_path)
    seed = runtime_lesson()
    store.save_lesson(seed)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TRIGGER reject_generation_record_insert
            BEFORE INSERT ON lesson_generation_records
            BEGIN
                SELECT RAISE(ABORT, 'forced private record failure');
            END
            """
        )
    lesson = seed.model_copy(update={"lesson_id": "lesson-atomic-new"})
    record = private_generation_record(
        lesson,
        generation_id="generation-atomic-new",
    )

    with pytest.raises(
        sqlite3.IntegrityError,
        match="forced private record failure",
    ):
        store.save_lesson(lesson, record)

    assert lesson.lesson_id not in store._lessons
    assert store.get_lesson(lesson.lesson_id) is None
    assert store.get_generation_record(lesson.lesson_id) is None


def test_duplicate_generation_id_rolls_back_second_lesson(tmp_path):
    database_path = tmp_path / "lessons.sqlite3"
    store = MemoryStore(database_path)
    first = runtime_lesson()
    first_record = private_generation_record(first)
    store.save_lesson(first, first_record)
    second = first.model_copy(update={"lesson_id": "lesson-second"})
    second_record = private_generation_record(
        second,
        generation_id=first_record.generation_id,
    )

    with pytest.raises(sqlite3.IntegrityError):
        store.save_lesson(second, second_record)

    assert store.get_lesson(second.lesson_id) is None
    assert store.get_generation_record(second.lesson_id) is None


def test_duplicate_lesson_id_with_record_preserves_exact_value_error(tmp_path):
    database_path = tmp_path / "lessons.sqlite3"
    lesson = runtime_lesson()
    store = MemoryStore(database_path)
    store.save_lesson(lesson, private_generation_record(lesson))

    with pytest.raises(ValueError, match="^lesson id already exists$"):
        store.save_lesson(
            lesson,
            private_generation_record(
                lesson,
                generation_id="generation-replacement",
            ),
        )


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("generation_id", "generation-forged"),
        ("rubric_version", "rubric-forged"),
        ("created_at", "2020-01-01T00:00:00+00:00"),
    ],
)
def test_sqlite_store_fails_closed_for_mismatched_record_columns(
    tmp_path,
    column,
    value,
):
    database_path = tmp_path / "lessons.sqlite3"
    lesson = runtime_lesson()
    record = private_generation_record(lesson)
    MemoryStore(database_path).save_lesson(lesson, record)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            f"UPDATE lesson_generation_records SET {column} = ?",
            (value,),
        )

    assert MemoryStore(database_path).get_generation_record(
        lesson.lesson_id
    ) is None


def test_sqlite_store_fails_closed_for_corrupt_private_record_json(tmp_path):
    database_path = tmp_path / "lessons.sqlite3"
    lesson = runtime_lesson()
    MemoryStore(database_path).save_lesson(
        lesson,
        private_generation_record(lesson),
    )
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE lesson_generation_records SET record_json = ?",
            ("{private corrupt json",),
        )

    restarted = MemoryStore(database_path)
    assert restarted.get_generation_record(lesson.lesson_id) is None
    assert restarted.get_lesson(lesson.lesson_id) == lesson


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lesson_id", "lesson-forged"),
        ("generation_id", "generation-forged"),
        ("created_at", "2020-01-01T00:00:00+00:00"),
    ],
)
def test_sqlite_store_fails_closed_for_mismatched_record_json(
    tmp_path,
    field,
    value,
):
    database_path = tmp_path / "lessons.sqlite3"
    lesson = runtime_lesson()
    record = private_generation_record(lesson)
    MemoryStore(database_path).save_lesson(lesson, record)
    payload = record.model_dump(mode="json")
    payload[field] = value
    forged_json = GenerationRecord.model_validate(payload).model_dump_json()
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE lesson_generation_records SET record_json = ?",
            (forged_json,),
        )

    assert MemoryStore(database_path).get_generation_record(
        lesson.lesson_id
    ) is None


def test_old_database_without_private_record_table_still_reads_lesson(
    tmp_path,
):
    database_path = tmp_path / "lessons.sqlite3"
    lesson = runtime_lesson()
    MemoryStore(database_path).save_lesson(lesson)
    with sqlite3.connect(database_path) as connection:
        connection.execute("DROP TABLE lesson_generation_records")

    restarted = MemoryStore(database_path)
    assert restarted.get_lesson(lesson.lesson_id) == lesson
    assert restarted.get_generation_record(lesson.lesson_id) is None


def test_store_rejects_generation_record_for_another_lesson_before_writing(
    tmp_path,
):
    database_path = tmp_path / "lessons.sqlite3"
    lesson = runtime_lesson()
    mismatched = private_generation_record(lesson).model_copy(
        update={"lesson_id": "lesson-other"}
    )
    store = MemoryStore(database_path)

    with pytest.raises(ValueError, match="generation record lesson id mismatch"):
        store.save_lesson(lesson, mismatched)

    assert not database_path.exists()
    assert store._lessons == {}


def test_store_rejects_record_rubric_that_disagrees_with_lesson(tmp_path):
    database_path = tmp_path / "lessons.sqlite3"
    lesson = runtime_lesson()
    report = dict(lesson.validation_report)
    report["pedagogy_rubric_version"] = "different-rubric"
    lesson = lesson.model_copy(update={"validation_report": report})
    store = MemoryStore(database_path)

    with pytest.raises(
        ValueError,
        match="generation record rubric version mismatch",
    ):
        store.save_lesson(lesson, private_generation_record(lesson))

    assert not database_path.exists()


def test_store_revalidates_generation_record_before_writing(tmp_path):
    database_path = tmp_path / "lessons.sqlite3"
    lesson = runtime_lesson()
    unvalidated = private_generation_record(lesson).model_copy(
        update={"generation_id": ""}
    )
    store = MemoryStore(database_path)

    with pytest.raises(ValidationError):
        store.save_lesson(lesson, unvalidated)

    assert not database_path.exists()


def test_concurrent_duplicate_generation_ids_roll_back_losing_lesson(
    tmp_path,
):
    database_path = tmp_path / "lessons.sqlite3"
    seed_store = MemoryStore(database_path)
    seed_store.save_lesson(runtime_lesson())
    with sqlite3.connect(database_path) as connection:
        connection.execute("DELETE FROM lessons")

    lessons = [
        runtime_lesson().model_copy(update={"lesson_id": "lesson-record-a"}),
        runtime_lesson().model_copy(update={"lesson_id": "lesson-record-b"}),
    ]
    stores = [MemoryStore(database_path), MemoryStore(database_path)]
    start = Barrier(2, timeout=2)

    def save(store, lesson):
        start.wait()
        try:
            store.save_lesson(
                lesson,
                private_generation_record(
                    lesson,
                    generation_id="generation-shared",
                ),
            )
        except sqlite3.IntegrityError:
            return "rejected"
        return "saved"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(
            executor.map(save, stores, lessons, timeout=10)
        )

    assert sorted(results) == ["rejected", "saved"]
    with sqlite3.connect(database_path) as connection:
        lesson_ids = {
            row[0]
            for row in connection.execute("SELECT lesson_id FROM lessons")
        }
        record_rows = connection.execute(
            "SELECT lesson_id, generation_id FROM lesson_generation_records"
        ).fetchall()
    assert len(lesson_ids) == 1
    assert record_rows == [(next(iter(lesson_ids)), "generation-shared")]


def test_memory_store_rejects_duplicate_lesson_id_without_overwrite():
    store = MemoryStore()
    lesson = runtime_lesson()
    store.save_lesson(lesson)
    replacement = lesson.model_copy(update={"summary": "forged"})

    with pytest.raises(ValueError, match="^lesson id already exists$"):
        store.save_lesson(replacement)

    assert store.get_lesson(lesson.lesson_id) == lesson


def test_memory_store_enforces_unique_generation_id_across_lessons():
    store = MemoryStore()
    first = runtime_lesson()
    store.save_lesson(first, private_generation_record(first))
    second = first.model_copy(update={"lesson_id": "lesson-memory-second"})
    duplicate_record = private_generation_record(second)

    with pytest.raises(sqlite3.IntegrityError):
        store.save_lesson(second, duplicate_record)

    assert store.get_lesson(second.lesson_id) is None
    assert store.get_generation_record(second.lesson_id) is None


def test_cached_private_record_fails_closed_if_cached_lesson_pair_mutates():
    store = MemoryStore()
    lesson = runtime_lesson()
    record = private_generation_record(lesson)
    store.save_lesson(lesson, record)
    cached_lesson = store.get_lesson(lesson.lesson_id)
    cached_lesson.validation_report["review_status"] = "revision_required"

    assert store.get_generation_record(lesson.lesson_id) is None


@pytest.mark.parametrize(
    ("report_update", "expected_message"),
    [
        (
            {"teaching_route_fingerprint": "forged-route"},
            "route fingerprint mismatch",
        ),
        (
            {"pedagogy_rubric_version": None},
            "pedagogy_rubric_version invalid",
        ),
        (
            {"review_status": "revision_required"},
            "review status mismatch",
        ),
        ({"repair_count": 7}, "repair count mismatch"),
        ({"artifact_versions": {}}, "artifact versions mismatch"),
    ],
)
def test_store_rejects_lesson_record_pair_integrity_mismatches(
    report_update,
    expected_message,
):
    lesson = runtime_lesson()
    report = dict(lesson.validation_report)
    report.update(report_update)
    forged = lesson.model_copy(update={"validation_report": report})
    store = MemoryStore()

    with pytest.raises(ValueError, match=expected_message):
        store.save_lesson(forged, private_generation_record(forged))

    assert store.get_lesson(forged.lesson_id) is None


def test_store_rejects_noncurrent_prepared_rubric_even_when_pair_agrees():
    lesson = runtime_lesson()
    report = dict(lesson.validation_report)
    report["pedagogy_rubric_version"] = "forged-rubric"
    lesson = lesson.model_copy(update={"validation_report": report})
    record = private_generation_record(lesson)
    forged_prepared = record.prepared_lesson.model_copy(
        update={"rubric_version": "forged-rubric"}
    )
    record = record.model_copy(update={"prepared_lesson": forged_prepared})

    with pytest.raises(ValueError, match="prepared rubric version invalid"):
        MemoryStore().save_lesson(lesson, record)


@pytest.mark.parametrize("value", [False, True, 1.0])
def test_store_requires_exact_integer_report_repair_count(value):
    lesson = runtime_lesson()
    report = dict(lesson.validation_report)
    report["repair_count"] = value
    lesson = lesson.model_copy(update={"validation_report": report})

    with pytest.raises(ValueError, match="report repair count invalid"):
        MemoryStore().save_lesson(lesson, private_generation_record(lesson))


@pytest.mark.parametrize("value", [True, 1.0])
def test_store_requires_exact_positive_integer_artifact_versions(value):
    lesson = runtime_lesson()
    report = dict(lesson.validation_report)
    versions = dict(report["artifact_versions"])
    versions["solution_trace"] = value
    report["artifact_versions"] = versions
    lesson = lesson.model_copy(update={"validation_report": report})

    with pytest.raises(ValueError, match="report artifact versions invalid"):
        MemoryStore().save_lesson(lesson, private_generation_record(lesson))


@pytest.mark.parametrize("field", ("lesson_step_id", "display_text"))
def test_current_seven_artifact_record_requires_complete_provenance(field):
    lesson = runtime_lesson()
    record = private_generation_record(lesson)
    provenance = list(record.cue_provenance)
    provenance[0] = provenance[0].model_copy(update={field: None})
    forged = record.model_copy(update={"cue_provenance": provenance})

    with pytest.raises(ValueError, match="matches preparation"):
        MemoryStore().save_lesson(lesson, forged)


def test_historical_rubric_0_1_record_remains_readable_privately(
    tmp_path,
):
    database_path = tmp_path / "historical.sqlite3"
    lesson = runtime_lesson()
    historical_beat = lesson.beats[0].model_copy(
        update={
            "sync_cues": [
                cue.model_copy(
                    update={
                        "teaching_step_id": None,
                        "display_text": None,
                    }
                )
                for cue in lesson.beats[0].sync_cues
            ]
        }
    )
    lesson = lesson.model_copy(update={"beats": [historical_beat]})
    report = dict(lesson.validation_report)
    report["pedagogy_rubric_version"] = "0.1"
    report["artifact_versions"] = {
        artifact_type: version
        for artifact_type, version in report["artifact_versions"].items()
        if artifact_type != "teaching_progression"
    }
    lesson = lesson.model_copy(update={"validation_report": report})
    record_payload = private_generation_record(lesson).model_dump(
        mode="python"
    )
    prepared = record_payload["prepared_lesson"]
    prepared["rubric_version"] = "0.1"
    prepared["teaching_progression"] = None
    for item in record_payload["cue_provenance"]:
        item.pop("lesson_step_id", None)
        item.pop("display_text", None)
        item.pop("response_id", None)
    revisions = {
        revision["artifact_type"]: revision
        for revision in prepared["artifact_history"]
    }
    prepared["artifact_history"] = [
        revisions[artifact_type]
        for artifact_type in (
            "solution_trace",
            "reasoning_trajectory",
            "teaching_script",
            "interaction_plan",
            "performance_score",
            "simulation_report",
        )
    ]
    record = GenerationRecord.model_validate(record_payload)
    MemoryStore(database_path).save_lesson(lesson)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO lesson_generation_records (
                lesson_id, generation_id, rubric_version,
                record_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                lesson.lesson_id,
                record.generation_id,
                record.prepared_lesson.rubric_version,
                record.model_dump_json(),
                record.created_at,
            ),
        )

    restarted = MemoryStore(database_path)
    assert restarted.get_generation_record(lesson.lesson_id) == record
    new_lesson = lesson.model_copy(update={"lesson_id": "lesson-new-write"})
    old_record = record.model_copy(
        update={
            "generation_id": "generation-new-write",
            "lesson_id": new_lesson.lesson_id,
        }
    )
    with pytest.raises(ValueError, match="rubric version invalid"):
        restarted.save_lesson(new_lesson, old_record)


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
    lesson = runtime_lesson(include_interaction=True)
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


def test_lesson_exists_treats_precreated_empty_database_as_empty(tmp_path):
    database_path = tmp_path / "empty.sqlite3"
    sqlite3.connect(database_path).close()

    assert MemoryStore(database_path).lesson_exists("lesson-empty") is False


def test_lesson_exists_propagates_malformed_lessons_schema(tmp_path):
    database_path = tmp_path / "malformed.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE lessons (wrong_column TEXT)")

    with pytest.raises(sqlite3.OperationalError, match="lesson_id"):
        MemoryStore(database_path).lesson_exists("lesson-malformed")


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


def test_sqlite_store_fails_closed_when_runtime_lesson_id_does_not_match(
    tmp_path,
):
    database_path = tmp_path / "lessons.sqlite3"
    lesson = runtime_lesson()
    MemoryStore(database_path).save_lesson(lesson)
    mismatched = lesson.model_copy(update={"lesson_id": "lesson-other"})
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE lessons SET runtime_json = ? WHERE lesson_id = ?",
            (mismatched.model_dump_json(), lesson.lesson_id),
        )
    restarted = MemoryStore(database_path)

    assert restarted.get_lesson(lesson.lesson_id) is None

    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "UPDATE lessons SET runtime_json = ? WHERE lesson_id = ?",
            (lesson.model_dump_json(), lesson.lesson_id),
        )
    assert restarted.get_lesson(lesson.lesson_id) == lesson


def test_sqlite_stores_atomically_reject_concurrent_duplicate_ids(tmp_path):
    database_path = tmp_path / "lessons.sqlite3"
    lessons = [
        runtime_lesson().model_copy(update={"summary": "writer one"}),
        runtime_lesson().model_copy(update={"summary": "writer two"}),
    ]
    stores = [MemoryStore(database_path), MemoryStore(database_path)]
    start = Barrier(2, timeout=2)

    def save(store, lesson):
        start.wait()
        try:
            store.save_lesson(lesson)
        except ValueError as exc:
            return "duplicate", str(exc)
        return "saved", lesson.summary

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(save, store, lesson)
            for store, lesson in zip(stores, lessons)
        ]
        results = [future.result(timeout=10) for future in futures]

    assert [status for status, _value in results].count("saved") == 1
    assert [status for status, _value in results].count("duplicate") == 1
    assert next(
        value for status, value in results if status == "duplicate"
    ) == "lesson id already exists"
    saved_summary = next(
        value for status, value in results if status == "saved"
    )
    with sqlite3.connect(database_path) as connection:
        rows = connection.execute(
            "SELECT runtime_json FROM lessons WHERE lesson_id = ?",
            (lessons[0].lesson_id,),
        ).fetchall()

    assert len(rows) == 1
    restored = RuntimeLesson.model_validate_json(rows[0][0])
    assert restored.summary == saved_summary


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
