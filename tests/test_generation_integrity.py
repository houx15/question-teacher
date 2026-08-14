import asyncio
import pytest

from app.generation_integrity import validate_lesson_generation_pair
from app.preparation_models import GenerationRecord
from tests.test_api import BundleGenerator, problem_input
from tests.test_preparation_models import teaching_progression_payload


ARTIFACT_ORDER = (
    "solution_trace",
    "reasoning_trajectory",
    "teaching_progression",
    "interaction_plan",
    "teaching_script",
    "performance_score",
    "simulation_report",
)
ARTIFACT_ROLES = {
    "solution_trace": "reference_analyst",
    "reasoning_trajectory": "teaching_designer",
    "teaching_progression": "teaching_designer",
    "interaction_plan": "interaction_designer",
    "teaching_script": "script_teacher",
    "performance_score": "classroom_director",
    "simulation_report": "student_simulator",
}


def current_pair():
    bundle = asyncio.run(BundleGenerator().generate_bundle(problem_input()))
    record_payload = bundle.generation_record.model_dump(mode="python")
    prepared = record_payload["prepared_lesson"]
    prepared["teaching_progression"] = teaching_progression_payload()
    prepared["artifact_history"] = [
        {
            "artifact_type": artifact_type,
            "version": 1,
            "responsible_role": ARTIFACT_ROLES[artifact_type],
        }
        for artifact_type in ARTIFACT_ORDER
    ]
    record = GenerationRecord.model_validate(record_payload)
    report = dict(bundle.lesson.validation_report)
    report["artifact_versions"] = {
        artifact_type: 1 for artifact_type in ARTIFACT_ORDER
    }
    lesson = bundle.lesson.model_copy(
        update={"validation_report": report}
    )
    return lesson, record


def legacy_six_artifact_pair(rubric_version="0.1"):
    lesson, record = current_pair()
    payload = record.model_dump(mode="python")
    prepared = payload["prepared_lesson"]
    prepared["rubric_version"] = rubric_version
    prepared["teaching_progression"] = None
    revisions = {
        item["artifact_type"]: item
        for item in prepared["artifact_history"]
    }
    legacy_order = (
        "solution_trace",
        "reasoning_trajectory",
        "teaching_script",
        "interaction_plan",
        "performance_score",
        "simulation_report",
    )
    prepared["artifact_history"] = [
        revisions[artifact_type] for artifact_type in legacy_order
    ]
    report = dict(lesson.validation_report)
    report["pedagogy_rubric_version"] = rubric_version
    report["artifact_versions"] = {
        artifact_type: 1 for artifact_type in legacy_order
    }
    lesson = lesson.model_copy(update={"validation_report": report})
    return lesson, GenerationRecord.model_validate(payload)


def test_current_generation_record_accepts_exact_seven_artifact_initial_build():
    lesson, record = current_pair()

    validated_lesson, validated_record = validate_lesson_generation_pair(
        lesson, record
    )

    assert validated_lesson.lesson_id == lesson.lesson_id
    assert tuple(
        revision.artifact_type
        for revision in validated_record.prepared_lesson.artifact_history
    ) == ARTIFACT_ORDER


def test_historical_rubric_0_1_six_artifact_record_is_readable_only_privately():
    lesson, record = legacy_six_artifact_pair()

    validate_lesson_generation_pair(
        lesson, record, require_current_rubric=False
    )
    with pytest.raises(ValueError, match="teaching progression missing"):
        validate_lesson_generation_pair(
            lesson, record, require_current_rubric=True
        )


def test_unknown_historical_rubric_cannot_select_legacy_six_artifact_shape():
    lesson, record = legacy_six_artifact_pair("unknown-rubric")

    with pytest.raises(ValueError, match="artifact history invalid"):
        validate_lesson_generation_pair(
            lesson, record, require_current_rubric=False
        )


def test_current_record_accepts_simulation_only_repair_suffix():
    lesson, record = current_pair()
    payload = record.model_dump(mode="python")
    payload["prepared_lesson"]["repair_count"] = 1
    payload["prepared_lesson"]["artifact_history"].append(
        {
            "artifact_type": "simulation_report",
            "version": 2,
            "responsible_role": "student_simulator",
        }
    )
    report = dict(lesson.validation_report)
    report["repair_count"] = 1
    report["artifact_versions"] = {
        **report["artifact_versions"],
        "simulation_report": 2,
    }

    validate_lesson_generation_pair(
        lesson.model_copy(update={"validation_report": report}),
        GenerationRecord.model_validate(payload),
    )


def test_current_generation_record_rejects_wrong_progression_owner():
    lesson, record = current_pair()
    payload = record.model_dump(mode="python")
    payload["prepared_lesson"]["artifact_history"][2][
        "responsible_role"
    ] = "interaction_designer"

    with pytest.raises(ValueError, match="artifact history invalid"):
        validate_lesson_generation_pair(
            lesson, GenerationRecord.model_validate(payload)
        )


@pytest.mark.parametrize("require_current_rubric", [True, False])
def test_current_generation_record_requires_non_null_progression_even_with_seven_history(
    require_current_rubric,
):
    lesson, record = current_pair()
    payload = record.model_dump(mode="python")
    payload["prepared_lesson"]["teaching_progression"] = None

    with pytest.raises(ValueError, match="teaching progression missing"):
        validate_lesson_generation_pair(
            lesson,
            GenerationRecord.model_validate(payload),
            require_current_rubric=require_current_rubric,
        )


def test_current_generation_record_accepts_exact_repair_suffix_and_versions():
    lesson, record = current_pair()
    payload = record.model_dump(mode="python")
    payload["prepared_lesson"]["repair_count"] = 1
    repaired_types = ARTIFACT_ORDER[3:]
    payload["prepared_lesson"]["artifact_history"].extend(
        {
            "artifact_type": artifact_type,
            "version": 2,
            "responsible_role": ARTIFACT_ROLES[artifact_type],
        }
        for artifact_type in repaired_types
    )
    report = dict(lesson.validation_report)
    report["repair_count"] = 1
    report["artifact_versions"] = {
        artifact_type: (2 if artifact_type in repaired_types else 1)
        for artifact_type in ARTIFACT_ORDER
    }
    lesson = lesson.model_copy(update={"validation_report": report})

    validate_lesson_generation_pair(
        lesson, GenerationRecord.model_validate(payload)
    )


@pytest.mark.parametrize("mutation", ["missing_suffix_item", "wrong_version"])
def test_current_generation_record_rejects_inexact_repair_history(mutation):
    lesson, record = current_pair()
    payload = record.model_dump(mode="python")
    payload["prepared_lesson"]["repair_count"] = 1
    repaired = [
        {
            "artifact_type": artifact_type,
            "version": 2,
            "responsible_role": ARTIFACT_ROLES[artifact_type],
        }
        for artifact_type in ARTIFACT_ORDER[3:]
    ]
    if mutation == "missing_suffix_item":
        repaired.pop(1)
    else:
        repaired[1]["version"] = 3
    payload["prepared_lesson"]["artifact_history"].extend(repaired)
    report = dict(lesson.validation_report)
    report["repair_count"] = 1
    lesson = lesson.model_copy(update={"validation_report": report})

    with pytest.raises(ValueError, match="artifact history invalid"):
        validate_lesson_generation_pair(
            lesson, GenerationRecord.model_validate(payload)
        )


@pytest.mark.parametrize("require_current_rubric", [True, False])
def test_current_generation_record_rejects_legacy_six_artifact_initial_build(
    require_current_rubric,
):
    lesson, record = current_pair()
    payload = record.model_dump(mode="python")
    prepared = payload["prepared_lesson"]
    prepared["teaching_progression"] = None
    prepared["artifact_history"] = [
        revision
        for revision in prepared["artifact_history"]
        if revision["artifact_type"] != "teaching_progression"
    ]
    report = dict(lesson.validation_report)
    report["artifact_versions"] = {
        artifact_type: 1
        for artifact_type in ARTIFACT_ORDER
        if artifact_type != "teaching_progression"
    }
    lesson = lesson.model_copy(update={"validation_report": report})

    with pytest.raises(ValueError, match="teaching progression missing"):
        validate_lesson_generation_pair(
            lesson,
            GenerationRecord.model_validate(payload),
            require_current_rubric=require_current_rubric,
        )
