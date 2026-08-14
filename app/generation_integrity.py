import json
from typing import Dict, Tuple

from app.pedagogy_rubric import PEDAGOGY_RUBRIC_VERSION
from app.preparation_models import GenerationRecord
from app.schemas import FIXED_RUNTIME_CUE_IDS, RuntimeLesson


_AUTHORITATIVE_ARTIFACT_ORDER = (
    "solution_trace",
    "reasoning_trajectory",
    "teaching_progression",
    "interaction_plan",
    "teaching_script",
    "performance_score",
    "simulation_report",
)
_AUTHORITATIVE_ARTIFACT_ROLES = {
    "solution_trace": "reference_analyst",
    "reasoning_trajectory": "teaching_designer",
    "teaching_progression": "teaching_designer",
    "interaction_plan": "interaction_designer",
    "teaching_script": "script_teacher",
    "performance_score": "classroom_director",
    "simulation_report": "student_simulator",
}
_LEGACY_ARTIFACT_ORDER = (
    "solution_trace",
    "reasoning_trajectory",
    "teaching_script",
    "interaction_plan",
    "performance_score",
    "simulation_report",
)
_LEGACY_ARTIFACT_ROLES = {
    artifact_type: _AUTHORITATIVE_ARTIFACT_ROLES[artifact_type]
    for artifact_type in _LEGACY_ARTIFACT_ORDER
}
_LEGACY_PRE_PROGRESSION_RUBRIC_VERSION = "0.1"


def _model_payload(value: object) -> object:
    model_dump = getattr(value, "model_dump", None)
    if not callable(model_dump):
        return value
    return model_dump(mode="python")


def _latest_artifact_versions(
    record: GenerationRecord,
    *,
    allow_legacy: bool = False,
) -> Dict[str, int]:
    historical_legacy = (
        allow_legacy
        and record.prepared_lesson.rubric_version
        == _LEGACY_PRE_PROGRESSION_RUBRIC_VERSION
        and record.prepared_lesson.teaching_progression is None
    )
    artifact_order = (
        _LEGACY_ARTIFACT_ORDER
        if historical_legacy
        else _AUTHORITATIVE_ARTIFACT_ORDER
    )
    artifact_roles = (
        _LEGACY_ARTIFACT_ROLES
        if historical_legacy
        else _AUTHORITATIVE_ARTIFACT_ROLES
    )
    history = record.prepared_lesson.artifact_history
    initial = history[: len(artifact_order)]
    if (
        tuple(item.artifact_type for item in initial)
        != artifact_order
        or any(
            item.version != 1
            or item.responsible_role
            != artifact_roles[item.artifact_type]
            for item in initial
        )
    ):
        raise ValueError("generation record artifact history invalid")
    latest = {
        artifact_type: 1
        for artifact_type in artifact_order
    }
    cursor = len(artifact_order)
    for _repair in range(record.prepared_lesson.repair_count):
        if cursor >= len(history):
            raise ValueError("generation record artifact history invalid")
        first_type = history[cursor].artifact_type
        try:
            start = artifact_order.index(first_type)
        except ValueError:
            raise ValueError(
                "generation record artifact history invalid"
            ) from None
        expected_types = artifact_order[start:]
        segment = history[cursor : cursor + len(expected_types)]
        if tuple(item.artifact_type for item in segment) != expected_types:
            raise ValueError("generation record artifact history invalid")
        for item, artifact_type in zip(segment, expected_types):
            if (
                item.version != latest[artifact_type] + 1
                or item.responsible_role
                != artifact_roles[artifact_type]
            ):
                raise ValueError("generation record artifact history invalid")
            latest[artifact_type] = item.version
        cursor += len(expected_types)
    if cursor != len(history):
        raise ValueError("generation record artifact history invalid")
    return latest


def validate_lesson_generation_pair(
    lesson: object,
    generation_record: object,
    *,
    require_current_rubric: bool = True,
) -> Tuple[RuntimeLesson, GenerationRecord]:
    """Defensively reconstruct and validate one runtime/private pair."""
    validated_lesson = RuntimeLesson.model_validate(_model_payload(lesson))
    validated_record = GenerationRecord.model_validate(
        _model_payload(generation_record)
    )
    if validated_record.lesson_id != validated_lesson.lesson_id:
        raise ValueError("generation record lesson id mismatch")

    prepared = validated_record.prepared_lesson
    report = validated_lesson.validation_report
    if (
        require_current_rubric
        and prepared.rubric_version != PEDAGOGY_RUBRIC_VERSION
    ):
        raise ValueError("generation record prepared rubric version invalid")
    if (
        prepared.rubric_version == PEDAGOGY_RUBRIC_VERSION
        and prepared.teaching_progression is None
        and (
            require_current_rubric
            or tuple(
                item.artifact_type
                for item in prepared.artifact_history[
                    : len(_LEGACY_ARTIFACT_ORDER)
                ]
            )
            != _LEGACY_ARTIFACT_ORDER
        )
    ):
        raise ValueError("generation record teaching progression missing")
    for field in (
        "teaching_route_fingerprint",
        "pedagogy_rubric_version",
        "review_status",
    ):
        if type(report.get(field)) is not str or not report[field].strip():
            raise ValueError(f"generation report {field} invalid")
    report_repair_count = report.get("repair_count")
    if type(report_repair_count) is not int or report_repair_count < 0:
        raise ValueError("generation report repair count invalid")
    report_artifact_versions = report.get("artifact_versions")
    if (
        type(report_artifact_versions) is not dict
        or any(type(key) is not str for key in report_artifact_versions)
        or any(
            type(value) is not int or value <= 0
            for value in report_artifact_versions.values()
        )
    ):
        raise ValueError("generation report artifact versions invalid")
    if (
        report.get("teaching_route_fingerprint")
        != validated_record.route_fingerprint
    ):
        raise ValueError("generation record route fingerprint mismatch")
    if report.get("pedagogy_rubric_version") != prepared.rubric_version:
        raise ValueError("generation record rubric version mismatch")
    if (
        report.get("review_status") != "approved"
        or report.get("review_status") != prepared.review.status
    ):
        raise ValueError("generation record review status mismatch")
    if report_repair_count != prepared.repair_count:
        raise ValueError("generation record repair count mismatch")
    if report_artifact_versions != _latest_artifact_versions(
        validated_record,
        allow_legacy=not require_current_rubric,
    ):
        raise ValueError("generation record artifact versions mismatch")

    runtime_cues = [
        cue for beat in validated_lesson.beats for cue in beat.sync_cues
    ]
    runtime_by_id = {cue.cue_id: cue for cue in runtime_cues}
    if len(runtime_by_id) != len(runtime_cues):
        raise ValueError("compiled runtime cue ids must be unique")
    authored_runtime_ids = set(runtime_by_id) - set(
        FIXED_RUNTIME_CUE_IDS.values()
    )
    provenance = validated_record.cue_provenance
    clauses = prepared.teaching_script.clauses
    if [item.clause_id for item in provenance] != [
        item.clause_id for item in clauses
    ]:
        raise ValueError("cue provenance clause order changed")
    clause_by_id = {item.clause_id: item for item in clauses}
    original_cue_by_clause = {
        clause_id: cue.cue_id
        for cue in prepared.performance_score.cues
        for clause_id in cue.clause_ids
    }
    if any(
        item.episode_id != clause_by_id[item.clause_id].episode_id
        or item.spoken_text != clause_by_id[item.clause_id].spoken_text
        or item.original_performance_cue_id
        != original_cue_by_clause.get(item.clause_id)
        for item in provenance
    ):
        raise ValueError("cue provenance no longer matches preparation")
    provenance_ids = {item.runtime_cue_id for item in provenance}
    if provenance_ids != authored_runtime_ids:
        raise ValueError("compiled authored cue ids changed")
    grouped = {cue_id: [] for cue_id in provenance_ids}
    for item in provenance:
        grouped[item.runtime_cue_id].append(item.spoken_text)
    if any(
        "".join(grouped[cue_id]) != runtime_by_id[cue_id].spoken_text
        for cue_id in provenance_ids
    ):
        raise ValueError("compiled authored cue text changed")
    return validated_lesson, validated_record


def audio_neutral_lesson_json(lesson: object) -> str:
    """Canonicalize a runtime lesson while ignoring only audio URL fields."""
    validated = RuntimeLesson.model_validate(_model_payload(lesson))
    payload = validated.model_dump(mode="json")
    for beat in payload["beats"]:
        beat["audio_url"] = None
        for cue in beat["sync_cues"]:
            cue["audio_url"] = None
        interaction = beat.get("interaction")
        if interaction is None:
            continue
        interaction["hint_audio_urls"] = []
        interaction["correct_audio_url"] = None
        for option in interaction["options"]:
            option["feedback_audio_url"] = None
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
