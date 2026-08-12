import json
from typing import Dict, Tuple

from app.pedagogy_rubric import PEDAGOGY_RUBRIC_VERSION
from app.preparation_models import GenerationRecord
from app.schemas import FIXED_RUNTIME_CUE_IDS, RuntimeLesson


_AUTHORITATIVE_ARTIFACT_ORDER = (
    "solution_trace",
    "reasoning_trajectory",
    "teaching_script",
    "interaction_plan",
    "performance_score",
    "simulation_report",
)
_AUTHORITATIVE_ARTIFACT_ROLES = {
    "solution_trace": "reference_analyst",
    "reasoning_trajectory": "teaching_designer",
    "teaching_script": "script_teacher",
    "interaction_plan": "interaction_designer",
    "performance_score": "classroom_director",
    "simulation_report": "student_simulator",
}


def _model_payload(value: object) -> object:
    model_dump = getattr(value, "model_dump", None)
    if not callable(model_dump):
        return value
    return model_dump(mode="python")


def _latest_artifact_versions(record: GenerationRecord) -> Dict[str, int]:
    history = record.prepared_lesson.artifact_history
    initial = history[: len(_AUTHORITATIVE_ARTIFACT_ORDER)]
    if (
        tuple(item.artifact_type for item in initial)
        != _AUTHORITATIVE_ARTIFACT_ORDER
        or any(
            item.version != 1
            or item.responsible_role
            != _AUTHORITATIVE_ARTIFACT_ROLES[item.artifact_type]
            for item in initial
        )
    ):
        raise ValueError("generation record artifact history invalid")
    latest = {
        artifact_type: 1
        for artifact_type in _AUTHORITATIVE_ARTIFACT_ORDER
    }
    cursor = len(_AUTHORITATIVE_ARTIFACT_ORDER)
    for _repair in range(record.prepared_lesson.repair_count):
        if cursor >= len(history):
            raise ValueError("generation record artifact history invalid")
        first_type = history[cursor].artifact_type
        try:
            start = _AUTHORITATIVE_ARTIFACT_ORDER.index(first_type)
        except ValueError:
            raise ValueError(
                "generation record artifact history invalid"
            ) from None
        if start == len(_AUTHORITATIVE_ARTIFACT_ORDER) - 1:
            raise ValueError("generation record artifact history invalid")
        expected_types = _AUTHORITATIVE_ARTIFACT_ORDER[start:]
        segment = history[cursor : cursor + len(expected_types)]
        if tuple(item.artifact_type for item in segment) != expected_types:
            raise ValueError("generation record artifact history invalid")
        for item, artifact_type in zip(segment, expected_types):
            if (
                item.version != latest[artifact_type] + 1
                or item.responsible_role
                != _AUTHORITATIVE_ARTIFACT_ROLES[artifact_type]
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
    if prepared.rubric_version != PEDAGOGY_RUBRIC_VERSION:
        raise ValueError("generation record prepared rubric version invalid")
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
    if report.get("repair_count") != prepared.repair_count:
        raise ValueError("generation record repair count mismatch")
    if report.get("artifact_versions") != _latest_artifact_versions(
        validated_record
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
