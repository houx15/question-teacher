import json
from typing import Dict, List, Tuple

from app.pedagogy_rubric import PEDAGOGY_RUBRIC_VERSION
from app.preparation_models import GenerationRecord
from app.problem_focus import compile_problem_focus_targets
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


def _is_historical_legacy(
    record: GenerationRecord,
    *,
    allow_legacy: bool,
) -> bool:
    prepared = record.prepared_lesson
    initial = prepared.artifact_history[: len(_LEGACY_ARTIFACT_ORDER)]
    return (
        allow_legacy
        and prepared.rubric_version
        == _LEGACY_PRE_PROGRESSION_RUBRIC_VERSION
        and prepared.teaching_progression is None
        and tuple(item.artifact_type for item in initial)
        == _LEGACY_ARTIFACT_ORDER
        and all(
            item.version == 1
            and item.responsible_role
            == _LEGACY_ARTIFACT_ROLES[item.artifact_type]
            for item in initial
        )
    )


def _latest_artifact_versions(
    record: GenerationRecord,
    *,
    allow_legacy: bool = False,
) -> Dict[str, int]:
    historical_legacy = _is_historical_legacy(
        record,
        allow_legacy=allow_legacy,
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


def _performance_cues_by_clause(prepared: object) -> Dict[str, object]:
    result = {}
    for cue in prepared.performance_score.cues:
        for clause_id in cue.clause_ids:
            if clause_id in result:
                raise ValueError(
                    "generation record performance cue membership changed"
                )
            result[clause_id] = cue
    return result


def _bound_actions(
    cue: object,
    phase: str,
    clause_ids: List[str],
) -> list:
    included = set(clause_ids)
    return [
        item.action
        for item in getattr(cue, phase)
        if item.clause_id in included
    ]


def _layer_by_clause(prepared: object) -> Dict[str, str]:
    transitions = {
        item.after_clause_id: item
        for item in prepared.performance_score.overlay_transitions
    }
    active_layer = "base"
    result = {}
    for clause in prepared.teaching_script.clauses:
        result[clause.clause_id] = active_layer
        transition = transitions.get(clause.clause_id)
        if transition is not None:
            active_layer = (
                transition.layer
                if transition.action == "enter"
                else "base"
            )
    return result


def _validate_current_runtime_semantics(
    lesson: RuntimeLesson,
    record: GenerationRecord,
) -> None:
    prepared = record.prepared_lesson
    if any(
        ability is not True
        for result in prepared.simulation_report.episode_results
        for ability in (
            result.can_align_display_and_spoken_math,
            result.can_recover_with_adaptive_support,
            result.can_locate_current_step,
        )
    ):
        raise ValueError("current simulation structured ability failed or missing")
    script = prepared.teaching_script
    main_clauses = list(script.clauses)
    response_pairs = [
        (response, clause)
        for response in script.response_scripts
        for clause in response.clauses
    ]
    provenance = record.cue_provenance
    expected_clause_ids = [
        *[clause.clause_id for clause in main_clauses],
        *[clause.clause_id for _response, clause in response_pairs],
    ]
    if [item.clause_id for item in provenance] != expected_clause_ids:
        raise ValueError("cue provenance clause order changed")

    performance_by_clause = _performance_cues_by_clause(prepared)
    response_by_clause = {
        clause.clause_id: response
        for response, clause in response_pairs
    }
    all_clauses = {
        clause.clause_id: clause
        for clause in [
            *main_clauses,
            *[clause for _response, clause in response_pairs],
        ]
    }
    if set(performance_by_clause) != set(all_clauses):
        raise ValueError("cue provenance no longer matches preparation")
    for item in provenance:
        clause = all_clauses[item.clause_id]
        response = response_by_clause.get(item.clause_id)
        if (
            item.lesson_step_id is None
            or item.display_text is None
            or item.episode_id != clause.episode_id
            or item.lesson_step_id != clause.lesson_step_id
            or item.display_text != clause.display_text
            or item.spoken_text != clause.spoken_text
            or item.original_performance_cue_id
            != performance_by_clause[item.clause_id].cue_id
            or item.response_id
            != (response.response_id if response is not None else None)
        ):
            raise ValueError("cue provenance no longer matches preparation")

    runtime_cues = [
        cue for beat in lesson.beats for cue in beat.sync_cues
    ]
    runtime_by_id = {cue.cue_id: cue for cue in runtime_cues}
    if len(runtime_by_id) != len(runtime_cues):
        raise ValueError("compiled runtime cue ids must be unique")
    authored_runtime_ids = set(runtime_by_id) - set(
        FIXED_RUNTIME_CUE_IDS.values()
    )
    main_provenance = [
        item for item in provenance if item.response_id is None
    ]
    if {
        item.runtime_cue_id for item in main_provenance
    } != authored_runtime_ids:
        raise ValueError("compiled authored cue ids changed")

    beat_by_cue_id = {
        cue.cue_id: beat
        for beat in lesson.beats
        for cue in beat.sync_cues
    }
    grouped = {}
    for item in main_provenance:
        grouped.setdefault(item.runtime_cue_id, []).append(item)
    layers = _layer_by_clause(prepared)
    for cue_id, items in grouped.items():
        runtime_cue = runtime_by_id[cue_id]
        clause_ids = [item.clause_id for item in items]
        original_ids = {
            item.original_performance_cue_id for item in items
        }
        expected_layers = {layers[item] for item in clause_ids}
        if len(original_ids) != 1 or len(expected_layers) != 1:
            raise ValueError("compiled authored cue grouping changed")
        performance_cue = performance_by_clause[clause_ids[0]]
        if (
            runtime_cue.teaching_step_id != items[0].lesson_step_id
            or runtime_cue.display_text
            != "".join(item.display_text or "" for item in items)
            or runtime_cue.spoken_text
            != "".join(item.spoken_text for item in items)
            or beat_by_cue_id[cue_id].layer
            != next(iter(expected_layers))
            or runtime_cue.lead_actions
            != _bound_actions(performance_cue, "lead_actions", clause_ids)
            or runtime_cue.start_actions
            != _bound_actions(performance_cue, "start_actions", clause_ids)
            or runtime_cue.end_actions
            != _bound_actions(performance_cue, "end_actions", clause_ids)
        ):
            raise ValueError("compiled authored cue semantics changed")

    runtime_interaction_pairs = [
        (beat, beat.interaction)
        for beat in lesson.beats
        if beat.interaction is not None
        and beat.interaction.interaction_id != "near-transfer"
    ]
    runtime_interactions = {
        interaction.interaction_id: (beat, interaction)
        for beat, interaction in runtime_interaction_pairs
    }
    if len(runtime_interactions) != len(runtime_interaction_pairs):
        raise ValueError("compiled interaction ids must be unique")
    if set(runtime_interactions) != {
        item.interaction_id
        for item in prepared.interaction_plan.interactions
    }:
        raise ValueError("compiled interaction binding changed")
    provenance_by_clause = {
        item.clause_id: item for item in provenance
    }
    authored_interactions = {
        item.interaction_id: item for item in script.interaction_scripts
    }
    for planned in prepared.interaction_plan.interactions:
        beat, runtime = runtime_interactions[planned.interaction_id]
        authored = authored_interactions.get(planned.interaction_id)
        if authored is None:
            raise ValueError("compiled interaction ownership changed")
        boundary_clause_id = planned.after_clause_id
        if planned.teaching_step_id is not None:
            matching = [
                clause
                for clause in main_clauses
                if clause.lesson_step_id == planned.teaching_step_id
            ]
            if matching:
                question = next(
                    (
                        clause
                        for clause in matching
                        if clause.pedagogical_function == "question"
                    ),
                    matching[0],
                )
                boundary_clause_id = question.clause_id
        if (
            boundary_clause_id is None
            or provenance_by_clause[boundary_clause_id].runtime_cue_id
            not in {cue.cue_id for cue in beat.sync_cues}
        ):
            raise ValueError("compiled interaction binding changed")
        expected_responses = {
            response.option_id: response
            for response in script.response_scripts
            if response.interaction_id == planned.interaction_id
        }
        if (
            runtime.kind != "choice"
            or runtime.prompt != authored.prompt
            or runtime.expected_answer != planned.correct_option_id
            or runtime.hints
            != ([authored.hint] if authored.hint is not None else [])
            or runtime.explanation_after_correct != ""
            or runtime.advance_after_response is not True
            or [option.option_id for option in runtime.options]
            != [option.option_id for option in planned.options]
            or [option.label for option in runtime.options]
            != [option.label for option in authored.options]
            or any(option.feedback is not None for option in runtime.options)
        ):
            raise ValueError("compiled interaction semantics changed")
        for option in runtime.options:
            response = expected_responses.get(option.option_id)
            if response is None or [
                cue.cue_id for cue in option.support_cues
            ] != [clause.clause_id for clause in response.clauses]:
                raise ValueError("compiled support cue binding changed")
            for support_cue, clause in zip(
                option.support_cues, response.clauses
            ):
                item = provenance_by_clause[clause.clause_id]
                performance_cue = performance_by_clause[clause.clause_id]
                if (
                    item.runtime_cue_id != support_cue.cue_id
                    or item.response_id != response.response_id
                    or support_cue.display_text != item.display_text
                    or support_cue.spoken_text != item.spoken_text
                    or support_cue.lead_actions
                    != _bound_actions(
                        performance_cue,
                        "lead_actions",
                        [clause.clause_id],
                    )
                    or support_cue.start_actions
                    != _bound_actions(
                        performance_cue,
                        "start_actions",
                        [clause.clause_id],
                    )
                    or support_cue.end_actions
                    != _bound_actions(
                        performance_cue,
                        "end_actions",
                        [clause.clause_id],
                    )
                ):
                    raise ValueError("compiled support cue semantics changed")

    transfer_script = script.transfer_script
    private_transfer = prepared.interaction_plan.transfer_item
    if transfer_script is None or (
        lesson.transfer_item.problem_text != transfer_script.problem_text
        or lesson.transfer_item.method_signal != transfer_script.method_signal
        or lesson.transfer_item.expected_answer != private_transfer.expected_answer
        or lesson.transfer_item.correct_option_id != private_transfer.correct_option_id
        or [item.option_id for item in lesson.transfer_item.options]
        != [item.option_id for item in transfer_script.options]
        or [item.label for item in lesson.transfer_item.options]
        != [item.label for item in transfer_script.options]
        or [item.feedback for item in lesson.transfer_item.options]
        != [item.feedback for item in transfer_script.options]
        or [item.canonical_answer for item in lesson.transfer_item.options]
        != [
            {item.option_id: item for item in private_transfer.options}[
                authored.option_id
            ].canonical_answer
            for authored in transfer_script.options
        ]
    ):
        raise ValueError("compiled transfer semantics changed")

    if lesson.problem_focus_targets != compile_problem_focus_targets(
        lesson.problem.problem_text
    ):
        raise ValueError("compiled problem focus targets changed")
    if (
        lesson.title != script.title
        or lesson.learning_goal != script.learning_goal
        or lesson.summary
        != "".join(
            clause.spoken_text
            for clause in main_clauses
            if clause.clause_id in script.closing_summary_clause_ids
        )
    ):
        raise ValueError("compiled lesson script semantics changed")

    legacy_action_types = {"write", "transform", "focus", "reveal"}
    for beat in lesson.beats:
        if beat.narration != "".join(
            cue.spoken_text for cue in beat.sync_cues
        ):
            raise ValueError("compiled beat narration changed")
        expected_board_actions = [
            action
            for cue in beat.sync_cues
            for action in cue.start_actions
            if action.surface == "board"
            and action.type in legacy_action_types
        ]
        actual_board_actions = [
            action.model_dump(mode="python")
            for action in beat.board_actions
        ]
        projected_actions = [
            {
                key: value
                for key, value in action.model_dump(mode="python").items()
                if key
                in {
                    "type",
                    "target",
                    "content",
                    "source",
                    "relation_target",
                    "annotation",
                }
            }
            for action in expected_board_actions
        ]
        if actual_board_actions != projected_actions:
            raise ValueError("compiled legacy board actions changed")


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
    historical_legacy = _is_historical_legacy(
        validated_record,
        allow_legacy=not require_current_rubric,
    )
    if prepared.teaching_progression is None and not historical_legacy:
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

    if not historical_legacy:
        _validate_current_runtime_semantics(
            validated_lesson,
            validated_record,
        )
        return validated_lesson, validated_record

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
            for support_cue in option.get("support_cues", []):
                support_cue["audio_url"] = None
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
