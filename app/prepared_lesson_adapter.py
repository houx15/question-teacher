"""Adapt an approved private lesson preparation to the runtime draft."""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from pydantic import ValidationError

from app.math_expression import render_typed_math_action
from app.preparation_models import (
    PerformanceCue,
    PlannedInteraction,
    PreparedLesson,
    TeachingScript,
)
from app.preparation_validation import (
    PreparationValidationError,
    validate_prepared_lesson,
)
from app.problem_focus import compile_problem_focus_targets
from app.schemas import (
    Interaction,
    InteractionOption,
    LessonDraft,
    LessonLayer,
    LessonMoment,
    MathStep,
    NarrativeSyncCue,
    ProblemInput,
    RESERVED_RUNTIME_CUE_IDS,
    SupportSyncCue,
    TransferItem,
    TransferOption,
)
from app.teaching_route import FrozenTeachingRoute, TeachingRouteMode


@dataclass(frozen=True)
class _AdaptedCue:
    episode_id: str
    layer: LessonLayer
    cue: NarrativeSyncCue
    interaction_bearing_episode: bool = False
    interaction: Optional[Interaction] = None


_GENERATED_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class PreparedLessonAdaptationError(ValueError):
    """An approved preparation could not be adapted to a runtime draft."""


@dataclass(frozen=True)
class CueProvenanceRecord:
    """Immutable private link from an authored clause to its runtime cue."""

    episode_id: str
    lesson_step_id: Optional[str]
    clause_id: str
    original_performance_cue_id: str
    runtime_cue_id: str
    display_text: Optional[str]
    spoken_text: str
    response_id: Optional[str]

    def __post_init__(self) -> None:
        for value in (
            self.episode_id,
            self.clause_id,
            self.original_performance_cue_id,
            self.runtime_cue_id,
        ):
            if (
                type(value) is not str
                or _GENERATED_ID_PATTERN.fullmatch(value) is None
            ):
                raise PreparedLessonAdaptationError(
                    "provenance ids must be generated ids"
                )
        for value in (self.lesson_step_id, self.response_id):
            if (
                value is not None
                and (
                    type(value) is not str
                    or _GENERATED_ID_PATTERN.fullmatch(value) is None
                )
            ):
                raise PreparedLessonAdaptationError(
                    "optional provenance ids must be generated ids"
                )
        if (
            self.display_text is not None
            and (
                type(self.display_text) is not str
                or not self.display_text.strip()
            )
        ):
            raise PreparedLessonAdaptationError(
                "provenance display text must be nonblank"
            )
        if type(self.spoken_text) is not str or not self.spoken_text.strip():
            raise PreparedLessonAdaptationError(
                "provenance spoken text must be nonblank"
            )


@dataclass(frozen=True, init=False)
class PreparedDraftRun:
    """Defensive draft plus request-private clause-to-cue provenance."""

    _draft_json: str
    _cue_provenance: Tuple[CueProvenanceRecord, ...]
    _expected_clause_ids: Tuple[str, ...]

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise TypeError("use from_prepared_lesson")

    @classmethod
    def from_prepared_lesson(
        cls,
        draft: LessonDraft,
        prepared: PreparedLesson,
        runtime_cue_by_clause: Dict[str, str],
    ) -> "PreparedDraftRun":
        if type(draft) is not LessonDraft:
            raise PreparedLessonAdaptationError(
                "draft must be an exact LessonDraft"
            )
        if type(prepared) is not PreparedLesson:
            raise PreparedLessonAdaptationError(
                "prepared must be an exact PreparedLesson"
            )
        if type(runtime_cue_by_clause) is not dict:
            raise PreparedLessonAdaptationError(
                "runtime cue assignment must be an exact built-in dict"
            )

        clauses = prepared.teaching_script.clauses
        response_clauses = [
            (response, clause)
            for response in prepared.teaching_script.response_scripts
            for clause in response.clauses
        ]
        expected_main_clause_ids = tuple(
            clause.clause_id for clause in clauses
        )
        expected_clause_ids = (
            *expected_main_clause_ids,
            *(clause.clause_id for _response, clause in response_clauses),
        )
        if any(
            type(clause_id) is not str
            or _GENERATED_ID_PATTERN.fullmatch(clause_id) is None
            for clause_id in expected_clause_ids
        ):
            raise PreparedLessonAdaptationError(
                "expected clause ids must be generated ids"
            )
        if len(expected_clause_ids) != len(set(expected_clause_ids)):
            raise PreparedLessonAdaptationError(
                "expected clause ids must be unique"
            )
        if any(
            type(clause_id) is not str
            or type(runtime_cue_id) is not str
            or _GENERATED_ID_PATTERN.fullmatch(clause_id) is None
            or _GENERATED_ID_PATTERN.fullmatch(runtime_cue_id) is None
            for clause_id, runtime_cue_id in runtime_cue_by_clause.items()
        ):
            raise PreparedLessonAdaptationError(
                "runtime cue assignment must contain generated ids"
            )
        if set(runtime_cue_by_clause) != set(expected_main_clause_ids):
            raise PreparedLessonAdaptationError(
                "runtime cue assignment must be a complete clause mapping"
            )

        clause_ids = set(expected_clause_ids)
        original_cue_by_clause = {}
        for cue in prepared.performance_score.cues:
            for clause_id in cue.clause_ids:
                if clause_id not in clause_ids:
                    continue
                if clause_id in original_cue_by_clause:
                    raise PreparedLessonAdaptationError(
                        "performance cue membership must be unique and exact"
                    )
                original_cue_by_clause[clause_id] = cue.cue_id
        if set(original_cue_by_clause) != clause_ids:
            raise PreparedLessonAdaptationError(
                "performance cue membership must be complete"
            )

        normalized_main = tuple(
            CueProvenanceRecord(
                episode_id=clause.episode_id,
                lesson_step_id=clause.lesson_step_id,
                clause_id=clause.clause_id,
                original_performance_cue_id=original_cue_by_clause[
                    clause.clause_id
                ],
                runtime_cue_id=runtime_cue_by_clause[clause.clause_id],
                display_text=clause.display_text,
                spoken_text=clause.spoken_text,
                response_id=None,
            )
            for clause in clauses
        )
        normalized_responses = tuple(
            CueProvenanceRecord(
                episode_id=clause.episode_id,
                lesson_step_id=clause.lesson_step_id,
                clause_id=clause.clause_id,
                original_performance_cue_id=original_cue_by_clause[
                    clause.clause_id
                ],
                runtime_cue_id=clause.clause_id,
                display_text=clause.display_text,
                spoken_text=clause.spoken_text,
                response_id=response.response_id,
            )
            for response, clause in response_clauses
        )
        normalized = normalized_main + normalized_responses

        runtime_cues = {
            cue.cue_id: cue
            for cues in (
                draft.opening_sync_cues or [],
                draft.method_introduction_sync_cues or [],
                [
                    cue
                    for moment in draft.moments
                    for cue in moment.sync_cues
                ],
                draft.summary_sync_cues or [],
            )
            for cue in cues
        }
        mapped_runtime_ids = {
            item.runtime_cue_id for item in normalized_main
        }
        if mapped_runtime_ids != set(runtime_cues):
            raise PreparedLessonAdaptationError(
                "provenance runtime cue ids must exactly match the draft"
            )
        grouped_text = {cue_id: [] for cue_id in runtime_cues}
        for item in normalized_main:
            grouped_text[item.runtime_cue_id].append(item.spoken_text)
        if any(
            "".join(grouped_text[cue_id]) != cue.spoken_text
            for cue_id, cue in runtime_cues.items()
        ):
            raise PreparedLessonAdaptationError(
                "grouped provenance text must equal runtime cue narration"
            )

        runtime_interactions = [
            *draft.fixed_section_interactions_after_cue.values(),
            *(
                moment.interaction
                for moment in draft.moments
                if moment.interaction is not None
            ),
        ]
        support_entries = [
            (interaction.interaction_id, option.option_id, cue)
            for interaction in runtime_interactions
            for option in interaction.options
            for cue in option.support_cues
        ]
        support_cues = {
            cue.cue_id: cue
            for _interaction_id, _option_id, cue in support_entries
        }
        support_bindings = {
            cue.cue_id: (interaction_id, option_id)
            for interaction_id, option_id, cue in support_entries
        }
        if len(support_cues) != len(support_entries):
            raise PreparedLessonAdaptationError(
                "support provenance runtime cue ids must be unique"
            )
        if {
            item.runtime_cue_id for item in normalized_responses
        } != set(support_cues):
            raise PreparedLessonAdaptationError(
                "support provenance must exactly match the draft"
            )
        expected_support_bindings = {
            clause.clause_id: (
                response.interaction_id,
                response.option_id,
            )
            for response, clause in response_clauses
        }
        if support_bindings != expected_support_bindings:
            raise PreparedLessonAdaptationError(
                "support provenance response binding must match the draft"
            )
        if any(
            support_cues[item.runtime_cue_id].display_text
            != item.display_text
            or support_cues[item.runtime_cue_id].spoken_text
            != item.spoken_text
            for item in normalized_responses
        ):
            raise PreparedLessonAdaptationError(
                "support provenance text must equal runtime support cues"
            )

        instance = object.__new__(cls)
        object.__setattr__(
            instance,
            "_draft_json",
            draft.model_dump_json(),
        )
        object.__setattr__(instance, "_cue_provenance", normalized)
        object.__setattr__(
            instance,
            "_expected_clause_ids",
            tuple(expected_clause_ids),
        )
        return instance

    @property
    def draft(self) -> LessonDraft:
        return LessonDraft.model_validate_json(self._draft_json)

    @property
    def cue_provenance(self) -> Tuple[CueProvenanceRecord, ...]:
        return self._cue_provenance

    @property
    def expected_clause_ids(self) -> Tuple[str, ...]:
        return self._expected_clause_ids


def _runtime_cue_id(
    original_id: str,
    score_index: int,
    part_index: int,
    reserved_ids: Set[str],
    allocated_ids: Set[str],
) -> str:
    if (
        part_index == 0
        and original_id not in RESERVED_RUNTIME_CUE_IDS
        and original_id not in allocated_ids
    ):
        allocated_ids.add(original_id)
        return original_id
    suffix = 0
    while True:
        candidate = "prepared-cue-%03d-%03d-%03d" % (
            score_index,
            part_index,
            suffix,
        )
        if (
            candidate not in reserved_ids
            and candidate not in allocated_ids
            and candidate not in RESERVED_RUNTIME_CUE_IDS
        ):
            allocated_ids.add(candidate)
            return candidate
        suffix += 1


def _narrative_cue_part(
    cue: PerformanceCue,
    script: TeachingScript,
    clause_ids: Tuple[str, ...],
    cue_id: str,
) -> NarrativeSyncCue:
    clauses = {item.clause_id: item for item in script.clauses}
    included = set(clause_ids)
    step_ids = {clauses[clause_id].lesson_step_id for clause_id in clause_ids}
    if len(step_ids) != 1 or None in step_ids:
        raise PreparedLessonAdaptationError(
            "runtime cue clauses must share one teaching step"
        )
    return NarrativeSyncCue(
        cue_id=cue_id,
        teaching_step_id=next(iter(step_ids)),
        display_text="".join(
            clauses[clause_id].display_text or ""
            for clause_id in clause_ids
        ),
        spoken_text="".join(
            clauses[clause_id].spoken_text for clause_id in clause_ids
        ),
        lead_actions=[
            item.action.model_copy(deep=True)
            for item in cue.lead_actions
            if item.clause_id in included
        ],
        start_actions=[
            item.action.model_copy(deep=True)
            for item in cue.start_actions
            if item.clause_id in included
        ],
        end_actions=[
            item.action.model_copy(deep=True)
            for item in cue.end_actions
            if item.clause_id in included
        ],
    )


def _runtime_interaction(
    planned: PlannedInteraction,
    script: TeachingScript,
    performance_cues_by_clause: dict,
) -> Interaction:
    authored = next(
        (
            item for item in script.interaction_scripts
            if item.interaction_id == planned.interaction_id
        ),
        None,
    )
    if authored is None:
        raise PreparedLessonAdaptationError(
            "current interaction has no authored student-visible script"
        )
    authored_options = {item.option_id: item for item in authored.options}
    responses = {
        response.option_id: response
        for response in script.response_scripts
        if response.interaction_id == planned.interaction_id
    }
    return Interaction(
        interaction_id=planned.interaction_id,
        kind="choice",
        prompt=authored.prompt,
        expected_answer=planned.correct_option_id,
        options=[
            InteractionOption(
                option_id=option.option_id,
                label=authored_options[option.option_id].label,
                support_cues=[
                    SupportSyncCue(
                        cue_id=clause.clause_id,
                        display_text=clause.display_text,
                        spoken_text=clause.spoken_text,
                        lead_actions=[
                            item.action.model_copy(deep=True)
                            for item in performance_cues_by_clause[
                                clause.clause_id
                            ].lead_actions
                            if item.clause_id == clause.clause_id
                        ],
                        start_actions=[
                            item.action.model_copy(deep=True)
                            for item in performance_cues_by_clause[
                                clause.clause_id
                            ].start_actions
                            if item.clause_id == clause.clause_id
                        ],
                        end_actions=[
                            item.action.model_copy(deep=True)
                            for item in performance_cues_by_clause[
                                clause.clause_id
                            ].end_actions
                            if item.clause_id == clause.clause_id
                        ],
                    )
                    for clause in responses[option.option_id].clauses
                ],
            )
            for option in planned.options
        ],
        hints=[authored.hint] if authored.hint is not None else [],
        explanation_after_correct="",
        advance_after_response=True,
    )


def _layer_by_clause(prepared: PreparedLesson) -> dict:
    transitions = {
        item.after_clause_id: item
        for item in prepared.performance_score.overlay_transitions
    }
    active_layer: LessonLayer = "base"
    result = {}
    for clause in prepared.teaching_script.clauses:
        result[clause.clause_id] = active_layer
        transition = transitions.get(clause.clause_id)
        if transition is None:
            continue
        active_layer = transition.layer if transition.action == "enter" else "base"
    return result


def _runtime_sections(prepared: PreparedLesson) -> tuple:
    script = prepared.teaching_script
    clauses = {item.clause_id: item for item in script.clauses}
    opening_ids = set(script.opening_clause_ids)
    method_ids = set(script.method_introduction_clause_ids)
    summary_ids = set(script.closing_summary_clause_ids)
    layer_by_clause = _layer_by_clause(prepared)
    section_cues = {"opening": [], "method": [], "body": [], "summary": []}
    interactions_after_clause = {}
    performance_cues_by_clause = {
        clause_id: cue
        for cue in prepared.performance_score.cues
        for clause_id in cue.clause_ids
    }
    for item in prepared.interaction_plan.interactions:
        boundary_clause_id = None
        if item.teaching_step_id is not None:
            matching_clauses = [
                clause
                for clause in script.clauses
                if clause.lesson_step_id == item.teaching_step_id
            ]
            question_clauses = [
                clause
                for clause in matching_clauses
                if clause.pedagogical_function == "question"
            ]
            if question_clauses:
                boundary_clause_id = question_clauses[0].clause_id
            elif matching_clauses:
                boundary_clause_id = matching_clauses[0].clause_id
        else:
            boundary_clause_id = item.after_clause_id
        if boundary_clause_id is None:
            raise PreparedLessonAdaptationError(
                "current interaction has no authored teaching-step boundary"
            )
        interactions_after_clause[boundary_clause_id] = _runtime_interaction(
            item,
            script,
            performance_cues_by_clause,
        )
    interaction_episode_ids = {
        item.episode_id for item in prepared.interaction_plan.interactions
    }
    reserved_ids = {
        cue.cue_id for cue in prepared.performance_score.cues
    }
    allocated_ids = set()
    fixed_interactions = {}
    fixed_layers = {}
    runtime_cue_by_clause = {}

    def clause_section(clause_id: str) -> str:
        if clause_id in opening_ids:
            return "opening"
        if clause_id in method_ids:
            return "method"
        if clause_id in summary_ids:
            return "summary"
        return "body"

    for score_index, score_cue in enumerate(prepared.performance_score.cues):
        if any(clause_id not in clauses for clause_id in score_cue.clause_ids):
            continue
        parts = []
        current = []
        current_key = None
        current_length = 0
        action_clause_ids = {
            item.clause_id
            for item in (
                *score_cue.lead_actions,
                *score_cue.start_actions,
                *score_cue.end_actions,
            )
        }
        for clause_id in score_cue.clause_ids:
            clause = clauses[clause_id]
            key = (
                clause_section(clause_id),
                clause.episode_id,
                clause.lesson_step_id,
                layer_by_clause[clause_id],
            )
            spoken_length = len(clause.spoken_text)
            display_length = len(clause.display_text or "")
            if current and (
                key != current_key
                or current_length + spoken_length > 90
                or sum(
                    len(clauses[item].display_text or "")
                    for item in current
                ) + display_length > 500
                or clause_id in action_clause_ids
            ):
                parts.append(tuple(current))
                current = []
                current_length = 0
            current.append(clause_id)
            current_key = key
            current_length += spoken_length
            if (
                clause_id in interactions_after_clause
                or clause_id in action_clause_ids
            ):
                parts.append(tuple(current))
                current = []
                current_key = None
                current_length = 0
        if current:
            parts.append(tuple(current))

        for part_index, clause_ids in enumerate(parts):
            first_clause = clauses[clause_ids[0]]
            section = clause_section(clause_ids[0])
            cue_id = _runtime_cue_id(
                score_cue.cue_id,
                score_index,
                part_index,
                reserved_ids,
                allocated_ids,
            )
            narrative_cue = _narrative_cue_part(
                score_cue,
                script,
                clause_ids,
                cue_id,
            )
            for clause_id in clause_ids:
                if clause_id in runtime_cue_by_clause:
                    raise PreparedLessonAdaptationError(
                        "runtime cue assignment contains a duplicate clause"
                    )
                runtime_cue_by_clause[clause_id] = cue_id
            interaction = interactions_after_clause.get(clause_ids[-1])
            if section == "body":
                section_cues["body"].append(
                    _AdaptedCue(
                        episode_id=first_clause.episode_id,
                        layer=layer_by_clause[clause_ids[0]],
                        cue=narrative_cue,
                        interaction_bearing_episode=(
                            first_clause.episode_id
                            in interaction_episode_ids
                        ),
                        interaction=interaction,
                    )
                )
            else:
                section_cues[section].append(narrative_cue)
                fixed_layers[cue_id] = layer_by_clause[clause_ids[0]]
                if interaction is not None:
                    fixed_interactions[cue_id] = interaction
    if any(not section_cues[name] for name in ("opening", "method", "summary")):
        raise PreparedLessonAdaptationError(
            "prepared script section has no performance cue"
        )
    if not section_cues["body"]:
        raise PreparedLessonAdaptationError(
            "prepared script requires explanatory body clauses"
        )
    return (
        section_cues["opening"],
        section_cues["method"],
        section_cues["body"],
        section_cues["summary"],
        fixed_interactions,
        fixed_layers,
        runtime_cue_by_clause,
    )


def _lesson_moments(prepared: PreparedLesson, body: list) -> List[LessonMoment]:
    episodes = {
        item.episode_id: item for item in prepared.reasoning_trajectory.episodes
    }
    trace_steps = {
        item.source_step_id: item
        for item in prepared.solution_trace.source_steps
    }
    moments = []
    group = []

    def flush(interaction: Optional[Interaction] = None) -> None:
        nonlocal group
        if not group:
            return
        episode_order = []
        for item in group:
            if item.episode_id not in episode_order:
                episode_order.append(item.episode_id)
        operation_labels = []
        for episode_id in episode_order:
            for source_step_id in episodes[episode_id].source_step_ids:
                trace_step = trace_steps.get(source_step_id)
                if trace_step is None:
                    raise PreparedLessonAdaptationError(
                        "trajectory source step is missing from solution trace"
                    )
                label = render_typed_math_action(
                    trace_step.operation_kind,
                    [],
                )
                if label not in operation_labels:
                    operation_labels.append(label)
        purpose = "；".join(operation_labels)
        moments.append(
            LessonMoment(
                purpose=purpose,
                sync_cues=[item.cue for item in group],
                layer=group[0].layer,
                interaction=interaction,
            )
        )
        group = []

    for adapted in body:
        if group and (
            adapted.layer != group[0].layer
            or len(group) == 5
            or adapted.interaction_bearing_episode
            != group[0].interaction_bearing_episode
            or (
                adapted.interaction_bearing_episode
                and adapted.episode_id != group[0].episode_id
            )
        ):
            flush()
        group.append(adapted)
        if adapted.interaction is not None:
            flush(adapted.interaction)
    flush()
    return moments


def _transfer_item(prepared: PreparedLesson) -> TransferItem:
    private = prepared.interaction_plan.transfer_item
    source = prepared.teaching_script.transfer_script
    if source is None:
        raise PreparedLessonAdaptationError(
            "current transfer item has no authored student-visible script"
        )
    private_options = {item.option_id: item for item in private.options}
    return TransferItem(
        problem_text=source.problem_text,
        expected_answer=private.expected_answer,
        method_signal=source.method_signal,
        options=[
            TransferOption(
                option_id=option.option_id,
                label=option.label,
                canonical_answer=private_options[
                    option.option_id
                ].canonical_answer,
                feedback=option.feedback,
            )
            for option in source.options
        ],
        correct_option_id=private.correct_option_id,
    )


def _prepared_lesson_to_draft_with_provenance(
    problem: ProblemInput,
    prepared: PreparedLesson,
    teaching_route: FrozenTeachingRoute,
    verified_math_steps: Optional[List[MathStep]] = None,
) -> PreparedDraftRun:
    if type(problem) is not ProblemInput:
        raise PreparedLessonAdaptationError(
            "problem must be an exact ProblemInput"
        )
    if type(prepared) is not PreparedLesson:
        raise PreparedLessonAdaptationError(
            "prepared must be an exact PreparedLesson"
        )
    if type(teaching_route) is not FrozenTeachingRoute:
        raise PreparedLessonAdaptationError(
            "teaching_route must be an exact FrozenTeachingRoute"
        )
    if prepared.review.status != "approved":
        raise PreparedLessonAdaptationError(
            "prepared lesson must be approved before adaptation"
        )

    route_payload = teaching_route.to_prompt_payload()
    if teaching_route.mode is TeachingRouteMode.SYMBOLIC_VERIFIED:
        if not verified_math_steps:
            raise PreparedLessonAdaptationError(
                "symbolic route requires non-empty math steps"
            )
        math_steps = [item.model_copy(deep=True) for item in verified_math_steps]
    else:
        if verified_math_steps is not None:
            raise PreparedLessonAdaptationError(
                "grounded route rejects verified math steps"
            )
        math_steps = []

    problem_targets = compile_problem_focus_targets(problem.problem_text)
    validate_prepared_lesson(prepared, teaching_route, problem_targets)
    (
        opening,
        method,
        body,
        summary,
        fixed_interactions,
        fixed_layers,
        runtime_cue_by_clause,
    ) = _runtime_sections(prepared)
    route_payload["teaching_route_fingerprint"] = teaching_route.fingerprint
    script = prepared.teaching_script
    draft = LessonDraft(
        title=script.title,
        learning_goal=script.learning_goal,
        opening="".join(cue.spoken_text for cue in opening),
        method_rationale=script.method_rationale,
        method_introduction=script.method_introduction.model_copy(deep=True),
        opening_sync_cues=opening,
        method_introduction_sync_cues=method,
        summary_sync_cues=summary,
        fixed_section_interactions_after_cue=fixed_interactions,
        fixed_section_layers_by_cue=fixed_layers,
        transfer_feedback_is_authoritative=True,
        math_steps=math_steps,
        teaching_route=route_payload,
        moments=_lesson_moments(prepared, body),
        summary="".join(cue.spoken_text for cue in summary),
        transfer_item=_transfer_item(prepared),
    )
    return PreparedDraftRun.from_prepared_lesson(
        draft,
        prepared,
        runtime_cue_by_clause,
    )


def prepared_lesson_to_draft_with_provenance(
    problem: ProblemInput,
    prepared: PreparedLesson,
    teaching_route: FrozenTeachingRoute,
    verified_math_steps: Optional[List[MathStep]] = None,
) -> PreparedDraftRun:
    try:
        return _prepared_lesson_to_draft_with_provenance(
            problem,
            prepared,
            teaching_route,
            verified_math_steps,
        )
    except PreparedLessonAdaptationError:
        raise
    except (PreparationValidationError, ValidationError) as error:
        raise PreparedLessonAdaptationError(
            "prepared lesson adaptation failed"
        ) from error


def prepared_lesson_to_draft(
    problem: ProblemInput,
    prepared: PreparedLesson,
    teaching_route: FrozenTeachingRoute,
    verified_math_steps: Optional[List[MathStep]] = None,
) -> LessonDraft:
    """Compatibility adapter returning only the defensive runtime draft."""
    return prepared_lesson_to_draft_with_provenance(
        problem,
        prepared,
        teaching_route,
        verified_math_steps,
    ).draft
