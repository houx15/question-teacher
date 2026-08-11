"""Adapt an approved private lesson preparation to the runtime draft."""

from dataclasses import dataclass
from typing import List, Optional, Set, Tuple

from app.preparation_models import (
    PerformanceCue,
    PlannedInteraction,
    PreparedLesson,
    TeachingScript,
)
from app.preparation_validation import validate_prepared_lesson
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


@dataclass(frozen=True)
class RuntimeCueProvenance:
    """Immutable private link from an authored clause to its runtime cue."""

    episode_id: str
    clause_id: str
    original_performance_cue_id: str
    runtime_cue_id: str
    spoken_text: str


@dataclass(frozen=True)
class PreparedDraftRun:
    """Defensive draft plus request-private clause-to-cue provenance."""

    _draft_json: str
    cue_provenance: Tuple[RuntimeCueProvenance, ...]

    @classmethod
    def from_draft(
        cls,
        draft: LessonDraft,
        cue_provenance: List[RuntimeCueProvenance],
    ) -> "PreparedDraftRun":
        if type(draft) is not LessonDraft:
            raise TypeError("draft must be an exact LessonDraft")
        return cls(
            _draft_json=draft.model_dump_json(),
            cue_provenance=tuple(cue_provenance),
        )

    @property
    def draft(self) -> LessonDraft:
        return LessonDraft.model_validate_json(self._draft_json)


_COMPILER_RESERVED_CUE_IDS = {
    "runtime-opening-cue",
    "runtime-method-introduction-cue",
    "runtime-summary-cue",
    "runtime-transfer-intro-cue",
}


def _runtime_cue_id(
    original_id: str,
    score_index: int,
    part_index: int,
    reserved_ids: Set[str],
    allocated_ids: Set[str],
) -> str:
    if (
        part_index == 0
        and original_id not in _COMPILER_RESERVED_CUE_IDS
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
            and candidate not in _COMPILER_RESERVED_CUE_IDS
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
    return NarrativeSyncCue(
        cue_id=cue_id,
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


def _runtime_interaction(planned: PlannedInteraction) -> Interaction:
    return Interaction(
        interaction_id=planned.interaction_id,
        kind="choice",
        prompt=planned.prompt,
        expected_answer=planned.correct_option_id,
        options=[
            InteractionOption(
                option_id=option.option_id,
                label=option.display_text,
                feedback=(
                    planned.correct_feedback
                    if option.option_id == planned.correct_option_id
                    else planned.incorrect_feedback_by_option[
                        option.option_id
                    ]
                ),
            )
            for option in planned.options
        ],
        hints=[planned.hint],
        explanation_after_correct=planned.correct_feedback,
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
    interactions_after_clause = {
        item.after_clause_id: _runtime_interaction(item)
        for item in prepared.interaction_plan.interactions
    }
    interaction_episode_ids = {
        item.episode_id for item in prepared.interaction_plan.interactions
    }
    reserved_ids = {
        cue.cue_id for cue in prepared.performance_score.cues
    }
    allocated_ids = set()
    fixed_interactions = {}
    cue_provenance = []

    def clause_section(clause_id: str) -> str:
        if clause_id in opening_ids:
            return "opening"
        if clause_id in method_ids:
            return "method"
        if clause_id in summary_ids:
            return "summary"
        return "body"

    for score_index, score_cue in enumerate(prepared.performance_score.cues):
        parts = []
        current = []
        current_key = None
        current_length = 0
        for clause_id in score_cue.clause_ids:
            clause = clauses[clause_id]
            key = (
                clause_section(clause_id),
                clause.episode_id,
                layer_by_clause[clause_id],
            )
            spoken_length = len(clause.spoken_text)
            if current and (
                key != current_key or current_length + spoken_length > 90
            ):
                parts.append(tuple(current))
                current = []
                current_length = 0
            current.append(clause_id)
            current_key = key
            current_length += spoken_length
            if clause_id in interactions_after_clause:
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
            cue_provenance.extend(
                RuntimeCueProvenance(
                    episode_id=clauses[clause_id].episode_id,
                    clause_id=clause_id,
                    original_performance_cue_id=score_cue.cue_id,
                    runtime_cue_id=cue_id,
                    spoken_text=clauses[clause_id].spoken_text,
                )
                for clause_id in clause_ids
            )
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
                if interaction is not None:
                    fixed_interactions[cue_id] = interaction
    if any(not section_cues[name] for name in ("opening", "method", "summary")):
        raise ValueError("prepared script section has no performance cue")
    if not section_cues["body"]:
        raise ValueError("prepared script requires explanatory body clauses")
    return (
        section_cues["opening"],
        section_cues["method"],
        section_cues["body"],
        section_cues["summary"],
        fixed_interactions,
        cue_provenance,
    )


def _lesson_moments(prepared: PreparedLesson, body: list) -> List[LessonMoment]:
    episodes = {
        item.episode_id: item for item in prepared.reasoning_trajectory.episodes
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
        purpose = "；".join(episodes[item].decision for item in episode_order)
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
    source = prepared.interaction_plan.transfer_item
    return TransferItem(
        problem_text=source.problem_text,
        expected_answer=source.expected_answer,
        method_signal=source.method_signal,
        options=[
            TransferOption(
                option_id=option.option_id,
                label=option.label,
                canonical_answer=option.canonical_answer,
                feedback=option.feedback,
            )
            for option in source.options
        ],
        correct_option_id=source.correct_option_id,
    )


def prepared_lesson_to_draft_with_provenance(
    problem: ProblemInput,
    prepared: PreparedLesson,
    teaching_route: FrozenTeachingRoute,
    verified_math_steps: Optional[List[MathStep]] = None,
) -> PreparedDraftRun:
    if type(problem) is not ProblemInput:
        raise TypeError("problem must be an exact ProblemInput")
    if type(prepared) is not PreparedLesson:
        raise TypeError("prepared must be an exact PreparedLesson")
    if type(teaching_route) is not FrozenTeachingRoute:
        raise TypeError("teaching_route must be an exact FrozenTeachingRoute")
    if prepared.review.status != "approved":
        raise ValueError("prepared lesson must be approved before adaptation")

    route_payload = teaching_route.to_prompt_payload()
    if teaching_route.mode is TeachingRouteMode.SYMBOLIC_VERIFIED:
        if not verified_math_steps:
            raise ValueError("symbolic route requires non-empty math steps")
        math_steps = [item.model_copy(deep=True) for item in verified_math_steps]
    else:
        if verified_math_steps is not None:
            raise ValueError("grounded route rejects verified math steps")
        math_steps = []

    problem_targets = compile_problem_focus_targets(problem.problem_text)
    validate_prepared_lesson(prepared, teaching_route, problem_targets)
    (
        opening,
        method,
        body,
        summary,
        fixed_interactions,
        cue_provenance,
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
        transfer_feedback_is_authoritative=True,
        math_steps=math_steps,
        teaching_route=route_payload,
        moments=_lesson_moments(prepared, body),
        summary="".join(cue.spoken_text for cue in summary),
        transfer_item=_transfer_item(prepared),
    )
    return PreparedDraftRun.from_draft(draft, cue_provenance)


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
