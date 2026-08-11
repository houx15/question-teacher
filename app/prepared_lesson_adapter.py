"""Adapt an approved private lesson preparation to the runtime draft."""

from dataclasses import dataclass
from typing import List, Optional

from app.preparation_models import (
    InteractionPlan,
    PerformanceCue,
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


def _narrative_cue(
    cue: PerformanceCue,
    script: TeachingScript,
) -> NarrativeSyncCue:
    clauses = {item.clause_id: item for item in script.clauses}
    return NarrativeSyncCue(
        cue_id=cue.cue_id,
        spoken_text="".join(
            clauses[clause_id].spoken_text for clause_id in cue.clause_ids
        ),
        lead_actions=[
            item.action.model_copy(deep=True) for item in cue.lead_actions
        ],
        start_actions=[
            item.action.model_copy(deep=True) for item in cue.start_actions
        ],
        end_actions=[
            item.action.model_copy(deep=True) for item in cue.end_actions
        ],
    )


def _interaction_by_episode(plan: InteractionPlan) -> dict:
    result = {}
    for planned in plan.interactions:
        if planned.episode_id in result:
            raise ValueError(
                "one runtime moment cannot contain multiple interactions"
            )
        result[planned.episode_id] = Interaction(
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
    return result


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
    opening = set(script.opening_clause_ids)
    method = set(script.method_introduction_clause_ids)
    summary = set(script.closing_summary_clause_ids)
    layer_by_clause = _layer_by_clause(prepared)
    section_cues = {"opening": [], "method": [], "body": [], "summary": []}

    for score_cue in prepared.performance_score.cues:
        sections = {
            "opening"
            if clause_id in opening
            else "method"
            if clause_id in method
            else "summary"
            if clause_id in summary
            else "body"
            for clause_id in score_cue.clause_ids
        }
        if len(sections) != 1:
            raise ValueError(
                "performance cue cannot cross a runtime section boundary"
            )
        section = sections.pop()
        narrative_cue = _narrative_cue(score_cue, script)
        if section != "body":
            section_cues[section].append(narrative_cue)
            continue
        episode_ids = {
            clauses[clause_id].episode_id
            for clause_id in score_cue.clause_ids
        }
        cue_layers = {
            layer_by_clause[clause_id] for clause_id in score_cue.clause_ids
        }
        if len(episode_ids) != 1 or len(cue_layers) != 1:
            raise ValueError(
                "body performance cue cannot cross episode or overlay boundaries"
            )
        section_cues["body"].append(
            _AdaptedCue(
                episode_id=episode_ids.pop(),
                layer=cue_layers.pop(),
                cue=narrative_cue,
            )
        )
    if any(not section_cues[name] for name in ("opening", "method", "summary")):
        raise ValueError("prepared script section has no performance cue")
    if not section_cues["body"]:
        raise ValueError("prepared script requires explanatory body clauses")
    return (
        section_cues["opening"],
        section_cues["method"],
        section_cues["body"],
        section_cues["summary"],
    )


def _lesson_moments(prepared: PreparedLesson, body: list) -> List[LessonMoment]:
    episodes = {
        item.episode_id: item for item in prepared.reasoning_trajectory.episodes
    }
    interactions = _interaction_by_episode(prepared.interaction_plan)
    moments = []
    index = 0
    while index < len(body):
        first = body[index]
        interaction = interactions.get(first.episode_id)
        group = [first]
        index += 1
        if interaction is not None:
            while index < len(body) and body[index].episode_id == first.episode_id:
                if body[index].layer != first.layer:
                    raise ValueError(
                        "interaction episode cannot cross an overlay boundary"
                    )
                group.append(body[index])
                index += 1
        else:
            while index < len(body):
                candidate = body[index]
                if (
                    candidate.layer != first.layer
                    or candidate.episode_id in interactions
                ):
                    break
                group.append(candidate)
                index += 1
        if len(group) > 5:
            raise ValueError("runtime moment exceeds the supported cue count")
        episode_order = []
        for item in group:
            if item.episode_id not in episode_order:
                episode_order.append(item.episode_id)
        purpose = "；".join(episodes[item].decision for item in episode_order)
        moments.append(
            LessonMoment(
                purpose=purpose,
                sync_cues=[item.cue for item in group],
                layer="interaction" if interaction is not None else first.layer,
                interaction=interaction,
            )
        )
    missing_interactions = set(interactions) - {
        item.episode_id for item in body
    }
    if missing_interactions:
        raise ValueError("interaction episode has no runtime body cue")
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


def prepared_lesson_to_draft(
    problem: ProblemInput,
    prepared: PreparedLesson,
    teaching_route: FrozenTeachingRoute,
    verified_math_steps: Optional[List[MathStep]] = None,
) -> LessonDraft:
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
    opening, method, body, summary = _runtime_sections(prepared)
    route_payload["teaching_route_fingerprint"] = teaching_route.fingerprint
    script = prepared.teaching_script
    return LessonDraft(
        title=script.title,
        learning_goal=script.learning_goal,
        opening="".join(cue.spoken_text for cue in opening),
        method_rationale=script.method_rationale,
        method_introduction=script.method_introduction.model_copy(deep=True),
        opening_sync_cues=opening,
        method_introduction_sync_cues=method,
        summary_sync_cues=summary,
        math_steps=math_steps,
        teaching_route=route_payload,
        moments=_lesson_moments(prepared, body),
        summary="".join(cue.spoken_text for cue in summary),
        transfer_item=_transfer_item(prepared),
    )
