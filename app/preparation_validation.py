"""Deterministic cross-artifact validation for prepared lessons."""

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

from app.math_content import (
    contains_bounded_answer_token,
    contains_cross_artifact_math_identity,
    contains_explicit_choice_answer_leak,
    contains_internal_control_syntax,
    contains_math_markup,
    contains_normalized_cross_artifact_math_identity,
    is_valid_generated_display_content,
    normalize_answer_leak_text,
    normalize_grounded_choice_option_label,
    normalize_cross_artifact_math_identity,
)
from app.pedagogy_rubric import PEDAGOGY_RUBRIC_VERSION
from app.problem_focus import MAX_PROBLEM_FOCUS_TARGETS
from app.preparation_models import (
    InteractionPlan,
    LessonReviewDecision,
    MAX_PREPARATION_ITEMS,
    PerformanceScore,
    PreparedLesson,
    ReasoningTrajectory,
    SimulationReport,
    SolutionTrace,
    TeachingScript,
)
from app.schemas import ProblemFocusTarget, SyncVisualAction
from app.teaching_route import FrozenTeachingRoute


ProblemTargets = Union[List[ProblemFocusTarget], Tuple[ProblemFocusTarget, ...]]
ActionKey = Tuple[str, str]
MAX_PERFORMANCE_CLAUSES = MAX_PREPARATION_ITEMS
MAX_PERFORMANCE_ACTIONS = 2048
MAX_PERFORMANCE_MATH_REFERENCES = 2048
_REVIEW_ROLE_ORDER = {
    "reference_analyst": 0,
    "teaching_designer": 1,
    "script_teacher": 2,
    "interaction_designer": 3,
    "classroom_director": 4,
}
_ARTIFACT_DEPENDENCY_ORDER = (
    "solution_trace",
    "reasoning_trajectory",
    "teaching_script",
    "interaction_plan",
    "performance_score",
    "simulation_report",
)
_ARTIFACT_OWNER_ORDER = {
    artifact_type: index
    for index, artifact_type in enumerate(_ARTIFACT_DEPENDENCY_ORDER)
}
_REPAIR_ROLE_ARTIFACT = {
    "reference_analyst": "solution_trace",
    "teaching_designer": "reasoning_trajectory",
    "script_teacher": "teaching_script",
    "interaction_designer": "interaction_plan",
    "classroom_director": "performance_score",
}
_ARTIFACT_HISTORY_ROLES = {
    "solution_trace": "reference_analyst",
    "reasoning_trajectory": "teaching_designer",
    "teaching_script": "script_teacher",
    "interaction_plan": "interaction_designer",
    "performance_score": "classroom_director",
    "simulation_report": "student_simulator",
}
MAX_REPAIR_CYCLES = 8


class PreparationValidationError(ValueError):
    def __init__(self, code: str, artifact_id: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.artifact_id = artifact_id
        self.detail = detail


class VisualActionValidationError(ValueError):
    """Internal shared semantic-action error with a stable preparation code."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


def _require_exact(value: object, expected: type, label: str) -> None:
    if type(value) is not expected:
        raise TypeError("%s must be an exact %s model" % (label, expected.__name__))


def _fail(code: str, artifact_id: str, detail: str) -> None:
    raise PreparationValidationError(code, artifact_id, detail)


def _contains_exact_id_token(notes: Sequence[str], item_id: str) -> bool:
    pattern = re.compile(
        r"(?<![A-Za-z0-9_-])%s(?![A-Za-z0-9_-])" % re.escape(item_id)
    )
    return any(pattern.search(note) is not None for note in notes)


def validate_solution_trace(
    trace: SolutionTrace,
    teaching_route: FrozenTeachingRoute,
) -> None:
    _require_exact(trace, SolutionTrace, "trace")
    _require_exact(teaching_route, FrozenTeachingRoute, "teaching_route")
    route_payload = teaching_route.to_prompt_payload()
    if normalize_cross_artifact_math_identity(
        trace.task_target
    ) != normalize_cross_artifact_math_identity(route_payload["target"]):
        _fail(
            "trace_target_mismatch",
            "solution_trace",
            "Solution trace target does not match the frozen teaching route.",
        )
    if normalize_cross_artifact_math_identity(
        trace.reference_conclusion
    ) != normalize_cross_artifact_math_identity(
        teaching_route.final_conclusion
    ):
        _fail(
            "trace_conclusion_mismatch",
            "solution_trace",
            "Solution trace conclusion does not match the frozen teaching route.",
        )

    route_assumptions = {
        item["assumption_id"]: item
        for item in route_payload["assumptions"]
    }
    trace_assumptions = {
        item.assumption_id: item.content for item in trace.assumptions
    }
    if set(trace_assumptions) != set(route_assumptions) or any(
        normalize_cross_artifact_math_identity(trace_assumptions[item_id])
        != normalize_cross_artifact_math_identity(item["expression"])
        for item_id, item in route_assumptions.items()
    ):
        _fail(
            "trace_assumption_mismatch",
            "solution_trace",
            "Solution trace assumptions must match the frozen route.",
        )
    assumption_ids = set(trace_assumptions)
    for assumption in trace.assumptions:
        if assumption.source_anchor.source_kind != route_assumptions[
            assumption.assumption_id
        ]["source_kind"]:
            _fail(
                "trace_assumption_provenance_mismatch",
                assumption.assumption_id,
                "Trace assumption provenance does not match the frozen route.",
            )
    route_steps = route_payload["steps"]
    trace_step_ids = [item.source_step_id for item in trace.source_steps]
    if trace_step_ids != [item["step_id"] for item in route_steps]:
        _fail(
            "trace_step_order_mismatch",
            "solution_trace",
            "Solution trace steps must exactly match frozen route order.",
        )
    for step in trace.source_steps:
        if (
            step.source_anchor.source_kind != "verified_route"
            or step.source_anchor.source_id != step.source_step_id
        ):
            _fail(
                "trace_source_anchor_invalid",
                step.source_step_id,
                "Trace step must bind exactly to its frozen route step.",
            )
    for step, route_step in zip(trace.source_steps, route_steps):
        if (
            normalize_cross_artifact_math_identity(step.state_before)
            != normalize_cross_artifact_math_identity(
                route_step["statement_before"]
            )
            or normalize_cross_artifact_math_identity(step.state_after)
            != normalize_cross_artifact_math_identity(
                route_step["statement_after"]
            )
        ):
            _fail(
                "trace_state_mismatch",
                step.source_step_id,
                "Solution trace state does not match its frozen route step.",
            )
        missing = set(step.assumption_ids_used) - assumption_ids
        if missing:
            _fail(
                "trace_assumption_missing",
                step.source_step_id,
                "Trace step references an undeclared assumption.",
            )
        route_operands = [
            normalize_cross_artifact_math_identity(item)
            for item in route_step["operands"]
        ]
        trace_operands = [
            normalize_cross_artifact_math_identity(item)
            for item in step.operands
        ]
        if (
            step.operation_kind != route_step["operation_kind"]
            or trace_operands != route_operands
            or step.assumption_ids_used
            != route_step.get("assumption_ids_used", [])
            or step.evidence_status != route_step["evidence_status"]
        ):
            _fail(
                "trace_typed_decision_mismatch",
                step.source_step_id,
                "Trace typed decision does not match the frozen route.",
            )
        if not set(step.reasoning_gap_codes) <= set(
            route_step.get("allowed_reasoning_gap_codes", [])
        ):
            _fail(
                "trace_gap_basis_invalid",
                step.source_step_id,
                "Trace reasoning gap lacks a frozen route basis.",
            )


def validate_reasoning_trajectory(
    trajectory: ReasoningTrajectory,
    trace: SolutionTrace,
) -> None:
    _require_exact(trajectory, ReasoningTrajectory, "trajectory")
    _require_exact(trace, SolutionTrace, "trace")
    trace_order = {
        step.source_step_id: index
        for index, step in enumerate(trace.source_steps)
    }
    covered = set()
    selected_gaps = {
        (step.source_step_id, gap_code)
        for step in trace.source_steps
        for gap_code in step.reasoning_gap_codes
    }
    resolved_gaps = set()
    last_position = -1
    for episode in trajectory.episodes:
        episode_must_teach_ids = {
            item.must_teach_id for item in episode.must_teach
        }
        for gap_ref in episode.resolved_gap_refs:
            key = (gap_ref.source_step_id, gap_ref.gap_code)
            if (
                gap_ref.source_step_id not in episode.source_step_ids
                or gap_ref.must_teach_id not in episode_must_teach_ids
                or key not in selected_gaps
            ):
                _fail(
                    "trajectory_gap_ref_invalid",
                    episode.episode_id,
                    "Reasoning gap reference is not bound to this episode and must-teach item.",
                )
            resolved_gaps.add(key)
        for step_id in episode.source_step_ids:
            if step_id not in trace_order:
                _fail(
                    "episode_source_missing",
                    episode.episode_id,
                    "Reasoning episode references an unknown trace step.",
                )
            position = trace_order[step_id]
            if step_id not in covered:
                if position < last_position:
                    _fail(
                        "episode_source_order_invalid",
                        episode.episode_id,
                        "Reasoning episode order violates trace-step dependencies.",
                    )
                last_position = position
            covered.add(step_id)
    if selected_gaps - resolved_gaps:
        _fail(
            "trajectory_gap_unresolved",
            "reasoning_trajectory",
            "Every selected reasoning gap requires explicit must-teach coverage.",
        )
    for step_id in trace_order:
        if step_id not in covered and not _contains_exact_id_token(
            trace.audit_notes, step_id
        ):
            _fail(
                "trace_step_uncovered",
                step_id,
                "Trace step has no reasoning episode coverage or exact audit note.",
            )


def validate_teaching_script(
    script: TeachingScript,
    trajectory: ReasoningTrajectory,
) -> None:
    _require_exact(script, TeachingScript, "script")
    _require_exact(trajectory, ReasoningTrajectory, "trajectory")
    episode_positions = {
        episode.episode_id: index
        for index, episode in enumerate(trajectory.episodes)
    }
    must_teach_ids = [
        item.must_teach_id
        for episode in trajectory.episodes
        for item in episode.must_teach
    ]
    if len(must_teach_ids) != len(set(must_teach_ids)):
        _fail(
            "must_teach_id_duplicate",
            "reasoning_trajectory",
            "Must-teach IDs must be globally unique across episodes.",
        )
    must_teach_owner = {
        item.must_teach_id: episode.episode_id
        for episode in trajectory.episodes
        for item in episode.must_teach
    }
    section_clause_ids = {
        *script.opening_clause_ids,
        *script.method_introduction_clause_ids,
        *script.closing_summary_clause_ids,
    }
    if all(
        clause.clause_id in section_clause_ids for clause in script.clauses
    ):
        _fail(
            "script_body_missing",
            "teaching_script",
            "Teaching script requires at least one explanatory body clause.",
        )
    covered = set()
    last_episode_position = -1
    for clause in script.clauses:
        if clause.episode_id not in episode_positions:
            _fail(
                "clause_episode_missing",
                clause.clause_id,
                "Script clause references an unknown reasoning episode.",
            )
        episode_position = episode_positions[clause.episode_id]
        if episode_position < last_episode_position:
            _fail(
                "clause_episode_order_invalid",
                clause.clause_id,
                "Script clause order violates trajectory order.",
            )
        last_episode_position = episode_position
        if contains_math_markup(clause.spoken_text) or contains_internal_control_syntax(
            clause.spoken_text
        ):
            _fail(
                "spoken_markup_invalid",
                clause.clause_id,
                "Spoken text contains math or internal visual markup.",
            )
        for reference in clause.must_teach_refs:
            if (
                reference not in must_teach_owner
                or must_teach_owner[reference] != clause.episode_id
            ):
                _fail(
                    "must_teach_ref_invalid",
                    clause.clause_id,
                    "Script clause has an invalid must-teach reference.",
                )
            covered.add(reference)
    for must_teach_id in must_teach_owner:
        if must_teach_id not in covered:
            _fail(
                "must_teach_uncovered",
                must_teach_id,
                "Must-teach item has no script-clause evidence.",
            )


def _labels_are_unique(labels: Sequence[str]) -> bool:
    normalized = [normalize_grounded_choice_option_label(item) for item in labels]
    return len(normalized) == len(set(normalized))


def _leaks_correct_answer(interaction: object) -> bool:
    correct = next(
        item
        for item in interaction.options
        if item.option_id == interaction.correct_option_id
    )
    visible_text = "|".join((interaction.prompt, interaction.hint))
    if contains_explicit_choice_answer_leak(
        visible_text,
        correct.option_id,
        correct.display_text,
    ) or contains_explicit_choice_answer_leak(
        visible_text,
        correct.option_id,
        correct.canonical_answer,
    ):
        return True
    correct_forms = {correct.display_text, correct.canonical_answer}
    correct_forms.discard("")
    for option in interaction.options:
        if option.option_id == interaction.correct_option_id:
            continue
        wrong_visible = (option.display_text,)
        if any(
            contains_bounded_answer_token(item, form)
            for form in correct_forms
            for item in wrong_visible
        ):
            return True
    return False


def _concealed_registry(
    episode_id: str,
    trajectory: ReasoningTrajectory,
    script: TeachingScript,
) -> dict:
    episode = next(
        item for item in trajectory.episodes if item.episode_id == episode_id
    )
    registry = {
        item.must_teach_id: [item.content]
        for item in episode.must_teach
    }
    for clause in script.clauses:
        if clause.episode_id != episode_id:
            continue
        registry[clause.clause_id] = [
            clause.spoken_text,
            *clause.math_references,
        ]
    return registry


def _leaks_concealed_content(
    interaction: object,
    concealed_registry: dict,
) -> bool:
    visible_parts = [interaction.prompt, interaction.hint]
    visible_parts.extend(
        item.display_text
        for item in interaction.options
        if item.option_id != interaction.correct_option_id
    )
    for target_id in interaction.concealed_targets:
        for content in concealed_registry[target_id]:
            if content and any(
                contains_bounded_answer_token(visible, content)
                for visible in visible_parts
            ):
                return True
    return False


def validate_interaction_plan(
    plan: InteractionPlan,
    trajectory: ReasoningTrajectory,
    script: TeachingScript,
) -> None:
    _require_exact(plan, InteractionPlan, "plan")
    _require_exact(trajectory, ReasoningTrajectory, "trajectory")
    _require_exact(script, TeachingScript, "script")
    episode_ids = {item.episode_id for item in trajectory.episodes}
    clauses = {item.clause_id: item for item in script.clauses}
    clause_positions = {
        item.clause_id: index for index, item in enumerate(script.clauses)
    }
    interaction_episode_ids = [
        interaction.episode_id for interaction in plan.interactions
    ]
    if len(interaction_episode_ids) != len(set(interaction_episode_ids)):
        _fail(
            "interaction_episode_duplicate",
            "interaction_plan",
            "An episode may contain at most one runtime interaction.",
        )
    for interaction in plan.interactions:
        after = clauses.get(interaction.after_clause_id)
        resume = clauses.get(interaction.resume_clause_id)
        if (
            interaction.episode_id not in episode_ids
            or after is None
            or resume is None
            or after.episode_id != interaction.episode_id
            or resume.episode_id != interaction.episode_id
            or clause_positions[resume.clause_id]
            != clause_positions[after.clause_id] + 1
        ):
            _fail(
                "interaction_clause_invalid",
                interaction.interaction_id,
                "Interaction clause boundary is inconsistent with its episode.",
            )
        if not _labels_are_unique(
            [item.display_text for item in interaction.options]
        ):
            _fail(
                "choice_formula_duplicate",
                interaction.interaction_id,
                "Interaction option labels are display-equivalent.",
            )
        if any(
            not is_valid_generated_display_content(item.display_text)
            for item in interaction.options
        ):
            _fail(
                "choice_formula_invalid",
                interaction.interaction_id,
                "Interaction option contains invalid display markup.",
            )
        concealed_registry = _concealed_registry(
            interaction.episode_id,
            trajectory,
            script,
        )
        if (
            interaction.correct_option_id in interaction.concealed_targets
            or interaction.resume_clause_id in interaction.concealed_targets
            or any(
                item not in concealed_registry
                for item in interaction.concealed_targets
            )
            or _leaks_correct_answer(interaction)
            or _leaks_concealed_content(
                interaction,
                concealed_registry,
            )
        ):
            _fail(
                "interaction_answer_leakage",
                interaction.interaction_id,
                "Interaction exposes or misbinds a concealed answer target.",
            )

    transfer_labels = [item.label for item in plan.transfer_item.options]
    if not _labels_are_unique(transfer_labels):
        _fail(
            "choice_formula_duplicate",
            "transfer_item",
            "Transfer option labels are display-equivalent.",
        )
    if any(not is_valid_generated_display_content(item) for item in transfer_labels):
        _fail(
            "choice_formula_invalid",
            "transfer_item",
            "Transfer option contains invalid display markup.",
        )


def _visual_error(code: str, detail: str) -> None:
    raise VisualActionValidationError(code, detail)


def validate_current_cue_cleanup(
    action: SyncVisualAction,
    focused_targets: Set[ActionKey],
    emphasized_targets: Set[ActionKey],
) -> None:
    action_key = (action.surface, action.target)
    if action.type == "fade" and action_key not in emphasized_targets:
        _visual_error(
            "visual_target_invalid",
            "结束动作没有匹配当前 cue 的强调活动状态。",
        )
    if action.type == "clear_focus" and action_key not in focused_targets:
        _visual_error(
            "visual_target_invalid",
            "结束动作没有匹配当前 cue 的聚焦活动状态。",
        )


def validate_visual_action_references(
    moments: Sequence[object],
    problem_focus_targets: Sequence[ProblemFocusTarget] = (),
    board_target_ids: Optional[Set[str]] = None,
) -> None:
    """Shared phase, target, lifecycle, and cue-cleanup legality core."""
    problem_target_ids = {target.target_id for target in problem_focus_targets}
    base_targets: Set[str] = set()
    allowed_phase_types = {
        "lead_actions": {"focus", "emphasize"},
        "start_actions": {
            "write",
            "transform",
            "focus",
            "emphasize",
            "annotate",
            "reveal",
        },
        "end_actions": {"clear_focus", "fade"},
    }
    for moment in moments:
        active_targets = set(base_targets)
        for cue in moment.sync_cues:
            cue_focused_targets: Set[ActionKey] = set()
            cue_emphasized_targets: Set[ActionKey] = set()
            action_phases = (
                ("lead_actions", cue.lead_actions),
                ("start_actions", cue.start_actions),
                ("end_actions", cue.end_actions),
            )
            for phase, actions in action_phases:
                for bound_action in actions:
                    action = getattr(bound_action, "action", bound_action)
                    if action.type not in allowed_phase_types[phase]:
                        _visual_error(
                            "visual_target_invalid",
                            "%s 包含不允许的动作类型。" % phase,
                        )
                    action_key = (action.surface, action.target)
                    if action.surface == "problem":
                        if action.target not in problem_target_ids:
                            _visual_error(
                                "visual_target_invalid",
                                "视觉动作引用了未知的题面目标。",
                            )
                        if phase == "end_actions":
                            validate_current_cue_cleanup(
                                action,
                                cue_focused_targets,
                                cue_emphasized_targets,
                            )
                        if action.type == "focus":
                            cue_focused_targets.add(action_key)
                        elif action.type == "emphasize":
                            cue_emphasized_targets.add(action_key)
                        continue
                    if action.type in {"write", "transform"}:
                        if board_target_ids is not None and action.target not in board_target_ids:
                            _visual_error(
                                "visual_target_invalid",
                                "板书动作引用了未声明的对象。",
                            )
                        if action.source is not None and action.source not in active_targets:
                            _visual_error(
                                "visual_target_invalid",
                                "板书变形引用了尚未创建的来源对象。",
                            )
                        active_targets.add(action.target)
                        continue
                    required_targets = [action.target]
                    if action.type == "annotate" and action.annotation == "arrow":
                        required_targets.append(action.relation_target)
                    if any(target not in active_targets for target in required_targets):
                        _visual_error(
                            "visual_target_invalid",
                            "板书动作引用了尚未创建的对象。",
                        )
                    if phase == "end_actions":
                        validate_current_cue_cleanup(
                            action,
                            cue_focused_targets,
                            cue_emphasized_targets,
                        )
                    if action.type == "focus":
                        cue_focused_targets.add(action_key)
                    elif action.type == "emphasize":
                        cue_emphasized_targets.add(action_key)
        if getattr(moment, "layer", "base") == "base":
            base_targets = active_targets


@dataclass(frozen=True)
class _ScoreMoment:
    sync_cues: Sequence[object]
    layer: str = "base"


def _score_moments(
    score: PerformanceScore,
    script_positions: dict,
) -> List[_ScoreMoment]:
    cue_end_ids = {cue.clause_ids[-1] for cue in score.cues}
    transitions_by_clause = {}
    previous_position = -1
    for transition in score.overlay_transitions:
        position = script_positions.get(transition.after_clause_id)
        if (
            position is None
            or position < previous_position
            or transition.after_clause_id not in cue_end_ids
        ):
            _fail(
                "overlay_transition_invalid",
                transition.transition_id,
                "Overlay transition must reference an ordered cue boundary.",
            )
        previous_position = position
        transitions_by_clause.setdefault(
            transition.after_clause_id, []
        ).append(transition)

    moments = []
    active_layer = "base"
    active_layer_start_position = None
    current_cues = []
    for cue in score.cues:
        current_cues.append(cue)
        transitions = transitions_by_clause.get(cue.clause_ids[-1], [])
        if not transitions:
            continue
        moments.append(
            _ScoreMoment(tuple(current_cues), active_layer)
        )
        current_cues = []
        for transition in transitions:
            if transition.action == "enter":
                if active_layer != "base":
                    _fail(
                        "overlay_transition_invalid",
                        transition.transition_id,
                        "Nested overlay layers are not supported.",
                    )
                active_layer = transition.layer
                active_layer_start_position = script_positions[
                    transition.after_clause_id
                ]
            elif active_layer != transition.layer:
                _fail(
                    "overlay_transition_invalid",
                    transition.transition_id,
                    "Overlay return does not match the active layer.",
                )
            elif (
                active_layer_start_position is not None
                and script_positions[transition.after_clause_id]
                <= active_layer_start_position
            ):
                _fail(
                    "overlay_transition_invalid",
                    transition.transition_id,
                    "Overlay enter and return require an intervening cue.",
                )
            else:
                active_layer = "base"
                active_layer_start_position = None
    if active_layer != "base":
        _fail(
            "overlay_transition_invalid",
            "performance_score",
            "Overlay transition does not return to the base sequence.",
        )
    if current_cues:
        moments.append(_ScoreMoment(tuple(current_cues), active_layer))
    return moments


def _validate_score_layers(
    moments: Sequence[_ScoreMoment],
    board_objects: dict,
) -> None:
    for moment in moments:
        allowed_reference_layers = {"base", moment.layer}
        for cue in moment.sync_cues:
            for bound in (
                *cue.lead_actions,
                *cue.start_actions,
                *cue.end_actions,
            ):
                action = bound.action
                if action.surface != "board":
                    continue
                referenced_ids = [action.target]
                if action.source is not None:
                    referenced_ids.append(action.source)
                if action.relation_target is not None:
                    referenced_ids.append(action.relation_target)
                referenced_layers = {
                    board_objects[item].layer
                    for item in referenced_ids
                    if item in board_objects
                }
                if not referenced_layers.issubset(allowed_reference_layers):
                    _fail(
                        "visual_target_invalid",
                        bound.clause_id,
                        "Visual action crosses a returned overlay lifecycle.",
                    )
                if (
                    action.type in {"write", "transform"}
                    and action.target in board_objects
                    and board_objects[action.target].layer != moment.layer
                ):
                    _fail(
                        "overlay_transition_invalid",
                        bound.clause_id,
                        "Overlay segment continues a different board layer.",
                    )


def _validate_problem_targets(problem_targets: ProblemTargets) -> None:
    if type(problem_targets) not in (list, tuple) or any(
        type(item) is not ProblemFocusTarget for item in problem_targets
    ):
        raise TypeError(
            "problem_targets must be a list or tuple of exact ProblemFocusTarget models"
        )


def validate_performance_score(
    score: PerformanceScore,
    problem_targets: ProblemTargets,
    script: TeachingScript,
    plan: InteractionPlan,
) -> None:
    _require_exact(score, PerformanceScore, "score")
    _validate_problem_targets(problem_targets)
    _require_exact(script, TeachingScript, "script")
    _require_exact(plan, InteractionPlan, "plan")
    action_count = sum(
        len(cue.lead_actions)
        + len(cue.start_actions)
        + len(cue.end_actions)
        for cue in score.cues
    )
    math_reference_count = sum(
        len(clause.math_references) for clause in script.clauses
    )
    if (
        len(script.clauses) > MAX_PERFORMANCE_CLAUSES
        or len(score.cues) > MAX_PERFORMANCE_CLAUSES
        or len(score.board_objects) > MAX_PERFORMANCE_CLAUSES
        or len(problem_targets) > MAX_PROBLEM_FOCUS_TARGETS
        or action_count > MAX_PERFORMANCE_ACTIONS
        or math_reference_count > MAX_PERFORMANCE_MATH_REFERENCES
    ):
        _fail(
            "artifact_size_invalid",
            "performance_score",
            "Performance artifact exceeds deterministic validation bounds.",
        )
    script_ids = [item.clause_id for item in script.clauses]
    script_positions = {item: index for index, item in enumerate(script_ids)}
    clauses = {item.clause_id: item for item in script.clauses}

    for cue in score.cues:
        positions = [script_positions.get(item, -1) for item in cue.clause_ids]
        if -1 in positions:
            _fail(
                "cue_clause_coverage_invalid",
                cue.cue_id,
                "Performance cue references an unknown script clause.",
            )
        if positions != list(range(positions[0], positions[0] + len(positions))):
            _fail(
                "cue_clause_nonadjacent",
                cue.cue_id,
                "Performance cue combines nonadjacent script clauses.",
            )
        for bound in (*cue.lead_actions, *cue.start_actions, *cue.end_actions):
            if bound.clause_id not in cue.clause_ids:
                _fail(
                    "visual_clause_invalid",
                    cue.cue_id,
                    "Visual action is bound to a clause outside its cue.",
                )
    flattened = [item for cue in score.cues for item in cue.clause_ids]
    if flattened != script_ids:
        _fail(
            "cue_clause_coverage_invalid",
            "performance_score",
            "Performance cues must cover every script clause exactly once in order.",
        )

    board_objects = {item.board_object_id: item for item in score.board_objects}
    for board_object in score.board_objects:
        if not is_valid_generated_display_content(board_object.content):
            _fail(
                "board_formula_invalid",
                board_object.board_object_id,
                "Board object contains invalid formula or internal visual markup.",
            )
    for cue in score.cues:
        for bound in (*cue.lead_actions, *cue.start_actions, *cue.end_actions):
            if bound.action.content is not None and not is_valid_generated_display_content(
                bound.action.content
            ):
                _fail(
                    "board_formula_invalid",
                    bound.clause_id,
                    "Visual action contains invalid formula or internal visual markup.",
                )
            if (
                bound.action.type == "write"
                and bound.action.target in board_objects
                and normalize_cross_artifact_math_identity(
                    bound.action.content or ""
                )
                != normalize_cross_artifact_math_identity(
                    board_objects[bound.action.target].content
                )
            ):
                _fail(
                    "visual_target_invalid",
                    bound.clause_id,
                    "Board write content does not match its declared object.",
                )
    score_moments = _score_moments(score, script_positions)
    _validate_score_layers(score_moments, board_objects)
    layer_by_clause = {
        clause_id: moment.layer
        for moment in score_moments
        for cue in moment.sync_cues
        for clause_id in cue.clause_ids
    }
    try:
        validate_visual_action_references(
            score_moments,
            problem_targets,
            set(board_objects),
        )
    except VisualActionValidationError as error:
        _fail(error.code, "performance_score", error.detail)

    references_by_position = []
    first_reference_position = {}
    for position, clause in enumerate(script.clauses):
        references = tuple(
            normalize_cross_artifact_math_identity(item)
            for item in clause.math_references
        )
        references_by_position.append(references)
        for reference in references:
            first_reference_position.setdefault(reference, position)
    problem_content = {
        item.target_id: normalize_cross_artifact_math_identity(item.math_text)
        for item in problem_targets
    }
    problem_first_reference_position = {}
    for target_id, target_content in problem_content.items():
        first_position = None
        for position, references in enumerate(references_by_position):
            if any(
                contains_normalized_cross_artifact_math_identity(
                    reference,
                    target_content,
                )
                for reference in references
            ):
                first_position = position
                break
        problem_first_reference_position[target_id] = first_position
    visible_board_content_by_layer = {"base": {}}

    def active_visible_content(layer: str) -> dict:
        visible = dict(visible_board_content_by_layer["base"])
        if layer != "base":
            visible.update(
                visible_board_content_by_layer.get(layer, {})
            )
        return visible

    for cue in score.cues:
        for bound in (*cue.lead_actions, *cue.start_actions, *cue.end_actions):
            action = bound.action
            bound_position = script_positions[bound.clause_id]
            active_layer = layer_by_clause[bound.clause_id]
            if action.surface == "problem":
                first_position = problem_first_reference_position[action.target]
                if first_position is None or first_position > bound_position:
                    _fail(
                        "visual_action_too_early",
                        bound.clause_id,
                        "Problem visual target appears before its bound clause references it.",
                    )
                if not any(
                    contains_normalized_cross_artifact_math_identity(
                        reference,
                        problem_content[action.target],
                    )
                    for reference in references_by_position[bound_position]
                ):
                    _fail(
                        "visual_clause_invalid",
                        bound.clause_id,
                        "Problem visual target is not discussed by its bound clause.",
                    )
            if action.type in {"write", "transform"}:
                normalized_content = normalize_cross_artifact_math_identity(
                    action.content or ""
                )
                if first_reference_position.get(
                    normalized_content,
                    len(script.clauses),
                ) > bound_position:
                    _fail(
                        "visual_action_too_early",
                        bound.clause_id,
                        "Visual content appears before its bound clause references it.",
                    )
                target_layer = board_objects[action.target].layer
                visible_board_content_by_layer.setdefault(
                    target_layer,
                    {},
                )[action.target] = action.content or ""
            elif (
                action.surface == "board"
                and action.target in active_visible_content(active_layer)
                and action.type not in {"clear_focus", "fade"}
                and first_reference_position.get(
                    normalize_cross_artifact_math_identity(
                        active_visible_content(active_layer)[action.target]
                    ),
                    len(script.clauses),
                )
                > bound_position
            ):
                _fail(
                    "visual_action_too_early",
                    bound.clause_id,
                    "Board visual target appears before its bound clause references it.",
                )
            visible_content = active_visible_content(active_layer)
            if (
                action.type == "emphasize"
                and action.surface == "board"
                and len(visible_content) == 1
                and action.target in visible_content
            ):
                _fail(
                    "non_discriminating_emphasis",
                    bound.clause_id,
                    "Emphasis on the sole visible board object is not discriminating.",
                )
            if (
                action.type == "annotate"
                and action.annotation == "label"
                and action.content is not None
                and len(visible_content) == 1
                and action.target in visible_content
                and normalize_cross_artifact_math_identity(action.content)
                == normalize_cross_artifact_math_identity(
                    visible_content[action.target]
                )
            ):
                _fail(
                    "non_discriminating_emphasis",
                    bound.clause_id,
                    "Label annotation exactly repeats the sole visible target content.",
                )


def validate_simulation_report(
    report: SimulationReport,
    trajectory: ReasoningTrajectory,
    plan: InteractionPlan,
) -> None:
    _require_exact(report, SimulationReport, "report")
    _require_exact(trajectory, ReasoningTrajectory, "trajectory")
    _require_exact(plan, InteractionPlan, "plan")
    expected = [item.episode_id for item in trajectory.episodes]
    actual = [item.episode_id for item in report.episode_results]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        _fail(
            "simulation_episode_coverage_invalid",
            "simulation_report",
            "Simulation must contain exactly one result for every reasoning episode.",
        )
    private_values = {"correct_option_id", "canonical_answer"}
    private_option_ids = []
    for interaction in plan.interactions:
        correct = next(
            item
            for item in interaction.options
            if item.option_id == interaction.correct_option_id
        )
        private_option_ids.append(interaction.correct_option_id)
        private_values.update(
            {
                correct.display_text,
                correct.canonical_answer,
                interaction.correct_feedback,
            }
        )
    transfer = plan.transfer_item
    transfer_correct = next(
        item
        for item in transfer.options
        if item.option_id == transfer.correct_option_id
    )
    private_option_ids.append(transfer.correct_option_id)
    private_values.update(
        {
            transfer.expected_answer,
            transfer_correct.canonical_answer,
            transfer_correct.label,
        }
    )
    for result in report.interaction_results:
        normalized = normalize_answer_leak_text(result)
        leaks_semantic_value = any(
            item and contains_bounded_answer_token(result, item)
            for item in private_values
        )
        leaks_option_id = any(
            (
                option_id_normalized in normalized
                if len(option_id_normalized) >= 4
                else contains_explicit_choice_answer_leak(
                    result,
                    option_id,
                    "",
                )
            )
            for option_id in private_option_ids
            for option_id_normalized in [
                normalize_answer_leak_text(option_id)
            ]
        )
        if leaks_semantic_value or leaks_option_id:
            _fail(
                "simulation_private_answer_invalid",
                "simulation_report",
                "Simulation interaction result contains private answer material.",
            )


def validate_review_decision(
    decision: LessonReviewDecision,
    trace: Optional[SolutionTrace] = None,
    trajectory: Optional[ReasoningTrajectory] = None,
    script: Optional[TeachingScript] = None,
    plan: Optional[InteractionPlan] = None,
    score: Optional[PerformanceScore] = None,
    report: Optional[SimulationReport] = None,
) -> None:
    _require_exact(decision, LessonReviewDecision, "decision")
    if decision.status == "approved" and any(
        item.severity in {"blocking", "material"}
        for item in decision.findings
    ):
        _fail(
            "review_approval_invalid",
            "review",
            "Approved review contains a blocking or material finding.",
        )
    supplied = (trace, trajectory, script, plan, score, report)
    if all(item is None for item in supplied):
        return
    if any(item is None for item in supplied):
        raise TypeError("review validation requires all prepared artifacts")
    _require_exact(trace, SolutionTrace, "trace")
    _require_exact(trajectory, ReasoningTrajectory, "trajectory")
    _require_exact(script, TeachingScript, "script")
    _require_exact(plan, InteractionPlan, "plan")
    _require_exact(score, PerformanceScore, "score")
    _require_exact(report, SimulationReport, "report")

    artifact_ids = {
        "solution_trace": {
            "solution_trace",
            *(item.assumption_id for item in trace.assumptions),
            *(item.source_anchor.source_id for item in trace.assumptions),
            *(item.source_step_id for item in trace.source_steps),
            *(item.source_anchor.source_id for item in trace.source_steps),
        },
        "reasoning_trajectory": {
            "reasoning_trajectory",
            *(item.episode_id for item in trajectory.episodes),
            *(
                must_teach.must_teach_id
                for episode in trajectory.episodes
                for must_teach in episode.must_teach
            ),
        },
        "teaching_script": {
            "teaching_script",
            *(item.clause_id for item in script.clauses),
        },
        "interaction_plan": {
            "interaction_plan",
            "transfer_item",
            *(item.interaction_id for item in plan.interactions),
            *(
                option.option_id
                for interaction in plan.interactions
                for option in interaction.options
            ),
            *(item.option_id for item in plan.transfer_item.options),
        },
        "performance_score": {
            "performance_score",
            *(item.cue_id for item in score.cues),
            *(item.board_object_id for item in score.board_objects),
            *(item.transition_id for item in score.overlay_transitions),
        },
        "simulation_report": {
            "simulation_report",
            *(item.episode_id for item in report.episode_results),
        },
    }
    for finding in decision.findings:
        if finding.artifact_id not in artifact_ids[finding.artifact_type]:
            _fail(
                "review_evidence_invalid",
                finding.finding_id,
                "Review finding must cite an existing concrete artifact ID.",
            )
        if (
            _REVIEW_ROLE_ORDER[finding.responsible_role]
            > _ARTIFACT_OWNER_ORDER[finding.artifact_type]
        ):
            _fail(
                "review_responsibility_invalid",
                finding.finding_id,
                "Review finding must route to the earliest responsible role.",
            )

    _validate_review_dependency_metadata(decision)

    novice_gate_failed = bool(report.blocking_findings) or any(
        not (
            result.can_identify_attention_target
            and result.can_explain_decision
            and result.can_execute_action
            and result.can_use_result_to_continue
        )
        for result in report.episode_results
    )
    if decision.status == "approved" and novice_gate_failed:
        _fail(
            "review_non_compensable_gate_invalid",
            "review",
            "Review cannot approve when a non-compensable novice gate fails.",
        )


def _validate_review_dependency_metadata(
    decision: LessonReviewDecision,
) -> None:
    material_findings = [
        item for item in decision.findings if item.severity != "polish"
    ]
    for finding in decision.findings:
        expected_invalidated: List[str] = []
        if finding.severity != "polish":
            responsible_artifact = _REPAIR_ROLE_ARTIFACT[
                finding.responsible_role
            ]
            start_index = (
                _ARTIFACT_OWNER_ORDER[responsible_artifact] + 1
            )
            expected_invalidated = list(
                _ARTIFACT_DEPENDENCY_ORDER[start_index:]
            )
        if finding.invalidated_downstream_artifacts != expected_invalidated:
            _fail(
                "review_dependency_invalid",
                finding.finding_id,
                "Invalidated artifacts must equal the complete ordered downstream suffix.",
            )

    expected_retained: List[str] = []
    if material_findings:
        earliest_role = min(
            material_findings,
            key=lambda item: _REVIEW_ROLE_ORDER[item.responsible_role],
        ).responsible_role
        earliest_artifact = _REPAIR_ROLE_ARTIFACT[earliest_role]
        expected_retained = list(
            _ARTIFACT_DEPENDENCY_ORDER[
                : _ARTIFACT_OWNER_ORDER[earliest_artifact]
            ]
        )
    if decision.retained_artifacts != expected_retained:
        _fail(
            "review_dependency_invalid",
            "review",
            "Retained artifacts must equal the complete ordered upstream prefix.",
        )


def blocking_signature(decision: LessonReviewDecision) -> str:
    _require_exact(decision, LessonReviewDecision, "decision")
    canonical = sorted(
        set(
            (
                item.severity,
                item.artifact_type,
                item.artifact_id,
                item.criterion,
                item.responsible_role,
            )
            for item in decision.findings
            if item.severity in {"blocking", "material"}
        )
    )
    payload = json.dumps(
        canonical,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_prepared_lesson(
    prepared: PreparedLesson,
    teaching_route: FrozenTeachingRoute,
    problem_targets: ProblemTargets,
    active_versions: Optional[Dict[str, int]] = None,
) -> None:
    _require_exact(prepared, PreparedLesson, "prepared")
    _require_exact(teaching_route, FrozenTeachingRoute, "teaching_route")
    _validate_problem_targets(problem_targets)
    if (
        not prepared.rubric_version.strip()
        or prepared.rubric_version != PEDAGOGY_RUBRIC_VERSION
    ):
        _fail(
            "rubric_version_invalid",
            "prepared_lesson",
            "Prepared lesson rubric version does not match the current rubric.",
        )
    validate_solution_trace(prepared.solution_trace, teaching_route)
    validate_reasoning_trajectory(
        prepared.reasoning_trajectory, prepared.solution_trace
    )
    validate_teaching_script(
        prepared.teaching_script, prepared.reasoning_trajectory
    )
    validate_interaction_plan(
        prepared.interaction_plan,
        prepared.reasoning_trajectory,
        prepared.teaching_script,
    )
    validate_performance_score(
        prepared.performance_score,
        problem_targets,
        prepared.teaching_script,
        prepared.interaction_plan,
    )
    validate_simulation_report(
        prepared.simulation_report,
        prepared.reasoning_trajectory,
        prepared.interaction_plan,
    )
    _validate_artifact_history(prepared, active_versions)
    validate_review_decision(
        prepared.review,
        prepared.solution_trace,
        prepared.reasoning_trajectory,
        prepared.teaching_script,
        prepared.interaction_plan,
        prepared.performance_score,
        prepared.simulation_report,
    )
    if prepared.review.status != "approved":
        _fail(
            "review_approval_invalid",
            "review",
            "Prepared lesson requires an approved review.",
        )


def _validate_artifact_history(
    prepared: PreparedLesson,
    active_versions: Optional[Dict[str, int]],
) -> None:
    if not 0 <= prepared.repair_count <= MAX_REPAIR_CYCLES:
        _fail(
            "artifact_history_invalid",
            "repair_count",
            "Repair count exceeds the preparation convergence budget.",
        )

    history = prepared.artifact_history
    initial_count = len(_ARTIFACT_DEPENDENCY_ORDER)
    initial = history[:initial_count]
    if len(initial) != initial_count or [
        item.artifact_type for item in initial
    ] != list(_ARTIFACT_DEPENDENCY_ORDER):
        _fail(
            "artifact_history_invalid",
            "artifact_history",
            "Artifact history must begin with the complete ordered initial build.",
        )
    current_versions = {
        artifact_type: 1 for artifact_type in _ARTIFACT_DEPENDENCY_ORDER
    }
    for item in initial:
        if (
            item.version != 1
            or item.responsible_role
            != _ARTIFACT_HISTORY_ROLES[item.artifact_type]
        ):
            _fail(
                "artifact_history_invalid",
                item.artifact_type,
                "Initial artifact revisions must be authoritative version one.",
            )

    cursor = initial_count
    repairable = _ARTIFACT_DEPENDENCY_ORDER[:-1]
    for _ in range(prepared.repair_count):
        if cursor >= len(history) or history[cursor].artifact_type not in repairable:
            _fail(
                "artifact_history_invalid",
                "artifact_history",
                "Each repair cycle must start at one repairable artifact.",
            )
        start_index = _ARTIFACT_OWNER_ORDER[
            history[cursor].artifact_type
        ]
        expected_types = _ARTIFACT_DEPENDENCY_ORDER[start_index:]
        segment = history[cursor : cursor + len(expected_types)]
        if [item.artifact_type for item in segment] != list(expected_types):
            _fail(
                "artifact_history_invalid",
                "artifact_history",
                "Each repair cycle must be one complete ordered dependency suffix.",
            )
        for item, artifact_type in zip(segment, expected_types):
            expected_version = current_versions[artifact_type] + 1
            if (
                item.version != expected_version
                or item.responsible_role
                != _ARTIFACT_HISTORY_ROLES[artifact_type]
            ):
                _fail(
                    "artifact_history_invalid",
                    artifact_type,
                    "Repair revisions must increment once under the authoritative role.",
                )
            current_versions[artifact_type] = expected_version
        cursor += len(expected_types)
    if cursor != len(history):
        _fail(
            "artifact_history_invalid",
            "artifact_history",
            "Artifact history contains revisions outside declared repair cycles.",
        )

    if active_versions is not None:
        if type(active_versions) is not dict or active_versions != current_versions:
            _fail(
                "artifact_history_invalid",
                "active_versions",
                "Current active artifact versions must match history maxima.",
            )
