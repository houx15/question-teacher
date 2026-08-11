"""Deterministic cross-artifact validation for prepared lessons."""

import hashlib
import json
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Set, Tuple, Union

from app.math_content import (
    contains_explicit_choice_answer_leak,
    contains_internal_control_syntax,
    contains_math_markup,
    is_valid_generated_display_content,
    normalize_answer_leak_text,
    normalize_grounded_choice_option_label,
    normalize_reference_text,
)
from app.pedagogy_rubric import PEDAGOGY_RUBRIC_VERSION
from app.preparation_models import (
    InteractionPlan,
    LessonReviewDecision,
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
    if normalize_reference_text(trace.reference_conclusion) != normalize_reference_text(
        teaching_route.final_conclusion
    ):
        _fail(
            "trace_conclusion_mismatch",
            "solution_trace",
            "Solution trace conclusion does not match the frozen teaching route.",
        )

    assumption_ids = {item.assumption_id for item in trace.assumptions}
    route_step_ids = {item["step_id"] for item in route_payload["steps"]}
    anchors = [item.source_anchor for item in trace.assumptions]
    anchors.extend(item.source_anchor for item in trace.source_steps)
    for anchor in anchors:
        # Only frozen-route IDs have an authoritative ID set in this API.
        # Problem, answer, and solution IDs remain structurally opaque here.
        consistent = (
            anchor.source_kind != "verified_route"
            or anchor.source_id in route_step_ids
        )
        if not consistent:
            _fail(
                "trace_source_anchor_invalid",
                anchor.source_id,
                "Source anchor is inconsistent with its declared source kind.",
            )
    for step in trace.source_steps:
        missing = set(step.assumption_ids_used) - assumption_ids
        if missing:
            _fail(
                "trace_assumption_missing",
                step.source_step_id,
                "Trace step references an undeclared assumption.",
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
    last_position = -1
    for episode in trajectory.episodes:
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
    must_teach_owner = {
        item.must_teach_id: episode.episode_id
        for episode in trajectory.episodes
        for item in episode.must_teach
    }
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
    correct_forms = {
        normalize_answer_leak_text(correct.display_text),
        normalize_answer_leak_text(correct.canonical_answer),
    }
    correct_forms.discard("")
    for option in interaction.options:
        if option.option_id == interaction.correct_option_id:
            continue
        wrong_visible = (
            normalize_answer_leak_text(option.display_text),
            normalize_answer_leak_text(option.canonical_answer),
        )
        if any(form and form in item for form in correct_forms for item in wrong_visible):
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
    semantic_ids = set(episode_ids) | set(clauses)
    semantic_ids.update(
        item.must_teach_id
        for episode in trajectory.episodes
        for item in episode.must_teach
    )
    for interaction in plan.interactions:
        semantic_ids.add(interaction.interaction_id)
        semantic_ids.update(item.option_id for item in interaction.options)
        after = clauses.get(interaction.after_clause_id)
        resume = clauses.get(interaction.resume_clause_id)
        if (
            interaction.episode_id not in episode_ids
            or after is None
            or resume is None
            or after.episode_id != interaction.episode_id
            or resume.episode_id != interaction.episode_id
            or clause_positions[resume.clause_id] <= clause_positions[after.clause_id]
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
        if (
            interaction.correct_option_id in interaction.concealed_targets
            or interaction.resume_clause_id in interaction.concealed_targets
            or any(item not in semantic_ids for item in interaction.concealed_targets)
            or _leaks_correct_answer(interaction)
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
                and normalize_reference_text(bound.action.content or "")
                != normalize_reference_text(
                    board_objects[bound.action.target].content
                )
            ):
                _fail(
                    "visual_target_invalid",
                    bound.clause_id,
                    "Board write content does not match its declared object.",
                )
    try:
        validate_visual_action_references(
            [_ScoreMoment(score.cues)],
            problem_targets,
            set(board_objects),
        )
    except VisualActionValidationError as error:
        _fail(error.code, "performance_score", error.detail)

    available_by_clause = {}
    referenced_math = set()
    for clause in script.clauses:
        referenced_math.update(
            normalize_reference_text(item)
            for item in clause.math_references
        )
        available_by_clause[clause.clause_id] = set(referenced_math)
    problem_content = {
        item.target_id: normalize_reference_text(item.math_text)
        for item in problem_targets
    }
    current_board_content = {}
    for cue in score.cues:
        for bound in (*cue.lead_actions, *cue.start_actions, *cue.end_actions):
            action = bound.action
            available = available_by_clause[bound.clause_id]
            if action.surface == "problem":
                target_content = problem_content[action.target]
                if not any(
                    target_content == item or target_content in item
                    for item in available
                ):
                    _fail(
                        "visual_action_too_early",
                        bound.clause_id,
                        "Problem visual target appears before its bound clause references it.",
                    )
            if action.type in {"write", "transform"}:
                normalized_content = normalize_reference_text(action.content or "")
                if normalized_content not in available:
                    _fail(
                        "visual_action_too_early",
                        bound.clause_id,
                        "Visual content appears before its bound clause references it.",
                    )
                current_board_content[action.target] = action.content or ""
            elif (
                action.surface == "board"
                and action.target in current_board_content
                and action.type not in {"clear_focus", "fade"}
                and normalize_reference_text(current_board_content[action.target])
                not in available
            ):
                _fail(
                    "visual_action_too_early",
                    bound.clause_id,
                    "Board visual target appears before its bound clause references it.",
                )
            if (
                action.type == "annotate"
                and action.annotation == "label"
                and action.content is not None
                and len(current_board_content) == 1
                and action.target in current_board_content
                and normalize_reference_text(action.content)
                == normalize_reference_text(current_board_content[action.target])
            ):
                _fail(
                    "non_discriminating_emphasis",
                    bound.clause_id,
                    "Label annotation exactly repeats the sole visible target content.",
                )

    transitions_by_position = {}
    previous_position = -1
    for transition in score.overlay_transitions:
        position = script_positions.get(transition.after_clause_id)
        if position is None or position < previous_position:
            _fail(
                "overlay_transition_invalid",
                transition.transition_id,
                "Overlay transition references an unknown or out-of-order clause.",
            )
        previous_position = position
        transitions_by_position.setdefault(position, []).append(transition)
    actions_by_position = {position: [] for position in range(len(script_ids))}
    for cue in score.cues:
        for bound in (*cue.lead_actions, *cue.start_actions, *cue.end_actions):
            actions_by_position[script_positions[bound.clause_id]].append(
                bound.action
            )

    stack: List[str] = []
    for position in range(len(script_ids)):
        expected_layer = stack[-1] if stack else "base"
        for action in actions_by_position[position]:
            if action.surface != "board" or action.target not in board_objects:
                continue
            if board_objects[action.target].layer != expected_layer:
                _fail(
                    "overlay_transition_invalid",
                    script_ids[position],
                    "Board action continues a different layer before overlay return.",
                )
        for transition in transitions_by_position.get(position, []):
            if transition.action == "enter":
                stack.append(transition.layer)
            elif not stack or stack.pop() != transition.layer:
                _fail(
                    "overlay_transition_invalid",
                    transition.transition_id,
                    "Overlay transitions are not deterministically nested.",
                )
    if stack:
        _fail(
            "overlay_transition_invalid",
            "performance_score",
            "Overlay transition does not return to the base sequence.",
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
    private_tokens = {"correct_option_id", "canonical_answer"}
    for interaction in plan.interactions:
        correct = next(
            item
            for item in interaction.options
            if item.option_id == interaction.correct_option_id
        )
        private_tokens.update(
            {
                interaction.correct_option_id,
                correct.display_text,
                correct.canonical_answer,
                interaction.correct_feedback,
            }
        )
    normalized_private = {
        normalize_answer_leak_text(item) for item in private_tokens
    }
    for result in report.interaction_results:
        normalized = normalize_answer_leak_text(result)
        if any(item and item in normalized for item in normalized_private):
            _fail(
                "simulation_private_answer_invalid",
                "simulation_report",
                "Simulation interaction result contains private answer material.",
            )


def validate_review_decision(decision: LessonReviewDecision) -> None:
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


def blocking_signature(decision: LessonReviewDecision) -> str:
    _require_exact(decision, LessonReviewDecision, "decision")
    canonical = sorted(
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
    validate_review_decision(prepared.review)
