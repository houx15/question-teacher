"""Deterministic cross-artifact validation for prepared lessons."""

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple, Union

from pydantic import ValidationError

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
from app.math_expression import MAX_STRICT_MATH_EXPRESSION_LENGTH
from app.math_speech import (
    MathSpeechError,
    contains_deterministic_math_speech,
    validate_display_spoken_alignment,
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
    TeachingProgression,
    TeachingScript,
)
from app.schemas import ProblemFocusTarget, SyncVisualAction
from app.teaching_route import FrozenTeachingRoute
from app.teaching_progression_validation import (
    TeachingProgressionValidationError,
    validate_teaching_progression,
)


ProblemTargets = Union[List[ProblemFocusTarget], Tuple[ProblemFocusTarget, ...]]
ActionKey = Tuple[str, str]
MAX_PERFORMANCE_CLAUSES = MAX_PREPARATION_ITEMS
MAX_PERFORMANCE_ACTIONS = 2048
MAX_PERFORMANCE_MATH_REFERENCES = 2048
_ARTIFACT_DEPENDENCY_ORDER = (
    "solution_trace",
    "reasoning_trajectory",
    "teaching_progression",
    "interaction_plan",
    "teaching_script",
    "performance_score",
    "simulation_report",
)
_ARTIFACT_OWNER_ORDER = {
    artifact_type: index
    for index, artifact_type in enumerate(_ARTIFACT_DEPENDENCY_ORDER)
}
_ARTIFACT_HISTORY_ROLES = {
    "solution_trace": "reference_analyst",
    "reasoning_trajectory": "teaching_designer",
    "teaching_progression": "teaching_designer",
    "interaction_plan": "interaction_designer",
    "teaching_script": "script_teacher",
    "performance_score": "classroom_director",
    "simulation_report": "student_simulator",
}
MAX_REPAIR_CYCLES = 8


def build_structured_performance_score(
    progression: TeachingProgression,
    script: TeachingScript,
) -> PerformanceScore:
    """Build the reliable tablet board lifecycle from authored lesson semantics."""
    _require_exact(progression, TeachingProgression, "progression")
    _require_exact(script, TeachingScript, "script")
    steps = {item.step_id: item for item in progression.steps}
    clauses_by_step: Dict[str, List[object]] = {}
    for clause in script.clauses:
        if clause.lesson_step_id is None:
            _fail(
                "structured_action_step_invalid",
                clause.clause_id,
                "Structured clauses require a teaching step.",
            )
        clauses_by_step.setdefault(clause.lesson_step_id, []).append(clause)

    board_objects = []
    cues = []
    board_index = 0
    for clause in script.clauses:
        step_id = clause.lesson_step_id
        step = steps.get(step_id or "")
        if step is None:
            _fail(
                "structured_action_step_invalid",
                clause.clause_id,
                "Structured clauses require a known teaching step.",
            )
        step_clauses = clauses_by_step[step_id]
        start_actions = []
        end_actions = []
        if clause is step_clauses[0]:
            start_actions.extend(
                [
                    {
                        "clause_id": clause.clause_id,
                        "action": {
                            "surface": "board",
                            "type": "reveal_step_header",
                            "target": step_id,
                            "teaching_step_id": step_id,
                            "step_label": step.directory_label,
                        },
                    },
                    {
                        "clause_id": clause.clause_id,
                        "action": {
                            "surface": "board",
                            "type": "scroll_to_step",
                            "target": step_id,
                            "teaching_step_id": step_id,
                        },
                    },
                ]
            )
        authored_board_content = list(clause.math_references)
        if not authored_board_content and clause.display_text is not None:
            authored_board_content.append(clause.display_text)
        for reference_index, reference in enumerate(authored_board_content):
            board_index += 1
            board_id = "board-main-%03d" % board_index
            line_role = (
                "knowledge_anchor"
                if clause is step_clauses[0] and reference_index == 0
                else "working"
            )
            board_objects.append(
                {
                    "board_object_id": board_id,
                    "content": reference,
                    "teaching_step_id": step_id,
                    "line_role": line_role,
                }
            )
            start_actions.append(
                {
                    "clause_id": clause.clause_id,
                    "action": {
                        "surface": "board",
                        "type": "write",
                        "target": board_id,
                        "content": reference,
                        "teaching_step_id": step_id,
                        "board_role": line_role,
                    },
                }
            )
        if clause is step_clauses[-1]:
            end_actions.append(
                {
                    "clause_id": clause.clause_id,
                    "action": {
                        "surface": "board",
                        "type": "complete_step",
                        "target": step_id,
                        "teaching_step_id": step_id,
                    },
                }
            )
        cues.append(
            {
                "cue_id": "cue-main-%03d" % len(cues),
                "clause_ids": [clause.clause_id],
                "start_actions": start_actions,
                "end_actions": end_actions,
            }
        )

    support_index = 0
    for response in script.response_scripts:
        panel_id = None
        if response.classification == "incorrect":
            support_index += 1
            panel_id = "support-panel-%03d" % support_index
        for clause_index, clause in enumerate(response.clauses):
            start_actions = []
            end_actions = []
            if response.classification == "incorrect":
                board_id = "board-support-%03d-%03d" % (
                    support_index,
                    clause_index,
                )
                content = clause.display_text or clause.math_references[0]
                board_objects.append(
                    {
                        "board_object_id": board_id,
                        "content": content,
                        "teaching_step_id": clause.lesson_step_id,
                        "line_role": "support",
                    }
                )
                if clause_index == 0:
                    start_actions.append(
                        {
                            "clause_id": clause.clause_id,
                            "action": {
                                "surface": "board",
                                "type": "open_supporting_explanation",
                                "target": panel_id,
                                "teaching_step_id": clause.lesson_step_id,
                            },
                        }
                    )
                start_actions.append(
                    {
                        "clause_id": clause.clause_id,
                        "action": {
                            "surface": "board",
                            "type": "write",
                            "target": board_id,
                            "content": content,
                            "teaching_step_id": clause.lesson_step_id,
                            "board_role": "support",
                        },
                    }
                )
                if clause_index == len(response.clauses) - 1:
                    end_actions.extend(
                        [
                            {
                                "clause_id": clause.clause_id,
                                "action": {
                                    "surface": "board",
                                    "type": "close_supporting_explanation",
                                    "target": panel_id,
                                    "teaching_step_id": clause.lesson_step_id,
                                },
                            },
                            {
                                "clause_id": clause.clause_id,
                                "action": {
                                    "surface": "board",
                                    "type": "scroll_to_step",
                                    "target": clause.lesson_step_id,
                                    "teaching_step_id": clause.lesson_step_id,
                                },
                            },
                        ]
                    )
            cues.append(
                {
                    "cue_id": "cue-response-%03d" % len(cues),
                    "clause_ids": [clause.clause_id],
                    "start_actions": start_actions,
                    "end_actions": end_actions,
                }
            )

    return PerformanceScore.model_validate(
        {
            "cues": cues,
            "board_objects": board_objects,
            "overlay_transitions": [],
        }
    )


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


def _defensively_revalidate(
    value: object,
    expected: type,
    label: str,
    code: str,
    detail: str,
) -> object:
    try:
        _require_exact(value, expected, label)
        return expected.model_validate(value.model_dump(mode="python"))
    except (ValidationError, TypeError):
        _fail(code, label, detail)


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
        for item in episode.must_teach:
            if (
                item.student_display_evidence is None
                or item.student_spoken_evidence is None
            ):
                _fail(
                    "must_teach_evidence_missing",
                    item.must_teach_id,
                    "Current must-teach items require student-visible display and spoken evidence.",
                )
            if not _contains_semantic_anchor(
                item.student_display_evidence, item.content
            ):
                _fail(
                    "must_teach_content_anchor_missing",
                    item.must_teach_id,
                    "Must-teach display evidence must preserve its authored content as a semantic anchor.",
                )
            try:
                validate_display_spoken_alignment(
                    item.student_display_evidence,
                    item.student_spoken_evidence,
                )
            except MathSpeechError:
                _fail(
                    "must_teach_evidence_alignment_invalid",
                    item.must_teach_id,
                    "Must-teach display and spoken evidence are not aligned.",
                )
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
    progression: TeachingProgression,
    interaction_plan: InteractionPlan,
) -> None:
    script = _defensively_revalidate(
        script,
        TeachingScript,
        "teaching_script",
        "teaching_script_content_invalid",
        "Teaching script content failed defensive model validation.",
    )
    trajectory = _defensively_revalidate(
        trajectory,
        ReasoningTrajectory,
        "reasoning_trajectory",
        "reasoning_trajectory_content_invalid",
        "Reasoning trajectory content failed defensive model validation.",
    )
    progression = _defensively_revalidate(
        progression,
        TeachingProgression,
        "teaching_progression",
        "teaching_progression_content_invalid",
        "Teaching progression content failed defensive model validation.",
    )
    interaction_plan = _defensively_revalidate(
        interaction_plan,
        InteractionPlan,
        "interaction_plan",
        "interaction_plan_content_invalid",
        "Interaction plan content failed defensive model validation.",
    )
    episode_positions = {
        episode.episode_id: index
        for index, episode in enumerate(trajectory.episodes)
    }
    step_positions = {
        step.step_id: index for index, step in enumerate(progression.steps)
    }
    step_by_id = {step.step_id: step for step in progression.steps}
    episode_steps = {
        episode_id: step.step_id
        for step in progression.steps
        for episode_id in step.episode_ids
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
    last_step_position = -1
    covered_steps = set()
    for clause in script.clauses:
        if clause.lesson_step_id is None:
            _fail(
                "clause_lesson_step_missing",
                clause.clause_id,
                "Current script clauses require a lesson-step binding.",
            )
        if clause.display_text is None:
            _fail(
                "clause_display_missing",
                clause.clause_id,
                "Current script clauses require display text.",
            )
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
        if clause.lesson_step_id not in step_positions:
            _fail(
                "clause_lesson_step_missing",
                clause.clause_id,
                "Script clause references an unknown lesson step.",
            )
        step = step_by_id[clause.lesson_step_id]
        if clause.episode_id not in step.episode_ids:
            _fail(
                "clause_episode_step_mismatch",
                clause.clause_id,
                "Script clause episode does not belong to its lesson step.",
            )
        step_position = step_positions[clause.lesson_step_id]
        if step_position < last_step_position:
            _fail(
                "clause_step_order_invalid",
                clause.clause_id,
                "Script clause order violates teaching progression order.",
            )
        last_step_position = step_position
        covered_steps.add(clause.lesson_step_id)
        _validate_authored_clause_language(clause)
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
    for step in progression.steps:
        if step.step_id not in covered_steps:
            _fail(
                "progression_step_uncovered",
                step.step_id,
                "Every teaching-progression step requires a main script clause.",
            )
    for must_teach_id in must_teach_owner:
        if must_teach_id not in covered:
            _fail(
                "must_teach_uncovered",
                must_teach_id,
                "Must-teach item has no script-clause evidence.",
            )
    _validate_must_teach_script_evidence(script, trajectory)
    _validate_authored_student_interactions(script, interaction_plan)
    _validate_response_scripts(
        script,
        interaction_plan,
        episode_steps,
    )


def _contains_evidence(container: str, evidence: str) -> bool:
    return contains_bounded_answer_token(container, evidence) or (
        normalize_cross_artifact_math_identity(evidence)
        in normalize_cross_artifact_math_identity(container)
    )


def _contains_semantic_anchor(container: str, anchor: str) -> bool:
    """Match an authored prose anchor through presentation-only variation."""
    normalized_anchor = normalize_answer_leak_text(anchor)
    return _contains_evidence(container, anchor) or (
        bool(normalized_anchor)
        and normalized_anchor in normalize_answer_leak_text(container)
    )


def _validate_must_teach_script_evidence(
    script: TeachingScript,
    trajectory: ReasoningTrajectory,
) -> None:
    items = {
        item.must_teach_id: item
        for episode in trajectory.episodes
        for item in episode.must_teach
    }
    clauses_by_ref: Dict[str, List[object]] = {item_id: [] for item_id in items}
    for clause in script.clauses:
        for item_id in clause.must_teach_refs:
            clauses_by_ref.setdefault(item_id, []).append(clause)
    for item_id, item in items.items():
        if (
            item.student_display_evidence is None
            or item.student_spoken_evidence is None
        ):
            _fail(
                "must_teach_evidence_missing",
                item_id,
                "Current must-teach items require student-visible display and spoken evidence.",
            )
        if not _contains_semantic_anchor(
            item.student_display_evidence, item.content
        ):
            _fail(
                "must_teach_content_anchor_missing",
                item_id,
                "Must-teach display evidence must preserve its authored content as a semantic anchor.",
            )
        if not any(
            _contains_evidence(
                clause.display_text or "", item.student_display_evidence
            )
            and _contains_evidence(
                clause.spoken_text, item.student_spoken_evidence
            )
            for clause in clauses_by_ref.get(item_id, [])
        ):
            _fail(
                "must_teach_evidence_missing",
                item_id,
                "Must-teach evidence must appear in its bound student-visible script clause.",
            )
        try:
            validate_display_spoken_alignment(
                item.student_display_evidence,
                item.student_spoken_evidence,
            )
        except MathSpeechError:
            _fail(
                "must_teach_evidence_alignment_invalid",
                item_id,
                "Must-teach display and spoken evidence are not aligned.",
            )


def _validate_authored_student_interactions(
    script: TeachingScript,
    plan: InteractionPlan,
) -> None:
    authored = {item.interaction_id: item for item in script.interaction_scripts}
    planned = {item.interaction_id: item for item in plan.interactions}
    if set(authored) != set(planned):
        _fail(
            "interaction_script_coverage_invalid",
            "teaching_script",
            "Teaching script must author every planned interaction exactly once.",
        )
    for interaction_id, private in planned.items():
        public = authored[interaction_id]
        public_options = {item.option_id: item for item in public.options}
        private_options = {item.option_id: item for item in private.options}
        if set(public_options) != set(private_options):
            _fail(
                "interaction_script_coverage_invalid",
                interaction_id,
                "Authored option IDs must exactly cover the private answer intent.",
            )
        labels = [item.label for item in public.options]
        if (
            not is_valid_generated_display_content(public.prompt)
            or contains_internal_control_syntax(public.prompt)
            or (
                public.hint is not None
                and (
                    not is_valid_generated_display_content(public.hint)
                    or contains_internal_control_syntax(public.hint)
                    or contains_math_markup(public.hint)
                )
            )
        ):
            _fail(
                "interaction_script_content_invalid",
                interaction_id,
                "Authored interaction prompt or spoken hint is unsafe.",
            )
        if not _labels_are_unique(labels) or any(
            not is_valid_generated_display_content(label) for label in labels
        ):
            _fail(
                "choice_formula_invalid",
                interaction_id,
                "Authored interaction options must be unique safe display content.",
            )
        correct = private_options[private.correct_option_id]
        correct_label = public_options[private.correct_option_id].label
        visible = "|".join((public.prompt, public.hint or ""))
        if contains_explicit_choice_answer_leak(
            visible, correct.option_id, correct_label
        ) or contains_explicit_choice_answer_leak(
            visible, correct.option_id, correct.canonical_answer
        ):
            _fail(
                "interaction_answer_leakage",
                interaction_id,
                "Authored prompt or hint exposes the private correct answer binding.",
            )

    transfer = script.transfer_script
    if transfer is None:
        _fail(
            "transfer_script_missing",
            "teaching_script",
            "Current teaching script must own final transfer text.",
        )
    planned_transfer = plan.transfer_item
    public_transfer_options = {item.option_id: item for item in transfer.options}
    private_transfer_options = {
        item.option_id: item for item in planned_transfer.options
    }
    if set(public_transfer_options) != set(private_transfer_options):
        _fail(
            "transfer_script_coverage_invalid",
            "transfer_item",
            "Authored transfer option IDs must exactly cover private answer intent.",
        )
    transfer_labels = [item.label for item in transfer.options]
    transfer_display_values = [transfer.problem_text, *transfer_labels]
    if (
        not _labels_are_unique(transfer_labels)
        or any(
            not is_valid_generated_display_content(item)
            or contains_internal_control_syntax(item)
            for item in transfer_display_values
        )
    ):
        _fail(
            "choice_formula_invalid",
            "transfer_item",
            "Authored transfer problem and options must be safe display content.",
        )
    transfer_spoken_values = [
        transfer.method_signal,
        *(item.feedback for item in transfer.options),
    ]
    if any(
        contains_math_markup(item) or contains_internal_control_syntax(item)
        for item in transfer_spoken_values
    ):
        _fail(
            "transfer_script_content_invalid",
            "transfer_item",
            "Authored transfer hints and feedback must be safe spoken content.",
        )


def _validate_authored_clause_language(clause: object) -> None:
    if contains_math_markup(clause.spoken_text) or contains_internal_control_syntax(
        clause.spoken_text
    ):
        _fail(
            "spoken_markup_invalid",
            clause.clause_id,
            "Spoken text contains math or internal visual markup.",
        )
    if clause.display_text is None:
        _fail(
            "clause_display_missing",
            clause.clause_id,
            "Current script clauses require display text.",
        )
    if not is_valid_generated_display_content(clause.display_text):
        _fail(
            "display_content_invalid",
            clause.clause_id,
            "Display text is not safe bounded generated content.",
        )
    try:
        validate_display_spoken_alignment(
            clause.display_text,
            clause.spoken_text,
        )
    except MathSpeechError as error:
        _fail(
            error.code,
            clause.clause_id,
            "Displayed mathematics is not safely aligned with spoken text.",
        )


def _contains_private_canonical_answer(
    visible: str,
    canonical_answer: str,
) -> bool:
    if contains_bounded_answer_token(visible, canonical_answer):
        return True
    compact_visible = re.sub(r"\s+", "", visible)
    compact_canonical = re.sub(r"\s+", "", canonical_answer)
    if contains_bounded_answer_token(
        compact_visible,
        compact_canonical,
    ):
        return True
    if contains_deterministic_math_speech(
        canonical_answer,
        visible,
    ):
        return True
    fraction_form = _slash_fractions_to_latex(canonical_answer)
    return fraction_form is not None and contains_deterministic_math_speech(
        fraction_form,
        visible,
    )


def _canonical_is_public(option: object) -> bool:
    display_identity = normalize_cross_artifact_math_identity(
        option.display_text
    )
    canonical_identity = normalize_cross_artifact_math_identity(
        option.canonical_answer
    )
    return bool(
        display_identity
        and canonical_identity
        and display_identity == canonical_identity
    )


_SLASH_FRACTION_TOKEN = re.compile(
    r"(?<![A-Za-z0-9_.])(\d+)\s*/\s*(\d+)(?![A-Za-z0-9_.])"
)


def _slash_fractions_to_latex(value: str) -> Optional[str]:
    if len(value) > MAX_STRICT_MATH_EXPRESSION_LENGTH:
        return None
    pieces = []
    cursor = 0
    matched = False
    for match in _SLASH_FRACTION_TOKEN.finditer(value):
        denominator = match.group(2)
        if not denominator.strip("0"):
            return None
        pieces.extend(
            (
                value[cursor : match.start()],
                r"\frac{%s}{%s}" % (match.group(1), denominator),
            )
        )
        cursor = match.end()
        matched = True
    if not matched:
        return None
    pieces.append(value[cursor:])
    transformed = "".join(pieces)
    if len(transformed) > MAX_STRICT_MATH_EXPRESSION_LENGTH:
        return None
    return transformed


def _all_occurrence_intervals(
    value: str,
    fragment: str,
) -> List[Tuple[int, int]]:
    if not fragment:
        return []
    intervals = []
    search_from = 0
    while search_from <= len(value) - len(fragment):
        start = value.find(fragment, search_from)
        if start < 0:
            break
        intervals.append((start, start + len(fragment)))
        search_from = start + 1
    return intervals


def _has_non_overlapping_interval_pair(
    first: Sequence[Tuple[int, int]],
    second: Sequence[Tuple[int, int]],
) -> bool:
    if not first or not second:
        return False
    return (
        first[0][1] <= second[-1][0]
        or second[0][1] <= first[-1][0]
    )


def _has_locally_independent_occurrence(
    own: Sequence[Tuple[int, int]],
    other: Sequence[Tuple[int, int]],
) -> bool:
    if not own:
        return False
    if not other:
        return True
    other_index = 0
    for start, end in own:
        while (
            other_index < len(other)
            and other[other_index][1] <= start
        ):
            other_index += 1
        if other_index == len(other) or other[other_index][0] >= end:
            return True
    return False


def _response_evidence_units(response: object) -> List[str]:
    units = []
    seen = set()
    for clause in response.clauses:
        for value in (clause.display_text or "", clause.spoken_text):
            normalized = normalize_answer_leak_text(value)
            if normalized and normalized not in seen:
                seen.add(normalized)
                units.append(normalized)
    return units


def _has_distinct_response_anchor_evidence(
    units: Sequence[str],
    first_anchor: str,
    second_anchor: str,
) -> bool:
    first_units = set()
    second_units = set()
    for index, unit in enumerate(units):
        first = _all_occurrence_intervals(unit, first_anchor)
        second = _all_occurrence_intervals(unit, second_anchor)
        if _has_non_overlapping_interval_pair(first, second):
            return True
        if _has_locally_independent_occurrence(first, second):
            first_units.add(index)
        if _has_locally_independent_occurrence(second, first):
            second_units.add(index)
    return bool(
        first_units
        and second_units
        and len(first_units | second_units) > 1
    )


def _validate_response_scripts(
    script: TeachingScript,
    interaction_plan: InteractionPlan,
    episode_steps: Dict[str, str],
) -> None:
    interactions = {
        item.interaction_id: item for item in interaction_plan.interactions
    }
    expected = {
        (interaction.interaction_id, option.option_id)
        for interaction in interaction_plan.interactions
        for option in interaction.options
    }
    actual: List[Tuple[str, str]] = []
    response_by_binding = {}
    for response in script.response_scripts:
        interaction = interactions.get(response.interaction_id)
        if interaction is None or response.option_id not in {
            item.option_id for item in interaction.options
        }:
            _fail(
                "response_script_binding_invalid",
                response.response_id,
                "Response script must bind to an exact interaction option.",
            )
        binding = (response.interaction_id, response.option_id)
        actual.append(binding)
        response_by_binding[binding] = response
    if len(actual) != len(set(actual)) or set(actual) != expected:
        _fail(
            "response_script_coverage_invalid",
            "teaching_script",
            "Every interaction option requires exactly one response script.",
        )

    for interaction in interaction_plan.interactions:
        for option in interaction.options:
            response = response_by_binding[
                (interaction.interaction_id, option.option_id)
            ]
            is_correct = option.option_id == interaction.correct_option_id
            expected_classification = "correct" if is_correct else "incorrect"
            if response.classification != expected_classification:
                _fail(
                    "response_classification_invalid",
                    response.response_id,
                    "Response classification does not match the bound option.",
                )
            if is_correct:
                if response.depth != "brief":
                    _fail(
                        "response_depth_invalid",
                        response.response_id,
                        "Correct responses must be brief.",
                    )
                if response.error_code is not None:
                    _fail(
                        "response_error_code_invalid",
                        response.response_id,
                        "Correct responses cannot carry an error code.",
                    )
            elif response.depth != option.remediation_depth:
                _fail(
                    "response_depth_invalid",
                    response.response_id,
                    "Incorrect response depth must match its bound option intent.",
                )

            if not is_correct and response.error_code != option.error_code:
                _fail(
                    "response_error_code_invalid",
                    response.response_id,
                    "Incorrect response error code must match its bound option intent.",
                )

            if not is_correct and option.misconception is None:
                _fail(
                    "response_misconception_missing",
                    response.response_id,
                    "Current incorrect options require a misconception anchor.",
                )

            expected_step_id = episode_steps.get(interaction.episode_id)
            if expected_step_id is None:
                _fail(
                    "response_clause_step_invalid",
                    response.response_id,
                    "Interaction episode has no teaching-progression step.",
                )
            for clause in response.clauses:
                if clause.lesson_step_id is None:
                    _fail(
                        "clause_lesson_step_missing",
                        clause.clause_id,
                        "Current response clauses require a lesson-step binding.",
                    )
                if clause.display_text is None:
                    _fail(
                        "clause_display_missing",
                        clause.clause_id,
                        "Current response clauses require display text.",
                    )
                if (
                    clause.episode_id != interaction.episode_id
                    or clause.lesson_step_id != expected_step_id
                ):
                    _fail(
                        "response_clause_step_invalid",
                        clause.clause_id,
                        "Response clause must stay in the interaction episode step.",
                    )
                _validate_authored_clause_language(clause)

            if not is_correct:
                visible = "|".join(
                    "%s|%s" % (clause.display_text or "", clause.spoken_text)
                    for clause in response.clauses
                )
                misconception_anchor = normalize_answer_leak_text(
                    option.misconception or ""
                )
                correction_anchor = normalize_answer_leak_text(
                    interaction.incorrect_feedback_by_option.get(
                        option.option_id, ""
                    )
                )
                if not misconception_anchor:
                    _fail(
                        "response_semantic_anchor_missing",
                        response.response_id,
                        "Incorrect response must directly express its misconception anchor.",
                    )
                if correction_anchor and misconception_anchor == correction_anchor:
                    _fail(
                        "response_remediation_insufficient",
                        response.response_id,
                        "Incorrect response requires distinct reason and correction units.",
                    )
                evidence_units = _response_evidence_units(response)
                has_anchors = (
                    _has_distinct_response_anchor_evidence(
                        evidence_units,
                        misconception_anchor,
                        correction_anchor,
                    )
                    if correction_anchor
                    else any(
                        contains_bounded_answer_token(unit, misconception_anchor)
                        for unit in evidence_units
                    )
                    and any(
                        unit != misconception_anchor
                        and not contains_bounded_answer_token(
                            misconception_anchor, unit
                        )
                        for unit in evidence_units
                    )
                )
                if not has_anchors:
                    _fail(
                        "response_semantic_anchor_missing",
                        response.response_id,
                        "Incorrect response must directly express its misconception and correction anchors.",
                    )
                if any(
                    not _canonical_is_public(private_option)
                    and _contains_private_canonical_answer(
                        visible, private_option.canonical_answer
                    )
                    for private_option in interaction.options
                ):
                    _fail(
                        "response_private_answer_leakage",
                        response.response_id,
                        "Incorrect response exposes private canonical answer data.",
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
    visible_text = "|".join((interaction.prompt, interaction.hint or ""))
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
    visible_parts = [interaction.prompt, interaction.hint or ""]
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


def normalize_interaction_control_metadata(
    plan: InteractionPlan,
    trajectory: ReasoningTrajectory,
    script: TeachingScript,
) -> InteractionPlan:
    """Drop only control targets that are already visibly disclosed."""
    _require_exact(plan, InteractionPlan, "plan")
    _require_exact(trajectory, ReasoningTrajectory, "trajectory")
    _require_exact(script, TeachingScript, "script")
    payload = plan.model_dump(mode="python")
    changed = False
    for interaction, interaction_payload in zip(
        plan.interactions, payload["interactions"]
    ):
        registry = _concealed_registry(
            interaction.episode_id, trajectory, script
        )
        visible_parts = [interaction.prompt, interaction.hint or ""]
        visible_parts.extend(
            option.display_text
            for option in interaction.options
            if option.option_id != interaction.correct_option_id
        )
        normalized_targets = []
        for target_id in interaction.concealed_targets:
            if target_id == interaction.resume_clause_id:
                changed = True
                continue
            disclosed = target_id in registry and any(
                contains_bounded_answer_token(visible, content)
                for content in registry[target_id]
                if content
                for visible in visible_parts
            )
            if disclosed:
                changed = True
                continue
            normalized_targets.append(target_id)
        interaction_payload["concealed_targets"] = normalized_targets
    if not changed:
        return plan
    return InteractionPlan.model_validate(payload)


def normalize_performance_control_metadata(
    score: PerformanceScore,
    problem_targets: ProblemTargets,
    script: TeachingScript,
    progression: Optional[TeachingProgression] = None,
) -> PerformanceScore:
    """Conservatively align recoverable visual controls to authored speech."""
    _require_exact(score, PerformanceScore, "score")
    _validate_problem_targets(problem_targets)
    _require_exact(script, TeachingScript, "script")
    if progression is not None:
        _require_exact(progression, TeachingProgression, "progression")
    payload = score.model_dump(mode="python")
    if progression is not None:
        clauses = {
            clause.clause_id: (None, clause)
            for clause in script.clauses
        }
        clauses.update(
            {
                clause.clause_id: (response, clause)
                for response in script.response_scripts
                for clause in response.clauses
            }
        )
        board_objects = {
            item["board_object_id"]: item
            for item in payload["board_objects"]
            if item.get("teaching_step_id") is not None
            and item.get("line_role") is not None
        }
        written_targets = set()
        for cue in payload["cues"]:
            for phase in (
                "lead_actions",
                "start_actions",
                "end_actions",
            ):
                normalized_actions = []
                for bound in cue[phase]:
                    owner = clauses.get(bound["clause_id"])
                    if owner is None:
                        continue
                    response, clause = owner
                    action = bound["action"]
                    if action["surface"] == "problem":
                        problem_target = next(
                            (
                                item
                                for item in problem_targets
                                if item.target_id == action["target"]
                            ),
                            None,
                        )
                        if (
                            problem_target is not None
                            and any(
                                contains_normalized_cross_artifact_math_identity(
                                    reference,
                                    problem_target.math_text,
                                )
                                for reference in clause.math_references
                            )
                        ):
                            normalized_actions.append(bound)
                        continue
                    if action["type"] in {
                        "reveal_step_header",
                        "complete_step",
                        "scroll_to_step",
                        "open_supporting_explanation",
                        "close_supporting_explanation",
                    }:
                        normalized_actions.append(bound)
                        continue
                    if action["type"] == "write":
                        board_object = board_objects.get(action["target"])
                        normalized_content = normalize_cross_artifact_math_identity(
                            action.get("content") or ""
                        )
                        main_reference = any(
                            normalized_content
                            == normalize_cross_artifact_math_identity(reference)
                            for reference in clause.math_references
                        )
                        if clause.display_text is not None:
                            main_reference = main_reference or (
                                normalized_content
                                == normalize_cross_artifact_math_identity(
                                    clause.display_text
                                )
                            )
                        support_reference = (
                            response is not None
                            and action.get("board_role") == "support"
                            and normalized_content
                            == normalize_cross_artifact_math_identity(
                                clause.display_text or ""
                            )
                        )
                        if (
                            board_object is not None
                            and board_object.get("teaching_step_id")
                            == action.get("teaching_step_id")
                            == clause.lesson_step_id
                            and board_object.get("line_role")
                            == action.get("board_role")
                            and (main_reference or support_reference)
                        ):
                            normalized_actions.append(bound)
                            written_targets.add(action["target"])
                        continue
                    if action["target"] in board_objects:
                        normalized_actions.append(bound)
                cue[phase] = normalized_actions
        payload["board_objects"] = [
            item
            for item in payload["board_objects"]
            if item["board_object_id"] in written_targets
        ]
        transitions = payload["overlay_transitions"]
        zero_width = {
            (first["transition_id"], second["transition_id"])
            for first, second in zip(transitions, transitions[1:])
            if first["action"] == "enter"
            and second["action"] == "return"
            and first["layer"] == second["layer"]
            and first["after_clause_id"] == second["after_clause_id"]
        }
        dropped_transition_ids = {
            transition_id
            for pair in zero_width
            for transition_id in pair
        }
        payload["overlay_transitions"] = [
            item
            for item in transitions
            if item["transition_id"] not in dropped_transition_ids
        ]
        return PerformanceScore.model_validate(payload)
    clause_positions = {
        clause.clause_id: index
        for index, clause in enumerate(script.clauses)
    }
    cue_index_by_clause = {
        clause_id: cue_index
        for cue_index, cue in enumerate(payload["cues"])
        for clause_id in cue["clause_ids"]
    }
    references_by_clause = {
        clause.clause_id: tuple(
            normalize_cross_artifact_math_identity(reference)
            for reference in clause.math_references
        )
        for clause in script.clauses
    }
    problem_content = {
        target.target_id: normalize_cross_artifact_math_identity(
            target.math_text
        )
        for target in problem_targets
    }
    declared_board_targets = {
        item["board_object_id"] for item in payload["board_objects"]
    }
    entries = []
    sequence = 0
    for cue_index, cue in enumerate(payload["cues"]):
        for phase in ("lead_actions", "start_actions", "end_actions"):
            for item in cue[phase]:
                entries.append((sequence, cue_index, phase, item))
                sequence += 1
            cue[phase] = []

    creation_candidates = []
    other_candidates = []
    for sequence, cue_index, phase, item in entries:
        action = item["action"]
        action_type = action["type"]
        if action_type in {"write", "transform"}:
            content = normalize_cross_artifact_math_identity(
                action.get("content") or ""
            )
            matching_clauses = [
                clause.clause_id
                for clause in script.clauses
                if content
                and content in references_by_clause[clause.clause_id]
            ]
            if not matching_clauses:
                continue
            clause_id = matching_clauses[0]
            item["clause_id"] = clause_id
            creation_candidates.append(
                (
                    clause_positions[clause_id],
                    sequence,
                    cue_index_by_clause[clause_id],
                    item,
                )
            )
            continue
        if action["surface"] == "problem":
            target_id = action["target"]
            clause_id = item["clause_id"]
            if target_id in declared_board_targets:
                action["surface"] = "board"
                normalized_phase = phase
            elif target_id in problem_content:
                matching_clauses = [
                    clause.clause_id
                    for clause in script.clauses
                    if any(
                        contains_normalized_cross_artifact_math_identity(
                            reference,
                            problem_content[target_id],
                        )
                        for reference in references_by_clause[
                            clause.clause_id
                        ]
                    )
                ]
                if not matching_clauses:
                    continue
                clause_id = matching_clauses[0]
                item["clause_id"] = clause_id
                normalized_phase = (
                    "lead_actions"
                    if action_type in {"focus", "emphasize"}
                    else "start_actions"
                )
            else:
                continue
            other_candidates.append(
                (
                    clause_positions[clause_id],
                    sequence,
                    cue_index_by_clause[clause_id],
                    normalized_phase,
                    item,
                )
            )
            continue
        clause_id = item["clause_id"]
        if clause_id not in clause_positions:
            continue
        other_candidates.append(
            (
                clause_positions[clause_id],
                sequence,
                cue_index_by_clause[clause_id],
                phase,
                item,
            )
        )

    created_at = {}
    accepted = []
    for position, sequence, cue_index, item in sorted(
        creation_candidates
    ):
        action = item["action"]
        source = action.get("source")
        if source is not None and (
            source not in created_at or created_at[source] > position
        ):
            continue
        created_at[action["target"]] = position
        accepted.append(
            (
                position,
                0,
                sequence,
                cue_index,
                "start_actions",
                item,
            )
        )

    for position, sequence, cue_index, phase, item in other_candidates:
        action = item["action"]
        if action["type"] in {"clear_focus", "fade"}:
            continue
        if action["surface"] == "board":
            target = action["target"]
            if target not in created_at or created_at[target] > position:
                continue
            relation_target = action.get("relation_target")
            if relation_target is not None and (
                relation_target not in created_at
                or created_at[relation_target] > position
            ):
                continue
            phase = "start_actions"
        accepted.append(
            (position, 1, sequence, cue_index, phase, item)
        )

    for _, _, _, cue_index, phase, item in sorted(accepted):
        payload["cues"][cue_index][phase].append(item)

    board_layers = {
        item["board_object_id"]: item["layer"]
        for item in payload["board_objects"]
    }
    used_layers = {"base"}
    for _, _, _, _, _, item in accepted:
        action = item["action"]
        if action["surface"] != "board":
            continue
        for key in ("target", "source", "relation_target"):
            target_id = action.get(key)
            if target_id in board_layers:
                used_layers.add(board_layers[target_id])
    payload["overlay_transitions"] = [
        transition
        for transition in payload["overlay_transitions"]
        if transition["layer"] in used_layers
    ]
    return PerformanceScore.model_validate(payload)


def validate_interaction_plan(
    plan: InteractionPlan,
    progression: Union[TeachingProgression, ReasoningTrajectory],
    legacy_script: Optional[TeachingScript] = None,
) -> None:
    """Validate the current pre-script diagnostic intent contract."""
    if legacy_script is not None:
        validate_legacy_interaction_plan(plan, progression, legacy_script)
        return
    _require_exact(plan, InteractionPlan, "plan")
    _require_exact(progression, TeachingProgression, "progression")
    steps = {item.step_id: item for item in progression.steps}
    episode_ids = []
    teaching_step_ids = []
    for interaction in plan.interactions:
        if any(
            value is None
            for value in (
                interaction.episode_id,
                interaction.teaching_step_id,
                interaction.why_pause,
                interaction.resume_step_id,
            )
        ):
            _fail(
                "interaction_intent_missing",
                interaction.interaction_id,
                "Current interactions require episode, teaching step, pause reason, and resume step intent.",
            )
        if interaction.resume_policy != "continue":
            _fail(
                "interaction_resume_policy_invalid",
                interaction.interaction_id,
                "Current interactions continue after their authored response support.",
            )
        step = steps.get(interaction.teaching_step_id)
        if (
            step is None
            or interaction.resume_step_id != interaction.teaching_step_id
            or interaction.episode_id not in step.episode_ids
        ):
            _fail(
                "interaction_step_invalid",
                interaction.interaction_id,
                "Interaction and resume intent must bind to the same owning teaching step.",
            )
        if step.checkpoint is None:
            _fail(
                "interaction_checkpoint_missing",
                interaction.interaction_id,
                "Current interactions require a declared teaching-step checkpoint.",
            )
        if normalize_answer_leak_text(step.checkpoint.diagnostic_goal) not in (
            normalize_answer_leak_text(interaction.why_pause or "")
        ):
            _fail(
                "interaction_why_pause_invalid",
                interaction.interaction_id,
                "Pause reason must explain the exact checkpoint diagnostic goal.",
            )

        correct = next(
            item
            for item in interaction.options
            if item.option_id == interaction.correct_option_id
        )
        if any(
            value is not None
            for value in (
                correct.misconception,
                correct.error_code,
                correct.remediation_depth,
            )
        ):
            _fail(
                "interaction_option_diagnosis_invalid",
                correct.option_id,
                "Correct options cannot carry private error diagnosis metadata.",
            )
        wrong_options = [
            item
            for item in interaction.options
            if item.option_id != interaction.correct_option_id
        ]
        if any(
            item.misconception is None
            or item.error_code is None
            or item.remediation_depth not in {"conceptual", "worked"}
            for item in wrong_options
        ):
            _fail(
                "interaction_option_diagnosis_invalid",
                interaction.interaction_id,
                "Every incorrect option requires misconception, error code, and remediation depth.",
            )
        error_codes = [item.error_code for item in wrong_options]
        if len(error_codes) != len(set(error_codes)):
            _fail(
                "interaction_error_code_duplicate",
                interaction.interaction_id,
                "Incorrect option error codes must be unique within an interaction.",
            )
        # The current plan owns private diagnostic intent only. Its historical
        # prompt, hint, and display-text drafts are never runtime sources, so
        # student-visible choice safety is enforced on TeachingScript instead.
        episode_ids.append(interaction.episode_id)
        teaching_step_ids.append(interaction.teaching_step_id)

    if len(episode_ids) != len(set(episode_ids)):
        _fail(
            "interaction_episode_duplicate",
            "interaction_plan",
            "An episode may contain at most one runtime interaction.",
        )
    if len(teaching_step_ids) != len(set(teaching_step_ids)):
        _fail(
            "interaction_step_duplicate",
            "interaction_plan",
            "A teaching step may contain at most one runtime interaction.",
        )
    _validate_transfer_choice_safety(plan)


def _validate_interaction_choice_safety(interaction: object) -> None:
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
    if _leaks_correct_answer(interaction):
        _fail(
            "interaction_answer_leakage",
            interaction.interaction_id,
            "Interaction exposes its private correct answer binding.",
        )


def _validate_transfer_choice_safety(plan: InteractionPlan) -> None:
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


def validate_legacy_interaction_plan(
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
        _validate_interaction_choice_safety(interaction)
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

    _validate_transfer_choice_safety(plan)


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
            "reveal_step_header",
            "scroll_to_step",
            "open_supporting_explanation",
        },
        "end_actions": {
            "clear_focus",
            "fade",
            "write",
            "reveal_step_header",
            "complete_step",
            "scroll_to_step",
            "close_supporting_explanation",
        },
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
                    if action.type in {
                        "reveal_step_header",
                        "complete_step",
                        "scroll_to_step",
                        "open_supporting_explanation",
                        "close_supporting_explanation",
                    }:
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


def _validate_legacy_performance_score(
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
            for item in (
                *clause.math_references,
                *(
                    (clause.display_text,)
                    if clause.display_text is not None
                    else ()
                ),
            )
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


def _validate_structured_performance_score(
    score: PerformanceScore,
    progression: TeachingProgression,
    script: TeachingScript,
) -> None:
    steps = {item.step_id: item for item in progression.steps}
    main_clauses = {item.clause_id: item for item in script.clauses}
    response_clauses = {
        clause.clause_id: (response, clause)
        for response in script.response_scripts
        for clause in response.clauses
    }
    expected_ids = [item.clause_id for item in script.clauses]
    expected_ids.extend(
        clause.clause_id
        for response in script.response_scripts
        for clause in response.clauses
    )
    flattened = [item for cue in score.cues for item in cue.clause_ids]
    if flattened != expected_ids:
        _fail(
            "cue_clause_coverage_invalid",
            "performance_score",
            "Structured performance must cover main and response clauses exactly once in order.",
        )

    response_by_clause = {
        clause_id: response.response_id
        for clause_id, (response, _) in response_clauses.items()
    }
    for cue in score.cues:
        owners = {
            "main" if clause_id in main_clauses else response_by_clause.get(clause_id)
            for clause_id in cue.clause_ids
        }
        if None in owners or len(owners) != 1:
            _fail(
                "visual_clause_invalid",
                cue.cue_id,
                "A performance cue cannot cross main or response ownership.",
            )
        for bound in (*cue.lead_actions, *cue.start_actions, *cue.end_actions):
            if bound.clause_id not in cue.clause_ids:
                _fail(
                    "visual_clause_invalid",
                    cue.cue_id,
                    "Visual action is bound outside its exact cue.",
                )

    board_objects = {item.board_object_id: item for item in score.board_objects}
    for item in score.board_objects:
        if (
            item.teaching_step_id is None
            or item.line_role is None
            or item.teaching_step_id not in steps
        ):
            _fail(
                "structured_board_object_invalid",
                item.board_object_id,
                "Current board objects require a known step and line role.",
            )

    clause_positions = {
        clause_id: position for position, clause_id in enumerate(expected_ids)
    }
    events = []
    written_targets = set()
    phase_order = {
        "lead_actions": 0,
        "start_actions": 1,
        "end_actions": 2,
    }
    for cue in score.cues:
        for phase_name in (
            "lead_actions",
            "start_actions",
            "end_actions",
        ):
            for action_index, bound in enumerate(getattr(cue, phase_name)):
                clause = main_clauses.get(bound.clause_id)
                response = None
                if clause is None:
                    response, clause = response_clauses[bound.clause_id]
                action = bound.action
                if clause.lesson_step_id not in steps:
                    _fail(
                        "structured_action_step_invalid",
                        bound.clause_id,
                        "Structured actions require a known clause step.",
                    )
                if (
                    action.surface == "board"
                    and action.teaching_step_id != clause.lesson_step_id
                ):
                    _fail(
                        "structured_action_step_invalid",
                        bound.clause_id,
                        "Board action ownership must match its clause step.",
                    )
                if action.type == "write":
                    board_object = board_objects.get(action.target)
                    if (
                        board_object is None
                        or board_object.teaching_step_id
                        != clause.lesson_step_id
                        or board_object.line_role != action.board_role
                        or normalize_cross_artifact_math_identity(
                            board_object.content
                        )
                        != normalize_cross_artifact_math_identity(
                            action.content or ""
                        )
                    ):
                        _fail(
                            "structured_action_ownership_invalid",
                            bound.clause_id,
                            "Board write must match its declared step-aware object.",
                        )
                    if not is_valid_generated_display_content(
                        action.content or ""
                    ):
                        _fail(
                            "board_formula_invalid",
                            bound.clause_id,
                            "Structured board write contains invalid display content.",
                        )
                    written_targets.add(action.target)
                elif (
                    action.surface == "board"
                    and action.target in board_objects
                    and board_objects[action.target].teaching_step_id
                    != clause.lesson_step_id
                ):
                    _fail(
                        "structured_action_ownership_invalid",
                        bound.clause_id,
                        "Board action crosses teaching-step ownership.",
                    )
                events.append(
                    (
                        clause_positions[bound.clause_id],
                        phase_order[phase_name],
                        action_index,
                        bound.clause_id,
                        action,
                        response.response_id if response is not None else None,
                    )
                )

    if written_targets != set(board_objects):
        _fail(
            "structured_board_object_invalid",
            "performance_score",
            "Every structured board object must be written exactly through the score.",
        )

    main_events = [item for item in events if item[5] is None]
    for step_id, step in steps.items():
        step_clauses = [
            clause for clause in script.clauses
            if clause.lesson_step_id == step_id
        ]
        if not step_clauses:
            _fail(
                "structured_step_lifecycle_invalid",
                step_id,
                "Every teaching step requires main script clauses.",
            )
        step_events = [
            item for item in main_events
            if item[4].teaching_step_id == step_id
        ]
        reveals = [
            item for item in step_events
            if item[4].type == "reveal_step_header"
        ]
        completes = [
            item for item in step_events
            if item[4].type == "complete_step"
        ]
        writes = [item for item in step_events if item[4].type == "write"]
        scrolls = [
            item for item in step_events
            if item[4].type == "scroll_to_step"
        ]
        if len(reveals) != 1 or len(completes) != 1 or not writes:
            _fail(
                "structured_step_lifecycle_invalid",
                step_id,
                "Each step requires one reveal, one completion, and a board write.",
            )
        reveal = reveals[0]
        complete = completes[0]
        if reveal[4].step_label != step.directory_label:
            _fail(
                "structured_step_label_invalid",
                step_id,
                "Step header label must match the teaching progression exactly.",
            )
        question_positions = [
            clause_positions[clause.clause_id]
            for clause in step_clauses
            if clause.pedagogical_function == "question"
        ]
        activation_position = (
            question_positions[0]
            if question_positions
            else clause_positions[step_clauses[0].clause_id]
        )
        if reveal[:3] <= (activation_position, 1, -1):
            _fail(
                "structured_step_lifecycle_invalid",
                step_id,
                "Step header must follow its authored step-opening clause.",
            )
        first_write = min(writes, key=lambda item: item[:3])
        activation_scrolls = [
            item for item in scrolls
            if reveal[:3] < item[:3] <= first_write[:3]
        ]
        if not activation_scrolls or reveal[:3] >= first_write[:3]:
            _fail(
                "structured_step_lifecycle_invalid",
                step_id,
                "Step activation must reveal, scroll, then write.",
            )
        last_clause_id = step_clauses[-1].clause_id
        if (
            complete[3] != last_clause_id
            or complete[1] != phase_order["end_actions"]
            or any(complete[:3] <= item[:3] for item in writes)
        ):
            _fail(
                "structured_step_lifecycle_invalid",
                step_id,
                "Step completion must follow its final main-clause write.",
            )

    events_by_response = {
        response.response_id: [
            item for item in events if item[5] == response.response_id
        ]
        for response in script.response_scripts
    }
    for response in script.response_scripts:
        response_events = events_by_response[response.response_id]
        support_events = [
            item
            for item in response_events
            if item[4].type in {
                "open_supporting_explanation",
                "close_supporting_explanation",
            }
            or (
                item[4].type == "write"
                and item[4].board_role == "support"
            )
        ]
        if response.classification == "correct":
            if support_events:
                _fail(
                    "structured_support_lifecycle_invalid",
                    response.response_id,
                    "Correct responses cannot open worked support.",
                )
            continue
        ordered = sorted(response_events, key=lambda item: item[:3])
        opens = [
            item for item in ordered
            if item[4].type == "open_supporting_explanation"
        ]
        closes = [
            item for item in ordered
            if item[4].type == "close_supporting_explanation"
        ]
        support_writes = [
            item for item in ordered
            if item[4].type == "write" and item[4].board_role == "support"
        ]
        scrolls = [
            item for item in ordered if item[4].type == "scroll_to_step"
        ]
        if (
            len(opens) != 1
            or len(closes) != 1
            or not support_writes
            or not ordered
            or ordered[0][4].type != "open_supporting_explanation"
            or opens[0][:3] >= support_writes[0][:3]
            or support_writes[-1][:3] >= closes[0][:3]
            or not any(item[:3] > closes[0][:3] for item in scrolls)
            or opens[0][4].target != closes[0][4].target
        ):
            _fail(
                "structured_support_lifecycle_invalid",
                response.response_id,
                "Incorrect response support must open, write, close, and restore scroll.",
            )


def validate_performance_score(
    score: PerformanceScore,
    problem_targets: ProblemTargets,
    progression: Optional[Union[TeachingProgression, TeachingScript]] = None,
    script: Optional[Union[TeachingScript, InteractionPlan]] = None,
    plan: Optional[InteractionPlan] = None,
) -> None:
    if progression is None:
        current_progression = None
        current_script = script
        current_plan = plan
    elif type(progression) is TeachingScript and type(script) is InteractionPlan:
        current_progression = None
        current_script = progression
        current_plan = script
        if plan is not None:
            raise TypeError("legacy performance validation accepts four arguments")
    else:
        current_progression = progression
        current_script = script
        current_plan = plan
    _require_exact(current_script, TeachingScript, "script")
    _require_exact(current_plan, InteractionPlan, "plan")
    if current_progression is None:
        main_clause_ids = {item.clause_id for item in current_script.clauses}
        response_clause_ids = {
            clause.clause_id
            for response in current_script.response_scripts
            for clause in response.clauses
        }
        if any(
            clause_id in response_clause_ids
            for cue in score.cues
            for clause_id in cue.clause_ids
        ):
            legacy_payload = score.model_dump(mode="python")
            legacy_payload["cues"] = [
                cue.model_dump(mode="python")
                for cue in score.cues
                if all(
                    clause_id in main_clause_ids
                    for clause_id in cue.clause_ids
                )
            ]
            score = PerformanceScore.model_validate(legacy_payload)
        _validate_legacy_performance_score(
            score,
            problem_targets,
            current_script,
            current_plan,
        )
        return
    _require_exact(current_progression, TeachingProgression, "progression")
    main_clause_ids = {item.clause_id for item in current_script.clauses}
    main_cues = [
        cue
        for cue in score.cues
        if all(item in main_clause_ids for item in cue.clause_ids)
    ]
    main_payload = score.model_dump(mode="python")
    main_payload["cues"] = [
        cue.model_dump(mode="python") for cue in main_cues
    ]
    _validate_legacy_performance_score(
        PerformanceScore.model_validate(main_payload),
        problem_targets,
        current_script,
        current_plan,
    )
    _validate_structured_performance_score(
        score,
        current_progression,
        current_script,
    )


def validate_simulation_report(
    report: SimulationReport,
    trajectory: ReasoningTrajectory,
    plan: InteractionPlan,
    *,
    require_structured_abilities: bool = False,
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
    if require_structured_abilities:
        for result in report.episode_results:
            if any(
                value is None
                for value in (
                    result.can_align_display_and_spoken_math,
                    result.can_recover_with_adaptive_support,
                    result.can_locate_current_step,
                )
            ):
                _fail(
                    "simulation_structured_ability_missing",
                    result.episode_id,
                    "Current-rubric simulation must evaluate every structured teaching ability.",
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
    progression: Optional[TeachingProgression] = None,
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
    if progression is not None:
        _require_exact(progression, TeachingProgression, "progression")

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
        "teaching_progression": {
            "teaching_progression",
            *(
                item.step_id
                for item in progression.steps
            ),
        } if progression is not None else set(),
        "teaching_script": {
            "teaching_script",
            *(item.clause_id for item in script.clauses),
            *(item.response_id for item in script.response_scripts),
            *(
                clause.clause_id
                for response in script.response_scripts
                for clause in response.clauses
            ),
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
            finding.responsible_role
            != _ARTIFACT_HISTORY_ROLES[finding.artifact_type]
        ):
            _fail(
                "review_responsibility_invalid",
                finding.finding_id,
                "Review finding must route to the earliest responsible role.",
            )

    _validate_review_dependency_metadata(decision)

    novice_gate_failed = bool(report.blocking_findings) or any(
        not all(
            ability is not False
            for ability in (
                result.can_identify_attention_target,
                result.can_explain_decision,
                result.can_execute_action,
                result.can_use_result_to_continue,
                result.can_align_display_and_spoken_math,
                result.can_recover_with_adaptive_support,
                result.can_locate_current_step,
            )
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
            start_index = _ARTIFACT_OWNER_ORDER[finding.artifact_type] + 1
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
        earliest_artifact = min(
            material_findings,
            key=lambda item: _ARTIFACT_OWNER_ORDER[item.artifact_type],
        ).artifact_type
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
    if prepared.teaching_progression is None:
        _fail(
            "artifact_history_invalid",
            "teaching_progression",
            "Current-rubric prepared lessons require teaching progression.",
        )
    _require_exact(
        prepared.teaching_progression,
        TeachingProgression,
        "teaching_progression",
    )
    try:
        validate_teaching_progression(
            prepared.teaching_progression,
            prepared.reasoning_trajectory,
            problem_targets,
        )
    except TeachingProgressionValidationError as error:
        _fail(
            error.code,
            error.artifact_id,
            "Teaching progression failed deterministic semantic validation.",
        )
    validate_teaching_script(
        prepared.teaching_script,
        prepared.reasoning_trajectory,
        prepared.teaching_progression,
        prepared.interaction_plan,
    )
    validate_interaction_plan(
        prepared.interaction_plan,
        prepared.teaching_progression,
    )
    validate_performance_score(
        prepared.performance_score,
        problem_targets,
        prepared.teaching_progression,
        prepared.teaching_script,
        prepared.interaction_plan,
    )
    _validate_must_teach_board_evidence(prepared)
    validate_simulation_report(
        prepared.simulation_report,
        prepared.reasoning_trajectory,
        prepared.interaction_plan,
        require_structured_abilities=True,
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
        progression=prepared.teaching_progression,
    )
    if prepared.review.status != "approved":
        _fail(
            "review_approval_invalid",
            "review",
            "Prepared lesson requires an approved review.",
        )


def _validate_must_teach_board_evidence(prepared: PreparedLesson) -> None:
    items = {
        item.must_teach_id: item
        for episode in prepared.reasoning_trajectory.episodes
        for item in episode.must_teach
    }
    clause_refs = {
        clause.clause_id: set(clause.must_teach_refs)
        for clause in prepared.teaching_script.clauses
    }
    action_contents: Dict[str, List[str]] = {}
    for cue in prepared.performance_score.cues:
        for phase in (cue.lead_actions, cue.start_actions, cue.end_actions):
            for bound in phase:
                if (
                    bound.action.surface == "board"
                    and bound.action.type in {"write", "transform"}
                    and bound.action.content
                ):
                    action_contents.setdefault(bound.clause_id, []).append(
                        bound.action.content
                    )
    for item_id, item in items.items():
        matching_clauses = [
            clause_id
            for clause_id, refs in clause_refs.items()
            if item_id in refs
        ]
        if item.student_display_evidence is None or not any(
            action_contents.get(clause_id) for clause_id in matching_clauses
        ):
            _fail(
                "must_teach_board_evidence_missing",
                item_id,
                "Every current must-teach item requires a board write bound to its evidence clause.",
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
    repairable = _ARTIFACT_DEPENDENCY_ORDER
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
