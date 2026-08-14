"""Deterministic semantic validation for the teaching progression bridge."""

import re
from typing import Dict, List, Tuple, Union

from app.math_content import (
    is_valid_generated_display_content,
    normalize_cross_artifact_math_identity,
)
from app.preparation_models import ReasoningTrajectory, TeachingProgression
from app.problem_focus import MAX_PROBLEM_FOCUS_TARGETS
from app.schemas import ProblemFocusTarget


ProblemTargets = Union[List[ProblemFocusTarget], Tuple[ProblemFocusTarget, ...]]
_GENERIC_WHY_NOW = frozenset(
    {
        "然后计算",
        "继续计算",
        "接着计算",
        "再计算",
        "进行计算",
        "开始计算",
    }
)


class TeachingProgressionValidationError(ValueError):
    """Stable, content-free semantic progression error."""

    def __init__(self, code: str, artifact_id: str) -> None:
        super().__init__("%s:%s" % (code, artifact_id))
        self.code = code
        self.artifact_id = artifact_id


def _fail(code: str, artifact_id: str) -> None:
    raise TeachingProgressionValidationError(code, artifact_id)


def _require_exact(value: object, expected: type, label: str) -> None:
    if type(value) is not expected:
        raise TypeError("%s must be an exact %s model" % (label, expected.__name__))


def _validate_problem_targets(problem_targets: ProblemTargets) -> None:
    if (
        type(problem_targets) not in (list, tuple)
        or len(problem_targets) > MAX_PROBLEM_FOCUS_TARGETS
        or any(type(item) is not ProblemFocusTarget for item in problem_targets)
    ):
        raise TypeError(
            "problem_targets must be a bounded list or tuple of exact "
            "ProblemFocusTarget models"
        )


def derive_misconception_vocabulary(
    trajectory: ReasoningTrajectory,
) -> List[Dict[str, str]]:
    """Project trajectory misconception text to bounded server-owned IDs."""
    _require_exact(trajectory, ReasoningTrajectory, "trajectory")
    return [
        {
            "misconception_id": "misconception-%03d-%03d"
            % (episode_index, misconception_index),
            "episode_id": episode.episode_id,
            "description": description,
        }
        for episode_index, episode in enumerate(trajectory.episodes, start=1)
        for misconception_index, description in enumerate(
            episode.likely_misconceptions,
            start=1,
        )
    ]


def _why_now_is_explanatory(value: str) -> bool:
    normalized = re.sub(r"[\s，。；;,.!?！？]", "", value).lower()
    return normalized not in _GENERIC_WHY_NOW


def validate_teaching_progression(
    progression: TeachingProgression,
    trajectory: ReasoningTrajectory,
    problem_targets: ProblemTargets,
) -> None:
    """Validate exact semantic coverage across trajectory and progression."""
    _require_exact(progression, TeachingProgression, "progression")
    _require_exact(trajectory, ReasoningTrajectory, "trajectory")
    _validate_problem_targets(problem_targets)

    expected_episode_ids = [episode.episode_id for episode in trajectory.episodes]
    actual_episode_ids = [
        episode_id
        for step in progression.steps
        for episode_id in step.episode_ids
    ]
    if actual_episode_ids != expected_episode_ids:
        _fail("progression_episode_coverage_invalid", "teaching_progression")

    episode_by_id = {
        episode.episode_id: episode for episode in trajectory.episodes
    }
    expected_must_teach_ids = [
        item.must_teach_id
        for episode in trajectory.episodes
        for item in episode.must_teach
    ]
    if len(expected_must_teach_ids) != len(set(expected_must_teach_ids)):
        _fail(
            "progression_must_teach_coverage_invalid",
            "reasoning_trajectory",
        )
    must_teach_owner = {
        item.must_teach_id: episode.episode_id
        for episode in trajectory.episodes
        for item in episode.must_teach
    }
    actual_must_teach_ids = [
        item for step in progression.steps for item in step.must_teach_refs
    ]

    allowed_target_ids = {item.target_id for item in problem_targets}
    vocabulary = derive_misconception_vocabulary(trajectory)
    misconception_owner = {
        item["misconception_id"]: item["episode_id"] for item in vocabulary
    }
    seen_misconceptions = set()
    directory_labels = set()

    for step in progression.steps:
        owned_episode_ids = set(step.episode_ids)
        for reference in step.must_teach_refs:
            if (
                reference not in must_teach_owner
                or must_teach_owner[reference] not in owned_episode_ids
            ):
                _fail("progression_must_teach_ref_invalid", step.step_id)

        if len(step.evidence_target_ids) != len(set(step.evidence_target_ids)):
            _fail("progression_evidence_target_duplicate", step.step_id)
        if not set(step.evidence_target_ids) <= allowed_target_ids:
            _fail("progression_evidence_target_invalid", step.step_id)

        if step.directory_label in directory_labels:
            _fail("progression_directory_label_duplicate", step.step_id)
        directory_labels.add(step.directory_label)

        if not _why_now_is_explanatory(step.why_now):
            _fail("progression_why_not_explanatory", step.step_id)

        for summary in step.board_summary:
            if not is_valid_generated_display_content(summary):
                _fail("progression_board_content_invalid", step.step_id)
        if len(step.board_summary) == 1:
            normalized_summary = normalize_cross_artifact_math_identity(
                step.board_summary[0]
            )
            visible_episode_results = {
                normalize_cross_artifact_math_identity(
                    episode_by_id[episode_id].result
                )
                for episode_id in step.episode_ids
            }
            if normalized_summary in visible_episode_results:
                _fail(
                    "progression_board_summary_not_explanatory",
                    step.step_id,
                )

        if step.checkpoint is None:
            continue
        references = step.checkpoint.misconception_ids
        if len(references) != len(set(references)) or any(
            item in seen_misconceptions for item in references
        ):
            _fail("progression_misconception_ref_duplicate", step.step_id)
        for reference in references:
            if (
                reference not in misconception_owner
                or misconception_owner[reference] not in owned_episode_ids
            ):
                _fail("progression_misconception_ref_invalid", step.step_id)
        seen_misconceptions.update(references)

    if (
        len(actual_must_teach_ids) != len(set(actual_must_teach_ids))
        or set(actual_must_teach_ids) != set(expected_must_teach_ids)
    ):
        _fail(
            "progression_must_teach_coverage_invalid",
            "teaching_progression",
        )
