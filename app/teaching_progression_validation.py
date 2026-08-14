"""Deterministic semantic validation for the teaching progression bridge."""

import re
import unicodedata
from typing import Dict, List, Tuple, Union

from pydantic import ValidationError

from app.math_content import (
    is_valid_generated_display_content,
    normalize_cross_artifact_math_identity,
)
from app.preparation_models import ReasoningTrajectory, TeachingProgression
from app.problem_focus import MAX_PROBLEM_FOCUS_TARGETS
from app.schemas import ProblemFocusTarget


ProblemTargets = Union[List[ProblemFocusTarget], Tuple[ProblemFocusTarget, ...]]
_GENERIC_WHY_NOW_TOKENS = (
    "算一算",
    "接下来",
    "下一步",
    "然后",
    "之后",
    "继续",
    "接着",
    "开始",
    "再来",
    "进行",
    "计算",
    "运算",
    "整理",
    "化简",
    "处理",
    "一下",
    "现在",
    "往下",
    "题意",
    "即可",
    "先",
    "吧",
    "做",
    "再",
    "算",
)
_WHY_NOW_MARKERS = (
    "因为",
    "由于",
    "所以",
    "从而",
    "为了",
    "以便",
    "便于",
    "避免",
    "保证",
    "确保",
    "根据",
    "利用",
    "结合",
    "条件",
    "已知",
    "题目",
    "前一步",
    "上一步",
    "依赖",
    "才能",
    "才",
    "观察",
    "判断",
    "确认",
    "需要",
    "必须",
    "否则",
    "注意",
    "关键",
    "目标",
)
_BOARD_NUMBER_PREFIX = re.compile(
    r"^\s*(?:(?:[（(]\s*(?:\d{1,3}|[IVXLCDMivxlcdm]{1,8}|"
    r"[零〇一二三四五六七八九十百]{1,4})\s*[）)]\s*[.、]?|"
    r"(?:\d{1,3}|[IVXLCDMivxlcdm]{1,8}|[零〇一二三四五六七八九十百]{1,4})[.、]))\s*"
)
_BOARD_LABEL_PREFIX = re.compile(
    r"^(?:当前推理得到|最终结果|结果|答案|得到|结论)\s*:\s*"
)
_DEFAULT_IGNORABLE_RANGES = (
    (0x034F, 0x034F),
    (0x115F, 0x1160),
    (0x17B4, 0x17B5),
    (0x180B, 0x180F),
    (0x2065, 0x2065),
    (0x3164, 0x3164),
    (0xFE00, 0xFE0F),
    (0xFFA0, 0xFFA0),
    (0xFFF0, 0xFFF8),
    (0x1BCA0, 0x1BCA3),
    (0x1D173, 0x1D17A),
    (0xE0000, 0xE0FFF),
)
_HAN_RANGES = (
    (0x3400, 0x4DBF),
    (0x4E00, 0x9FFF),
    (0xF900, 0xFAFF),
    (0x20000, 0x2FA1F),
    (0x30000, 0x3134F),
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


def _is_default_ignorable(character: str) -> bool:
    codepoint = ord(character)
    return (
        unicodedata.category(character) == "Cf"
        or any(
            start <= codepoint <= end
            for start, end in _DEFAULT_IGNORABLE_RANGES
        )
    )


def _without_default_ignorables(value: str) -> str:
    return "".join(
        character
        for character in value
        if not _is_default_ignorable(character)
    )


def _normalize_unicode(value: str) -> str:
    return _without_default_ignorables(unicodedata.normalize("NFKC", value))


def _remove_semantic_tokens(value: str, tokens: Tuple[str, ...]) -> str:
    remainder = value
    for token in sorted(tokens, key=len, reverse=True):
        remainder = remainder.replace(token, "")
    return remainder


def _is_han_character(character: str) -> bool:
    codepoint = ord(character)
    return any(start <= codepoint <= end for start, end in _HAN_RANGES)


def _why_now_is_explanatory(value: str) -> bool:
    if type(value) is not str:
        return False
    normalized = "".join(
        character
        for character in _normalize_unicode(value).lower()
        if not character.isspace()
        and not unicodedata.category(character).startswith("P")
    )
    if not normalized or not any(
        marker in normalized for marker in _WHY_NOW_MARKERS
    ):
        return False

    remainder = _remove_semantic_tokens(normalized, _WHY_NOW_MARKERS)
    remainder = _remove_semantic_tokens(
        remainder,
        _GENERIC_WHY_NOW_TOKENS,
    )
    han_count = sum(_is_han_character(character) for character in remainder)
    has_alnum_or_math_token = any(
        character.isalnum() and not _is_han_character(character)
        for character in remainder
    )
    return han_count >= 2 or has_alnum_or_math_token


def _is_unicode_numbering(character: str) -> bool:
    name = unicodedata.name(character, "")
    return (
        (
            ("CIRCLED" in name or "PARENTHESIZED" in name)
            and (
                "DIGIT" in name
                or "NUMBER" in name
                or "IDEOGRAPH" in name
            )
        )
        or "ROMAN NUMERAL" in name
    )


def _strip_raw_unicode_numbering(value: str) -> str:
    index = 0
    while index < len(value) and (
        value[index].isspace()
        or (
            value[index] != "-"
            and unicodedata.category(value[index]).startswith("P")
        )
    ):
        index += 1
    if index < len(value) and _is_unicode_numbering(value[index]):
        return value[:index] + value[index + 1 :].lstrip()
    return value


def _strip_board_numbering(value: str) -> str:
    without_unicode_numbering = _strip_raw_unicode_numbering(value)
    return _BOARD_NUMBER_PREFIX.sub(
        "",
        without_unicode_numbering,
        count=1,
    )


def _strip_endpoint_punctuation(value: str) -> str:
    start = 0
    end = len(value)
    while start < end and (
        value[start].isspace()
        or (
            value[start] != "-"
            and unicodedata.category(value[start]).startswith("P")
        )
    ):
        start += 1
    while end > start and (
        value[end - 1].isspace()
        or unicodedata.category(value[end - 1]).startswith("P")
    ):
        end -= 1
    return value[start:end]


def _normalize_board_identity(value: str) -> str:
    without_ignorables = _without_default_ignorables(value.strip())
    without_raw_numbering = _strip_board_numbering(without_ignorables)
    normalized = _normalize_unicode(without_raw_numbering)
    without_numbering = _strip_board_numbering(normalized)
    without_endpoint_punctuation = _strip_endpoint_punctuation(without_numbering)
    without_label = _BOARD_LABEL_PREFIX.sub(
        "",
        without_endpoint_punctuation,
        count=1,
    )
    without_decoration = _strip_endpoint_punctuation(without_label)
    return normalize_cross_artifact_math_identity(without_decoration)


def _board_summary_has_semantic_character(value: str) -> bool:
    return any(character.isalnum() for character in value)


def validate_teaching_progression(
    progression: TeachingProgression,
    trajectory: ReasoningTrajectory,
    problem_targets: ProblemTargets,
) -> None:
    """Validate exact semantic coverage across trajectory and progression."""
    _require_exact(progression, TeachingProgression, "progression")
    _require_exact(trajectory, ReasoningTrajectory, "trajectory")
    _validate_problem_targets(problem_targets)
    try:
        trajectory = ReasoningTrajectory.model_validate(
            trajectory.model_dump(mode="python", warnings="none")
        )
    except (ValidationError, TypeError):
        _fail("trajectory_structure_invalid", "reasoning_trajectory")
    try:
        problem_targets = type(problem_targets)(
            ProblemFocusTarget.model_validate(
                item.model_dump(mode="python", warnings="none")
            )
            for item in problem_targets
        )
    except (ValidationError, TypeError):
        _fail("problem_target_structure_invalid", "problem_targets")
    try:
        progression = TeachingProgression.model_validate(
            progression.model_dump(mode="python", warnings="none")
        )
    except (ValidationError, TypeError):
        _fail("progression_structure_invalid", "teaching_progression")

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
    seen_evidence_targets = set()
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
        if seen_evidence_targets.intersection(step.evidence_target_ids):
            _fail("progression_evidence_target_duplicate", step.step_id)
        seen_evidence_targets.update(step.evidence_target_ids)

        if step.directory_label in directory_labels:
            _fail("progression_directory_label_duplicate", step.step_id)
        directory_labels.add(step.directory_label)

        if not _why_now_is_explanatory(step.why_now):
            _fail("progression_why_not_explanatory", step.step_id)

        for summary in step.board_summary:
            if not is_valid_generated_display_content(summary):
                _fail("progression_board_content_invalid", step.step_id)
        normalized_summaries = {
            _normalize_board_identity(item) for item in step.board_summary
        }
        if any(
            not _board_summary_has_semantic_character(item)
            for item in normalized_summaries
        ):
            _fail(
                "progression_board_summary_not_explanatory",
                step.step_id,
            )
        visible_episode_results = {
            _normalize_board_identity(episode_by_id[episode_id].result)
            for episode_id in step.episode_ids
        }
        if normalized_summaries <= visible_episode_results:
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

    if seen_evidence_targets != allowed_target_ids:
        _fail(
            "progression_evidence_target_coverage_invalid",
            "teaching_progression",
        )

    if (
        len(actual_must_teach_ids) != len(set(actual_must_teach_ids))
        or set(actual_must_teach_ids) != set(expected_must_teach_ids)
    ):
        _fail(
            "progression_must_teach_coverage_invalid",
            "teaching_progression",
        )
