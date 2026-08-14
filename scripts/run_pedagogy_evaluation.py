#!/usr/bin/env python3
"""Run bounded pedagogy contract evaluations without claiming learning effects."""

import argparse
import asyncio
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import time
from typing import Callable, Dict, List, Literal, Optional, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.config import Settings
from app.generation import LessonGenerationService
from app.generation_diagnostics import InternalGenerationDiagnostic
from app.generation_integrity import validate_lesson_generation_pair
from app.llm_client import OpenAICompatibleClient
from app.math_engine import MathEngine
from app.math_content import (
    normalize_answer_leak_text,
    normalize_cross_artifact_math_identity,
)
from app.math_speech import MathSpeechError, validate_display_spoken_alignment
from app.pedagogy_rubric import PEDAGOGY_RUBRIC_VERSION
from app.preparation_models import GenerationRecord
from app.schemas import (
    LessonLayer,
    NonEmptyString,
    ProblemInput,
    ProblemText,
    RuntimeLesson,
    SchemaModel,
    SyncVisualAction,
)


DEFAULT_FIXTURE = REPOSITORY_ROOT / "tests" / "fixtures" / "pedagogy_golden_cases.json"
MAX_CASES = 64
MAX_RUNS_PER_CASE = 10
MAX_JSON_INPUT_BYTES = 16 * 1024 * 1024
MAX_JSON_OUTPUT_BYTES = 32 * 1024 * 1024
MANIFEST_SCHEMA_VERSION = 1
PRIVATE_DIRECTORY_MODE = 0o700
PRIVATE_FILE_MODE = 0o600
UNSAFE_DIRECTORY_WRITE_MASK = stat.S_IWGRP | stat.S_IWOTH
_SAFE_ID = re.compile(r"[a-z][a-z0-9_]{2,63}")
_SAFE_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
_CASE_KEYS = {
    "case_id",
    "problem",
    "coverage_tags",
    "trace_anchors",
    "required_reasoning_modes",
    "required_must_teach",
    "typical_misconceptions",
    "required_board_states",
    "acceptable_excerpt_patterns",
    "unacceptable_excerpt_patterns",
}
_STRUCTURED_EXPECTATION_KEYS = {
    "required_step_labels",
    "required_must_teach_anchors",
    "required_spoken_forms",
    "required_error_codes",
}
_PROBLEM_KEYS = {
    "problem_text",
    "reference_answer",
    "reference_solution_text",
    "lesson_length",
}
_COVERAGE_TAGS = {
    "concept_condition_conversion",
    "algebra_execution",
    "equation_parameter",
    "method_selection",
    "text_only_geometry",
    "function_relationship",
    "omitted_condition",
    "exploration_or_revision",
    "concept_overlay",
    "no_forced_interaction",
    "no_forced_emphasis",
}
_REASONING_MODES = {
    "understand",
    "plan",
    "explore",
    "execute",
    "monitor",
    "revise",
    "reflect",
}
_SAFE_FAILURE_CATEGORIES = {
    "provider_error",
    "invalid_structure",
    "reference_trace_failed",
    "reasoning_design_failed",
    "review_not_converged",
    "compile_failed",
    "tts_failed",
    "persistence_failed",
}
_STAGE_CODES = {
    "正在验证数学路线": "math_route",
    "正在审阅参考解析": "reference_audit",
    "正在规划数学路线": "math_route",
    "正在整理参考教学路线": "reference_grounding",
    "正在整理参考解析": "reference_trace",
    "正在设计解题思维轨迹": "reasoning_trajectory",
    "正在编写讲稿": "script",
    "正在设计互动": "interaction",
    "正在编排板书与高亮": "performance",
    "正在模拟学生并审核课程": "review",
    "正在编译课堂": "compile",
}
_METRIC_COUNT_KEYS = {
    "must_teach_coverage": "covered",
    "step_coverage": "covered",
    "must_teach_to_script_coverage": "covered",
    "must_teach_to_board_coverage": "covered",
    "display_speech_alignment": "aligned",
    "diagnostic_branch_coverage": "covered",
    "step_lifecycle_coverage": "complete",
    "clause_action_binding": "valid",
}
_METRIC_SCOPE = [
    "generation_success",
    "hard_gate_review_pass",
    *_METRIC_COUNT_KEYS,
    "schema_runtime_pass",
    "duration_ms",
    "call_count",
]
_FORBIDDEN_PUBLIC_KEYS = {
    "candidate_version",
    "validation_report",
    "private_feedback",
    "reference_answer",
    "reference_solution_text",
    "expected_answer",
    "correct_option_id",
    "canonical_answer",
    "feedback",
    "feedback_audio_url",
    "correct_audio_url",
}


class EvaluationConfigurationError(RuntimeError):
    """Safe operator error raised before or independently of provider calls."""


class GoldenEvidenceError(ValueError):
    category = "invalid_structure"


class _PublicProblem(SchemaModel):
    problem_text: ProblemText
    required_method: Optional[
        Literal["factor", "quadratic_formula", "complete_the_square"]
    ] = None
    lesson_length: Literal["concise", "standard"]


class _PublicOption(SchemaModel):
    option_id: NonEmptyString
    label: Optional[NonEmptyString] = None


class _PublicInteraction(SchemaModel):
    interaction_id: NonEmptyString
    kind: Literal[
        "point_select",
        "choice",
        "expression",
        "free_text",
        "transfer",
    ]
    prompt: NonEmptyString
    options: List[_PublicOption]
    hints: List[NonEmptyString]


class _PublicCue(SchemaModel):
    cue_id: NonEmptyString
    spoken_text: NonEmptyString
    lead_actions: List[SyncVisualAction]
    start_actions: List[SyncVisualAction]
    end_actions: List[SyncVisualAction]


class _PublicBeat(SchemaModel):
    beat_id: NonEmptyString
    purpose: NonEmptyString
    narration: NonEmptyString
    layer: LessonLayer
    sync_cues: List[_PublicCue]
    interaction: Optional[_PublicInteraction] = None
    next_beat_id: Optional[NonEmptyString] = None


class _PublicTransferItem(SchemaModel):
    problem_text: ProblemText
    method_signal: NonEmptyString
    options: List[_PublicOption]


class _PublicEvaluationArtifact(SchemaModel):
    schema_version: Literal[1]
    lesson_id: NonEmptyString
    problem: _PublicProblem
    title: NonEmptyString
    learning_goal: NonEmptyString
    beats: List[_PublicBeat]
    summary: NonEmptyString
    transfer_item: _PublicTransferItem


class StepClock:
    """A bounded deterministic clock useful for unit-level evaluation tests."""

    def __init__(self, values: Sequence[float]) -> None:
        self._values = iter(values)

    def __call__(self) -> float:
        return float(next(self._values))


class CountingModelClient:
    """Count every provider call without retaining prompt or response text."""

    def __init__(self, delegate: object) -> None:
        self._delegate = delegate
        self.call_count = 0

    async def complete_json(self, system_prompt: str, user_prompt: str) -> object:
        self.call_count += 1
        return await self._delegate.complete_json(system_prompt, user_prompt)

    async def complete_json_with_metadata(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> object:
        self.call_count += 1
        return await self._delegate.complete_json_with_metadata(
            system_prompt,
            user_prompt,
        )

    async def close(self) -> None:
        close = getattr(self._delegate, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the reviewed pedagogy golden set or create a blinded "
            "comparison from two completed run directories."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New or empty output directory; existing content is never overwritten.",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--rubric-version",
        help="Explicit rubric version expected from this checkout.",
    )
    mode.add_argument(
        "--compare-run",
        type=Path,
        nargs=2,
        metavar=("RUN_A", "RUN_B"),
        help="Create blinded pairs from two completed evaluation directories.",
    )
    parser.add_argument(
        "--runs-per-case",
        type=int,
        help="Required in generation mode; use 3 for the standard comparison.",
    )
    parser.add_argument(
        "--candidate-version",
        help=(
            "Optional prompt/build candidate label. Defaults to the rubric "
            "version; use distinct labels for prompt-only comparisons."
        ),
    )
    parser.add_argument(
        "--fixture",
        type=Path,
        default=DEFAULT_FIXTURE,
        help="Reviewed JSON fixture (defaults to the repository golden set).",
    )
    return parser


def _json_content(payload: object) -> str:
    try:
        content = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        ) + "\n"
    except (TypeError, ValueError, RecursionError):
        raise EvaluationConfigurationError(
            "evaluation JSON output is invalid"
        ) from None
    if len(content.encode("utf-8")) > MAX_JSON_OUTPUT_BYTES:
        raise EvaluationConfigurationError("evaluation JSON output exceeds size limit")
    return content


class _OutputGuard:
    """Own one stable, non-symlink output tree and never overwrite files."""

    def __init__(
        self,
        root: Path,
        resolved_root: Path,
        root_identity: tuple,
    ) -> None:
        self.root = root
        self._resolved_root = resolved_root
        self._root_identity = root_identity

    @staticmethod
    def _secure_dirfd_supported() -> bool:
        return (
            os.name == "posix"
            and hasattr(os, "O_DIRECTORY")
            and hasattr(os, "O_NOFOLLOW")
            and hasattr(os, "geteuid")
            and hasattr(os, "fchmod")
            and os.open in os.supports_dir_fd
            and os.mkdir in os.supports_dir_fd
            and os.stat in os.supports_dir_fd
            and os.chmod in os.supports_dir_fd
            and os.chmod in os.supports_follow_symlinks
        )

    @staticmethod
    def _validate_directory_stat(value: object, label: str) -> None:
        if not stat.S_ISDIR(value.st_mode):
            raise EvaluationConfigurationError("%s is not a directory" % label)
        if value.st_uid != os.geteuid():
            raise EvaluationConfigurationError(
                "%s owner is not the current user" % label
            )
        if value.st_mode & UNSAFE_DIRECTORY_WRITE_MASK:
            raise EvaluationConfigurationError(
                "%s permissions allow group or other writes" % label
            )

    @classmethod
    def create(cls, output_dir: Path) -> "_OutputGuard":
        if not cls._secure_dirfd_supported():
            raise EvaluationConfigurationError(
                "secure descriptor-relative output is unavailable"
            )
        if type(output_dir) is not Path:
            output_dir = Path(output_dir)
        if output_dir.is_symlink():
            raise EvaluationConfigurationError("output root cannot be a symlink")
        created = not output_dir.exists()
        try:
            if created:
                output_dir.mkdir(parents=True, mode=PRIVATE_DIRECTORY_MODE)
                os.chmod(
                    output_dir,
                    PRIVATE_DIRECTORY_MODE,
                    follow_symlinks=False,
                )
        except OSError:
            raise EvaluationConfigurationError(
                "output root could not be created securely"
            ) from None
        if output_dir.is_symlink():
            raise EvaluationConfigurationError("output root cannot be a symlink")
        try:
            resolved = output_dir.resolve(strict=True)
            root_stat = os.stat(output_dir, follow_symlinks=False)
        except OSError:
            raise EvaluationConfigurationError("output root is unavailable") from None
        cls._validate_directory_stat(root_stat, "output root")
        if created and stat.S_IMODE(root_stat.st_mode) != PRIVATE_DIRECTORY_MODE:
            raise EvaluationConfigurationError(
                "new output root permissions are not private"
            )
        if not created:
            try:
                if any(output_dir.iterdir()):
                    raise EvaluationConfigurationError(
                        "output directory is non-empty"
                    )
            except EvaluationConfigurationError:
                raise
            except OSError:
                raise EvaluationConfigurationError(
                    "output root is unavailable"
                ) from None
        guard = cls(
            output_dir,
            resolved,
            (root_stat.st_dev, root_stat.st_ino),
        )
        guard._assert_stable()
        return guard

    def _assert_stable(self) -> None:
        if self.root.is_symlink() or not self.root.is_dir():
            raise EvaluationConfigurationError("output root symlink changed")
        try:
            current = self.root.resolve(strict=True)
        except OSError:
            raise EvaluationConfigurationError("output root is unavailable") from None
        if current != self._resolved_root:
            raise EvaluationConfigurationError("output root resolution changed")
        try:
            root_stat = os.stat(self.root, follow_symlinks=False)
        except OSError:
            raise EvaluationConfigurationError("output root is unavailable") from None
        self._validate_directory_stat(root_stat, "output root")
        if (root_stat.st_dev, root_stat.st_ino) != self._root_identity:
            raise EvaluationConfigurationError("output root identity changed")

    @staticmethod
    def _parts(relative: str) -> Sequence[str]:
        if type(relative) is not str:
            raise EvaluationConfigurationError("controlled output path is invalid")
        path = Path(relative)
        if (
            not relative
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise EvaluationConfigurationError("controlled output path is invalid")
        return path.parts

    @staticmethod
    def _directory_open_flags() -> int:
        return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW

    @staticmethod
    def _close_descriptor(descriptor: Optional[int]) -> None:
        if descriptor is None:
            return
        try:
            os.close(descriptor)
        except OSError:
            pass

    def _open_root_fd(self) -> int:
        self._assert_stable()
        descriptor = None
        try:
            descriptor = os.open(self.root, self._directory_open_flags())
            root_stat = os.fstat(descriptor)
        except OSError:
            self._close_descriptor(descriptor)
            raise EvaluationConfigurationError(
                "output root could not be opened safely"
            ) from None
        except BaseException:
            self._close_descriptor(descriptor)
            raise
        try:
            self._validate_directory_stat(root_stat, "output root")
        except EvaluationConfigurationError:
            self._close_descriptor(descriptor)
            raise
        if (root_stat.st_dev, root_stat.st_ino) != self._root_identity:
            self._close_descriptor(descriptor)
            raise EvaluationConfigurationError("output root identity changed")
        return descriptor

    def _open_directory_chain(
        self,
        parts: Sequence[str],
        *,
        create: bool,
    ) -> int:
        current_fd = self._open_root_fd()
        try:
            for part in parts:
                next_fd = None
                created = False
                if create:
                    try:
                        os.mkdir(
                            part,
                            mode=PRIVATE_DIRECTORY_MODE,
                            dir_fd=current_fd,
                        )
                        created = True
                    except FileExistsError:
                        pass
                    if created:
                        os.chmod(
                            part,
                            PRIVATE_DIRECTORY_MODE,
                            dir_fd=current_fd,
                            follow_symlinks=False,
                        )
                try:
                    next_fd = os.open(
                        part,
                        self._directory_open_flags(),
                        dir_fd=current_fd,
                    )
                    if created:
                        os.fchmod(next_fd, PRIVATE_DIRECTORY_MODE)
                    next_stat = os.fstat(next_fd)
                    self._validate_directory_stat(
                        next_stat,
                        "controlled output directory",
                    )
                    if (
                        created
                        and stat.S_IMODE(next_stat.st_mode)
                        != PRIVATE_DIRECTORY_MODE
                    ):
                        raise EvaluationConfigurationError(
                            "new controlled directory permissions are not private"
                        )
                except BaseException:
                    self._close_descriptor(next_fd)
                    raise
                self._close_descriptor(current_fd)
                current_fd = next_fd
            return current_fd
        except EvaluationConfigurationError:
            self._close_descriptor(current_fd)
            raise
        except OSError:
            self._close_descriptor(current_fd)
            raise EvaluationConfigurationError(
                "controlled output path cannot traverse a symlink safely"
            ) from None
        except BaseException:
            self._close_descriptor(current_fd)
            raise

    def ensure_directory(self, relative: str) -> Path:
        parts = self._parts(relative)
        descriptor = self._open_directory_chain(parts, create=True)
        self._close_descriptor(descriptor)
        self._assert_stable()
        return self.root.joinpath(*parts)

    def _write_text(self, relative: str, content: str) -> None:
        parts = self._parts(relative)
        parent_fd = self._open_directory_chain(parts[:-1], create=True)
        file_fd = None
        try:
            file_fd = os.open(
                parts[-1],
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | os.O_NOFOLLOW,
                PRIVATE_FILE_MODE,
                dir_fd=parent_fd,
            )
            os.fchmod(file_fd, PRIVATE_FILE_MODE)
            opened_stat = os.fstat(file_fd)
            if opened_stat.st_uid != os.geteuid():
                raise EvaluationConfigurationError(
                    "controlled output file owner is not the current user"
                )
            if (
                not stat.S_ISREG(opened_stat.st_mode)
                or opened_stat.st_nlink != 1
                or stat.S_IMODE(opened_stat.st_mode) != PRIVATE_FILE_MODE
            ):
                raise EvaluationConfigurationError(
                    "controlled output file is not a regular file"
                )
            path_stat = os.stat(
                parts[-1],
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(path_stat.st_mode)
                or path_stat.st_uid != os.geteuid()
                or stat.S_IMODE(path_stat.st_mode) != PRIVATE_FILE_MODE
                or (path_stat.st_dev, path_stat.st_ino)
                != (opened_stat.st_dev, opened_stat.st_ino)
            ):
                raise EvaluationConfigurationError(
                    "controlled output file identity changed"
                )
            remaining = memoryview(content.encode("utf-8"))
            while remaining:
                written = os.write(file_fd, remaining)
                if written <= 0:
                    raise OSError("short output write")
                remaining = remaining[written:]
            final_stat = os.fstat(file_fd)
            final_path_stat = os.stat(
                parts[-1],
                dir_fd=parent_fd,
                follow_symlinks=False,
            )
            if (
                not stat.S_ISREG(final_stat.st_mode)
                or final_stat.st_uid != os.geteuid()
                or stat.S_IMODE(final_stat.st_mode) != PRIVATE_FILE_MODE
                or final_path_stat.st_uid != os.geteuid()
                or stat.S_IMODE(final_path_stat.st_mode) != PRIVATE_FILE_MODE
                or (final_stat.st_dev, final_stat.st_ino)
                != (final_path_stat.st_dev, final_path_stat.st_ino)
            ):
                raise EvaluationConfigurationError(
                    "controlled output file identity changed"
                )
        except FileExistsError:
            raise EvaluationConfigurationError(
                "controlled output file already exists"
            ) from None
        except EvaluationConfigurationError:
            raise
        except OSError:
            raise EvaluationConfigurationError(
                "controlled output file could not be written"
            ) from None
        finally:
            self._close_descriptor(file_fd)
            self._close_descriptor(parent_fd)
        self._assert_stable()

    def write_json(self, relative: str, payload: object) -> None:
        self._write_text(relative, _json_content(payload))

    def write_log(self, relative: str, content: str) -> None:
        if type(content) is not str or len(content.encode("utf-8")) > 4096:
            raise EvaluationConfigurationError("evaluation log is invalid")
        self._write_text(relative, content)


def _read_json(path: Path, label: str) -> object:
    try:
        if not path.is_file() or path.stat().st_size > MAX_JSON_INPUT_BYTES:
            raise EvaluationConfigurationError("%s is invalid" % label)
        content = path.read_text(encoding="utf-8")
        if len(content.encode("utf-8")) > MAX_JSON_INPUT_BYTES:
            raise EvaluationConfigurationError("%s is invalid" % label)
        return json.loads(content)
    except EvaluationConfigurationError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError):
        raise EvaluationConfigurationError("%s is invalid" % label) from None


def _validate_candidate_version(value: str) -> str:
    if type(value) is not str or _SAFE_VERSION.fullmatch(value) is None:
        raise EvaluationConfigurationError(
            "candidate version must be a bounded safe label"
        )
    return value


def _validate_cases(cases: object) -> List[Dict[str, object]]:
    if type(cases) is not list or not 1 <= len(cases) <= MAX_CASES:
        raise EvaluationConfigurationError("golden case count is out of bounds")
    case_ids = []
    validated = []
    for case in cases:
        if type(case) is not dict or not (
            set(case) == _CASE_KEYS
            or (
                case.get("case_id") == "parameter_root_01"
                and set(case) == _CASE_KEYS | _STRUCTURED_EXPECTATION_KEYS
            )
        ):
            raise EvaluationConfigurationError("golden case contract is invalid")
        case_id = case.get("case_id")
        if type(case_id) is not str or _SAFE_ID.fullmatch(case_id) is None:
            raise EvaluationConfigurationError("golden case id is invalid")
        problem = case.get("problem")
        if type(problem) is not dict or set(problem) != _PROBLEM_KEYS:
            raise EvaluationConfigurationError("golden case problem is invalid")
        if (
            any(
                type(problem[field]) is not str
                or not problem[field].strip()
                for field in (
                    "problem_text",
                    "reference_answer",
                    "reference_solution_text",
                )
            )
            or type(problem["lesson_length"]) is not str
            or problem["lesson_length"] not in {"concise", "standard"}
        ):
            raise EvaluationConfigurationError("golden case problem is invalid")
        try:
            ProblemInput.model_validate(problem)
        except Exception:
            raise EvaluationConfigurationError("golden case problem is invalid") from None
        for key in _CASE_KEYS - {"case_id", "problem"}:
            values = case.get(key)
            if (
                type(values) is not list
                or not 1 <= len(values) <= 8
                or any(
                    type(value) is not str
                    or value.strip() != value
                    or not 1 <= len(value) <= 160
                    for value in values
                )
            ):
                raise EvaluationConfigurationError(
                    "golden case metadata is invalid"
                )
            if len(values) != len(set(values)):
                raise EvaluationConfigurationError(
                    "golden case metadata is invalid"
                )
        if not set(case["coverage_tags"]).issubset(_COVERAGE_TAGS):
            raise EvaluationConfigurationError(
                "golden case coverage tags are invalid"
            )
        if not set(case["required_reasoning_modes"]).issubset(
            _REASONING_MODES
        ):
            raise EvaluationConfigurationError(
                "golden case reasoning modes are invalid"
            )
        if _STRUCTURED_EXPECTATION_KEYS <= set(case):
            labels = case["required_step_labels"]
            must_teach_anchors = case["required_must_teach_anchors"]
            spoken_forms = case["required_spoken_forms"]
            error_codes = case["required_error_codes"]
            if (
                type(labels) is not list
                or len(labels) != 5
                or any(
                    type(item) is not str
                    or item.strip() != item
                    or not item
                    for item in labels
                )
                or type(spoken_forms) is not list
                or len(spoken_forms) != 2
                or any(
                    type(item) is not dict
                    or set(item) != {"display", "spoken_contains"}
                    or any(
                        type(value) is not str
                        or value.strip() != value
                        or not value
                        for value in item.values()
                    )
                    for item in spoken_forms
                )
                or type(must_teach_anchors) is not list
                or len(must_teach_anchors) != 5
                or any(
                    type(item) is not dict
                    or set(item) != {
                        "content",
                        "display_anchor",
                        "spoken_anchor",
                    }
                    or any(
                        type(value) is not str
                        or value.strip() != value
                        or not value
                        or len(value) > 160
                        for value in item.values()
                    )
                    for item in must_teach_anchors
                )
                or len(
                    {item["content"] for item in must_teach_anchors}
                ) != len(must_teach_anchors)
                or {
                    item["content"] for item in must_teach_anchors
                } != set(case["required_must_teach"])
                or type(error_codes) is not list
                or len(error_codes) != 4
                or len(error_codes) != len(set(error_codes))
                or any(
                    type(item) is not str
                    or _SAFE_ID.fullmatch(item) is None
                    for item in error_codes
                )
            ):
                raise EvaluationConfigurationError(
                    "golden structured expectations are invalid"
                )
        case_ids.append(case_id)
        validated.append(case)
    if len(case_ids) != len(set(case_ids)):
        raise EvaluationConfigurationError("golden case ids must be unique")
    return validated


def _case_set_fingerprint(cases: Sequence[Dict[str, object]]) -> str:
    canonical = json.dumps(
        cases,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_structured_case_evidence(
    case: Dict[str, object],
    record: object,
) -> None:
    if not _STRUCTURED_EXPECTATION_KEYS <= set(case):
        return
    prepared = record.prepared_lesson
    progression = prepared.teaching_progression
    labels = [step.directory_label for step in progression.steps]
    if labels != case["required_step_labels"]:
        raise GoldenEvidenceError("required teaching step labels are missing")

    clauses = list(prepared.teaching_script.clauses) + [
        clause
        for response in prepared.teaching_script.response_scripts
        for clause in response.clauses
    ]
    must_teach = {
        item.content: item
        for episode in prepared.reasoning_trajectory.episodes
        for item in episode.must_teach
    }
    actions_by_clause = {}
    completed_steps = set()
    for cue in prepared.performance_score.cues:
        for field in ("lead_actions", "start_actions", "end_actions"):
            for binding in getattr(cue, field):
                action = binding.action
                if (
                    action.type in {"write", "transform"}
                    and action.surface == "board"
                ):
                    actions_by_clause.setdefault(binding.clause_id, []).append(
                        action.content or ""
                    )
                if action.type == "complete_step":
                    completed_steps.add(action.teaching_step_id)

    def contains_fixture_anchor(container: str, anchor: str) -> bool:
        normalized_anchor = normalize_answer_leak_text(anchor)
        return (
            bool(normalized_anchor)
            and normalized_anchor in normalize_answer_leak_text(container)
        ) or (
            normalize_cross_artifact_math_identity(anchor)
            in normalize_cross_artifact_math_identity(container)
        )

    for expected in case["required_must_teach_anchors"]:
        item = must_teach.get(expected["content"])
        if (
            item is None
            or not item.student_display_evidence
            or not item.student_spoken_evidence
            or not contains_fixture_anchor(
                item.student_display_evidence,
                expected["display_anchor"],
            )
            or not contains_fixture_anchor(
                item.student_spoken_evidence,
                expected["spoken_anchor"],
            )
        ):
            raise GoldenEvidenceError(
                "required must-teach structured evidence is missing"
            )
        evidence_clauses = [
            clause for clause in clauses
            if item.must_teach_id in clause.must_teach_refs
            and contains_fixture_anchor(
                clause.display_text or "", expected["display_anchor"]
            )
            and contains_fixture_anchor(
                clause.spoken_text, expected["spoken_anchor"]
            )
        ]
        if not evidence_clauses or not any(
            contains_fixture_anchor(
                action_content, expected["display_anchor"]
            )
            for clause in evidence_clauses
            for action_content in actions_by_clause.get(clause.clause_id, [])
        ):
            raise GoldenEvidenceError(
                "required must-teach script and board evidence is missing"
            )

    if completed_steps != {step.step_id for step in progression.steps}:
        raise GoldenEvidenceError("teaching step completion evidence is missing")

    responses = list(prepared.teaching_script.response_scripts)
    if not any(
        response.classification == "correct"
        and response.depth == "brief"
        for response in responses
    ):
        raise GoldenEvidenceError("brief correct response evidence is missing")
    if not any(
        response.classification == "incorrect"
        and response.depth in {"conceptual", "worked"}
        and response.clauses
        for response in responses
    ):
        raise GoldenEvidenceError("deeper wrong-answer support evidence is missing")

    for expected in case["required_spoken_forms"]:
        if not any(
            expected["display"] in clause.display_text
            and expected["spoken_contains"] in clause.spoken_text
            for clause in clauses
        ):
            raise GoldenEvidenceError(
                "required display and spoken form evidence is missing"
            )

    actual_error_codes = {
        option.error_code
        for interaction in prepared.interaction_plan.interactions
        for option in interaction.options
        if option.error_code is not None
    }
    if not set(case["required_error_codes"]).issubset(actual_error_codes):
        raise GoldenEvidenceError("required diagnostic error code is missing")


def _model_dump(value: object) -> Dict[str, object]:
    dump = getattr(value, "model_dump", None)
    if not callable(dump):
        raise TypeError("evaluation artifact is not serializable")
    payload = dump(mode="json")
    if type(payload) is not dict:
        raise TypeError("evaluation artifact payload is invalid")
    return payload


def _project_public_option(payload: object) -> Dict[str, object]:
    if type(payload) is not dict:
        raise ValueError("public option source is invalid")
    return {
        "option_id": payload.get("option_id"),
        "label": payload.get("label"),
    }


def _project_public_interaction(payload: object) -> Optional[Dict[str, object]]:
    if payload is None:
        return None
    if type(payload) is not dict:
        raise ValueError("public interaction source is invalid")
    options = payload.get("options")
    hints = payload.get("hints")
    if type(options) is not list or type(hints) is not list:
        raise ValueError("public interaction source is invalid")
    return {
        "interaction_id": payload.get("interaction_id"),
        "kind": payload.get("kind"),
        "prompt": payload.get("prompt"),
        "options": [_project_public_option(item) for item in options],
        "hints": list(hints),
    }


def _public_runtime_payload(lesson: object) -> Dict[str, object]:
    source = _model_dump(lesson)
    problem = source.get("problem")
    transfer = source.get("transfer_item")
    beats = source.get("beats")
    if type(problem) is not dict or type(transfer) is not dict or type(beats) is not list:
        raise ValueError("public runtime source is invalid")
    transfer_options = transfer.get("options")
    if type(transfer_options) is not list:
        raise ValueError("public transfer source is invalid")

    projected_beats = []
    for beat in beats:
        if type(beat) is not dict or type(beat.get("sync_cues")) is not list:
            raise ValueError("public beat source is invalid")
        projected_cues = []
        for cue in beat["sync_cues"]:
            if type(cue) is not dict:
                raise ValueError("public cue source is invalid")
            action_fields = {}
            for field in ("lead_actions", "start_actions", "end_actions"):
                actions = cue.get(field)
                if type(actions) is not list:
                    raise ValueError("public cue source is invalid")
                action_fields[field] = actions
            projected_cues.append({
                "cue_id": cue.get("cue_id"),
                "spoken_text": cue.get("spoken_text"),
                **action_fields,
            })
        projected_beats.append({
            "beat_id": beat.get("beat_id"),
            "purpose": beat.get("purpose"),
            "narration": beat.get("narration"),
            "layer": beat.get("layer"),
            "sync_cues": projected_cues,
            "interaction": _project_public_interaction(
                beat.get("interaction")
            ),
            "next_beat_id": beat.get("next_beat_id"),
        })

    artifact = _PublicEvaluationArtifact.model_validate({
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "lesson_id": source.get("lesson_id"),
        "problem": {
            "problem_text": problem.get("problem_text"),
            "required_method": problem.get("required_method"),
            "lesson_length": problem.get("lesson_length"),
        },
        "title": source.get("title"),
        "learning_goal": source.get("learning_goal"),
        "beats": projected_beats,
        "summary": source.get("summary"),
        "transfer_item": {
            "problem_text": transfer.get("problem_text"),
            "method_signal": transfer.get("method_signal"),
            "options": [
                _project_public_option(item)
                for item in transfer_options
            ],
        },
    }, strict=True)
    public_payload = artifact.model_dump(mode="json")
    _assert_public_keys_safe(public_payload)
    return public_payload


def _validate_public_artifact(payload: object) -> Dict[str, object]:
    try:
        artifact = _PublicEvaluationArtifact.model_validate(
            payload,
            strict=True,
        )
    except Exception:
        raise EvaluationConfigurationError(
            "comparison public artifact schema is invalid"
        ) from None
    public_payload = artifact.model_dump(mode="json")
    _assert_public_keys_safe(public_payload)
    return public_payload


def _public_artifact_sha256(payload: object) -> str:
    return hashlib.sha256(
        _json_content(payload).encode("utf-8")
    ).hexdigest()


def _read_public_artifact(path: Path) -> tuple:
    if path.is_symlink():
        raise EvaluationConfigurationError(
            "comparison public artifact cannot be a symlink"
        )
    try:
        if not path.is_file() or path.stat().st_size > MAX_JSON_INPUT_BYTES:
            raise EvaluationConfigurationError(
                "comparison public runtime is invalid"
            )
        content = path.read_bytes()
        if len(content) > MAX_JSON_INPUT_BYTES:
            raise EvaluationConfigurationError(
                "comparison public runtime is invalid"
            )
        payload = json.loads(content.decode("utf-8"))
    except EvaluationConfigurationError:
        raise
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        RecursionError,
    ):
        raise EvaluationConfigurationError(
            "comparison public runtime is invalid"
        ) from None
    return payload, hashlib.sha256(content).hexdigest()


def _assert_public_keys_safe(payload: object) -> None:
    if type(payload) is dict:
        for key, value in payload.items():
            if type(key) is not str or key in _FORBIDDEN_PUBLIC_KEYS:
                raise EvaluationConfigurationError(
                    "public artifact contains a forbidden private field"
                )
            _assert_public_keys_safe(value)
    elif type(payload) is list:
        for value in payload:
            _assert_public_keys_safe(value)
    elif payload is not None and type(payload) not in {str, int, float, bool}:
        raise EvaluationConfigurationError("public artifact value is invalid")


def _contract_metrics(
    lesson: object,
    record: object,
    duration_ms: int,
    rubric_version: str,
    call_count: Optional[int] = None,
) -> Dict[str, object]:
    prepared = record.prepared_lesson
    if prepared.rubric_version != rubric_version:
        raise ValueError("generated rubric version does not match requested version")

    required_must_teach = {
        item.must_teach_id
        for episode in prepared.reasoning_trajectory.episodes
        for item in episode.must_teach
    }
    referenced_must_teach = {
        item
        for clause in prepared.teaching_script.clauses
        for item in clause.must_teach_refs
    }
    covered = len(required_must_teach & referenced_must_teach)
    must_teach_metric = {
        "covered": covered,
        "total": len(required_must_teach),
        "ratio": (
            covered / len(required_must_teach)
            if required_must_teach
            else 1.0
        ),
    }

    response_scripts = list(
        getattr(prepared.teaching_script, "response_scripts", []) or []
    )
    response_clauses = [
        clause
        for response in response_scripts
        for clause in response.clauses
    ]
    all_clauses = list(prepared.teaching_script.clauses) + response_clauses
    script_clause_ids = {clause.clause_id for clause in all_clauses}
    all_actions = []
    valid_actions = 0
    for cue in prepared.performance_score.cues:
        cue_clause_ids = set(cue.clause_ids)
        for field in ("lead_actions", "start_actions", "end_actions"):
            for action in getattr(cue, field):
                all_actions.append(action)
                if (
                    action.clause_id in cue_clause_ids
                    and action.clause_id in script_clause_ids
                ):
                    valid_actions += 1
    binding_metric = {
        "valid": valid_actions,
        "total": len(all_actions),
        "ratio": (
            valid_actions / len(all_actions)
            if all_actions
            else 1.0
        ),
    }

    progression = getattr(prepared, "teaching_progression", None)
    steps = list(getattr(progression, "steps", []) or [])
    step_ids = {step.step_id for step in steps}
    covered_step_ids = {
        getattr(clause, "lesson_step_id", None)
        for clause in prepared.teaching_script.clauses
    } & step_ids
    step_metric = _ratio_metric("covered", len(covered_step_ids), len(step_ids))

    aligned = 0
    alignment_total = 0
    for clause in all_clauses:
        display_text = getattr(clause, "display_text", None)
        spoken_text = getattr(clause, "spoken_text", None)
        if not isinstance(display_text, str) or not isinstance(spoken_text, str):
            continue
        alignment_total += 1
        try:
            validate_display_spoken_alignment(display_text, spoken_text)
        except MathSpeechError:
            continue
        aligned += 1
    display_speech_metric = _ratio_metric(
        "aligned", aligned, alignment_total
    )

    board_clause_ids = set()
    step_lifecycle = {step_id: set() for step_id in step_ids}
    for cue in prepared.performance_score.cues:
        for field in ("lead_actions", "start_actions", "end_actions"):
            for binding in getattr(cue, field):
                action = getattr(binding, "action", None)
                action_type = getattr(action, "type", None)
                if (
                    getattr(action, "surface", None) == "board"
                    and action_type in {"write", "transform"}
                ):
                    board_clause_ids.add(binding.clause_id)
                action_step_id = getattr(action, "teaching_step_id", None)
                if (
                    action_step_id in step_lifecycle
                    and action_type in {"reveal_step_header", "complete_step"}
                ):
                    step_lifecycle[action_step_id].add(action_type)
    board_must_teach = {
        must_teach_id
        for clause in all_clauses
        if clause.clause_id in board_clause_ids
        for must_teach_id in clause.must_teach_refs
    }
    must_teach_to_board_metric = _ratio_metric(
        "covered",
        len(required_must_teach & board_must_teach),
        len(required_must_teach),
    )
    complete_steps = sum(
        actions == {"reveal_step_header", "complete_step"}
        for actions in step_lifecycle.values()
    )
    lifecycle_metric = _ratio_metric(
        "complete", complete_steps, len(step_ids)
    )

    responses_by_pair = {
        (response.interaction_id, response.option_id): response
        for response in response_scripts
    }
    interactions = list(
        getattr(
            getattr(prepared, "interaction_plan", None),
            "interactions",
            [],
        )
        or []
    )
    diagnostic_covered = 0
    for interaction in interactions:
        branches_valid = True
        for option in interaction.options:
            response = responses_by_pair.get(
                (interaction.interaction_id, option.option_id)
            )
            if response is None:
                branches_valid = False
                break
            is_correct = option.option_id == interaction.correct_option_id
            expected_depth = (
                "brief" if is_correct else option.remediation_depth
            )
            if response.depth != expected_depth or not response.clauses:
                branches_valid = False
                break
        diagnostic_covered += int(branches_valid)
    diagnostic_metric = _ratio_metric(
        "covered", diagnostic_covered, len(interactions)
    )

    schema_runtime_pass = False
    if isinstance(lesson, RuntimeLesson) and isinstance(record, GenerationRecord):
        validate_lesson_generation_pair(lesson, record)
        schema_runtime_pass = True
    else:
        _model_dump(lesson)
        _model_dump(record)
        schema_runtime_pass = True

    return {
        "generation_success": True,
        "hard_gate_review_pass": prepared.review.status == "approved",
        "must_teach_coverage": must_teach_metric,
        "step_coverage": step_metric,
        "must_teach_to_script_coverage": must_teach_metric.copy(),
        "must_teach_to_board_coverage": must_teach_to_board_metric,
        "display_speech_alignment": display_speech_metric,
        "diagnostic_branch_coverage": diagnostic_metric,
        "step_lifecycle_coverage": lifecycle_metric,
        "clause_action_binding": binding_metric,
        "schema_runtime_pass": schema_runtime_pass,
        "duration_ms": duration_ms,
        "call_count": (
            call_count if call_count is not None else len(record.role_calls)
        ),
    }


def _ratio_metric(
    count_key: str,
    count: int,
    total: int,
) -> Dict[str, object]:
    return {
        count_key: count,
        "total": total,
        "ratio": count / total if total else 1.0,
    }


def _safe_failure(error: BaseException) -> str:
    category = getattr(error, "category", None)
    if category in _SAFE_FAILURE_CATEGORIES:
        return category
    try:
        return InternalGenerationDiagnostic(category=category).category
    except Exception:
        return "invalid_structure"


def _safe_stage(stage: str) -> str:
    return _STAGE_CODES.get(stage, "generation")


def _validate_run_inputs(
    cases: Sequence[Dict[str, object]],
    rubric_version: str,
    runs_per_case: int,
) -> None:
    if os.getenv("RUN_INTEGRATION") != "1":
        raise EvaluationConfigurationError(
            "RUN_INTEGRATION=1 is required for live evaluation"
        )
    if type(rubric_version) is not str or _SAFE_VERSION.fullmatch(rubric_version) is None:
        raise EvaluationConfigurationError("rubric version must be explicit")
    if type(runs_per_case) is not int or not 1 <= runs_per_case <= MAX_RUNS_PER_CASE:
        raise EvaluationConfigurationError("runs per case must be from 1 to 10")
    _validate_cases(cases)


async def _run_evaluation_async(
    cases: Sequence[Dict[str, object]],
    *,
    rubric_version: str,
    runs_per_case: int,
    output: _OutputGuard,
    service_factory: Callable[[], object],
    clock: Callable[[], float],
    candidate_version: str,
) -> None:
    output.ensure_directory("private/records")
    output.ensure_directory("public/runtime")
    runs: List[Dict[str, object]] = []
    service = service_factory()
    try:
        for case in cases:
            case_id = str(case["case_id"])
            problem = ProblemInput.model_validate(case["problem"])
            for run_index in range(1, runs_per_case + 1):
                stem = "%s__run-%02d" % (case_id, run_index)
                stage = "generation"

                async def on_stage(value: str) -> None:
                    nonlocal stage
                    stage = _safe_stage(value)

                client = getattr(service, "client", None)
                before_calls = getattr(client, "call_count", None)
                started = clock()
                try:
                    bundle = await service.generate_bundle(
                        problem,
                        on_stage=on_stage,
                    )
                    _validate_structured_case_evidence(
                        case, bundle.generation_record
                    )
                    duration_ms = max(0, round((clock() - started) * 1000))
                    metrics = _contract_metrics(
                        bundle.lesson,
                        bundle.generation_record,
                        duration_ms,
                        rubric_version,
                        call_count=(
                            getattr(client, "call_count", 0) - before_calls
                            if type(before_calls) is int
                            else None
                        ),
                    )
                    public_payload = _public_runtime_payload(bundle.lesson)
                    public_sha256 = _public_artifact_sha256(public_payload)
                    output.write_json(
                        "private/records/%s.json" % stem,
                        _model_dump(bundle.generation_record),
                    )
                    output.write_json(
                        "public/runtime/%s.json" % stem,
                        public_payload,
                    )
                    runs.append({
                        "case_id": case_id,
                        "run_index": run_index,
                        "status": "succeeded",
                        "public_sha256": public_sha256,
                        "metrics": metrics,
                    })
                except (KeyboardInterrupt, asyncio.CancelledError):
                    raise
                except EvaluationConfigurationError:
                    raise
                except Exception as error:
                    duration_ms = max(0, round((clock() - started) * 1000))
                    runs.append({
                        "case_id": case_id,
                        "run_index": run_index,
                        "status": "failed",
                        "failure": {
                            "category": _safe_failure(error),
                            "stage": stage,
                        },
                        "metrics": {
                            "generation_success": False,
                            "hard_gate_review_pass": False,
                            "must_teach_coverage": None,
                            "step_coverage": None,
                            "must_teach_to_script_coverage": None,
                            "must_teach_to_board_coverage": None,
                            "display_speech_alignment": None,
                            "diagnostic_branch_coverage": None,
                            "step_lifecycle_coverage": None,
                            "clause_action_binding": None,
                            "schema_runtime_pass": False,
                            "duration_ms": duration_ms,
                            "call_count": (
                                getattr(client, "call_count", 0) - before_calls
                                if type(before_calls) is int
                                else None
                            ),
                        },
                    })
    finally:
        client = getattr(service, "client", None)
        close = getattr(client, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result

    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "rubric_version": rubric_version,
        "candidate_version": candidate_version,
        "runs_per_case": runs_per_case,
        "case_ids": [case["case_id"] for case in cases],
        "case_set_sha256": _case_set_fingerprint(cases),
        "metric_scope": list(_METRIC_SCOPE),
        "evidence_boundary": (
            "Deterministic generation contracts only; no teacher preference "
            "or student learning inference."
        ),
        "runs": runs,
    }
    output.write_json("manifest.json", manifest)
    output.write_log(
        "run.log",
        "Evaluation completed. Provider content and credentials are not logged.\n",
    )


def run_evaluation(
    cases: Sequence[Dict[str, object]],
    *,
    rubric_version: str,
    runs_per_case: int,
    output_dir: Path,
    service_factory: Callable[[], object],
    clock: Callable[[], float] = time.monotonic,
    candidate_version: Optional[str] = None,
) -> None:
    _validate_run_inputs(cases, rubric_version, runs_per_case)
    resolved_candidate = _validate_candidate_version(
        candidate_version or rubric_version
    )
    output = _OutputGuard.create(output_dir)
    asyncio.run(_run_evaluation_async(
        cases,
        rubric_version=rubric_version,
        runs_per_case=runs_per_case,
        output=output,
        service_factory=service_factory,
        clock=clock,
        candidate_version=resolved_candidate,
    ))


def _read_manifest(run_dir: Path) -> Dict[str, object]:
    payload = _read_json(
        run_dir / "manifest.json",
        "comparison run manifest",
    )
    manifest_keys = {
        "schema_version",
        "rubric_version",
        "candidate_version",
        "runs_per_case",
        "case_ids",
        "case_set_sha256",
        "metric_scope",
        "evidence_boundary",
        "runs",
    }
    if type(payload) is not dict or set(payload) != manifest_keys:
        raise EvaluationConfigurationError("comparison run manifest is invalid")
    rubric_version = payload["rubric_version"]
    candidate_version = payload["candidate_version"]
    runs_per_case = payload["runs_per_case"]
    case_ids = payload["case_ids"]
    runs = payload["runs"]
    metric_scope = payload["metric_scope"]
    expected_metric_scope = list(_METRIC_SCOPE)
    if (
        payload["schema_version"] != MANIFEST_SCHEMA_VERSION
        or type(rubric_version) is not str
        or _SAFE_VERSION.fullmatch(rubric_version) is None
        or type(candidate_version) is not str
        or _SAFE_VERSION.fullmatch(candidate_version) is None
        or type(runs_per_case) is not int
        or not 1 <= runs_per_case <= MAX_RUNS_PER_CASE
        or type(case_ids) is not list
        or not 1 <= len(case_ids) <= MAX_CASES
        or type(runs) is not list
        or type(metric_scope) is not list
        or metric_scope != expected_metric_scope
        or type(payload["evidence_boundary"]) is not str
        or not payload["evidence_boundary"].strip()
        or type(payload["case_set_sha256"]) is not str
        or re.fullmatch(r"[0-9a-f]{64}", payload["case_set_sha256"])
        is None
    ):
        raise EvaluationConfigurationError("comparison run manifest is invalid")
    if any(
        type(case_id) is not str or _SAFE_ID.fullmatch(case_id) is None
        for case_id in case_ids
    ):
        raise EvaluationConfigurationError("comparison run manifest is invalid")
    if len(case_ids) != len(set(case_ids)):
        raise EvaluationConfigurationError("comparison run manifest is invalid")

    expected_identities = [
        (case_id, run_index)
        for case_id in case_ids
        for run_index in range(1, runs_per_case + 1)
    ]
    if len(runs) != len(expected_identities):
        raise EvaluationConfigurationError(
            "comparison run manifest is not a complete run matrix"
        )
    actual_identities = []
    for item in runs:
        if type(item) is not dict:
            raise EvaluationConfigurationError(
                "comparison run manifest is invalid"
            )
        case_id = item.get("case_id")
        run_index = item.get("run_index")
        status = item.get("status")
        if (
            type(case_id) is not str
            or case_id not in case_ids
            or type(run_index) is not int
            or not 1 <= run_index <= runs_per_case
            or type(status) is not str
            or status not in {"succeeded", "failed"}
        ):
            raise EvaluationConfigurationError(
                "comparison run manifest is invalid"
            )
        expected_keys = (
            {"case_id", "run_index", "status", "public_sha256", "metrics"}
            if status == "succeeded"
            else {"case_id", "run_index", "status", "failure", "metrics"}
        )
        if set(item) != expected_keys:
            raise EvaluationConfigurationError(
                "comparison run manifest is invalid"
            )
        _validate_manifest_metrics(item["metrics"], status)
        if status == "succeeded":
            digest = item["public_sha256"]
            if (
                type(digest) is not str
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            ):
                raise EvaluationConfigurationError(
                    "comparison run manifest is invalid"
                )
        else:
            failure = item["failure"]
            if (
                type(failure) is not dict
                or set(failure) != {"category", "stage"}
                or type(failure.get("category")) is not str
                or failure["category"] not in _SAFE_FAILURE_CATEGORIES
                or type(failure.get("stage")) is not str
                or not failure["stage"].strip()
                or len(failure["stage"]) > 64
            ):
                raise EvaluationConfigurationError(
                    "comparison run manifest is invalid"
                )
        actual_identities.append((case_id, run_index))
    if actual_identities != expected_identities:
        raise EvaluationConfigurationError(
            "comparison run manifest is not a complete ordered run matrix"
        )
    return payload


def _validate_manifest_metrics(payload: object, status: str) -> None:
    keys = {
        "generation_success",
        "hard_gate_review_pass",
        *_METRIC_COUNT_KEYS,
        "schema_runtime_pass",
        "duration_ms",
        "call_count",
    }
    if type(payload) is not dict or set(payload) != keys:
        raise EvaluationConfigurationError("comparison run metrics are invalid")
    if status not in {"succeeded", "failed"}:
        raise EvaluationConfigurationError("comparison run metrics are invalid")
    expected_success = status == "succeeded"
    if (
        type(payload["generation_success"]) is not bool
        or payload["generation_success"] is not expected_success
        or type(payload["hard_gate_review_pass"]) is not bool
        or payload["hard_gate_review_pass"] is not expected_success
        or type(payload["schema_runtime_pass"]) is not bool
        or payload["schema_runtime_pass"] is not expected_success
        or type(payload["duration_ms"]) is not int
        or payload["duration_ms"] < 0
        or (
            payload["call_count"] is not None
            and (
                type(payload["call_count"]) is not int
                or payload["call_count"] < 0
            )
        )
    ):
        raise EvaluationConfigurationError("comparison run metrics are invalid")
    for field, count_key in _METRIC_COUNT_KEYS.items():
        metric = payload[field]
        if not expected_success:
            if metric is not None:
                raise EvaluationConfigurationError(
                    "comparison run metrics are invalid"
                )
            continue
        if (
            type(metric) is not dict
            or set(metric) != {count_key, "total", "ratio"}
            or type(metric[count_key]) is not int
            or type(metric["total"]) is not int
            or metric[count_key] < 0
            or metric["total"] < metric[count_key]
            or type(metric["ratio"]) is not float
            or not math.isfinite(metric["ratio"])
            or metric["ratio"]
            != (
                metric[count_key] / metric["total"]
                if metric["total"]
                else 1.0
            )
        ):
            raise EvaluationConfigurationError(
                "comparison run metrics are invalid"
            )


def create_blind_comparison(left: Path, right: Path, output_dir: Path) -> None:
    left = left.resolve()
    right = right.resolve()
    if left == right:
        raise EvaluationConfigurationError("comparison requires two different runs")
    left_manifest = _read_manifest(left)
    right_manifest = _read_manifest(right)
    _validate_candidate_version(left_manifest["candidate_version"])
    _validate_candidate_version(right_manifest["candidate_version"])
    if left_manifest["candidate_version"] == right_manifest["candidate_version"]:
        raise EvaluationConfigurationError("comparison candidate versions must differ")
    if (
        left_manifest["case_ids"] != right_manifest["case_ids"]
        or left_manifest["runs_per_case"] != right_manifest["runs_per_case"]
        or left_manifest.get("case_set_sha256")
        != right_manifest.get("case_set_sha256")
    ):
        raise EvaluationConfigurationError("comparison run contracts do not match")
    output = _OutputGuard.create(output_dir)

    left_by_identity = {
        (item["case_id"], item["run_index"]): item
        for item in left_manifest["runs"]
    }
    right_by_identity = {
        (item["case_id"], item["run_index"]): item
        for item in right_manifest["runs"]
    }
    keys = [
        (case_id, run_index)
        for case_id in left_manifest["case_ids"]
        for run_index in range(1, left_manifest["runs_per_case"] + 1)
    ]
    left_successes = sum(
        item["status"] == "succeeded"
        for item in left_manifest["runs"]
    )
    right_successes = sum(
        item["status"] == "succeeded"
        for item in right_manifest["runs"]
    )
    both_success = 0
    both_failed = 0
    one_sided_failure = 0
    pairs = []
    mappings = []
    for case_id, run_index in keys:
        left_run = left_by_identity[(case_id, run_index)]
        right_run = right_by_identity[(case_id, run_index)]
        if left_run["status"] != "succeeded" or right_run["status"] != "succeeded":
            if left_run["status"] == right_run["status"] == "failed":
                both_failed += 1
            else:
                one_sided_failure += 1
            continue
        both_success += 1
        stem = "%s__run-%02d" % (case_id, run_index)
        left_raw, left_sha256 = _read_public_artifact(
            left / "public" / "runtime" / (stem + ".json")
        )
        right_raw, right_sha256 = _read_public_artifact(
            right / "public" / "runtime" / (stem + ".json")
        )
        if (
            left_sha256 != left_run["public_sha256"]
            or right_sha256 != right_run["public_sha256"]
        ):
            raise EvaluationConfigurationError(
                "comparison public artifact hash mismatch"
            )
        left_payload = _validate_public_artifact(left_raw)
        right_payload = _validate_public_artifact(right_raw)
        swap = hashlib.sha256(stem.encode("utf-8")).digest()[0] % 2 == 1
        candidates = [right_payload, left_payload] if swap else [left_payload, right_payload]
        labels = ["right", "left"] if swap else ["left", "right"]
        pairs.append({
            "pair_id": stem,
            "case_id": case_id,
            "run_index": run_index,
            "candidate_a": candidates[0],
            "candidate_b": candidates[1],
        })
        mappings.append({
            "pair_id": stem,
            "candidate_a": {
                "source": labels[0],
                "rubric_version": (
                    left_manifest["rubric_version"]
                    if labels[0] == "left"
                    else right_manifest["rubric_version"]
                ),
                "candidate_version": (
                    left_manifest["candidate_version"]
                    if labels[0] == "left"
                    else right_manifest["candidate_version"]
                ),
            },
            "candidate_b": {
                "source": labels[1],
                "rubric_version": (
                    left_manifest["rubric_version"]
                    if labels[1] == "left"
                    else right_manifest["rubric_version"]
                ),
                "candidate_version": (
                    left_manifest["candidate_version"]
                    if labels[1] == "left"
                    else right_manifest["candidate_version"]
                ),
            },
        })
    excluded = both_failed + one_sided_failure
    output.write_json("public/blind_pairs.json", pairs)
    output.write_json("private/candidate_mapping.json", mappings)
    output.write_json("manifest.json", {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "pair_count": len(pairs),
        "case_ids": left_manifest["case_ids"],
        "runs_per_case": left_manifest["runs_per_case"],
        "comparison_counts": {
            "matched_runs": len(keys),
            "left_successes": left_successes,
            "left_failures": len(keys) - left_successes,
            "right_successes": right_successes,
            "right_failures": len(keys) - right_successes,
            "both_success": both_success,
            "both_failed": both_failed,
            "one_sided_failure": one_sided_failure,
            "excluded_from_blind_pairs": excluded,
            "blind_pairs": len(pairs),
        },
        "evidence_boundary": (
            "Candidate labels are blinded. A teacher may record a pairwise "
            "preference; this file contains no inferred preference or learning effect."
        ),
    })


def _load_cases(path: Path) -> List[Dict[str, object]]:
    return _validate_cases(_read_json(path, "golden fixture"))


def _real_service_factory(settings: Settings) -> Callable[[], LessonGenerationService]:
    def create() -> LessonGenerationService:
        client = CountingModelClient(OpenAICompatibleClient(settings))
        return LessonGenerationService(
            client,
            MathEngine(),
        )

    return create


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.compare_run is not None:
        if args.runs_per_case is not None or args.candidate_version is not None:
            raise SystemExit(
                "--runs-per-case and --candidate-version are only valid in generation mode"
            )
        create_blind_comparison(
            args.compare_run[0],
            args.compare_run[1],
            args.output_dir,
        )
        return
    if args.runs_per_case is None:
        raise SystemExit("--runs-per-case is required in generation mode")
    if os.getenv("RUN_INTEGRATION") != "1":
        raise SystemExit("RUN_INTEGRATION=1 is required; no network request was made")
    settings = Settings.from_env()
    if not settings.model_configured:
        raise SystemExit(
            "Missing model configuration: %s; no network request was made"
            % ", ".join(settings.missing_model_settings)
        )
    if args.rubric_version != PEDAGOGY_RUBRIC_VERSION:
        raise SystemExit(
            "--rubric-version must match this checkout (%s)"
            % PEDAGOGY_RUBRIC_VERSION
        )
    run_evaluation(
        _load_cases(args.fixture),
        rubric_version=args.rubric_version,
        runs_per_case=args.runs_per_case,
        output_dir=args.output_dir,
        service_factory=_real_service_factory(settings),
        candidate_version=args.candidate_version,
    )


if __name__ == "__main__":
    main()
