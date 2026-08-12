#!/usr/bin/env python3
"""Run bounded pedagogy contract evaluations without claiming learning effects."""

import argparse
import asyncio
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Callable, Dict, List, Optional, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.config import Settings
from app.generation import LessonGenerationService
from app.generation_diagnostics import InternalGenerationDiagnostic
from app.generation_integrity import validate_lesson_generation_pair
from app.llm_client import OpenAICompatibleClient
from app.math_engine import MathEngine
from app.pedagogy_rubric import PEDAGOGY_RUBRIC_VERSION
from app.preparation_models import GenerationRecord
from app.schemas import ProblemInput, RuntimeLesson


DEFAULT_FIXTURE = REPOSITORY_ROOT / "tests" / "fixtures" / "pedagogy_golden_cases.json"
MAX_CASES = 64
MAX_RUNS_PER_CASE = 10
MAX_JSON_INPUT_BYTES = 16 * 1024 * 1024
MAX_JSON_OUTPUT_BYTES = 32 * 1024 * 1024
MANIFEST_SCHEMA_VERSION = 1
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
_PROBLEM_KEYS = {
    "problem_text",
    "reference_answer",
    "reference_solution_text",
    "lesson_length",
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


class EvaluationConfigurationError(RuntimeError):
    """Safe operator error raised before or independently of provider calls."""


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


def _ensure_empty_output(output_dir: Path) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise EvaluationConfigurationError("output path is not a directory")
        if any(output_dir.iterdir()):
            raise EvaluationConfigurationError("output directory is non-empty")
    else:
        output_dir.mkdir(parents=True)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    if len(content.encode("utf-8")) > MAX_JSON_OUTPUT_BYTES:
        raise EvaluationConfigurationError("evaluation JSON output exceeds size limit")
    path.write_text(content, encoding="utf-8")


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
    except (OSError, UnicodeError, json.JSONDecodeError):
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
        if type(case) is not dict or set(case) != _CASE_KEYS:
            raise EvaluationConfigurationError("golden case contract is invalid")
        case_id = case.get("case_id")
        if type(case_id) is not str or _SAFE_ID.fullmatch(case_id) is None:
            raise EvaluationConfigurationError("golden case id is invalid")
        problem = case.get("problem")
        if type(problem) is not dict or set(problem) != _PROBLEM_KEYS:
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
                or len(values) != len(set(values))
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


def _model_dump(value: object) -> Dict[str, object]:
    dump = getattr(value, "model_dump", None)
    if not callable(dump):
        raise TypeError("evaluation artifact is not serializable")
    payload = dump(mode="json")
    if type(payload) is not dict:
        raise TypeError("evaluation artifact payload is invalid")
    return payload


def _public_runtime_payload(lesson: object) -> Dict[str, object]:
    payload = _model_dump(lesson)
    problem = payload.get("problem")
    if type(problem) is dict:
        problem.pop("reference_answer", None)
        problem.pop("reference_solution_text", None)
    transfer = payload.get("transfer_item")
    if type(transfer) is dict:
        transfer.pop("expected_answer", None)
        transfer.pop("correct_option_id", None)
        for option in transfer.get("options", []):
            if type(option) is dict:
                option.pop("canonical_answer", None)
                option.pop("feedback", None)
    payload.pop("validation_report", None)
    beats = payload.get("beats", [])
    if type(beats) is list:
        for beat in beats:
            if type(beat) is not dict:
                continue
            interaction = beat.get("interaction")
            if type(interaction) is not dict:
                continue
            interaction.pop("expected_answer", None)
            interaction.pop("explanation_after_correct", None)
            interaction.pop("correct_audio_url", None)
            for option in interaction.get("options", []):
                if type(option) is dict:
                    option.pop("feedback", None)
                    option.pop("feedback_audio_url", None)
    return payload


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
            round(covered / len(required_must_teach), 6)
            if required_must_teach
            else 1.0
        ),
    }

    script_clause_ids = {
        clause.clause_id for clause in prepared.teaching_script.clauses
    }
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
            round(valid_actions / len(all_actions), 6)
            if all_actions
            else 1.0
        ),
    }

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
        "clause_action_binding": binding_metric,
        "schema_runtime_pass": schema_runtime_pass,
        "duration_ms": duration_ms,
        "call_count": (
            call_count if call_count is not None else len(record.role_calls)
        ),
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
    output_dir: Path,
    service_factory: Callable[[], object],
    clock: Callable[[], float],
    candidate_version: str,
) -> None:
    private_root = output_dir / "private" / "records"
    public_root = output_dir / "public" / "runtime"
    private_root.mkdir(parents=True)
    public_root.mkdir(parents=True)
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
                    _write_json(
                        private_root / (stem + ".json"),
                        _model_dump(bundle.generation_record),
                    )
                    _write_json(
                        public_root / (stem + ".json"),
                        _public_runtime_payload(bundle.lesson),
                    )
                    runs.append({
                        "case_id": case_id,
                        "run_index": run_index,
                        "status": "succeeded",
                        "metrics": metrics,
                    })
                except (KeyboardInterrupt, asyncio.CancelledError):
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
        "metric_scope": [
            "generation_success",
            "hard_gate_review_pass",
            "must_teach_coverage",
            "clause_action_binding",
            "schema_runtime_pass",
            "duration_ms",
            "call_count",
        ],
        "evidence_boundary": (
            "Deterministic generation contracts only; no teacher preference "
            "or student learning inference."
        ),
        "runs": runs,
    }
    _write_json(output_dir / "manifest.json", manifest)
    (output_dir / "run.log").write_text(
        "Evaluation completed. Provider content and credentials are not logged.\n",
        encoding="utf-8",
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
    _ensure_empty_output(output_dir)
    asyncio.run(_run_evaluation_async(
        cases,
        rubric_version=rubric_version,
        runs_per_case=runs_per_case,
        output_dir=output_dir,
        service_factory=service_factory,
        clock=clock,
        candidate_version=resolved_candidate,
    ))


def _read_manifest(run_dir: Path) -> Dict[str, object]:
    payload = _read_json(
        run_dir / "manifest.json",
        "comparison run manifest",
    )
    if (
        type(payload) is not dict
        or payload.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or type(payload.get("rubric_version")) is not str
        or type(payload.get("candidate_version")) is not str
        or type(payload.get("runs_per_case")) is not int
        or type(payload.get("case_ids")) is not list
        or type(payload.get("runs")) is not list
    ):
        raise EvaluationConfigurationError("comparison run manifest is invalid")
    if (
        not 1 <= payload["runs_per_case"] <= MAX_RUNS_PER_CASE
        or not 1 <= len(payload["case_ids"]) <= MAX_CASES
        or len(payload["case_ids"]) != len(set(payload["case_ids"]))
        or any(
            type(case_id) is not str
            or _SAFE_ID.fullmatch(case_id) is None
            for case_id in payload["case_ids"]
        )
        or type(payload.get("case_set_sha256")) is not str
        or re.fullmatch(r"[0-9a-f]{64}", payload["case_set_sha256"])
        is None
        or len(payload["runs"])
        > len(payload["case_ids"]) * payload["runs_per_case"]
    ):
        raise EvaluationConfigurationError("comparison run manifest is invalid")
    identities = []
    allowed_cases = set(payload["case_ids"])
    for item in payload["runs"]:
        if (
            type(item) is not dict
            or item.get("status") not in {"succeeded", "failed"}
            or item.get("case_id") not in allowed_cases
            or type(item.get("run_index")) is not int
            or not 1 <= item["run_index"] <= payload["runs_per_case"]
        ):
            raise EvaluationConfigurationError(
                "comparison run manifest is invalid"
            )
        identities.append((item["case_id"], item["run_index"]))
    if len(identities) != len(set(identities)):
        raise EvaluationConfigurationError("comparison run manifest is invalid")
    return payload


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
    _ensure_empty_output(output_dir)

    left_successes = {
        (item.get("case_id"), item.get("run_index"))
        for item in left_manifest["runs"]
        if type(item) is dict and item.get("status") == "succeeded"
    }
    right_successes = {
        (item.get("case_id"), item.get("run_index"))
        for item in right_manifest["runs"]
        if type(item) is dict and item.get("status") == "succeeded"
    }
    keys = sorted(left_successes & right_successes)
    pairs = []
    mappings = []
    for case_id, run_index in keys:
        if type(case_id) is not str or type(run_index) is not int:
            raise EvaluationConfigurationError("comparison run identity is invalid")
        stem = "%s__run-%02d" % (case_id, run_index)
        try:
            left_payload = _read_json(
                left / "public" / "runtime" / (stem + ".json"),
                "comparison public runtime",
            )
            right_payload = _read_json(
                right / "public" / "runtime" / (stem + ".json"),
                "comparison public runtime",
            )
        except EvaluationConfigurationError:
            raise
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
    _write_json(output_dir / "public" / "blind_pairs.json", pairs)
    _write_json(output_dir / "private" / "candidate_mapping.json", mappings)
    _write_json(output_dir / "manifest.json", {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "pair_count": len(pairs),
        "case_ids": left_manifest["case_ids"],
        "runs_per_case": left_manifest["runs_per_case"],
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
