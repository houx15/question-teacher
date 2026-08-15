import hashlib
import inspect
import json
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import (
    Any,
    Awaitable,
    Callable,
    Optional,
    Union,
)
from uuid import uuid4

from pydantic import ValidationError, model_validator

from app.compiler import LessonCompileError, LessonCompiler
from app.claim_checker import (
    ClaimCheckResult,
    ClaimChecker,
    ClaimCheckerUnavailableError,
    ClaimStatus,
)
from app.deterministic_route import DeterministicRoutePlanner
from app.llm_client import ModelResponseError
from app.math_engine import MathValidationError
from app.preparation_models import (
    GenerationRecord,
    PreparedLesson,
    RuntimeCueProvenanceRecord,
)
from app.generation_integrity import validate_lesson_generation_pair
from app.generation_diagnostics import (
    GenerationFailureCategory,
    InternalGenerationDiagnostic,
)
from app.preparation_pipeline import LessonPreparationPipeline
from app.prepared_lesson_adapter import (
    PreparedLessonAdaptationError,
    prepared_lesson_to_draft_with_provenance,
)
from app.prompts import (
    MATH_ROUTE_SYSTEM,
    REFERENCE_GROUNDING_SYSTEM,
    REFERENCE_AUDITOR_SYSTEM,
    math_route_prompt,
    reference_grounding_prompt,
    reference_audit_prompt,
)
from app.schemas import (
    LessonDraft,
    MathRouteDraft,
    ProblemInput,
    ReferenceGroundingBrief,
    ReferenceMaterialAudit,
    RuntimeLesson,
    SchemaModel,
)
from app.problem_capability import (
    ProblemCapabilityProbe,
    ProblemIntakeStatus,
)
from app.problem_focus import compile_problem_focus_targets
from app.reference_safety import (
    ReferenceContentSafetyError,
    ReferenceSafetyPolicy,
)
from app.teaching_route import (
    FrozenTeachingRoute,
    TeachingRouteEvidenceError,
    freeze_grounded_route,
    freeze_symbolic_route,
)


StageCallback = Callable[
    [str],
    Union[None, Awaitable[None]],
]
REQUIRED_METHODS = {
    "factor": {
        "display_name": "因式分解法",
        "operation": "factor",
    },
    "quadratic_formula": {
        "display_name": "公式法",
        "operation": "quadratic_formula",
    },
    "complete_the_square": {
        "display_name": "配方法",
        "operation": "complete_the_square",
    },
}
RESOLVED_METHODS = {
    **REQUIRED_METHODS,
    "basic_equation_operations": {
        "display_name": "等式基本变形",
        "operation": None,
    },
}


class LessonQualityError(RuntimeError):
    """Raised when a generated lesson cannot pass safe quality gates."""


class LessonGenerationFailure(LessonQualityError):
    """Safe typed generation failure with no provider-authored detail."""

    def __init__(
        self,
        category: GenerationFailureCategory,
        detail: str,
    ) -> None:
        diagnostic = InternalGenerationDiagnostic(category=category)
        super().__init__(detail)
        self.category = diagnostic.category


class GeneratedLessonBundle(SchemaModel):
    lesson: RuntimeLesson
    generation_record: GenerationRecord

    @model_validator(mode="after")
    def validate_private_runtime_links(self) -> "GeneratedLessonBundle":
        validate_lesson_generation_pair(
            self.lesson,
            self.generation_record,
        )
        return self


_PUBLIC_INPUT_ERROR_MESSAGES = frozenset(
    {
        "输入包含不安全或过长的内容。",
        "参考答案与题目实际结果不一致。",
        "题目格式不正确。",
        "参考答案格式不正确。",
        "题目不能为空。",
        "参考答案不能为空。",
        "题目格式不完整，请检查后再试。",
        "参考答案与题目不一致，请检查后再试。",
        "参考解析与题目或参考答案存在数学冲突，请检查后再试。",
    }
)


class LessonInputError(LessonQualityError):
    """Safe, user-correctable input failure that may be shown publicly."""

    def __init__(self, public_message: str) -> None:
        if public_message not in _PUBLIC_INPUT_ERROR_MESSAGES:
            raise ValueError("unknown input error message")
        super().__init__(public_message)
        self.public_message = public_message

    @staticmethod
    def validated_public_message(error: Exception) -> Optional[str]:
        if type(error) is not LessonInputError:
            return None
        message = getattr(error, "public_message", None)
        args = getattr(error, "args", None)
        if (
            type(message) is not str
            or message not in _PUBLIC_INPUT_ERROR_MESSAGES
            or type(args) is not tuple
            or len(args) != 1
            or type(args[0]) is not str
            or args[0] != message
        ):
            return None
        return message


class _RouteValidationError(LessonQualityError):
    """Carries one allow-listed retry code without model output."""

    def __init__(self, code: str, public_message: str) -> None:
        super().__init__(public_message)
        self.code = code


def _canonical_report(report: dict) -> str:
    return json.dumps(
        report,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _runtime_semantics(lesson: RuntimeLesson) -> dict:
    payload = lesson.model_dump(mode="json")
    payload.pop("lesson_id", None)
    for beat in payload["beats"]:
        beat["audio_url"] = None
        for cue in beat["sync_cues"]:
            cue["audio_url"] = None
        interaction = beat.get("interaction")
        if interaction is None:
            continue
        interaction["hint_audio_urls"] = []
        interaction["correct_audio_url"] = None
        for option in interaction["options"]:
            option["feedback_audio_url"] = None
            for support_cue in option.get("support_cues", []):
                support_cue["audio_url"] = None
    return payload


def _has_audio_urls(lesson: RuntimeLesson) -> bool:
    return any(
        beat.audio_url is not None
        or any(cue.audio_url is not None for cue in beat.sync_cues)
        or (
            beat.interaction is not None
            and (
                bool(beat.interaction.hint_audio_urls)
                or beat.interaction.correct_audio_url is not None
                or any(
                    option.feedback_audio_url is not None
                    or any(
                        cue.audio_url is not None
                        for cue in option.support_cues
                    )
                    for option in beat.interaction.options
                )
            )
        )
        for beat in lesson.beats
    )


@dataclass(frozen=True)
class _VerifiedMathRoute:
    canonical_json: str
    fingerprint: str
    method_family: str
    source: str

    @classmethod
    def freeze(
        cls,
        route: MathRouteDraft,
        method_family: str,
        source: str = "agent",
    ) -> "_VerifiedMathRoute":
        canonical_json = route.model_copy(deep=True).model_dump_json()
        return cls(
            canonical_json=canonical_json,
            fingerprint=hashlib.sha256(
                canonical_json.encode("utf-8")
            ).hexdigest(),
            method_family=method_family,
            source=source,
        )

    def thaw(self) -> MathRouteDraft:
        route = MathRouteDraft.model_validate_json(self.canonical_json)
        actual = hashlib.sha256(
            route.model_dump_json().encode("utf-8")
        ).hexdigest()
        if actual != self.fingerprint:
            raise LessonQualityError("已验证数学路线的完整性检查失败。")
        return route


class LessonGenerationService:
    MAX_GROUNDING_ATTEMPTS = 3

    def __init__(
        self,
        client: Any,
        math_engine: Any,
        compiler: Optional[LessonCompiler] = None,
        deterministic_route_planner: Any = None,
        capability_probe: Any = None,
        claim_checker: Any = None,
        preparation_pipeline: Optional[LessonPreparationPipeline] = None,
    ) -> None:
        self.client = client
        self.math_engine = math_engine
        self.compiler = compiler or LessonCompiler()
        self.deterministic_route_planner = (
            deterministic_route_planner
            if deterministic_route_planner is not None
            else DeterministicRoutePlanner(math_engine)
        )
        self.capability_probe = (
            capability_probe
            if capability_probe is not None
            else ProblemCapabilityProbe(math_engine)
        )
        self.claim_checker = claim_checker or ClaimChecker()
        self.preparation_pipeline = (
            preparation_pipeline
            if preparation_pipeline is not None
            else LessonPreparationPipeline(client)
        )

    async def generate(
        self,
        problem: ProblemInput,
        on_stage: Optional[StageCallback] = None,
    ) -> RuntimeLesson:
        return (await self.generate_bundle(problem, on_stage=on_stage)).lesson

    async def generate_bundle(
        self,
        problem: ProblemInput,
        on_stage: Optional[StageCallback] = None,
    ) -> GeneratedLessonBundle:
        await self._emit(on_stage, "正在验证数学路线")
        assessment = self.capability_probe.assess(
            problem.problem_text,
            problem.reference_answer,
        )
        if assessment.status == ProblemIntakeStatus.INVALID_INPUT:
            raise LessonInputError(
                assessment.public_message or "题目格式不完整，请检查后再试。"
            )
        if assessment.status == ProblemIntakeStatus.CONTRADICTION:
            raise LessonInputError(
                assessment.public_message or "参考答案与题目不一致，请检查后再试。"
            )

        reference_audit = None
        verified_route = None
        problem_report = assessment.problem_validation
        if assessment.status == ProblemIntakeStatus.SYMBOLIC_VERIFIED:
            assert problem_report is not None
            if problem.reference_solution_text is not None:
                await self._emit(on_stage, "正在审阅参考解析")
                reference_audit = await self._audit_reference(
                    problem,
                    problem_report.solution_strings,
                )
                self._validate_reference_audit(problem, reference_audit)

            await self._emit(on_stage, "正在规划数学路线")
            verified_route = await self._create_validated_route(
                problem,
                problem_report.solution_strings,
                problem_report.equation_degree,
            )
            teaching_route = freeze_symbolic_route(
                verified_route,
                method_name=self._resolved_method_display_name(
                    verified_route
                ),
                equation_degree=problem_report.equation_degree,
                independent_solutions=list(
                    problem_report.solution_strings
                ),
            )
        else:
            teaching_route = await self._build_grounded_teaching_route(
                problem,
                on_stage,
            )

        problem_focus_targets = compile_problem_focus_targets(
            problem.problem_text
        )
        prepared_run = await self.preparation_pipeline.prepare_with_audit(
            problem,
            teaching_route,
            list(problem_focus_targets),
            on_stage=on_stage,
        )
        frozen_prepared_json = (
            prepared_run.prepared_lesson.model_copy(deep=True).model_dump_json()
        )
        prepared = PreparedLesson.model_validate_json(frozen_prepared_json)
        frozen_progression_json = (
            prepared.teaching_progression.model_copy(deep=True).model_dump_json()
            if prepared.teaching_progression is not None
            else None
        )
        try:
            prepared_draft_run = prepared_lesson_to_draft_with_provenance(
                problem,
                prepared,
                teaching_route,
                verified_math_steps=(
                    verified_route.thaw().math_steps
                    if verified_route is not None
                    else None
                ),
            )
        except PreparedLessonAdaptationError:
            raise LessonGenerationFailure(
                "compile_failed", "课程编排失败。"
            ) from None
        draft = prepared_draft_run.draft

        await self._emit(on_stage, "正在编译课堂")
        validation_report = {
            "verification_mode": teaching_route.mode.value,
            "consistency_status": teaching_route.consistency.value,
            "teaching_route_fingerprint": teaching_route.fingerprint,
            "pedagogy_rubric_version": prepared.rubric_version,
            "artifact_versions": prepared_run.audit.active_versions,
            "repair_count": prepared.repair_count,
            "review_status": prepared.review.status,
        }
        if verified_route is not None and problem_report is not None:
            validation_report.update(
                {
                    "math_status": "verified",
                    "independent_solutions": (
                        problem_report.solution_strings
                    ),
                    "math_route_status": "verified",
                    "math_route_fingerprint": verified_route.fingerprint,
                    "math_route_method_family": (
                        verified_route.method_family
                    ),
                    "math_route_source": verified_route.source,
                }
            )
        if reference_audit is not None:
            validation_report["reference_material_status"] = (
                reference_audit.status
            )
        frozen_problem_json = problem.model_dump_json()
        frozen_draft_json = draft.model_dump_json()
        frozen_report_json = _canonical_report(validation_report)
        try:
            expected_lesson = LessonCompiler().compile(
                ProblemInput.model_validate_json(frozen_problem_json),
                LessonDraft.model_validate_json(frozen_draft_json),
                json.loads(frozen_report_json),
                lesson_id="integrity-baseline",
            )
        except LessonCompileError:
            raise LessonGenerationFailure(
                "compile_failed", "课堂编译失败。"
            ) from None
        compiler_problem = ProblemInput.model_validate_json(
            frozen_problem_json
        )
        compiler_draft = LessonDraft.model_validate_json(
            frozen_draft_json
        )
        compiler_report = json.loads(frozen_report_json)
        try:
            lesson = self.compiler.compile(
                compiler_problem,
                compiler_draft,
                compiler_report,
            )
        except LessonCompileError:
            raise LessonGenerationFailure(
                "compile_failed", "课堂编译失败。"
            ) from None
        if type(lesson) is not RuntimeLesson:
            raise LessonGenerationFailure(
                "compile_failed", "课堂编译完整性检查失败。"
            )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                lesson_payload = lesson.model_dump(mode="python")
            lesson = RuntimeLesson.model_validate(lesson_payload)
        except (ValidationError, TypeError, ValueError):
            raise LessonGenerationFailure(
                "compile_failed", "课堂编译完整性检查失败。"
            ) from None
        if (
            compiler_problem.model_dump_json() != frozen_problem_json
            or compiler_draft.model_dump_json() != frozen_draft_json
            or _canonical_report(compiler_report) != frozen_report_json
            or lesson.problem.model_dump_json() != frozen_problem_json
            or _canonical_report(lesson.validation_report)
            != frozen_report_json
            or _has_audio_urls(lesson)
            or _runtime_semantics(lesson)
            != _runtime_semantics(expected_lesson)
            or prepared.model_dump_json() != frozen_prepared_json
            or (
                prepared.teaching_progression.model_dump_json()
                if prepared.teaching_progression is not None
                else None
            )
            != frozen_progression_json
        ):
            raise LessonGenerationFailure(
                "compile_failed", "课堂编译完整性检查失败。"
            )
        generation_record = GenerationRecord(
            generation_id=str(uuid4()),
            lesson_id=lesson.lesson_id,
            route_fingerprint=teaching_route.fingerprint,
            prepared_lesson=prepared,
            role_calls=prepared_run.audit.role_calls,
            cue_provenance=[
                RuntimeCueProvenanceRecord(
                    episode_id=item.episode_id,
                    lesson_step_id=item.lesson_step_id,
                    clause_id=item.clause_id,
                    original_performance_cue_id=(
                        item.original_performance_cue_id
                    ),
                    runtime_cue_id=item.runtime_cue_id,
                    display_text=item.display_text,
                    spoken_text=item.spoken_text,
                    response_id=item.response_id,
                )
                for item in prepared_draft_run.cue_provenance
            ],
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        try:
            return GeneratedLessonBundle(
                lesson=lesson,
                generation_record=generation_record,
            )
        except ValidationError:
            raise LessonGenerationFailure(
                "compile_failed", "课堂编译完整性检查失败。"
            ) from None

    async def _build_grounded_teaching_route(
        self,
        problem: ProblemInput,
        on_stage: Optional[StageCallback],
    ) -> FrozenTeachingRoute:
        await self._emit(on_stage, "正在整理参考教学路线")
        grounding_prompt = reference_grounding_prompt(problem)
        validation_guidance = ""
        for attempt in range(self.MAX_GROUNDING_ATTEMPTS):
            attempt_prompt = grounding_prompt
            if attempt:
                attempt_prompt += (
                    "\n上一次输出结构无效。"
                    "请仅返回符合 Schema 的 JSON 对象。"
                    + validation_guidance
                )
            payload = await self._complete_json(
                REFERENCE_GROUNDING_SYSTEM,
                attempt_prompt,
                ReferenceGroundingBrief,
            )
            try:
                brief = ReferenceGroundingBrief.validate_for_reference_answer(
                    payload,
                    problem.reference_answer,
                )
                break
            except ValidationError as error:
                if attempt + 1 == self.MAX_GROUNDING_ATTEMPTS:
                    raise LessonQualityError(
                        "参考教学路线结构无效。"
                    ) from None
                invalid_paths = sorted(
                    {
                        ".".join(str(part) for part in item["loc"])
                        for item in error.errors()
                    }
                )[:16]
                validation_guidance = (
                    "未通过校验的字段路径："
                    + "、".join(invalid_paths)
                    + "。数学字段只能填写纯数学表达式，"
                    + "不得加入解释性文字或不受支持的 LaTeX 命令。"
                    + "若路径位于 reasoning_steps，请同时检查 operation_kind "
                    + "与 operands 数量：算术操作恰好一个操作数，代入等操作"
                    + "为一至四个，化简、整理、结论等操作必须为零个。"
                )
        try:
            brief = ReferenceSafetyPolicy.from_problem(
                problem
            ).sanitize_grounding_brief(
                brief,
                problem.reference_answer,
            )
        except ReferenceContentSafetyError:
            raise LessonQualityError("参考教学路线内容无效。") from None

        results = []
        for request in brief.check_requests:
            try:
                result = self.claim_checker.check(request)
            except ClaimCheckerUnavailableError:
                result = ClaimCheckResult(
                    check_id=request.check_id,
                    status=ClaimStatus.UNSUPPORTED,
                    conclusion_linked=request.conclusion_linked,
                    reason_code="checker_unavailable",
                    request_fingerprint=ClaimChecker.request_fingerprint(
                        request
                    ),
                )
            results.append(result)
        try:
            return freeze_grounded_route(brief, results)
        except TeachingRouteEvidenceError:
            raise LessonQualityError("参考教学路线证据无效。") from None

    async def _create_route(
        self,
        problem: ProblemInput,
        solution_strings: Any,
        equation_degree: int,
        previous_validation_code: Optional[str],
    ) -> MathRouteDraft:
        payload = await self._complete_json(
            MATH_ROUTE_SYSTEM,
            math_route_prompt(
                problem,
                list(solution_strings),
                equation_degree,
                previous_validation_code,
            ),
        )
        try:
            return MathRouteDraft.model_validate(payload)
        except ValidationError:
            raise _RouteValidationError(
                "route_schema_invalid",
                "数学路线结构无效。",
            ) from None

    async def _create_validated_route(
        self,
        problem: ProblemInput,
        solution_strings: Any,
        equation_degree: int,
    ) -> _VerifiedMathRoute:
        deterministic = self._create_deterministic_route(
            problem,
            solution_strings,
            equation_degree,
        )
        if deterministic is not None:
            return deterministic

        previous_validation_code = None
        for attempt in range(2):
            try:
                route = await self._create_route(
                    problem,
                    solution_strings,
                    equation_degree,
                    previous_validation_code,
                )
                method_family = self._validate_math_route_draft(
                    problem,
                    route,
                    equation_degree,
                )
                return _VerifiedMathRoute.freeze(
                    route,
                    method_family,
                    source="agent",
                )
            except _RouteValidationError as error:
                if attempt == 1:
                    raise LessonQualityError(str(error)) from None
                previous_validation_code = error.code
        raise AssertionError("unreachable route retry state")

    def _create_deterministic_route(
        self,
        problem: ProblemInput,
        solution_strings: Any = None,
        equation_degree: Optional[int] = None,
    ) -> Optional[_VerifiedMathRoute]:
        if solution_strings is None or equation_degree is None:
            report = self.math_engine.validate_problem(
                problem.problem_text,
                problem.reference_answer,
            )
            solution_strings = report.solution_strings
            equation_degree = report.equation_degree
        route = self.deterministic_route_planner.plan(
            problem,
            equation_degree,
            list(solution_strings),
        )
        if route is None:
            return None
        try:
            method_family = self._validate_math_route_draft(
                problem,
                route,
                equation_degree,
            )
        except _RouteValidationError as error:
            raise LessonQualityError(str(error)) from None
        return _VerifiedMathRoute.freeze(
            route,
            method_family,
            source="deterministic",
        )

    async def _audit_reference(
        self,
        problem: ProblemInput,
        solution_strings: Any,
    ) -> ReferenceMaterialAudit:
        payload = await self._complete_json(
            REFERENCE_AUDITOR_SYSTEM,
            reference_audit_prompt(problem, list(solution_strings)),
        )
        try:
            return ReferenceMaterialAudit.model_validate(payload)
        except ValidationError:
            raise LessonQualityError("参考解析审阅结构无效。") from None

    def _validate_reference_audit(
        self,
        problem: ProblemInput,
        audit: ReferenceMaterialAudit,
    ) -> None:
        public_message = (
            "参考解析与题目或参考答案存在数学冲突，请检查后再试。"
        )
        if audit.status == "rejected":
            raise LessonInputError(public_message)

        try:
            if (
                audit.claimed_answer is not None
                and not self.math_engine.answers_equivalent(
                    audit.claimed_answer,
                    problem.reference_answer,
                )
            ):
                raise MathValidationError(
                    "Reference solution answer conflicts."
                )
            for step in audit.key_steps:
                self.math_engine.validate_step(step)
        except MathValidationError:
            raise LessonInputError(public_message) from None

    async def _complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
        model_type: Optional[type] = None,
    ) -> Any:
        for attempt in range(2):
            try:
                structured_method = getattr(
                    self.client,
                    "complete_model",
                    None,
                )
                if model_type is not None and callable(structured_method):
                    return await structured_method(
                        system_prompt,
                        user_prompt,
                        model_type,
                    )
                return await self.client.complete_json(
                    system_prompt,
                    user_prompt,
                )
            except ModelResponseError:
                if attempt == 1:
                    raise
        raise AssertionError("unreachable model retry state")

    @staticmethod
    def _resolved_method_display_name(
        verified_route: _VerifiedMathRoute,
    ) -> str:
        profile = RESOLVED_METHODS.get(verified_route.method_family)
        if profile is None:
            raise LessonQualityError("已验证数学路线的方法族无效。")
        return profile["display_name"]

    def _validate_math_route_draft(
        self,
        problem: ProblemInput,
        route: MathRouteDraft,
        equation_degree: int,
    ) -> str:
        named_families = {
            "factor",
            "complete_the_square",
            "quadratic_formula",
        }
        used_families = {
            step.operation
            for step in route.math_steps
            if step.operation in named_families
        }
        if problem.required_method is not None:
            if used_families != {problem.required_method}:
                code = (
                    "route_required_method_missing"
                    if problem.required_method not in used_families
                    else "route_method_family_conflict"
                )
                raise _RouteValidationError(
                    code,
                    "数学路线未通过验证。",
                )
            method_family = problem.required_method
        elif equation_degree == 2:
            if len(used_families) != 1:
                raise _RouteValidationError(
                    "route_method_family_conflict",
                    "数学路线未通过验证。",
                )
            method_family = next(iter(used_families))
        elif equation_degree == 1:
            basic_operations = {
                "simplify",
                "add_both_sides",
                "subtract_both_sides",
                "multiply_both_sides",
                "divide_both_sides",
                "expand",
                "combine_like_terms",
            }
            if used_families or any(
                step.operation not in basic_operations
                for step in route.math_steps
            ):
                raise _RouteValidationError(
                    "route_method_family_conflict",
                    "数学路线未通过验证。",
                )
            method_family = "basic_equation_operations"
        else:
            raise _RouteValidationError(
                "route_method_family_conflict",
                "数学路线未通过验证。",
            )

        if method_family == "factor" and (
            route.math_steps[-1].operation != "factor"
            or len(route.math_steps[-1].state_after) != 1
        ):
            raise _RouteValidationError(
                "route_method_family_conflict",
                "数学路线未通过验证。",
            )

        for step in route.math_steps:
            try:
                self.math_engine.validate_step(step)
            except MathValidationError:
                raise _RouteValidationError(
                    "route_step_invalid",
                    "数学路线未通过验证。",
                ) from None

        try:
            original_equation = self.math_engine.extract_problem_equation(
                problem.problem_text
            )
            if (
                len(route.math_steps[0].state_before) != 1
                or self._normalized_state(
                    route.math_steps[0].state_before
                )
                != self._normalized_state([original_equation])
            ):
                raise _RouteValidationError(
                    "route_first_state_mismatch",
                    "数学路线未通过验证。",
                )

            for previous, current in zip(
                route.math_steps,
                route.math_steps[1:],
            ):
                if self._normalized_state(previous.state_after) != (
                    self._normalized_state(current.state_before)
                ):
                    raise _RouteValidationError(
                        "route_disconnected",
                        "数学路线未通过验证。",
                    )

            final_solutions = self.math_engine.solution_set(
                route.math_steps[-1].state_after
            )
            expected_solutions = self.math_engine.solution_set(
                [original_equation]
            )
            if final_solutions != expected_solutions:
                raise _RouteValidationError(
                    "route_final_solution_mismatch",
                    "数学路线未通过验证。",
                )
        except _RouteValidationError:
            raise
        except MathValidationError:
            raise _RouteValidationError(
                "route_step_invalid",
                "数学路线未通过验证。",
            ) from None
        return method_family

    def _normalized_state(self, state: Any) -> Any:
        if (
            len(state) == 1
            and isinstance(state[0], str)
            and state[0].strip() == "无实数解"
        ):
            return (("empty-real-solution-set",),)

        equations = []
        for equation_text in state:
            equation = self.math_engine.parse_equation(equation_text)
            equations.append((str(equation.lhs), str(equation.rhs)))
        return tuple(sorted(equations))

    @staticmethod
    async def _emit(
        callback: Optional[StageCallback],
        stage: str,
    ) -> None:
        if callback is None:
            return
        result = callback(stage)
        if inspect.isawaitable(result):
            await result
