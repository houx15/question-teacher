import hashlib
import inspect
import json
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Union

from pydantic import ValidationError

from app.compiler import (
    NEAR_TRANSFER_INTERACTION_ID,
    LessonCompileError,
    LessonCompiler,
)
from app.llm_client import ModelResponseError
from app.math_engine import MathValidationError
from app.prompts import (
    DIRECTOR_SYSTEM,
    MATERIALS_SYSTEM,
    MATH_ROUTE_SYSTEM,
    REFERENCE_AUDITOR_SYSTEM,
    REVIEWER_SYSTEM,
    REVISION_SYSTEM,
    director_prompt,
    materials_prompt,
    math_route_prompt,
    reference_audit_prompt,
    reviewer_prompt,
    revision_prompt,
)
from app.schemas import (
    LessonMoment,
    LessonDraft,
    MAX_NARRATIVE_SERIALIZED_BYTES,
    MaterialsDraft,
    MathRouteDraft,
    NarrativeDraft,
    ProblemInput,
    ReferenceMaterialAudit,
    ReviewDecision,
    RuntimeLesson,
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


_INLINE_MATH_SEGMENT = re.compile(r"\\\((.*?)\\\)")
_BLOCK_MATH_SEGMENT = re.compile(r"\\\[(.*?)\\\]")


def _normalize_choice_option_label(label: str) -> str:
    """Approximate browser display normalization for choice-label equality."""
    normalized = " ".join(label.split())
    normalized = _INLINE_MATH_SEGMENT.sub(
        lambda match: r"\(" + re.sub(r"\s+", "", match.group(1)) + r"\)",
        normalized,
    )
    return _BLOCK_MATH_SEGMENT.sub(
        lambda match: r"\[" + re.sub(r"\s+", "", match.group(1)) + r"\]",
        normalized,
    )


class LessonQualityError(RuntimeError):
    """Raised when a generated lesson cannot pass safe quality gates."""


class LessonInputError(LessonQualityError):
    """Safe, user-correctable input failure that may be shown publicly."""

    def __init__(self, public_message: str) -> None:
        super().__init__(public_message)
        self.public_message = public_message


class _DraftSchemaValidationError(LessonQualityError):
    """Carries a bounded schema-only retry summary inside the service."""

    def __init__(self, validation_summary: str) -> None:
        super().__init__("模型生成的讲解结构无效。")
        self.validation_summary = validation_summary


class _MaterialsValidationError(LessonQualityError):
    """Carries a safe materials-only retry reason."""

    def __init__(self, public_message: str, retry_summary: str) -> None:
        super().__init__(public_message)
        self.retry_summary = retry_summary


class _RouteValidationError(LessonQualityError):
    """Carries one allow-listed retry code without model output."""

    def __init__(self, code: str, public_message: str) -> None:
        super().__init__(public_message)
        self.code = code


@dataclass(frozen=True)
class _VerifiedMathRoute:
    canonical_json: str
    fingerprint: str
    method_family: str

    @classmethod
    def freeze(
        cls,
        route: MathRouteDraft,
        method_family: str,
    ) -> "_VerifiedMathRoute":
        canonical_json = route.model_dump_json()
        return cls(
            canonical_json=canonical_json,
            fingerprint=hashlib.sha256(
                canonical_json.encode("utf-8")
            ).hexdigest(),
            method_family=method_family,
        )

    def thaw(self) -> MathRouteDraft:
        route = MathRouteDraft.model_validate_json(self.canonical_json)
        actual = hashlib.sha256(
            route.model_dump_json().encode("utf-8")
        ).hexdigest()
        if actual != self.fingerprint:
            raise LessonQualityError("已验证数学路线的完整性检查失败。")
        return route


def _schema_property_names(model: Any) -> set:
    names = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            properties = value.get("properties")
            if isinstance(properties, dict):
                names.update(properties)
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(model.model_json_schema())
    return names


_NARRATIVE_DRAFT_PROPERTY_NAMES = _schema_property_names(NarrativeDraft)
_MATERIALS_DRAFT_PROPERTY_NAMES = _schema_property_names(MaterialsDraft)
_MAX_SCHEMA_RETRY_ISSUES = 12


def _schema_validation_summary(
    error: ValidationError,
    *,
    category: str,
    property_names: set,
) -> str:
    raw_issues = error.errors(
        include_url=False,
        include_context=False,
        include_input=False,
    )
    issues = []
    for raw_issue in raw_issues[:_MAX_SCHEMA_RETRY_ISSUES]:
        path_parts = []
        for part in raw_issue.get("loc", ()):
            if isinstance(part, int):
                path_parts.append("[]")
            elif part in property_names:
                path_parts.append(part)
            else:
                path_parts.append("<unknown>")
        issue_type = raw_issue.get("type")
        if not (
            isinstance(issue_type, str)
            and len(issue_type) <= 40
            and re.fullmatch(r"[a-z_]+", issue_type)
        ):
            issue_type = "validation_error"
        issues.append(
            {
                "path": ".".join(path_parts) or "<model>",
                "type": issue_type,
            }
        )
    return json.dumps(
        {
            "category": category,
            "issue_count": len(raw_issues),
            "issues": issues,
            "truncated": len(raw_issues) > len(issues),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _narrative_schema_validation_summary(error: ValidationError) -> str:
    return _schema_validation_summary(
        error,
        category="narrative_draft_schema_validation",
        property_names=_NARRATIVE_DRAFT_PROPERTY_NAMES,
    )


def _materials_schema_validation_summary(error: ValidationError) -> str:
    return _schema_validation_summary(
        error,
        category="materials_draft_schema_validation",
        property_names=_MATERIALS_DRAFT_PROPERTY_NAMES,
    )


class LessonGenerationService:
    MAX_REVISIONS = 2
    MAX_DRAFT_ATTEMPTS = 2

    def __init__(
        self,
        client: Any,
        math_engine: Any,
        compiler: Optional[LessonCompiler] = None,
    ) -> None:
        self.client = client
        self.math_engine = math_engine
        self.compiler = compiler or LessonCompiler()

    async def generate(
        self,
        problem: ProblemInput,
        on_stage: Optional[StageCallback] = None,
    ) -> RuntimeLesson:
        await self._emit(on_stage, "正在验证数学路线")
        try:
            problem_report = self.math_engine.validate_problem(
                problem.problem_text,
                problem.reference_answer,
            )
        except MathValidationError:
            raise LessonInputError(
                "题目或参考答案未通过数学验证，请检查后再试。"
            ) from None

        reference_audit = None
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
        await self._emit(on_stage, "正在设计完整讲解")
        narrative = await self._create_validated_narrative(
            problem,
            problem_report.solution_strings,
            reference_audit,
            problem_report.equation_degree,
            verified_route,
        )
        await self._emit(on_stage, "正在准备互动素材")
        draft = await self._create_validated_materials(
            problem,
            narrative,
            problem_report.solution_strings,
            None,
            problem_report.equation_degree,
            verified_route,
        )
        revision_count = 0

        while True:
            self._validate_narrative_size(narrative)
            self._validate_draft(problem, draft)
            self._assert_route_fingerprint(draft, verified_route)
            await self._emit(on_stage, "正在进行整篇审稿")
            review = await self._review(problem, draft, reference_audit)
            if review.status == "approved":
                break
            if revision_count >= self.MAX_REVISIONS:
                raise LessonQualityError("整篇讲稿在两轮修订后仍未通过。")

            await self._emit(on_stage, "正在修订完整讲解")
            narrative = await self._revise(
                problem,
                narrative,
                review,
                reference_audit,
                verified_route,
            )
            await self._emit(on_stage, "正在准备互动素材")
            draft = await self._create_validated_materials(
                problem,
                narrative,
                problem_report.solution_strings,
                review,
                problem_report.equation_degree,
                verified_route,
            )
            revision_count += 1

        await self._emit(on_stage, "正在编译课堂")
        self._validate_narrative_size(narrative)
        validation_report = {
            "math_status": "verified",
            "review_status": review.status,
            "revision_count": revision_count,
            "independent_solutions": problem_report.solution_strings,
            "review_assessment": review.overall_assessment,
            "math_route_status": "verified",
            "math_route_fingerprint": verified_route.fingerprint,
            "math_route_method_family": verified_route.method_family,
        }
        if reference_audit is not None:
            validation_report["reference_material_status"] = (
                reference_audit.status
            )
        try:
            return self.compiler.compile(
                problem,
                draft,
                validation_report,
            )
        except LessonCompileError:
            raise
        except Exception:
            raise LessonQualityError("课堂编译失败。") from None

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
                return _VerifiedMathRoute.freeze(route, method_family)
            except _RouteValidationError as error:
                if attempt == 1:
                    raise LessonQualityError(str(error)) from None
                previous_validation_code = error.code
        raise AssertionError("unreachable route retry state")

    async def _create_narrative(
        self,
        problem: ProblemInput,
        solution_strings: Any,
        reference_audit: Optional[ReferenceMaterialAudit],
        previous_validation_error: Optional[str] = None,
        original_equation_degree: Optional[int] = None,
        verified_route: Optional[_VerifiedMathRoute] = None,
    ) -> NarrativeDraft:
        payload = await self._complete_json(
            DIRECTOR_SYSTEM,
            director_prompt(
                problem,
                list(solution_strings),
                reference_audit,
                previous_validation_error,
                original_equation_degree,
                verified_math_route=(
                    verified_route.thaw()
                    if verified_route is not None
                    else None
                ),
            ),
        )
        try:
            return NarrativeDraft.model_validate(payload)
        except ValidationError as error:
            raise _DraftSchemaValidationError(
                _narrative_schema_validation_summary(error)
            ) from None

    async def _create_validated_narrative(
        self,
        problem: ProblemInput,
        solution_strings: Any,
        reference_audit: Optional[ReferenceMaterialAudit],
        original_equation_degree: Optional[int],
        verified_route: _VerifiedMathRoute,
    ) -> NarrativeDraft:
        previous_validation_error = None
        for attempt in range(self.MAX_DRAFT_ATTEMPTS):
            try:
                narrative = await self._create_narrative(
                    problem,
                    solution_strings,
                    reference_audit,
                    previous_validation_error,
                    original_equation_degree,
                    verified_route,
                )
            except _DraftSchemaValidationError as error:
                if attempt + 1 >= self.MAX_DRAFT_ATTEMPTS:
                    raise LessonQualityError(str(error)) from None
                previous_validation_error = error.validation_summary
                continue
            try:
                self._validate_narrative(problem, narrative)
                return narrative
            except LessonQualityError as error:
                if attempt + 1 >= self.MAX_DRAFT_ATTEMPTS:
                    raise
                previous_validation_error = str(error)
        raise LessonQualityError("完整讲解生成失败。")

    async def _create_materials(
        self,
        problem: ProblemInput,
        narrative: NarrativeDraft,
        solution_strings: Any,
        review: Optional[ReviewDecision],
        previous_validation_error: Optional[str],
        original_equation_degree: Optional[int],
        verified_route: _VerifiedMathRoute,
    ) -> MaterialsDraft:
        payload = await self._complete_json(
            MATERIALS_SYSTEM,
            materials_prompt(
                problem,
                narrative,
                list(solution_strings),
                review,
                previous_validation_error,
                original_equation_degree,
                verified_math_route=verified_route.thaw(),
            ),
        )
        try:
            return MaterialsDraft.model_validate(payload)
        except ValidationError as error:
            raise _MaterialsValidationError(
                "互动素材结构无效。",
                _materials_schema_validation_summary(error),
            ) from None

    async def _create_validated_materials(
        self,
        problem: ProblemInput,
        narrative: NarrativeDraft,
        solution_strings: Any,
        review: Optional[ReviewDecision],
        original_equation_degree: Optional[int],
        verified_route: _VerifiedMathRoute,
    ) -> LessonDraft:
        self._validate_narrative_size(narrative)
        previous_validation_error = None
        last_error: Optional[LessonQualityError] = None
        for attempt in range(2):
            try:
                materials = await self._create_materials(
                    problem,
                    narrative,
                    solution_strings,
                    review,
                    previous_validation_error,
                    original_equation_degree,
                    verified_route,
                )
                draft = self._compose_draft(
                    narrative,
                    materials,
                    verified_route,
                )
                draft = self._canonicalize_transfer_labels(draft)
                self._validate_draft(problem, draft)
                return draft
            except _MaterialsValidationError as error:
                last_error = error
                previous_validation_error = error.retry_summary
            except LessonQualityError as error:
                last_error = error
                previous_validation_error = str(error)
            if attempt == 1:
                assert last_error is not None
                raise last_error
        raise AssertionError("unreachable materials retry state")

    def _compose_draft(
        self,
        narrative: NarrativeDraft,
        materials: MaterialsDraft,
        verified_route: _VerifiedMathRoute,
    ) -> LessonDraft:
        narrative_ids = {
            moment.moment_id
            for moment in narrative.moments
        }
        intended_ids = {
            moment.moment_id
            for moment in narrative.moments
            if moment.interaction_intent is not None
        }
        bound_ids = [
            binding.moment_id
            for binding in materials.interactions
        ]
        if any(moment_id not in narrative_ids for moment_id in bound_ids):
            raise LessonQualityError("互动素材的绑定位置无效。")
        if len(bound_ids) != len(set(bound_ids)):
            raise LessonQualityError("互动素材不能重复绑定同一时刻。")
        if set(bound_ids) != intended_ids:
            raise LessonQualityError(
                "互动素材必须完整填写已声明的互动意图。"
            )

        by_id = {
            binding.moment_id: binding.interaction
            for binding in materials.interactions
        }
        moments = [
            LessonMoment(
                purpose=moment.purpose,
                narration=moment.narration,
                board_actions=[
                    action.model_copy(deep=True)
                    for action in moment.board_actions
                ],
                layer=moment.layer,
                interaction=(
                    by_id[moment.moment_id].model_dump()
                    if moment.moment_id in by_id
                    else None
                ),
            )
            for moment in narrative.moments
        ]
        return LessonDraft(
            **narrative.model_dump(exclude={"moments"}),
            math_steps=[
                step.model_copy(deep=True)
                for step in verified_route.thaw().math_steps
            ],
            moments=moments,
            transfer_item=materials.transfer_item.model_dump(),
        )

    async def _review(
        self,
        problem: ProblemInput,
        draft: LessonDraft,
        reference_audit: Optional[ReferenceMaterialAudit],
    ) -> ReviewDecision:
        payload = await self._complete_json(
            REVIEWER_SYSTEM,
            reviewer_prompt(problem, draft, reference_audit),
        )
        try:
            return ReviewDecision.model_validate(payload)
        except ValidationError:
            raise LessonQualityError("模型返回的审稿结构无效。") from None

    async def _revise(
        self,
        problem: ProblemInput,
        narrative: NarrativeDraft,
        review: ReviewDecision,
        reference_audit: Optional[ReferenceMaterialAudit],
        verified_route: _VerifiedMathRoute,
    ) -> NarrativeDraft:
        previous_validation_error = None
        for attempt in range(self.MAX_DRAFT_ATTEMPTS):
            payload = await self._complete_json(
                REVISION_SYSTEM,
                revision_prompt(
                    problem,
                    narrative,
                    review,
                    reference_audit,
                    previous_validation_error,
                    verified_math_route=verified_route.thaw(),
                ),
            )
            try:
                revised = NarrativeDraft.model_validate(payload)
            except ValidationError as error:
                if attempt + 1 >= self.MAX_DRAFT_ATTEMPTS:
                    raise LessonQualityError(
                        "模型修订的讲解结构无效。"
                    ) from None
                previous_validation_error = (
                    _narrative_schema_validation_summary(error)
                )
                continue
            try:
                self._validate_narrative(problem, revised)
                return revised
            except LessonQualityError as error:
                if attempt + 1 >= self.MAX_DRAFT_ATTEMPTS:
                    raise
                previous_validation_error = str(error)
        raise AssertionError("unreachable revision retry state")

    def _canonicalize_transfer_labels(
        self,
        draft: LessonDraft,
    ) -> LessonDraft:
        try:
            labels = [
                self.math_engine.format_answer_label(
                    option.canonical_answer
                )
                for option in draft.transfer_item.options
            ]
        except MathValidationError:
            return draft

        options = [
            option.model_copy(update={"label": label})
            for option, label in zip(
                draft.transfer_item.options,
                labels,
            )
        ]
        transfer_item = draft.transfer_item.model_copy(
            update={"options": options}
        )
        return draft.model_copy(
            update={"transfer_item": transfer_item}
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
    ) -> Any:
        for attempt in range(2):
            try:
                return await self.client.complete_json(
                    system_prompt,
                    user_prompt,
                )
            except ModelResponseError:
                if attempt == 1:
                    raise
        raise AssertionError("unreachable model retry state")

    def _validate_narrative(
        self,
        problem: ProblemInput,
        narrative: NarrativeDraft,
    ) -> None:
        self._validate_narrative_size(narrative)
        required_method = REQUIRED_METHODS.get(problem.required_method)
        moment_ids = [
            moment.moment_id
            for moment in narrative.moments
        ]
        if len(moment_ids) != len(set(moment_ids)):
            raise LessonQualityError("教学主线的时刻标识必须唯一。")
        intent_count = sum(
            moment.interaction_intent is not None
            for moment in narrative.moments
        )
        if intent_count not in {1, 2, 3}:
            raise LessonQualityError(
                "教学主线必须声明 1 至 3 个互动意图。"
            )
        if (
            required_method is not None
            and narrative.method_introduction.method_name
            != required_method["display_name"]
        ):
            raise LessonQualityError(
                "讲解的方法介绍与指定方法不一致。"
            )
        if len(narrative.method_introduction.spoken_narration) > 90:
            raise LessonQualityError("方法介绍的口语讲稿过长。")

    @staticmethod
    def _validate_narrative_size(narrative: NarrativeDraft) -> None:
        serialized_size = len(
            narrative.model_dump_json().encode("utf-8")
        )
        if serialized_size > MAX_NARRATIVE_SERIALIZED_BYTES:
            raise LessonQualityError("教学主线整体内容过长。")

    def _validate_draft(
        self,
        problem: ProblemInput,
        draft: LessonDraft,
    ) -> None:
        required_method = REQUIRED_METHODS.get(problem.required_method)
        if (
            required_method is not None
            and draft.method_introduction.method_name
            != required_method["display_name"]
        ):
            raise LessonQualityError(
                "讲解的方法介绍与指定方法不一致。"
            )

        if len(draft.method_introduction.spoken_narration) > 90:
            raise LessonQualityError("方法介绍的口语讲稿过长。")

        for step in draft.math_steps:
            try:
                self.math_engine.validate_step(step)
            except MathValidationError:
                raise LessonQualityError(
                    "讲解中的数学步骤未通过验证。"
                ) from None

        self._validate_math_route(problem, draft)

        try:
            self.math_engine.validate_problem(
                draft.transfer_item.problem_text,
                draft.transfer_item.expected_answer,
            )
        except MathValidationError:
            raise LessonQualityError(
                "近迁移题未通过数学验证。"
            ) from None

        transfer_options = draft.transfer_item.options
        if len(transfer_options) not in {3, 4}:
            raise LessonQualityError(
                "近迁移题必须提供 3 至 4 个诊断选项。"
            )
        try:
            equivalent_option_ids = []
            for option in transfer_options:
                if not self.math_engine.answers_equivalent(
                    option.canonical_answer,
                    option.canonical_answer,
                ):
                    raise MathValidationError(
                        "Transfer option answer is not self-equivalent."
                    )
                if self.math_engine.answers_equivalent(
                    option.canonical_answer,
                    draft.transfer_item.expected_answer,
                ):
                    equivalent_option_ids.append(option.option_id)
            if (
                len(equivalent_option_ids) != 1
                or draft.transfer_item.correct_option_id
                != equivalent_option_ids[0]
            ):
                raise MathValidationError(
                    "Transfer options must identify one correct answer."
                )
        except MathValidationError:
            raise LessonQualityError(
                "近迁移选项未通过数学验证。"
            ) from None
        expected_labels = [
            self.math_engine.format_answer_label(option.canonical_answer)
            for option in transfer_options
        ]
        if (
            any(
                option.label is None
                or option.label.strip() != expected_label
                for option, expected_label in zip(
                    transfer_options,
                    expected_labels,
                )
            )
            or len(expected_labels) != len(set(expected_labels))
        ):
            raise LessonQualityError("近迁移选项显示格式无效。")

        interactions = [
            moment.interaction
            for moment in draft.moments
            if moment.interaction is not None
        ]
        if len(interactions) not in {1, 2, 3}:
            raise LessonQualityError(
                "讲解只能设置 1 至 3 个学生互动。"
            )

        interaction_ids = [
            interaction.interaction_id
            for interaction in interactions
        ]
        if NEAR_TRANSFER_INTERACTION_ID in interaction_ids:
            raise LessonQualityError(
                "学生互动标识不能使用系统保留值。"
            )
        if len(interaction_ids) != len(set(interaction_ids)):
            raise LessonQualityError("学生互动标识必须全课唯一。")

        for interaction in interactions:
            if interaction.kind != "choice":
                raise LessonQualityError(
                    "新讲解中的自动判分互动必须使用选择题。"
                )
            if len(interaction.options) not in {3, 4}:
                raise LessonQualityError(
                    "选择互动需要 3 至 4 个选项。"
                )
            option_labels = [
                _normalize_choice_option_label(option.label)
                for option in interaction.options
            ]
            if len(option_labels) != len(set(option_labels)):
                raise LessonQualityError("选择互动选项标签不能重复。")
            if any(
                option.feedback is None
                for option in interaction.options
            ):
                raise LessonQualityError("选择互动缺少诊断反馈。")
            if any(
                option.feedback_audio_url is not None
                for option in interaction.options
            ):
                raise LessonQualityError(
                    "选择互动不能预填反馈音频地址。"
                )

        if (
            required_method is not None
            and required_method["operation"] not in {
            step.operation for step in draft.math_steps
            }
        ):
            raise LessonQualityError("讲解没有真正使用指定方法。")

    def _validate_math_route(
        self,
        problem: ProblemInput,
        draft: LessonDraft,
    ) -> None:
        first_state = draft.math_steps[0].state_before
        try:
            if len(first_state) != 1:
                raise MathValidationError(
                    "The route must begin from one original equation."
                )
            self.math_engine.validate_problem(
                first_state[0],
                problem.reference_answer,
            )
            original_solutions = self.math_engine.solution_set(first_state)

            for previous, current in zip(
                draft.math_steps,
                draft.math_steps[1:],
            ):
                if self._normalized_state(previous.state_after) != (
                    self._normalized_state(current.state_before)
                ):
                    raise MathValidationError(
                        "Consecutive route states do not connect."
                    )

            final_solutions = self.math_engine.solution_set(
                draft.math_steps[-1].state_after
            )
            if final_solutions != original_solutions:
                raise MathValidationError(
                    "The final route does not preserve the original solutions."
                )
        except MathValidationError:
            raise LessonQualityError(
                "讲解中的数学路线未通过验证。"
            ) from None

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

    def _assert_route_fingerprint(
        self,
        draft: LessonDraft,
        verified_route: _VerifiedMathRoute,
    ) -> None:
        assembled = MathRouteDraft(
            math_steps=[
                step.model_copy(deep=True)
                for step in draft.math_steps
            ]
        )
        actual = hashlib.sha256(
            assembled.model_dump_json().encode("utf-8")
        ).hexdigest()
        if actual != verified_route.fingerprint:
            raise LessonQualityError("已验证数学路线的完整性检查失败。")

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
