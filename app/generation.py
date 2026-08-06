import inspect
import json
import re
from typing import Any, Awaitable, Callable, Optional, Union

from pydantic import ValidationError

from app.compiler import LessonCompileError, LessonCompiler
from app.llm_client import ModelResponseError
from app.math_engine import MathValidationError
from app.prompts import (
    DIRECTOR_SYSTEM,
    REFERENCE_AUDITOR_SYSTEM,
    REVIEWER_SYSTEM,
    REVISION_SYSTEM,
    director_prompt,
    reference_audit_prompt,
    reviewer_prompt,
    revision_prompt,
)
from app.schemas import (
    LessonDraft,
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


def _lesson_draft_property_names() -> set:
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

    visit(LessonDraft.model_json_schema())
    return names


_LESSON_DRAFT_PROPERTY_NAMES = _lesson_draft_property_names()
_MAX_SCHEMA_RETRY_ISSUES = 12


def _draft_schema_validation_summary(error: ValidationError) -> str:
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
            elif part in _LESSON_DRAFT_PROPERTY_NAMES:
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
            "category": "lesson_draft_schema_validation",
            "issue_count": len(raw_issues),
            "issues": issues,
            "truncated": len(raw_issues) > len(issues),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
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

        await self._emit(on_stage, "正在设计完整讲解")
        draft = await self._create_validated_draft(
            problem,
            problem_report.solution_strings,
            reference_audit,
        )
        revision_count = 0

        while True:
            self._validate_draft(problem, draft)
            await self._emit(on_stage, "正在进行整篇审稿")
            review = await self._review(problem, draft, reference_audit)
            if review.status == "approved":
                break
            if revision_count >= self.MAX_REVISIONS:
                raise LessonQualityError("整篇讲稿在两轮修订后仍未通过。")

            await self._emit(on_stage, "正在修订完整讲解")
            draft = await self._revise(
                problem,
                draft,
                review,
                reference_audit,
            )
            revision_count += 1

        await self._emit(on_stage, "正在编译课堂")
        validation_report = {
            "math_status": "verified",
            "review_status": review.status,
            "revision_count": revision_count,
            "independent_solutions": problem_report.solution_strings,
            "review_assessment": review.overall_assessment,
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

    async def _create_draft(
        self,
        problem: ProblemInput,
        solution_strings: Any,
        reference_audit: Optional[ReferenceMaterialAudit],
        previous_validation_error: Optional[str] = None,
    ) -> LessonDraft:
        payload = await self._complete_json(
            DIRECTOR_SYSTEM,
            director_prompt(
                problem,
                list(solution_strings),
                reference_audit,
                previous_validation_error,
            ),
        )
        try:
            return LessonDraft.model_validate(payload)
        except ValidationError as error:
            raise _DraftSchemaValidationError(
                _draft_schema_validation_summary(error)
            ) from None

    async def _create_validated_draft(
        self,
        problem: ProblemInput,
        solution_strings: Any,
        reference_audit: Optional[ReferenceMaterialAudit],
    ) -> LessonDraft:
        previous_validation_error = None
        for attempt in range(self.MAX_DRAFT_ATTEMPTS):
            try:
                draft = await self._create_draft(
                    problem,
                    solution_strings,
                    reference_audit,
                    previous_validation_error,
                )
            except _DraftSchemaValidationError as error:
                if attempt + 1 >= self.MAX_DRAFT_ATTEMPTS:
                    raise LessonQualityError(str(error)) from None
                previous_validation_error = error.validation_summary
                continue
            try:
                self._validate_draft(problem, draft)
                return draft
            except LessonQualityError as error:
                if attempt + 1 >= self.MAX_DRAFT_ATTEMPTS:
                    raise
                previous_validation_error = str(error)
        raise LessonQualityError("完整讲解生成失败。")

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
        draft: LessonDraft,
        review: ReviewDecision,
        reference_audit: Optional[ReferenceMaterialAudit],
    ) -> LessonDraft:
        payload = await self._complete_json(
            REVISION_SYSTEM,
            revision_prompt(problem, draft, review, reference_audit),
        )
        try:
            return LessonDraft.model_validate(payload)
        except ValidationError:
            raise LessonQualityError("模型修订的讲解结构无效。") from None

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
        if any(
            option.label.strip()
            != self.math_engine.format_answer_label(option.canonical_answer)
            for option in transfer_options
        ):
            raise LessonQualityError("近迁移选项显示格式无效。")

        interactions = [
            moment.interaction
            for moment in draft.moments
            if moment.interaction is not None
        ]
        if not interactions:
            raise LessonQualityError("讲解没有设置学生互动。")

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
