import inspect
from typing import Any, Awaitable, Callable, Optional, Union

from pydantic import ValidationError

from app.compiler import LessonCompileError, LessonCompiler
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


class LessonQualityError(RuntimeError):
    """Raised when a generated lesson cannot pass safe quality gates."""


class LessonInputError(LessonQualityError):
    """Safe, user-correctable input failure that may be shown publicly."""

    def __init__(self, public_message: str) -> None:
        super().__init__(public_message)
        self.public_message = public_message


class LessonGenerationService:
    MAX_REVISIONS = 2

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
        draft = await self._create_draft(
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
    ) -> LessonDraft:
        payload = await self._complete_json(
            DIRECTOR_SYSTEM,
            director_prompt(
                problem,
                list(solution_strings),
                reference_audit,
            ),
            "完整讲解生成失败。",
        )
        try:
            return LessonDraft.model_validate(payload)
        except ValidationError:
            raise LessonQualityError("模型生成的讲解结构无效。") from None

    async def _review(
        self,
        problem: ProblemInput,
        draft: LessonDraft,
        reference_audit: Optional[ReferenceMaterialAudit],
    ) -> ReviewDecision:
        payload = await self._complete_json(
            REVIEWER_SYSTEM,
            reviewer_prompt(problem, draft, reference_audit),
            "整篇审稿失败。",
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
            "完整讲解修订失败。",
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
            "参考解析审阅失败。",
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
        safe_error: str,
    ) -> Any:
        for _attempt in range(2):
            try:
                return await self.client.complete_json(
                    system_prompt,
                    user_prompt,
                )
            except Exception:
                continue
        raise LessonQualityError(safe_error) from None

    def _validate_draft(
        self,
        problem: ProblemInput,
        draft: LessonDraft,
    ) -> None:
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

        interactions = [
            moment.interaction
            for moment in draft.moments
            if moment.interaction is not None
        ]
        if not interactions:
            raise LessonQualityError("讲解没有设置学生互动。")

        for interaction in interactions:
            try:
                if interaction.kind == "expression":
                    self.math_engine.parse_expression(
                        interaction.expected_answer
                    )
                elif interaction.kind == "transfer":
                    if not self.math_engine.answers_equivalent(
                        interaction.expected_answer,
                        interaction.expected_answer,
                    ):
                        raise MathValidationError(
                            "Transfer answer is not self-equivalent."
                        )
            except MathValidationError:
                raise LessonQualityError(
                    "讲解中的互动答案未通过数学验证。"
                ) from None

        required_operations = {
            "factor": "factor",
            "quadratic_formula": "quadratic_formula",
            "complete_the_square": "complete_the_square",
        }
        expected_operation = required_operations.get(problem.required_method)
        if expected_operation and expected_operation not in {
            step.operation for step in draft.math_steps
        }:
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
