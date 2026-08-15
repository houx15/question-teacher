"""Deterministic orchestration for the private lesson-preparation chain."""

import inspect
import re
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import (
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from pydantic import BaseModel, ValidationError

from app.pedagogy_rubric import PEDAGOGY_RUBRIC_VERSION
from app.llm_client import (
    ModelCompletion,
    ModelResponseError,
    ModelStructureError,
)
from app.preparation_models import (
    ArtifactRevision,
    InteractionPlan,
    LessonReviewDecision,
    MAX_ROLE_CALL_TOKEN_COUNTER,
    PerformanceScore,
    PreparedLesson,
    ResponsibleRole,
    ReviewFinding,
    ROLE_CALL_TOKEN_USAGE_KEYS,
    ReasoningTrajectory,
    RoleCallRecord,
    SimulationReport,
    SolutionTrace,
    TeachingProgression,
    TeachingScript,
)
from app.preparation_prompts import (
    CLASSROOM_DIRECTOR_SYSTEM,
    INTERACTION_DESIGNER_SYSTEM,
    LESSON_REVIEWER_SYSTEM,
    SCRIPT_TEACHER_SYSTEM,
    SOLUTION_TRACE_SYSTEM,
    STUDENT_SIMULATOR_SYSTEM,
    TEACHING_DESIGNER_SYSTEM,
    TEACHING_PROGRESSION_SYSTEM,
    interaction_plan_prompt,
    performance_score_prompt,
    reasoning_trajectory_prompt,
    lesson_review_prompt,
    solution_trace_prompt,
    student_simulation_prompt,
    teaching_progression_prompt,
    teaching_script_prompt,
    with_output_schema,
)
from app.preparation_validation import (
    PreparationValidationError,
    build_structured_performance_score,
    blocking_signature,
    normalize_performance_control_metadata,
    validate_interaction_plan,
    validate_performance_score,
    validate_reasoning_trajectory,
    validate_prepared_lesson,
    validate_review_decision,
    validate_simulation_report,
    validate_solution_trace,
    validate_teaching_script,
)
from app.schemas import ProblemFocusTarget, ProblemInput
from app.reference_safety import (
    ReferenceContentSafetyError,
    ReferenceSafetyPolicy,
)
from app.teaching_route import FrozenTeachingRoute
from app.teaching_progression_validation import (
    TeachingProgressionValidationError,
    validate_teaching_progression,
)


StageCallback = Callable[[str], Union[None, Awaitable[None]]]
_ModelType = TypeVar("_ModelType", bound=BaseModel)
_SAFE_STRUCTURE_TOKEN = re.compile(r"^[A-Za-z0-9_]{1,64}$")


def _schema_property_names(model_type: Type[BaseModel]) -> set[str]:
    names: set[str] = set()
    stack: List[object] = [model_type.model_json_schema()]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            properties = item.get("properties")
            if isinstance(properties, dict):
                names.update(
                    key
                    for key in properties
                    if isinstance(key, str)
                    and _SAFE_STRUCTURE_TOKEN.fullmatch(key)
                )
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    return names


def _safe_validation_retry_guidance(
    model_type: Type[BaseModel],
    error: ValidationError,
) -> str:
    allowed_names = _schema_property_names(model_type)
    issues = []
    for item in error.errors()[:12]:
        error_type = item.get("type")
        if (
            not isinstance(error_type, str)
            or _SAFE_STRUCTURE_TOKEN.fullmatch(error_type) is None
        ):
            error_type = "validation_error"
        path = []
        for part in item.get("loc", ()):
            if isinstance(part, int) and 0 <= part <= 9999:
                path.append(str(part))
            elif isinstance(part, str) and part in allowed_names:
                path.append(part)
            else:
                path = []
                break
        location = ".".join(path) if path else "schema"
        issue = "%s(%s)" % (location, error_type)
        if issue not in issues:
            issues.append(issue)
    if not issues:
        return "请逐项检查所有必填字段、类型和长度限制。"
    guidance = "请修正这些字段问题：%s。" % "、".join(issues)
    if model_type.__name__ == "PerformanceScore" and any(
        phase in issue
        for issue in issues
        for phase in ("lead_actions", "start_actions", "end_actions")
    ):
        guidance += (
            "lead_actions/start_actions/end_actions 的每一项必须是两层对象："
            "外层仅放 clause_id 和 action，surface/type/target 等只能放在 action 内。"
            "动作字段必须严格使用 system 中的受控模板；"
            "步骤 reveal/scroll/complete 的 target 必须等于 teaching_step_id，"
            "step-aware write 的 board_role 只能是 knowledge_anchor、working、"
            "summary、error_tip、support 之一；lead focus/emphasize 不得带 content "
            "或 board_role；未列字段必须省略或为 null。"
        )
    if model_type.__name__ == "PerformanceScore" and any(
        issue.startswith("board_objects.") for issue in issues
    ):
        guidance += (
            "board_objects 中字段名必须是 line_role，绝不能写 board_role；"
            "对应 write 动作中才使用 board_role，且其值必须与对象 line_role 相同。"
        )
    return guidance


def _safe_structure_error_guidance(error: ModelStructureError) -> str:
    code = error.code
    if not isinstance(code, str) or _SAFE_STRUCTURE_TOKEN.fullmatch(code) is None:
        code = "invalid_structure"
    return "结构错误类型：%s；请返回完整且可解析的 JSON 对象。" % code
DEFAULT_PREPARATION_CAPABILITIES = {
    "interaction_kinds": ["choice"],
    "surfaces": ["problem", "board"],
    "semantic_actions": [
        "write",
        "transform",
        "focus",
        "emphasize",
        "annotate",
        "fade",
        "reveal",
        "clear_focus",
    ],
    "layers": ["base", "micro_explanation", "comparison"],
    "supports_overlays": True,
    "max_interactions": 3,
    "max_options_per_interaction": 4,
}
ROLE_ORDER = {
    "reference_analyst": 0,
    "teaching_designer": 1,
    "interaction_designer": 2,
    "script_teacher": 3,
    "classroom_director": 4,
}
ARTIFACT_DEPENDENCY_ORDER = (
    "solution_trace",
    "reasoning_trajectory",
    "teaching_progression",
    "interaction_plan",
    "teaching_script",
    "performance_score",
    "simulation_report",
)
ARTIFACT_RESPONSIBLE_ROLE = {
    "solution_trace": "reference_analyst",
    "reasoning_trajectory": "teaching_designer",
    "teaching_progression": "teaching_designer",
    "interaction_plan": "interaction_designer",
    "teaching_script": "script_teacher",
    "performance_score": "classroom_director",
    "simulation_report": "student_simulator",
}


def _normalize_solution_trace_control_metadata(
    trace: SolutionTrace,
) -> SolutionTrace:
    """Bind verified-route anchors to their typed source-step authority."""
    payload = trace.model_dump(mode="python")
    changed = False
    for step in payload["source_steps"]:
        anchor = step["source_anchor"]
        if (
            anchor["source_kind"] == "verified_route"
            and anchor["source_id"] != step["source_step_id"]
        ):
            anchor["source_id"] = step["source_step_id"]
            changed = True
    if not changed:
        return trace
    return SolutionTrace.model_validate(payload)


def _normalize_review_control_metadata(
    decision: LessonReviewDecision,
) -> LessonReviewDecision:
    """Approved, finding-free reviews have no repair dependency prefix."""
    if (
        decision.status != "approved"
        or decision.findings
        or not decision.retained_artifacts
    ):
        return decision
    payload = decision.model_dump(mode="python")
    payload["retained_artifacts"] = []
    return LessonReviewDecision.model_validate(payload)


def _normalize_script_section_metadata(
    script: TeachingScript,
    trajectory: ReasoningTrajectory,
    progression: TeachingProgression,
) -> TeachingScript:
    """Bind script metadata to authoritative episode and step ownership."""
    first_episode_id = trajectory.episodes[0].episode_id
    last_episode_id = trajectory.episodes[-1].episode_id
    must_teach_owner = {
        item.must_teach_id: episode.episode_id
        for episode in trajectory.episodes
        for item in episode.must_teach
    }
    episode_steps = {
        episode_id: step.step_id
        for step in progression.steps
        for episode_id in step.episode_ids
    }
    opening_ids = {
        *script.opening_clause_ids,
        *script.method_introduction_clause_ids,
    }
    closing_ids = set(script.closing_summary_clause_ids)
    payload = script.model_dump(mode="python")
    changed = False
    for clause in payload["clauses"]:
        clause_id = clause["clause_id"]
        if clause_id in opening_ids:
            episode_id = first_episode_id
        elif clause_id in closing_ids:
            episode_id = last_episode_id
        else:
            continue
        if clause["episode_id"] != episode_id:
            clause["episode_id"] = episode_id
            changed = True
        normalized_refs = [
            reference
            for reference in clause["must_teach_refs"]
            if must_teach_owner.get(reference) == episode_id
        ]
        if normalized_refs != clause["must_teach_refs"]:
            clause["must_teach_refs"] = normalized_refs
            changed = True
    all_clauses = list(payload["clauses"])
    all_clauses.extend(
        clause
        for response in payload["response_scripts"]
        for clause in response["clauses"]
    )
    for clause in all_clauses:
        authoritative_step_id = episode_steps.get(clause["episode_id"])
        if (
            authoritative_step_id is not None
            and clause["lesson_step_id"] != authoritative_step_id
        ):
            clause["lesson_step_id"] = authoritative_step_id
            changed = True
    if not changed:
        return script
    return TeachingScript.model_validate(payload)


def earliest_responsible_role(
    findings: List[ReviewFinding],
) -> ResponsibleRole:
    artifact_type = earliest_repair_artifact(findings)
    return ARTIFACT_RESPONSIBLE_ROLE[artifact_type]


def earliest_repair_artifact(findings: List[ReviewFinding]) -> str:
    material = [
        finding for finding in findings if finding.severity != "polish"
    ]
    if not material:
        raise RuntimeError("no material review finding to route")
    return min(
        material,
        key=lambda finding: ARTIFACT_DEPENDENCY_ORDER.index(
            finding.artifact_type
        ),
    ).artifact_type


class PreparationFailure(RuntimeError):
    def __init__(self, category: str, role: str, detail: str) -> None:
        super().__init__(detail)
        self.category = category
        self.role = role
        self.detail = detail
        self.audit: Optional["PreparationAuditSnapshot"] = None

    def _attach_audit(self, audit: "PreparationAuditSnapshot") -> None:
        if self.audit is not None:
            raise RuntimeError("preparation failure audit is already attached")
        self.audit = audit


@dataclass
class PreparationState:
    reference_safety: Optional[ReferenceSafetyPolicy] = None
    solution_trace: Optional[SolutionTrace] = None
    reasoning_trajectory: Optional[ReasoningTrajectory] = None
    teaching_progression: Optional[TeachingProgression] = None
    teaching_script: Optional[TeachingScript] = None
    interaction_plan: Optional[InteractionPlan] = None
    performance_score: Optional[PerformanceScore] = None
    simulation_report: Optional[SimulationReport] = None
    review: Optional[LessonReviewDecision] = None
    versions: Dict[str, int] = field(default_factory=dict)
    active_versions: Dict[str, int] = field(default_factory=dict)
    history: List[ArtifactRevision] = field(default_factory=list)
    role_calls: List[RoleCallRecord] = field(default_factory=list)


@dataclass(frozen=True)
class PreparationContext:
    problem: ProblemInput
    teaching_route: FrozenTeachingRoute
    problem_focus_targets: List[ProblemFocusTarget]
    on_stage: Optional[StageCallback]


@dataclass(frozen=True)
class PreparationAuditSnapshot:
    """Content-free, immutable storage for one request's audit metadata."""

    _version_items: Tuple[Tuple[str, int], ...]
    _active_version_items: Tuple[Tuple[str, int], ...]
    _history_json: Tuple[str, ...]
    _role_call_json: Tuple[str, ...]

    @classmethod
    def from_state(
        cls,
        state: PreparationState,
    ) -> "PreparationAuditSnapshot":
        return cls(
            _version_items=tuple(sorted(state.versions.items())),
            _active_version_items=tuple(
                sorted(state.active_versions.items())
            ),
            _history_json=tuple(
                item.model_dump_json() for item in state.history
            ),
            _role_call_json=tuple(
                item.model_dump_json() for item in state.role_calls
            ),
        )

    @property
    def versions(self) -> Dict[str, int]:
        return dict(self._version_items)

    @property
    def active_versions(self) -> Dict[str, int]:
        return dict(self._active_version_items)

    @property
    def history(self) -> List[ArtifactRevision]:
        return [
            ArtifactRevision.model_validate_json(item)
            for item in self._history_json
        ]

    @property
    def role_calls(self) -> List[RoleCallRecord]:
        return [
            RoleCallRecord.model_validate_json(item)
            for item in self._role_call_json
        ]


@dataclass(frozen=True)
class PreparationRunSnapshot:
    """Request-scoped Task 4 result; it is not a complete PreparedLesson."""

    _solution_trace_json: str
    _reasoning_trajectory_json: str
    audit: PreparationAuditSnapshot

    @classmethod
    def from_state(
        cls,
        state: PreparationState,
    ) -> "PreparationRunSnapshot":
        if state.solution_trace is None or state.reasoning_trajectory is None:
            raise RuntimeError("early preparation artifacts are incomplete")
        return cls(
            _solution_trace_json=state.solution_trace.model_dump_json(),
            _reasoning_trajectory_json=(
                state.reasoning_trajectory.model_dump_json()
            ),
            audit=PreparationAuditSnapshot.from_state(state),
        )

    @property
    def solution_trace(self) -> SolutionTrace:
        return SolutionTrace.model_validate_json(self._solution_trace_json)

    @property
    def reasoning_trajectory(self) -> ReasoningTrajectory:
        return ReasoningTrajectory.model_validate_json(
            self._reasoning_trajectory_json
        )

    @property
    def versions(self) -> Dict[str, int]:
        return self.audit.versions

    @property
    def active_versions(self) -> Dict[str, int]:
        return self.audit.active_versions

    @property
    def history(self) -> List[ArtifactRevision]:
        return self.audit.history

    @property
    def role_calls(self) -> List[RoleCallRecord]:
        return self.audit.role_calls


@dataclass(frozen=True)
class PreparedLessonRun:
    """Approved lesson and audit captured from the same preparation request."""

    _prepared_lesson_json: str
    audit: PreparationAuditSnapshot

    @classmethod
    def from_state(
        cls,
        prepared_lesson: PreparedLesson,
        state: PreparationState,
    ) -> "PreparedLessonRun":
        if type(prepared_lesson) is not PreparedLesson:
            raise TypeError("prepared lesson must be an exact PreparedLesson")
        if prepared_lesson.review.status != "approved":
            raise RuntimeError("prepared lesson run requires approval")
        return cls(
            _prepared_lesson_json=prepared_lesson.model_dump_json(),
            audit=PreparationAuditSnapshot.from_state(state),
        )

    @property
    def prepared_lesson(self) -> PreparedLesson:
        return PreparedLesson.model_validate_json(
            self._prepared_lesson_json
        )


class LessonPreparationPipeline:
    MAX_STRUCTURE_ATTEMPTS = 3
    MAX_REPAIR_CYCLES = 8

    def __init__(
        self,
        client: object,
        capabilities: Optional[Dict[str, object]] = None,
    ) -> None:
        self.client = client
        self.capabilities = dict(
            DEFAULT_PREPARATION_CAPABILITIES
            if capabilities is None
            else capabilities
        )
        self._active_state: ContextVar[Optional[PreparationState]] = ContextVar(
            "active_preparation_state_%x" % id(self),
            default=None,
        )

    async def prepare(
        self,
        problem: ProblemInput,
        teaching_route: FrozenTeachingRoute,
        problem_focus_targets: List[ProblemFocusTarget],
        on_stage: Optional[StageCallback] = None,
    ) -> PreparedLesson:
        run = await self.prepare_with_audit(
            problem,
            teaching_route,
            problem_focus_targets,
            on_stage,
        )
        return run.prepared_lesson

    async def prepare_with_audit(
        self,
        problem: ProblemInput,
        teaching_route: FrozenTeachingRoute,
        problem_focus_targets: List[ProblemFocusTarget],
        on_stage: Optional[StageCallback] = None,
    ) -> PreparedLessonRun:
        state = PreparationState(
            reference_safety=ReferenceSafetyPolicy.from_problem(problem)
        )
        state_token = self._active_state.set(state)
        try:
            await self._populate_early_state(
                state,
                problem,
                teaching_route,
                problem_focus_targets,
                on_stage,
            )
            prepared_lesson = await self._continue_preparation(
                state,
                problem,
                teaching_route,
                problem_focus_targets,
                on_stage,
            )
            return PreparedLessonRun.from_state(prepared_lesson, state)
        except PreparationFailure as failure:
            if failure.audit is None:
                failure._attach_audit(
                    PreparationAuditSnapshot.from_state(state)
                )
            raise
        finally:
            self._active_state.reset(state_token)

    async def prepare_early(
        self,
        problem: ProblemInput,
        teaching_route: FrozenTeachingRoute,
        problem_focus_targets: List[ProblemFocusTarget],
        on_stage: Optional[StageCallback] = None,
    ) -> PreparationRunSnapshot:
        """Run Task 4 only and return a defensive, request-scoped snapshot."""
        state = PreparationState(
            reference_safety=ReferenceSafetyPolicy.from_problem(problem)
        )
        state_token = self._active_state.set(state)
        try:
            await self._populate_early_state(
                state,
                problem,
                teaching_route,
                problem_focus_targets,
                on_stage,
            )
            return PreparationRunSnapshot.from_state(state)
        except PreparationFailure as failure:
            if failure.audit is None:
                failure._attach_audit(
                    PreparationAuditSnapshot.from_state(state)
                )
            raise
        finally:
            self._active_state.reset(state_token)

    async def _populate_early_state(
        self,
        state: PreparationState,
        problem: ProblemInput,
        teaching_route: FrozenTeachingRoute,
        problem_focus_targets: List[ProblemFocusTarget],
        on_stage: Optional[StageCallback],
    ) -> None:
        await self._create_solution_trace(
            state,
            problem,
            teaching_route,
            problem_focus_targets,
            on_stage,
        )
        await self._create_reasoning_trajectory(
            state,
            problem,
            on_stage,
        )

    async def _continue_preparation(
        self,
        state: PreparationState,
        problem: ProblemInput,
        teaching_route: FrozenTeachingRoute,
        problem_focus_targets: List[ProblemFocusTarget],
        on_stage: Optional[StageCallback],
    ) -> PreparedLesson:
        """Build, simulate, review, and converge one request-scoped lesson."""
        context = PreparationContext(
            problem=problem,
            teaching_route=teaching_route,
            problem_focus_targets=list(problem_focus_targets),
            on_stage=on_stage,
        )
        await self._create_teaching_progression(
            state,
            problem_focus_targets,
            on_stage,
        )
        await self._create_interaction_plan(state, on_stage)
        await self._create_teaching_script(state, on_stage)
        await self._create_performance_score(
            state,
            problem_focus_targets,
            on_stage,
        )
        await self._emit(on_stage, "模拟学生并审核课程")
        await self._simulate_student(state)
        review_context_id = "review-context-1"
        await self._review_lesson(state, review_context_id)

        repair_count = 0
        previous_signature: Optional[str] = None
        while state.review is not None and state.review.status != "approved":
            if state.review.status == "failed":
                self._raise_not_converged()
            signature = blocking_signature(state.review)
            if signature == previous_signature:
                fresh_context_id = "review-context-%d" % (repair_count + 2)
                await self._emit(on_stage, "模拟学生并审核课程")
                await self._review_lesson(state, fresh_context_id)
                if (
                    state.review.status != "approved"
                    and blocking_signature(state.review) == signature
                ):
                    self._raise_not_converged()
                previous_signature = None
                continue
            if repair_count >= self.MAX_REPAIR_CYCLES:
                self._raise_not_converged()
            previous_signature = signature
            artifact_type = earliest_repair_artifact(state.review.findings)
            await self._repair_from(
                artifact_type,
                state,
                state.review.findings,
                context,
            )
            repair_count += 1
            await self._emit(on_stage, "模拟学生并审核课程")
            if artifact_type != "simulation_report":
                await self._simulate_student(state)
            await self._review_lesson(state, review_context_id)

        if state.review is None or state.review.status != "approved":
            self._raise_not_converged()
        expected_active = {
            "solution_trace",
            "reasoning_trajectory",
            "teaching_progression",
            "interaction_plan",
            "teaching_script",
            "performance_score",
            "simulation_report",
        }
        if (
            set(state.active_versions) != expected_active
            or any(
                state.active_versions[item] != state.versions.get(item)
                for item in expected_active
            )
        ):
            self._raise_not_converged()
        required = (
            state.solution_trace,
            state.reasoning_trajectory,
            state.teaching_progression,
            state.interaction_plan,
            state.teaching_script,
            state.performance_score,
            state.simulation_report,
        )
        if any(item is None for item in required):
            raise RuntimeError("approved preparation state is incomplete")
        prepared = PreparedLesson(
            rubric_version=PEDAGOGY_RUBRIC_VERSION,
            solution_trace=state.solution_trace,
            reasoning_trajectory=state.reasoning_trajectory,
            teaching_progression=state.teaching_progression,
            teaching_script=state.teaching_script,
            interaction_plan=state.interaction_plan,
            performance_score=state.performance_score,
            simulation_report=state.simulation_report,
            review=state.review,
            repair_count=repair_count,
            artifact_history=list(state.history),
        )
        try:
            validate_prepared_lesson(
                prepared,
                teaching_route,
                problem_focus_targets,
                active_versions=state.active_versions,
            )
        except PreparationValidationError:
            raise PreparationFailure(
                category="review_not_converged",
                role="lesson_reviewer",
                detail="课程审核未收敛。",
            ) from None
        return prepared

    @staticmethod
    def _raise_not_converged() -> None:
        raise PreparationFailure(
            category="review_not_converged",
            role="lesson_reviewer",
            detail="课程审核未收敛。",
        )

    async def _create_solution_trace(
        self,
        state: PreparationState,
        problem: ProblemInput,
        teaching_route: FrozenTeachingRoute,
        problem_focus_targets: List[ProblemFocusTarget],
        on_stage: Optional[StageCallback],
        repair: Optional[Dict[str, object]] = None,
        finding_ids: Optional[List[str]] = None,
    ) -> SolutionTrace:
        self._require_active_state(state)
        await self._emit(on_stage, "整理参考解析")
        base_prompt = self._build_prompt(
            state,
            "reference_analyst",
            solution_trace_prompt,
            problem,
            teaching_route,
            problem_focus_targets,
            repair,
        )
        trace = await self._complete_model(
            "reference_analyst",
            SOLUTION_TRACE_SYSTEM,
            base_prompt,
            SolutionTrace,
        )
        trace = _normalize_solution_trace_control_metadata(trace)
        try:
            validate_solution_trace(trace, teaching_route)
        except PreparationValidationError:
            self._mark_last_call_failed(
                state,
                "reference_analyst",
                "reference_trace_failed",
            )
            raise PreparationFailure(
                category="reference_trace_failed",
                role="reference_analyst",
                detail="参考解析轨迹未通过确定性校验。",
            ) from None
        try:
            trace = self._reference_safety(state).sanitize_solution_trace(
                trace,
                teaching_route,
            )
        except ReferenceContentSafetyError:
            self._raise_reference_content_leak(state, "reference_analyst")
        # Validate the server-owned projection too, so route reconstruction
        # cannot silently weaken the cross-artifact contract.
        try:
            validate_solution_trace(trace, teaching_route)
        except PreparationValidationError:
            self._mark_last_call_failed(
                state,
                "reference_analyst",
                "reference_trace_failed",
            )
            raise PreparationFailure(
                category="reference_trace_failed",
                role="reference_analyst",
                detail="参考解析轨迹未通过确定性校验。",
            ) from None
        self._accept_artifact(
            state,
            artifact_type="solution_trace",
            responsible_role="reference_analyst",
            artifact=trace,
            finding_ids=finding_ids,
        )
        return trace

    async def _create_reasoning_trajectory(
        self,
        state: PreparationState,
        problem: ProblemInput,
        on_stage: Optional[StageCallback],
        repair: Optional[Dict[str, object]] = None,
        finding_ids: Optional[List[str]] = None,
    ) -> ReasoningTrajectory:
        self._require_active_state(state)
        if state.solution_trace is None:
            raise RuntimeError("solution trace must exist before trajectory design")
        await self._emit(on_stage, "设计解题思维轨迹")
        base_prompt = self._build_prompt(
            state,
            "teaching_designer",
            reasoning_trajectory_prompt,
            problem,
            state.solution_trace,
            self.capabilities,
            repair,
        )
        validation_error: Optional[PreparationValidationError] = None
        for semantic_attempt in range(3):
            attempt_prompt = base_prompt
            if semantic_attempt and validation_error is not None:
                correction_guidance = (
                    "episode.source_step_ids 只能逐字复制输入 "
                    "SolutionTrace.source_steps 的 source_step_id；"
                    "按原顺序覆盖全部 source step，不得杜撰、改名或遗漏。"
                )
                if validation_error.code in {
                    "must_teach_evidence_missing",
                    "must_teach_content_anchor_missing",
                    "must_teach_evidence_alignment_invalid",
                }:
                    correction_guidance = (
                        "请重写该 must_teach：student_display_evidence 必须完整保留"
                        "本项 content 原文作为语义锚点，可在前后补充自然解释；"
                        "student_spoken_evidence 必须自然读出同一内容及所有数学运算。"
                    )
                attempt_prompt += (
                    "\n上一次解题思维轨迹未通过确定性校验。"
                    "请完整重写 ReasoningTrajectory，不要局部补丁。"
                    "稳定错误码："
                    + validation_error.code
                    + "；对象："
                    + validation_error.artifact_id
                    + "。"
                    + correction_guidance
                    + "不要复述上一次输出。"
                )
            trajectory = await self._complete_model(
                "teaching_designer",
                TEACHING_DESIGNER_SYSTEM,
                attempt_prompt,
                ReasoningTrajectory,
            )
            try:
                validate_reasoning_trajectory(trajectory, state.solution_trace)
                break
            except PreparationValidationError as error:
                validation_error = error
                self._mark_last_call_failed(
                    state,
                    "teaching_designer",
                    "reasoning_design_failed",
                )
                if error.code not in {
                    "episode_source_missing",
                    "trace_step_uncovered",
                    "must_teach_evidence_missing",
                    "must_teach_content_anchor_missing",
                    "must_teach_evidence_alignment_invalid",
                } or semantic_attempt == 2:
                    raise PreparationFailure(
                        category="reasoning_design_failed",
                        role="teaching_designer",
                        detail="解题思维轨迹未通过确定性校验。",
                    ) from None
        self._accept_artifact(
            state,
            artifact_type="reasoning_trajectory",
            responsible_role="teaching_designer",
            artifact=trajectory,
            finding_ids=finding_ids,
        )
        return trajectory

    async def _create_teaching_progression(
        self,
        state: PreparationState,
        problem_focus_targets: List[ProblemFocusTarget],
        on_stage: Optional[StageCallback],
        repair: Optional[Dict[str, object]] = None,
        finding_ids: Optional[List[str]] = None,
    ) -> TeachingProgression:
        self._require_active_state(state)
        if state.reasoning_trajectory is None:
            raise RuntimeError(
                "reasoning trajectory must exist before progression design"
            )
        await self._emit(on_stage, "设计教学推进")
        base_prompt = self._build_prompt(
            state,
            "teaching_designer",
            teaching_progression_prompt,
            state.reasoning_trajectory,
            problem_focus_targets,
            repair,
        )
        validation_error: Optional[TeachingProgressionValidationError] = None
        for semantic_attempt in range(2):
            attempt_prompt = base_prompt
            if semantic_attempt and validation_error is not None:
                correction_guidance = "请逐字段对照系统规则修正该对象。"
                if (
                    validation_error.code
                    == "progression_why_not_explanatory"
                ):
                    correction_guidance = (
                        "请重写该步骤的 why_now，点明已知条件、前一步结果或"
                        "当前学生困难与本步动作之间的具体因果。"
                    )
                elif validation_error.code in {
                    "progression_evidence_target_duplicate",
                    "progression_evidence_target_coverage_invalid",
                }:
                    correction_guidance = (
                        "请先删除所有 evidence_target_ids 重复项，再把 "
                        "problem_targets 中每个 target_id 分配给唯一一个负责步骤。"
                        "检查所有步骤 evidence_target_ids 的并集与输入 target_id "
                        "全集完全相同，且每个 ID 在整套推进中只出现一次。"
                    )
                attempt_prompt += (
                    "\n上一次教学推进未通过确定性校验。"
                    "请完整重写 TeachingProgression，不要局部补丁。"
                    "稳定错误码："
                    + validation_error.code
                    + "；对象："
                    + validation_error.artifact_id
                    + "。"
                    + correction_guidance
                )
            progression = await self._complete_model(
                "teaching_designer",
                TEACHING_PROGRESSION_SYSTEM,
                attempt_prompt,
                TeachingProgression,
            )
            try:
                validate_teaching_progression(
                    progression,
                    state.reasoning_trajectory,
                    problem_focus_targets,
                )
                break
            except TeachingProgressionValidationError as error:
                validation_error = error
                self._mark_last_call_failed(
                    state,
                    "teaching_designer",
                    "teaching_progression_failed",
                )
        else:
            raise PreparationFailure(
                category="teaching_progression_failed",
                role="teaching_designer",
                detail="教学推进未通过确定性校验。",
            ) from None
        self._accept_artifact(
            state,
            artifact_type="teaching_progression",
            responsible_role="teaching_designer",
            artifact=progression,
            finding_ids=finding_ids,
        )
        return progression

    async def _create_teaching_script(
        self,
        state: PreparationState,
        on_stage: Optional[StageCallback],
        repair: Optional[Dict[str, object]] = None,
        finding_ids: Optional[List[str]] = None,
    ) -> TeachingScript:
        self._require_active_state(state)
        if (
            state.reasoning_trajectory is None
            or state.teaching_progression is None
            or state.interaction_plan is None
        ):
            raise RuntimeError(
                "progression and interaction plan must exist before script writing"
            )
        await self._emit(on_stage, "编写讲稿")
        base_prompt = self._build_prompt(
            state,
            "script_teacher",
            teaching_script_prompt,
            state.teaching_progression,
            state.interaction_plan,
            state.reasoning_trajectory,
            repair,
        )
        validation_error: Optional[PreparationValidationError] = None
        for semantic_attempt in range(3):
            attempt_prompt = base_prompt
            if semantic_attempt and validation_error is not None:
                correction_guidance = (
                    "请确保每条主线和 response clause 都有非空 "
                    "display_text，并与本条 spoken_text 表达同一教学内容。"
                )
                if validation_error.code == "must_teach_evidence_missing":
                    correction_guidance = (
                        "请按 must_teach_id 找到输入 must_teach_evidence，"
                        "在绑定该 ID 的同一 clause 中逐字包含对应的 "
                        "student_display_evidence 和 student_spoken_evidence，"
                        "不得改写、遗漏或放到其他 episode。"
                    )
                elif validation_error.code in {
                    "response_semantic_anchor_missing",
                    "response_remediation_insufficient",
                }:
                    correction_guidance = (
                        "请找到该 response 对应的错误 option，在同一错误分支中"
                        "分别直接包含该 option 的 misconception 原文和 "
                        "incorrect_feedback_by_option 纠正动作原文。两项必须是"
                        "可独立识别、互不重叠的语义单位，不得用泛化安慰或填充文字替代。"
                    )
                elif validation_error.code in {
                    "response_classification_invalid",
                    "response_depth_invalid",
                    "response_error_code_invalid",
                }:
                    correction_guidance = (
                        "请按 interaction_id 与 option_id 逐项重建 response："
                        "正确项必须 classification=correct、depth=brief、"
                        "error_code=null；错误项必须 classification=incorrect，"
                        "并逐字复制该 option 的 error_code 与 remediation_depth，"
                        "不得翻译、改名或自造。"
                    )
                elif validation_error.code == "response_private_answer_leakage":
                    correction_guidance = (
                        "请删除所有 response display_text/spoken_text 中的私有 "
                        "canonical_answer、隐藏答案和正确结果。错误分支只使用"
                        "公开 option label、本项 misconception 与 "
                        "incorrect_feedback_by_option 纠正动作，不得复述答案值。"
                    )
                attempt_prompt += (
                    "\n上一次讲稿未通过确定性校验。"
                    "请完整重写 TeachingScript，不要局部补丁。"
                    "稳定错误码："
                    + validation_error.code
                    + "；对象："
                    + validation_error.artifact_id
                    + "。"
                    + correction_guidance
                    + "不要复述上一次输出。"
                )
            script = await self._complete_model(
                "script_teacher",
                SCRIPT_TEACHER_SYSTEM,
                attempt_prompt,
                TeachingScript,
            )
            script = _normalize_script_section_metadata(
                script,
                state.reasoning_trajectory,
                state.teaching_progression,
            )
            try:
                validate_teaching_script(
                    script,
                    state.reasoning_trajectory,
                    state.teaching_progression,
                    state.interaction_plan,
                )
                break
            except PreparationValidationError as error:
                validation_error = error
                self._mark_last_call_failed(
                    state,
                    "script_teacher",
                    "teaching_script_failed",
                )
                if error.code not in {
                    "clause_display_missing",
                    "must_teach_evidence_missing",
                    "response_semantic_anchor_missing",
                    "response_remediation_insufficient",
                    "response_classification_invalid",
                    "response_depth_invalid",
                    "response_error_code_invalid",
                    "response_private_answer_leakage",
                } or semantic_attempt == 2:
                    raise PreparationFailure(
                        category="teaching_script_failed",
                        role="script_teacher",
                        detail="讲稿未通过确定性校验。",
                    ) from None
        self._accept_artifact(
            state,
            artifact_type="teaching_script",
            responsible_role="script_teacher",
            artifact=script,
            finding_ids=finding_ids,
        )
        return script

    async def _create_interaction_plan(
        self,
        state: PreparationState,
        on_stage: Optional[StageCallback],
        repair: Optional[Dict[str, object]] = None,
        finding_ids: Optional[List[str]] = None,
    ) -> InteractionPlan:
        self._require_active_state(state)
        if state.teaching_progression is None:
            raise RuntimeError(
                "teaching progression must exist before interaction design"
            )
        await self._emit(on_stage, "设计互动")
        base_prompt = self._build_prompt(
            state,
            "interaction_designer",
            interaction_plan_prompt,
            state.teaching_progression,
            repair,
        )
        validation_error: Optional[PreparationValidationError] = None
        for semantic_attempt in range(2):
            attempt_prompt = base_prompt
            if semantic_attempt and validation_error is not None:
                correction_guidance = (
                    "请让 why_pause 逐字包含所属步骤的 "
                    "checkpoint.diagnostic_goal，并解释为什么要在当前步骤"
                    "暂停检查这个目标。"
                )
                if validation_error.code == "interaction_checkpoint_missing":
                    correction_guidance = (
                        "请先检查 TeachingProgression：每个 interaction 的 "
                        "teaching_step_id 必须逐字引用一个 checkpoint 非空的步骤，"
                        "episode_id 必须属于该同一步；没有 checkpoint 的步骤不得"
                        "放互动。why_pause 再逐字包含该 checkpoint.diagnostic_goal。"
                    )
                attempt_prompt += (
                    "\n上一次互动意图未通过确定性校验。"
                    "请完整重写 InteractionPlan，不要局部补丁。"
                    "稳定错误码："
                    + validation_error.code
                    + "；对象："
                    + validation_error.artifact_id
                    + "。"
                    + correction_guidance
                    + "不要复述上一次输出。"
                )
            plan = await self._complete_model(
                "interaction_designer",
                INTERACTION_DESIGNER_SYSTEM,
                attempt_prompt,
                InteractionPlan,
            )
            try:
                validate_interaction_plan(plan, state.teaching_progression)
                break
            except PreparationValidationError as error:
                validation_error = error
                self._mark_last_call_failed(
                    state,
                    "interaction_designer",
                    "interaction_plan_failed",
                )
                if error.code not in {
                    "interaction_why_pause_invalid",
                    "interaction_checkpoint_missing",
                } or semantic_attempt:
                    raise PreparationFailure(
                        category="interaction_plan_failed",
                        role="interaction_designer",
                        detail="互动意图未通过确定性校验。",
                    ) from None
        self._accept_artifact(
            state,
            artifact_type="interaction_plan",
            responsible_role="interaction_designer",
            artifact=plan,
            finding_ids=finding_ids,
        )
        return plan

    async def _create_performance_score(
        self,
        state: PreparationState,
        problem_focus_targets: List[ProblemFocusTarget],
        on_stage: Optional[StageCallback],
        repair: Optional[Dict[str, object]] = None,
        finding_ids: Optional[List[str]] = None,
    ) -> PerformanceScore:
        self._require_active_state(state)
        if state.teaching_script is None or state.interaction_plan is None:
            raise RuntimeError(
                "script and interaction plan must exist before direction"
            )
        if state.teaching_progression is None:
            raise RuntimeError(
                "teaching progression must exist before direction"
            )
        await self._emit(on_stage, "编排板书与高亮")
        base_prompt = self._build_prompt(
            state,
            "classroom_director",
            performance_score_prompt,
            problem_focus_targets,
            state.teaching_script,
            state.interaction_plan,
            self.capabilities,
            repair,
            state.teaching_progression,
        )
        validation_error: Optional[PreparationValidationError] = None
        retryable_codes = {
            "visual_target_invalid",
            "visual_action_too_early",
            "overlay_transition_invalid",
            "cue_clause_coverage_invalid",
            "structured_step_lifecycle_invalid",
            "structured_support_lifecycle_invalid",
        }
        for semantic_attempt in range(3):
            attempt_prompt = base_prompt
            if semantic_attempt and validation_error is not None:
                correction_guidance = (
                    "所有 problem 动作 target 只能逐字引用 problem_targets.target_id；"
                    "所有普通 board 动作 target/source/relation_target 只能引用已声明"
                    "且在当前时点已由 write/transform 创建的 board_object_id。"
                    "write.content 必须与对应 board_object.content 完全一致；"
                    "同一 cue 的 clear_focus/fade 只能关闭该 cue 已开启的 focus/emphasize。"
                )
                if (
                    validation_error.code == "visual_target_invalid"
                    and "包含不允许的动作类型" in validation_error.detail
                ):
                    correction_guidance = (
                        "严格按阶段重排动作：lead_actions 仅 focus/emphasize；"
                        "start_actions 仅 write/transform/focus/emphasize/annotate/"
                        "reveal/reveal_step_header/scroll_to_step/"
                        "open_supporting_explanation；end_actions 仅 clear_focus/fade/"
                        "write/reveal_step_header/complete_step/scroll_to_step/"
                        "close_supporting_explanation。不得把 complete、close、fade、"
                        "clear_focus 放进 start_actions。"
                    )
                if validation_error.code == "visual_action_too_early":
                    correction_guidance = (
                        "请按 cue 顺序重排动作：problem 的 focus/emphasize 只能绑定到"
                        "其 clause.math_references 已明确包含该 problem target 数学内容的"
                        "子句；如果当前子句尚未提到它，就删除该可选动作。board 对象必须"
                        "先 write/transform 创建，之后的 cue 才能 focus/emphasize/"
                        "annotate/fade；lead_actions 不得引用本 cue 才创建的对象。"
                    )
                elif validation_error.code == "overlay_transition_invalid":
                    correction_guidance = (
                        "请删除不必要的 overlay；如确需 overlay，enter 与 return 必须"
                        "位于不同 cue 边界，中间至少一个 cue，且 overlay 内不得引用"
                        "已经 return 的对象。"
                    )
                elif validation_error.code == "cue_clause_coverage_invalid":
                    correction_guidance = (
                        "请先列出 TeachingScript 的全部主线 clauses 与全部 response "
                        "clauses，并按输入原顺序分配到 cues.clause_ids：每个 clause_id "
                        "恰好出现一次，不得重复、遗漏、改名或新增。每个动作外层的 "
                        "clause_id 必须属于所在 cue.clause_ids。"
                    )
                attempt_prompt += (
                    "\n上一次板书编排未通过确定性校验。"
                    "请完整重写 PerformanceScore，不要局部补丁。"
                    "稳定错误码："
                    + validation_error.code
                    + "；对象："
                    + validation_error.artifact_id
                    + "。"
                    + correction_guidance
                    + "不要复述上一次输出。"
                )
            score = await self._complete_model(
                "classroom_director",
                CLASSROOM_DIRECTOR_SYSTEM,
                attempt_prompt,
                PerformanceScore,
            )
            try:
                validate_performance_score(
                    score,
                    problem_focus_targets,
                    state.teaching_progression,
                    state.teaching_script,
                    state.interaction_plan,
                )
                break
            except PreparationValidationError as initial_error:
                validation_error = initial_error
                if initial_error.code in {
                    "overlay_transition_invalid",
                    "visual_action_too_early",
                }:
                    score = normalize_performance_control_metadata(
                        score,
                        problem_focus_targets,
                        state.teaching_script,
                        state.teaching_progression,
                    )
                    try:
                        validate_performance_score(
                            score,
                            problem_focus_targets,
                            state.teaching_progression,
                            state.teaching_script,
                            state.interaction_plan,
                        )
                        break
                    except PreparationValidationError as normalized_error:
                        validation_error = normalized_error
                if validation_error.code in retryable_codes:
                    try:
                        deterministic_score = build_structured_performance_score(
                            state.teaching_progression,
                            state.teaching_script,
                        )
                        validate_performance_score(
                            deterministic_score,
                            problem_focus_targets,
                            state.teaching_progression,
                            state.teaching_script,
                            state.interaction_plan,
                        )
                        score = deterministic_score
                        break
                    except PreparationValidationError as fallback_error:
                        validation_error = fallback_error
                if semantic_attempt < 2:
                    self._mark_last_call_failed(
                        state,
                        "classroom_director",
                        "performance_score_failed",
                    )
                if validation_error.code not in retryable_codes or semantic_attempt == 2:
                    raise PreparationFailure(
                        category="performance_score_failed",
                        role="classroom_director",
                        detail="板书与高亮编排未通过确定性校验。",
                    ) from None
        self._accept_artifact(
            state,
            artifact_type="performance_score",
            responsible_role="classroom_director",
            artifact=score,
            finding_ids=finding_ids,
        )
        return score

    async def _simulate_student(
        self,
        state: PreparationState,
        repair: Optional[Dict[str, object]] = None,
        finding_ids: Optional[List[str]] = None,
    ) -> SimulationReport:
        self._require_active_state(state)
        if (
            state.reasoning_trajectory is None
            or state.teaching_script is None
            or state.interaction_plan is None
            or state.performance_score is None
        ):
            raise RuntimeError(
                "all classroom artifacts must exist before simulation"
            )
        report = await self._complete_model(
            "student_simulator",
            STUDENT_SIMULATOR_SYSTEM,
            self._build_prompt(
                state,
                "student_simulator",
                student_simulation_prompt,
                state.reasoning_trajectory,
                state.teaching_script,
                state.interaction_plan,
                state.performance_score,
                repair,
            ),
            SimulationReport,
        )
        try:
            validate_simulation_report(
                report,
                state.reasoning_trajectory,
                state.interaction_plan,
            )
        except PreparationValidationError:
            self._mark_last_call_failed(
                state,
                "student_simulator",
                "simulation_failed",
            )
            raise PreparationFailure(
                category="simulation_failed",
                role="student_simulator",
                detail="学生模拟未通过确定性校验。",
            ) from None
        self._accept_artifact(
            state,
            artifact_type="simulation_report",
            responsible_role="student_simulator",
            artifact=report,
            finding_ids=finding_ids,
        )
        return report

    async def _review_lesson(
        self,
        state: PreparationState,
        reviewer_context_id: str,
    ) -> LessonReviewDecision:
        self._require_active_state(state)
        if any(
            item is None
            for item in (
                state.solution_trace,
                state.reasoning_trajectory,
                state.teaching_progression,
                state.teaching_script,
                state.interaction_plan,
                state.performance_score,
                state.simulation_report,
            )
        ):
            raise RuntimeError("all prepared artifacts must exist before review")
        decision = await self._complete_model(
            "lesson_reviewer",
            LESSON_REVIEWER_SYSTEM,
            self._build_prompt(
                state,
                "lesson_reviewer",
                lesson_review_prompt,
                {
                    "solution_trace": state.solution_trace,
                    "reasoning_trajectory": state.reasoning_trajectory,
                    "teaching_progression": state.teaching_progression,
                    "interaction_plan": state.interaction_plan,
                    "teaching_script": state.teaching_script,
                    "performance_score": state.performance_score,
                },
                state.simulation_report,
                reviewer_context_id,
            ),
            LessonReviewDecision,
        )
        decision = _normalize_review_control_metadata(decision)
        try:
            self._reference_safety(state).ensure_safe(
                decision,
                downstream_of_sanitized_trace=True,
            )
        except ReferenceContentSafetyError:
            self._raise_reference_content_leak(state, "lesson_reviewer")
        try:
            validate_review_decision(
                decision,
                state.solution_trace,
                state.reasoning_trajectory,
                state.teaching_script,
                state.interaction_plan,
                state.performance_score,
                state.simulation_report,
                progression=state.teaching_progression,
            )
        except PreparationValidationError:
            self._mark_last_call_failed(
                state,
                "lesson_reviewer",
                "review_not_converged",
            )
            self._raise_not_converged()
        self._set_last_call_finding_ids(
            state,
            "lesson_reviewer",
            [item.finding_id for item in decision.findings],
        )
        state.review = decision
        return decision

    async def _repair_from(
        self,
        artifact_type: str,
        state: PreparationState,
        findings: List[ReviewFinding],
        context: PreparationContext,
    ) -> None:
        repairable = ARTIFACT_DEPENDENCY_ORDER
        if artifact_type not in repairable:
            raise RuntimeError("unknown repair artifact")
        self._require_active_state(state)
        routed = [
            finding
            for finding in findings
            if finding.severity != "polish"
            and finding.artifact_type == artifact_type
        ]
        if not routed:
            raise RuntimeError("repair route has no material finding")
        await self._emit(context.on_stage, "正在修订完整讲解")
        repair = self._repair_request(artifact_type, state, routed)
        finding_ids = [item.finding_id for item in routed]
        state.review = None
        start_index = ARTIFACT_DEPENDENCY_ORDER.index(artifact_type)
        self._deactivate_artifacts(
            state,
            *ARTIFACT_DEPENDENCY_ORDER[start_index:],
        )

        if artifact_type == "solution_trace":
            await self._create_solution_trace(
                state,
                context.problem,
                context.teaching_route,
                context.problem_focus_targets,
                context.on_stage,
                repair,
                finding_ids,
            )
            await self._create_reasoning_trajectory(
                state, context.problem, context.on_stage
            )
            await self._create_teaching_progression(
                state,
                context.problem_focus_targets,
                context.on_stage,
            )
            await self._create_interaction_plan(state, context.on_stage)
            await self._create_teaching_script(state, context.on_stage)
            await self._create_performance_score(
                state,
                context.problem_focus_targets,
                context.on_stage,
            )
        elif artifact_type == "reasoning_trajectory":
            await self._create_reasoning_trajectory(
                state,
                context.problem,
                context.on_stage,
                repair,
                finding_ids,
            )
            await self._create_teaching_progression(
                state,
                context.problem_focus_targets,
                context.on_stage,
            )
            await self._create_interaction_plan(state, context.on_stage)
            await self._create_teaching_script(state, context.on_stage)
            await self._create_performance_score(
                state,
                context.problem_focus_targets,
                context.on_stage,
            )
        elif artifact_type == "teaching_progression":
            await self._create_teaching_progression(
                state,
                context.problem_focus_targets,
                context.on_stage,
                repair,
                finding_ids,
            )
            await self._create_interaction_plan(state, context.on_stage)
            await self._create_teaching_script(state, context.on_stage)
            await self._create_performance_score(
                state,
                context.problem_focus_targets,
                context.on_stage,
            )
        elif artifact_type == "interaction_plan":
            await self._create_interaction_plan(
                state,
                context.on_stage,
                repair,
                finding_ids,
            )
            await self._create_teaching_script(state, context.on_stage)
            await self._create_performance_score(
                state,
                context.problem_focus_targets,
                context.on_stage,
            )
        elif artifact_type == "teaching_script":
            await self._create_teaching_script(
                state,
                context.on_stage,
                repair,
                finding_ids,
            )
            await self._create_performance_score(
                state,
                context.problem_focus_targets,
                context.on_stage,
            )
        elif artifact_type == "performance_score":
            await self._create_performance_score(
                state,
                context.problem_focus_targets,
                context.on_stage,
                repair,
                finding_ids,
            )
        elif artifact_type == "simulation_report":
            await self._simulate_student(
                state,
                repair=repair,
                finding_ids=finding_ids,
            )

    @staticmethod
    def _deactivate_artifacts(
        state: PreparationState,
        *artifact_types: str,
    ) -> None:
        for artifact_type in artifact_types:
            setattr(state, artifact_type, None)
            state.active_versions.pop(artifact_type, None)

    @staticmethod
    def _repair_request(
        artifact_type: str,
        state: PreparationState,
        findings: List[ReviewFinding],
    ) -> Dict[str, object]:
        artifact_types = list(ARTIFACT_DEPENDENCY_ORDER)
        artifact_index = artifact_types.index(artifact_type)
        retained = {}
        for retained_type in artifact_types[: artifact_index + 1]:
            artifact = getattr(state, retained_type)
            if artifact is None:
                raise RuntimeError("repair source artifact is missing")
            retained[retained_type] = artifact
        return {
            "finding_ids": [item.finding_id for item in findings],
            "evidence": [item.evidence for item in findings],
            "requested_changes": [
                item.requested_change for item in findings
            ],
            "current_artifact_version": state.versions[artifact_type],
            "retained_artifacts": retained,
        }

    def _build_prompt(
        self,
        state: PreparationState,
        role: str,
        builder: Callable[..., str],
        *args: object,
    ) -> str:
        started = time.monotonic()
        try:
            return builder(*args)
        except ValueError as error:
            message = str(error)
            if (
                message != "prompt_payload_too_large"
                and re.fullmatch(
                    r"repair_request_(finding_ids|evidence|requested_changes)_(item|text)_limit",
                    message,
                )
                is None
            ):
                raise
            self._append_call_record(
                state,
                role,
                started,
                0,
                "prompt_payload_too_large",
                None,
            )
            raise PreparationFailure(
                category="prompt_payload_too_large",
                role=role,
                detail="备课内容超出可处理范围。",
            ) from None

    async def _complete_model(
        self,
        role: str,
        system: str,
        prompt: str,
        model_type: Type[_ModelType],
    ) -> _ModelType:
        state = self._current_state()
        started = time.monotonic()
        try:
            prompt = with_output_schema(prompt, model_type)
        except ValueError as error:
            if str(error) != "prompt_payload_too_large":
                raise
            self._append_call_record(
                state,
                role,
                started,
                0,
                "prompt_payload_too_large",
                None,
            )
            raise PreparationFailure(
                category="prompt_payload_too_large",
                role=role,
                detail="备课内容超出可处理范围。",
            ) from None
        retry_count = 0
        token_usage: Optional[Dict[str, int]] = {}
        structure_guidance = ""
        for attempt in range(self.MAX_STRUCTURE_ATTEMPTS):
            attempt_prompt = prompt
            if attempt:
                attempt_prompt = (
                    prompt
                    + "\n上一次输出结构无效。"
                    + "请仅返回符合 Schema 的 JSON 对象。"
                    + structure_guidance
                )
            try:
                structured_metadata_method = getattr(
                    self.client,
                    "complete_model_with_metadata",
                    None,
                )
                metadata_method = getattr(
                    self.client,
                    "complete_json_with_metadata",
                    None,
                )
                if callable(structured_metadata_method):
                    completion = await structured_metadata_method(
                        system,
                        attempt_prompt,
                        model_type,
                    )
                elif callable(metadata_method):
                    completion = await metadata_method(system, attempt_prompt)
                else:
                    completion = await self.client.complete_json(
                        system,
                        attempt_prompt,
                    )
                if isinstance(completion, ModelCompletion):
                    payload = completion.payload
                    token_usage = self._merge_token_usage(
                        token_usage,
                        completion.token_usage,
                    )
                else:
                    payload = completion
                model = model_type.model_validate(payload)
            except ModelStructureError as error:
                structure_guidance = _safe_structure_error_guidance(error)
                token_usage = self._merge_token_usage(
                    token_usage,
                    error.token_usage,
                )
                if attempt + 1 < self.MAX_STRUCTURE_ATTEMPTS:
                    retry_count += 1
                    continue
                self._append_call_record(
                    state,
                    role,
                    started,
                    retry_count,
                    "invalid_structure",
                    self._optional_token_usage(token_usage),
                )
                raise PreparationFailure(
                    category="invalid_structure",
                    role=role,
                    detail="模型输出结构无效。",
                ) from None
            except ValidationError as error:
                structure_guidance = _safe_validation_retry_guidance(
                    model_type,
                    error,
                )
                if attempt + 1 < self.MAX_STRUCTURE_ATTEMPTS:
                    retry_count += 1
                    continue
                self._append_call_record(
                    state,
                    role,
                    started,
                    retry_count,
                    "invalid_structure",
                    self._optional_token_usage(token_usage),
                )
                raise PreparationFailure(
                    category="invalid_structure",
                    role=role,
                    detail="模型输出结构无效。",
                ) from None
            except ModelResponseError:
                self._append_call_record(
                    state,
                    role,
                    started,
                    retry_count,
                    "provider_error",
                    self._optional_token_usage(token_usage),
                )
                raise PreparationFailure(
                    category="provider_error",
                    role=role,
                    detail="模型服务暂时不可用。",
                ) from None

            self._append_call_record(
                state,
                role,
                started,
                retry_count,
                None,
                self._optional_token_usage(token_usage),
            )
            return model
        raise AssertionError("unreachable model completion state")

    def _accept_artifact(
        self,
        state: PreparationState,
        artifact_type: str,
        responsible_role: str,
        artifact: BaseModel,
        finding_ids: Optional[List[str]] = None,
    ) -> None:
        if artifact_type not in {
            "solution_trace",
            "reasoning_trajectory",
            "teaching_progression",
            "teaching_script",
            "interaction_plan",
            "performance_score",
            "simulation_report",
        }:
            raise RuntimeError("unknown preparation artifact type")
        if not state.role_calls:
            raise RuntimeError("accepted artifact has no model call record")
        record = state.role_calls[-1]
        if record.role != responsible_role or record.failure_category is not None:
            raise RuntimeError("model call record does not match accepted artifact")
        try:
            self._reference_safety(state).ensure_safe(
                artifact,
                downstream_of_sanitized_trace=(
                    artifact_type != "solution_trace"
                ),
            )
        except ReferenceContentSafetyError:
            self._raise_reference_content_leak(state, responsible_role)
        version = state.versions.get(artifact_type, 0) + 1
        revision = ArtifactRevision(
            artifact_type=artifact_type,
            version=version,
            responsible_role=responsible_role,
            finding_ids=list(finding_ids or []),
        )
        updated_record_payload = record.model_dump(mode="python")
        updated_record_payload.update(
            {
                "output_artifact_type": artifact_type,
                "output_artifact_version": version,
                "review_finding_ids": list(finding_ids or []),
            }
        )
        updated_record = RoleCallRecord.model_validate(updated_record_payload)
        versions = dict(state.versions)
        versions[artifact_type] = version
        active_versions = dict(state.active_versions)
        active_versions[artifact_type] = version
        history = list(state.history)
        history.append(revision)
        role_calls = list(state.role_calls)
        role_calls[-1] = updated_record

        setattr(state, artifact_type, artifact)
        state.versions = versions
        state.active_versions = active_versions
        state.history = history
        state.role_calls = role_calls

    @staticmethod
    def _reference_safety(
        state: PreparationState,
    ) -> ReferenceSafetyPolicy:
        if state.reference_safety is None:
            raise RuntimeError("reference safety policy is missing")
        return state.reference_safety

    def _raise_reference_content_leak(
        self,
        state: PreparationState,
        role: str,
    ) -> None:
        self._mark_last_call_failed(
            state,
            role,
            "reference_content_leak",
        )
        raise PreparationFailure(
            category="reference_content_leak",
            role=role,
            detail="备课内容包含不安全的参考材料复述。",
        ) from None

    @staticmethod
    def _set_last_call_finding_ids(
        state: PreparationState,
        role: str,
        finding_ids: List[str],
    ) -> None:
        if not state.role_calls:
            raise RuntimeError("review has no model call record")
        record = state.role_calls[-1]
        if record.role != role or record.failure_category is not None:
            raise RuntimeError("model call record does not match review")
        payload = record.model_dump(mode="python")
        payload["review_finding_ids"] = list(finding_ids)
        role_calls = list(state.role_calls)
        role_calls[-1] = RoleCallRecord.model_validate(payload)
        state.role_calls = role_calls

    @staticmethod
    def _mark_last_call_failed(
        state: PreparationState,
        role: str,
        category: str,
    ) -> None:
        if not state.role_calls:
            raise RuntimeError("failed artifact has no model call record")
        record = state.role_calls[-1]
        if record.role != role or record.output_artifact_type is not None:
            raise RuntimeError("model call record does not match failed artifact")
        payload = record.model_dump(mode="python")
        payload["failure_category"] = category
        updated_record = RoleCallRecord.model_validate(payload)
        role_calls = list(state.role_calls)
        role_calls[-1] = updated_record
        state.role_calls = role_calls

    @staticmethod
    def _deactivate_invalidated_artifact(
        state: PreparationState,
        artifact_type: str,
        role: str,
        category: str,
    ) -> None:
        current_version = state.active_versions.get(artifact_type)
        if current_version is None:
            raise RuntimeError("rejected artifact is not active")
        matching = [
            index
            for index, record in enumerate(state.role_calls)
            if record.role == role
            and record.output_artifact_type == artifact_type
            and record.output_artifact_version == current_version
        ]
        if not matching:
            raise RuntimeError("rejected artifact has no model call record")
        record_index = matching[-1]
        payload = state.role_calls[record_index].model_dump(mode="python")
        payload["failure_category"] = category
        state.role_calls[record_index] = RoleCallRecord.model_validate(payload)
        state.active_versions.pop(artifact_type, None)
        setattr(state, artifact_type, None)

    @staticmethod
    def _append_call_record(
        state: PreparationState,
        role: str,
        started: float,
        retry_count: int,
        failure_category: Optional[str],
        token_usage: Optional[Dict[str, int]],
    ) -> None:
        duration_ms = max(0, int((time.monotonic() - started) * 1000))
        state.role_calls.append(
            RoleCallRecord(
                role=role,
                input_artifact_versions=dict(state.active_versions),
                duration_ms=duration_ms,
                retry_count=retry_count,
                failure_category=failure_category,
                token_usage=token_usage,
                review_finding_ids=[],
            )
        )

    @staticmethod
    def _merge_token_usage(
        accumulated: Optional[Dict[str, int]],
        supplied: Optional[Dict[str, int]],
    ) -> Optional[Dict[str, int]]:
        if accumulated is None:
            return None
        merged = dict(accumulated)
        if type(supplied) is not dict:
            return merged
        for key, value in supplied.items():
            if key not in ROLE_CALL_TOKEN_USAGE_KEYS:
                continue
            if type(value) is not int or value < 0:
                continue
            if value > MAX_ROLE_CALL_TOKEN_COUNTER:
                return None
            total = merged.get(key, 0) + value
            if total <= MAX_ROLE_CALL_TOKEN_COUNTER:
                merged[key] = total
            else:
                return None
        return merged

    @staticmethod
    def _optional_token_usage(
        usage: Optional[Dict[str, int]],
    ) -> Optional[Dict[str, int]]:
        return dict(usage) if usage else None

    def _require_active_state(self, state: PreparationState) -> None:
        if self._active_state.get() is not state:
            raise RuntimeError("preparation state is not active")

    def _current_state(self) -> PreparationState:
        state = self._active_state.get()
        if state is None:
            raise RuntimeError("preparation state is not active")
        return state

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
