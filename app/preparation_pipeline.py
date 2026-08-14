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
) -> TeachingScript:
    """Bind fixed script sections to boundary episodes, not invented ones."""
    first_episode_id = trajectory.episodes[0].episode_id
    last_episode_id = trajectory.episodes[-1].episode_id
    must_teach_owner = {
        item.must_teach_id: episode.episode_id
        for episode in trajectory.episodes
        for item in episode.must_teach
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
    MAX_STRUCTURE_ATTEMPTS = 2
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
        prompt = self._build_prompt(
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
            prompt,
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
        prompt = self._build_prompt(
            state,
            "teaching_designer",
            reasoning_trajectory_prompt,
            problem,
            state.solution_trace,
            self.capabilities,
            repair,
        )
        trajectory = await self._complete_model(
            "teaching_designer",
            TEACHING_DESIGNER_SYSTEM,
            prompt,
            ReasoningTrajectory,
        )
        try:
            validate_reasoning_trajectory(trajectory, state.solution_trace)
        except PreparationValidationError:
            self._mark_last_call_failed(
                state,
                "teaching_designer",
                "reasoning_design_failed",
            )
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
        progression = await self._complete_model(
            "teaching_designer",
            TEACHING_PROGRESSION_SYSTEM,
            self._build_prompt(
                state,
                "teaching_designer",
                teaching_progression_prompt,
                state.reasoning_trajectory,
                problem_focus_targets,
                repair,
            ),
            TeachingProgression,
        )
        try:
            validate_teaching_progression(
                progression,
                state.reasoning_trajectory,
                problem_focus_targets,
            )
        except TeachingProgressionValidationError:
            self._mark_last_call_failed(
                state,
                "teaching_designer",
                "teaching_progression_failed",
            )
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
        script = await self._complete_model(
            "script_teacher",
            SCRIPT_TEACHER_SYSTEM,
            self._build_prompt(
                state,
                "script_teacher",
                teaching_script_prompt,
                state.teaching_progression,
                state.interaction_plan,
                repair,
            ),
            TeachingScript,
        )
        script = _normalize_script_section_metadata(
            script, state.reasoning_trajectory
        )
        try:
            validate_teaching_script(
                script,
                state.reasoning_trajectory,
                state.teaching_progression,
                state.interaction_plan,
            )
        except PreparationValidationError:
            self._mark_last_call_failed(
                state,
                "script_teacher",
                "teaching_script_failed",
            )
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
        prompt = self._build_prompt(
            state,
            "interaction_designer",
            interaction_plan_prompt,
            state.teaching_progression,
            repair,
        )
        plan = await self._complete_model(
            "interaction_designer",
            INTERACTION_DESIGNER_SYSTEM,
            prompt,
            InteractionPlan,
        )
        try:
            validate_interaction_plan(plan, state.teaching_progression)
        except PreparationValidationError:
            self._mark_last_call_failed(
                state,
                "interaction_designer",
                "interaction_plan_failed",
            )
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
        await self._emit(on_stage, "编排板书与高亮")
        prompt = self._build_prompt(
            state,
            "classroom_director",
            performance_score_prompt,
            problem_focus_targets,
            state.teaching_script,
            state.interaction_plan,
            self.capabilities,
            repair,
        )
        score = await self._complete_model(
            "classroom_director",
            CLASSROOM_DIRECTOR_SYSTEM,
            prompt,
            PerformanceScore,
        )
        try:
            validate_performance_score(
                score,
                problem_focus_targets,
                state.teaching_script,
                state.interaction_plan,
            )
        except PreparationValidationError as initial_error:
            if initial_error.code not in {
                "overlay_transition_invalid",
                "visual_action_too_early",
            }:
                self._mark_last_call_failed(
                    state,
                    "classroom_director",
                    "performance_score_failed",
                )
                raise PreparationFailure(
                    category="performance_score_failed",
                    role="classroom_director",
                    detail="板书与高亮编排未通过确定性校验。",
                ) from None
            score = normalize_performance_control_metadata(
                score,
                problem_focus_targets,
                state.teaching_script,
            )
            try:
                validate_performance_score(
                    score,
                    problem_focus_targets,
                    state.teaching_script,
                    state.interaction_plan,
                )
            except PreparationValidationError:
                self._mark_last_call_failed(
                    state,
                    "classroom_director",
                    "performance_score_failed",
                )
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
        for attempt in range(self.MAX_STRUCTURE_ATTEMPTS):
            attempt_prompt = prompt
            if attempt:
                attempt_prompt = (
                    prompt
                    + "\n上一次输出结构无效。"
                    + "请仅返回符合 Schema 的 JSON 对象。"
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
            except ValidationError:
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
