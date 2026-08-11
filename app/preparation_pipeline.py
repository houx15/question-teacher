"""Deterministic orchestration for the private lesson-preparation chain."""

import inspect
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
    ROLE_CALL_TOKEN_USAGE_KEYS,
    ReasoningTrajectory,
    RoleCallRecord,
    SimulationReport,
    SolutionTrace,
    TeachingScript,
)
from app.preparation_prompts import (
    CLASSROOM_DIRECTOR_SYSTEM,
    INTERACTION_DESIGNER_SYSTEM,
    SCRIPT_TEACHER_SYSTEM,
    SOLUTION_TRACE_SYSTEM,
    TEACHING_DESIGNER_SYSTEM,
    interaction_plan_prompt,
    performance_score_prompt,
    reasoning_trajectory_prompt,
    solution_trace_prompt,
    teaching_script_prompt,
)
from app.preparation_validation import (
    PreparationValidationError,
    validate_interaction_plan,
    validate_performance_score,
    validate_reasoning_trajectory,
    validate_solution_trace,
    validate_teaching_script,
)
from app.schemas import ProblemFocusTarget, ProblemInput
from app.teaching_route import FrozenTeachingRoute


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
    solution_trace: Optional[SolutionTrace] = None
    reasoning_trajectory: Optional[ReasoningTrajectory] = None
    teaching_script: Optional[TeachingScript] = None
    interaction_plan: Optional[InteractionPlan] = None
    performance_score: Optional[PerformanceScore] = None
    simulation_report: Optional[SimulationReport] = None
    review: Optional[LessonReviewDecision] = None
    versions: Dict[str, int] = field(default_factory=dict)
    history: List[ArtifactRevision] = field(default_factory=list)
    role_calls: List[RoleCallRecord] = field(default_factory=list)


@dataclass(frozen=True)
class PreparationAuditSnapshot:
    """Content-free, immutable storage for one request's audit metadata."""

    _version_items: Tuple[Tuple[str, int], ...]
    _history_json: Tuple[str, ...]
    _role_call_json: Tuple[str, ...]

    @classmethod
    def from_state(
        cls,
        state: PreparationState,
    ) -> "PreparationAuditSnapshot":
        return cls(
            _version_items=tuple(sorted(state.versions.items())),
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
        state = PreparationState()
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
        state = PreparationState()
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
        """Run Task 5 artifacts, retaining an honest Task 6 boundary."""
        del problem, teaching_route
        await self._create_teaching_script(state, on_stage)
        await self._create_interaction_plan(state, on_stage)
        await self._create_performance_score(
            state,
            problem_focus_targets,
            on_stage,
        )
        raise NotImplementedError(
            "simulation and review stages are not implemented in Task 5"
        )

    async def _create_solution_trace(
        self,
        state: PreparationState,
        problem: ProblemInput,
        teaching_route: FrozenTeachingRoute,
        problem_focus_targets: List[ProblemFocusTarget],
        on_stage: Optional[StageCallback],
    ) -> SolutionTrace:
        self._require_active_state(state)
        await self._emit(on_stage, "整理参考解析")
        prompt = solution_trace_prompt(
            problem,
            teaching_route,
            problem_focus_targets,
        )
        trace = await self._complete_model(
            "reference_analyst",
            SOLUTION_TRACE_SYSTEM,
            prompt,
            SolutionTrace,
        )
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
        )
        return trace

    async def _create_reasoning_trajectory(
        self,
        state: PreparationState,
        problem: ProblemInput,
        on_stage: Optional[StageCallback],
    ) -> ReasoningTrajectory:
        self._require_active_state(state)
        if state.solution_trace is None:
            raise RuntimeError("solution trace must exist before trajectory design")
        await self._emit(on_stage, "设计解题思维轨迹")
        prompt = reasoning_trajectory_prompt(
            problem,
            state.solution_trace,
            self.capabilities,
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
        )
        return trajectory

    async def _create_teaching_script(
        self,
        state: PreparationState,
        on_stage: Optional[StageCallback],
    ) -> TeachingScript:
        self._require_active_state(state)
        if state.reasoning_trajectory is None:
            raise RuntimeError(
                "reasoning trajectory must exist before script writing"
            )
        await self._emit(on_stage, "编写讲稿")
        script = await self._complete_model(
            "script_teacher",
            SCRIPT_TEACHER_SYSTEM,
            teaching_script_prompt(state.reasoning_trajectory),
            TeachingScript,
        )
        try:
            validate_teaching_script(script, state.reasoning_trajectory)
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
        )
        return script

    async def _create_interaction_plan(
        self,
        state: PreparationState,
        on_stage: Optional[StageCallback],
    ) -> InteractionPlan:
        self._require_active_state(state)
        if state.reasoning_trajectory is None or state.teaching_script is None:
            raise RuntimeError(
                "trajectory and script must exist before interaction design"
            )
        await self._emit(on_stage, "设计互动")
        plan = await self._complete_model(
            "interaction_designer",
            INTERACTION_DESIGNER_SYSTEM,
            interaction_plan_prompt(
                state.reasoning_trajectory,
                state.teaching_script,
            ),
            InteractionPlan,
        )
        try:
            validate_interaction_plan(
                plan,
                state.reasoning_trajectory,
                state.teaching_script,
            )
        except PreparationValidationError:
            self._mark_last_call_failed(
                state,
                "interaction_designer",
                "interaction_plan_failed",
            )
            raise PreparationFailure(
                category="interaction_plan_failed",
                role="interaction_designer",
                detail="互动方案未通过确定性校验。",
            ) from None
        self._accept_artifact(
            state,
            artifact_type="interaction_plan",
            responsible_role="interaction_designer",
            artifact=plan,
        )
        return plan

    async def _create_performance_score(
        self,
        state: PreparationState,
        problem_focus_targets: List[ProblemFocusTarget],
        on_stage: Optional[StageCallback],
    ) -> PerformanceScore:
        self._require_active_state(state)
        if state.teaching_script is None or state.interaction_plan is None:
            raise RuntimeError(
                "script and interaction plan must exist before direction"
            )
        await self._emit(on_stage, "编排板书与高亮")
        score = await self._complete_model(
            "classroom_director",
            CLASSROOM_DIRECTOR_SYSTEM,
            performance_score_prompt(
                problem_focus_targets,
                state.teaching_script,
                state.interaction_plan,
                self.capabilities,
            ),
            PerformanceScore,
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
        )
        return score

    async def _complete_model(
        self,
        role: str,
        system: str,
        prompt: str,
        model_type: Type[_ModelType],
    ) -> _ModelType:
        state = self._current_state()
        started = time.monotonic()
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
                metadata_method = getattr(
                    self.client,
                    "complete_json_with_metadata",
                    None,
                )
                if callable(metadata_method):
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
    ) -> None:
        if artifact_type not in {
            "solution_trace",
            "reasoning_trajectory",
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
        version = state.versions.get(artifact_type, 0) + 1
        revision = ArtifactRevision(
            artifact_type=artifact_type,
            version=version,
            responsible_role=responsible_role,
            finding_ids=[],
        )
        updated_record_payload = record.model_dump(mode="python")
        updated_record_payload.update(
            {
                "output_artifact_type": artifact_type,
                "output_artifact_version": version,
            }
        )
        updated_record = RoleCallRecord.model_validate(updated_record_payload)
        versions = dict(state.versions)
        versions[artifact_type] = version
        history = list(state.history)
        history.append(revision)
        role_calls = list(state.role_calls)
        role_calls[-1] = updated_record

        setattr(state, artifact_type, artifact)
        state.versions = versions
        state.history = history
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
                input_artifact_versions=dict(state.versions),
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
