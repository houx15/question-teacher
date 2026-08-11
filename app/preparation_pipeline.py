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
    Type,
    TypeVar,
    Union,
)

from pydantic import BaseModel, ValidationError

from app.llm_client import ModelResponseError
from app.preparation_models import (
    ArtifactRevision,
    InteractionPlan,
    LessonReviewDecision,
    PerformanceScore,
    PreparedLesson,
    ReasoningTrajectory,
    RoleCallRecord,
    SimulationReport,
    SolutionTrace,
    TeachingScript,
)
from app.preparation_prompts import (
    SOLUTION_TRACE_SYSTEM,
    TEACHING_DESIGNER_SYSTEM,
    reasoning_trajectory_prompt,
    solution_trace_prompt,
)
from app.preparation_validation import (
    PreparationValidationError,
    validate_reasoning_trajectory,
    validate_solution_trace,
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
        self.last_state: Optional[PreparationState] = None
        self._active_state: ContextVar[Optional[PreparationState]] = ContextVar(
            "active_preparation_state_%x" % id(self),
            default=None,
        )

    @property
    def role_calls(self) -> List[RoleCallRecord]:
        if self.last_state is None:
            return []
        return list(self.last_state.role_calls)

    async def prepare(
        self,
        problem: ProblemInput,
        teaching_route: FrozenTeachingRoute,
        problem_focus_targets: List[ProblemFocusTarget],
        on_stage: Optional[StageCallback] = None,
    ) -> PreparedLesson:
        state = PreparationState()
        self.last_state = state
        state_token = self._active_state.set(state)
        try:
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
            return await self._continue_preparation(
                state,
                problem,
                teaching_route,
                problem_focus_targets,
                on_stage,
            )
        finally:
            self._active_state.reset(state_token)

    async def _continue_preparation(
        self,
        state: PreparationState,
        problem: ProblemInput,
        teaching_route: FrozenTeachingRoute,
        problem_focus_targets: List[ProblemFocusTarget],
        on_stage: Optional[StageCallback],
    ) -> PreparedLesson:
        """Task 4 boundary; Tasks 5 and 6 replace this with downstream stages."""
        del state, problem, teaching_route, problem_focus_targets, on_stage
        raise NotImplementedError(
            "downstream preparation stages are not implemented in Task 4"
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
        token_usage = None
        for attempt in range(self.MAX_STRUCTURE_ATTEMPTS):
            attempt_prompt = prompt
            if attempt:
                attempt_prompt = (
                    prompt
                    + "\n上一次输出结构无效。"
                    + "请仅返回符合 Schema 的 JSON 对象。"
                )
            try:
                payload = await self.client.complete_json(system, attempt_prompt)
                token_usage = self._safe_token_usage()
                model = model_type.model_validate(payload)
            except ModelResponseError as error:
                token_usage = self._safe_token_usage()
                if self._is_structure_response_error(error):
                    if attempt + 1 < self.MAX_STRUCTURE_ATTEMPTS:
                        retry_count += 1
                        continue
                    self._append_call_record(
                        state,
                        role,
                        started,
                        retry_count,
                        "invalid_structure",
                        token_usage,
                    )
                    raise PreparationFailure(
                        category="invalid_structure",
                        role=role,
                        detail="模型输出结构无效。",
                    ) from None
                self._append_call_record(
                    state,
                    role,
                    started,
                    retry_count,
                    "provider_error",
                    token_usage,
                )
                raise PreparationFailure(
                    category="provider_error",
                    role=role,
                    detail="模型服务暂时不可用。",
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
                    token_usage,
                )
                raise PreparationFailure(
                    category="invalid_structure",
                    role=role,
                    detail="模型输出结构无效。",
                ) from None
            except Exception:
                self._append_call_record(
                    state,
                    role,
                    started,
                    retry_count,
                    "provider_error",
                    token_usage,
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
                token_usage,
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
        version = state.versions.get(artifact_type, 0) + 1
        setattr(state, artifact_type, artifact)
        state.versions[artifact_type] = version
        state.history.append(
            ArtifactRevision(
                artifact_type=artifact_type,
                version=version,
                responsible_role=responsible_role,
                finding_ids=[],
            )
        )
        record = state.role_calls[-1]
        if record.role != responsible_role or record.failure_category is not None:
            raise RuntimeError("model call record does not match accepted artifact")
        state.role_calls[-1] = record.model_copy(
            update={
                "output_artifact_type": artifact_type,
                "output_artifact_version": version,
            }
        )

    @staticmethod
    def _mark_last_call_failed(
        state: PreparationState,
        role: str,
        category: str,
    ) -> None:
        record = state.role_calls[-1]
        if record.role != role or record.output_artifact_type is not None:
            raise RuntimeError("model call record does not match failed artifact")
        state.role_calls[-1] = record.model_copy(
            update={"failure_category": category}
        )

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

    def _safe_token_usage(self) -> Optional[Dict[str, int]]:
        usage = getattr(self.client, "last_token_usage", None)
        if type(usage) is not dict:
            return None
        if any(
            type(key) is not str
            or type(value) is not int
            or value < 0
            for key, value in usage.items()
        ):
            return None
        return dict(usage)

    @staticmethod
    def _is_structure_response_error(error: ModelResponseError) -> bool:
        return str(error).startswith("Model response content")

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
