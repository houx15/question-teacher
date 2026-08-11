from typing import Dict, List, Literal, Optional

from pydantic import Field, model_validator

from app.schemas import (
    CueSpokenText,
    GeneratedId,
    GeneratedMathAnswer,
    GeneratedTransferItem,
    MethodIntroduction,
    NarrativeBoardContent,
    NonEmptyString,
    SchemaModel,
    SyncVisualAction,
)


EvidenceStatus = Literal["quoted", "derived", "inferred", "verified_route"]
ReasoningMode = Literal["understand", "plan", "explore", "execute", "monitor", "revise", "reflect"]
TrajectoryType = Literal["planned", "exploratory", "hybrid"]
PedagogicalFunction = Literal["focus", "question", "explain", "decide", "execute", "observe", "correct", "transition", "review", "summarize"]
DiagnosticKind = Literal["conception", "execution"]
ArtifactType = Literal["solution_trace", "reasoning_trajectory", "teaching_script", "interaction_plan", "performance_score", "simulation_report"]
ResponsibleRole = Literal["reference_analyst", "teaching_designer", "script_teacher", "interaction_designer", "classroom_director"]
RoleName = Literal["reference_analyst", "teaching_designer", "script_teacher", "interaction_designer", "classroom_director", "student_simulator", "lesson_reviewer"]


def _require_unique(values: List[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError("%s must be unique" % label)


class SourceAnchor(SchemaModel):
    source_kind: Literal["problem", "answer", "solution", "verified_route"]
    source_id: GeneratedId
    excerpt: NonEmptyString


class SolutionTraceStep(SchemaModel):
    source_step_id: GeneratedId
    source_anchor: SourceAnchor
    state_before: NonEmptyString
    mathematical_action: NonEmptyString
    justification: NonEmptyString
    state_after: NonEmptyString
    new_information: NonEmptyString
    assumption_ids_used: List[GeneratedId] = Field(default_factory=list)
    omitted_reasoning: List[NonEmptyString] = Field(default_factory=list)
    evidence_status: EvidenceStatus


class TraceAssumption(SchemaModel):
    assumption_id: GeneratedId
    content: NonEmptyString
    source_anchor: SourceAnchor


class SolutionTrace(SchemaModel):
    task_target: NonEmptyString
    reference_conclusion: NonEmptyString
    assumptions: List[TraceAssumption] = Field(default_factory=list)
    source_steps: List[SolutionTraceStep] = Field(min_length=1)
    audit_notes: List[NonEmptyString] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_local_ids(self) -> "SolutionTrace":
        _require_unique([item.assumption_id for item in self.assumptions], "assumption ids")
        _require_unique([item.source_step_id for item in self.source_steps], "trace step ids")
        return self


class MustTeachItem(SchemaModel):
    must_teach_id: GeneratedId
    content: NonEmptyString
    why_it_matters: NonEmptyString


class ReasoningEpisode(SchemaModel):
    episode_id: GeneratedId
    sequence_index: int = Field(ge=0)
    mode: ReasoningMode
    source_step_ids: List[GeneratedId] = Field(min_length=1)
    learner_state_before: NonEmptyString
    attention_targets: List[NonEmptyString] = Field(min_length=1)
    thinking_question: NonEmptyString
    decision: NonEmptyString
    decision_reason: NonEmptyString
    mathematical_action: NonEmptyString
    action_justification: NonEmptyString
    result: NonEmptyString
    result_meaning: NonEmptyString
    transition_reason: NonEmptyString
    must_teach: List[MustTeachItem] = Field(min_length=1)
    likely_misconceptions: List[NonEmptyString] = Field(default_factory=list)
    interaction_intent: Optional[NonEmptyString] = None
    visual_intent: Optional[NonEmptyString] = None

    @model_validator(mode="after")
    def validate_must_teach_ids(self) -> "ReasoningEpisode":
        _require_unique([item.must_teach_id for item in self.must_teach], "must-teach ids")
        return self


class ReasoningTrajectory(SchemaModel):
    trajectory_type: TrajectoryType
    lesson_purpose: NonEmptyString
    episodes: List[ReasoningEpisode] = Field(min_length=1)
    method_summary: NonEmptyString
    error_summary: NonEmptyString

    @model_validator(mode="after")
    def validate_episodes(self) -> "ReasoningTrajectory":
        _require_unique([item.episode_id for item in self.episodes], "episode ids")
        if [item.sequence_index for item in self.episodes] != list(range(len(self.episodes))):
            raise ValueError("episode sequence indexes must be contiguous starting at zero")
        return self


class ScriptClause(SchemaModel):
    clause_id: GeneratedId
    episode_id: GeneratedId
    pedagogical_function: PedagogicalFunction
    spoken_text: CueSpokenText
    math_references: List[GeneratedMathAnswer] = Field(default_factory=list)
    learner_gain: NonEmptyString
    answer_exposure: bool
    must_teach_refs: List[GeneratedId] = Field(default_factory=list)


class TeachingScript(SchemaModel):
    title: NonEmptyString
    learning_goal: NonEmptyString
    method_rationale: NonEmptyString
    method_introduction: MethodIntroduction
    opening_clause_ids: List[GeneratedId] = Field(min_length=1)
    method_introduction_clause_ids: List[GeneratedId] = Field(min_length=1)
    clauses: List[ScriptClause] = Field(min_length=1)
    closing_summary_clause_ids: List[GeneratedId] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_clause_sections(self) -> "TeachingScript":
        clause_ids = [item.clause_id for item in self.clauses]
        _require_unique(clause_ids, "clause ids")
        sections = (self.opening_clause_ids, self.method_introduction_clause_ids, self.closing_summary_clause_ids)
        flattened = [item for section in sections for item in section]
        if any(item not in clause_ids for item in flattened):
            raise ValueError("script section clause ids must exist in clauses")
        if len(flattened) != len(set(flattened)):
            raise ValueError("script sections must not overlap")
        positions = [clause_ids.index(item) for item in flattened]
        if positions != sorted(positions):
            raise ValueError("script sections must retain script order")
        return self


class PlannedInteractionOption(SchemaModel):
    option_id: GeneratedId
    display_text: NonEmptyString
    canonical_answer: NonEmptyString
    misconception: Optional[NonEmptyString] = None


class PlannedInteraction(SchemaModel):
    interaction_id: GeneratedId
    episode_id: GeneratedId
    after_clause_id: GeneratedId
    diagnostic_target: NonEmptyString
    diagnostic_kind: DiagnosticKind
    prompt: NonEmptyString
    options: List[PlannedInteractionOption] = Field(min_length=3, max_length=4)
    correct_option_id: GeneratedId
    correct_feedback: NonEmptyString
    incorrect_feedback_by_option: Dict[GeneratedId, NonEmptyString]
    hint: NonEmptyString
    resume_clause_id: GeneratedId
    concealed_targets: List[GeneratedId] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_options(self) -> "PlannedInteraction":
        option_ids = [item.option_id for item in self.options]
        _require_unique(option_ids, "interaction option ids")
        if self.correct_option_id not in option_ids:
            raise ValueError("correct_option_id must match an interaction option_id")
        incorrect_ids = set(option_ids) - {self.correct_option_id}
        if set(self.incorrect_feedback_by_option) != incorrect_ids:
            raise ValueError("incorrect feedback must cover exactly all incorrect option ids")
        return self


class InteractionPlan(SchemaModel):
    interactions: List[PlannedInteraction] = Field(default_factory=list, max_length=3)
    transfer_item: GeneratedTransferItem

    @model_validator(mode="after")
    def validate_interaction_ids(self) -> "InteractionPlan":
        _require_unique([item.interaction_id for item in self.interactions], "interaction ids")
        return self


class ClauseBoundVisualAction(SchemaModel):
    clause_id: GeneratedId
    action: SyncVisualAction


class PerformanceCue(SchemaModel):
    cue_id: GeneratedId
    clause_ids: List[GeneratedId] = Field(min_length=1)
    lead_actions: List[ClauseBoundVisualAction] = Field(default_factory=list)
    start_actions: List[ClauseBoundVisualAction] = Field(default_factory=list)
    end_actions: List[ClauseBoundVisualAction] = Field(default_factory=list)


class PerformanceBoardObject(SchemaModel):
    board_object_id: GeneratedId
    content: NarrativeBoardContent
    layer: Literal["base", "micro_explanation", "comparison"] = "base"


class OverlayTransition(SchemaModel):
    transition_id: GeneratedId
    after_clause_id: GeneratedId
    action: Literal["enter", "return"]
    layer: Literal["micro_explanation", "comparison"]


class PerformanceScore(SchemaModel):
    cues: List[PerformanceCue] = Field(min_length=1)
    board_objects: List[PerformanceBoardObject] = Field(default_factory=list)
    overlay_transitions: List[OverlayTransition] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_local_ids(self) -> "PerformanceScore":
        _require_unique([item.cue_id for item in self.cues], "cue ids")
        _require_unique([item.board_object_id for item in self.board_objects], "board object ids")
        _require_unique([item.transition_id for item in self.overlay_transitions], "overlay transition ids")
        return self


class EpisodeSimulationResult(SchemaModel):
    episode_id: GeneratedId
    learner_profile: NonEmptyString
    can_identify_attention_target: bool
    can_explain_decision: bool
    can_execute_action: bool
    can_use_result_to_continue: bool
    evidence: List[NonEmptyString] = Field(min_length=1)


class SimulationReport(SchemaModel):
    episode_results: List[EpisodeSimulationResult] = Field(min_length=1)
    interaction_results: List[NonEmptyString] = Field(default_factory=list)
    end_of_lesson_recall: NonEmptyString
    blocking_findings: List[NonEmptyString] = Field(default_factory=list)


class ReviewFinding(SchemaModel):
    finding_id: GeneratedId
    severity: Literal["blocking", "material", "polish"]
    artifact_type: ArtifactType
    artifact_id: GeneratedId
    criterion: NonEmptyString
    evidence: NonEmptyString
    responsible_role: ResponsibleRole
    requested_change: NonEmptyString
    invalidated_downstream_artifacts: List[ArtifactType] = Field(default_factory=list)


class LessonReviewDecision(SchemaModel):
    status: Literal["approved", "revision_required", "failed"]
    findings: List[ReviewFinding] = Field(default_factory=list)
    retained_artifacts: List[ArtifactType] = Field(default_factory=list)
    approval_summary: NonEmptyString

    @model_validator(mode="after")
    def validate_decision(self) -> "LessonReviewDecision":
        _require_unique([item.finding_id for item in self.findings], "finding ids")
        blocking_or_material = any(item.severity in {"blocking", "material"} for item in self.findings)
        if self.status == "approved" and blocking_or_material:
            raise ValueError("approved review cannot contain blocking or material findings")
        if self.status in {"revision_required", "failed"} and not blocking_or_material:
            raise ValueError("revision_required or failed review requires a blocking or material finding")
        return self


class ArtifactRevision(SchemaModel):
    artifact_type: ArtifactType
    version: int = Field(ge=1)
    responsible_role: ResponsibleRole
    finding_ids: List[GeneratedId] = Field(default_factory=list)


class PreparedLesson(SchemaModel):
    rubric_version: NonEmptyString
    solution_trace: SolutionTrace
    reasoning_trajectory: ReasoningTrajectory
    teaching_script: TeachingScript
    interaction_plan: InteractionPlan
    performance_score: PerformanceScore
    simulation_report: SimulationReport
    review: LessonReviewDecision
    repair_count: int = Field(ge=0)
    artifact_history: List[ArtifactRevision] = Field(min_length=5)


class RoleCallRecord(SchemaModel):
    role: RoleName
    input_artifact_versions: Dict[ArtifactType, int] = Field(default_factory=dict)
    output_artifact_type: Optional[ArtifactType] = None
    output_artifact_version: Optional[int] = Field(default=None, ge=1)
    duration_ms: int = Field(ge=0)
    retry_count: int = Field(ge=0)
    failure_category: Optional[NonEmptyString] = None
    token_usage: Optional[Dict[str, int]] = None
    review_finding_ids: List[GeneratedId] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_output_and_tokens(self) -> "RoleCallRecord":
        if (self.output_artifact_type is None) != (self.output_artifact_version is None):
            raise ValueError("output type and version must be present or absent together")
        if self.token_usage is not None and any(value < 0 for value in self.token_usage.values()):
            raise ValueError("token usage values cannot be negative")
        return self


class GenerationRecord(SchemaModel):
    generation_id: NonEmptyString
    lesson_id: NonEmptyString
    route_fingerprint: NonEmptyString
    prepared_lesson: PreparedLesson
    role_calls: List[RoleCallRecord] = Field(min_length=7)
    created_at: NonEmptyString
