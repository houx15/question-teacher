from typing import Annotated, Dict, List, Literal, Optional

from pydantic import (
    Field,
    PositiveInt,
    StringConstraints,
    field_validator,
    model_validator,
)

from app.math_expression import (
    MathOperationKind,
    ReasoningGapCode,
    StrictMathExpression,
    validate_operation_operands,
)
from app.pedagogy_rubric import ReviewCriterionId
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


EvidenceStatus = Literal[
    "quoted",
    "derived",
    "inferred",
    "verified_route",
    "reference_only",
    "checked",
    "check_warning",
]
ReasoningMode = Literal["understand", "plan", "explore", "execute", "monitor", "revise", "reflect"]
TrajectoryType = Literal["planned", "exploratory", "hybrid"]
PedagogicalFunction = Literal["focus", "question", "explain", "decide", "execute", "observe", "correct", "transition", "review", "summarize"]
DiagnosticKind = Literal["conception", "execution"]
ArtifactType = Literal["solution_trace", "reasoning_trajectory", "teaching_script", "interaction_plan", "performance_score", "simulation_report"]
ResponsibleRole = Literal["reference_analyst", "teaching_designer", "script_teacher", "interaction_designer", "classroom_director"]
RoleName = Literal["reference_analyst", "teaching_designer", "script_teacher", "interaction_designer", "classroom_director", "student_simulator", "lesson_reviewer"]
ArtifactResponsibleRole = Literal["reference_analyst", "teaching_designer", "script_teacher", "interaction_designer", "classroom_director", "student_simulator"]
ROLE_CALL_TOKEN_USAGE_KEYS = frozenset(
    {
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_tokens",
        "reasoning_tokens",
    }
)
MAX_ROLE_CALL_TOKEN_COUNTER = 1_000_000_000
MAX_PREPARATION_ITEMS = 256
MAX_DETAIL_ITEMS = 64
MAX_MATH_REFERENCE_ITEMS = 2048
MAX_VISUAL_ACTION_ITEMS = 2048
MAX_ARTIFACT_HISTORY_ITEMS = 64
MAX_SIMULATION_EVIDENCE_ITEMS = 16
MAX_REVIEW_ARTIFACT_ITEMS = 6
MAX_SIMULATION_SERIALIZED_BYTES = 128 * 1024
MAX_REVIEW_SERIALIZED_BYTES = 128 * 1024
LearnerProfileText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=120),
]
BoundedReviewText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=800),
]


def _require_unique(values: List[str], label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError("%s must be unique" % label)


class SourceAnchor(SchemaModel):
    source_kind: Literal[
        "problem",
        "problem_derived",
        "answer",
        "solution",
        "verified_route",
    ]
    source_id: GeneratedId
    excerpt: NonEmptyString


class SolutionTraceStep(SchemaModel):
    source_step_id: GeneratedId
    source_anchor: SourceAnchor
    state_before: StrictMathExpression
    operation_kind: MathOperationKind
    operands: List[StrictMathExpression] = Field(
        default_factory=list,
        max_length=8,
    )
    mathematical_action: NonEmptyString
    justification: NonEmptyString
    state_after: StrictMathExpression
    new_information: NonEmptyString
    assumption_ids_used: List[GeneratedId] = Field(
        default_factory=list, max_length=MAX_DETAIL_ITEMS
    )
    reasoning_gap_codes: List[ReasoningGapCode] = Field(
        default_factory=list, max_length=MAX_DETAIL_ITEMS
    )
    evidence_status: EvidenceStatus

    @model_validator(mode="after")
    def validate_operand_arity(self) -> "SolutionTraceStep":
        validate_operation_operands(self.operation_kind, self.operands)
        return self


class TraceAssumption(SchemaModel):
    assumption_id: GeneratedId
    content: StrictMathExpression
    source_anchor: SourceAnchor


class SolutionTrace(SchemaModel):
    task_target: StrictMathExpression
    reference_conclusion: StrictMathExpression
    assumptions: List[TraceAssumption] = Field(
        default_factory=list, max_length=MAX_PREPARATION_ITEMS
    )
    source_steps: List[SolutionTraceStep] = Field(
        min_length=1, max_length=MAX_PREPARATION_ITEMS
    )
    audit_notes: List[NonEmptyString] = Field(
        default_factory=list, max_length=MAX_DETAIL_ITEMS
    )

    @model_validator(mode="after")
    def validate_local_ids(self) -> "SolutionTrace":
        _require_unique([item.assumption_id for item in self.assumptions], "assumption ids")
        _require_unique([item.source_step_id for item in self.source_steps], "trace step ids")
        return self


class MustTeachItem(SchemaModel):
    must_teach_id: GeneratedId
    content: NonEmptyString
    why_it_matters: NonEmptyString


class ResolvedReasoningGap(SchemaModel):
    source_step_id: GeneratedId
    gap_code: ReasoningGapCode
    must_teach_id: GeneratedId


class ReasoningEpisode(SchemaModel):
    episode_id: GeneratedId
    sequence_index: int = Field(ge=0)
    mode: ReasoningMode
    source_step_ids: List[GeneratedId] = Field(
        min_length=1, max_length=MAX_PREPARATION_ITEMS
    )
    learner_state_before: NonEmptyString
    attention_targets: List[NonEmptyString] = Field(
        min_length=1, max_length=MAX_DETAIL_ITEMS
    )
    thinking_question: NonEmptyString
    decision: NonEmptyString
    decision_reason: NonEmptyString
    mathematical_action: NonEmptyString
    action_justification: NonEmptyString
    result: NonEmptyString
    result_meaning: NonEmptyString
    transition_reason: NonEmptyString
    must_teach: List[MustTeachItem] = Field(
        min_length=1, max_length=MAX_DETAIL_ITEMS
    )
    resolved_gap_refs: List[ResolvedReasoningGap] = Field(
        default_factory=list,
        max_length=MAX_DETAIL_ITEMS,
    )
    likely_misconceptions: List[NonEmptyString] = Field(
        default_factory=list, max_length=MAX_DETAIL_ITEMS
    )
    interaction_intent: Optional[NonEmptyString] = None
    visual_intent: Optional[NonEmptyString] = None

    @model_validator(mode="after")
    def validate_must_teach_ids(self) -> "ReasoningEpisode":
        _require_unique([item.must_teach_id for item in self.must_teach], "must-teach ids")
        _require_unique(
            [
                "%s:%s" % (item.source_step_id, item.gap_code)
                for item in self.resolved_gap_refs
            ],
            "resolved gap refs",
        )
        return self


class ReasoningTrajectory(SchemaModel):
    trajectory_type: TrajectoryType
    lesson_purpose: NonEmptyString
    episodes: List[ReasoningEpisode] = Field(
        min_length=1, max_length=MAX_PREPARATION_ITEMS
    )
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
    math_references: List[GeneratedMathAnswer] = Field(
        default_factory=list, max_length=MAX_MATH_REFERENCE_ITEMS
    )
    learner_gain: NonEmptyString
    answer_exposure: bool
    must_teach_refs: List[GeneratedId] = Field(
        default_factory=list, max_length=MAX_DETAIL_ITEMS
    )


class TeachingScript(SchemaModel):
    title: NonEmptyString
    learning_goal: NonEmptyString
    method_rationale: NonEmptyString
    method_introduction: MethodIntroduction
    opening_clause_ids: List[GeneratedId] = Field(
        min_length=1, max_length=MAX_PREPARATION_ITEMS
    )
    method_introduction_clause_ids: List[GeneratedId] = Field(
        min_length=1, max_length=MAX_PREPARATION_ITEMS
    )
    clauses: List[ScriptClause] = Field(
        min_length=1, max_length=MAX_PREPARATION_ITEMS
    )
    closing_summary_clause_ids: List[GeneratedId] = Field(
        min_length=1, max_length=MAX_PREPARATION_ITEMS
    )

    @model_validator(mode="after")
    def validate_clause_sections(self) -> "TeachingScript":
        clause_ids = [item.clause_id for item in self.clauses]
        _require_unique(clause_ids, "clause ids")
        clause_positions = {
            clause_id: index for index, clause_id in enumerate(clause_ids)
        }
        sections = (self.opening_clause_ids, self.method_introduction_clause_ids, self.closing_summary_clause_ids)
        flattened = [item for section in sections for item in section]
        if any(item not in clause_ids for item in flattened):
            raise ValueError("script section clause ids must exist in clauses")
        if len(flattened) != len(set(flattened)):
            raise ValueError("script sections must not overlap")
        positions = [clause_positions[item] for item in flattened]
        if positions != sorted(positions):
            raise ValueError("script sections must retain script order")
        opening_count = len(self.opening_clause_ids)
        method_count = len(self.method_introduction_clause_ids)
        if clause_ids[:opening_count] != self.opening_clause_ids:
            raise ValueError("opening section must be the exact opening prefix")
        method_start = opening_count
        method_end = method_start + method_count
        if clause_ids[method_start:method_end] != self.method_introduction_clause_ids:
            raise ValueError("method-introduction section must immediately follow opening")
        if clause_ids[-len(self.closing_summary_clause_ids):] != self.closing_summary_clause_ids:
            raise ValueError("closing summary section must be the exact closing suffix")
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
    concealed_targets: List[GeneratedId] = Field(
        default_factory=list,
        max_length=MAX_DETAIL_ITEMS,
        description=(
            "互动前必须隐藏的同 episode 语义目标 ID；"
            "resume_clause_id 不得放入 concealed_targets。"
        ),
    )

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
    clause_ids: List[GeneratedId] = Field(
        min_length=1, max_length=MAX_PREPARATION_ITEMS
    )
    lead_actions: List[ClauseBoundVisualAction] = Field(
        default_factory=list,
        max_length=MAX_VISUAL_ACTION_ITEMS,
        description="仅允许 focus 或 emphasize，在口播前引导注意。",
    )
    start_actions: List[ClauseBoundVisualAction] = Field(
        default_factory=list,
        max_length=MAX_VISUAL_ACTION_ITEMS,
        description=(
            "仅允许 write、transform、focus、emphasize、annotate 或 reveal；"
            "write/transform content 必须精确对应绑定子句的 math_references。"
        ),
    )
    end_actions: List[ClauseBoundVisualAction] = Field(
        default_factory=list,
        max_length=MAX_VISUAL_ACTION_ITEMS,
        description="仅允许 clear_focus 或 fade，用于弱化当前划过的重点。",
    )


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
    cues: List[PerformanceCue] = Field(
        min_length=1, max_length=MAX_PREPARATION_ITEMS
    )
    board_objects: List[PerformanceBoardObject] = Field(
        default_factory=list, max_length=MAX_PREPARATION_ITEMS
    )
    overlay_transitions: List[OverlayTransition] = Field(
        default_factory=list,
        max_length=MAX_PREPARATION_ITEMS,
        description=(
            "enter 和 return 必须在不同 cue 边界，中间至少有一个 cue；"
            "无真正新图层时使用空列表。"
        ),
    )

    @model_validator(mode="after")
    def validate_local_ids(self) -> "PerformanceScore":
        _require_unique([item.cue_id for item in self.cues], "cue ids")
        _require_unique([item.board_object_id for item in self.board_objects], "board object ids")
        _require_unique([item.transition_id for item in self.overlay_transitions], "overlay transition ids")
        return self


class EpisodeSimulationResult(SchemaModel):
    episode_id: GeneratedId
    learner_profile: LearnerProfileText
    can_identify_attention_target: bool
    can_explain_decision: bool
    can_execute_action: bool
    can_use_result_to_continue: bool
    evidence: List[BoundedReviewText] = Field(
        min_length=1, max_length=MAX_SIMULATION_EVIDENCE_ITEMS
    )


class SimulationReport(SchemaModel):
    episode_results: List[EpisodeSimulationResult] = Field(
        min_length=1, max_length=MAX_PREPARATION_ITEMS
    )
    interaction_results: List[BoundedReviewText] = Field(
        default_factory=list, max_length=MAX_DETAIL_ITEMS
    )
    end_of_lesson_recall: BoundedReviewText
    blocking_findings: List[BoundedReviewText] = Field(
        default_factory=list, max_length=MAX_DETAIL_ITEMS
    )

    @model_validator(mode="after")
    def validate_serialized_size(self) -> "SimulationReport":
        if len(self.model_dump_json().encode("utf-8")) > MAX_SIMULATION_SERIALIZED_BYTES:
            raise ValueError("simulation report exceeds serialized byte limit")
        return self


class ReviewFinding(SchemaModel):
    finding_id: GeneratedId
    severity: Literal["blocking", "material", "polish"]
    artifact_type: ArtifactType
    artifact_id: GeneratedId
    criterion: ReviewCriterionId
    evidence: BoundedReviewText
    responsible_role: ResponsibleRole
    requested_change: BoundedReviewText
    invalidated_downstream_artifacts: List[ArtifactType] = Field(
        default_factory=list, max_length=MAX_REVIEW_ARTIFACT_ITEMS
    )

    @model_validator(mode="after")
    def validate_invalidated_artifacts(self) -> "ReviewFinding":
        _require_unique(
            self.invalidated_downstream_artifacts,
            "invalidated downstream artifacts",
        )
        return self


class LessonReviewDecision(SchemaModel):
    status: Literal["approved", "revision_required", "failed"]
    findings: List[ReviewFinding] = Field(
        default_factory=list, max_length=MAX_DETAIL_ITEMS
    )
    retained_artifacts: List[ArtifactType] = Field(
        default_factory=list, max_length=MAX_REVIEW_ARTIFACT_ITEMS
    )
    approval_summary: BoundedReviewText

    @model_validator(mode="after")
    def validate_decision(self) -> "LessonReviewDecision":
        _require_unique([item.finding_id for item in self.findings], "finding ids")
        _require_unique(self.retained_artifacts, "retained artifacts")
        blocking_or_material = any(item.severity in {"blocking", "material"} for item in self.findings)
        if self.status == "approved" and blocking_or_material:
            raise ValueError("approved review cannot contain blocking or material findings")
        if self.status in {"revision_required", "failed"} and not blocking_or_material:
            raise ValueError("revision_required or failed review requires a blocking or material finding")
        if len(self.model_dump_json().encode("utf-8")) > MAX_REVIEW_SERIALIZED_BYTES:
            raise ValueError("review decision exceeds serialized byte limit")
        return self


class ArtifactRevision(SchemaModel):
    artifact_type: ArtifactType
    version: PositiveInt
    responsible_role: ArtifactResponsibleRole
    finding_ids: List[GeneratedId] = Field(
        default_factory=list, max_length=MAX_DETAIL_ITEMS
    )


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
    artifact_history: List[ArtifactRevision] = Field(
        min_length=5, max_length=MAX_ARTIFACT_HISTORY_ITEMS
    )


class RoleCallRecord(SchemaModel):
    role: RoleName
    input_artifact_versions: Dict[ArtifactType, PositiveInt] = Field(default_factory=dict)
    output_artifact_type: Optional[ArtifactType] = None
    output_artifact_version: Optional[PositiveInt] = None
    duration_ms: int = Field(ge=0)
    retry_count: int = Field(ge=0)
    failure_category: Optional[NonEmptyString] = None
    token_usage: Optional[Dict[str, int]] = None
    review_finding_ids: List[GeneratedId] = Field(default_factory=list)

    @field_validator("token_usage", mode="before")
    @classmethod
    def validate_token_usage_vocabulary(
        cls,
        value: object,
    ) -> object:
        if value is None:
            return None
        if type(value) is not dict:
            raise ValueError("token usage must be an exact mapping")
        if not set(value).issubset(ROLE_CALL_TOKEN_USAGE_KEYS):
            raise ValueError("token usage contains an unknown counter")
        if any(
            type(item) is not int
            or not 0 <= item <= MAX_ROLE_CALL_TOKEN_COUNTER
            for item in value.values()
        ):
            raise ValueError("token usage values must be bounded integers")
        return dict(value)

    @model_validator(mode="after")
    def validate_output_and_tokens(self) -> "RoleCallRecord":
        if (self.output_artifact_type is None) != (self.output_artifact_version is None):
            raise ValueError("output type and version must be present or absent together")
        return self


class RuntimeCueProvenanceRecord(SchemaModel):
    """Private authored-clause link to one compiled runtime cue."""

    episode_id: GeneratedId
    clause_id: GeneratedId
    original_performance_cue_id: GeneratedId
    runtime_cue_id: GeneratedId
    spoken_text: CueSpokenText


class GenerationRecord(SchemaModel):
    generation_id: NonEmptyString
    lesson_id: NonEmptyString
    route_fingerprint: NonEmptyString
    prepared_lesson: PreparedLesson
    role_calls: List[RoleCallRecord] = Field(
        min_length=7, max_length=MAX_ARTIFACT_HISTORY_ITEMS
    )
    cue_provenance: List[RuntimeCueProvenanceRecord] = Field(
        min_length=1, max_length=MAX_PREPARATION_ITEMS
    )
    created_at: NonEmptyString
