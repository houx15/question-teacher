from typing import Annotated, Dict, List, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)


NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
ProblemText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=3),
]
ReferenceSolutionText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=12000),
]
MomentNarration = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=90),
]
METHOD_NAME_MAX_LENGTH = 8
METHOD_DEFINITION_MAX_LENGTH = 36
METHOD_TARGET_FORM_MAX_LENGTH = 80
METHOD_WHY_MAX_LENGTH = 32
MethodName = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=METHOD_NAME_MAX_LENGTH,
    ),
]
MethodDefinition = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=METHOD_DEFINITION_MAX_LENGTH,
    ),
]
MethodTargetForm = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=METHOD_TARGET_FORM_MAX_LENGTH,
    ),
]
MethodBenefit = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=METHOD_WHY_MAX_LENGTH,
    ),
]
GeneratedId = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]*$",
    ),
]
InteractionIntentText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=160),
]
GeneratedPromptText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=160),
]
GeneratedLabelText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=160),
]
GeneratedFeedbackText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=180),
]
GeneratedHintText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=120),
]
GeneratedProblemText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=3, max_length=500),
]
GeneratedMathAnswer = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=160),
]
MathOperation = Literal[
    "simplify",
    "add_both_sides",
    "subtract_both_sides",
    "multiply_both_sides",
    "divide_both_sides",
    "expand",
    "factor",
    "combine_like_terms",
    "take_square_root_both_sides",
    "split_plus_minus",
    "complete_the_square",
    "quadratic_formula",
]
LessonLayer = Literal[
    "base",
    "micro_explanation",
    "comparison",
    "interaction",
]
NarrativeLayer = Literal[
    "base",
    "micro_explanation",
    "comparison",
]


class SchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProblemInput(SchemaModel):
    problem_text: ProblemText
    reference_answer: NonEmptyString
    reference_solution_text: Optional[ReferenceSolutionText] = None
    required_method: Optional[
        Literal["factor", "quadratic_formula", "complete_the_square"]
    ] = None
    lesson_length: Literal["concise", "standard"] = "standard"


class MathStep(SchemaModel):
    purpose: NonEmptyString
    operation: MathOperation
    operands: List[NonEmptyString] = Field(default_factory=list)
    state_before: List[NonEmptyString] = Field(min_length=1)
    state_after: List[NonEmptyString] = Field(min_length=1)
    reason: NonEmptyString

    @model_validator(mode="after")
    def validate_operand_count(self) -> "MathStep":
        operand_required = {
            "add_both_sides",
            "subtract_both_sides",
            "multiply_both_sides",
            "divide_both_sides",
            "complete_the_square",
        }
        if self.operation in operand_required:
            if len(self.operands) != 1:
                raise ValueError(
                    f"{self.operation} requires exactly one operand"
                )
        elif self.operands:
            raise ValueError(f"{self.operation} does not accept operands")
        return self


class ReferenceMaterialAudit(SchemaModel):
    status: Literal["approved", "rejected"]
    claimed_answer: Optional[NonEmptyString] = None
    method_summary: Optional[NonEmptyString] = None
    key_steps: List[MathStep] = Field(default_factory=list)
    teaching_assets: List[NonEmptyString] = Field(default_factory=list)
    warnings: List[NonEmptyString] = Field(default_factory=list)
    blocking_issues: List[NonEmptyString] = Field(default_factory=list)
    evidence: List[NonEmptyString] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_decision(self) -> "ReferenceMaterialAudit":
        if self.status == "approved" and self.blocking_issues:
            raise ValueError("approved audit cannot contain blocking issues")
        if self.status == "rejected":
            if not self.blocking_issues or not self.evidence:
                raise ValueError(
                    "rejected audit requires blocking issues and evidence"
                )
        return self


class BoardAction(SchemaModel):
    type: Literal[
        "write",
        "transform",
        "focus",
        "annotate",
        "compare",
        "mask",
        "reveal",
        "fade",
        "pause",
        "clear",
    ]
    target: Optional[NonEmptyString] = None
    content: Optional[NonEmptyString] = None
    source: Optional[NonEmptyString] = None
    relation_target: Optional[NonEmptyString] = None
    annotation: Optional[
        Literal["circle", "box", "underline", "arrow", "bracket", "label"]
    ] = None

    @model_validator(mode="after")
    def require_executable_payload(self) -> "BoardAction":
        if self.type in {"write", "transform"}:
            if not self.target or not self.content:
                raise ValueError(
                    f"{self.type} requires both target and content"
                )
        elif self.type in {"focus", "mask", "reveal", "fade"}:
            if not self.target:
                raise ValueError(f"{self.type} requires target")
        elif self.type == "annotate":
            if not self.target or not self.annotation:
                raise ValueError("annotate requires target and annotation")
            if self.annotation == "label" and not self.content:
                raise ValueError("label annotation requires content")
            if self.annotation == "arrow" and not self.relation_target:
                raise ValueError("arrow annotation requires relation_target")
        elif self.type == "compare":
            if not self.target or not self.relation_target:
                raise ValueError(
                    "compare requires target and relation_target"
                )
        return self


class InteractionOption(SchemaModel):
    option_id: NonEmptyString
    label: NonEmptyString
    feedback: Optional[NonEmptyString] = None
    feedback_audio_url: Optional[NonEmptyString] = None


class Interaction(SchemaModel):
    interaction_id: NonEmptyString
    kind: Literal[
        "point_select",
        "choice",
        "expression",
        "free_text",
        "transfer",
    ]
    prompt: NonEmptyString
    expected_answer: NonEmptyString
    options: List[InteractionOption] = Field(default_factory=list)
    hints: List[NonEmptyString] = Field(default_factory=list)
    explanation_after_correct: str = ""
    hint_audio_urls: List[NonEmptyString] = Field(default_factory=list)
    correct_audio_url: Optional[NonEmptyString] = None

    @field_validator("explanation_after_correct")
    @classmethod
    def normalize_optional_feedback(cls, value: str) -> str:
        if value == "":
            return value
        normalized = value.strip()
        if not normalized:
            raise ValueError(
                "explanation_after_correct must be nonempty when provided"
            )
        return normalized

    @model_validator(mode="after")
    def require_executable_options(self) -> "Interaction":
        if self.kind == "choice":
            if not self.options:
                raise ValueError("choice interactions require options")
            option_ids = [option.option_id for option in self.options]
            if len(option_ids) != len(set(option_ids)):
                raise ValueError("choice option ids must be unique")
            if self.expected_answer not in option_ids:
                raise ValueError(
                    "choice expected_answer must match an option_id"
                )
        elif self.options:
            raise ValueError(
                "only choice interactions may provide options"
            )
        return self


class LessonMoment(SchemaModel):
    purpose: NonEmptyString
    narration: MomentNarration
    board_actions: List[BoardAction] = Field(default_factory=list)
    layer: LessonLayer = "base"
    interaction: Optional[Interaction] = None


class NarrativeMoment(SchemaModel):
    moment_id: GeneratedId
    purpose: NonEmptyString
    narration: MomentNarration
    board_actions: List[BoardAction] = Field(default_factory=list)
    layer: NarrativeLayer = "base"
    interaction_intent: Optional[InteractionIntentText] = None


class TransferOption(SchemaModel):
    option_id: NonEmptyString
    label: Optional[NonEmptyString] = None
    canonical_answer: NonEmptyString
    feedback: NonEmptyString


class TransferItem(SchemaModel):
    problem_text: ProblemText
    expected_answer: NonEmptyString
    method_signal: NonEmptyString
    options: List[TransferOption] = Field(default_factory=list)
    correct_option_id: Optional[NonEmptyString] = None

    @model_validator(mode="after")
    def validate_diagnostic_options(self) -> "TransferItem":
        if not self.options:
            if self.correct_option_id is not None:
                raise ValueError(
                    "correct_option_id requires diagnostic options"
                )
            return self

        if len(self.options) not in {3, 4}:
            raise ValueError("diagnostic options must contain 3 or 4 items")

        option_ids = [option.option_id for option in self.options]
        if len(option_ids) != len(set(option_ids)):
            raise ValueError("transfer option ids must be unique")

        if self.correct_option_id not in option_ids:
            raise ValueError(
                "correct_option_id must match a transfer option_id"
            )
        return self


class MethodIntroduction(SchemaModel):
    method_name: MethodName
    student_definition: MethodDefinition
    target_form: MethodTargetForm
    why_it_helps: MethodBenefit

    @property
    def spoken_narration(self) -> str:
        return "".join(
            self._as_sentence(fragment)
            for fragment in (
                f"今天用{self.method_name}",
                self.student_definition,
                self.why_it_helps,
            )
        )

    @staticmethod
    def _as_sentence(fragment: str) -> str:
        if fragment.endswith(("。", "！", "？", "!", "?")):
            return fragment
        return f"{fragment}。"


class NarrativeDraft(SchemaModel):
    title: NonEmptyString
    learning_goal: NonEmptyString
    opening: NonEmptyString
    method_rationale: NonEmptyString
    method_introduction: MethodIntroduction
    math_steps: List[MathStep] = Field(min_length=1)
    moments: List[NarrativeMoment] = Field(min_length=1)
    summary: NonEmptyString

    @model_validator(mode="after")
    def validate_moment_slots(self) -> "NarrativeDraft":
        moment_ids = [moment.moment_id for moment in self.moments]
        if len(moment_ids) != len(set(moment_ids)):
            raise ValueError("narrative moment ids must be unique")
        intent_count = sum(
            moment.interaction_intent is not None
            for moment in self.moments
        )
        if intent_count not in {1, 2, 3}:
            raise ValueError(
                "narrative requires 1 to 3 interaction intents"
            )
        return self


class GeneratedInteractionOption(SchemaModel):
    option_id: GeneratedId
    label: GeneratedLabelText
    feedback: GeneratedFeedbackText


class GeneratedChoiceInteraction(SchemaModel):
    interaction_id: GeneratedId
    kind: Literal["choice"]
    prompt: GeneratedPromptText
    expected_answer: GeneratedId
    options: List[GeneratedInteractionOption] = Field(
        min_length=3,
        max_length=4,
    )
    hints: List[GeneratedHintText] = Field(default_factory=list, max_length=3)
    explanation_after_correct: Annotated[
        str,
        StringConstraints(strip_whitespace=True, max_length=180),
    ] = ""

    @model_validator(mode="after")
    def validate_choice(self) -> "GeneratedChoiceInteraction":
        option_ids = [option.option_id for option in self.options]
        if len(option_ids) != len(set(option_ids)):
            raise ValueError("choice option ids must be unique")
        if self.expected_answer not in option_ids:
            raise ValueError(
                "choice expected_answer must match an option_id"
            )
        return self


class GeneratedTransferOption(SchemaModel):
    option_id: GeneratedId
    canonical_answer: GeneratedMathAnswer
    feedback: GeneratedFeedbackText


class GeneratedTransferItem(SchemaModel):
    problem_text: GeneratedProblemText
    expected_answer: GeneratedMathAnswer
    method_signal: GeneratedHintText
    options: List[GeneratedTransferOption] = Field(
        min_length=3,
        max_length=4,
    )
    correct_option_id: GeneratedId

    @model_validator(mode="after")
    def validate_options(self) -> "GeneratedTransferItem":
        option_ids = [option.option_id for option in self.options]
        if len(option_ids) != len(set(option_ids)):
            raise ValueError("transfer option ids must be unique")
        if self.correct_option_id not in option_ids:
            raise ValueError(
                "correct_option_id must match a transfer option_id"
            )
        return self


class InteractionBinding(SchemaModel):
    moment_id: GeneratedId
    interaction: GeneratedChoiceInteraction


class MaterialsDraft(SchemaModel):
    interactions: List[InteractionBinding] = Field(
        min_length=1,
        max_length=3,
    )
    transfer_item: GeneratedTransferItem


class LessonDraft(SchemaModel):
    title: NonEmptyString
    learning_goal: NonEmptyString
    opening: NonEmptyString
    method_rationale: NonEmptyString
    method_introduction: MethodIntroduction
    math_steps: List[MathStep] = Field(min_length=1)
    moments: List[LessonMoment] = Field(min_length=1)
    summary: NonEmptyString
    transfer_item: TransferItem


class ReviewDecision(SchemaModel):
    status: Literal["approved", "revision_required"]
    overall_assessment: NonEmptyString
    must_fix: List[NonEmptyString] = Field(default_factory=list)
    evidence: List[NonEmptyString] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_must_fix_for_status(self) -> "ReviewDecision":
        if self.status == "revision_required" and not self.must_fix:
            raise ValueError(
                "must_fix must be nonempty when revision is required"
            )
        if self.status == "approved" and self.must_fix:
            raise ValueError("approved reviews must not include must_fix")
        return self


class RuntimeBeat(SchemaModel):
    beat_id: NonEmptyString
    purpose: NonEmptyString
    narration: NonEmptyString
    board_actions: List[BoardAction]
    layer: LessonLayer
    interaction: Optional[Interaction] = None
    audio_url: Optional[NonEmptyString] = None
    next_beat_id: Optional[NonEmptyString] = None


class RuntimeLesson(SchemaModel):
    lesson_id: NonEmptyString
    problem: ProblemInput
    title: NonEmptyString
    learning_goal: NonEmptyString
    beats: List[RuntimeBeat] = Field(min_length=1)
    summary: NonEmptyString
    transfer_item: TransferItem
    validation_report: Dict[str, object]


class GenerationJob(SchemaModel):
    """Validated snapshot; reconstruct jobs instead of model_copy(update=...)."""

    job_id: NonEmptyString
    status: Literal["queued", "running", "completed", "failed"]
    stage: NonEmptyString
    lesson_id: Optional[NonEmptyString] = None
    error: Optional[NonEmptyString] = None

    @model_validator(mode="after")
    def validate_status_payload(self) -> "GenerationJob":
        if self.status == "completed":
            if not self.lesson_id:
                raise ValueError("completed jobs require lesson_id")
            if self.error:
                raise ValueError("completed jobs must not include error")
        elif self.status == "failed":
            if not self.error:
                raise ValueError("failed jobs require error")
            if self.lesson_id:
                raise ValueError("failed jobs must not include lesson_id")
        else:
            if self.lesson_id or self.error:
                raise ValueError(
                    f"{self.status} jobs must not include lesson_id or error"
                )
        return self
