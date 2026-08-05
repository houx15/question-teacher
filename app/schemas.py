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
MomentNarration = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=90),
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


class SchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProblemInput(SchemaModel):
    problem_text: ProblemText
    reference_answer: NonEmptyString
    required_method: Optional[
        Literal["factor", "quadratic_formula", "complete_the_square"]
    ] = None
    lesson_length: Literal["concise", "standard"] = "standard"


class MathStep(SchemaModel):
    purpose: NonEmptyString
    operation: MathOperation
    state_before: List[NonEmptyString]
    state_after: List[NonEmptyString]
    reason: NonEmptyString


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


class LessonMoment(SchemaModel):
    purpose: NonEmptyString
    narration: MomentNarration
    board_actions: List[BoardAction] = Field(default_factory=list)
    layer: LessonLayer = "base"
    interaction: Optional[Interaction] = None


class TransferItem(SchemaModel):
    problem_text: ProblemText
    expected_answer: NonEmptyString
    method_signal: NonEmptyString


class LessonDraft(SchemaModel):
    title: NonEmptyString
    learning_goal: NonEmptyString
    opening: NonEmptyString
    method_rationale: NonEmptyString
    math_steps: List[MathStep]
    moments: List[LessonMoment]
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
    beats: List[RuntimeBeat]
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
        if self.status == "completed" and not self.lesson_id:
            raise ValueError("completed jobs require lesson_id")
        if self.status == "failed" and not self.error:
            raise ValueError("failed jobs require error")
        if self.status in {"queued", "running"}:
            if self.lesson_id or self.error:
                raise ValueError(
                    f"{self.status} jobs must not include lesson_id or error"
                )
        return self
