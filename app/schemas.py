from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class ProblemInput(BaseModel):
    problem_text: str = Field(min_length=3)
    reference_answer: str = Field(min_length=1)
    required_method: Optional[
        Literal["factor", "quadratic_formula", "complete_the_square"]
    ] = None
    lesson_length: Literal["concise", "standard"] = "standard"


class MathStep(BaseModel):
    purpose: str
    operation: str
    state_before: List[str]
    state_after: List[str]
    reason: str


class BoardAction(BaseModel):
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
    target: Optional[str] = None
    content: Optional[str] = None
    source: Optional[str] = None
    relation_target: Optional[str] = None
    annotation: Optional[
        Literal["circle", "box", "underline", "arrow", "bracket", "label"]
    ] = None


class InteractionOption(BaseModel):
    option_id: str
    label: str


class Interaction(BaseModel):
    interaction_id: str
    kind: Literal[
        "point_select",
        "choice",
        "expression",
        "free_text",
        "transfer",
    ]
    prompt: str
    expected_answer: str
    options: List[InteractionOption] = Field(default_factory=list)
    hints: List[str] = Field(default_factory=list)
    explanation_after_correct: str = ""
    hint_audio_urls: List[str] = Field(default_factory=list)
    correct_audio_url: Optional[str] = None


class LessonMoment(BaseModel):
    purpose: str
    narration: str = Field(min_length=1, max_length=90)
    board_actions: List[BoardAction] = Field(default_factory=list)
    layer: Literal[
        "base",
        "micro_explanation",
        "comparison",
        "interaction",
    ] = "base"
    interaction: Optional[Interaction] = None


class TransferItem(BaseModel):
    problem_text: str
    expected_answer: str
    method_signal: str


class LessonDraft(BaseModel):
    title: str
    learning_goal: str
    opening: str
    method_rationale: str
    math_steps: List[MathStep]
    moments: List[LessonMoment]
    summary: str
    transfer_item: TransferItem


class ReviewDecision(BaseModel):
    status: Literal["approved", "revision_required"]
    overall_assessment: str
    must_fix: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_must_fix_for_revision(self) -> "ReviewDecision":
        if self.status == "revision_required" and not self.must_fix:
            raise ValueError(
                "must_fix must be nonempty when revision is required"
            )
        return self


class RuntimeBeat(BaseModel):
    beat_id: str
    purpose: str
    narration: str
    board_actions: List[BoardAction]
    layer: str
    interaction: Optional[Interaction] = None
    audio_url: Optional[str] = None
    next_beat_id: Optional[str] = None


class RuntimeLesson(BaseModel):
    lesson_id: str
    problem: str
    title: str
    learning_goal: str
    beats: List[RuntimeBeat]
    summary: str
    transfer_item: TransferItem
    validation_report: Dict[str, object]


class GenerationJob(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    stage: str
    lesson_id: Optional[str] = None
    error: Optional[str] = None
