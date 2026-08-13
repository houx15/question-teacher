from types import MappingProxyType
from typing import Annotated, Dict, List, Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from app.math_content import (
    contains_internal_control_syntax,
    contains_math_markup as _contains_math_markup,
)
from app.math_expression import (
    MathOperationKind,
    StrictMathExpression,
    validate_operation_operands,
)


NonEmptyString = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
ProblemText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=3, max_length=12000),
]
ReferenceAnswerText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=12000),
]
ReferenceSolutionText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=12000),
]
MomentNarration = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=90),
]
CueSpokenText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=90),
]
METHOD_NAME_MAX_LENGTH = 8
METHOD_DEFINITION_MAX_LENGTH = 36
METHOD_TARGET_FORM_MAX_LENGTH = 80
METHOD_WHY_MAX_LENGTH = 32
MAX_NARRATIVE_SERIALIZED_BYTES = 64 * 1024
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
NarrativeTitle = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=120),
]
NarrativeLearningGoal = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=240),
]
NarrativeRationale = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=300),
]
NarrativePurpose = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=120),
]
NarrativeMathText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
]
NarrativeReason = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=300),
]
NarrativeBoardTarget = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=120),
]
NarrativeBoardContent = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
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
FIXED_RUNTIME_CUE_IDS = MappingProxyType(
    {
        "opening": "runtime-opening-cue",
        "method_introduction": "runtime-method-introduction-cue",
        "summary": "runtime-summary-cue",
        "transfer_intro": "runtime-transfer-intro-cue",
    }
)
RESERVED_RUNTIME_CUE_IDS = frozenset(FIXED_RUNTIME_CUE_IDS.values())
NarrativeLayer = Literal[
    "base",
    "micro_explanation",
    "comparison",
]


class SchemaModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProblemInput(SchemaModel):
    problem_text: ProblemText
    reference_answer: ReferenceAnswerText
    reference_solution_text: Optional[ReferenceSolutionText] = None
    required_method: Optional[
        Literal["factor", "quadratic_formula", "complete_the_square"]
    ] = None
    lesson_length: Literal["concise", "standard"] = "standard"


class ProblemFocusTarget(SchemaModel):
    target_id: NonEmptyString
    math_text: NonEmptyString
    display_mode: bool = False
    ordinal: int = Field(ge=1, le=64)


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


class NarrativeMathStep(MathStep):
    purpose: NarrativePurpose
    operands: List[NarrativeMathText] = Field(
        default_factory=list,
        max_length=1,
    )
    state_before: List[NarrativeMathText] = Field(
        min_length=1,
        max_length=4,
    )
    state_after: List[NarrativeMathText] = Field(
        min_length=1,
        max_length=4,
    )
    reason: NarrativeReason


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


SingleLetterSymbol = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z]$"),
]


def _normalize_reference_text(value: str) -> str:
    normalized = "".join(value.split())
    delimiters = (
        (r"\(", r"\)"),
        (r"\[", r"\]"),
        ("$$", "$$"),
        ("$", "$"),
    )
    unwrapped = True
    while unwrapped:
        unwrapped = False
        for opening, closing in delimiters:
            if (
                normalized.startswith(opening)
                and normalized.endswith(closing)
                and len(normalized) > len(opening) + len(closing)
            ):
                normalized = normalized[
                    len(opening) : -len(closing)
                ]
                unwrapped = True
                break
    return normalized


class GroundedAssumption(SchemaModel):
    assumption_id: GeneratedId
    expression: StrictMathExpression
    source_kind: Literal[
        "problem",
        "problem_derived",
        "solution",
    ] = "solution"


class GroundedReasoningStep(SchemaModel):
    step_id: GeneratedId
    statement_before: StrictMathExpression
    operation_kind: MathOperationKind
    operands: List[StrictMathExpression] = Field(
        default_factory=list,
        max_length=8,
    )
    statement_after: StrictMathExpression
    assumption_ids_used: List[GeneratedId] = Field(
        default_factory=list,
        max_length=8,
    )

    @model_validator(mode="after")
    def validate_operand_arity(self) -> "GroundedReasoningStep":
        validate_operation_operands(self.operation_kind, self.operands)
        return self


class GroundingCheckRequest(SchemaModel):
    check_id: GeneratedId
    source_step_id: GeneratedId
    kind: Literal[
        "substitution",
        "equivalence",
        "nonzero_division",
        "back_substitution",
    ]
    expression: StrictMathExpression
    expected: StrictMathExpression
    substitutions: Dict[
        SingleLetterSymbol,
        StrictMathExpression,
    ] = Field(default_factory=dict, max_length=4)
    nonzero_symbols: List[SingleLetterSymbol] = Field(
        default_factory=list,
        max_length=4,
    )
    conclusion_linked: bool = False


class ReferenceGroundingBrief(SchemaModel):
    task_summary: GeneratedFeedbackText
    target: StrictMathExpression
    assumptions: List[GroundedAssumption] = Field(max_length=8)
    reference_conclusion: StrictMathExpression
    method_name: MethodName
    reasoning_steps: List[GroundedReasoningStep] = Field(
        min_length=1,
        max_length=12,
    )
    check_requests: List[GroundingCheckRequest] = Field(max_length=8)
    audit_notes: List[GeneratedFeedbackText] = Field(max_length=8)

    @classmethod
    def validate_for_reference_answer(
        cls,
        value: object,
        reference_answer: str,
    ) -> "ReferenceGroundingBrief":
        return cls.model_validate(
            value,
            context={"reference_answer": reference_answer},
        )

    @model_validator(mode="before")
    @classmethod
    def require_reference_answer_context(
        cls,
        value: object,
        info: ValidationInfo,
    ) -> object:
        context = info.context
        reference_answer = (
            context.get("reference_answer")
            if isinstance(context, dict)
            else None
        )
        if (
            not isinstance(reference_answer, str)
            or not reference_answer.strip()
        ):
            raise ValueError(
                "nonblank reference_answer context is required"
            )
        return value

    @field_validator("reference_conclusion")
    @classmethod
    def validate_reference_conclusion(
        cls,
        value: str,
        info: ValidationInfo,
    ) -> str:
        reference_answer = info.context["reference_answer"]

        conclusion = _normalize_reference_text(value)
        answer = _normalize_reference_text(reference_answer)
        target = _normalize_reference_text(info.data.get("target", ""))

        if "=" in answer:
            agrees = conclusion == answer
        elif conclusion == answer:
            agrees = True
        elif conclusion.count("=") == 1:
            conclusion_target, conclusion_answer = conclusion.split("=")
            agrees = (
                bool(target)
                and conclusion_target == target
                and conclusion_answer == answer
            )
        else:
            agrees = False

        if not agrees:
            raise ValueError(
                "reference_conclusion must agree with reference_answer"
            )
        return value

    @model_validator(mode="after")
    def validate_check_request_ids(self) -> "ReferenceGroundingBrief":
        assumption_ids = [
            assumption.assumption_id for assumption in self.assumptions
        ]
        if len(assumption_ids) != len(set(assumption_ids)):
            raise ValueError("grounded assumption ids must be unique")
        known_assumptions = set(assumption_ids)
        for step in self.reasoning_steps:
            if not set(step.assumption_ids_used) <= known_assumptions:
                raise ValueError(
                    "grounded step references an unknown assumption"
                )
        step_ids = [step.step_id for step in self.reasoning_steps]
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("grounded step ids must be unique")
        known_steps = set(step_ids)
        check_ids = [
            request.check_id for request in self.check_requests
        ]
        if len(check_ids) != len(set(check_ids)):
            raise ValueError("check request ids must be unique")
        if any(
            request.source_step_id not in known_steps
            for request in self.check_requests
        ):
            raise ValueError("check request references an unknown step")
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


class NarrativeBoardAction(BoardAction):
    target: Optional[NarrativeBoardTarget] = None
    content: Optional[NarrativeBoardContent] = None
    source: Optional[NarrativeBoardTarget] = None
    relation_target: Optional[NarrativeBoardTarget] = None


class SyncVisualAction(SchemaModel):
    surface: Literal["problem", "board"]
    type: Literal[
        "write",
        "transform",
        "focus",
        "emphasize",
        "annotate",
        "fade",
        "reveal",
        "clear_focus",
    ]
    target: GeneratedId
    content: Optional[NarrativeBoardContent] = None
    source: Optional[GeneratedId] = None
    relation_target: Optional[GeneratedId] = None
    annotation: Optional[
        Literal["underline", "arrow", "bracket", "label"]
    ] = None
    emphasis_style: Optional[
        Literal["highlight", "underline", "red"]
    ] = None
    persistence: Optional[Literal["transient", "trace"]] = None

    @model_validator(mode="before")
    @classmethod
    def discard_known_focus_emphasis_style(cls, value: object) -> object:
        if type(value) is not dict or value.get("type") != "focus":
            return value
        style = value.get("emphasis_style")
        if style not in {"highlight", "underline", "red"}:
            return value
        normalized = dict(value)
        normalized["emphasis_style"] = None
        return normalized

    @model_validator(mode="after")
    def require_executable_payload(self) -> "SyncVisualAction":
        if self.surface == "problem" and self.type in {
            "write",
            "transform",
            "annotate",
        }:
            raise ValueError(
                f"problem actions cannot use {self.type}"
            )

        if self.type in {"write", "transform"} and not self.content:
            raise ValueError(f"{self.type} requires content")
        if self.type == "emphasize" and not self.emphasis_style:
            raise ValueError("emphasize requires emphasis_style")
        if self.persistence and self.type != "emphasize":
            raise ValueError("persistence is valid only for emphasize")

        allowed_fields = {
            "write": {"content"},
            "transform": {"content", "source"},
            "focus": set(),
            "emphasize": {"emphasis_style", "persistence"},
            "annotate": {"annotation", "content", "relation_target"},
            "fade": set(),
            "reveal": set(),
            "clear_focus": set(),
        }[self.type]
        optional_fields = {
            "content": self.content,
            "source": self.source,
            "relation_target": self.relation_target,
            "annotation": self.annotation,
            "emphasis_style": self.emphasis_style,
            "persistence": self.persistence,
        }
        irrelevant_fields = [
            field
            for field, value in optional_fields.items()
            if value is not None and field not in allowed_fields
        ]
        if irrelevant_fields:
            raise ValueError(
                f"{self.type} does not accept "
                f"{', '.join(irrelevant_fields)}"
            )

        if self.type == "annotate":
            if not self.annotation:
                raise ValueError("annotate requires annotation")
            if self.annotation == "label" and not self.content:
                raise ValueError("label annotation requires content")
            if self.annotation != "label" and self.content:
                raise ValueError(
                    "content is valid only for label annotation"
                )
            if self.annotation == "arrow" and not self.relation_target:
                raise ValueError(
                    "arrow annotation requires relation_target"
                )
            if self.annotation != "arrow" and self.relation_target:
                raise ValueError(
                    "relation_target is valid only for arrow annotation"
                )
        return self


class NarrativeSyncCue(SchemaModel):
    cue_id: GeneratedId
    spoken_text: CueSpokenText
    lead_actions: List[SyncVisualAction] = Field(
        default_factory=list,
        max_length=6,
    )
    start_actions: List[SyncVisualAction] = Field(
        default_factory=list,
        max_length=8,
    )
    end_actions: List[SyncVisualAction] = Field(
        default_factory=list,
        max_length=6,
    )

    @field_validator("spoken_text")
    @classmethod
    def reject_math_markup_in_spoken_text(cls, value: str) -> str:
        if _contains_math_markup(value) or contains_internal_control_syntax(
            value
        ):
            raise ValueError(
                "spoken_text must be natural speech without math markup"
            )
        return value


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


_LEGACY_LESSON_CUE_ID = "legacy-lesson-moment-cue"
_LEGACY_ACTION_COMPATIBILITY_ERROR = (
    "legacy action is not losslessly representable as SyncVisualAction"
)
_LEGACY_NARRATION_COMPATIBILITY_ERROR = (
    "legacy narration is not compatible with TTS spoken_text"
)
_LOSSLESS_LEGACY_ACTION_TYPES = {
    "write",
    "transform",
    "focus",
    "reveal",
    "fade",
    "annotate",
}
_LOSSLESS_LEGACY_ANNOTATIONS = {
    "underline",
    "arrow",
    "bracket",
    "label",
}


def _legacy_action_to_sync(
    action: BoardAction,
) -> SyncVisualAction:
    """Convert only legacy actions with a lossless SyncVisualAction form."""
    if action.type not in _LOSSLESS_LEGACY_ACTION_TYPES or (
        action.type == "annotate"
        and action.annotation not in _LOSSLESS_LEGACY_ANNOTATIONS
    ):
        raise ValueError(_LEGACY_ACTION_COMPATIBILITY_ERROR)

    payload = action.model_dump()
    payload["surface"] = "board"
    try:
        return SyncVisualAction.model_validate(payload)
    except ValidationError as error:
        raise ValueError(
            _LEGACY_ACTION_COMPATIBILITY_ERROR
        ) from error


def _sync_action_to_legacy(
    action: SyncVisualAction,
) -> Optional[BoardAction]:
    """Project only the exact common subset; sync_cues remain authoritative."""
    if action.surface != "board":
        return None
    if action.type not in _LOSSLESS_LEGACY_ACTION_TYPES:
        return None
    return BoardAction(
        type=action.type,
        target=action.target,
        content=action.content,
        source=action.source,
        relation_target=action.relation_target,
        annotation=action.annotation,
    )


def _flatten_legacy_board_actions(
    sync_cues: List[NarrativeSyncCue],
) -> List[BoardAction]:
    legacy_actions = []
    for cue in sync_cues:
        for action in (
            *cue.lead_actions,
            *cue.start_actions,
            *cue.end_actions,
        ):
            legacy_action = _sync_action_to_legacy(action)
            if legacy_action is not None:
                legacy_actions.append(legacy_action)
    return legacy_actions


def _canonicalize_legacy_moment_payload(
    value: dict,
    cue_id: str,
) -> dict:
    canonical = dict(value)
    spoken_text = canonical.pop("narration")
    if (
        isinstance(spoken_text, str)
        and (
            _contains_math_markup(spoken_text)
            or contains_internal_control_syntax(spoken_text)
        )
    ):
        raise ValueError(_LEGACY_NARRATION_COMPATIBILITY_ERROR)
    legacy_actions = canonical.pop("board_actions", [])
    start_actions = []
    for legacy_action in legacy_actions:
        action = (
            legacy_action
            if isinstance(legacy_action, BoardAction)
            else BoardAction.model_validate(legacy_action)
        )
        start_actions.append(_legacy_action_to_sync(action))
    canonical["sync_cues"] = [
        {
            "cue_id": cue_id,
            "spoken_text": spoken_text,
            "start_actions": start_actions,
        }
    ]
    return canonical


class LessonMoment(SchemaModel):
    purpose: NonEmptyString
    sync_cues: List[NarrativeSyncCue] = Field(
        min_length=1,
        max_length=5,
    )
    layer: LessonLayer = "base"
    interaction: Optional[Interaction] = None

    @model_validator(mode="before")
    @classmethod
    def canonicalize_legacy_moment(cls, value):
        if not isinstance(value, dict) or "sync_cues" in value:
            return value
        if "narration" not in value:
            return value

        # Standalone construction has no lesson ordinal; the temporary cue id
        # remains authoritative once this model has been parsed.
        return _canonicalize_legacy_moment_payload(
            value,
            _LEGACY_LESSON_CUE_ID,
        )

    @property
    def narration(self) -> str:
        return "".join(cue.spoken_text for cue in self.sync_cues)

    @property
    def board_actions(self) -> List[BoardAction]:
        return _flatten_legacy_board_actions(self.sync_cues)


class NarrativeMoment(SchemaModel):
    moment_id: GeneratedId
    purpose: NarrativePurpose
    sync_cues: List[NarrativeSyncCue] = Field(
        min_length=1,
        max_length=5,
    )
    layer: NarrativeLayer = "base"
    interaction_intent: Optional[InteractionIntentText] = None

    @property
    def spoken_narration(self) -> str:
        return "".join(cue.spoken_text for cue in self.sync_cues)


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


class MathRouteDraft(SchemaModel):
    math_steps: List[NarrativeMathStep] = Field(
        min_length=1,
        max_length=16,
    )


class NarrativeDraft(SchemaModel):
    title: NarrativeTitle
    learning_goal: NarrativeLearningGoal
    opening: MomentNarration
    method_rationale: NarrativeRationale
    method_introduction: MethodIntroduction
    moments: List[NarrativeMoment] = Field(
        min_length=1,
        max_length=16,
    )
    summary: MomentNarration

    @model_validator(mode="after")
    def validate_moment_slots(self) -> "NarrativeDraft":
        moment_ids = [moment.moment_id for moment in self.moments]
        if len(moment_ids) != len(set(moment_ids)):
            raise ValueError("narrative moment ids must be unique")
        cue_ids = [
            cue.cue_id
            for moment in self.moments
            for cue in moment.sync_cues
        ]
        if len(cue_ids) != len(set(cue_ids)):
            raise ValueError("narrative cue ids must be unique")
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
    label: GeneratedLabelText
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
    opening_sync_cues: Optional[List[NarrativeSyncCue]] = Field(
        default=None,
        min_length=1,
    )
    method_introduction_sync_cues: Optional[List[NarrativeSyncCue]] = Field(
        default=None,
        min_length=1,
    )
    summary_sync_cues: Optional[List[NarrativeSyncCue]] = Field(
        default=None,
        min_length=1,
    )
    fixed_section_interactions_after_cue: Dict[
        GeneratedId,
        Interaction,
    ] = Field(default_factory=dict)
    fixed_section_layers_by_cue: Dict[
        GeneratedId,
        NarrativeLayer,
    ] = Field(default_factory=dict)
    transfer_feedback_is_authoritative: bool = False
    math_steps: List[MathStep] = Field(default_factory=list)
    teaching_route: Dict[str, object] = Field(
        default_factory=lambda: {
            "verification_mode": "symbolic_verified",
            "teaching_route_fingerprint": "legacy-direct-draft",
        }
    )
    moments: List[LessonMoment] = Field(min_length=1)
    summary: NonEmptyString
    transfer_item: TransferItem

    @model_validator(mode="before")
    @classmethod
    def canonicalize_legacy_cue_ids(cls, value):
        if not isinstance(value, dict):
            return value
        raw_moments = value.get("moments")
        if not isinstance(raw_moments, list):
            return value

        canonical = dict(value)
        canonical_moments = []
        for ordinal, raw_moment in enumerate(raw_moments, start=1):
            # Legacy provenance exists only in this raw mapping shape. Parsed
            # LessonMoment objects and mappings with sync_cues are authoritative.
            if (
                isinstance(raw_moment, dict)
                and "narration" in raw_moment
                and "sync_cues" not in raw_moment
            ):
                canonical_moments.append(
                    _canonicalize_legacy_moment_payload(
                        raw_moment,
                        f"{_LEGACY_LESSON_CUE_ID}-{ordinal:03d}",
                    )
                )
            else:
                canonical_moments.append(raw_moment)
        canonical["moments"] = canonical_moments
        return canonical

    @model_validator(mode="after")
    def require_route_evidence(self) -> "LessonDraft":
        cue_ids = [
            cue.cue_id
            for cues in (
                self.opening_sync_cues or [],
                self.method_introduction_sync_cues or [],
                [
                    cue
                    for moment in self.moments
                    for cue in moment.sync_cues
                ],
                self.summary_sync_cues or [],
            )
            for cue in cues
        ]
        if len(cue_ids) != len(set(cue_ids)):
            raise ValueError("lesson cue ids must be unique")
        fixed_cue_ids = {
            cue.cue_id
            for cues in (
                self.opening_sync_cues or [],
                self.method_introduction_sync_cues or [],
                self.summary_sync_cues or [],
            )
            for cue in cues
        }
        if not set(self.fixed_section_interactions_after_cue).issubset(
            fixed_cue_ids
        ):
            raise ValueError(
                "fixed-section interaction must follow an authored fixed cue"
            )
        if not set(self.fixed_section_layers_by_cue).issubset(fixed_cue_ids):
            raise ValueError(
                "fixed-section layer must reference an authored fixed cue"
            )
        mode = self.teaching_route.get("verification_mode")
        if mode == "symbolic_verified" and not self.math_steps:
            raise ValueError("symbolic lessons require math_steps")
        if mode != "symbolic_verified" and self.math_steps:
            raise ValueError("grounded lessons do not use legacy math_steps")
        return self


class GroundedTransferDistractorReview(SchemaModel):
    option_id: GeneratedId
    misconception: GeneratedFeedbackText


class GroundedTransferReview(SchemaModel):
    transfer_problem_text: GeneratedProblemText
    correct_option_id: GeneratedId
    correct_canonical_answer: GeneratedMathAnswer
    distractors: List[GroundedTransferDistractorReview] = Field(
        min_length=2,
        max_length=3,
    )

    @model_validator(mode="after")
    def validate_distractor_ids(self) -> "GroundedTransferReview":
        option_ids = [
            distractor.option_id for distractor in self.distractors
        ]
        if len(option_ids) != len(set(option_ids)):
            raise ValueError("grounded distractor ids must be unique")
        return self


class ReviewDecision(SchemaModel):
    status: Literal["approved", "revision_required"]
    overall_assessment: NonEmptyString
    must_fix: List[NonEmptyString] = Field(default_factory=list)
    evidence: List[NonEmptyString] = Field(default_factory=list)
    grounded_transfer_review: Optional[GroundedTransferReview] = None

    @model_validator(mode="after")
    def validate_must_fix_for_status(self) -> "ReviewDecision":
        if self.status == "revision_required" and not self.must_fix:
            raise ValueError(
                "must_fix must be nonempty when revision is required"
            )
        if self.status == "approved" and self.must_fix:
            raise ValueError("approved reviews must not include must_fix")
        return self


class RuntimeSyncCue(SchemaModel):
    cue_id: NonEmptyString
    spoken_text: NonEmptyString
    lead_actions: List[SyncVisualAction] = Field(default_factory=list)
    start_actions: List[SyncVisualAction] = Field(default_factory=list)
    end_actions: List[SyncVisualAction] = Field(default_factory=list)
    audio_url: Optional[NonEmptyString] = None


class RuntimeBeat(SchemaModel):
    beat_id: NonEmptyString
    purpose: NonEmptyString
    narration: NonEmptyString
    board_actions: List[BoardAction]
    layer: LessonLayer
    sync_cues: List[RuntimeSyncCue] = Field(default_factory=list)
    interaction: Optional[Interaction] = None
    audio_url: Optional[NonEmptyString] = None
    next_beat_id: Optional[NonEmptyString] = None


class RuntimeLesson(SchemaModel):
    lesson_id: NonEmptyString
    problem: ProblemInput
    title: NonEmptyString
    learning_goal: NonEmptyString
    beats: List[RuntimeBeat] = Field(min_length=1)
    problem_focus_targets: List[ProblemFocusTarget] = Field(
        default_factory=list
    )
    summary: NonEmptyString
    transfer_item: TransferItem
    validation_report: Dict[str, object]

    @model_validator(mode="after")
    def require_unique_runtime_cue_ids(self) -> "RuntimeLesson":
        cue_ids = [
            cue.cue_id
            for beat in self.beats
            for cue in beat.sync_cues
        ]
        if len(cue_ids) != len(set(cue_ids)):
            raise ValueError("runtime cue ids must be unique")
        return self


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
