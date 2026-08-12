from typing import Literal

from pydantic import BaseModel, ConfigDict


GenerationFailureCategory = Literal[
    "provider_error",
    "invalid_structure",
    "reference_trace_failed",
    "reasoning_design_failed",
    "review_not_converged",
    "compile_failed",
    "tts_failed",
    "persistence_failed",
]


class InternalGenerationDiagnostic(BaseModel):
    """Private, content-free diagnostic for one failed generation job."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    category: GenerationFailureCategory
