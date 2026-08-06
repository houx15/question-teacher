import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from app.math_engine import (
    MathEngine,
    MathValidationError,
    MathValidationReason,
    ProblemValidation,
)


_MAX_PROBLEM_LENGTH = 4000
_MAX_ANSWER_LENGTH = 1000
_CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_DANGEROUS_PROTOCOL = re.compile(r"__[A-Za-z0-9_]+__\s*\(")
_CJK_CHARACTER = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_MATH_ONLY = re.compile(r"[0-9A-Za-z+\-*/^().=＝×÷−－–—＋²\s]+")
_DIRECT_EQUATION_COMMAND = re.compile(
    r"^[\u3400-\u4dbf\u4e00-\u9fff\s，,、]+方程\s*(.+)$"
)
_EXPLICIT_MATH_COMMAND = re.compile(
    r"(?:解方程|求解|计算|化简|求值|因式分解)\s*$"
)


class ProblemIntakeStatus(str, Enum):
    SYMBOLIC_VERIFIED = "symbolic_verified"
    CONTRADICTION = "contradiction"
    INVALID_INPUT = "invalid_input"
    UNSUPPORTED = "unsupported"


@dataclass(frozen=True)
class ProblemIntakeAssessment:
    status: ProblemIntakeStatus
    problem_validation: Optional[ProblemValidation] = None
    public_message: Optional[str] = None


class ProblemCapabilityProbe:
    def __init__(self, math_engine: MathEngine) -> None:
        self.math_engine = math_engine

    def assess(
        self,
        problem_text: str,
        reference_answer: str,
    ) -> ProblemIntakeAssessment:
        empty_assessment = _empty_input_assessment(
            problem_text,
            reference_answer,
        )
        if empty_assessment is not None:
            return empty_assessment
        if not _safe_broad_input(problem_text, reference_answer):
            return ProblemIntakeAssessment(
                ProblemIntakeStatus.INVALID_INPUT,
                public_message="输入包含不安全或过长的内容。",
            )

        try:
            report = self.math_engine.try_validate_supported_problem(
                problem_text,
                reference_answer,
            )
        except MathValidationError as error:
            if error.reason == MathValidationReason.CONTRADICTION:
                return ProblemIntakeAssessment(
                    ProblemIntakeStatus.CONTRADICTION,
                    public_message="参考答案与题目实际结果不一致。",
                )
        else:
            return ProblemIntakeAssessment(
                ProblemIntakeStatus.SYMBOLIC_VERIFIED,
                problem_validation=report,
            )

        try:
            equation_text = self.math_engine.extract_problem_equation(
                problem_text
            )
        except MathValidationError as error:
            if _is_safe_unsupported_task(problem_text, error):
                return ProblemIntakeAssessment(
                    ProblemIntakeStatus.UNSUPPORTED,
                )
            return ProblemIntakeAssessment(
                ProblemIntakeStatus.INVALID_INPUT,
                public_message="题目格式不正确。",
            )

        try:
            self.math_engine.solution_set([equation_text])
        except MathValidationError:
            return ProblemIntakeAssessment(
                ProblemIntakeStatus.UNSUPPORTED,
            )

        try:
            self.math_engine.answers_equivalent(
                reference_answer,
                reference_answer,
            )
        except MathValidationError:
            return ProblemIntakeAssessment(
                ProblemIntakeStatus.INVALID_INPUT,
                public_message="参考答案格式不正确。",
            )

        return ProblemIntakeAssessment(
            ProblemIntakeStatus.CONTRADICTION,
            public_message="参考答案与题目实际结果不一致。",
        )


def _empty_input_assessment(
    problem_text: object,
    reference_answer: object,
) -> Optional[ProblemIntakeAssessment]:
    if not isinstance(problem_text, str) or not problem_text.strip():
        return ProblemIntakeAssessment(
            ProblemIntakeStatus.INVALID_INPUT,
            public_message="题目不能为空。",
        )
    if not isinstance(reference_answer, str) or not reference_answer.strip():
        return ProblemIntakeAssessment(
            ProblemIntakeStatus.INVALID_INPUT,
            public_message="参考答案不能为空。",
        )
    return None


def _safe_broad_input(problem_text: str, reference_answer: str) -> bool:
    return (
        1 <= len(problem_text) <= _MAX_PROBLEM_LENGTH
        and 1 <= len(reference_answer) <= _MAX_ANSWER_LENGTH
        and _CONTROL_CHARACTERS.search(problem_text) is None
        and _CONTROL_CHARACTERS.search(reference_answer) is None
        and _DANGEROUS_PROTOCOL.search(problem_text) is None
        and _DANGEROUS_PROTOCOL.search(reference_answer) is None
    )


def _is_safe_unsupported_task(
    problem_text: str,
    validation_error: MathValidationError,
) -> bool:
    if validation_error.reason == MathValidationReason.UNSUPPORTED:
        return True
    if _looks_like_narrative_task(problem_text):
        return True
    return False


def _looks_like_narrative_task(problem_text: str) -> bool:
    stripped = problem_text.strip()
    if _looks_like_direct_math_input(stripped):
        return False
    if _CJK_CHARACTER.search(stripped):
        return True
    prose_words = re.findall(r"\b[A-Za-z]{2,}\b", stripped)
    return len(prose_words) >= 2


def _looks_like_direct_math_input(text: str) -> bool:
    colon_index = max(text.rfind(":"), text.rfind("："))
    if colon_index >= 0:
        prefix = text[:colon_index]
        suffix = text[colon_index + 1 :].strip()
        suffix_has_equality = "=" in suffix or "＝" in suffix
        if suffix_has_equality and _EXPLICIT_MATH_COMMAND.search(prefix):
            return True
        if (
            suffix_has_equality
            and _CJK_CHARACTER.search(suffix) is None
            and not _has_english_prose(suffix)
        ):
            return True
        if "方程" in prefix and suffix_has_equality:
            return True
        if (
            suffix
            and _MATH_ONLY.fullmatch(suffix)
            and not _has_english_prose(suffix)
        ):
            return True

    if _MATH_ONLY.fullmatch(text) and not _has_english_prose(text):
        if "=" in text or "＝" in text:
            return True
        return True

    command_match = _DIRECT_EQUATION_COMMAND.fullmatch(text)
    if command_match is None:
        return False
    command_suffix = command_match.group(1).lstrip(":：").strip()
    return bool(
        _MATH_ONLY.fullmatch(command_suffix)
        or "=" in command_suffix
        or "＝" in command_suffix
    )


def _has_english_prose(text: str) -> bool:
    return len(re.findall(r"\b[A-Za-z]{2,}\b", text)) >= 2
