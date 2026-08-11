"""Server-owned boundary for untrusted reference-solution text."""

import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence, Tuple

from pydantic import BaseModel

from app.preparation_models import SolutionTrace
from app.schemas import ProblemInput, ReferenceGroundingBrief


_OPAQUE_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{7,}")
_PHRASE_SPLIT = re.compile(r"[\r\n。！？!?；;]+")
_WHITESPACE = re.compile(r"\s+")
_MAX_CANDIDATES = 256
_MAX_WALK_NODES = 20_000


class ReferenceContentSafetyError(ValueError):
    """Raised without echoing the sensitive literal."""


def _normalize(value: str) -> str:
    return _WHITESPACE.sub("", value).casefold()


def _bounded_strings(value: Any) -> Iterable[str]:
    remaining = _MAX_WALK_NODES
    stack = [value]
    while stack and remaining:
        current = stack.pop()
        remaining -= 1
        if isinstance(current, str):
            yield current
        elif isinstance(current, BaseModel):
            stack.append(current.model_dump(mode="python"))
        elif isinstance(current, Mapping):
            stack.extend(current.values())
        elif isinstance(current, Sequence) and not isinstance(
            current, (bytes, bytearray)
        ):
            stack.extend(current)
    if stack:
        raise ReferenceContentSafetyError(
            "reference safety traversal exceeded its bound"
        )


@dataclass(frozen=True)
class ReferenceSafetyPolicy:
    """Detect raw-only literals while allowing public question/answer text."""

    sensitive_literals: Tuple[str, ...]

    @classmethod
    def from_problem(cls, problem: ProblemInput) -> "ReferenceSafetyPolicy":
        raw = problem.reference_solution_text or ""
        public = _normalize(
            problem.problem_text + "\n" + problem.reference_answer
        )
        candidates = []

        def add(candidate: str) -> None:
            normalized = _normalize(candidate)
            if (
                len(normalized) < 8
                or normalized in public
                or normalized in candidates
                or len(candidates) >= _MAX_CANDIDATES
            ):
                return
            candidates.append(normalized)

        for line in raw.splitlines():
            add(line)
        for phrase in _PHRASE_SPLIT.split(raw):
            add(phrase)
        for token in _OPAQUE_TOKEN.findall(raw):
            add(token)
        return cls(tuple(candidates))

    def ensure_safe(self, value: Any) -> None:
        if not self.sensitive_literals:
            return
        for text in _bounded_strings(value):
            normalized = _normalize(text)
            if any(item in normalized for item in self.sensitive_literals):
                raise ReferenceContentSafetyError(
                    "reference-only content crossed the safe boundary"
                )

    def sanitize_solution_trace(self, trace: SolutionTrace) -> SolutionTrace:
        payload = trace.model_dump(mode="python")
        payload["audit_notes"] = []
        anchors = [
            item["source_anchor"] for item in payload["assumptions"]
        ] + [item["source_anchor"] for item in payload["source_steps"]]
        counters = {}
        labels = {
            "problem": "题目结构依据",
            "answer": "参考答案结构依据",
            "solution": "参考解析结构依据",
            "verified_route": "已验证路线结构依据",
        }
        for anchor in anchors:
            source_kind = anchor["source_kind"]
            counters[source_kind] = counters.get(source_kind, 0) + 1
            if source_kind != "verified_route":
                anchor["source_id"] = "%s-anchor-%03d" % (
                    source_kind,
                    counters[source_kind],
                )
            anchor["excerpt"] = labels[source_kind]
        sanitized = SolutionTrace.model_validate(payload)
        self.ensure_safe(sanitized)
        return sanitized

    def sanitize_grounding_brief(
        self,
        brief: ReferenceGroundingBrief,
        reference_answer: str,
    ) -> ReferenceGroundingBrief:
        payload = brief.model_dump(mode="python")
        payload["audit_notes"] = []
        sanitized = ReferenceGroundingBrief.validate_for_reference_answer(
            payload,
            reference_answer,
        )
        self.ensure_safe(sanitized)
        return sanitized
