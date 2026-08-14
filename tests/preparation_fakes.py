import copy
from dataclasses import dataclass
from typing import Dict, List, Optional

from app.llm_client import ModelCompletion
from app.preparation_prompts import (
    CLASSROOM_DIRECTOR_SYSTEM,
    INTERACTION_DESIGNER_SYSTEM,
    LESSON_REVIEWER_SYSTEM,
    SCRIPT_TEACHER_SYSTEM,
    SOLUTION_TRACE_SYSTEM,
    STUDENT_SIMULATOR_SYSTEM,
    TEACHING_DESIGNER_SYSTEM,
    TEACHING_PROGRESSION_SYSTEM,
)


_ROLE_BY_SYSTEM = {
    SOLUTION_TRACE_SYSTEM: "reference_analyst",
    TEACHING_DESIGNER_SYSTEM: "teaching_designer",
    TEACHING_PROGRESSION_SYSTEM: "teaching_designer",
    SCRIPT_TEACHER_SYSTEM: "script_teacher",
    INTERACTION_DESIGNER_SYSTEM: "interaction_designer",
    CLASSROOM_DIRECTOR_SYSTEM: "classroom_director",
    STUDENT_SIMULATOR_SYSTEM: "student_simulator",
    LESSON_REVIEWER_SYSTEM: "lesson_reviewer",
}


def role_for_system(system: str) -> str:
    try:
        return _ROLE_BY_SYSTEM[system]
    except KeyError:
        raise AssertionError("unknown preparation system prompt") from None


@dataclass(frozen=True)
class RecordedCall:
    role: str
    system: str
    user: str


@dataclass(frozen=True)
class PreparationFakeResponse:
    payload: object
    token_usage: Optional[Dict[str, int]] = None


class PreparationFakeClient:
    def __init__(self, responses_by_role: Dict[str, List[object]]) -> None:
        self.responses_by_role = {
            role: list(responses)
            for role, responses in responses_by_role.items()
        }
        self.calls: List[RecordedCall] = []

    async def complete_json_with_metadata(
        self,
        system: str,
        user: str,
    ) -> object:
        role = role_for_system(system)
        self.calls.append(RecordedCall(role=role, system=system, user=user))
        try:
            response = self.responses_by_role[role].pop(0)
        except (KeyError, IndexError):
            raise AssertionError("no fake response remaining for %s" % role) from None
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, PreparationFakeResponse):
            return ModelCompletion(
                payload=copy.deepcopy(response.payload),
                token_usage=copy.deepcopy(response.token_usage),
            )
        return copy.deepcopy(response)

    async def complete_json(self, system: str, user: str) -> object:
        completion = await self.complete_json_with_metadata(system, user)
        if isinstance(completion, ModelCompletion):
            return copy.deepcopy(completion.payload)
        return completion
