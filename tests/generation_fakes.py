import copy
import json

from app.prompts import (
    DIRECTOR_SYSTEM,
    MATERIALS_SYSTEM,
    MATH_ROUTE_SYSTEM,
    REVISION_SYSTEM,
)


def cover_symbolic_narrative_route(response, user_prompt):
    """Make broad legacy fixtures behave like a route-following Director."""
    if not (
        isinstance(response, dict)
        and isinstance(response.get("moments"), list)
        and response["moments"]
    ):
        return response
    try:
        route = json.loads(user_prompt)["teaching_route"]
    except (KeyError, TypeError, json.JSONDecodeError):
        return response
    if route.get("verification_mode") != "symbolic_verified":
        return response

    covered = copy.deepcopy(response)
    steps = list(route.get("steps", []))
    available = sum(
        12 - len(moment.get("board_actions", []))
        for moment in covered["moments"]
    )
    if available < len(steps):
        return covered

    moment_index = 0
    for index, step in enumerate(steps, start=1):
        while len(
            covered["moments"][moment_index].get("board_actions", [])
        ) >= 12:
            moment_index += 1
        board_actions = covered["moments"][moment_index].setdefault(
            "board_actions",
            [],
        )
        board_actions.append(
            {
                "type": "write",
                "target": f"fake-symbolic-route-step-{index}",
                "content": step["statement_after"],
            }
        )
    return covered


class FakeClient:
    """Compatibility fake that decomposes legacy whole-lesson fixtures."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.all_calls = []
        self.route_calls = []
        self._pending_narrative = None
        self._pending_materials = None

    @property
    def system_prompts(self):
        return [system_prompt for system_prompt, _ in self.all_calls]

    @property
    def user_prompts(self):
        return [user_prompt for _, user_prompt in self.all_calls]

    def prompt_values(self, key):
        values = []
        for user_prompt in self.user_prompts:
            payload = json.loads(user_prompt)
            self._collect_prompt_values(payload, key, values)
        return values

    @classmethod
    def _collect_prompt_values(cls, value, key, values):
        if isinstance(value, dict):
            if key in value:
                values.append(value[key])
            for child in value.values():
                cls._collect_prompt_values(child, key, values)
        elif isinstance(value, list):
            for child in value:
                cls._collect_prompt_values(child, key, values)

    async def complete_json(self, system_prompt, user_prompt):
        call = (system_prompt, user_prompt)
        if system_prompt == MATH_ROUTE_SYSTEM:
            self.route_calls.append(call)
        else:
            self.all_calls.append(call)
        if system_prompt not in {MATH_ROUTE_SYSTEM, MATERIALS_SYSTEM}:
            self.calls.append(call)

        if (
            system_prompt in {DIRECTOR_SYSTEM, REVISION_SYSTEM}
            and self._pending_narrative is not None
        ):
            response = self._pending_narrative
            self._pending_narrative = None
            return cover_symbolic_narrative_route(
                response,
                user_prompt,
            )
        if (
            system_prompt == MATERIALS_SYSTEM
            and self._pending_materials is not None
        ):
            response = self._pending_materials
            self._pending_materials = None
            return copy.deepcopy(response)

        if (
            system_prompt == MATH_ROUTE_SYSTEM
            and self.responses
            and self._looks_like_route_free_narrative(self.responses[0])
        ):
            return self._default_factor_route()

        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response

        if system_prompt == MATH_ROUTE_SYSTEM:
            route = self._decompose_route_source(response)
            if route is not None:
                return route
        if system_prompt in {DIRECTOR_SYSTEM, REVISION_SYSTEM}:
            narrative = self._decompose_narrative_source(response)
            if narrative is not None:
                return cover_symbolic_narrative_route(
                    narrative,
                    user_prompt,
                )
        if system_prompt == MATERIALS_SYSTEM:
            materials = self._extract_materials(response)
            if materials is not None:
                return materials
        return copy.deepcopy(response)

    @staticmethod
    def _looks_like_route_free_narrative(response):
        return (
            isinstance(response, dict)
            and "title" in response
            and "moments" in response
            and "math_steps" not in response
        )

    @staticmethod
    def _default_factor_route():
        return {
            "math_steps": [
                {
                    "purpose": "因式分解",
                    "operation": "factor",
                    "operands": [],
                    "state_before": ["x^2-5x+6=0"],
                    "state_after": ["(x-2)(x-3)=0"],
                    "reason": "两个数相乘为 6、相加为 -5。",
                }
            ]
        }

    def _decompose_route_source(self, response):
        if not (
            isinstance(response, dict)
            and isinstance(response.get("math_steps"), list)
        ):
            return None

        route = {"math_steps": copy.deepcopy(response["math_steps"])}
        if "moments" not in response:
            return route

        narrative = self._decompose_narrative_source(response)
        assert narrative is not None
        self._pending_narrative = narrative
        return route

    def _decompose_narrative_source(self, response):
        if not (
            isinstance(response, dict)
            and isinstance(response.get("moments"), list)
            and "title" in response
        ):
            return None
        narrative = copy.deepcopy(response)
        narrative.pop("math_steps", None)
        transfer_item = narrative.pop("transfer_item", None)
        interaction_bindings = []
        for index, moment in enumerate(narrative["moments"]):
            moment["moment_id"] = moment.get(
                "moment_id",
                f"moment-{index}",
            )
            interaction = moment.pop("interaction", None)
            if interaction is not None:
                moment["interaction_intent"] = (
                    moment.get("interaction_intent")
                    or "诊断这个关键认知转折。"
                )
                interaction_bindings.append(
                    {
                        "moment_id": moment["moment_id"],
                        "interaction": interaction,
                    }
                )
            if moment.get("layer") == "interaction":
                moment["layer"] = "base"
        if transfer_item is not None:
            for option in transfer_item.get("options", []):
                option.pop("label", None)
            self._pending_materials = {
                "interactions": interaction_bindings,
                "transfer_item": transfer_item,
            }
        return narrative

    @staticmethod
    def _extract_materials(response):
        if not (
            isinstance(response, dict)
            and "moments" in response
            and "transfer_item" in response
        ):
            return None
        bindings = []
        for index, moment in enumerate(response["moments"]):
            if moment.get("interaction") is not None:
                bindings.append(
                    {
                        "moment_id": moment.get(
                            "moment_id",
                            f"moment-{index}",
                        ),
                        "interaction": copy.deepcopy(
                            moment["interaction"]
                        ),
                    }
                )
        transfer_item = copy.deepcopy(response["transfer_item"])
        for option in transfer_item.get("options", []):
            option.pop("label", None)
        return {
            "interactions": bindings,
            "transfer_item": transfer_item,
        }
