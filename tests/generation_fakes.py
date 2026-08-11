import copy

from app.prompts import MATH_ROUTE_SYSTEM
from tests.preparation_fakes import role_for_system


class CompositeGenerationClient:
    """Route calls by role without encoding one global prompt sequence."""

    def __init__(self, route_client, preparation_client):
        self.route_client = route_client
        self.preparation_client = preparation_client

    @property
    def preparation_calls(self):
        return list(self.preparation_client.calls)

    async def complete_json_with_metadata(self, system, user):
        try:
            role_for_system(system)
        except AssertionError:
            return await self.route_client.complete_json(system, user)
        return await self.preparation_client.complete_json_with_metadata(
            system,
            user,
        )

    async def complete_json(self, system, user):
        try:
            role_for_system(system)
        except AssertionError:
            return await self.route_client.complete_json(system, user)
        return await self.preparation_client.complete_json(system, user)


class FakeClient:
    """Small sequential fake for route, grounding, and reference-audit calls."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.all_calls = []
        self.route_calls = []

    @property
    def system_prompts(self):
        return [system for system, _ in self.all_calls]

    @property
    def user_prompts(self):
        return [user for _, user in self.all_calls]

    async def complete_json(self, system, user):
        call = (system, user)
        self.all_calls.append(call)
        self.calls.append(call)
        if system == MATH_ROUTE_SYSTEM:
            self.route_calls.append(call)
        if not self.responses:
            raise AssertionError("no route fake response remaining")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return copy.deepcopy(response)
