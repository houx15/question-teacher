import json
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

import httpx
from pydantic import BaseModel

from app.config import Settings


class ModelResponseError(RuntimeError):
    """Raised when a model request or response cannot be used safely."""


class ModelStructureError(ModelResponseError):
    """Raised when provider output cannot be decoded as the requested shape."""

    def __init__(
        self,
        code: str,
        detail: Optional[str] = None,
        token_usage: Optional[Dict[str, int]] = None,
    ) -> None:
        super().__init__(detail or "Model response structure is invalid.")
        self.code = code
        self.token_usage = deepcopy(token_usage)


@dataclass(frozen=True)
class ModelCompletion:
    """Atomically binds one completion payload to its optional usage counters."""

    payload: object
    token_usage: Optional[Dict[str, int]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", deepcopy(self.payload))
        if self.token_usage is not None:
            object.__setattr__(
                self,
                "token_usage",
                deepcopy(self.token_usage),
            )


class OpenAICompatibleClient:
    _FENCED_CONTENT = re.compile(
        r"\A```[ \t]*(?:json)?[ \t]*(?:\r?\n)?(?P<content>.*?)(?:\r?\n)?[ \t]*```\Z",
        flags=re.IGNORECASE | re.DOTALL,
    )

    def __init__(
        self,
        settings: Settings,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self._settings = settings
        self._client = httpx.AsyncClient(
            timeout=settings.openai_timeout_seconds,
            transport=transport,
        )

    @staticmethod
    def parse_json_content(content: str) -> Dict[str, Any]:
        if not isinstance(content, str) or not content.strip():
            raise ModelStructureError(
                "missing_content",
                "Model response content is missing.",
            )

        normalized = content.strip()
        if normalized.startswith("```"):
            fenced = OpenAICompatibleClient._FENCED_CONTENT.fullmatch(normalized)
            if fenced is None:
                raise ModelStructureError(
                    "invalid_json_fence",
                    "Model response content has an invalid JSON code fence.",
                )
            normalized = fenced.group("content").strip()
            if not normalized:
                raise ModelStructureError(
                    "missing_content",
                    "Model response content is missing.",
                )

        try:
            parsed = json.loads(normalized)
        except (TypeError, ValueError):
            raise ModelStructureError(
                "invalid_json",
                "Model response content is not valid JSON.",
            ) from None

        if not isinstance(parsed, dict):
            raise ModelStructureError(
                "non_object_json",
                "Model response content must be a top-level JSON object.",
            )
        return parsed

    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Dict[str, Any]:
        completion = await self.complete_json_with_metadata(
            system_prompt,
            user_prompt,
        )
        payload = completion.payload
        if type(payload) is not dict:
            raise AssertionError("JSON completion payload must be an object")
        return payload

    async def complete_json_with_metadata(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> ModelCompletion:
        self._validate_configuration()
        payload = {
            "model": self._settings.openai_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.5,
            "response_format": {"type": "json_object"},
        }

        response = await self._post(payload)
        if response.status_code in (400, 422):
            fallback_payload = dict(payload)
            fallback_payload.pop("response_format")
            response = await self._post(fallback_payload)

        if response.is_error:
            raise ModelResponseError(
                f"Model request failed with HTTP status {response.status_code}."
            )

        try:
            response_payload = response.json()
        except (TypeError, ValueError):
            raise ModelStructureError(
                "invalid_response_envelope",
                "Model response content is missing or invalid.",
            ) from None
        usage = (
            response_payload.get("usage")
            if type(response_payload) is dict
            else None
        )
        try:
            content = response_payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ModelStructureError(
                "invalid_response_envelope",
                "Model response content is missing or invalid.",
                token_usage=usage,
            ) from None

        try:
            parsed = self.parse_json_content(content)
        except ModelStructureError as error:
            raise ModelStructureError(
                error.code,
                str(error),
                token_usage=usage,
            ) from None
        return ModelCompletion(payload=parsed, token_usage=usage)

    async def complete_model(
        self,
        system_prompt: str,
        user_prompt: str,
        model_type: Type[BaseModel],
    ) -> Dict[str, Any]:
        completion = await self.complete_model_with_metadata(
            system_prompt,
            user_prompt,
            model_type,
        )
        payload = completion.payload
        if type(payload) is not dict:
            raise AssertionError("JSON completion payload must be an object")
        return payload

    async def complete_model_with_metadata(
        self,
        system_prompt: str,
        user_prompt: str,
        model_type: Type[BaseModel],
    ) -> ModelCompletion:
        """Prefer provider-native schema decoding, with compatible fallback."""
        self._validate_configuration()
        payload = {
            "model": self._settings.openai_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": model_type.__name__,
                    "strict": True,
                    "schema": model_type.model_json_schema(),
                },
            },
        }
        response = await self._post(payload)
        if response.status_code in (400, 422):
            fallback_payload = dict(payload)
            fallback_payload["response_format"] = {"type": "json_object"}
            response = await self._post(fallback_payload)
        if response.status_code in (400, 422):
            fallback_payload = dict(payload)
            fallback_payload.pop("response_format")
            response = await self._post(fallback_payload)
        if response.is_error:
            raise ModelResponseError(
                f"Model request failed with HTTP status {response.status_code}."
            )

        return self._parse_completion_response(response)

    def _parse_completion_response(
        self,
        response: httpx.Response,
    ) -> ModelCompletion:
        try:
            response_payload = response.json()
        except (TypeError, ValueError):
            raise ModelStructureError(
                "invalid_response_envelope",
                "Model response content is missing or invalid.",
            ) from None
        usage = (
            response_payload.get("usage")
            if type(response_payload) is dict
            else None
        )
        try:
            content = response_payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise ModelStructureError(
                "invalid_response_envelope",
                "Model response content is missing or invalid.",
                token_usage=usage,
            ) from None
        try:
            parsed = self.parse_json_content(content)
        except ModelStructureError as error:
            raise ModelStructureError(
                error.code,
                str(error),
                token_usage=usage,
            ) from None
        return ModelCompletion(payload=parsed, token_usage=usage)

    async def close(self) -> None:
        await self._client.aclose()

    def _validate_configuration(self) -> None:
        missing = self._settings.missing_model_settings
        if missing:
            raise ModelResponseError(
                "Model configuration is incomplete; missing "
                + ", ".join(missing)
                + "."
            )

    async def _post(self, payload: Dict[str, Any]) -> httpx.Response:
        try:
            return await self._client.post(
                self._settings.chat_completions_url,
                headers={
                    "Authorization": f"Bearer {self._settings.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        except httpx.HTTPError:
            raise ModelResponseError("Model request failed.") from None
