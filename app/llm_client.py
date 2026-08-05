import json
import re
from typing import Any, Dict, Optional

import httpx

from app.config import Settings


class ModelResponseError(RuntimeError):
    """Raised when a model request or response cannot be used safely."""


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
            raise ModelResponseError("Model response content is missing.")

        normalized = content.strip()
        if normalized.startswith("```"):
            fenced = OpenAICompatibleClient._FENCED_CONTENT.fullmatch(normalized)
            if fenced is None:
                raise ModelResponseError(
                    "Model response content has an invalid JSON code fence."
                )
            normalized = fenced.group("content").strip()
            if not normalized:
                raise ModelResponseError("Model response content is missing.")

        try:
            parsed = json.loads(normalized)
        except (TypeError, ValueError):
            raise ModelResponseError(
                "Model response content is not valid JSON."
            ) from None

        if not isinstance(parsed, dict):
            raise ModelResponseError(
                "Model response content must be a top-level JSON object."
            )
        return parsed

    async def complete_json(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> Dict[str, Any]:
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
            content = response_payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError):
            raise ModelResponseError(
                "Model response content is missing or invalid."
            ) from None

        return self.parse_json_content(content)

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
