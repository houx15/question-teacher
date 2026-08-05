from typing import Optional

import httpx

from app.config import Settings


class SpeechGenerationError(RuntimeError):
    pass


class OpenAISpeechClient:
    def __init__(
        self,
        settings: Settings,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ):
        self.settings = settings
        self.http = httpx.AsyncClient(
            timeout=settings.openai_timeout_seconds,
            transport=transport,
        )

    def _voice_configuration(self):
        values = (
            self.settings.tts_base_url,
            self.settings.tts_api_key,
            self.settings.tts_model,
            self.settings.tts_voice,
        )
        if not all(
            isinstance(value, str) and value.strip()
            for value in values
        ):
            raise SpeechGenerationError(
                "Speech service is not configured"
            )
        return tuple(value.strip() for value in values)

    async def synthesize(self, text: str) -> bytes:
        base_url, api_key, model, voice = self._voice_configuration()
        if not isinstance(text, str):
            raise SpeechGenerationError("Speech input must be text")
        if not text.strip():
            raise SpeechGenerationError("Speech input must not be blank")

        try:
            response = await self.http.post(
                f"{base_url.rstrip('/')}/audio/speech",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "voice": voice,
                    "input": text,
                    "response_format": "mp3",
                },
            )
        except httpx.HTTPError:
            raise SpeechGenerationError("Speech request failed") from None

        if not 200 <= response.status_code < 300:
            raise SpeechGenerationError(
                "Speech generation failed with status "
                f"{response.status_code}"
            )
        if not response.content:
            raise SpeechGenerationError(
                "Speech generation returned empty audio"
            )
        return response.content

    async def close(self) -> None:
        await self.http.aclose()
