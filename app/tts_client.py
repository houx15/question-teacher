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

    async def synthesize(self, text: str) -> bytes:
        if not self.settings.voice_configured:
            raise SpeechGenerationError("Speech service is not configured")
        if not text.strip():
            raise SpeechGenerationError("Speech input must not be blank")

        try:
            response = await self.http.post(
                self.settings.speech_url,
                headers={
                    "Authorization": (
                        f"Bearer {self.settings.tts_api_key}"
                    ),
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.settings.tts_model,
                    "voice": self.settings.tts_voice,
                    "input": text,
                    "response_format": "mp3",
                },
            )
        except httpx.HTTPError:
            raise SpeechGenerationError("Speech request failed") from None

        if response.status_code >= 400:
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
