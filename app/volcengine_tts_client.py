import base64
import binascii
import codecs
import json
import math
import uuid
from typing import Any, List, Optional, Tuple

import httpx

from app.config import Settings
from app.tts_client import SpeechGenerationError


class _JsonFrameDecoder:
    def __init__(self) -> None:
        self._utf8 = codecs.getincrementaldecoder("utf-8")()
        self._json = json.JSONDecoder()
        self._buffer = ""

    def feed(self, chunk: bytes, *, final: bool = False) -> List[Any]:
        try:
            self._buffer += self._utf8.decode(chunk, final=final)
        except UnicodeDecodeError:
            raise SpeechGenerationError(
                "Speech generation returned invalid response"
            ) from None

        frames: List[Any] = []
        while True:
            self._buffer = self._buffer.lstrip()
            if not self._buffer:
                return frames
            try:
                frame, end = self._json.raw_decode(self._buffer)
            except json.JSONDecodeError:
                if final:
                    raise SpeechGenerationError(
                        "Speech generation returned invalid response"
                    ) from None
                return frames
            frames.append(frame)
            self._buffer = self._buffer[end:]


class VolcengineSpeechClient:
    def __init__(
        self,
        settings: Settings,
        transport: Optional[httpx.AsyncBaseTransport] = None,
    ) -> None:
        self.settings = settings
        self.http = httpx.AsyncClient(
            timeout=settings.openai_timeout_seconds,
            transport=transport,
        )

    def _voice_configuration(
        self,
    ) -> Tuple[str, str, str, str, float, int]:
        endpoint = self.settings.volcengine_tts_endpoint
        resource_id = self.settings.volcengine_tts_resource_id
        voice = self.settings.volcengine_tts_voice
        uid = self.settings.volcengine_tts_uid
        speed_ratio = self.settings.volcengine_tts_speed_ratio
        sample_rate = self.settings.volcengine_tts_sample_rate
        required_strings = (endpoint, resource_id, voice, uid)
        valid_core = all(
            isinstance(value, str) and value.strip()
            for value in required_strings
        )
        valid_audio = (
            isinstance(speed_ratio, (int, float))
            and not isinstance(speed_ratio, bool)
            and math.isfinite(speed_ratio)
            and 0.5 <= speed_ratio <= 2.0
            and sample_rate in {8000, 16000, 24000}
        )

        api_key = self.settings.volcengine_tts_api_key
        valid_auth = isinstance(api_key, str) and bool(api_key.strip())

        if not valid_core or not valid_audio or not valid_auth:
            raise SpeechGenerationError(
                "Speech service is not configured"
            )
        return (
            endpoint.strip(),
            api_key.strip(),
            resource_id.strip(),
            voice.strip(),
            float(speed_ratio),
            sample_rate,
        )

    async def synthesize(self, text: str) -> bytes:
        (
            endpoint,
            api_key,
            resource_id,
            voice,
            speed_ratio,
            sample_rate,
        ) = self._voice_configuration()
        if not isinstance(text, str):
            raise SpeechGenerationError("Speech input must be text")
        if not text.strip():
            raise SpeechGenerationError("Speech input must not be blank")

        headers = {
            "X-Api-Key": api_key,
            "X-Api-Resource-Id": resource_id,
            "X-Api-Request-Id": str(uuid.uuid4()),
            "Content-Type": "application/json",
        }
        body = {
            "user": {"uid": self.settings.volcengine_tts_uid.strip()},
            "req_params": {
                "text": text,
                "speaker": voice,
                "audio_params": {
                    "format": "mp3",
                    "sample_rate": sample_rate,
                    "speech_rate": round((speed_ratio - 1) * 100),
                },
            },
        }

        decoder = _JsonFrameDecoder()
        audio_parts: List[bytes] = []
        completed = False
        try:
            async with self.http.stream(
                "POST",
                endpoint,
                headers=headers,
                json=body,
            ) as response:
                if not 200 <= response.status_code < 300:
                    raise SpeechGenerationError(
                        "Speech generation failed with status "
                        f"{response.status_code}"
                    )
                async for chunk in response.aiter_bytes():
                    for frame in decoder.feed(chunk):
                        completed = self._consume_frame(
                            frame,
                            audio_parts,
                            completed,
                        )
                for frame in decoder.feed(b"", final=True):
                    completed = self._consume_frame(
                        frame,
                        audio_parts,
                        completed,
                    )
        except SpeechGenerationError:
            raise
        except httpx.HTTPError:
            raise SpeechGenerationError("Speech request failed") from None

        if not completed:
            raise SpeechGenerationError(
                "Speech generation returned missing completion"
            )
        audio = b"".join(audio_parts)
        if not audio:
            raise SpeechGenerationError(
                "Speech generation returned empty audio"
            )
        return audio

    @staticmethod
    def _consume_frame(
        frame: Any,
        audio_parts: List[bytes],
        completed: bool,
    ) -> bool:
        if not isinstance(frame, dict):
            raise SpeechGenerationError(
                "Speech generation returned invalid response"
            )
        code = frame.get("code")
        if not isinstance(code, int) or isinstance(code, bool):
            raise SpeechGenerationError(
                "Speech generation returned invalid response"
            )
        if code == 20000000:
            return True
        if code != 0:
            raise SpeechGenerationError(
                "Speech generation failed with vendor code "
                f"{code}"
            )
        encoded = frame.get("data")
        if encoded is None and isinstance(frame.get("sentence"), dict):
            return completed
        if not isinstance(encoded, str):
            raise SpeechGenerationError(
                "Speech generation returned invalid response"
            )
        try:
            decoded = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            raise SpeechGenerationError(
                "Speech generation returned invalid response"
            ) from None
        audio_parts.append(decoded)
        return completed

    async def close(self) -> None:
        await self.http.aclose()
