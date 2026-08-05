import os
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Settings:
    openai_base_url: Optional[str] = None
    openai_api_key: Optional[str] = None
    openai_model: Optional[str] = None
    timeout: float = 90.0
    tts_base_url: Optional[str] = None
    tts_api_key: Optional[str] = None
    tts_model: Optional[str] = None
    tts_voice: Optional[str] = None

    @classmethod
    def from_env(cls) -> "Settings":
        openai_base_url = cls._normalize_url(os.getenv("OPENAI_BASE_URL"))
        openai_api_key = os.getenv("OPENAI_API_KEY")

        return cls(
            openai_base_url=openai_base_url,
            openai_api_key=openai_api_key,
            openai_model=os.getenv("OPENAI_MODEL"),
            timeout=float(os.getenv("OPENAI_TIMEOUT_SECONDS", "90")),
            tts_base_url=cls._normalize_url(
                os.getenv("TTS_BASE_URL") or openai_base_url
            ),
            tts_api_key=os.getenv("TTS_API_KEY") or openai_api_key,
            tts_model=os.getenv("TTS_MODEL"),
            tts_voice=os.getenv("TTS_VOICE"),
        )

    @staticmethod
    def _normalize_url(url: Optional[str]) -> Optional[str]:
        return url.rstrip("/") if url else None

    @property
    def missing_model_settings(self) -> Tuple[str, ...]:
        values = (
            ("OPENAI_BASE_URL", self.openai_base_url),
            ("OPENAI_API_KEY", self.openai_api_key),
            ("OPENAI_MODEL", self.openai_model),
        )
        return tuple(name for name, value in values if not value)

    @property
    def model_configured(self) -> bool:
        return not self.missing_model_settings

    @property
    def chat_completions_url(self) -> Optional[str]:
        if not self.openai_base_url:
            return None
        return f"{self.openai_base_url}/chat/completions"

    @property
    def voice_configured(self) -> bool:
        return all(
            (
                self.tts_base_url,
                self.tts_api_key,
                self.tts_model,
                self.tts_voice,
            )
        )

    @property
    def speech_url(self) -> Optional[str]:
        if not self.tts_base_url:
            return None
        return f"{self.tts_base_url}/audio/speech"
