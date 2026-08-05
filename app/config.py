import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class Settings:
    openai_base_url: Optional[str] = None
    openai_api_key: Optional[str] = None
    openai_model: Optional[str] = None
    openai_timeout_seconds: float = 90.0
    tts_base_url: Optional[str] = None
    tts_api_key: Optional[str] = None
    tts_model: Optional[str] = None
    tts_voice: Optional[str] = None

    @classmethod
    def from_env(cls) -> "Settings":
        openai_base_url = cls._normalize_url(os.getenv("OPENAI_BASE_URL"))
        openai_api_key = cls._normalize_string(os.getenv("OPENAI_API_KEY"))
        explicit_tts_base_url = cls._normalize_url(
            os.getenv("TTS_BASE_URL")
        )
        explicit_tts_api_key = cls._normalize_string(
            os.getenv("TTS_API_KEY")
        )
        tts_base_url = explicit_tts_base_url or openai_base_url
        if explicit_tts_api_key:
            tts_api_key = explicit_tts_api_key
        elif tts_base_url == openai_base_url:
            tts_api_key = openai_api_key
        else:
            tts_api_key = None

        return cls(
            openai_base_url=openai_base_url,
            openai_api_key=openai_api_key,
            openai_model=cls._normalize_string(os.getenv("OPENAI_MODEL")),
            openai_timeout_seconds=cls._parse_timeout(
                os.getenv("OPENAI_TIMEOUT_SECONDS", "90")
            ),
            tts_base_url=tts_base_url,
            tts_api_key=tts_api_key,
            tts_model=cls._normalize_string(os.getenv("TTS_MODEL")),
            tts_voice=cls._normalize_string(os.getenv("TTS_VOICE")),
        )

    @staticmethod
    def _normalize_string(value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return value.strip() or None

    @staticmethod
    def _parse_timeout(value: str) -> float:
        error = (
            "OPENAI_TIMEOUT_SECONDS must be a finite positive number; "
            f"got {value!r}"
        )
        try:
            timeout = float(value)
        except ValueError as exc:
            raise ValueError(error) from exc
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError(error)
        return timeout

    @classmethod
    def _normalize_url(cls, url: Optional[str]) -> Optional[str]:
        normalized = cls._normalize_string(url)
        if not normalized:
            return None
        return normalized.rstrip("/") or None

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
