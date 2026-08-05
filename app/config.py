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
    tts_provider: str = "openai_compatible"
    volcengine_tts_endpoint: Optional[str] = (
        "https://openspeech.bytedance.com/api/v3/tts/unidirectional"
    )
    volcengine_tts_api_key: Optional[str] = None
    volcengine_tts_resource_id: Optional[str] = None
    volcengine_tts_voice: Optional[str] = None
    volcengine_tts_speed_ratio: float = 1.0
    volcengine_tts_sample_rate: int = 24000
    volcengine_tts_uid: str = "ai-math-demo"

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

        provider = cls._parse_tts_provider(
            os.getenv("TTS_PROVIDER", "openai_compatible")
        )
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
            tts_provider=provider,
            volcengine_tts_endpoint=cls._normalize_url(
                os.getenv(
                    "VOLCENGINE_TTS_ENDPOINT",
                    (
                        "https://openspeech.bytedance.com/"
                        "api/v3/tts/unidirectional"
                    ),
                )
            ),
            volcengine_tts_api_key=cls._normalize_string(
                os.getenv("VOLCENGINE_TTS_API_KEY")
            ),
            volcengine_tts_resource_id=cls._normalize_string(
                os.getenv("VOLCENGINE_TTS_RESOURCE_ID")
            ),
            volcengine_tts_voice=cls._normalize_string(
                os.getenv("VOLCENGINE_TTS_VOICE")
            ),
            volcengine_tts_speed_ratio=cls._parse_speed_ratio(
                os.getenv("VOLCENGINE_TTS_SPEED_RATIO", "1.0")
            ),
            volcengine_tts_sample_rate=cls._parse_sample_rate(
                os.getenv("VOLCENGINE_TTS_SAMPLE_RATE", "24000")
            ),
            volcengine_tts_uid=cls._parse_uid(
                os.getenv("VOLCENGINE_TTS_UID", "ai-math-demo")
            ),
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

    @staticmethod
    def _parse_tts_provider(value: str) -> str:
        if value not in {"openai_compatible", "volcengine"}:
            raise ValueError(
                "TTS_PROVIDER must be 'openai_compatible' or 'volcengine'; "
                f"got {value!r}"
            )
        return value

    @staticmethod
    def _parse_speed_ratio(value: str) -> float:
        error = (
            "VOLCENGINE_TTS_SPEED_RATIO must be a finite number from "
            f"0.5 to 2.0; got {value!r}"
        )
        try:
            speed_ratio = float(value)
        except ValueError as exc:
            raise ValueError(error) from exc
        if (
            not math.isfinite(speed_ratio)
            or speed_ratio < 0.5
            or speed_ratio > 2.0
        ):
            raise ValueError(error)
        return speed_ratio

    @staticmethod
    def _parse_sample_rate(value: str) -> int:
        error = (
            "VOLCENGINE_TTS_SAMPLE_RATE must be one of "
            f"8000, 16000, or 24000; got {value!r}"
        )
        try:
            sample_rate = int(value)
        except ValueError as exc:
            raise ValueError(error) from exc
        if sample_rate not in {8000, 16000, 24000}:
            raise ValueError(error)
        return sample_rate

    @classmethod
    def _parse_uid(cls, value: str) -> str:
        uid = cls._normalize_string(value)
        if not uid:
            raise ValueError(
                "VOLCENGINE_TTS_UID must be a nonblank string; "
                f"got {value!r}"
            )
        return uid

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
        if self.tts_provider == "volcengine":
            has_auth = bool(
                isinstance(self.volcengine_tts_api_key, str)
                and self.volcengine_tts_api_key.strip()
            )
            has_required_strings = all(
                isinstance(value, str) and value.strip()
                for value in (
                    self.volcengine_tts_endpoint,
                    self.volcengine_tts_resource_id,
                    self.volcengine_tts_voice,
                    self.volcengine_tts_uid,
                )
            )
            has_valid_speed = (
                isinstance(
                    self.volcengine_tts_speed_ratio,
                    (int, float),
                )
                and not isinstance(self.volcengine_tts_speed_ratio, bool)
                and math.isfinite(self.volcengine_tts_speed_ratio)
                and 0.5 <= self.volcengine_tts_speed_ratio <= 2.0
            )
            return bool(
                has_required_strings
                and has_auth
                and has_valid_speed
                and self.volcengine_tts_sample_rate
                in {8000, 16000, 24000}
            )
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
