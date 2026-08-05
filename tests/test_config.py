import asyncio

import httpx
import pytest

from app.config import Settings
from app.tts_client import OpenAISpeechClient, SpeechGenerationError


MODEL_ENV_NAMES = (
    "OPENAI_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
)
TTS_ENV_NAMES = (
    "TTS_BASE_URL",
    "TTS_API_KEY",
    "TTS_MODEL",
    "TTS_VOICE",
)


def clear_settings_env(monkeypatch):
    for name in (*MODEL_ENV_NAMES, "OPENAI_TIMEOUT_SECONDS", *TTS_ENV_NAMES):
        monkeypatch.delenv(name, raising=False)


def test_missing_model_environment_is_reported(monkeypatch):
    clear_settings_env(monkeypatch)

    settings = Settings.from_env()

    assert settings.model_configured is False
    assert settings.missing_model_settings == MODEL_ENV_NAMES


def test_whitespace_only_model_environment_is_reported_missing(monkeypatch):
    clear_settings_env(monkeypatch)
    monkeypatch.setenv("OPENAI_BASE_URL", "   ")
    monkeypatch.setenv("OPENAI_API_KEY", "\t")
    monkeypatch.setenv("OPENAI_MODEL", "\n")

    settings = Settings.from_env()

    assert settings.model_configured is False
    assert settings.missing_model_settings == MODEL_ENV_NAMES


def test_timeout_uses_default_and_environment_override(monkeypatch):
    clear_settings_env(monkeypatch)

    assert Settings.from_env().openai_timeout_seconds == 90.0

    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "12.5")

    assert Settings.from_env().openai_timeout_seconds == 12.5


@pytest.mark.parametrize(
    "value",
    ("malformed", "0", "-1", "nan", "inf", "-inf"),
)
def test_timeout_rejects_invalid_values(monkeypatch, value):
    clear_settings_env(monkeypatch)
    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", value)

    with pytest.raises(ValueError, match="OPENAI_TIMEOUT_SECONDS"):
        Settings.from_env()


def test_chat_completions_url_normalizes_trailing_slash(monkeypatch):
    clear_settings_env(monkeypatch)
    monkeypatch.setenv("OPENAI_BASE_URL", " https://example.test/v1/ ")
    monkeypatch.setenv("OPENAI_API_KEY", " secret ")
    monkeypatch.setenv("OPENAI_MODEL", " demo-model ")

    settings = Settings.from_env()

    assert settings.chat_completions_url == (
        "https://example.test/v1/chat/completions"
    )
    assert settings.openai_api_key == "secret"
    assert settings.openai_model == "demo-model"
    assert settings.model_configured is True


def test_tts_credentials_inherit_openai_credentials(monkeypatch):
    clear_settings_env(monkeypatch)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1/")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("OPENAI_MODEL", "demo-model")
    monkeypatch.setenv("TTS_BASE_URL", " ")
    monkeypatch.setenv("TTS_API_KEY", "\t")
    monkeypatch.setenv("TTS_MODEL", "demo-tts")
    monkeypatch.setenv("TTS_VOICE", "teacher")

    settings = Settings.from_env()

    assert settings.speech_url == "https://example.test/v1/audio/speech"
    assert settings.tts_api_key == "secret"
    assert settings.voice_configured is True


def test_tts_overrides_are_normalized(monkeypatch):
    clear_settings_env(monkeypatch)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("OPENAI_MODEL", "demo-model")
    monkeypatch.setenv("TTS_BASE_URL", " https://speech.test/v1/ ")
    monkeypatch.setenv("TTS_API_KEY", " tts-secret ")
    monkeypatch.setenv("TTS_MODEL", " demo-tts ")
    monkeypatch.setenv("TTS_VOICE", " teacher ")

    settings = Settings.from_env()

    assert settings.tts_base_url == "https://speech.test/v1"
    assert settings.tts_api_key == "tts-secret"
    assert settings.tts_model == "demo-tts"
    assert settings.tts_voice == "teacher"
    assert settings.speech_url == "https://speech.test/v1/audio/speech"
    assert settings.voice_configured is True


@pytest.mark.parametrize(
    (
        "tts_base_url",
        "tts_api_key",
        "expected_base_url",
        "expected_api_key",
    ),
    [
        (None, None, "https://model.test/v1", "model-secret"),
        (
            " https://model.test/v1/ ",
            None,
            "https://model.test/v1",
            "model-secret",
        ),
        (
            "https://speech.test/v1",
            None,
            "https://speech.test/v1",
            None,
        ),
        (
            "https://speech.test/v1",
            "speech-secret",
            "https://speech.test/v1",
            "speech-secret",
        ),
    ],
)
def test_tts_credentials_only_inherit_for_the_same_endpoint(
    monkeypatch,
    tts_base_url,
    tts_api_key,
    expected_base_url,
    expected_api_key,
):
    clear_settings_env(monkeypatch)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://model.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "model-secret")
    monkeypatch.setenv("OPENAI_MODEL", "demo-model")
    monkeypatch.setenv("TTS_MODEL", "demo-tts")
    monkeypatch.setenv("TTS_VOICE", "teacher")
    if tts_base_url is not None:
        monkeypatch.setenv("TTS_BASE_URL", tts_base_url)
    if tts_api_key is not None:
        monkeypatch.setenv("TTS_API_KEY", tts_api_key)

    settings = Settings.from_env()

    assert settings.tts_base_url == expected_base_url
    assert settings.tts_api_key == expected_api_key


def test_cross_origin_tts_without_key_never_sends_model_credential(
    monkeypatch,
):
    clear_settings_env(monkeypatch)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://model.test/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "model-secret")
    monkeypatch.setenv("OPENAI_MODEL", "demo-model")
    monkeypatch.setenv("TTS_BASE_URL", "https://speech.test/v1")
    monkeypatch.setenv("TTS_MODEL", "demo-tts")
    monkeypatch.setenv("TTS_VOICE", "teacher")
    requests = []

    def unexpected_request(request):
        requests.append(request)
        return httpx.Response(200, content=b"must-not-be-used")

    client = OpenAISpeechClient(
        Settings.from_env(),
        transport=httpx.MockTransport(unexpected_request),
    )

    async def scenario():
        try:
            with pytest.raises(
                SpeechGenerationError,
                match="not configured",
            ):
                await client.synthesize("有效讲解")
        finally:
            await client.close()

    asyncio.run(scenario())

    assert requests == []
