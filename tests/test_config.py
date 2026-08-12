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
    "TTS_PROVIDER",
    "TTS_BASE_URL",
    "TTS_API_KEY",
    "TTS_MODEL",
    "TTS_VOICE",
    "VOLCENGINE_TTS_ENDPOINT",
    "VOLCENGINE_TTS_API_KEY",
    "VOLCENGINE_TTS_RESOURCE_ID",
    "VOLCENGINE_TTS_VOICE",
    "VOLCENGINE_TTS_SPEED_RATIO",
    "VOLCENGINE_TTS_SAMPLE_RATE",
    "VOLCENGINE_TTS_UID",
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

    assert Settings.from_env().openai_timeout_seconds == 180.0

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


def test_volcengine_new_key_configuration_never_inherits_openai_key(
    monkeypatch,
):
    clear_settings_env(monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "model-secret")
    monkeypatch.setenv("TTS_PROVIDER", "volcengine")
    monkeypatch.setenv("VOLCENGINE_TTS_API_KEY", " voice-secret ")
    monkeypatch.setenv("VOLCENGINE_TTS_RESOURCE_ID", " seed-tts-2.0 ")
    monkeypatch.setenv("VOLCENGINE_TTS_VOICE", " teacher ")

    settings = Settings.from_env()

    assert settings.tts_provider == "volcengine"
    assert settings.volcengine_tts_endpoint == (
        "https://openspeech.bytedance.com/api/v3/tts/unidirectional"
    )
    assert settings.volcengine_tts_api_key == "voice-secret"
    assert settings.volcengine_tts_resource_id == "seed-tts-2.0"
    assert settings.volcengine_tts_voice == "teacher"
    assert settings.voice_configured is True


def test_volcengine_requires_api_key(monkeypatch):
    clear_settings_env(monkeypatch)
    monkeypatch.setenv("TTS_PROVIDER", "volcengine")
    monkeypatch.setenv("VOLCENGINE_TTS_RESOURCE_ID", "seed-tts-2.0")
    monkeypatch.setenv("VOLCENGINE_TTS_VOICE", "teacher")

    assert Settings.from_env().voice_configured is False

    monkeypatch.setenv("VOLCENGINE_TTS_API_KEY", "voice-secret")

    assert Settings.from_env().voice_configured is True


@pytest.mark.parametrize("provider", ["unknown", "VOLCENGINE"])
def test_tts_provider_rejects_unsupported_values(monkeypatch, provider):
    clear_settings_env(monkeypatch)
    monkeypatch.setenv("TTS_PROVIDER", provider)

    with pytest.raises(ValueError, match="TTS_PROVIDER"):
        Settings.from_env()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("VOLCENGINE_TTS_SPEED_RATIO", "0.49"),
        ("VOLCENGINE_TTS_SPEED_RATIO", "2.01"),
        ("VOLCENGINE_TTS_SPEED_RATIO", "nan"),
        ("VOLCENGINE_TTS_SAMPLE_RATE", "44100"),
        ("VOLCENGINE_TTS_SAMPLE_RATE", "not-a-number"),
        ("VOLCENGINE_TTS_UID", " "),
    ],
)
def test_volcengine_rejects_invalid_audio_settings(
    monkeypatch,
    name,
    value,
):
    clear_settings_env(monkeypatch)
    monkeypatch.setenv("TTS_PROVIDER", "volcengine")
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=name):
        Settings.from_env()
