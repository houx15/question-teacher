from app.config import Settings


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


def test_timeout_uses_default_and_environment_override(monkeypatch):
    clear_settings_env(monkeypatch)

    assert Settings.from_env().timeout == 90.0

    monkeypatch.setenv("OPENAI_TIMEOUT_SECONDS", "12.5")

    assert Settings.from_env().timeout == 12.5


def test_chat_completions_url_normalizes_trailing_slash(monkeypatch):
    clear_settings_env(monkeypatch)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1/")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("OPENAI_MODEL", "demo-model")

    settings = Settings.from_env()

    assert settings.chat_completions_url == (
        "https://example.test/v1/chat/completions"
    )
    assert settings.model_configured is True


def test_tts_credentials_inherit_openai_credentials(monkeypatch):
    clear_settings_env(monkeypatch)
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test/v1/")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    monkeypatch.setenv("OPENAI_MODEL", "demo-model")
    monkeypatch.setenv("TTS_MODEL", "demo-tts")
    monkeypatch.setenv("TTS_VOICE", "teacher")

    settings = Settings.from_env()

    assert settings.speech_url == "https://example.test/v1/audio/speech"
    assert settings.tts_api_key == "secret"
    assert settings.voice_configured is True
