import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys

import httpx
import pytest
from pydantic import BaseModel, ConfigDict

from app.config import Settings
from app.llm_client import (
    ModelResponseError,
    ModelStructureError,
    OpenAICompatibleClient,
)


def configured_settings(**overrides):
    values = {
        "openai_base_url": "https://model.example/v1",
        "openai_api_key": "test-secret-key",
        "openai_model": "demo-model",
        "openai_timeout_seconds": 12.5,
    }
    values.update(overrides)
    return Settings(**values)


class StructuredProbe(BaseModel):
    model_config = ConfigDict(extra="forbid")
    answer: int
    ok: bool


def test_complete_json_posts_chat_completion_and_returns_object():
    captured = {}

    def handler(request):
        captured["request"] = request
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"answer": 42, "ok": true}'}}
                ]
            },
        )

    client = OpenAICompatibleClient(
        configured_settings(),
        transport=httpx.MockTransport(handler),
    )

    result = asyncio.run(
        client.complete_json("You return JSON.", "Solve the problem.")
    )
    asyncio.run(client.close())

    request = captured["request"]
    assert request.url.path == "/v1/chat/completions"
    assert request.headers["Authorization"] == "Bearer test-secret-key"
    assert json.loads(request.content) == {
        "model": "demo-model",
        "messages": [
            {"role": "system", "content": "You return JSON."},
            {"role": "user", "content": "Solve the problem."},
        ],
        "temperature": 0.5,
        "response_format": {"type": "json_object"},
    }
    assert result == {"answer": 42, "ok": True}


def test_complete_json_with_metadata_returns_payload_and_usage_in_one_request():
    request_count = 0

    def handler(request):
        nonlocal request_count
        request_count += 1
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"answer":7}'}}],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 3,
                    "total_tokens": 11,
                },
            },
        )

    client = OpenAICompatibleClient(
        configured_settings(),
        transport=httpx.MockTransport(handler),
    )

    completion = asyncio.run(
        client.complete_json_with_metadata("system", "user")
    )

    asyncio.run(client.close())
    assert request_count == 1
    assert completion.payload == {"answer": 7}
    assert completion.token_usage == {
        "prompt_tokens": 8,
        "completion_tokens": 3,
        "total_tokens": 11,
    }


def test_complete_model_uses_native_json_schema_and_returns_json_object():
    request_bodies = []

    def handler(request):
        request_bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"answer":7,"ok":true}'}}
                ]
            },
        )

    client = OpenAICompatibleClient(
        configured_settings(), transport=httpx.MockTransport(handler)
    )

    completion = asyncio.run(
        client.complete_model_with_metadata(
            "system", "user", StructuredProbe
        )
    )
    asyncio.run(client.close())

    response_format = request_bodies[0]["response_format"]
    assert request_bodies[0]["temperature"] == 0.2
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["name"] == "StructuredProbe"
    assert response_format["json_schema"]["strict"] is True
    assert response_format["json_schema"]["schema"] == (
        StructuredProbe.model_json_schema()
    )
    assert completion.payload == {"answer": 7, "ok": True}


def test_complete_model_falls_back_to_json_object_when_schema_is_unsupported():
    request_bodies = []

    def handler(request):
        request_bodies.append(json.loads(request.content))
        if len(request_bodies) == 1:
            return httpx.Response(400, json={"error": "unsupported schema"})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"answer":7,"ok":true}'}}
                ]
            },
        )

    client = OpenAICompatibleClient(
        configured_settings(), transport=httpx.MockTransport(handler)
    )

    result = asyncio.run(
        client.complete_model("system", "user", StructuredProbe)
    )
    asyncio.run(client.close())

    assert result == {"answer": 7, "ok": True}
    assert request_bodies[0]["response_format"]["type"] == "json_schema"
    assert request_bodies[1]["response_format"] == {"type": "json_object"}


def test_complete_model_uses_json_object_directly_for_deepseek():
    request_bodies = []

    def handler(request):
        request_bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"answer":7,"ok":true}'}}
                ]
            },
        )

    client = OpenAICompatibleClient(
        configured_settings(openai_base_url="https://api.deepseek.com"),
        transport=httpx.MockTransport(handler),
    )

    result = asyncio.run(
        client.complete_model("system", "user", StructuredProbe)
    )
    asyncio.run(client.close())

    assert result == {"answer": 7, "ok": True}
    assert len(request_bodies) == 1
    assert request_bodies[0]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize(
    "content",
    [
        '```json\n{"answer": 7}\n```',
        '  ```JSON\r\n{"answer": 7}\r\n```  ',
        '```\n{"answer": 7}\n```',
    ],
)
def test_parse_json_content_accepts_optional_code_fence(content):
    assert OpenAICompatibleClient.parse_json_content(content) == {"answer": 7}


@pytest.mark.parametrize("unsupported_status", [400, 422])
def test_complete_json_retries_without_response_format(
    unsupported_status,
):
    request_bodies = []

    def handler(request):
        request_bodies.append(json.loads(request.content))
        if len(request_bodies) == 1:
            return httpx.Response(
                unsupported_status,
                json={"error": "unsupported response_format"},
            )
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": '{"answer": "ok"}'}}]},
        )

    client = OpenAICompatibleClient(
        configured_settings(),
        transport=httpx.MockTransport(handler),
    )
    result = asyncio.run(client.complete_json("system", "user"))
    asyncio.run(client.close())

    assert result == {"answer": "ok"}
    assert len(request_bodies) == 2
    assert request_bodies[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in request_bodies[1]


@pytest.mark.parametrize("second_status", [400, 422])
def test_complete_json_stops_after_one_fallback_attempt(second_status):
    request_bodies = []

    def handler(request):
        request_bodies.append(json.loads(request.content))
        status_code = 400 if len(request_bodies) == 1 else second_status
        return httpx.Response(
            status_code,
            text="provider body that must not leak",
        )

    client = OpenAICompatibleClient(
        configured_settings(),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(
        ModelResponseError,
        match=f"HTTP status {second_status}",
    ) as exc_info:
        asyncio.run(client.complete_json("private system", "private user"))
    asyncio.run(client.close())

    assert len(request_bodies) == 2
    assert "response_format" in request_bodies[0]
    assert "response_format" not in request_bodies[1]
    assert "provider body" not in str(exc_info.value)
    assert "private system" not in str(exc_info.value)
    assert "private user" not in str(exc_info.value)


def test_live_smoke_validates_environment_before_network_access():
    repository_root = Path(__file__).resolve().parents[1]
    script = repository_root / "scripts" / "smoke_live.py"
    environment = os.environ.copy()
    for name in (
        "OPENAI_BASE_URL",
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_TIMEOUT_SECONDS",
        "TTS_BASE_URL",
        "TTS_API_KEY",
        "TTS_MODEL",
        "TTS_VOICE",
    ):
        environment.pop(name, None)

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert "缺少环境变量" in result.stderr
    assert "OPENAI_API_KEY" in result.stderr
    assert "TTS_MODEL" in result.stderr
    assert "Traceback" not in result.stderr


def test_live_smoke_sanitizes_invalid_timeout_configuration():
    repository_root = Path(__file__).resolve().parents[1]
    script = repository_root / "scripts" / "smoke_live.py"
    environment = os.environ.copy()
    environment.update(
        {
            "OPENAI_BASE_URL": "https://model.example/v1",
            "OPENAI_API_KEY": "model-secret",
            "OPENAI_MODEL": "demo-model",
            "OPENAI_TIMEOUT_SECONDS": "private-invalid-timeout",
            "TTS_BASE_URL": "https://speech.example/v1",
            "TTS_API_KEY": "speech-secret",
            "TTS_MODEL": "demo-tts",
            "TTS_VOICE": "demo-voice",
        }
    )

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 1
    assert result.stdout == ""
    assert result.stderr.strip() == (
        "OPENAI_TIMEOUT_SECONDS 配置无效。未发起网络请求。"
    )
    assert "private-invalid-timeout" not in result.stderr
    assert "Traceback" not in result.stderr


def test_parse_json_content_rejects_malformed_json_safely():
    with pytest.raises(ModelResponseError, match="valid JSON") as exc_info:
        OpenAICompatibleClient.parse_json_content('{"answer":')

    assert "JSONDecodeError" not in str(exc_info.value)


def test_invalid_json_error_atomically_carries_response_usage():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"answer":'}}],
                "usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 2,
                    "total_tokens": 9,
                },
            },
        )
    )
    client = OpenAICompatibleClient(configured_settings(), transport=transport)

    with pytest.raises(ModelStructureError) as captured:
        asyncio.run(client.complete_json("system", "user"))

    asyncio.run(client.close())
    assert captured.value.code == "invalid_json"
    assert captured.value.token_usage == {
        "prompt_tokens": 7,
        "completion_tokens": 2,
        "total_tokens": 9,
    }

@pytest.mark.parametrize("content", ['["answer"]', '"answer"', "42", "null"])
def test_parse_json_content_rejects_non_object_json(content):
    with pytest.raises(ModelResponseError, match="JSON object"):
        OpenAICompatibleClient.parse_json_content(content)


@pytest.mark.parametrize(
    "content",
    [
        None,
        "",
        "   ",
        "```json\n{\"answer\": 1}\n``` trailing",
        "prefix ```json\n{\"answer\": 1}\n```",
    ],
)
def test_parse_json_content_rejects_missing_content_and_fence_junk(content):
    with pytest.raises(ModelResponseError):
        OpenAICompatibleClient.parse_json_content(content)


@pytest.mark.parametrize(
    "response_json",
    [
        {},
        {"choices": []},
        {"choices": [{}]},
        {"choices": [{"message": {}}]},
        {"choices": [{"message": {"content": None}}]},
    ],
)
def test_complete_json_rejects_missing_choices_or_content(response_json):
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json=response_json)
    )
    client = OpenAICompatibleClient(configured_settings(), transport=transport)

    with pytest.raises(ModelResponseError, match="content"):
        asyncio.run(client.complete_json("system", "user"))

    asyncio.run(client.close())


def test_non_fallback_http_error_is_safe():
    response_body = "provider-internal-secret"
    api_key = "api-key-that-must-not-leak"
    transport = httpx.MockTransport(
        lambda request: httpx.Response(503, text=response_body)
    )
    client = OpenAICompatibleClient(
        configured_settings(openai_api_key=api_key),
        transport=transport,
    )

    with pytest.raises(ModelResponseError) as exc_info:
        asyncio.run(client.complete_json("private system", "private user"))

    asyncio.run(client.close())
    message = str(exc_info.value)
    assert "503" in message
    assert response_body not in message
    assert api_key not in message
    assert "private system" not in message
    assert "private user" not in message


def test_missing_model_configuration_fails_without_request():
    request_count = 0

    def handler(request):
        nonlocal request_count
        request_count += 1
        return httpx.Response(200, json={})

    client = OpenAICompatibleClient(
        configured_settings(openai_api_key=None, openai_model=None),
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(ModelResponseError) as exc_info:
        asyncio.run(client.complete_json("system", "user"))

    asyncio.run(client.close())
    message = str(exc_info.value)
    assert "OPENAI_API_KEY" in message
    assert "OPENAI_MODEL" in message
    assert request_count == 0


def test_close_closes_async_client():
    client = OpenAICompatibleClient(
        configured_settings(),
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={})
        ),
    )

    asyncio.run(client.close())

    assert client._client.is_closed
