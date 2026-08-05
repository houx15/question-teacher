import asyncio
import json

import httpx
import pytest

from app.config import Settings
from app.llm_client import ModelResponseError, OpenAICompatibleClient


def configured_settings(**overrides):
    values = {
        "openai_base_url": "https://model.example/v1",
        "openai_api_key": "test-secret-key",
        "openai_model": "demo-model",
        "openai_timeout_seconds": 12.5,
    }
    values.update(overrides)
    return Settings(**values)


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


def test_complete_json_retries_400_without_response_format():
    request_bodies = []

    def handler(request):
        request_bodies.append(json.loads(request.content))
        if len(request_bodies) == 1:
            return httpx.Response(400, json={"error": "unsupported response_format"})
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


def test_parse_json_content_rejects_malformed_json_safely():
    with pytest.raises(ModelResponseError, match="valid JSON") as exc_info:
        OpenAICompatibleClient.parse_json_content('{"answer":')

    assert "JSONDecodeError" not in str(exc_info.value)


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
