import asyncio
import base64
import importlib
import json
import uuid

import httpx
import pytest

from app.config import Settings
from app.tts_client import SpeechGenerationError


class ChunkStream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


def client_class():
    return importlib.import_module(
        "app.volcengine_tts_client"
    ).VolcengineSpeechClient


def volcengine_settings(**overrides):
    values = {
        "openai_api_key": "model-secret-must-not-leak",
        "openai_timeout_seconds": 12.5,
        "tts_provider": "volcengine",
        "volcengine_tts_endpoint": (
            "https://openspeech.bytedance.com/api/v3/tts/unidirectional"
        ),
        "volcengine_tts_api_key": "voice-secret",
        "volcengine_tts_resource_id": "seed-tts-2.0",
        "volcengine_tts_voice": "teacher-voice",
        "volcengine_tts_speed_ratio": 1.2,
        "volcengine_tts_sample_rate": 24000,
        "volcengine_tts_uid": "demo-user",
    }
    values.update(overrides)
    return Settings(**values)


def run(coroutine):
    return asyncio.run(coroutine)


def response_frames(*frames):
    return "".join(json.dumps(frame) for frame in frames).encode()


def test_v3_client_decodes_split_concatenated_frames_and_builds_request():
    requests = []
    first = base64.b64encode(b"first-").decode()
    second = base64.b64encode(b"second").decode()
    payload = b" \n ".join(
        json.dumps(frame).encode()
        for frame in (
            {"code": 0, "data": first},
            {"code": 0, "data": second},
            {"code": 20000000},
        )
    )
    chunks = [payload[:7], payload[7:23], payload[23:51], payload[51:]]

    def handler(request):
        requests.append(request)
        return httpx.Response(200, stream=ChunkStream(chunks))

    async def scenario():
        client = client_class()(
            volcengine_settings(
                volcengine_tts_app_id="unused-legacy-app",
                volcengine_tts_access_key="unused-legacy-access",
            ),
            transport=httpx.MockTransport(handler),
        )
        try:
            return await client.synthesize("先看等式两边。")
        finally:
            await client.close()

    assert run(scenario()) == b"first-second"
    request = requests[0]
    assert str(request.url) == (
        "https://openspeech.bytedance.com/api/v3/tts/unidirectional"
    )
    assert request.headers["X-Api-Key"] == "voice-secret"
    assert "X-Api-App-Id" not in request.headers
    assert "X-Api-Access-Key" not in request.headers
    assert request.headers["X-Api-Resource-Id"] == "seed-tts-2.0"
    uuid.UUID(request.headers["X-Api-Request-Id"])
    assert "Authorization" not in request.headers
    assert "model-secret-must-not-leak" not in str(request.headers)
    assert json.loads(request.content) == {
        "user": {"uid": "demo-user"},
        "req_params": {
            "text": "先看等式两边。",
            "speaker": "teacher-voice",
            "audio_params": {"format": "mp3", "sample_rate": 24000},
            "speed_ratio": 1.2,
        },
    }


def test_v3_client_uses_legacy_header_pair_when_new_key_is_absent():
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(
            200,
            stream=ChunkStream(
                [
                    response_frames(
                        {
                            "code": 0,
                            "data": base64.b64encode(b"audio").decode(),
                        },
                        {"code": 20000000},
                    )
                ]
            ),
        )

    async def scenario():
        client = client_class()(
            volcengine_settings(
                volcengine_tts_api_key=None,
                volcengine_tts_app_id="legacy-app",
                volcengine_tts_access_key="legacy-access",
            ),
            transport=httpx.MockTransport(handler),
        )
        try:
            await client.synthesize("有效讲解")
        finally:
            await client.close()

    run(scenario())

    assert requests[0].headers["X-Api-App-Id"] == "legacy-app"
    assert requests[0].headers["X-Api-Access-Key"] == "legacy-access"
    assert "X-Api-Key" not in requests[0].headers


def test_v3_client_generates_a_unique_request_id_for_every_call():
    request_ids = []

    def handler(request):
        request_ids.append(request.headers["X-Api-Request-Id"])
        return httpx.Response(
            200,
            stream=ChunkStream(
                [
                    response_frames(
                        {
                            "code": 0,
                            "data": base64.b64encode(b"audio").decode(),
                        },
                        {"code": 20000000},
                    )
                ]
            ),
        )

    async def scenario():
        client = client_class()(
            volcengine_settings(),
            transport=httpx.MockTransport(handler),
        )
        try:
            await client.synthesize("第一段")
            await client.synthesize("第二段")
        finally:
            await client.close()

    run(scenario())

    assert len(set(request_ids)) == 2


@pytest.mark.parametrize(
    "settings",
    [
        volcengine_settings(volcengine_tts_api_key=None),
        volcengine_settings(volcengine_tts_resource_id=None),
        volcengine_settings(volcengine_tts_voice=None),
        volcengine_settings(volcengine_tts_endpoint=None),
    ],
)
def test_v3_client_rejects_incomplete_configuration_before_request(settings):
    def unexpected_request(_request):
        raise AssertionError("request must not be sent")

    async def scenario():
        client = client_class()(
            settings,
            transport=httpx.MockTransport(unexpected_request),
        )
        try:
            with pytest.raises(SpeechGenerationError, match="not configured"):
                await client.synthesize("有效讲解")
        finally:
            await client.close()

    run(scenario())


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (b'{"code":0', "invalid response"),
        (
            response_frames(
                {"code": 0, "data": "%%%"},
                {"code": 20000000},
            ),
            "invalid response",
        ),
        (
            response_frames(
                {
                    "code": 0,
                    "data": base64.b64encode(b"audio").decode(),
                }
            ),
            "missing completion",
        ),
        (response_frames({"code": 20000000}), "empty audio"),
    ],
)
def test_v3_client_rejects_invalid_or_incomplete_stream(body, expected):
    async def scenario():
        client = client_class()(
            volcengine_settings(),
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200,
                    stream=ChunkStream([body]),
                )
            ),
        )
        try:
            with pytest.raises(SpeechGenerationError, match=expected):
                await client.synthesize("有效讲解")
        finally:
            await client.close()

    run(scenario())


def test_v3_client_vendor_error_exposes_code_only():
    private_body = {
        "code": 55000031,
        "message": "voice-secret 私密讲稿",
    }

    async def scenario():
        client = client_class()(
            volcengine_settings(),
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    200,
                    stream=ChunkStream([response_frames(private_body)]),
                )
            ),
        )
        try:
            with pytest.raises(SpeechGenerationError) as error:
                await client.synthesize("私密讲稿")
            return str(error.value)
        finally:
            await client.close()

    message = run(scenario())

    assert message == "Speech generation failed with vendor code 55000031"
    assert private_body["message"] not in message


def test_v3_client_http_error_exposes_status_only():
    async def scenario():
        client = client_class()(
            volcengine_settings(),
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    403,
                    content=b"voice-secret private response",
                )
            ),
        )
        try:
            with pytest.raises(SpeechGenerationError) as error:
                await client.synthesize("私密讲稿")
            return str(error.value)
        finally:
            await client.close()

    assert run(scenario()) == "Speech generation failed with status 403"


def test_live_smoke_constructs_and_closes_selected_speech_provider():
    smoke_live = importlib.import_module("scripts.smoke_live")
    client = smoke_live.create_speech_client(volcengine_settings())

    assert client.__class__.__name__ == "VolcengineSpeechClient"
    assert client.http.is_closed is False

    run(client.close())

    assert client.http.is_closed is True
