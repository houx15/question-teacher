import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api import run_generation
from app.config import Settings
from app.main import create_app
from app.schemas import ProblemInput, RuntimeBeat, RuntimeLesson
from app.store import MemoryStore


def problem_input() -> ProblemInput:
    return ProblemInput(
        problem_text="2x+3=7",
        reference_answer="x=2",
        lesson_length="standard",
    )


def runtime_lesson(problem: ProblemInput) -> RuntimeLesson:
    return RuntimeLesson(
        lesson_id="lesson-1",
        problem=problem,
        title="测试课程",
        learning_goal="学会解方程",
        beats=[
            RuntimeBeat(
                beat_id="beat-1",
                purpose="建立等式平衡意识",
                narration="先观察等号两边。",
                board_actions=[],
                layer="base",
            )
        ],
        summary="完成",
        transfer_item={
            "problem_text": "2x=4",
            "expected_answer": "x=2",
            "method_signal": "保持等式平衡",
        },
        validation_report={"math_status": "verified"},
    )


class FakeGenerator:
    def __init__(self, error=None):
        self.error = error
        self.calls = 0

    async def generate(self, problem, on_stage=None):
        self.calls += 1
        if self.error is not None:
            raise self.error
        for stage in (
            "正在验证数学路线",
            "正在设计完整讲解",
            "正在进行整篇审稿",
            "正在修订完整讲解",
            "正在编译课堂",
        ):
            if on_stage:
                on_stage(stage)
        return runtime_lesson(problem)


class FakeAudioService:
    def __init__(self):
        self.calls = 0

    async def attach_audio(self, lesson, on_stage=None):
        self.calls += 1
        if on_stage:
            on_stage("正在生成讲解语音")
        return lesson


class RecordingStore(MemoryStore):
    def __init__(self):
        super().__init__()
        self.seen_stages = []

    def update_job(self, job_id, **changes):
        job = super().update_job(job_id, **changes)
        if "stage" in changes:
            self.seen_stages.append(job.stage)
        return job


def build_client(**overrides):
    generator = overrides.pop("generator", FakeGenerator())
    audio_service = overrides.pop("audio_service", FakeAudioService())
    app = create_app(
        generator=generator,
        audio_service=audio_service,
        **overrides,
    )
    return TestClient(app), generator, audio_service


def test_memory_store_revalidates_updates_without_partial_mutation():
    store = MemoryStore()
    job = store.create_job()

    with pytest.raises(ValidationError, match="completed jobs require lesson_id"):
        store.update_job(job.job_id, status="completed")

    assert store.get_job(job.job_id).status == "queued"


def test_memory_store_supports_concurrent_job_creation():
    store = MemoryStore()

    with ThreadPoolExecutor(max_workers=8) as executor:
        jobs = list(executor.map(lambda _: store.create_job(), range(40)))

    assert len({job.job_id for job in jobs}) == 40
    assert all(store.get_job(job.job_id) is not None for job in jobs)


def test_health_reports_configuration_without_exposing_secrets():
    settings = Settings(
        openai_base_url="https://model.example/v1",
        openai_api_key="model-secret",
        openai_model="demo-model",
        tts_base_url="https://speech.example/v1",
        tts_api_key="speech-secret",
        tts_model="demo-voice-model",
        tts_voice="demo-voice",
    )
    client, _, _ = build_client(settings=settings)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_configured": True,
        "voice_configured": True,
    }
    assert "model-secret" not in response.text
    assert "speech-secret" not in response.text


def test_generation_job_completes_with_public_stage_sequence():
    store = RecordingStore()
    client, generator, audio_service = build_client(store=store)

    response = client.post(
        "/api/lessons/generate",
        json=problem_input().model_dump(),
    )

    assert response.status_code == 202
    assert response.json() == {"job_id": response.json()["job_id"]}
    job_id = response.json()["job_id"]
    job = client.get(f"/api/jobs/{job_id}")
    assert job.status_code == 200
    assert job.json()["status"] == "completed"
    assert job.json()["stage"] == "已完成"
    lesson = client.get(f"/api/lessons/{job.json()['lesson_id']}")
    assert lesson.status_code == 200
    assert lesson.json()["lesson_id"] == "lesson-1"
    assert generator.calls == 1
    assert audio_service.calls == 1
    assert store.seen_stages == [
        "正在理解题目",
        "正在验证数学路线",
        "正在设计完整讲解",
        "正在进行整篇审稿",
        "正在修订并编译课堂",
        "正在生成讲解语音",
        "已完成",
    ]


def test_generation_failure_is_sanitized_and_has_no_lesson():
    private_error = RuntimeError(
        "provider body, api-key=secret, system prompt=private"
    )
    client, _, _ = build_client(generator=FakeGenerator(private_error))

    response = client.post(
        "/api/lessons/generate",
        json=problem_input().model_dump(),
    )

    assert response.status_code == 202
    job = client.get(f"/api/jobs/{response.json()['job_id']}").json()
    assert job["status"] == "failed"
    assert job["stage"] == "生成失败"
    assert job["error"] == "课程生成失败，请稍后重试。"
    assert "secret" not in str(job)
    assert "private" not in str(job)
    assert client.get("/api/lessons/lesson-1").status_code == 404


def test_missing_job_and_lesson_return_404():
    client, _, _ = build_client()

    job_response = client.get("/api/jobs/missing")
    lesson_response = client.get("/api/lessons/missing")

    assert job_response.status_code == 404
    assert lesson_response.status_code == 404


@pytest.mark.parametrize("kind", ["choice", "point_select"])
def test_choice_and_point_select_use_trimmed_exact_comparison(kind):
    client, _, _ = build_client()

    correct = client.post(
        "/api/interactions/evaluate",
        json={"kind": kind, "answer": " A ", "expected": "A"},
    )
    incorrect = client.post(
        "/api/interactions/evaluate",
        json={"kind": kind, "answer": "a", "expected": "A"},
    )

    assert correct.json() == {"classification": "correct"}
    assert incorrect.json() == {"classification": "incorrect"}


def test_expression_interaction_uses_math_equivalence():
    client, _, _ = build_client()

    equivalent = client.post(
        "/api/interactions/evaluate",
        json={
            "kind": "expression",
            "answer": "x^2-6x+9",
            "expected": "(x-3)^2",
        },
    )
    malformed = client.post(
        "/api/interactions/evaluate",
        json={
            "kind": "expression",
            "answer": "not math",
            "expected": "(x-3)^2",
        },
    )

    assert equivalent.json() == {"classification": "correct"}
    assert malformed.json() == {"classification": "incorrect"}


def test_transfer_interaction_uses_answer_equivalence():
    client, _, _ = build_client()

    response = client.post(
        "/api/interactions/evaluate",
        json={
            "kind": "transfer",
            "answer": "x=4/2",
            "expected": "x=2",
        },
    )

    assert response.json() == {"classification": "correct"}


def test_free_text_always_needs_review_without_calling_generator():
    client, generator, _ = build_client()

    response = client.post(
        "/api/interactions/evaluate",
        json={
            "kind": "free_text",
            "answer": "我的思路包含 private student text",
            "expected": "任意参考答案",
        },
    )

    assert response.json() == {
        "classification": "needs_review",
        "message": "首版暂不自动判错这类文字回答。",
    }
    assert generator.calls == 0


def test_run_generation_records_failed_job_without_raising():
    store = MemoryStore()
    job = store.create_job()

    asyncio.run(
        run_generation(
            job.job_id,
            problem_input(),
            store,
            FakeGenerator(RuntimeError("private")),
            FakeAudioService(),
        )
    )

    assert store.get_job(job.job_id).status == "failed"


def test_lifespan_does_not_close_injected_services():
    class ClosableGenerator(FakeGenerator):
        def __init__(self):
            super().__init__()
            self.close_calls = 0

        async def close(self):
            self.close_calls += 1

    class ClosableAudioService(FakeAudioService):
        def __init__(self):
            super().__init__()
            self.close_calls = 0

        async def close(self):
            self.close_calls += 1

    generator = ClosableGenerator()
    audio_service = ClosableAudioService()
    app = create_app(generator=generator, audio_service=audio_service)

    with TestClient(app) as client:
        assert client.get("/api/health").status_code == 200

    assert generator.close_calls == 0
    assert audio_service.close_calls == 0


def test_production_lifespan_constructs_services_and_closes_owned_clients():
    application = create_app(settings=Settings())

    with TestClient(application):
        services = application.state.services
        model_client = services.generator.client
        speech_client = services.audio_service.client
        assert model_client._client.is_closed is False
        assert speech_client.http.is_closed is False

    assert model_client._client.is_closed is True
    assert speech_client.http.is_closed is True


def test_static_and_audio_mounts_and_page_routes_are_registered():
    app = create_app(
        generator=FakeGenerator(),
        audio_service=FakeAudioService(),
    )
    paths = {route.path for route in app.routes}

    assert "/static" in paths
    assert "/audio" in paths
    assert "/" in paths
    assert "/lesson/{lesson_id}" in paths
