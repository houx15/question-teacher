import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from app.api import (
    _PUBLIC_GENERATION_STAGES,
    run_generation,
    safe_generation_error,
)
from app.compiler import LessonCompiler
from app.config import Settings
from app.generation import LessonInputError
from app.main import create_app
from app.schemas import (
    Interaction,
    InteractionOption,
    LessonDraft,
    ProblemInput,
    RuntimeBeat,
    RuntimeLesson,
    TransferItem,
    TransferOption,
)
from app.store import MemoryStore
from tests.test_generation import problem, valid_draft


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


def save_interaction_lesson(
    store,
    *,
    kind,
    expected,
    options=None,
):
    lesson = runtime_lesson(problem_input())
    interaction = Interaction(
        interaction_id="interaction-1",
        kind=kind,
        prompt="请作答。",
        expected_answer=expected,
        options=options or [],
    )
    beat = lesson.beats[0].model_copy(
        update={"interaction": interaction}
    )
    lesson = lesson.model_copy(update={"beats": [beat]})
    store.save_lesson(lesson)
    return lesson, interaction


class FakeGenerator:
    def __init__(self, error=None):
        self.error = error
        self.calls = 0

    async def generate(self, problem, on_stage=None):
        self.calls += 1
        if self.error is not None:
            raise self.error
        stages = ["正在验证数学路线"]
        if problem.reference_solution_text is not None:
            stages.append("正在审阅参考解析")
        stages.extend(
            [
                "正在设计完整讲解",
                "正在进行整篇审稿",
                "正在修订完整讲解",
                "正在编译课堂",
            ]
        )
        for stage in stages:
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


def test_internal_route_planning_reuses_existing_public_math_stage():
    assert _PUBLIC_GENERATION_STAGES["正在规划数学路线"] == (
        "正在验证数学路线"
    )


def test_generation_with_reference_solution_exposes_audit_stage():
    store = RecordingStore()
    client, _, _ = build_client(store=store)
    payload = problem_input().model_copy(
        update={
            "reference_solution_text": (
                "解：2x+3=7，所以 2x=4，最终 x=2。"
            )
        }
    )

    response = client.post(
        "/api/lessons/generate",
        json=payload.model_dump(),
    )

    assert response.status_code == 202
    assert "正在审阅参考解析" in store.seen_stages


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


def test_generation_failure_exposes_only_typed_safe_input_errors():
    public_message = "参考解析与题目或参考答案存在数学冲突，请检查后再试。"

    assert safe_generation_error(LessonInputError(public_message)) == (
        public_message
    )
    assert safe_generation_error(RuntimeError("private-provider-detail")) == (
        "课程生成失败，请稍后重试。"
    )


def test_missing_job_and_lesson_return_404():
    client, _, _ = build_client()

    job_response = client.get("/api/jobs/missing")
    lesson_response = client.get("/api/lessons/missing")

    assert job_response.status_code == 404
    assert lesson_response.status_code == 404


def test_public_lesson_payload_redacts_answers_and_review_internals():
    lesson = RuntimeLesson.model_validate(
        {
            "lesson_id": "lesson-public",
            "problem": {
                "problem_text": "解方程 x^2-6x+9=0",
                "reference_answer": "x=271828",
                "reference_solution_text": (
                    "解：这是不应进入学生课堂的内部参考解析。"
                ),
            },
            "title": "公开课程",
            "learning_goal": "理解完全平方结构。",
            "beats": [
                {
                    "beat_id": "beat-expression",
                    "purpose": "识别结构",
                    "narration": "请写出对应的完全平方。",
                    "board_actions": [],
                    "layer": "interaction",
                    "audio_url": "/audio/lesson-public/beat-expression.mp3",
                    "interaction": {
                        "interaction_id": "expression-check",
                        "kind": "expression",
                        "prompt": "写出完全平方。",
                        "expected_answer": "x^2-6*x+9",
                        "hints": ["观察中间项。"],
                        "hint_audio_urls": [
                            "/audio/lesson-public/hint-1.mp3",
                        ],
                        "correct_audio_url": (
                            "/audio/lesson-public/correct.mp3"
                        ),
                    },
                    "next_beat_id": "beat-choice",
                },
                {
                    "beat_id": "beat-choice",
                    "purpose": "选择方法",
                    "narration": "请选择下一步。",
                    "board_actions": [],
                    "layer": "interaction",
                    "interaction": {
                        "interaction_id": "choice-check",
                        "kind": "choice",
                        "prompt": "选择方法。",
                        "expected_answer": "correct-option",
                        "options": [
                            {
                                "option_id": "correct-option",
                                "label": "配方法",
                            },
                            {
                                "option_id": "other-option",
                                "label": "其他方法",
                                "feedback": "这个式子已经是完全平方。",
                                "feedback_audio_url": (
                                    "/audio/lesson-public/other-option.mp3"
                                ),
                            },
                        ],
                    },
                },
            ],
            "summary": "识别完全平方。",
            "transfer_item": {
                "problem_text": "解方程 x^2-8x+16=0",
                "expected_answer": "x=314159",
                "method_signal": "观察完全平方。",
            },
            "validation_report": {
                "independent_solutions": ["validation-secret"],
                "review_assessment": "private-review",
            },
        }
    )
    store = MemoryStore()
    store.save_lesson(lesson)
    client, _, _ = build_client(store=store)

    response = client.get("/api/lessons/lesson-public")

    assert response.status_code == 200
    payload = response.json()
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "reference_answer" not in payload["problem"]
    assert "reference_solution_text" not in payload["problem"]
    assert "expected_answer" not in payload["transfer_item"]
    assert "validation_report" not in payload
    assert all(
        "expected_answer" not in beat["interaction"]
        for beat in payload["beats"]
    )
    assert "x=271828" not in serialized
    assert "不应进入学生课堂" not in serialized
    assert "x^2-6*x+9" not in serialized
    assert "x=314159" not in serialized
    assert "validation-secret" not in serialized
    assert "private-review" not in serialized
    assert payload["problem"]["problem_text"] == "解方程 x^2-6x+9=0"
    assert payload["beats"][0]["audio_url"].endswith(
        "beat-expression.mp3"
    )
    assert payload["beats"][0]["interaction"]["hints"] == [
        "观察中间项。"
    ]
    assert payload["beats"][0]["interaction"]["hint_audio_urls"]
    assert payload["beats"][0]["interaction"]["correct_audio_url"]
    assert payload["beats"][1]["interaction"]["options"] == [
        {"option_id": "correct-option", "label": "配方法"},
        {
            "option_id": "other-option",
            "label": "其他方法",
            "feedback": "这个式子已经是完全平方。",
            "feedback_audio_url": (
                "/audio/lesson-public/other-option.mp3"
            ),
        },
    ]

    evaluation = client.post(
        "/api/interactions/evaluate",
        json={
            "lesson_id": "lesson-public",
            "interaction_id": "expression-check",
            "answer": "(x-3)^2",
        },
    )
    assert evaluation.json() == {"classification": "correct"}


def test_public_lesson_payload_redacts_diagnostic_transfer_answers():
    lesson = runtime_lesson(problem_input()).model_copy(
        update={
            "transfer_item": TransferItem(
                problem_text="用因式分解法解方程：x^2-7x+12=0",
                expected_answer="x=3 或 x=4",
                method_signal="寻找乘积为 12、和为 -7 的两个数。",
                options=[
                    TransferOption(
                        option_id="both-roots",
                        label="x=3 或 x=4",
                        canonical_answer="x=3 或 x=4",
                        feedback="两个根都能使原方程成立。",
                    ),
                    TransferOption(
                        option_id="only-three",
                        label="x=3",
                        canonical_answer="x=3",
                        feedback="还遗漏了另一个根。",
                    ),
                    TransferOption(
                        option_id="only-four",
                        label="x=4",
                        canonical_answer="x=4",
                        feedback="还遗漏了另一个根。",
                    ),
                ],
                correct_option_id="both-roots",
            )
        }
    )
    store = MemoryStore()
    store.save_lesson(lesson)
    client, _, _ = build_client(store=store)

    response = client.get(f"/api/lessons/{lesson.lesson_id}")

    assert response.status_code == 200
    transfer_item = response.json()["transfer_item"]
    assert "expected_answer" not in transfer_item
    assert "correct_option_id" not in transfer_item
    assert all(
        "canonical_answer" not in option
        for option in transfer_item["options"]
    )
    assert transfer_item["options"] == [
        {
            "option_id": "both-roots",
            "label": "x=3 或 x=4",
            "feedback": "两个根都能使原方程成立。",
        },
        {
            "option_id": "only-three",
            "label": "x=3",
            "feedback": "还遗漏了另一个根。",
        },
        {
            "option_id": "only-four",
            "label": "x=4",
            "feedback": "还遗漏了另一个根。",
        },
    ]


def test_public_compiled_diagnostic_transfer_hides_answer_key():
    draft = valid_draft()
    lesson = LessonCompiler(
        lesson_id_factory=lambda: "lesson-compiled-choice"
    ).compile(
        problem(),
        LessonDraft.model_validate(draft),
        {"review_status": "approved"},
    )
    store = MemoryStore()
    store.save_lesson(lesson)
    client, _, _ = build_client(store=store)

    response = client.get(f"/api/lessons/{lesson.lesson_id}")

    assert response.status_code == 200
    transfer = response.json()["beats"][-1]["interaction"]
    assert transfer["kind"] == "choice"
    assert "expected_answer" not in transfer
    assert transfer["options"] == [
        {
            "option_id": option["option_id"],
            "label": option["label"],
            "feedback": option["feedback"],
        }
        for option in draft["transfer_item"]["options"]
    ]


@pytest.mark.parametrize("kind", ["choice", "point_select"])
def test_choice_and_point_select_use_trimmed_exact_comparison(kind):
    store = MemoryStore()
    options = (
        [InteractionOption(option_id="A", label="选项 A")]
        if kind == "choice"
        else []
    )
    lesson, interaction = save_interaction_lesson(
        store,
        kind=kind,
        expected="A",
        options=options,
    )
    client, _, _ = build_client(store=store)

    correct = client.post(
        "/api/interactions/evaluate",
        json={
            "lesson_id": lesson.lesson_id,
            "interaction_id": interaction.interaction_id,
            "answer": " A ",
        },
    )
    incorrect = client.post(
        "/api/interactions/evaluate",
        json={
            "lesson_id": lesson.lesson_id,
            "interaction_id": interaction.interaction_id,
            "answer": "a",
        },
    )

    assert correct.json() == {"classification": "correct"}
    assert incorrect.json() == {"classification": "incorrect"}


def test_expression_interaction_uses_math_equivalence():
    store = MemoryStore()
    lesson, interaction = save_interaction_lesson(
        store,
        kind="expression",
        expected="(x-3)^2",
    )
    client, _, _ = build_client(store=store)

    equivalent = client.post(
        "/api/interactions/evaluate",
        json={
            "lesson_id": lesson.lesson_id,
            "interaction_id": interaction.interaction_id,
            "answer": "x^2-6x+9",
        },
    )
    malformed = client.post(
        "/api/interactions/evaluate",
        json={
            "lesson_id": lesson.lesson_id,
            "interaction_id": interaction.interaction_id,
            "answer": "not math",
        },
    )

    assert equivalent.json() == {"classification": "correct"}
    assert malformed.json() == {"classification": "incorrect"}


def test_transfer_interaction_uses_answer_equivalence():
    store = MemoryStore()
    lesson, interaction = save_interaction_lesson(
        store,
        kind="transfer",
        expected="x=2",
    )
    client, _, _ = build_client(store=store)

    response = client.post(
        "/api/interactions/evaluate",
        json={
            "lesson_id": lesson.lesson_id,
            "interaction_id": interaction.interaction_id,
            "answer": "x=4/2",
        },
    )

    assert response.json() == {"classification": "correct"}


def test_free_text_always_needs_review_without_calling_generator():
    store = MemoryStore()
    lesson, interaction = save_interaction_lesson(
        store,
        kind="free_text",
        expected="等式两边做相同运算。",
    )
    client, generator, _ = build_client(store=store)

    response = client.post(
        "/api/interactions/evaluate",
        json={
            "lesson_id": lesson.lesson_id,
            "interaction_id": interaction.interaction_id,
            "answer": "我的思路包含 private student text",
        },
    )

    assert response.json() == {
        "classification": "needs_review",
        "message": "首版暂不自动判错这类文字回答。",
    }
    assert generator.calls == 0


def test_interaction_evaluation_rejects_forged_expected_answer():
    store = MemoryStore()
    lesson, interaction = save_interaction_lesson(
        store,
        kind="choice",
        expected="A",
        options=[InteractionOption(option_id="A", label="选项 A")],
    )
    client, _, _ = build_client(store=store)

    forged = client.post(
        "/api/interactions/evaluate",
        json={
            "lesson_id": lesson.lesson_id,
            "interaction_id": interaction.interaction_id,
            "answer": "forged",
            "expected": "forged",
        },
    )
    incorrect = client.post(
        "/api/interactions/evaluate",
        json={
            "lesson_id": lesson.lesson_id,
            "interaction_id": interaction.interaction_id,
            "answer": "forged",
        },
    )

    assert forged.status_code == 422
    assert incorrect.json() == {"classification": "incorrect"}


@pytest.mark.parametrize(
    ("lesson_id", "interaction_id"),
    [
        ("missing", "interaction-1"),
        ("lesson-1", "missing"),
    ],
)
def test_interaction_evaluation_returns_safe_404_for_missing_authority(
    lesson_id,
    interaction_id,
):
    store = MemoryStore()
    save_interaction_lesson(
        store,
        kind="free_text",
        expected="参考答案",
    )
    client, _, _ = build_client(store=store)

    response = client.post(
        "/api/interactions/evaluate",
        json={
            "lesson_id": lesson_id,
            "interaction_id": interaction_id,
            "answer": "学生答案",
        },
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "课程互动不存在。"


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


def test_production_lifespan_selects_and_closes_volcengine_client():
    application = create_app(
        settings=Settings(
            tts_provider="volcengine",
            volcengine_tts_api_key="voice-secret",
            volcengine_tts_resource_id="seed-tts-2.0",
            volcengine_tts_voice="teacher",
        )
    )

    with TestClient(application):
        speech_client = application.state.services.audio_service.client
        assert speech_client.__class__.__name__ == "VolcengineSpeechClient"
        assert speech_client.http.is_closed is False

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
