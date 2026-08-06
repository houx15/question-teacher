import asyncio
import json

import httpx
import pytest

from app.audio_service import LessonAudioService
from app.config import Settings
from app.schemas import (
    Interaction,
    InteractionOption,
    ProblemInput,
    RuntimeBeat,
    RuntimeLesson,
    TransferItem,
    TransferOption,
)
from app.tts_client import OpenAISpeechClient, SpeechGenerationError


def run(coroutine):
    return asyncio.run(coroutine)


def speech_settings(**overrides):
    values = {
        "openai_timeout_seconds": 12.5,
        "tts_base_url": "https://speech.test/v1",
        "tts_api_key": "voice-secret",
        "tts_model": "demo-tts",
        "tts_voice": "teacher",
    }
    values.update(overrides)
    return Settings(**values)


def runtime_lesson(
    *,
    lesson_id="lesson-001",
    first_beat_id="beat-001",
    correct_explanation="对，先把未知数项集中到等号左边。",
):
    interaction = Interaction(
        interaction_id="check-first-step",
        kind="free_text",
        prompt="第一步应该做什么？",
        expected_answer="等式两边同时减一。",
        hints=["观察常数项。", "等式两边要做相同运算。"],
        explanation_after_correct=correct_explanation,
    )
    return RuntimeLesson(
        lesson_id=lesson_id,
        problem=ProblemInput(
            problem_text="解方程 x + 1 = 0",
            reference_answer="x = -1",
        ),
        title="用等式性质解方程",
        learning_goal="能用等式性质隔离未知数。",
        beats=[
            RuntimeBeat(
                beat_id=first_beat_id,
                purpose="建立目标",
                narration="先观察未知数和常数项的位置。",
                board_actions=[],
                layer="base",
            ),
            RuntimeBeat(
                beat_id="beat-002",
                purpose="检查第一步",
                narration="现在判断第一步应该做什么。",
                board_actions=[],
                layer="interaction",
                interaction=interaction,
                next_beat_id="beat-003",
            ),
        ],
        summary="等式两边做相同运算，等式仍然成立。",
        transfer_item=TransferItem(
            problem_text="解方程 x + 2 = 0",
            expected_answer="x = -2",
            method_signal="等式两边同时减二。",
            options=[
                TransferOption(
                    option_id="negative-two",
                    label="x = -2",
                    canonical_answer="x = -2",
                    feedback="对，等式两边同时减二得到 x = -2。",
                ),
                TransferOption(
                    option_id="positive-two",
                    label="x = 2",
                    canonical_answer="x = 2",
                    feedback="减二后符号应为负。",
                ),
                TransferOption(
                    option_id="zero",
                    label="x = 0",
                    canonical_answer="x = 0",
                    feedback="代回原方程会得到 2，不成立。",
                ),
            ],
            correct_option_id="negative-two",
        ),
        validation_report={"math_valid": True},
    )


def choice_runtime_lesson(*, with_feedback=True):
    options = [
        InteractionOption(
            option_id="option/with-path-like-id",
            label="两边同时减一",
            feedback=("对，先消去常数项。" if with_feedback else None),
        ),
        InteractionOption(
            option_id="wrong-sign",
            label="两边同时加一",
            feedback=("这会让常数项更远离零。" if with_feedback else None),
        ),
        InteractionOption(
            option_id="divide-first",
            label="两边同时除以一",
            feedback=("除以一没有改变等式。" if with_feedback else None),
        ),
    ]
    lesson = runtime_lesson()
    interaction = lesson.beats[1].interaction.model_copy(
        update={
            "kind": "choice",
            "expected_answer": options[0].option_id,
            "options": options,
        }
    )
    beats = list(lesson.beats)
    beats[1] = beats[1].model_copy(update={"interaction": interaction})
    return lesson.model_copy(update={"beats": beats})


class FakeSpeechClient:
    def __init__(self, outcomes=None):
        self.outcomes = list(outcomes or [])
        self.texts = []

    async def synthesize(self, text):
        self.texts.append(text)
        if self.outcomes:
            outcome = self.outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        return f"audio:{text}".encode()


class ControlledOptionFeedbackClient:
    def __init__(self, feedbacks):
        self.feedbacks = set(feedbacks)
        self.releases = {
            feedback: asyncio.Event() for feedback in feedbacks
        }
        self.active_feedback_calls = 0
        self.max_active_feedback_calls = 0
        self.started_feedbacks = []
        self.completed_feedbacks = []
        self._condition = asyncio.Condition()

    async def synthesize(self, text):
        if text not in self.feedbacks:
            return f"audio:{text}".encode()

        async with self._condition:
            self.active_feedback_calls += 1
            self.max_active_feedback_calls = max(
                self.max_active_feedback_calls,
                self.active_feedback_calls,
            )
            self.started_feedbacks.append(text)
            self._condition.notify_all()
        try:
            await self.releases[text].wait()
            async with self._condition:
                self.completed_feedbacks.append(text)
                self._condition.notify_all()
            return f"audio:{text}".encode()
        finally:
            async with self._condition:
                self.active_feedback_calls -= 1
                self._condition.notify_all()

    async def wait_for_started_feedbacks(self, count):
        async with self._condition:
            await self._condition.wait_for(
                lambda: len(self.started_feedbacks) >= count
            )

    async def wait_for_completed_feedbacks(self, count):
        async with self._condition:
            await self._condition.wait_for(
                lambda: len(self.completed_feedbacks) >= count
            )


class FailingOptionFeedbackClient:
    def __init__(self, first_feedback, failing_feedback, blocked_feedback):
        self.first_feedback = first_feedback
        self.failing_feedback = failing_feedback
        self.blocked_feedback = blocked_feedback
        self.failure_release = asyncio.Event()
        self.calls = []
        self.cancelled_feedbacks = []
        self.active_feedback_calls = 0
        self._condition = asyncio.Condition()

    async def synthesize(self, text):
        self.calls.append(text)
        if text not in {
            self.first_feedback,
            self.failing_feedback,
            self.blocked_feedback,
        }:
            return f"audio:{text}".encode()

        async with self._condition:
            self.active_feedback_calls += 1
            self._condition.notify_all()
        try:
            if text == self.first_feedback:
                return f"audio:{text}".encode()
            if text == self.failing_feedback:
                await self.failure_release.wait()
                raise SpeechGenerationError("private upstream detail")
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                self.cancelled_feedbacks.append(text)
                raise
        finally:
            async with self._condition:
                self.active_feedback_calls -= 1
                self._condition.notify_all()

    async def wait_for_active_feedback_calls(self, count):
        async with self._condition:
            await self._condition.wait_for(
                lambda: self.active_feedback_calls >= count
            )


def test_speech_client_posts_expected_request():
    async def scenario():
        def handler(request):
            payload = json.loads(request.content)
            assert request.url.path == "/v1/audio/speech"
            assert request.headers["Authorization"] == "Bearer voice-secret"
            assert request.headers["Content-Type"] == "application/json"
            assert payload == {
                "model": "demo-tts",
                "voice": "teacher",
                "input": "先看一次项系数。",
                "response_format": "mp3",
            }
            return httpx.Response(200, content=b"fake-mp3")

        client = OpenAISpeechClient(
            speech_settings(),
            transport=httpx.MockTransport(handler),
        )
        try:
            return await client.synthesize("先看一次项系数。")
        finally:
            await client.close()

    assert run(scenario()) == b"fake-mp3"


@pytest.mark.parametrize(
    "settings",
    [
        speech_settings(tts_api_key=None),
        speech_settings(tts_model=None),
        speech_settings(tts_voice=None),
        speech_settings(tts_base_url=None),
    ],
)
def test_speech_client_rejects_missing_voice_configuration_before_request(
    settings,
):
    def unexpected_request(_request):
        raise AssertionError("request must not be sent")

    async def scenario():
        client = OpenAISpeechClient(
            settings,
            transport=httpx.MockTransport(unexpected_request),
        )
        try:
            with pytest.raises(
                SpeechGenerationError,
                match="not configured",
            ):
                await client.synthesize("有效讲解")
        finally:
            await client.close()

    run(scenario())


@pytest.mark.parametrize(
    "settings",
    [
        speech_settings(tts_api_key=" \t"),
        speech_settings(tts_model="\n"),
        speech_settings(tts_voice=" "),
        speech_settings(tts_base_url="\t "),
    ],
)
def test_speech_client_rejects_whitespace_voice_configuration_before_request(
    settings,
):
    def unexpected_request(_request):
        raise AssertionError("request must not be sent")

    async def scenario():
        client = OpenAISpeechClient(
            settings,
            transport=httpx.MockTransport(unexpected_request),
        )
        try:
            with pytest.raises(
                SpeechGenerationError,
                match="not configured",
            ):
                await client.synthesize("有效讲解")
        finally:
            await client.close()

    run(scenario())


@pytest.mark.parametrize("text", ["", " ", "\t\n"])
def test_speech_client_rejects_blank_text_before_request(text):
    def unexpected_request(_request):
        raise AssertionError("request must not be sent")

    async def scenario():
        client = OpenAISpeechClient(
            speech_settings(),
            transport=httpx.MockTransport(unexpected_request),
        )
        try:
            with pytest.raises(SpeechGenerationError, match="blank"):
                await client.synthesize(text)
        finally:
            await client.close()

    run(scenario())


def test_speech_client_rejects_non_string_text_before_request():
    def unexpected_request(_request):
        raise AssertionError("request must not be sent")

    async def scenario():
        client = OpenAISpeechClient(
            speech_settings(),
            transport=httpx.MockTransport(unexpected_request),
        )
        try:
            with pytest.raises(SpeechGenerationError, match="must be text"):
                await client.synthesize(None)
        finally:
            await client.close()

    run(scenario())


def test_speech_client_http_error_exposes_status_only():
    private_body = "voice-secret 绝不能出现在错误里"

    async def scenario():
        client = OpenAISpeechClient(
            speech_settings(),
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    429,
                    content=private_body.encode(),
                )
            ),
        )
        try:
            with pytest.raises(SpeechGenerationError) as error:
                await client.synthesize("这是私密讲稿")
        finally:
            await client.close()
        return str(error.value)

    message = run(scenario())

    assert message == "Speech generation failed with status 429"
    assert private_body not in message
    assert "voice-secret" not in message
    assert "这是私密讲稿" not in message


def test_speech_client_rejects_redirect_with_status_only():
    private_body = "redirect body must remain private"

    async def scenario():
        client = OpenAISpeechClient(
            speech_settings(),
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(
                    302,
                    headers={"Location": "https://redirect.test"},
                    content=private_body.encode(),
                )
            ),
        )
        try:
            with pytest.raises(SpeechGenerationError) as error:
                await client.synthesize("这是私密讲稿")
        finally:
            await client.close()
        return str(error.value)

    message = run(scenario())

    assert message == "Speech generation failed with status 302"
    assert private_body not in message
    assert "这是私密讲稿" not in message


def test_speech_client_rejects_empty_audio():
    async def scenario():
        client = OpenAISpeechClient(
            speech_settings(),
            transport=httpx.MockTransport(
                lambda _request: httpx.Response(200, content=b"")
            ),
        )
        try:
            with pytest.raises(SpeechGenerationError, match="empty audio"):
                await client.synthesize("有效讲解")
        finally:
            await client.close()

    run(scenario())


def test_audio_service_writes_every_narration_hint_and_feedback(tmp_path):
    lesson = runtime_lesson()
    original_dump = lesson.model_dump()
    client = FakeSpeechClient()

    voiced = run(LessonAudioService(client, tmp_path).attach_audio(lesson))

    assert [beat.audio_url for beat in voiced.beats] == [
        "/audio/lesson-001/beat-001.mp3",
        "/audio/lesson-001/beat-002.mp3",
    ]
    interaction = voiced.beats[1].interaction
    assert interaction is not None
    assert interaction.hint_audio_urls == [
        "/audio/lesson-001/beat-002-hint-1.mp3",
        "/audio/lesson-001/beat-002-hint-2.mp3",
    ]
    assert (
        interaction.correct_audio_url
        == "/audio/lesson-001/beat-002-correct.mp3"
    )
    assert client.texts == [
        "先观察未知数和常数项的位置。",
        "现在判断第一步应该做什么。",
        "观察常数项。",
        "等式两边要做相同运算。",
        "对，先把未知数项集中到等号左边。",
    ]
    for filename, text in [
        ("beat-001.mp3", client.texts[0]),
        ("beat-002.mp3", client.texts[1]),
        ("beat-002-hint-1.mp3", client.texts[2]),
        ("beat-002-hint-2.mp3", client.texts[3]),
        ("beat-002-correct.mp3", client.texts[4]),
    ]:
        assert (tmp_path / "lesson-001" / filename).read_bytes() == (
            f"audio:{text}".encode()
        )
    assert voiced.beats[1].next_beat_id == "beat-003"
    assert voiced.problem == lesson.problem
    assert voiced.validation_report == {"math_valid": True}
    assert lesson.model_dump() == original_dump


def test_audio_service_writes_choice_feedback_with_numeric_asset_ids(tmp_path):
    lesson = choice_runtime_lesson()
    original_dump = lesson.model_dump()
    client = FakeSpeechClient()

    voiced = run(LessonAudioService(client, tmp_path).attach_audio(lesson))

    interaction = voiced.beats[1].interaction
    assert interaction is not None
    assert [option.feedback_audio_url for option in interaction.options] == [
        "/audio/lesson-001/beat-002-option-1.mp3",
        "/audio/lesson-001/beat-002-option-2.mp3",
        "/audio/lesson-001/beat-002-option-3.mp3",
    ]
    assert client.texts[:4] == [
        "先观察未知数和常数项的位置。",
        "现在判断第一步应该做什么。",
        "观察常数项。",
        "等式两边要做相同运算。",
    ]
    assert client.texts[-1] == "对，先把未知数项集中到等号左边。"
    assert set(client.texts[4:-1]) == {
        "对，先消去常数项。",
        "这会让常数项更远离零。",
        "除以一没有改变等式。",
    }
    for index, option in enumerate(interaction.options, start=1):
        assert (
            tmp_path
            / "lesson-001"
            / f"beat-002-option-{index}.mp3"
        ).read_bytes() == f"audio:{option.feedback}".encode()
    assert not (tmp_path / "lesson-001" / "option").exists()
    assert lesson.model_dump() == original_dump


def test_audio_service_limits_choice_feedback_concurrency_and_keeps_order(
    tmp_path,
):
    lesson = choice_runtime_lesson()
    feedbacks = [option.feedback for option in lesson.beats[1].interaction.options]

    async def scenario():
        client = ControlledOptionFeedbackClient(feedbacks)
        task = asyncio.create_task(
            LessonAudioService(client, tmp_path).attach_audio(lesson)
        )
        try:
            await asyncio.wait_for(
                client.wait_for_started_feedbacks(2),
                timeout=0.5,
            )
        except asyncio.TimeoutError:
            for release in client.releases.values():
                release.set()
            await task
            return None, client
        assert client.active_feedback_calls == 2
        client.releases[feedbacks[1]].set()
        await client.wait_for_started_feedbacks(3)
        client.releases[feedbacks[2]].set()
        await client.wait_for_completed_feedbacks(2)
        client.releases[feedbacks[0]].set()
        return await task, client

    voiced, client = run(scenario())

    assert client.max_active_feedback_calls == 2
    assert client.completed_feedbacks == [
        feedbacks[1],
        feedbacks[2],
        feedbacks[0],
    ]
    interaction = voiced.beats[1].interaction
    assert interaction is not None
    assert [option.feedback_audio_url for option in interaction.options] == [
        "/audio/lesson-001/beat-002-option-1.mp3",
        "/audio/lesson-001/beat-002-option-2.mp3",
        "/audio/lesson-001/beat-002-option-3.mp3",
    ]


def test_audio_service_cancels_settles_and_cleans_up_failed_option_audio(
    tmp_path,
):
    lesson = choice_runtime_lesson()
    feedbacks = [option.feedback for option in lesson.beats[1].interaction.options]

    async def scenario():
        client = FailingOptionFeedbackClient(*feedbacks)
        task = asyncio.create_task(
            LessonAudioService(client, tmp_path).attach_audio(lesson)
        )
        try:
            await asyncio.wait_for(
                client.wait_for_active_feedback_calls(2),
                timeout=0.5,
            )
        except asyncio.TimeoutError:
            client.failure_release.set()
            with pytest.raises(SpeechGenerationError) as error:
                await task
            return error.value, client
        while not (tmp_path / "lesson-001" / "beat-002-option-1.mp3").exists():
            await asyncio.sleep(0)
        await client.wait_for_active_feedback_calls(2)
        client.failure_release.set()
        with pytest.raises(SpeechGenerationError) as error:
            await task
        return error.value, client

    error, client = run(scenario())

    assert str(error) == "Audio generation failed for beat-002-option-2"
    assert "private upstream detail" not in str(error)
    assert client.calls.count(feedbacks[1]) == 2
    assert client.cancelled_feedbacks == [feedbacks[2]]
    assert client.active_feedback_calls == 0
    assert not (tmp_path / "lesson-001").exists()
    assert not list(tmp_path.rglob("*.mp3"))


def test_audio_service_skips_legacy_choice_options_without_feedback(tmp_path):
    lesson = choice_runtime_lesson(with_feedback=False)
    client = FakeSpeechClient()

    voiced = run(LessonAudioService(client, tmp_path).attach_audio(lesson))

    interaction = voiced.beats[1].interaction
    assert interaction is not None
    assert [option.feedback_audio_url for option in interaction.options] == [
        None,
        None,
        None,
    ]
    assert "" not in client.texts
    assert not list((tmp_path / "lesson-001").glob("beat-002-option-*.mp3"))


def test_audio_service_skips_empty_correct_explanation(tmp_path):
    lesson = runtime_lesson(correct_explanation="")
    client = FakeSpeechClient()

    voiced = run(LessonAudioService(client, tmp_path).attach_audio(lesson))

    interaction = voiced.beats[1].interaction
    assert interaction is not None
    assert interaction.correct_audio_url is None
    assert not (
        tmp_path / "lesson-001" / "beat-002-correct.mp3"
    ).exists()
    assert "" not in client.texts


def test_audio_service_retries_each_asset_once(tmp_path):
    client = FakeSpeechClient(
        outcomes=[
            SpeechGenerationError("temporary"),
            b"retry-audio",
        ]
    )
    lesson = runtime_lesson()

    voiced = run(LessonAudioService(client, tmp_path).attach_audio(lesson))

    assert client.texts[:2] == [
        lesson.beats[0].narration,
        lesson.beats[0].narration,
    ]
    assert (
        tmp_path / "lesson-001" / "beat-001.mp3"
    ).read_bytes() == b"retry-audio"
    assert voiced.beats[0].audio_url == "/audio/lesson-001/beat-001.mp3"


def test_audio_service_terminal_failure_removes_partial_lesson(tmp_path):
    client = FakeSpeechClient(
        outcomes=[
            b"first-audio",
            SpeechGenerationError("private upstream detail"),
            SpeechGenerationError("private upstream detail"),
        ]
    )
    lesson = runtime_lesson()

    with pytest.raises(SpeechGenerationError) as error:
        run(LessonAudioService(client, tmp_path).attach_audio(lesson))

    assert str(error.value) == "Audio generation failed for beat-002"
    assert "private upstream detail" not in str(error.value)
    assert error.value.__cause__ is None
    assert not (tmp_path / "lesson-001").exists()
    assert lesson.beats[0].audio_url is None


def test_audio_service_calls_sync_stage_callback(tmp_path):
    stages = []

    run(
        LessonAudioService(FakeSpeechClient(), tmp_path).attach_audio(
            runtime_lesson(),
            on_stage=stages.append,
        )
    )

    assert stages == ["正在生成讲解语音"]


def test_audio_service_awaits_async_stage_callback(tmp_path):
    stages = []

    async def on_stage(stage):
        await asyncio.sleep(0)
        stages.append(stage)

    run(
        LessonAudioService(FakeSpeechClient(), tmp_path).attach_audio(
            runtime_lesson(),
            on_stage=on_stage,
        )
    )

    assert stages == ["正在生成讲解语音"]


@pytest.mark.parametrize(
    ("lesson_id", "beat_id"),
    [
        ("../outside", "beat-001"),
        ("lesson/child", "beat-001"),
        ("lesson\\child", "beat-001"),
        ("lesson-001", "../outside"),
        ("lesson-001", "beat/child"),
        ("lesson-001", "beat\\child"),
    ],
)
def test_audio_service_rejects_path_traversal_before_synthesis(
    tmp_path,
    lesson_id,
    beat_id,
):
    client = FakeSpeechClient()
    lesson = runtime_lesson(
        lesson_id=lesson_id,
        first_beat_id=beat_id,
    )

    with pytest.raises(
        SpeechGenerationError,
        match="Invalid audio asset identifier",
    ):
        run(LessonAudioService(client, tmp_path).attach_audio(lesson))

    assert client.texts == []
    assert list(tmp_path.iterdir()) == []


def test_audio_service_refuses_symlink_destination(tmp_path):
    lesson_dir = tmp_path / "lesson-001"
    lesson_dir.mkdir()
    outside = tmp_path / "outside.mp3"
    outside.write_bytes(b"must-stay-unchanged")
    (lesson_dir / "beat-001.mp3").symlink_to(outside)
    client = FakeSpeechClient()

    with pytest.raises(
        SpeechGenerationError,
        match="Invalid audio asset destination",
    ):
        run(
            LessonAudioService(client, tmp_path).attach_audio(
                runtime_lesson()
            )
        )

    assert client.texts == []
    assert outside.read_bytes() == b"must-stay-unchanged"
