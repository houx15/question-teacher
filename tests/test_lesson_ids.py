import pytest

from app.audio_service import LessonAudioService
from app.compiler import LessonCompileError, LessonCompiler
from app.store import MemoryStore
from app.tts_client import SpeechGenerationError
from tests.test_compiler import compile_lesson
from tests.test_tts_client import FakeSpeechClient, run, runtime_lesson


@pytest.mark.parametrize(
    "lesson_id",
    [
        "lesson.safe:1",
        "L" + "a" * 127,
    ],
)
def test_lesson_id_contract_accepts_valid_ids_across_components(
    tmp_path,
    lesson_id,
):
    compiled = compile_lesson(
        LessonCompiler(lesson_id_factory=lambda: lesson_id)
    )
    lesson = runtime_lesson().model_copy(update={"lesson_id": lesson_id})
    database_path = tmp_path / "lessons.sqlite3"
    MemoryStore(database_path).save_lesson(lesson)
    restored = MemoryStore(database_path).get_lesson(lesson_id)
    voiced = run(
        LessonAudioService(FakeSpeechClient(), tmp_path / "audio").attach_audio(
            lesson
        )
    )

    assert compiled.lesson_id == lesson_id
    assert restored == lesson
    assert voiced.lesson_id == lesson_id
    assert (tmp_path / "audio" / lesson_id).is_dir()


@pytest.mark.parametrize(
    "lesson_id",
    [
        "x" * 129,
        "../lesson",
        "",
        "   ",
    ],
)
def test_lesson_id_contract_rejects_invalid_ids_across_components(
    tmp_path,
    lesson_id,
):
    compiler = LessonCompiler(lesson_id_factory=lambda: lesson_id)
    lesson = runtime_lesson().model_copy(update={"lesson_id": lesson_id})
    database_path = tmp_path / "lessons.sqlite3"
    client = FakeSpeechClient()

    with pytest.raises(LessonCompileError, match="lesson id"):
        compile_lesson(compiler)
    with pytest.raises(ValueError, match="^invalid lesson id$"):
        MemoryStore(database_path).save_lesson(lesson)
    with pytest.raises(
        SpeechGenerationError,
        match="^Invalid audio asset identifier$",
    ):
        run(
            LessonAudioService(client, tmp_path / "audio").attach_audio(
                lesson
            )
        )

    assert not database_path.exists()
    assert client.texts == []
    assert not (tmp_path / "audio").exists()
