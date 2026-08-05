import inspect
import shutil
from pathlib import Path
from typing import Callable, Optional

from app.schemas import RuntimeLesson
from app.tts_client import SpeechGenerationError


class LessonAudioService:
    def __init__(self, client, audio_root: Path):
        self.client = client
        self.audio_root = Path(audio_root)

    @staticmethod
    def _validate_identifier(value: str) -> None:
        if (
            value in {".", ".."}
            or "/" in value
            or "\\" in value
            or "\x00" in value
        ):
            raise SpeechGenerationError(
                "Invalid audio asset identifier"
            )

    def _lesson_directory(self, lesson_id: str) -> Path:
        self._validate_identifier(lesson_id)
        root = self.audio_root.resolve()
        lesson_dir = root / lesson_id
        if lesson_dir.resolve().parent != root:
            raise SpeechGenerationError(
                "Invalid audio asset identifier"
            )
        return lesson_dir

    async def _write(
        self,
        lesson_id: str,
        asset_id: str,
        text: str,
    ) -> str:
        self._validate_identifier(asset_id)
        for _attempt in range(2):
            try:
                data = await self.client.synthesize(text)
                if not data:
                    raise SpeechGenerationError(
                        "Speech generation returned empty audio"
                    )
                break
            except Exception:
                pass
        else:
            raise SpeechGenerationError(
                f"Audio generation failed for {asset_id}"
            ) from None

        lesson_dir = self._lesson_directory(lesson_id)
        filename = f"{asset_id}.mp3"
        (lesson_dir / filename).write_bytes(data)
        return f"/audio/{lesson_id}/{filename}"

    async def attach_audio(
        self,
        lesson: RuntimeLesson,
        on_stage: Optional[Callable] = None,
    ) -> RuntimeLesson:
        lesson_dir = self._lesson_directory(lesson.lesson_id)
        for beat in lesson.beats:
            self._validate_identifier(beat.beat_id)

        lesson_dir.mkdir(parents=True, exist_ok=True)
        try:
            if on_stage is not None:
                stage_result = on_stage("正在生成讲解语音")
                if inspect.isawaitable(stage_result):
                    await stage_result

            voiced_beats = []
            for beat in lesson.beats:
                audio_url = await self._write(
                    lesson.lesson_id,
                    beat.beat_id,
                    beat.narration,
                )
                interaction = beat.interaction
                if interaction is not None:
                    hint_audio_urls = []
                    for index, hint in enumerate(
                        interaction.hints,
                        start=1,
                    ):
                        hint_audio_urls.append(
                            await self._write(
                                lesson.lesson_id,
                                f"{beat.beat_id}-hint-{index}",
                                hint,
                            )
                        )

                    correct_audio_url = None
                    if interaction.explanation_after_correct:
                        correct_audio_url = await self._write(
                            lesson.lesson_id,
                            f"{beat.beat_id}-correct",
                            interaction.explanation_after_correct,
                        )
                    interaction = interaction.model_copy(
                        update={
                            "hint_audio_urls": hint_audio_urls,
                            "correct_audio_url": correct_audio_url,
                        }
                    )

                voiced_beats.append(
                    beat.model_copy(
                        update={
                            "audio_url": audio_url,
                            "interaction": interaction,
                        }
                    )
                )
            return lesson.model_copy(update={"beats": voiced_beats})
        except Exception:
            if lesson_dir.exists():
                shutil.rmtree(lesson_dir)
            raise
