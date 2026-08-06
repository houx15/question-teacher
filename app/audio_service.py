import asyncio
import inspect
import os
import shutil
import tempfile
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
        if lesson_dir.is_symlink():
            raise SpeechGenerationError(
                "Invalid audio asset destination"
            )
        if lesson_dir.resolve().parent != root:
            raise SpeechGenerationError(
                "Invalid audio asset identifier"
            )
        return lesson_dir

    def _asset_destination(
        self,
        lesson_id: str,
        asset_id: str,
    ):
        self._validate_identifier(asset_id)
        root = self.audio_root.resolve()
        lesson_dir = self._lesson_directory(lesson_id)
        resolved_lesson_dir = lesson_dir.resolve()
        destination = lesson_dir / f"{asset_id}.mp3"
        if (
            destination.is_symlink()
            or resolved_lesson_dir.parent != root
            or destination.resolve().parent != resolved_lesson_dir
        ):
            raise SpeechGenerationError(
                "Invalid audio asset destination"
            )
        return lesson_dir, destination

    async def _write(
        self,
        lesson_id: str,
        asset_id: str,
        text: str,
    ) -> str:
        lesson_dir, destination = self._asset_destination(
            lesson_id,
            asset_id,
        )
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

        filename = f"{asset_id}.mp3"
        file_descriptor, temporary_name = tempfile.mkstemp(
            dir=lesson_dir,
            prefix=f".{asset_id}-",
            suffix=".tmp",
        )
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "wb") as temporary_file:
                temporary_file.write(data)
            _, checked_destination = self._asset_destination(
                lesson_id,
                asset_id,
            )
            os.replace(temporary_path, checked_destination)
        finally:
            temporary_path.unlink(missing_ok=True)
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

                    option_feedback_semaphore = asyncio.Semaphore(2)

                    async def voice_option(index, option):
                        feedback_audio_url = None
                        if option.feedback:
                            async with option_feedback_semaphore:
                                feedback_audio_url = await self._write(
                                    lesson.lesson_id,
                                    f"{beat.beat_id}-option-{index}",
                                    option.feedback,
                                )
                        return option.model_copy(
                            update={
                                "feedback_audio_url": feedback_audio_url,
                            }
                        )

                    option_tasks = [
                        asyncio.create_task(voice_option(index, option))
                        for index, option in enumerate(
                            interaction.options,
                            start=1,
                        )
                    ]
                    try:
                        voiced_options = await asyncio.gather(
                            *option_tasks
                        )
                    except BaseException:
                        for task in option_tasks:
                            if not task.done():
                                task.cancel()
                        await asyncio.gather(
                            *option_tasks,
                            return_exceptions=True,
                        )
                        raise

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
                            "options": voiced_options,
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
