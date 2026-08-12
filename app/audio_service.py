import asyncio
import inspect
import os
import shutil
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from app.audio_manifest import (
    audio_asset_filename,
    audio_asset_url,
    correct_feedback_asset_id,
    cue_asset_id,
    hint_asset_id,
    option_feedback_asset_id,
)
from app.lesson_ids import is_valid_lesson_id
from app.schemas import RuntimeLesson, RuntimeSyncCue
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
        if not is_valid_lesson_id(lesson_id):
            raise SpeechGenerationError(
                "Invalid audio asset identifier"
            )
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
        destination = lesson_dir / audio_asset_filename(asset_id)
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
        return audio_asset_url(lesson_id, asset_id)

    def _preflight(self, lesson: RuntimeLesson) -> Path:
        lesson_dir = self._lesson_directory(lesson.lesson_id)
        planned_asset_ids = set()

        def plan_asset(asset_id: str) -> None:
            self._validate_identifier(asset_id)
            if asset_id in planned_asset_ids:
                raise SpeechGenerationError(
                    "Duplicate audio asset identifier"
                )
            self._asset_destination(
                lesson.lesson_id,
                asset_id,
            )
            planned_asset_ids.add(asset_id)

        for beat in lesson.beats:
            self._validate_identifier(beat.beat_id)
            if beat.sync_cues:
                for cue in beat.sync_cues:
                    self._validate_identifier(cue.cue_id)
                    plan_asset(
                        cue_asset_id(beat.beat_id, cue.cue_id),
                    )
            else:
                plan_asset(beat.beat_id)

            interaction = beat.interaction
            if interaction is None:
                continue
            for index, _hint in enumerate(
                interaction.hints,
                start=1,
            ):
                plan_asset(
                    hint_asset_id(beat.beat_id, index),
                )
            for index, option in enumerate(
                interaction.options,
                start=1,
            ):
                if option.feedback:
                    plan_asset(
                        option_feedback_asset_id(beat.beat_id, index),
                    )
            if interaction.explanation_after_correct:
                plan_asset(
                    correct_feedback_asset_id(beat.beat_id),
                )
        if lesson_dir.exists():
            raise SpeechGenerationError(
                "Audio destination already exists"
            )
        return lesson_dir

    def cleanup_lesson_audio(self, lesson_id: str) -> None:
        lesson_dir = self._lesson_directory(lesson_id)
        if lesson_dir.exists():
            shutil.rmtree(lesson_dir)

    async def _voice_sync_cues(
        self,
        lesson: RuntimeLesson,
    ) -> Dict[int, List[RuntimeSyncCue]]:
        cue_jobs: List[Tuple[int, str, RuntimeSyncCue]] = [
            (
                beat_index,
                beat.beat_id,
                cue,
            )
            for beat_index, beat in enumerate(lesson.beats)
            for cue in beat.sync_cues
        ]
        if not cue_jobs:
            return {}

        cue_semaphore = asyncio.Semaphore(3)

        async def voice_cue(
            beat_id: str,
            cue: RuntimeSyncCue,
        ) -> RuntimeSyncCue:
            async with cue_semaphore:
                audio_url = await self._write(
                    lesson.lesson_id,
                    cue_asset_id(beat_id, cue.cue_id),
                    cue.spoken_text,
                )
            return cue.model_copy(
                deep=True,
                update={"audio_url": audio_url},
            )

        cue_tasks = [
            asyncio.create_task(voice_cue(beat_id, cue))
            for _beat_index, beat_id, cue in cue_jobs
        ]
        try:
            voiced_cues = await asyncio.gather(*cue_tasks)
        except BaseException:
            for task in cue_tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(
                *cue_tasks,
                return_exceptions=True,
            )
            raise

        cues_by_beat: Dict[int, List[RuntimeSyncCue]] = {}
        for (beat_index, _beat_id, _cue), voiced_cue in zip(
            cue_jobs,
            voiced_cues,
        ):
            cues_by_beat.setdefault(beat_index, []).append(voiced_cue)
        return cues_by_beat

    async def attach_audio(
        self,
        lesson: RuntimeLesson,
        on_stage: Optional[Callable] = None,
    ) -> RuntimeLesson:
        lesson_dir = self._preflight(lesson)
        lesson_dir.mkdir(parents=True, exist_ok=False)
        try:
            if on_stage is not None:
                stage_result = on_stage("正在生成讲解语音")
                if inspect.isawaitable(stage_result):
                    await stage_result

            voiced_cues_by_beat = await self._voice_sync_cues(lesson)
            voiced_beats = []
            for beat_index, beat in enumerate(lesson.beats):
                if beat.sync_cues:
                    audio_url = None
                    sync_cues = voiced_cues_by_beat[beat_index]
                else:
                    audio_url = await self._write(
                        lesson.lesson_id,
                        beat.beat_id,
                        beat.narration,
                    )
                    sync_cues = []
                voiced_beats.append(
                    beat.model_copy(
                        update={
                            "audio_url": audio_url,
                            "sync_cues": sync_cues,
                        }
                    )
                )

            fully_voiced_beats = []
            for beat in voiced_beats:
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
                                hint_asset_id(beat.beat_id, index),
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
                                    option_feedback_asset_id(
                                        beat.beat_id,
                                        index,
                                    ),
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
                            correct_feedback_asset_id(beat.beat_id),
                            interaction.explanation_after_correct,
                        )
                    interaction = interaction.model_copy(
                        update={
                            "hint_audio_urls": hint_audio_urls,
                            "correct_audio_url": correct_audio_url,
                            "options": voiced_options,
                        }
                    )

                fully_voiced_beats.append(
                    beat.model_copy(
                        update={
                            "interaction": interaction,
                        }
                    )
                )
            return lesson.model_copy(
                update={"beats": fully_voiced_beats}
            )
        except BaseException:
            if lesson_dir.exists():
                shutil.rmtree(lesson_dir)
            raise
