from app.schemas import RuntimeLesson


def audio_asset_filename(asset_id: str) -> str:
    return f"{asset_id}.mp3"


def audio_asset_url(lesson_id: str, asset_id: str) -> str:
    return f"/audio/{lesson_id}/{audio_asset_filename(asset_id)}"


def cue_asset_id(beat_id: str, cue_id: str) -> str:
    return f"{beat_id}-{cue_id}"


def hint_asset_id(beat_id: str, index: int) -> str:
    return f"{beat_id}-hint-{index}"


def option_feedback_asset_id(beat_id: str, index: int) -> str:
    return f"{beat_id}-option-{index}"


def correct_feedback_asset_id(beat_id: str) -> str:
    return f"{beat_id}-correct"


def validate_lesson_audio_manifest(lesson: RuntimeLesson) -> None:
    """Require every voiced field to map to its exact local asset."""
    lesson_id = lesson.lesson_id
    for beat in lesson.beats:
        if beat.sync_cues:
            if beat.audio_url is not None:
                raise ValueError("cue-based beat cannot have beat audio")
            for cue in beat.sync_cues:
                expected = audio_asset_url(
                    lesson_id,
                    cue_asset_id(beat.beat_id, cue.cue_id),
                )
                if cue.audio_url != expected:
                    raise ValueError("runtime cue audio manifest mismatch")
        else:
            expected = audio_asset_url(lesson_id, beat.beat_id)
            if beat.audio_url != expected:
                raise ValueError("runtime beat audio manifest mismatch")

        interaction = beat.interaction
        if interaction is None:
            continue
        expected_hints = [
            audio_asset_url(
                lesson_id,
                hint_asset_id(beat.beat_id, index),
            )
            for index, _hint in enumerate(interaction.hints, start=1)
        ]
        if interaction.hint_audio_urls != expected_hints:
            raise ValueError("interaction hint audio manifest mismatch")
        for index, option in enumerate(interaction.options, start=1):
            expected_option = (
                audio_asset_url(
                    lesson_id,
                    option_feedback_asset_id(beat.beat_id, index),
                )
                if option.feedback
                else None
            )
            if option.feedback_audio_url != expected_option:
                raise ValueError("option feedback audio manifest mismatch")
        expected_correct = (
            audio_asset_url(
                lesson_id,
                correct_feedback_asset_id(beat.beat_id),
            )
            if interaction.explanation_after_correct
            else None
        )
        if interaction.correct_audio_url != expected_correct:
            raise ValueError("correct feedback audio manifest mismatch")
