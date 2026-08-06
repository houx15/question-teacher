import re
from typing import Callable, Dict, List, Optional
from uuid import uuid4

from app.schemas import (
    BoardAction,
    Interaction,
    InteractionOption,
    LessonDraft,
    ProblemInput,
    RuntimeBeat,
    RuntimeLesson,
)


class LessonCompileError(RuntimeError):
    """Raised when a validated lesson cannot be compiled safely."""


class LessonCompiler:
    _SAFE_LESSON_ID = re.compile(r"\A[A-Za-z0-9][A-Za-z0-9._:-]{0,127}\Z")

    def __init__(
        self,
        lesson_id_factory: Optional[Callable[[], str]] = None,
    ) -> None:
        self._lesson_id_factory = lesson_id_factory or (
            lambda: str(uuid4())
        )

    def compile(
        self,
        problem: ProblemInput,
        draft: LessonDraft,
        validation_report: Dict[str, object],
        lesson_id: Optional[str] = None,
    ) -> RuntimeLesson:
        resolved_lesson_id = self._resolve_lesson_id(lesson_id)
        beats: List[RuntimeBeat] = [
            RuntimeBeat(
                beat_id="pending",
                purpose="进入问题",
                narration=draft.opening,
                board_actions=[
                    BoardAction(
                        type="write",
                        target="original_problem",
                        content=problem.problem_text,
                    )
                ],
                layer="base",
            )
        ]

        method_introduction = draft.method_introduction
        method_narration = method_introduction.spoken_narration
        if len(method_narration) > 90:
            raise LessonCompileError("方法介绍的口语讲稿过长。")
        beats.append(
            RuntimeBeat(
                beat_id="pending",
                purpose="先认识方法",
                narration=method_narration,
                board_actions=[
                    BoardAction(
                        type="write",
                        target="method_name",
                        content=method_introduction.method_name,
                    ),
                    BoardAction(type="focus", target="method_name"),
                    BoardAction(
                        type="write",
                        target="method_target_form",
                        content=method_introduction.target_form,
                    ),
                ],
                layer="micro_explanation",
            )
        )

        for moment in draft.moments:
            beats.append(
                RuntimeBeat(
                    beat_id="pending",
                    purpose=moment.purpose,
                    narration=moment.narration,
                    board_actions=moment.board_actions,
                    layer=moment.layer,
                    interaction=moment.interaction,
                )
            )

        transfer_item = draft.transfer_item
        transfer_interaction = (
            Interaction(
                interaction_id="near-transfer",
                kind="choice",
                prompt=transfer_item.problem_text,
                expected_answer=transfer_item.correct_option_id,
                options=[
                    InteractionOption(
                        option_id=option.option_id,
                        label=option.label,
                        feedback=option.feedback,
                    )
                    for option in transfer_item.options
                ],
                hints=[transfer_item.method_signal],
                explanation_after_correct="你已经识别并使用了同一方法结构。",
            )
            if transfer_item.options
            else Interaction(
                interaction_id="near-transfer",
                kind="transfer",
                prompt=transfer_item.problem_text,
                expected_answer=transfer_item.expected_answer,
                hints=[transfer_item.method_signal],
                explanation_after_correct="你已经识别并使用了同一方法结构。",
            )
        )

        beats.extend(
            [
                RuntimeBeat(
                    beat_id="pending",
                    purpose="压缩方法",
                    narration=draft.summary,
                    board_actions=[
                        BoardAction(
                            type="write",
                            target="method_summary",
                            content=draft.summary,
                        )
                    ],
                    layer="base",
                ),
                RuntimeBeat(
                    beat_id="pending",
                    purpose="完成近迁移",
                    narration="现在换一道表面不同、结构相同的题。",
                    board_actions=[],
                    layer="interaction",
                    interaction=transfer_interaction,
                ),
            ]
        )

        numbered_beats = []
        for index, beat in enumerate(beats, start=1):
            next_beat_id = (
                f"beat-{index + 1:03d}"
                if index < len(beats)
                else None
            )
            numbered_beats.append(
                RuntimeBeat(
                    beat_id=f"beat-{index:03d}",
                    purpose=beat.purpose,
                    narration=beat.narration,
                    board_actions=beat.board_actions,
                    layer=beat.layer,
                    interaction=beat.interaction,
                    audio_url=beat.audio_url,
                    next_beat_id=next_beat_id,
                )
            )

        return RuntimeLesson(
            lesson_id=resolved_lesson_id,
            problem=problem,
            title=draft.title,
            learning_goal=draft.learning_goal,
            beats=numbered_beats,
            summary=draft.summary,
            transfer_item=draft.transfer_item,
            validation_report=dict(validation_report),
        )

    def _resolve_lesson_id(self, lesson_id: Optional[str]) -> str:
        if lesson_id is None:
            try:
                lesson_id = self._lesson_id_factory()
            except Exception:
                raise LessonCompileError(
                    "Unable to create a safe lesson id."
                ) from None

        if (
            not isinstance(lesson_id, str)
            or self._SAFE_LESSON_ID.fullmatch(lesson_id) is None
        ):
            raise LessonCompileError("Unable to create a safe lesson id.")
        return lesson_id
