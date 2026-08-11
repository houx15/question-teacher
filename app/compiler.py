from typing import Callable, Dict, List, Optional
from uuid import uuid4

from app.lesson_ids import is_valid_lesson_id
from app.problem_focus import compile_problem_focus_targets
from app.schemas import (
    BoardAction,
    FIXED_RUNTIME_CUE_IDS,
    Interaction,
    InteractionOption,
    LessonDraft,
    LessonLayer,
    NarrativeSyncCue,
    ProblemInput,
    RuntimeBeat,
    RuntimeLesson,
    RuntimeSyncCue,
    SyncVisualAction,
)

NEAR_TRANSFER_INTERACTION_ID = "near-transfer"
_LEGACY_BOARD_ACTION_TYPES = {
    "write",
    "transform",
    "focus",
    "reveal",
}


def _copy_runtime_cue(cue: NarrativeSyncCue) -> RuntimeSyncCue:
    return RuntimeSyncCue.model_validate(cue.model_dump())


def _legacy_board_actions(
    sync_cues: List[RuntimeSyncCue],
) -> List[BoardAction]:
    """Project the exact legacy board subset; runtime cues stay authoritative."""
    legacy_actions = []
    for cue in sync_cues:
        for action in cue.start_actions:
            if (
                action.surface != "board"
                or action.type not in _LEGACY_BOARD_ACTION_TYPES
            ):
                # Problem actions and lifecycle-only visual effects have no
                # place in the legacy beat-start board projection.
                continue
            legacy_actions.append(
                BoardAction(
                    type=action.type,
                    target=action.target,
                    content=action.content,
                    source=action.source,
                    relation_target=action.relation_target,
                    annotation=action.annotation,
                )
            )
    return legacy_actions


def _runtime_beat(
    *,
    purpose: str,
    layer: LessonLayer,
    sync_cues: List[RuntimeSyncCue],
    interaction: Optional[Interaction] = None,
) -> RuntimeBeat:
    return RuntimeBeat(
        beat_id="pending",
        purpose=purpose,
        narration="".join(cue.spoken_text for cue in sync_cues),
        board_actions=_legacy_board_actions(sync_cues),
        layer=layer,
        sync_cues=sync_cues,
        interaction=interaction,
    )


def _authored_section_beats(
    *,
    purpose: str,
    default_layer: LessonLayer,
    cues: List[RuntimeSyncCue],
    interactions_after_cue: Dict[str, Interaction],
    layers_by_cue: Dict[str, LessonLayer],
) -> List[RuntimeBeat]:
    """Split authored fixed speech at interaction and layer boundaries."""
    beats = []
    pending = []
    pending_layer = default_layer

    def flush(interaction: Optional[Interaction] = None) -> None:
        nonlocal pending
        if not pending:
            return
        beats.append(
            _runtime_beat(
                purpose=purpose,
                layer=pending_layer,
                sync_cues=pending,
                interaction=interaction,
            )
        )
        pending = []

    for cue in cues:
        cue_layer = layers_by_cue.get(cue.cue_id, default_layer)
        if pending and cue_layer != pending_layer:
            flush()
        pending_layer = cue_layer
        pending.append(cue)
        interaction = interactions_after_cue.get(cue.cue_id)
        if interaction is not None:
            flush(interaction)
    flush()
    return beats


class LessonCompileError(RuntimeError):
    """Raised when a validated lesson cannot be compiled safely."""


class LessonCompiler:
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
        problem_focus_targets = compile_problem_focus_targets(
            problem.problem_text
        )
        authored_cue_ids = {
            cue.cue_id
            for cues in (
                draft.opening_sync_cues or [],
                draft.method_introduction_sync_cues or [],
                [
                    cue
                    for moment in draft.moments
                    for cue in moment.sync_cues
                ],
                draft.summary_sync_cues or [],
            )
            for cue in cues
        }
        if authored_cue_ids.intersection(FIXED_RUNTIME_CUE_IDS.values()):
            raise LessonCompileError(
                "同步提示 ID 与编译器保留 ID 冲突。"
            )

        opening_cues = (
            [_copy_runtime_cue(cue) for cue in draft.opening_sync_cues]
            if draft.opening_sync_cues is not None
            else [
                RuntimeSyncCue(
                    cue_id=FIXED_RUNTIME_CUE_IDS["opening"],
                    spoken_text=draft.opening,
                )
            ]
        )
        beats: List[RuntimeBeat] = _authored_section_beats(
            purpose="进入问题",
            default_layer="base",
            cues=opening_cues,
            interactions_after_cue=(
                draft.fixed_section_interactions_after_cue
            ),
            layers_by_cue=draft.fixed_section_layers_by_cue,
        )

        method_introduction = draft.method_introduction
        method_narration = method_introduction.spoken_narration
        if len(method_narration) > 90:
            raise LessonCompileError("方法介绍的口语讲稿过长。")
        method_cues = (
            [
                _copy_runtime_cue(cue)
                for cue in draft.method_introduction_sync_cues
            ]
            if draft.method_introduction_sync_cues is not None
            else [
                RuntimeSyncCue(
                    cue_id=FIXED_RUNTIME_CUE_IDS["method_introduction"],
                    spoken_text=method_narration,
                    start_actions=[
                        SyncVisualAction(
                            surface="board",
                            type="write",
                            target="method_name",
                            content=method_introduction.method_name,
                        ),
                        SyncVisualAction(
                            surface="board",
                            type="focus",
                            target="method_name",
                        ),
                        SyncVisualAction(
                            surface="board",
                            type="write",
                            target="method_target_form",
                            content=method_introduction.target_form,
                        ),
                    ],
                )
            ]
        )
        beats.extend(
            _authored_section_beats(
                purpose="先认识方法",
                default_layer="micro_explanation",
                cues=method_cues,
                interactions_after_cue=(
                    draft.fixed_section_interactions_after_cue
                ),
                layers_by_cue=draft.fixed_section_layers_by_cue,
            )
        )

        for moment in draft.moments:
            sync_cues = [
                _copy_runtime_cue(cue)
                for cue in moment.sync_cues
            ]
            beats.append(
                _runtime_beat(
                    purpose=moment.purpose,
                    layer=moment.layer,
                    sync_cues=sync_cues,
                    interaction=moment.interaction,
                )
            )

        transfer_item = draft.transfer_item
        if transfer_item.options and any(
            option.label is None for option in transfer_item.options
        ):
            raise LessonCompileError(
                "近迁移选项缺少已规范化的显示标签。"
            )
        transfer_interaction = (
            Interaction(
                interaction_id=NEAR_TRANSFER_INTERACTION_ID,
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
                explanation_after_correct=(
                    ""
                    if draft.transfer_feedback_is_authoritative
                    else "你已经识别并使用了同一方法结构。"
                ),
            )
            if transfer_item.options
            else Interaction(
                interaction_id=NEAR_TRANSFER_INTERACTION_ID,
                kind="transfer",
                prompt=transfer_item.problem_text,
                expected_answer=transfer_item.expected_answer,
                hints=[transfer_item.method_signal],
                explanation_after_correct=(
                    ""
                    if draft.transfer_feedback_is_authoritative
                    else "你已经识别并使用了同一方法结构。"
                ),
            )
        )

        summary_cues = (
            [
                _copy_runtime_cue(cue)
                for cue in draft.summary_sync_cues
            ]
            if draft.summary_sync_cues is not None
            else [
                RuntimeSyncCue(
                    cue_id=FIXED_RUNTIME_CUE_IDS["summary"],
                    spoken_text=draft.summary,
                    start_actions=[
                        SyncVisualAction(
                            surface="board",
                            type="write",
                            target="method_summary",
                            content=draft.summary,
                        )
                    ],
                )
            ]
        )
        beats.extend(
            _authored_section_beats(
                purpose="压缩方法",
                default_layer="base",
                cues=summary_cues,
                interactions_after_cue=(
                    draft.fixed_section_interactions_after_cue
                ),
                layers_by_cue=draft.fixed_section_layers_by_cue,
            )
        )
        beats.append(
            _runtime_beat(
                purpose="完成近迁移",
                sync_cues=[
                    RuntimeSyncCue(
                        cue_id=FIXED_RUNTIME_CUE_IDS["transfer_intro"],
                        spoken_text=(
                            "现在换一道表面不同、结构相同的题。"
                        ),
                    )
                ],
                layer="interaction",
                interaction=transfer_interaction,
            )
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
                    sync_cues=[
                        cue.model_copy(deep=True)
                        for cue in beat.sync_cues
                    ],
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
            problem_focus_targets=problem_focus_targets,
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

        if not is_valid_lesson_id(lesson_id):
            raise LessonCompileError("Unable to create a safe lesson id.")
        return lesson_id
